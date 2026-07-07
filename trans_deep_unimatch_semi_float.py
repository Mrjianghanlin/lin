import argparse
import logging
import os
import pprint
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np
import torch.backends.cudnn as cudnn
from itertools import cycle

from dataset.semi import SemiDataset
from model.models.nets1.mix_deeplab import DeepLab
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from supervised import evaluate


parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation with AMP')
parser.add_argument('--labeled-id-path', type=str, default="../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC/sing_train_semi_random20/train.txt")
parser.add_argument('--val-id-path', type=str, default="../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC/VOC16_23Train/val.txt")
parser.add_argument('--unlabeled-id-path', type=str, default="../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC/sing_train_semi_random20/train80.txt")
parser.add_argument('--save-path', type=str, default="myEH/model_update_semi_float_random20/mix_semi")
parser.add_argument('--freeze-epochs', type=int, default=0)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

torch.cuda.set_device(0)
def main():
    args = parser.parse_args()
    
    cfg = {
        'dataset': 'pascal',
        'data_root':  r'../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC',
        'nclass': 6,
        'crop_size': 512,
        'pretrained': True,
        'epochs': 200,
        'batch_size': 2,
        'lr': 0.001,
        'lr_multi': 10.0,
        'criterion': {'kwargs': {'ignore_index': 255, 'min_kept': 200000, 'thresh': 0.7}, 'name': 'OHEM'},
        'conf_thresh': 0.95,
        'backbone': 'hrnet',
        'replace_stride_with_dilation': [False, False, True],
        'dilations': [6, 12, 18],
        'model_path': "pretrained/hrnetv2_w48-imagenet.pth",
        'num_workers': 0,
        'downsample_factor': 16,
        'in_channels': 3,
        'use_swin': True,
    }

    os.makedirs(args.save_path, exist_ok=True)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    
    writer = SummaryWriter(args.save_path)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    # 模型选择
    model = DeepLab(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'], 
                    downsample_factor=cfg['downsample_factor'], use_swin=cfg['use_swin'])
    model.cuda()

    # 优化器
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    # 损失函数
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()

    # 数据集
    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=None)
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', cfg['crop_size'], args.val_id_path)

    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=1, drop_last=True)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=1, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    total_iters = len(trainloader_l) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    rank = 0  # 单卡训练，无分布式

    # AMP 缩放器
    scaler = GradScaler(enabled=True)

    # 检查点恢复
    latest_path = os.path.join(args.save_path, 'latest.pth')
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        logger.info(f'************ Load from checkpoint at epoch {epoch}\n')

    # 动态数据加载器
    def safe_loader(loader_l, loader_u):
        iter_u = iter(loader_u)
        for batch_l in loader_l:
            try:
                batch_u1 = next(iter_u)
                batch_u2 = next(iter_u)
            except StopIteration:
                iter_u = iter(loader_u)
                batch_u1 = next(iter_u)
                batch_u2 = next(iter_u)
            # 转移到GPU
            batch_l = [t.cuda(non_blocking=True) for t in batch_l]
            batch_u1 = [t.cuda(non_blocking=True) for t in batch_u1]
            batch_u2 = [t.cuda(non_blocking=True) for t in batch_u2]
            yield batch_l, batch_u1, batch_u2

    # ================== Training Loop ==================
    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(f'===========> Epoch: {epoch}, LR: {optimizer.param_groups[0]["lr"]:.5f}, Previous best: {previous_best:.2f}')

        # 冻结骨干
        if epoch < args.freeze_epochs:
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            for param in model.backbone.parameters():
                param.requires_grad = True

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        loader = safe_loader(trainloader_l, trainloader_u)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):

            # ================== AMP 前向 ==================
            with autocast(device_type='cuda', dtype=torch.float16):
                img_x, mask_x = img_x.cuda(), mask_x.cuda()
                img_u_w = img_u_w.cuda()
                img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
                cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
                img_u_w_mix = img_u_w_mix.cuda()
                img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
                ignore_mask_mix = ignore_mask_mix.cuda()

                model.eval()
                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            # CutMix 应用
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            # ================== AMP 梯度缩放训练 ==================
            with autocast(device_type='cuda', dtype=torch.float16):
                preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
                pred_x, pred_u_w = preds.split([num_lb, num_ulb])
                pred_u_w_fp = preds_fp[num_lb:]

                pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

                pred_u_w = pred_u_w.detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

                # CutMix mask 更新
                mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
                mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
                mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
                conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
                ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]
                mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
                conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
                ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

                # 损失
                loss_x = criterion_l(pred_x, mask_x)
                loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
                loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
                loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

                loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
                loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
                loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

                loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
                loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

                loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 更新指标
            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                         (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio)

            # 动态学习率
            iters = epoch * len(trainloader_l) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            # Tensorboard
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

        # ================== Evaluation ==================
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        # 保存checkpoint
        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(model.state_dict(), os.path.join(args.save_path, "best_epoch_weights.pth"))


if __name__ == '__main__':
    main()
