
import argparse
import logging
import os
import pprint
import math
import torch
import numpy as np
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ====== AMP 混合精度 ======
from torch.amp import autocast, GradScaler

from dataset.semiact import SemiDataset
from model.models.nets1.mix_deeplab import DeepLab
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from supervised import evaluate

parser = argparse.ArgumentParser(description='Pure Supervised Semantic Segmentation')
parser.add_argument('--labeled-id-path', type=str,
                    default="../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC/sing_train_nosemi_active11/train.txt")
parser.add_argument('--val-id-path', type=str,
                    default="../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC/VOC16_23Train_nosemi_active11/val.txt")
parser.add_argument('--save-path', type=str, default="myEH/model_supervised")
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--freeze-epochs', type=int, default=0)

torch.cuda.set_device(1)


def main():
    args = parser.parse_args()

    cfg = {
        'dataset': 'pascal',
        'data_root': r'../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC',
        'nclass': 6,
        'crop_size': 512,
        'pretrained': True,
        'epochs': 200,
        'batch_size': 2,
        'lr': 0.001,
        'lr_multi': 10.0,
        'criterion': {'kwargs': {'ignore_index': 255,
                                 'min_kept': 200000,
                                 'thresh': 0.7},
                      'name': 'OHEM'},
        'backbone': 'hrnet',
        'replace_stride_with_dilation': [False, False, True],
        'dilations': [6, 12, 18],
        'model_path': "pretrained/hrnetv2_w48-imagenet.pth",
        'num_workers': 1,
        'downsample_factor': 16,
        'in_channels': 3,
    }

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))

    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    # ===== 模型 =====
    model = DeepLab(
        num_classes=cfg['nclass'],
        backbone=cfg['backbone'],
        pretrained=cfg['pretrained'],
        downsample_factor=cfg['downsample_factor']
    )
    model.cuda()

    optimizer = SGD(
        [
            {'params': model.backbone.parameters(), 'lr': cfg['lr']},
            {'params': [p for n, p in model.named_parameters() if 'backbone' not in n],
             'lr': cfg['lr'] * cfg['lr_multi']}
        ], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4
    )

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError

    scaler = GradScaler(enabled=True)

    # ===== 数据集 =====
    trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=1, pin_memory=True,
                             drop_last=True)

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', cfg['crop_size'], args.val_id_path)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    previous_best = 0.0
    epoch_start = 0

    # ===== 断点续训 =====
    last_ckpt = os.path.join(args.save_path, 'latest.pth')
    if os.path.exists(last_ckpt):
        checkpoint = torch.load(last_ckpt, map_location='cuda')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1
        previous_best = checkpoint['previous_best']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch_start)

    # =======================
    #      训练主循环
    # =======================
    total_iters = len(trainloader) * cfg['epochs']
    for epoch in range(epoch_start, cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best))

        # 可选冻结
        if epoch < args.freeze_epochs:
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            for param in model.backbone.parameters():
                param.requires_grad = True

        total_loss = AverageMeter()
        model.train()
        for i, (img, mask) in enumerate(trainloader):
            img, mask = img.cuda(), mask.cuda()

            with autocast(device_type='cuda', dtype=torch.float16):
                pred = model(img)
                loss = criterion_l(pred, mask)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss.update(loss.item())
            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

        # ===== 验证 =====
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)
        for cls_idx, iou in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}'.format(
                cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
        writer.add_scalar('eval/mIoU', mIoU, epoch)
        for i, iou in enumerate(iou_class):
            writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'epoch': epoch, 'previous_best': previous_best}
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if is_best:
            torch.save(model.state_dict(), os.path.join(args.save_path, "best_epoch_weights.pth"))


if __name__ == '__main__':
    main()
