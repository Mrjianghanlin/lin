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
from model.models.nets.deeplabv3_training import weights_init
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter

parser = argparse.ArgumentParser(description='Supervised Semantic Segmentation with Active Learning')
parser.add_argument('--labeled-id-path', type=str, default="../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC/sing_train_nosemi_active/train.txt")
parser.add_argument('--val-id-path', type=str, default="../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC/VOC16_23Train_nosemi_active/val.txt")
parser.add_argument('--unlabeled-id-path', type=str, default="../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC/VOC16_23Train_nosemi_active/train.txt")
parser.add_argument('--save-path', type=str, default="myEH/model_update_active_supervised/nosmei_active20%")
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--freeze-epochs', type=int, default=0)
parser.add_argument('--active-ratio', type=float, default=0.01, help='主动学习选择比例')
parser.add_argument('--active-start-epoch', type=int, default=30, help='开始主动学习的epoch')
parser.add_argument('--max-active-ratio', type=float, default=0.2, help='总选择比例上限')

torch.cuda.set_device(1)

def main():
    args = parser.parse_args()
    model_path = ""

    cfg = {
        'dataset': 'pascal',
        'data_root':  r'../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC',
        'nclass': 6,
        'crop_size': 512,
        'pretrained': True,
        'epochs': 204,
        'batch_size': 2,
        'lr': 0.001,
        'lr_multi': 10.0,
        'criterion': {'kwargs': {'ignore_index': 255,
                                 'min_kept': 200000,
                                 'thresh': 0.7},
                      'name': 'OHEM'},
        'conf_thresh': 0.95,
        'backbone': 'hrnet',
        'replace_stride_with_dilation': [False, False, True],
        'dilations': [6, 12, 18],
        'model_path': "pretrained/hrnetv2_w48-imagenet.pth",
        'num_workers': 0,
        'use_swin': True,
        'downsample_factor': 16,
        'in_channels': 3,
    }

    # ===== 主动学习组件 =====
    class ActiveLearningManager:
        def __init__(self, labeled_path, unlabeled_path, max_ratio=0.2):
            self.labeled_path = labeled_path
            self.unlabeled_path = unlabeled_path
            self.max_ratio = max_ratio

            self._reload_pools()

            self.initial_unlabeled = len(self.unlabeled_pool)
            self.total_quota = math.ceil(self.initial_unlabeled * max_ratio)
            self.total_selected = 0
            print(f"初始未标注样本数: {self.initial_unlabeled}, 总配额: {self.total_quota}")

        def _reload_pools(self):
            with open(self.labeled_path) as f:
                self.labeled_pool = list(set(line.strip() for line in f))
            with open(self.unlabeled_path) as f:
                self.unlabeled_pool = list(set(line.strip() for line in f))

        def update_pools(self, selected_ids):
            if self.total_selected >= self.total_quota:
                return 0

            valid_ids = list(set(selected_ids) & set(self.unlabeled_pool))
            if not valid_ids:
                return 0

            remaining = self.total_quota - self.total_selected
            valid_ids = valid_ids[:remaining]

            self.labeled_pool.extend(valid_ids)
            self.unlabeled_pool = [x for x in self.unlabeled_pool if x not in valid_ids]
            self.total_selected += len(valid_ids)

            tmp_labeled = f"{self.labeled_path}.tmp"
            tmp_unlabeled = f"{self.unlabeled_path}.tmp"

            with open(tmp_labeled, 'w') as f:
                f.write('\n'.join(self.labeled_pool))
            with open(tmp_unlabeled, 'w') as f:
                f.write('\n'.join(self.unlabeled_pool))

            os.replace(tmp_labeled, self.labeled_path)
            os.replace(tmp_unlabeled, self.unlabeled_path)

            self._reload_pools()
            return len(valid_ids)

        def get_status(self):
            return len(self.labeled_pool), len(self.unlabeled_pool), self.total_quota - self.total_selected

    class UncertaintySelector:
        @staticmethod
        def compute_confidence(logits):
            probs = torch.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            return max_probs.mean(dim=(1, 2))

        @classmethod
        def select_global(cls, confidences, all_ids, max_samples):
            if len(all_ids) == 0:
                return []
            sorted_indices = torch.argsort(confidences)
            select_num = min(max_samples, len(all_ids))
            selected = sorted_indices[:select_num]
            return [all_ids[i] for i in selected.tolist()]

    active_manager = ActiveLearningManager(
        args.labeled_id_path,
        args.unlabeled_id_path,
        max_ratio=args.max_active_ratio
    )
    selector = UncertaintySelector()

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))

    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLab(
        num_classes=cfg['nclass'],
        backbone=cfg['backbone'],
        pretrained=cfg['pretrained'],
        downsample_factor=cfg['downsample_factor'],
        use_swin=cfg['use_swin']
    )

    if not cfg['pretrained']:
        weights_init(model)

    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(model_path, map_location=device)
        temp_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    optimizer = SGD(
        [
            {'params': model.backbone.parameters(), 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
             'lr': cfg['lr'] * cfg['lr_multi']}
        ],
        lr=cfg['lr'],
        momentum=0.9,
        weight_decay=1e-4
    )

    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    model.cuda()

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', cfg['crop_size'], args.val_id_path)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    previous_best = 0.0
    epoch = -1

    scaler = GradScaler(enabled=True)

    last_ckpt = os.path.join(args.save_path, 'latest.pth')
    if os.path.exists(last_ckpt):
        checkpoint = torch.load(last_ckpt, map_location='cuda')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best))

        if epoch < args.freeze_epochs:
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            for param in model.backbone.parameters():
                param.requires_grad = True

        # ===== 主动学习 =====
        if epoch >= args.active_start_epoch and active_manager.total_selected < active_manager.total_quota:
            model.eval()
            all_confidences = []
            all_u_ids = []

            active_set = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                                     cfg['crop_size'], args.unlabeled_id_path)
            active_loader = DataLoader(active_set, batch_size=cfg['batch_size'],
                                       shuffle=False, num_workers=1)

            with torch.no_grad():
                for img, _, _, _, _, _, u_ids in active_loader:
                    img = img.cuda(non_blocking=True)
                    with autocast(device_type='cuda', dtype=torch.float16):
                        logits = model(img)
                    conf = selector.compute_confidence(logits)
                    all_confidences.append(conf.cpu())
                    all_u_ids.extend(u_ids)

            if all_confidences:
                total_conf = torch.cat(all_confidences)
                remaining_quota = active_manager.total_quota - active_manager.total_selected
                remaining_epochs = cfg['epochs'] - epoch
                max_samples = min(
                    remaining_quota,
                    max(1, math.ceil(remaining_quota / max(1, remaining_epochs)))
                )
                selected_ids = selector.select_global(total_conf, all_u_ids, max_samples)
                moved = active_manager.update_pools(selected_ids)

                logger.info(f"主动学习选择 {moved} 样本，剩余配额 {remaining_quota - moved}")
                writer.add_scalar('active/selected', moved, epoch)
                writer.add_scalar('active/remaining', remaining_quota - moved, epoch)

        # ===== 训练集 =====
        train_l_set = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                                  cfg['crop_size'], args.labeled_id_path, nsample=None)
        train_l_loader = DataLoader(
            train_l_set, batch_size=cfg['batch_size'],
            shuffle=True, num_workers=1, pin_memory=True, drop_last=True
        )

        total_iters = len(train_l_loader) * cfg['epochs']

        total_loss = AverageMeter()

        for i, (img_x, mask_x) in enumerate(train_l_loader):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()

            model.train()
            with autocast(device_type='cuda', dtype=torch.float16):
                preds = model(img_x)
                loss = criterion(preds, mask_x)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss.update(loss.item())

            iters = epoch * len(train_l_loader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            writer.add_scalar('train/loss', loss.item(), iters)

            if (i % (max(1, len(train_l_loader) // 8)) == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        # ===== 验证 =====
        model.eval()
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}'.format(
                cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))

        writer.add_scalar('eval/mIoU', mIoU, epoch)
        for i, iou in enumerate(iou_class):
            writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if is_best:
            torch.save(model.state_dict(), os.path.join(args.save_path, "best_epoch_weights.pth"))

        labeled_num, unlabeled_num, remaining_quota = active_manager.get_status()
        writer.add_scalar('pool/labeled', labeled_num, epoch)
        writer.add_scalar('pool/unlabeled', unlabeled_num, epoch)
        writer.add_scalar('pool/remaining_quota', remaining_quota, epoch)
        logger.info(f'当前数据池状态 | 有标签: {labeled_num} | 无标签: {unlabeled_num}')
        logger.info(f'验证集mIoU: {mIoU:.2f}% (最佳: {previous_best:.2f}%)\n')

if __name__ == '__main__':
    main()
