import argparse
import logging
import os
import pprint
import traceback
import sys

import torch
import numpy as np
from torch import nn
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.semi import SemiDataset
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log

# 导入Swin Transformer相关库
import timm
from timm.models.swin_transformer import SwinTransformer


# 基于Swin Transformer的语义分割模型
class SwinTransformerSeg(nn.Module):
    def __init__(self, num_classes=6, backbone='swin_tiny', pretrained=True, img_size=512):
        super().__init__()

        print(f"Initializing Swin Transformer with backbone: {backbone}, img_size: {img_size}")

        # 根据backbone类型设置参数
        if backbone == 'swin_tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
        elif backbone == 'swin_small':
            embed_dim = 96
            depths = [2, 2, 18, 2]
            num_heads = [3, 6, 12, 24]
        elif backbone == 'swin_base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
        elif backbone == 'swin_large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 创建Swin Transformer骨干网络
        self.backbone = SwinTransformer(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            num_classes=0,  # 不要分类头
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim * 8, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        # 分类头
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

        # 初始化权重
        if not pretrained:
            self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        # 提取特征
        features = self.backbone.forward_features(x)  # (B, H/32, W/32, C)

        # 转换形状: (B, H, W, C) -> (B, C, H, W)
        features = features.permute(0, 3, 1, 2).contiguous()

        # 解码
        x = self.decoder(features)

        # 上采样到输入尺寸
        if x.shape[-2:] != (H, W):
            x = nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        # 分类
        x = self.classifier(x)

        return x


# 主程序
parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--labeled-id-path', type=str,
                    default="../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC/Mix_Train_updata/train.txt")
parser.add_argument('--val-id-path', type=str,
                    default="../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC/Mix_Train_updata/val.txt")
parser.add_argument('--save-path', type=str, default="myEH/Mix_model_update/swin_transformer_EH_512")
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--freeze-epochs', type=int, default=0)

# 设置CUDA
torch.cuda.set_device(0)
# 移除可能导致问题的环境变量
if 'CUDA_LAUNCH_BLOCKING' in os.environ:
    del os.environ['CUDA_LAUNCH_BLOCKING']

# 添加多进程启动方法设置
if sys.platform != 'win32':
    import multiprocessing

    multiprocessing.set_start_method('spawn', force=True)


def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    total = 0
    correct = 0
    with torch.no_grad():
        loader = tqdm(loader, desc="Evaluating")
        for i, (img, mask, id) in enumerate(loader):
            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, cfg['nclass'], h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)
            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img).argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            intersection_meter.update(intersection)
            union_meter.update(union)
            total += img.size(0) * img.size(2) * img.size(3)
            correct += (pred == mask.cuda()).sum().item()

    if total == 0:
        acc = 0
    else:
        acc = correct / total

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)
    print(f"Mean Accuracy: {acc:.4f}")

    return mIOU, iou_class


def main():
    args = parser.parse_args()

    cfg = {
        'dataset': 'pascal',
        'data_root': r'../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC',
        'nclass': 6,
        'crop_size': 512,
        'pretrained': True,
        'epochs': 200,
        'batch_size': 12,
        'lr': 0.0001,
        'lr_multi': 10.0,
        'criterion': {
            'name': 'CELoss',
            'kwargs': {
                'ignore_index': 255
            }
        },
        'conf_thresh': 0.95,
        'backbone': 'swin_tiny',
        'img_size': 512,
        'window_size': 7,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'model_path': "",
        'num_workers': 0,  # 设置为0，避免多进程问题
        'in_channels': 3,
    }

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))

    writer = SummaryWriter(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)

    # 创建模型
    print("=" * 60)
    print("Creating model...")
    model = SwinTransformerSeg(
        num_classes=cfg['nclass'],
        backbone=cfg['backbone'],
        pretrained=cfg['pretrained'],
        img_size=cfg['img_size']
    )
    print("Model created successfully!")
    print("=" * 60)

    # 测试前向传播
    print("\nTesting forward pass...")
    test_input = torch.randn(2, 3, cfg['crop_size'], cfg['crop_size'])
    with torch.no_grad():
        test_output = model(test_input)
        print(f"✓ Forward test passed!")
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {test_output.shape}")

    # 将模型移到GPU
    model = model.cuda()

    # 统计参数量
    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    # 优化器
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    # 损失函数
    criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()

    # 数据集
    trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path)
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', cfg['crop_size'], args.val_id_path)

    # 数据加载器 - 使用更安全的设置
    trainloader = DataLoader(
        trainset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        pin_memory=False,  # 禁用pin_memory
        num_workers=0,  # 不使用多进程
        drop_last=True,
        persistent_workers=False
    )

    valloader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,  # 禁用pin_memory
        num_workers=0,  # 不使用多进程
        drop_last=False,
        persistent_workers=False
    )

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1

    # 加载检查点
    checkpoint_path = os.path.join(args.save_path, 'latest.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    else:
        print("No checkpoint found. Starting from scratch.")

    # 训练循环
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best))

        # 冻结策略
        if epoch < args.freeze_epochs:
            for name, param in model.named_parameters():
                if 'backbone' in name:
                    param.requires_grad = False
            print(f"Frozen backbone for epoch {epoch}")
        else:
            for param in model.parameters():
                param.requires_grad = True

        model.train()
        total_loss = AverageMeter()

        # 使用更简单的训练循环
        for batch_idx, (img, mask) in enumerate(trainloader):
            try:
                # 将数据移到GPU
                img, mask = img.cuda(), mask.cuda()

                # 前向传播
                pred = model(img)
                loss = criterion(pred, mask)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 记录损失
                total_loss.update(loss.item())
                iters = epoch * len(trainloader) + batch_idx

                # 学习率调度
                lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]['lr'] = lr

                # 记录到tensorboard
                writer.add_scalar('train/loss', loss.item(), iters)

                # 定期打印日志
                if batch_idx % 100 == 0:
                    logger.info('Epoch: [{}/{}], Batch: [{}/{}], Loss: {:.4f}, LR: {:.6f}'.format(
                        epoch, cfg['epochs'], batch_idx, len(trainloader),
                        loss.item(), optimizer.param_groups[0]['lr']))

            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                print("Skipping this batch and continuing...")
                traceback.print_exc()
                continue

        # 每个epoch后更新学习率
        scheduler.step()

        # 验证
        print(f"\n{'=' * 40}")
        print(f"Validating epoch {epoch}...")
        print(f"{'=' * 40}")

        model.eval()
        eval_mode = 'original'  # 使用原始模式
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        # 记录各类别的IoU
        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('Class [{:} {:}] IoU: {:.2f}'.format(
                cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        logger.info('MeanIoU: {:.2f}\n'.format(mIoU))

        # 记录到tensorboard
        writer.add_scalar('eval/mIoU', mIoU, epoch)
        for i, iou in enumerate(iou_class):
            writer.add_scalar(f'eval/{CLASSES[cfg["dataset"]][i]}_IoU', iou, epoch)

        # 保存检查点
        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
        }

        # 保存最新的检查点
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_model_path = os.path.join(args.save_path, "best_epoch_weights.pth")
            torch.save(model.state_dict(), best_model_path)
            best_checkpoint_filename = os.path.join(args.save_path, f"best_epoch_weights_val_mIoU_{mIoU:.4f}.pth")
            torch.save(model.state_dict(), best_checkpoint_filename)
            print(f"✓ Saved best model with mIoU: {mIoU:.4f}")

        print(f"✓ Epoch {epoch} completed. Loss: {total_loss.avg:.4f}, mIoU: {mIoU:.4f}")
        print(f"{'=' * 60}\n")

        # 定期清理缓存
        if epoch % 10 == 0:
            torch.cuda.empty_cache()

    print("Training completed!")
    writer.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)