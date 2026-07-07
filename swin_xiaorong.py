import argparse
import logging
import os
import pprint
import traceback
import sys

import torch
import numpy as np
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.semi import SemiDataset
from util.classes import CLASSES
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log

import timm
from timm.models.swin_transformer import SwinTransformer


def pad_to_window_size(x, window_size=7, downsample=32):
    """
    Pad input to be divisible by window_size * downsample
    """
    B, C, H, W = x.shape
    factor = window_size * downsample
    pad_h = (factor - H % factor) % factor
    pad_w = (factor - W % factor) % factor
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
    return x



# ==================== 模型定义 ====================
class MultiStageSwinSeg(nn.Module):
    def __init__(self, num_classes=6,
                 selected_stages=[2, 3],
                 fusion_type='concat',
                 pretrained_path=None):
        super().__init__()

        self.selected_stages = selected_stages
        self.fusion_type = fusion_type

        # ===== Swin backbone（不联网）=====
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,  # ❗关键：禁止联网
            features_only=True,
            out_indices=(0, 1, 2, 3),
            dynamic_img_size=True

        )
        # ===== 关键：关闭 PatchEmbed 尺寸断言 =====
        self._disable_patch_embed_size_check()

        # ===== 加载本地 ImageNet 权重 =====
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location='cpu')

            # 兼容 timm / 官方格式
            if 'model' in state_dict:
                state_dict = state_dict['model']

            missing, unexpected = self.backbone.load_state_dict(
                state_dict, strict=False
            )

            print(f"✅ Loaded ImageNet pretrained from: {pretrained_path}")
            print(f"   Missing keys: {len(missing)} (正常)")
            print(f"   Unexpected keys: {len(unexpected)} (正常)")

        stage_channels = [96, 192, 384, 768]

        # 对齐通道
        self.align_convs = nn.ModuleList([
            nn.Conv2d(stage_channels[s], 256, kernel_size=1)
            for s in selected_stages
        ])

        in_ch = 256 * len(selected_stages) if fusion_type == 'concat' else 256

        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(32, num_classes, 1)

        self.stage_weights = nn.Parameter(
            torch.ones(len(selected_stages))
        )

    # ------------------ 关闭尺寸检查 ------------------
    def _disable_patch_embed_size_check(self):
        from timm.layers.patch_embed import PatchEmbed
        for m in self.backbone.modules():
            if isinstance(m, PatchEmbed):
                m.img_size = None   # 彻底关闭 PatchEmbed 的 assert

    def forward(self, x):
        orig_H, orig_W = x.shape[2], x.shape[3]  # 保存原始大小

        # 先 pad 到 window_size 的倍数
        x = pad_to_window_size(x, window_size=7)
        B, C, H, W = x.shape
        # print(f"[DEBUG] Padded input shape: {x.shape}")

        # backbone 提取特征
        features = self.backbone(x)
        feats = [features[s] for s in self.selected_stages]

        # ====== 修改后（✅ 正确 & 稳定）======
        for i in range(len(feats)):
            feats[i] = feats[i].permute(0, 3, 1, 2).contiguous()

            # print(f"[DEBUG] Permuted feature {i} to NCHW: {feats[i].shape}")

        # 对齐通道
        aligned_feats = []
        target_size = feats[0].shape[-2:]
        for f, conv in zip(feats, self.align_convs):
            f = conv(f)
            if f.shape[-2:] != target_size:
                f = nn.functional.interpolate(
                    f, size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            aligned_feats.append(f)

        # 特征融合
        if self.fusion_type == 'concat':
            fused = torch.cat(aligned_feats, dim=1)

        elif self.fusion_type == 'add':
            fused = sum(aligned_feats)

        elif self.fusion_type == 'weighted_sum':
            weights = torch.softmax(self.stage_weights, dim=0)
            fused = sum(w * f for w, f in zip(weights, aligned_feats))

        # decoder
        x = self.decoder(fused)

        # ✅ 插值回原始 H/W
        if x.shape[-2:] != (orig_H, orig_W):
            x = nn.functional.interpolate(
                x, size=(orig_H, orig_W),
                mode='bilinear',
                align_corners=False
            )

        # 分类头
        return self.classifier(x)


# ==================== 验证函数 ====================
def evaluate(model, loader, mode, cfg):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    total = 0
    correct = 0

    with torch.no_grad():
        loader = tqdm(loader, desc="Evaluating")
        for i, (img, mask, _) in enumerate(loader):
            img = img.cuda()
            pred = model(img).argmax(dim=1)
            intersection, union, target = intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            intersection_meter.update(intersection)
            union_meter.update(union)
            total += img.size(0) * img.size(2) * img.size(3)
            correct += (pred == mask.cuda()).sum().item()

    acc = correct / total if total != 0 else 0
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIoU = np.mean(iou_class)
    print(f"Mean Accuracy: {acc:.4f}")
    return mIoU, iou_class


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser()




    parser = argparse.ArgumentParser(
        description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
    parser.add_argument('--labeled-id-path', type=str,
                        default="../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC/Mix_Train_updata/train.txt")
    parser.add_argument('--val-id-path', type=str,
                        default="../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC/Mix_Train_updata/val.txt")
    parser.add_argument('--save-path', type=str, default="myEH/swin_transformer111")
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)
    parser.add_argument('--freeze-epochs', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    if sys.platform != 'win32':
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)

    cfg = {
        'dataset': 'pascal',
        'data_root': r'../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC',
        'nclass': 6,
        'crop_size': 512,

        'epochs': 200,

        'batch_size': 8,
        'lr': 0.0001,
    }

    # 实验配置
    all_configs = [
        {'selected_stages':[3], 'fusion_type':'concat'},
        # {'selected_stages':[2,3], 'fusion_type':'concat'},
        # {'selected_stages':[1,3], 'fusion_type':'concat'},
        # {'selected_stages':[1,2,3], 'fusion_type':'concat'},
        {'selected_stages':[0,1,2,3], 'fusion_type':'concat'},
        {'selected_stages':[0,1,2,3], 'fusion_type':'weighted_sum'},
        {'selected_stages':[0,1,2,3], 'fusion_type':'add'},
    ]
    experiment_names = [
        'Baseline_stage3_only',
        # 'Baseline_stage23_concat',
        # 'Baseline_stage13_concat',
        # 'Baseline_stage123_concat',
        'Baseline_stage0123_concat',
        'Baseline_stage0123_weighted_sum',
        'Baseline_stage0123_add',
    ]

    for exp_idx, exp_cfg in enumerate(all_configs):
        exp_name = experiment_names[exp_idx]
        save_path = os.path.join(args.save_path, exp_name)
        os.makedirs(save_path, exist_ok=True)
        logger = init_log(exp_name, logging.INFO)
        logger.propagate = 0
        writer = SummaryWriter(save_path)
        logger.info(f"Running experiment: {exp_name}\nConfig: {exp_cfg}")

        # 模型
        model = MultiStageSwinSeg(
            num_classes=cfg['nclass'],
            selected_stages=exp_cfg['selected_stages'],
            fusion_type=exp_cfg['fusion_type'],
            pretrained_path='pretrained/swin_tiny_patch4_window7_224.pth'
        ).cuda()

        logger.info(f"Model params: {count_params(model):.2f}M")

        optimizer = AdamW(model.parameters(), lr=cfg['lr'])
        criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()

        trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path)
        valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', cfg['crop_size'], args.val_id_path)
        trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
        valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)

        previous_best = 0.0
        for epoch in range(cfg['epochs']):
            model.train()
            total_loss = AverageMeter()
            for batch_idx, (img, mask) in enumerate(trainloader):
                img, mask = img.cuda(), mask.cuda()
                pred = model(img)
                loss = criterion(pred, mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss.update(loss.item())

            # 验证
            mIoU, iou_class = evaluate(model, valloader, 'original', cfg)
            logger.info(f"Epoch {epoch} - Loss: {total_loss.avg:.4f}, mIoU: {mIoU:.4f}")
            # 记录各类别的IoU
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('Class [{:} {:}] IoU: {:.2f}'.format(
                    cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('MeanIoU: {:.2f}\n'.format(mIoU))

            # 保存模型
            is_best = mIoU > previous_best
            previous_best = max(mIoU, previous_best)
            checkpoint_path = os.path.join(save_path, "latest.pth")
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'previous_best': previous_best}, checkpoint_path)
            if is_best:
                best_path = os.path.join(save_path, f"best_mIoU_{mIoU:.4f}.pth")
                torch.save(model.state_dict(), best_path)
                logger.info(f"Saved best model at epoch {epoch} with mIoU {mIoU:.4f}")

        writer.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
