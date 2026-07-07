import argparse
import os
from tqdm import tqdm  # 导入 tqdm 库
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from model.Hrnets.hrnet import HRnet
from model.semseg.deeplabv3plus import DeepLabV3Plus

from model.models.nets1.deeplabv3_plus import DeepLab
from util.semi import SemiDataset
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log




import logging
import os
import pprint
import traceback
import sys


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


torch.cuda.set_device(0)
def test(model, loader, cfg, save_path):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    correct_pixels = 0
    total_pixels = 0
    class_correct_pixels = [0] * cfg['nclass']
    class_total_pixels = [0] * cfg['nclass']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        # 使用 tqdm 包装数据加载器以显示进度条
        for img, mask, id in tqdm(loader, desc="Testing", unit="batch"):
            img = img.cuda()
            # print(f"[DEBUG] img.shape: {mask.shape}")
            pred = model(img).argmax(dim=1)
            # print(f"[DEBUG] pred.shape: {pred.shape}")
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            intersection_meter.update(intersection)
            union_meter.update(union)
            # filename = valloader.dataset.ids[0] + '.tif'
            filename = id[0].split('/')[-1]
            save_file = os.path.join(save_path, filename)
            pred_img = pred.cpu().numpy()[0]
            pred_img = np.uint8(pred_img)
            pred_img = np.squeeze(pred_img)
            Image.fromarray(pred_img).save(save_file)

            # calculate accuracy
            correct_pixels += (pred == mask.cuda()).sum().item()
            total_pixels += mask.numel()
            for i in range(cfg['nclass']):
                class_correct_pixels[i] += ((pred == i) & (mask.cuda() == i)).sum().item()
                class_total_pixels[i] += (mask == i).sum().item()

    accuracy = correct_pixels / total_pixels
    class_accuracy = [class_correct_pixels[i] / class_total_pixels[i] for i in range(cfg['nclass'])]
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIoU = np.mean(iou_class)
    return mIoU, iou_class, accuracy, class_accuracy

if __name__ == '__main__':
    cfg = {
        'dataset': 'pascal',
        'data_root': r'../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC',

        'nclass': 6,
        'crop_size': 512,
        'pretrained': True,
        'epochs': 200,
        'batch_size': 10,
        'lr': 0.001,
        'lr_multi': 10.0,
        'criterion': {
            'name': 'CELoss',
            'kwargs': {
                'ignore_index': 255
            }
        },
        'conf_thresh': 0.95,
        'downsample_factor': 16,
        # 'model': 'deeplabv3plus',
        # 'backbone': 'xception',
        # 'backbone': 'hrnetv2_w32',
        # 'backbone': 'hrnetv2_w18',
        # 'backbone': 'mobilenet',
        'backbone': 'swin_tiny',
        'img_size': 512,
        # 'backbone': 'hrnet',
        # 'backbone': 'resnet50',
        # 'model_path': 'myEH/d_hrnet/mixpretrained512_1024down_rate1/best_epoch_weights.pth',
        # 'model_path': 'myEH/d_hrnet/random_16_23mix20/best_epoch_weights.pth',
        'model_path': 'myEH/Mix_model_update/deep_sing_EH_512/best_epoch_weights.pth',
        # 'model_path': "pretrained/hrnetv2_w18_imagenet_pretrained.pth",

        'replace_stride_with_dilation': [False, False, True],
        'dilations': [6, 12, 18],
        'use_swin':True,
    }
    parser = argparse.ArgumentParser(description='Test script for semantic segmentation model')
    parser.add_argument('--data-root', type=str, default='../../../../../../home/lin/remote_sense/EH/VOCdevkit')
    # parser.add_argument('--model-path', type=str, default='pretrained/best_epoch_weights1.pth')
    # parser.add_argument('--model-path', type=str, default='myEH/model_update/deep_sing_EH_512/best_epoch_weights.pth')
    parser.add_argument('--model-path', type=str, default='myEH/Mix_model_update/swin_transformer_EH_512/best_epoch_weights.pth')
    parser.add_argument('--save-path', type=str, default='predicted_images_sing_train1')
    # parser.add_argument('--val-id-path', type=str,
    #                     default="../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC/Mix_Train_updata/test.txt")

    # parser.add_argument('--val-id-path', type=str,
    #                     default="../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC/sing_train/test.txt")
    parser.add_argument('--val-id-path', type=str,
                        default="../../../../../../home/lin/remote_sense/EH/VOCdevkit/VOC/VOC16_23Train/val.txt")
    parser.add_argument('--use-swin', action='store_true', help='Enable Transformer branch')

    args = parser.parse_args()

    # Set up the dataset and data loader
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val',cfg['crop_size'],args.val_id_path)

    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    # print(id[0])
    # print(valloader.dataset.ids[0])
    #
    # 
    # 
    # 这里是需要修改的每次的模型不一致


    model = SwinTransformerSeg(
        num_classes=cfg['nclass'],
        backbone=cfg['backbone'],
        pretrained=cfg['pretrained'],
        img_size=cfg['img_size']
    )
    # model = DeepV3Plus(in_channels=3, n_classes=6, backbone="resnet50", pretrained=True)
    # model = DeepLabV3Plus(cfg)
    # model = HRnet(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'])
    # model = HRnet(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'])
    # model = DeepLab(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'], downsample_factor=cfg['downsample_factor'])
    # model = DeepLab(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'], downsample_factor=cfg['downsample_factor'])
    # model = PSPNet(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'], downsample_factor=cfg['downsample_factor'])
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    mIoU, iou_class, accuracy, class_accuracy = test(model, valloader,  cfg,args.save_path)
    # mIOU, iou_class = test(model, valloader, cfg)

    print('Mean IoU:', mIoU)
    print('Class-wise IoU:', iou_class)
    print('Class-wise class_accuracy:', class_accuracy)
    print(f"Mean Accuracy: {accuracy:.4f}")