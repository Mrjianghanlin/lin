import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

import model.backbone.resnet as resnet
from model.backbone.xception import xception
from model.models.model.modules.backbones.resnest import resnest50, resnest101
from model.models.model.modules.backbones.resnet import resnet18
from model.models.model.modules.backbones.resnet_ibn import (
    resnet18_ibn_a, resnet18_ibn_b,
    resnet50_ibn_a, resnet101_ibn_a
)
from model.models.nets1.hrnet import HRNet_Backbone


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()

        # --- Backbone & Channel setting ---
        self.backbone_name = cfg['backbone']
        if cfg['backbone'] in ["resnet50", "resnet101"]:
            self.backbone = resnet.__dict__[cfg['backbone']](
                pretrained=cfg['pretrained'],
                replace_stride_with_dilation=cfg['replace_stride_with_dilation']
            )
            self.low_channels, self.high_channels = 256, 2048

        elif cfg['backbone'] in ["resnet18"]:
            self.backbone = resnet18(pretrained=cfg['pretrained'])
            self.low_channels, self.high_channels = 64, 512

        elif cfg['backbone'] == "resnet18_ibn_a":
            self.backbone = resnet18_ibn_a(pretrained=cfg['pretrained'])
            self.low_channels, self.high_channels = 64, 512

        elif cfg['backbone'] == "resnet18_ibn_b":
            self.backbone = resnet18_ibn_b(pretrained=cfg['pretrained'])
            self.low_channels, self.high_channels = 64, 512

        elif cfg['backbone'] == "resnet50_ibn_a":
            self.backbone = resnet50_ibn_a(pretrained=cfg['pretrained'])
            self.low_channels, self.high_channels = 256, 2048

        elif cfg['backbone'] == "resnet101_ibn_a":
            self.backbone = resnet101_ibn_a(pretrained=cfg['pretrained'])
            self.low_channels, self.high_channels = 256, 2048

        elif cfg['backbone'] == "resnset50":
            self.backbone = resnest50(pretrained=cfg['pretrained'])
            self.low_channels, self.high_channels = 256, 2048

        elif cfg['backbone'] == "resnset101":
            self.backbone = resnest101(pretrained=cfg['pretrained'])
            self.low_channels, self.high_channels = 256, 2048

        elif cfg['backbone'] == "hrnet":
            self.backbone = HRNet_Backbone(backbone="hrnetv2_w32", pretrained=cfg['pretrained'])
            self.low_channels, self.high_channels = 256, 480
            # HRNet 输出不是 2048，需要对齐
            self.align_c4 = nn.Sequential(
                nn.Conv2d(self.high_channels, 2048, 1, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(True)
            )
            self.high_channels = 2048

        else:  # xception
            assert cfg['backbone'] == "xception"
            self.backbone = xception(pretrained=cfg['pretrained'])
            self.low_channels, self.high_channels = 256, 2048

        # --- DeepLab Head ---
        self.head = ASPPModule(self.high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(
            nn.Conv2d(self.low_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(self.high_channels // 8 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

    def forward(self, x, need_fp=False):
        h, w = x.shape[-2:]
        feats = self.backbone.base_forward(x)  # 统一返回 [c1, ..., c4]
        c1, c4 = feats[0], feats[-1]

        # HRNet 特殊处理
        if hasattr(self, "align_c4"):
            c4 = self.align_c4(c4)

        if need_fp:
            outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
                                torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            return out, out_fp
        else:
            out = self._decode(c1, c4)
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
            return out

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)
        c1 = self.reduce(c1)
        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)
        return self.classifier(feature)


# ---------------- ASPP 模块 ---------------- #
def ASPPConv(in_channels, out_channels, atrous_rate):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        feats = [self.b0(x), self.b1(x), self.b2(x), self.b3(x), self.b4(x)]
        return self.project(torch.cat(feats, dim=1))


# ---------------- 测试 ---------------- #
if __name__ == '__main__':
    cfg = {
        'nclass': 6,
        'pretrained': False,
        'backbone': 'resnet18',  # 换成 'hrnet', 'resnet50' 等都能跑
        'replace_stride_with_dilation': [False, False, True],
        'dilations': [6, 12, 18]
    }
    model = DeepLabV3Plus(cfg)
    summary(model, (3, 512, 512), device="cpu")
