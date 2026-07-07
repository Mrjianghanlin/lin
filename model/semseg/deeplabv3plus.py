import torch
from torch import nn
import torch.nn.functional as F

import model.backbone.resnet as resnet
from model.backbone.xception import xception
from model.models.model.modules.backbones.resnest import resnest50, resnest101
from model.models.model.modules.backbones.resnet import resnet18
from model.models.model.modules.backbones.resnet_ibn import (
    resnet18_ibn_a, resnet18_ibn_b,
    resnet50_ibn_a, resnet101_ibn_a
)
from model.models.nets1.hrnet import HRNet_Backbone


# ================= ASPP =================
def ASPPConv(in_channels, out_channels, rate):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        return F.interpolate(self.gap(x), (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, rates):
        super().__init__()
        out_channels = in_channels // 8
        r1, r2, r3 = rates

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.b1 = ASPPConv(in_channels, out_channels, r1)
        self.b2 = ASPPConv(in_channels, out_channels, r2)
        self.b3 = ASPPConv(in_channels, out_channels, r3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        feats = [
            self.b0(x), self.b1(x),
            self.b2(x), self.b3(x),
            self.b4(x)
        ]
        return self.project(torch.cat(feats, dim=1))


# ================= DeepLabV3+ with RC =================
class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # ---------- backbone ----------
        if cfg['backbone'] == "resnet50":
            self.backbone = resnet.resnet50(
                pretrained=cfg['pretrained'],
                replace_stride_with_dilation=cfg['replace_stride_with_dilation']
            )
            low_ch, high_ch = 256, 2048

        elif cfg['backbone'] == "resnet18":
            self.backbone = resnet18(pretrained=cfg['pretrained'])
            low_ch, high_ch = 64, 512

        elif cfg['backbone'] == "hrnet":
            self.backbone = HRNet_Backbone("hrnetv2_w32", pretrained=cfg['pretrained'])
            low_ch, high_ch = 256, 480
            self.align_c4 = nn.Sequential(
                nn.Conv2d(480, 2048, 1, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(True)
            )
            high_ch = 2048

        else:
            raise ValueError("Unsupported backbone")

        # ---------- decoder ----------
        self.head = ASPPModule(high_ch, cfg['dilations'])

        self.reduce = nn.Sequential(
            nn.Conv2d(low_ch, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(high_ch // 8 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1)

        # ================= RC: 区域中心记忆库 =================
        self.num_classes = cfg['nclass']
        self.rc_dim = 256
        self.rc_momentum = 0.9

        self.register_buffer(
            "region_centers",
            torch.zeros(self.num_classes, self.rc_dim)
        )

    # ---------- forward ----------
    def forward(self, x, return_feat=False):
        h, w = x.shape[-2:]

        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]

        if hasattr(self, "align_c4"):
            c4 = self.align_c4(c4)

        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)
        c1 = self.reduce(c1)

        feat = self.fuse(torch.cat([c1, c4], dim=1))
        out = self.classifier(feat)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        if return_feat:
            return out, feat
        return out

    # ================= RC: 更新区域中心 =================
    @torch.no_grad()
    def update_region_centers(self, feats, preds):
        """
        feats: [B, 256, H, W]
        preds: [B, H, W]
        """
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(-1, C)
        preds = preds.reshape(-1)

        for cls in range(self.num_classes):
            mask = preds == cls
            if mask.sum() < 20:
                continue
            cls_feat = feats[mask].mean(dim=0)
            self.region_centers[cls] = (
                self.rc_momentum * self.region_centers[cls]
                + (1 - self.rc_momentum) * cls_feat
            )

    # ================= RC loss =================
    def rc_loss(self, feats, preds):
        """
        feats: [B, 256, H, W]
        preds: [B, H, W]
        """
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(-1, C)
        preds = preds.reshape(-1)

        loss, cnt = 0.0, 0
        for cls in range(self.num_classes):
            mask = preds == cls
            if mask.sum() < 20:
                continue
            center = self.region_centers[cls].detach()
            loss += F.mse_loss(feats[mask], center.expand_as(feats[mask]))
            cnt += 1

        if cnt == 0:
            return torch.tensor(0.0, device=feats.device)
        return loss / cnt
