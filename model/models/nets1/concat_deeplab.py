import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用国内镜像

import timm
from transformers import ViTModel, ViTForImageClassification

# 骨干网络导入（请确保你的相对路径正确）
from model.models.nets1.hrnet import HRNet_Backbone
from model.models.nets1.mobilevit import mobile_vit_small_backbone
from model.models.nets1.resnet import resnet50_backbone
from model.models.nets1.resnext import resnext50_32x4d_backbone
from model.models.nets1.swin_transformer import Swin_Transformer_Backbone
from model.models.nets1.xception import xception
from model.models.nets1.mobilenetv2 import mobilenetv2
from model.models.nets1.hrnet_new import HRNet_Backbone_New
from model.models.nets1.mobilenetv3 import mobilenet_v3_large_backbone
from model.models.nets1.repvgg_new import repvgg_model_convert, repvgg_backbone_new


# ========================================== #
#           Concat (拼接) 融合策略
# ========================================== #
class Concat_Fusion(nn.Module):
    """
    使用 Concat 拼接融合策略。
    拼接后通过 1x1 卷积将通道数还原，确保与下游网络兼容。
    """

    def __init__(self, cnn_channels, swin_channels=256):
        super(Concat_Fusion, self).__init__()
        # Concat 之后通道数会变成 cnn_channels + swin_channels
        # 所以用 1x1 卷积将通道数降维回 cnn_channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(cnn_channels + swin_channels, cnn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, cnn_feat, swin_feat):
        # 1. 空间尺度对齐 (防止由于下采样倍数不一致导致报错)
        if swin_feat.shape[2:] != cnn_feat.shape[2:]:
            swin_feat = F.interpolate(
                swin_feat,
                size=cnn_feat.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        # 2. 在通道维度 (dim=1) 上进行 Concat 拼接
        concat_feat = torch.cat([cnn_feat, swin_feat], dim=1)

        # 3. 经过 1x1 卷积进行通道融合与降维
        out = self.fusion_conv(concat_feat)
        return out


# ========================================== #
#               辅助网络与分支
# ========================================== #
class Swin_Branch(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.swin = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=False,
            features_only=True,
            in_chans=3,
            img_size=512,
            drop_path_rate=0.3,
            window_size=8
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(1792, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        feats_4d = []
        swin_feats = self.swin(x)
        for f in swin_feats:
            f_4d = f.permute(0, 3, 1, 2).contiguous()
            feats_4d.append(f_4d)
        swin_feats = feats_4d

        target_h, target_w = x.shape[2] // 8, x.shape[3] // 8

        c2 = F.interpolate(swin_feats[1], size=(target_h, target_w), mode='bilinear', align_corners=True)
        c3 = F.interpolate(swin_feats[2], size=(target_h, target_w), mode='bilinear', align_corners=True)
        c4 = F.interpolate(swin_feats[3], size=(target_h, target_w), mode='bilinear', align_corners=True)

        fused_feat = torch.cat([c2, c3, c4], dim=1)
        return self.fusion_conv(fused_feat)


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=True, downsample_factor=8):
        super(MobileNetV2, self).__init__()
        from functools import partial
        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        global_feature = torch.mean(x, dim=2, keepdim=True)
        global_feature = torch.mean(global_feature, dim=3, keepdim=True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(
            input=global_feature, size=(row, col), scale_factor=None, mode="bilinear", align_corners=True
        )

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


# ========================================== #
#             DeepLab 网络主体
# ========================================== #
class DeepLab(nn.Module):
    def __init__(
            self,
            num_classes,
            backbone,
            pretrained=False,
            downsample_factor=8,
            backbone_path="",
            use_swin=True
    ):
        super(DeepLab, self).__init__()
        self.use_swin = use_swin
        self.backbone_name = backbone

        # ---------------- 骨干网络选择 ----------------
        if backbone == "xception":
            self.backbone = xception(pretrained, downsample_factor)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            self.backbone = MobileNetV2(pretrained, downsample_factor)
            in_channels = 320
            low_level_channels = 24
        elif backbone == "resnet50":
            self.backbone = resnet50_backbone(pretrained, backbone_path)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "resnext50":
            self.backbone = resnext50_32x4d_backbone(pretrained=False, downsample_factor=8)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "repvgg_new":
            self.backbone = repvgg_backbone_new(model_type="RepVGG-B2g4-new", pretrained=pretrained)
            in_channels = 2560
            low_level_channels = 160
        elif backbone == "hrnet":
            self.backbone = HRNet_Backbone(backbone="hrnetv2_w48", pretrained=pretrained)
            in_channels = 720
            low_level_channels = 256
        elif backbone == "hrnet_new":
            self.backbone = HRNet_Backbone_New(backbone="hrnet_w32", pretrained=pretrained)
            in_channels = 32
            low_level_channels = 256
        elif backbone == "swin_transformer":
            self.backbone = Swin_Transformer_Backbone()
            in_channels = 1024
            low_level_channels = 256
        elif backbone == "mobilevit":
            self.backbone = mobile_vit_small_backbone(model_type="small", pretrained=pretrained)
            in_channels = 640
            low_level_channels = 64
        elif backbone == "mobilenetv3":
            self.backbone = mobilenet_v3_large_backbone(pretrained=pretrained, model_type="large")
            in_channels = 160
            low_level_channels = 40
        else:
            raise ValueError("Unsupported backbone - `{}`.".format(backbone))

        # ========== 核心修改：引入 Concat 融合 ==========
        if use_swin:
            self.swin_branch = Swin_Branch(num_classes, pretrained)
            self.concat_fusion = Concat_Fusion(cnn_channels=in_channels, swin_channels=256)

        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(in_channels=low_level_channels, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.cls_conv = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1, stride=1)

    def forward(self, x, need_fp=False):
        assert x.shape[-2:] == (512, 512), f"输入尺寸错误: {x.shape}"
        H, W = x.size(2), x.size(3)
        raw_x = x

        # 1. 骨干网络提取特征
        if self.backbone_name in ["xception", "mobilenet", "repvgg_new", "hrnet", "swin_transformer", "mobilevit",
                                  "mobilenetv3", "hrnet_new"]:
            _, _, low_level_features, x = self.backbone(x)
        elif self.backbone_name in ["resnet50", "resnext50"]:
            features = self.backbone(x)
            low_level_features = features["low_features"]
            x = features["main"]

        # 2. Swin 分支与 Concat 融合
        if self.use_swin and hasattr(self, 'swin_branch'):
            swin_feat = self.swin_branch(raw_x)
            # 使用 Concat 进行融合降维
            x = self.concat_fusion(x, swin_feat)

        # 3. ASPP模块
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        # 4. 尺寸对齐与拼接
        x = F.interpolate(
            input=x,
            size=(low_level_features.size(2), low_level_features.size(3)),
            mode="bilinear",
            align_corners=True,
        )
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))

        # 5. 输出分类结果
        if need_fp:
            x = self.cls_conv(torch.cat((x, nn.Dropout2d(0.5)(x))))
            outs = F.interpolate(input=x, size=(H, W), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            return out, out_fp

        out = self.cls_conv(x)
        out = F.interpolate(input=out, size=(H, W), mode="bilinear", align_corners=True)
        return out

    def switch_to_deploy(self):
        if self.backbone_name in ["repvgg_new"]:
            self.backbone = repvgg_model_convert(model=self.backbone)
            print(f"\033[1;33;44m 🔬🔬🔬🔬 Switch {self.backbone_name} to deploy model \033[0m")
        else:
            print(f"\033[1;31;41m 🔬🔬🔬🔬 Can not Switch to deploy model \033[0m")


if __name__ == '__main__':
    # 构建测试张量
    input_tensor = torch.randn(2, 3, 512, 512)

    # 实例化基于 HRNet 骨干网络的 DeepLab 模型
    model = DeepLab(num_classes=3, backbone="hrnet", downsample_factor=16, pretrained=False, use_swin=True)

    # 前向传播测试
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")