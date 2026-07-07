import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tabulate import tabulate
import os

# 新增：引入ViT模型
from transformers import ViTModel
from model.models.nets1.hrnet import HRNet_Backbone
# from models.nets1.mobilenetv3 import mobilenet_v3_large_backbone
from model.models.nets1.mobilevit import mobile_vit_small_backbone
# from models.nets1.repvgg_new import repvgg_backbone_new, repvgg_model_convert
from model.models.nets1.resnet import resnet50_backbone
from model.models.nets1.resnext import resnext50_32x4d_backbone
from model.models.nets1.swin_transformer import Swin_Transformer_Backbone
from model.models.nets1.xception import xception
from model.models.nets1.mobilenetv2 import mobilenetv2

from model.models.nets1.hrnet_new import HRNet_Backbone_New
from model.models.nets1.mobilenetv3 import mobilenet_v3_large_backbone

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification
from transformers import ViTModel
from timm.models.swin_transformer import SwinTransformer
from model.models.nets1.repvgg_new import repvgg_model_convert, repvgg_backbone_new
import timm


class Swin_Branch_Configurable(nn.Module):
    """可配置的Swin分支，支持消融实验"""

    def __init__(self, num_classes, pretrained=True, selected_stages=[1, 2, 3],
                 fusion_channels=256, use_attention=False, fusion_type='concat',
                 pretrained_path=None, **kwargs):
        """
        Args:
            selected_stages: 选择哪些stage的特征，列表形式，如[1,2,3]表示使用stage1,2,3
                            索引对应关系: 0-stage0, 1-stage1, 2-stage2, 3-stage3
            fusion_channels: 融合后的通道数
            use_attention: 是否使用通道注意力
            fusion_type: 融合方式 ['concat', 'add', 'weighted_sum']
            pretrained_path: 本地预训练权重路径
        """
        super().__init__()
        self.selected_stages = selected_stages
        self.num_selected = len(selected_stages)
        self.fusion_type = fusion_type
        self.use_attention = use_attention

        # 每个stage的输出通道数（swin_base_patch4_window7_224）
        stage_channels = {
            0: 128,  # stage0: [B, 128, H/4, W/4]
            1: 256,  # stage1: [B, 256, H/8, W/8]
            2: 512,  # stage2: [B, 512, H/16, W/16]
            3: 1024  # stage3: [B, 1024, H/32, W/32]
        }

        # 计算融合的总通道数
        total_channels = 0
        for stage_idx in selected_stages:
            if stage_idx in stage_channels:
                total_channels += stage_channels[stage_idx]
            else:
                raise ValueError(f"Invalid stage index: {stage_idx}. Must be 0,1,2,3")

        # 尝试从本地加载预训练权重
        if pretrained and pretrained_path and os.path.exists(pretrained_path):
            print(f"📁 尝试从本地加载Swin权重: {pretrained_path}")
            self.swin = self._create_swin_from_local(pretrained_path)
        elif pretrained:
            # 如果没有指定路径，尝试在线下载
            try:
                print("🌐 尝试在线下载Swin预训练权重...")
                self.swin = timm.create_model(
                    'swin_base_patch4_window7_224',
                    pretrained=True,
                    features_only=True,
                    in_chans=3,
                    img_size=512,
                    drop_path_rate=0.3,
                    window_size=8
                )
                print("✅ Swin预训练权重在线下载成功")
            except Exception as e:
                print(f"⚠️ 在线下载失败: {e}")
                print("使用随机初始化的Swin模型")
                self.swin = timm.create_model(
                    'swin_base_patch4_window7_224',
                    pretrained=False,
                    features_only=True,
                    in_chans=3,
                    img_size=512,
                    drop_path_rate=0.3,
                    window_size=8
                )
        else:
            # 随机初始化
            self.swin = timm.create_model(
                'swin_base_patch4_window7_224',
                pretrained=False,
                features_only=True,
                in_chans=3,
                img_size=512,
                drop_path_rate=0.3,
                window_size=8
            )

        # 不同融合策略
        if fusion_type == 'concat':
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(total_channels, fusion_channels, 1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            )
        elif fusion_type == 'add':
            # 需要先调整到相同通道数
            self.adjust_convs = nn.ModuleList()
            for stage_idx in selected_stages:
                if stage_idx in stage_channels:
                    self.adjust_convs.append(
                        nn.Conv2d(stage_channels[stage_idx], fusion_channels, 1)
                    )
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(fusion_channels, fusion_channels, 1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            )
        elif fusion_type == 'weighted_sum':
            # 加权融合
            self.weights = nn.Parameter(torch.ones(self.num_selected) / self.num_selected)
            self.adjust_convs = nn.ModuleList()
            for stage_idx in selected_stages:
                if stage_idx in stage_channels:
                    self.adjust_convs.append(
                        nn.Conv2d(stage_channels[stage_idx], fusion_channels, 1)
                    )
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(fusion_channels, fusion_channels, 1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            )

        # # 通道注意力
        # if use_attention:
        #     self.channel_attention = nn.Sequential(
        #         nn.AdaptiveAvgPool2d(1),
        #         nn.Conv2d(total_channels, total_channels // 16, 1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(total_channels // 16, total_channels, 1),
        #         nn.Sigmoid()
        #     )
        # 通道注意力
        if use_attention:
            att_channels = fusion_channels  # ⭐ 关键改动
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(att_channels, att_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(att_channels // 4, att_channels, 1),
                nn.Sigmoid()
            )

        # 记录配置
        self.config = {
            'selected_stages': selected_stages,
            'total_channels': total_channels,
            'fusion_channels': fusion_channels,
            'use_attention': use_attention,
            'fusion_type': fusion_type
        }

        print(f"🚀 Swin_Branch 配置:")
        print(f"  - 选中的阶段: {selected_stages}")
        print(f"  - 总输入通道: {total_channels}")
        print(f"  - 输出通道: {fusion_channels}")
        print(f"  - 融合方式: {fusion_type}")
        print(f"  - 使用注意力: {use_attention}")
        print(f"  - 使用预训练: {pretrained}")

    def _create_swin_from_local(self, weight_path):
        """从本地文件创建Swin模型"""
        try:
            # 先创建不预训练的模型
            model = timm.create_model(
                'swin_base_patch4_window7_224',
                pretrained=False,
                features_only=True,
                in_chans=3,
                img_size=512,
                drop_path_rate=0.3,
                window_size=8
            )

            # 加载本地权重
            state_dict = torch.load(weight_path, map_location='cpu')

            # 处理state_dict的key
            if 'model' in state_dict:
                # 如果是包含'model'键的state_dict
                state_dict = state_dict['model']

            # 移除可能的module前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # 移除'module.'
                elif k.startswith('base_model.'):
                    k = k[11:]  # 移除'base_model.'
                new_state_dict[k] = v

            # 尝试加载权重
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

            if missing_keys:
                print(f"⚠️ 缺少的keys数量: {len(missing_keys)}")
                if len(missing_keys) < 10:  # 只显示前几个
                    print(f"  示例: {missing_keys[:5]}")

            if unexpected_keys:
                print(f"⚠️ 多余的keys数量: {len(unexpected_keys)}")
                if len(unexpected_keys) < 10:  # 只显示前几个
                    print(f"  示例: {unexpected_keys[:5]}")

            print("✅ Swin本地权重加载成功")
            return model

        except Exception as e:
            print(f"❌ 本地权重加载失败: {e}")
            print("使用随机初始化的Swin模型")
            return timm.create_model(
                'swin_base_patch4_window7_224',
                pretrained=False,
                features_only=True,
                in_chans=3,
                img_size=512,
                drop_path_rate=0.3,
                window_size=8
            )

    def forward(self, x):
        B, C, H, W = x.shape

        # 获取所有stage的特征
        swin_feats_3d = self.swin(x)  # 返回的是[B, H, W, C]格式

        # 转换维度到[B, C, H, W]
        swin_feats = []
        for f in swin_feats_3d:
            f_4d = f.permute(0, 3, 1, 2).contiguous()
            swin_feats.append(f_4d)

        # 获取目标尺寸（默认为输入尺寸的1/8）
        target_h, target_w = H // 8, W // 8

        # 选择并处理指定stage的特征
        selected_features = []

        for idx, stage_idx in enumerate(self.selected_stages):
            if stage_idx < len(swin_feats):
                feat = swin_feats[stage_idx]

                # 上采样到目标尺寸
                if feat.shape[2:] != (target_h, target_w):
                    feat = F.interpolate(feat, size=(target_h, target_w),
                                         mode='bilinear', align_corners=True)

                selected_features.append(feat)

        # 根据不同融合策略处理
        if self.fusion_type == 'concat':
            fused_feat = torch.cat(selected_features, dim=1)
        elif self.fusion_type == 'add':
            adjusted_feats = []
            for i, feat in enumerate(selected_features):
                adjusted = self.adjust_convs[i](feat)
                adjusted_feats.append(adjusted)
            fused_feat = torch.stack(adjusted_feats, dim=0).sum(dim=0)
        elif self.fusion_type == 'weighted_sum':
            adjusted_feats = []
            for i, feat in enumerate(selected_features):
                adjusted = self.adjust_convs[i](feat)
                adjusted_feats.append(adjusted)
            # 加权求和
            adjusted_stack = torch.stack(adjusted_feats, dim=0)
            weights = F.softmax(self.weights, dim=0).view(-1, 1, 1, 1, 1)
            fused_feat = (adjusted_stack * weights).sum(dim=0)

        # 通道注意力
        if self.use_attention and hasattr(self, 'channel_attention'):
            channel_weights = self.channel_attention(fused_feat)
            fused_feat = fused_feat * channel_weights

        # 最终融合
        output = self.fusion_conv(fused_feat)

        return output


class CAFM(nn.Module):
    """优化后的耦合注意力融合模块"""

    def __init__(self, channels):
        super().__init__()
        # 动态调整融合权重
        self.gamma = nn.Parameter(torch.zeros(1))
        # 通道压缩
        self.compression = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU()
        )
        # 注意力生成
        self.attention = nn.Sequential(
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        self.swin_align_conv = nn.Conv2d(256, 720, kernel_size=1)
        self.att_map_conv = nn.Sequential(
            nn.Conv2d(720, 720, kernel_size=1),  # 可以看成通道注意力
            nn.Sigmoid()
        )
        # 融合权重（可学习的缩放因子）
        self.gamma = nn.Parameter(torch.zeros(1))

    # 修改forward方法
    def forward(self, cnn_feat, swin_feat):
        # print(f"🔄 CAFM模块输入:")
        # print(f"  ├─CNN特征: {cnn_feat.shape} (主干网络输出)")
        # print(f"  ├─Swin特征: {swin_feat.shape} (Swin分支输出)")
        swin_feat = self.swin_align_conv(swin_feat)  # shape 变为 [6, 720, 128, 128]
        # print(f"  ├─对齐后Swin特征: {swin_feat.shape} (1x1卷积调整通道)")
        # 2. 生成注意力图（根据 CNN 特征）
        att_map = self.att_map_conv(cnn_feat)  # [B, 720, H, W]，值域在 [0, 1]
        # 残差连接避免特征淹没
        # print(f"  ├─注意力图: {att_map.shape}")

        output = cnn_feat + self.gamma * att_map * swin_feat
        # print(f"  └─融合后特征: {output.shape} (残差连接)")
        return cnn_feat + self.gamma * att_map * swin_feat


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features


class _DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPBlock, self).__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
                                      norm_layer, norm_kwargs)
        self.SP = StripPooling(320, up_kwargs={'mode': 'bilinear', 'align_corners': True})

    def forward(self, x):
        x1 = self.SP(x)
        aspp3 = self.aspp_3(x)

        x = torch.cat([aspp3, x], dim=1)

        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)

        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)

        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)

        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)
        x = torch.cat([x, x1], dim=1)

        return x


# -----------------------------------------#
#   SP条形池化模块，输入通道=输出通道=320
# -----------------------------------------#
class StripPooling(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))  # 1*W
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))  # H*1
        inter_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, bias=False),
                                   nn.BatchNorm2d(in_channels))
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(x)
        x2 = F.interpolate(self.conv2(self.pool1(x1)), (h, w), **self._up_kwargs)  # 结构图的1*W的部分
        x3 = F.interpolate(self.conv3(self.pool2(x1)), (h, w), **self._up_kwargs)  # 结构图的H*1的部分
        x4 = self.conv4(F.relu_(x2 + x3))  # 结合1*W和H*1的特征
        out = self.conv5(x4)
        return F.relu_(x + out)  # 将输出的特征与原始输入特征结


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


# -----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(
            self, dim_in, dim_out, rate=1, bn_mom=0.1
    ):  # dim_in=2048, dim_out=256, rate=2
        super(ASPP, self).__init__()
        # Conv1x1 branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=rate,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Conv3x3 branch dilation=6 * 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                padding=6 * rate,
                dilation=6 * rate,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Conv3x3 branch dilation=12 * 2
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                padding=12 * rate,
                dilation=12 * rate,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Conv3x3 branch dilation=18 * 2
        self.branch4 = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                padding=18 * rate,
                dilation=18 * rate,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Conv1x1 branch 全局平均池化层
        self.branch5_conv = nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.branch5_bn = nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        # 对ASPP模块concat后的结果进行卷积操作（降低维度）
        self.conv_cat = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_out * 5,
                out_channels=dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()  # x(bs,2048,64,64)
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)  # conv1x1(bs, 256, 64, 64)
        conv3x3_1 = self.branch2(x)  # conv3x3_1(bs, 256, 64, 64)
        conv3x3_2 = self.branch3(x)  # conv3x3_2(bs, 256, 64, 64)
        conv3x3_3 = self.branch4(x)  # conv3x3_3(bs, 256, 64, 64)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(
            input=x, dim=2, keepdim=True
        )  # global_feature(bs, 2048, 1, 64)
        global_feature = torch.mean(
            input=global_feature, dim=3, keepdim=True
        )  # global_feature(bs, 2048, 1, 1)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(
            global_feature
        )  # global_feature(bs, 256, 1, 1)
        global_feature = F.interpolate(
            input=global_feature,
            size=(row, col),
            scale_factor=None,
            mode="bilinear",
            align_corners=True,
        )  # global_feature(bs, 256, 64, 64)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征
        # -----------------------------------------#
        feature_cat = torch.cat(
            [conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1
        )  # feature_cat(bs, 1280, 64, 64)
        result = self.conv_cat(feature_cat)  # result(bs, 256, 64, 64)
        return result


class DeepLab(nn.Module):
    def __init__(
            self,
            num_classes,
            backbone,
            pretrained=False,
            downsample_factor=8,
            backbone_path="",
            use_swin=True,  # 启用Swin分支
            swin_config=None,  # Swin分支配置
            swin_pretrained_path='pretrained/swin_base_patch4_window7_224.pth'  # Swin本地权重路径
    ):
        super(DeepLab, self).__init__()
        self.use_swin = use_swin
        self.backbone_name = backbone

        # 默认Swin配置
        default_swin_config = {
            'selected_stages': [1, 2, 3],
            'fusion_channels': 256,
            'use_attention': False,
            'fusion_type': 'concat'
        }

        # 更新配置
        if swin_config is None:
            self.swin_config = default_swin_config
        else:
            self.swin_config = default_swin_config.copy()
            self.swin_config.update(swin_config)

        if backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = xception(pretrained, downsample_factor)
            in_channels = 2048  # 主干部分的特征 (2048,30,30)
            low_level_channels = 256  # 浅层特征 (256,128,128)
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(pretrained, downsample_factor)
            in_channels = 320  # 主干部分的特征(320,30,30)
            low_level_channels = 24  # 浅层特征(24,128,128)
        elif backbone == "resnet50":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [2048,H/8,W/8]
            #   浅层特征    [256,H/4,W/4]
            # ----------------------------------#
            self.backbone = resnet50_backbone(pretrained, backbone_path)
            in_channels = 2048  # 主干部分的特征
            low_level_channels = 256  # 浅层次特征

        elif backbone == "resnext50":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [2048,H/8,W/8]
            #   浅层特征    [256,H/4,W/4]
            # ----------------------------------#
            self.backbone = resnext50_32x4d_backbone(
                pretrained=False, downsample_factor=8
            )

            in_channels = 2048  # 主干部分的特征
            low_level_channels = 256  # 浅层次特征

        elif backbone == "repvgg_new":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [2560,H/8,W/8]
            #   浅层特征    [160,H/4,W/4]
            # ----------------------------------#
            self.backbone = repvgg_backbone_new(model_type="RepVGG-B2g4-new", pretrained=pretrained)
            in_channels = 2560  # 主干部分的特征
            low_level_channels = 160  # 浅层次特征

        elif backbone == "hrnet":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [480,H/8,W/8]
            #   浅层特征    [256,H/4,W/4]
            # ----------------------------------#
            self.backbone = HRNet_Backbone(backbone="hrnetv2_w48", pretrained=pretrained)
            # in_channels = 480  # 主干部分的特征
            in_channels = 720  # 主干部分的特征
            low_level_channels = 256  # 浅层次特征
            # low_level_features=64
            # the_three_features=128

        elif backbone == "hrnet_new":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [32,H/4,W/4]
            #   浅层特征    [256,H/4,W/4]
            #   注意：hrnet_new 的深浅层次融合特征尺寸是相同的
            # ----------------------------------#
            self.backbone = HRNet_Backbone_New(backbone="hrnet_w32", pretrained=pretrained)
            in_channels = 32  # 主干部分的特征
            low_level_channels = 256  # 浅层次特征

        elif backbone == "swin_transformer":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [1024,H/8,W/8]
            #   浅层特征    [256,H/4,W/4]
            # ----------------------------------#
            self.backbone = Swin_Transformer_Backbone()
            in_channels = 1024
            low_level_channels = 256

        elif backbone == "mobilevit":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [640,H/8,W/8]
            #   浅层特征    [64,H/4,W/4]
            # ----------------------------------#
            self.backbone = mobile_vit_small_backbone(model_type="small", pretrained=pretrained)
            in_channels = 640
            low_level_channels = 64

        elif backbone == "mobilenetv3":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [640,H/8,W/8]
            #   浅层特征    [64,H/4,W/4]
            # ----------------------------------#
            self.backbone = mobilenet_v3_large_backbone(pretrained=pretrained, model_type="large")
            in_channels = 160
            low_level_channels = 40
        else:
            raise ValueError(
                "Unsupported backbone - `{}`, Use mobilenet, xception.".format(backbone)
            )

        # ========== Swin分支 ==========
        if use_swin:
            print(f"🔧 创建Swin分支，配置: {self.swin_config}")

            # 检查权重文件是否存在
            if not os.path.exists(swin_pretrained_path):
                print(f"⚠️ 警告: Swin权重文件不存在: {swin_pretrained_path}")
                print("将尝试在线下载或使用随机初始化")

            self.swin_branch = Swin_Branch_Configurable(
                num_classes=num_classes,
                pretrained=True,  # 总是尝试使用预训练
                pretrained_path=swin_pretrained_path,  # 传递本地权重路径
                selected_stages=self.swin_config['selected_stages'],
                fusion_channels=self.swin_config['fusion_channels'],
                use_attention=self.swin_config['use_attention'],
                fusion_type=self.swin_config['fusion_type']
            )

            # 根据骨干动态设置CAFM输入通道
            self.cafm = CAFM(in_channels + self.swin_config['fusion_channels'])

        # ===================== 原有模块保持不变 =====================
        self.CA = CoordAtt(in_channels, in_channels)

        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256,
                         rate=16 // downsample_factor)  # dim_in=2048 dim_out=256 rate=2
        #
        # self.denseaspp = _DenseASPPBlock(in_channels, 512, 256, norm_layer=nn.BatchNorm2d, norm_kwargs=None)

        # ----------------------------------#
        #   浅层特征边的卷积处理模块 将通道维度调整为48
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=low_level_channels,
                out_channels=256,  # deeplabv3plus 48
                kernel_size=1,
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
        )

        # Concat拼接浅层特征和ASPP处理后的特征
        self.cat_conv = nn.Sequential(
            CoordAtt(512, 512),  # 添加坐标注意力
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        # 更改channels至num_classes
        self.cls_conv = nn.Conv2d(
            in_channels=256, out_channels=num_classes, kernel_size=1, stride=1
        )

    def forward(self, x, need_fp=False):
        assert x.shape[-2:] == (512, 512), f"输入尺寸错误: {x.shape}"
        H, W = x.size(2), x.size(3)  # x(bs,3,H,W)
        raw_x = x  # 保存原始输入图像

        # 特征提取
        if self.backbone_name in [
            "xception",
            "mobilenet",
            "repvgg_new",
            "hrnet",
            "swin_transformer",
            "mobilevit",
            "mobilenetv3",
            "hrnet_new",
        ]:
            _, _, low_level_features, x = self.backbone(x)
        elif self.backbone_name in ["resnet50", "resnext50"]:
            features = self.backbone(x)
            low_level_features = features["low_features"]  # (B, 256, H/4, W/4)
            x = features["main"]  # (B, 2048, H/8, W/8)

        # Swin分支处理
        if hasattr(self, 'swin_branch'):
            swin_feat = self.swin_branch(raw_x)  # [B, fusion_channels, H/8, W/8]

            # 尺寸对齐：确保Swin特征与CNN主干特征分辨率一致
            if swin_feat.shape[2:] != x.shape[2:]:
                swin_feat = F.interpolate(
                    swin_feat,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )

            x = self.cafm(x, swin_feat)  # 特征融合

        x = self.aspp(x)  # x(bs, 256, H/8, W/8)
        low_level_features = self.shortcut_conv(
            low_level_features
        )  # low_level_features(bs, 256, H/4, W/4)

        # 将加强特征边上采样并与浅层特征堆叠
        x = F.interpolate(
            input=x,
            size=(low_level_features.size(2), low_level_features.size(3)),
            mode="bilinear",
            align_corners=True,
        )  # x(bs, 256, H/8, W/8) -> x(bs, 256, H/4, W/4)

        x = self.cat_conv(
            torch.cat((x, low_level_features), dim=1)  # (bs,512,H/4,W/4)
        )  # x(bs, 256, H/4, W/4)

        if need_fp:
            x = self.cls_conv(torch.cat((x, nn.Dropout2d(0.5)(x))))
            outs = F.interpolate(input=x, size=(H, W), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            return out, out_fp

        out = self.cls_conv(x)  # x(bs, num_classes, H/4, W/4)
        out = F.interpolate(input=out, size=(H, W), mode="bilinear", align_corners=True)  # x(bs, num_classes, H, W)
        return out

    def get_swin_config(self):
        """获取Swin分支配置"""
        if hasattr(self, 'swin_branch'):
            return self.swin_branch.config
        return None

    def switch_to_deploy(self):
        if self.backbone_name in ["repvgg_new"]:
            self.backbone = repvgg_model_convert(model=self.backbone)
            print(
                f"\033[1;33;44m 🔬🔬🔬🔬 Switch {self.backbone_name} to deploy model \033[0m"
            )
        else:
            print(f"\033[1;31;41m 🔬🔬🔬🔬 Can not Switch to deploy model \033[0m")