import torch
import torch.nn as nn
import torch.nn.functional as F

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
import os
from transformers import ViTForImageClassification
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用国内镜像
from transformers import ViTModel
from timm.models.swin_transformer import SwinTransformer
from model.models.nets1.repvgg_new import repvgg_model_convert, repvgg_backbone_new
import timm


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
        # fusion_conv保持不变
        #         # 通道数修正：256+512+1024=1792
                # 通道数修正：256+512+1024=1792
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(1792, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        # print(f"Input batch size: {x.shape[0]}, device: {x.device}")
        feats_4d = []
        swin_feats = self.swin(x)  # 这里返回的是 list，元素shape: [B, C, H, W]
        for f in swin_feats:
            # f.shape = [B, H, W, C]
            f_4d = f.permute(0, 3, 1, 2).contiguous()
            feats_4d.append(f_4d)
        swin_feats=feats_4d
        # print(f"Swin output len: {len(swin_feats)}")
        for i, f in enumerate(swin_feats):
            pass
            # print(f"out[{i}] shape:", f.shape)

        # 选取第1,2,3个stage作为示范，尺寸可能不同，直接统一插值到相同尺寸
        target_h, target_w = x.shape[2] // 8, x.shape[3] // 8

        c2 = F.interpolate(swin_feats[1], size=(target_h, target_w), mode='bilinear', align_corners=True)
        c3 = F.interpolate(swin_feats[2], size=(target_h, target_w), mode='bilinear', align_corners=True)
        c4 = F.interpolate(swin_feats[3], size=(target_h, target_w), mode='bilinear', align_corners=True)


         # 添加关键打印点
        # print(f"🔄 Swin_Branch输入: {x.shape} (B,C,H,W)")
        # print(f"  ├─Stage1特征: {swin_feats[0].shape} (忽略)")
        # print(f"  ├─Stage2特征: {swin_feats[1].shape} -> 上采样后: {c2.shape}")
        # print(f"  ├─Stage3特征: {swin_feats[2].shape} -> 上采样后: {c3.shape}")
        # print(f"  ├─Stage4特征: {swin_feats[3].shape} -> 上采样后: {c4.shape}")

        fused_feat = torch.cat([c2, c3, c4], dim=1)
        # print(f"Fused feature shape: {fused_feat.shape}")  # 这里建议加上这行打印
        # print(f"  └─融合后特征: {fused_feat.shape} | 通过conv(1792→256)")
        # print(f"  └─融合后特征111: {self.fusion_conv(fused_feat).shape} | 通过conv(1792→256)")

        return self.fusion_conv(fused_feat)


# class Swin_Branch(nn.Module):
#     """重构的Swin分支：动态提取多尺度特征"""
#     def __init__(self, num_classes, pretrained=True):
#         super().__init__()
#         self.swin = SwinTransformer(
#             img_size=512,
#             patch_size=4,
#             in_chans=3,
#             num_classes=0,  # 禁用分类头
#             embed_dim=128,
#             features_only=True,  # 强制返回特征图而非logits
#             depths=[2, 2, 18, 2],
#             num_heads=[4, 8, 16, 32],
#             window_size=8,
#             mlp_ratio=4.,
#             qkv_bias=True,
#             drop_rate=0.0,
#             attn_drop_rate=0.0,
#             drop_path_rate=0.3,
#             out_indices=(1, 2, 3),  # 输出stage2-4特征
#             pretrained=pretrained
#         )
        # # 通道数修正：256+512+1024=1792
        # self.fusion_conv = nn.Sequential(
        #     nn.Conv2d(3072, 256, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU()
#         )
    
        
#     def forward(self, x):

     
#         print(f"Input batch size: {x.shape[0]}")     # 应该是 6
#         print(f"x.device: {x.device}")               # 比如 cuda:0

#         out = self.swin.forward_features(x)
#         print(f"Swin output batch size: {out[0].shape[0]}")




#         B, C, H, W = x.shape
#         target_h, target_w = H // 8, W // 8
        
#             # 1. 获取所有stage输出，shape均为 [B, L, C]
#         swin_feats = self.swin.forward_features(x)
#         print(f"Swin output batch size: {swin_feats[0].shape[0]}")
#         print(f"Swin output len: {len(swin_feats)}")
#         for i, f in enumerate(swin_feats):
#             print(f"out[{i}] shape:", f.shape)

#         if swin_feats is None or len(swin_feats) == 0:
#             raise RuntimeError("SwinTransformer返回空特征! 检查模型配置")

#         # 2. 将每个特征从[B, L, C]转换为[B, C, H, W]
#         feats_4d = []
#         for f in swin_feats:
#             B_f, L, C_f = f.shape
#             size = int(L ** 0.5)
#             f_4d = f.permute(0, 2, 1).contiguous().view(B_f, C_f, size, size)  # 变换维度
#             feats_4d.append(f_4d)

#         # 3. 选择第2、3、4层特征 (索引1,2,3)
#         c2, c3, c4 = feats_4d[1], feats_4d[2], feats_4d[3]

#         # 4. 验证维度
#         for i, f in enumerate([c2, c3, c4]):
#             assert f.dim() == 4, f"第{i}个特征维度不是4D: shape={f.shape}"
#         # 4. 安全上采样
#         c2 = self.safe_interpolate(c2, (target_h, target_w))
#         c3 = self.safe_interpolate(c3, (target_h, target_w))
#         c4 = self.safe_interpolate(c4, (target_h, target_w))
        
#         # 5. 特征融合
#         fused_feat = torch.cat([c2, c3, c4], dim=1)
#         return self.fusion_conv(fused_feat)
    
#     def safe_interpolate(self, feat, target_size):
#         """鲁棒的维度处理"""
#         if feat.dim() == 2:  # 处理1D输出
#             feat = feat.view(feat.size(0), feat.size(1), 1, 1)
#         elif feat.dim() == 3:  # 处理3D特征 (B, C, L)
#             L = feat.size(2)
#             feat = feat.view(feat.size(0), feat.size(1), int(L**0.5), int(L**0.5))
#         return F.interpolate(feat, size=target_size, 
#                            mode='bilinear', align_corners=True)

class CAFM(nn.Module):
    """优化后的耦合注意力融合模块"""
    def __init__(self, channels):
        super().__init__()
        # 动态调整融合权重
        self.gamma = nn.Parameter(torch.zeros(1))
        # 通道压缩
        self.compression = nn.Sequential(
            nn.Conv2d(channels, channels//4, 1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU()
        )
        # 注意力生成
        self.attention = nn.Sequential(
            nn.Conv2d(channels//4, 1, 1),
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
        use_swin=True  # 启用Swin分支
    ):
        super(DeepLab, self).__init__()
        self.use_swin = use_swin
        self.backbone_name = backbone
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
            self.backbone = repvgg_backbone_new(model_type="RepVGG-B2g4-new",pretrained=pretrained)
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
            in_channels =  720   # 主干部分的特征
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
            self.backbone = HRNet_Backbone_New(backbone="hrnet_w32",pretrained=pretrained)
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
            self.backbone = mobile_vit_small_backbone(model_type="small",pretrained=pretrained)
            in_channels = 640
            low_level_channels = 64

        elif backbone == "mobilenetv3":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [640,H/8,W/8]
            #   浅层特征    [64,H/4,W/4]
            # ----------------------------------#
            self.backbone = mobilenet_v3_large_backbone(pretrained=pretrained,model_type="large")
            in_channels = 160
            low_level_channels = 40
        else:
            raise ValueError(
                "Unsupported backbone - `{}`, Use mobilenet, xception.".format(backbone)
            )


           # ========== 关键修改：Swin分支替换ViT ==========
        # 在DeepLab初始化中
        if use_swin:
            self.swin_branch = Swin_Branch(num_classes, pretrained)
            self.fusion = nn.Sequential(
                nn.Conv2d(2048, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
       
            
            # 根据骨干动态设置CAFM输入通道
            self.cafm = CAFM(in_channels+256)  # 替换原固定值256
        
        


        # ===================== 原有模块保持不变 =====================
        self.CA = CoordAtt(in_channels, in_channels)



        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.aspp = ASPP( dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)  # dim_in=2048 dim_out=256 rate=2
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
            CoordAtt(512, 512),  # 添加坐标注意力[1](@ref)
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



    def forward(self, x,need_fp=False):

        assert x.shape[-2:] == (512, 512), f"输入尺寸错误: {x.shape}"
        H, W = x.size(2), x.size(3)  # x(bs,3,H,W)
        raw_x = x  # 保存原始输入图像
        # print("Input x shape:", x.shape)
        # print("raw_x shape:", raw_x.shape)

        # -----------------------------------------#
        #   特征提取 获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理 (B, 256, H/4, W/4)  处理4倍下采样feature maps
        #   x : 主干部分-利用ASPP结构进行加强特征提取 (B, 2048, H/8, W/8)  处理8倍下采样feature maps
        # -----------------------------------------#

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
            _,_,low_level_features, x = self.backbone(x)
        elif self.backbone_name in ["resnet50", "resnext50"]:
            features = self.backbone(x)
            low_level_features = features["low_features"]  # (B, 256, H/4, W/4)
            x = features["main"]  # (B, 2048, H/8, W/8)
        # print(f"📦 主干网络输出:")
        # print(f"  ├─浅层特征: {low_level_features.shape}")
        # print(f"  └─深层特征: {x.shape} (ASPP输入)")
        
          # ========== Swin分支处理 ==========
        if hasattr(self, 'swin_branch'):
            swin_feat = self.swin_branch(raw_x)  # [B,256,H/8,W/8]
            # print("swin_feat shape before interp111111:", swin_feat.shape)

             # 尺寸对齐：确保Swin特征与CNN主干特征分辨率一致
            if swin_feat.shape[2:] != x.shape[2:]:
                swin_feat = F.interpolate(
                    swin_feat, 
                    size=x.shape[2:], 
                    mode='bilinear', 
                    align_corners=True
                )


            # print("cnn_feat shape:", x.shape)
            # print("swin_feat shape before interp:", swin_feat.shape)
            swin_feat = F.interpolate(swin_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
            # print("swin_feat shape after interp:", swin_feat.shape)

            x = self.cafm(x, swin_feat)  # 特征融合
        
        

        x = self.aspp(x)  # x(bs, 256, H/8, W/8)
        # x = self.denseaspp(x)
        low_level_features = self.shortcut_conv(
            low_level_features
        )  # low_level_features(bs, 256, H/4, W/4)

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(
            input=x,
            size=(low_level_features.size(2), low_level_features.size(3)),
            mode="bilinear",
            align_corners=True,
        )  # x(bs, 256, H/8, W/8) -> x(bs, 256, H/4, W/4)
        x = self.cat_conv(
            torch.cat((x, low_level_features), dim=1)  # (bs,304,H/4,W/4)
        )  # x(bs, 256, H/4, W/4)
        # x = self.cls_conv(x)  # x(bs, num_classes, H/4, W/4)
        # x = F.interpolate(
        #     input=x, size=(H, W), mode="bilinear", align_corners=True
        # )  # x(bs, num_classes, H, W)
        # return x
        if need_fp:
            x= self.cls_conv(torch.cat((x, nn.Dropout2d(0.5)(x))))
            outs = F.interpolate(input=x, size=(H, W), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            return out, out_fp

        out = self.cls_conv(x)  # x(bs, num_classes, H/4, W/4)
        out = F.interpolate(input=out, size=(H, W), mode="bilinear", align_corners=True)  # x(bs, num_classes, H, W)
        return out

    def switch_to_deploy(self):
        if self.backbone_name in ["repvgg_new"]:
            self.backbone = repvgg_model_convert(model=self.backbone)
            print(
                f"\033[1;33;44m 🔬🔬🔬🔬 Switch {self.backbone_name} to deploy model \033[0m"
            )
        else:
            print(f"\033[1;31;41m 🔬🔬🔬🔬 Can not Switch to deploy model \033[0m")
if __name__ == '__main__':
    model = DeepLab(num_classes=3,backbone="hrnet", downsample_factor=16, pretrained=True,use_swin=True  )
    # summary(model, (3, 512, 512), device="cpu")
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(model)