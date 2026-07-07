import torch.nn as nn
from torchvision.models import resnet50, resnet101, resnext50_32x4d, resnext101_32x8d

class ResNetBackbone(nn.Module):
    def __init__(self, arch="resnet50", pretrained=True):
        super().__init__()
        if arch == "resnet50":
            backbone = resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        elif arch == "resnet101":
            backbone = resnet101(weights="IMAGENET1K_V1" if pretrained else None)
        elif arch == "resnext50":
            backbone = resnext50_32x4d(weights="IMAGENET1K_V1" if pretrained else None)
        elif arch == "resnext101":
            backbone = resnext101_32x8d(weights="IMAGENET1K_V1" if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone {arch}")

        # 按 stage 拆分
        self.conv1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # C1
        self.layer2 = backbone.layer2  # C2
        self.layer3 = backbone.layer3  # C3
        self.layer4 = backbone.layer4  # C4

    def base_forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]   # 返回统一格式
