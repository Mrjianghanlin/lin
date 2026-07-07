import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义特征提取钩子 (Hook)
# ==========================================
# 用于存储截取到的中间层特征
extracted_features = {}


def get_activation(name):
    """
    钩子函数：当网络执行到指定层时，自动将输出特征保存到字典中
    """

    def hook(model, input, output):
        extracted_features[name] = output.detach()

    return hook


# ==========================================
# 2. 特征图转热力图的核心函数
# ==========================================
def feature_to_heatmap(feature_tensor, target_size=(512, 512)):
    """
    将高维特征张量转换为伪彩色热力图
    :param feature_tensor: 模型输出的特征张量，形状为 [1, C, H, W]
    :param target_size: 输出热力图的尺寸，通常与原图对齐
    """
    # 沿通道维度(C)求均值，将 [1, C, H, W] 降维成 [H, W] 的单通道图
    feat_map = torch.mean(feature_tensor, dim=1).squeeze(0).cpu().numpy()

    # 过滤掉负值（类似 ReLU 的激活效果，突出高响应区域）
    feat_map = np.maximum(feat_map, 0)

    # 归一化到 0 ~ 1 之间
    feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)

    # 缩放到 0 ~ 255，并转为 uint8 类型
    feat_map = np.uint8(255 * feat_map)

    # 将小尺寸特征图放大到与原图相同的尺寸
    feat_map_resized = cv2.resize(feat_map, target_size, interpolation=cv2.INTER_LINEAR)

    # 应用 JET 伪彩色映射（红-黄-绿-蓝，红色代表高响应区域）
    heatmap = cv2.applyColorMap(feat_map_resized, cv2.COLORMAP_JET)

    # OpenCV 默认是 BGR，转换为 RGB 以便 matplotlib 正常显示
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap


# ==========================================
# 3. 主程序
# ==========================================
def main():
    # ---- 步骤 A：加载模型并注册钩子 ----
    # 此处以 ResNet-101 为例。实际应用时请替换为你的 SwinDeepLab 模型实例
    model = models.resnet101(pretrained=True)
    model.eval()  # 设置为评估模式

    # 将钩子挂载到 ResNet 的 Stage 1 (layer1) 和 Stage 4 (layer4) 上
    # 如果是你自己的模型，请打印 model 看看具体层的名称并替换
    model.layer1.register_forward_hook(get_activation('stage1'))
    model.layer4.register_forward_hook(get_activation('stage4'))

    # ---- 步骤 B：加载并预处理输入图像 ----
    image_path = "1.tif"  # 替换为你的遥感图像路径
    try:
        original_img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"找不到图像文件: {image_path}，请准备一张测试图。")
        return

    original_img = original_img.resize((512, 512))  # 统一缩放到 512x512

    # 图像预处理流水线：转为 Tensor 并进行标准化
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(original_img).unsqueeze(0)  # 增加 Batch 维度: [1, 3, 512, 512]

    # 把数据和模型放到 GPU 上（如果有的话）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # ---- 步骤 C：前向传播，触发钩子截取特征 ----
    with torch.no_grad():
        _ = model(input_tensor)  # 我们不需要最终输出，只要中间过程触发钩子即可

    # ---- 步骤 D：获取特征并生成热力图 ----
    feat_stage1 = extracted_features['stage1']
    feat_stage4 = extracted_features['stage4']

    heatmap_s1 = feature_to_heatmap(feat_stage1, target_size=(512, 512))
    heatmap_s4 = feature_to_heatmap(feat_stage4, target_size=(512, 512))

    # ---- 步骤 E：使用 Matplotlib 拼接并保存结果，方便放入毕业论文 ----
    plt.figure(figsize=(15, 5))

    # 1. 显示原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original Remote Sensing Image", fontsize=14)
    plt.axis('off')

    # 2. 显示 Stage 1 浅层特征（侧重边缘、纹理）
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_s1)
    plt.title("Stage 1 Feature Map (Low-level)", fontsize=14)
    plt.axis('off')

    # 3. 显示 Stage 4 深层特征（侧重全局语义）
    plt.subplot(1, 3, 3)
    plt.imshow(heatmap_s4)
    plt.title("Stage 4 Feature Map (High-level)", fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    # 将排版好的对比图保存为高分辨率的高清图片，可直接贴入 Word
    plt.savefig("feature_visualization_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("特征可视化对比图已保存为 feature_visualization_comparison.png")


if __name__ == '__main__':
    main()