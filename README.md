# SwinDeepLab & AL-RCNet: High-Resolution Remote Sensing Semantic Segmentation

本项目为高分辨率遥感影像语义分割提供了高效的解决方案，涵盖了从数据增强、混合模型设计到主动学习与半监督学习训练范式的完整框架。项目旨在解决遥感影像中精细边界丢失、尺度变化剧烈以及跨区域泛化能力差等核心痛点[cite: 1]。

## 核心架构与方法

本项目主要包含两个核心研究成果：

1. **SwinDeepLab 模型**：一种融合 CNN 与 Swin Transformer 的并行混合语义分割网络[cite: 1]。
    * **CNN 分支 (DeepLabv3+)**：负责提取局部精细边缘与纹理特征[cite: 1]。
    * **Swin Transformer 分支**：通过层次化自注意力机制捕获全局长程依赖[cite: 1]。
    * **CAFM 模块 (Coupled Attention Fusion Module)**：实现局部与全局特征的自适应动态融合[cite: 1]。
    * **多分辨率数据增强**：通过协同利用原生与下采样样本，显著提升模型对多尺度目标的适应能力[cite: 1]。

2. **AL-RCNet 训练框架**：融合主动学习与半监督学习的闭环训练范式[cite: 1]。
    * **UDJS 主动学习策略**：结合预测不确定性与特征空间多样性，精准筛选最具价值的难例样本[cite: 1]。
    * **RC 半监督学习 (Region-aware Contrastive Learning)**：构建全局类别原型记忆库，通过区域级对比约束，挖掘未标注数据的潜在语义价值[cite: 1]。


## 实验结果总结

[cite_start]本研究通过大量的消融实验与对比实验，验证了所提方法的先进性 [cite: 687, 745]。

### 1. 模型性能验证 (洱海数据集)
[cite_start]SwinDeepLab 在洱海流域无人机遥感数据集上表现卓越，其在复杂地表场景下的分割精度显著优于各类主流基线模型 [cite: 754]：
* [cite_start]**MIoU 表现**：SwinDeepLab 达到 **83.56%**，较传统 DeepLabv3+ (80.90%) 提升了 2.66 个百分点 [cite: 749, 754]。
* [cite_start]**类别表现**：在道路 (73.87%) 和植被 (81.31%) 等细粒度结构类别上实现了断层式领先 [cite: 754]。


### 2. 跨区域泛化能力验证 (LoveDA 数据集)
[cite_start]在极具挑战性的 LoveDA 数据集（跨域场景）上，AL-RCNet 展现了极强的泛化鲁棒性 [cite: 778]：
* [cite_start]**性能对比**：AL-RCNet 以 **55.6%** 的 MIoU 斩获全局最优，相较于基线 DeepLabv3+ (47.6%) 提升了 **8 个百分点** [cite: 776, 778]。
* [cite_start]**抗干扰能力**：在裸地 (28.2%)、建筑 (62.1%) 等难点类别上，精准滤除了严重的域偏移噪声，克服了传统模型边界模糊的缺陷 [cite: 778]。

### 3. 主动学习与半监督策略协同
* [cite_start]**标注效率**：在仅使用 **20%** 标注预算的情况下，通过主动学习与半监督策略协同，模型在跨域测试集上的 MIoU 达到 **80.45%**，较全监督基线大幅提升 **11.62 个百分点** [cite: 26, 1137]。

## 代码结构说明

* `/model`: 包含 SwinDeepLab 与 AL-RCNet 的核心网络架构实现[cite: 1]。
* `/util`: 包含数据预处理、多分辨率增强策略及评估指标等工具函数[cite: 1]。
* `Mix_EH.py / Mixtest.py`: 实现多分辨率混合数据增强策略的关键脚本[cite: 1]。
* `swin_xiaorong.py`: Swin 分支多层特征融合的消融实验相关代码[cite: 1]。
* `trans_deep_active.py`: AL-RCNet 框架下主动学习的训练实现[cite: 1]。
* `trans_deep_unimatch_semi.py`: AL-RCNet 框架下半监督学习的训练实现[cite: 1]。




