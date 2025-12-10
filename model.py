# import torch
# import torch.nn as nn
# from torch.autograd import Function
#
#
# # ================= 核心组件：梯度反转层 (GRL) =================
# # 这是迁移学习的灵魂。在前向传播时，它什么都不做（直接传递）；
# # 在反向传播时，它把梯度乘以负数 (-alpha)。
# # 原理：让特征提取器“欺骗”域判别器，提取出源域和目标域“共有”的特征。
# class GradientReverseLayer(Function):
#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.alpha = alpha
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         output = grad_output.neg() * ctx.alpha
#         return output, None
#
#
# # ================= 主模型架构 =================
# class DANN_Model(nn.Module):
#     def __init__(self, num_classes=4):
#         super(DANN_Model, self).__init__()
#
#         # 1. 特征提取器 (Feature Extractor) - 基于 1D-CNN
#         # 输入: (Batch, 1, 1024) -> 输出: (Batch, 64, 128) -> Flatten
#         # self.feature = nn.Sequential(
#         #     nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=1),
#         #     nn.BatchNorm1d(16),
#         #     nn.ReLU(),
#         #     nn.MaxPool1d(kernel_size=2),
#         #
#         #     nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=1),
#         #     nn.BatchNorm1d(32),
#         #     nn.ReLU(),
#         #     nn.MaxPool1d(kernel_size=2),
#         #
#         #     nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
#         #     nn.BatchNorm1d(64),
#         #     nn.ReLU(),
#         #     nn.MaxPool1d(kernel_size=2)
#         # )
#
#         # self.feature = nn.Sequential(
#         #     # 第一层卷积核稍微改小一点，适应 512 的长度
#         #     nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=1),
#         #     nn.BatchNorm1d(16),
#         #     nn.ReLU(),
#         #     nn.MaxPool1d(kernel_size=2),
#         #
#         #     nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=1),
#         #     nn.BatchNorm1d(32),
#         #     nn.ReLU(),
#         #     nn.MaxPool1d(kernel_size=2),
#         #
#         #     nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
#         #     nn.BatchNorm1d(64),
#         #     nn.ReLU(),
#         #     nn.MaxPool1d(kernel_size=2)
#         # )
#
#         # 特征提取器：把所有的 BatchNorm1d 换成 InstanceNorm1d
#         self.feature = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=1),
#             nn.InstanceNorm1d(16, affine=True),  # 【修改点】BN -> IN
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#
#             nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=1),
#             nn.InstanceNorm1d(32, affine=True),  # 【修改点】BN -> IN
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#
#             nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm1d(64, affine=True),  # 【修改点】BN -> IN
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2)
#         )
#
#         # 计算 Flatten 后的大小：经过多次下采样，需要根据输入维度自动计算或预估
#         # 这里假设输入长度 1024，经过层层下采样，最后大概剩 64 * 31 左右
#         # 为了通用性，我们在 forward 里加一个 Global Average Pooling 把它变成固定维度
#
#         # 2. 故障分类器 (Label Classifier) - 也就是常规的任务
#         self.class_classifier = nn.Sequential(
#             nn.Linear(64, 100),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(100, num_classes)  # 输出 4 类 (Normal, IR, OR, B)
#         )
#
#         # 3. 域判别器 (Domain Classifier) - 判断数据是源域还是目标域
#         self.domain_classifier = nn.Sequential(
#             nn.Linear(64, 100),
#             nn.ReLU(),
#             nn.Linear(100, 2)  # 输出 2 类 (Source=0, Target=1)
#         )
#
#     def forward(self, x, alpha=1.0):
#         # x shape: [batch, 1024] -> [batch, 1, 1024]
#         x = x.unsqueeze(1)
#
#         # 提取特征
#         features = self.feature(x)
#         # 使用自适应池化，强制变成 (Batch, 64, 1) 然后展平 -> (Batch, 64)
#         features = nn.functional.adaptive_avg_pool1d(features, 1)
#         features = features.view(features.size(0), -1)
#
#         # --- 分支 1：故障分类 ---
#         class_output = self.class_classifier(features)
#
#         # --- 分支 2：域判别 (关键点) ---
#         # 经过梯度反转层
#         reverse_features = GradientReverseLayer.apply(features, alpha)
#         domain_output = self.domain_classifier(reverse_features)
#
#         return class_output, domain_output, features

# -*- coding: utf-8 -*-
"""
model_new.py

DANN 模型定义 - 极致版（带残差连接）

模型结构：
- Stem: Conv1d(1→32) + InstanceNorm + ReLU + MaxPool
- Layer1: ResidualBlock1D(32→64)
- Layer2: ResidualBlock1D(64→128)
- Layer3: ResidualBlock1D(128→256)
- GlobalAvgPool
- ClassClassifier: 256→128→64→num_classes
- DomainClassifier: 256→128→2

特点：
1. 使用 InstanceNorm 替代 BatchNorm，更适合小 batch 和域适应
2. 残差连接提升梯度流动
3. Dropout 防止过拟合
4. 梯度反转层实现对抗训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# =========================================
# 梯度反转层 (Gradient Reversal Layer)
# =========================================
class GradientReverseLayer(Function):
    """
    梯度反转层：前向传播时不变，反向传播时梯度取反并乘以 alpha
    用于域对抗训练
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GRL(nn.Module):
    """梯度反转层的模块封装"""

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReverseLayer.apply(x, self.alpha)


# =========================================
# 残差块 (1D Residual Block)
# =========================================
class ResidualBlock1D(nn.Module):
    """
    1D 残差块
    结构：Conv1d → InstanceNorm → ReLU → Conv1d → InstanceNorm → (+shortcut) → ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.InstanceNorm1d(out_channels, affine=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.InstanceNorm1d(out_channels, affine=True)

        # 短接连接（如果维度不匹配）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.InstanceNorm1d(out_channels, affine=True)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# =========================================
# DANN 模型 - 极致版
# =========================================
class DANN_Model_Ultimate(nn.Module):
    """
    Domain-Adversarial Neural Network (极致版)

    特点：
    - 使用残差连接的特征提取器
    - InstanceNorm 替代 BatchNorm
    - 更深的分类器网络
    - 梯度反转实现域对抗

    Args:
        num_classes: 分类类别数，默认 4

    Returns:
        class_output: 分类输出 [batch, num_classes]
        domain_output: 域判别输出 [batch, 2]
        features: 特征向量 [batch, 256]
    """

    def __init__(self, num_classes=4):
        super(DANN_Model_Ultimate, self).__init__()

        self.num_classes = num_classes

        # ====== 特征提取器 ======
        # Stem 层
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(32, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 残差层
        self.layer1 = ResidualBlock1D(32, 64, stride=2)
        self.layer2 = ResidualBlock1D(64, 128, stride=2)
        self.layer3 = ResidualBlock1D(128, 256, stride=1)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ====== 分类器 ======
        self.class_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        # ====== 域判别器 ======
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, alpha=1.0):
        """
        前向传播

        Args:
            x: 输入信号 [batch, signal_length]
            alpha: 梯度反转系数，用于控制域对抗强度

        Returns:
            class_output: 分类输出
            domain_output: 域判别输出
            features: 特征向量
        """
        # 添加通道维度 [batch, 1, signal_length]
        x = x.unsqueeze(1)

        # 特征提取
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 全局池化
        features = self.global_pool(x)
        features = features.view(features.size(0), -1)  # [batch, 256]

        # 分类输出
        class_output = self.class_classifier(features)

        # 域判别（带梯度反转）
        reverse_features = GradientReverseLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)

        return class_output, domain_output, features

    def extract_features(self, x):
        """仅提取特征，不进行分类"""
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.global_pool(x)
        features = features.view(features.size(0), -1)
        return features

    def predict(self, x):
        """仅进行分类预测"""
        class_output, _, _ = self.forward(x, alpha=0.0)
        return class_output


# =========================================
# CNN 基线模型（不含域判别器）
# =========================================
class SimpleCNN(nn.Module):
    """
    简单 CNN 分类器（用于基线对比）
    与 DANN 使用相同的特征提取器，但没有域判别器
    """

    def __init__(self, input_length=512, num_classes=4):
        super(SimpleCNN, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(32, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.layer1 = ResidualBlock1D(32, 64, stride=2)
        self.layer2 = ResidualBlock1D(64, 128, stride=2)
        self.layer3 = ResidualBlock1D(128, 256, stride=1)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


# =========================================
# 测试代码
# =========================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 DANN_Model_Ultimate")
    print("=" * 60)

    # 创建模型
    model = DANN_Model_Ultimate(num_classes=4)
    print(f"\n模型结构：")
    print(model)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试前向传播
    batch_size = 8
    signal_length = 512
    x = torch.randn(batch_size, signal_length)

    class_out, domain_out, features = model(x, alpha=1.0)

    print(f"\n输入形状: {x.shape}")
    print(f"分类输出形状: {class_out.shape}")
    print(f"域判别输出形状: {domain_out.shape}")
    print(f"特征形状: {features.shape}")

    print("\n✅ 模型测试通过！")