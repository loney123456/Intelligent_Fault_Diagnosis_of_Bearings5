import torch
import torch.nn as nn
from torch.autograd import Function


# ================= 核心组件：梯度反转层 (GRL) =================
# 这是迁移学习的灵魂。在前向传播时，它什么都不做（直接传递）；
# 在反向传播时，它把梯度乘以负数 (-alpha)。
# 原理：让特征提取器“欺骗”域判别器，提取出源域和目标域“共有”的特征。
class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# ================= 主模型架构 =================
class DANN_Model(nn.Module):
    def __init__(self, num_classes=4):
        super(DANN_Model, self).__init__()

        # 1. 特征提取器 (Feature Extractor) - 基于 1D-CNN
        # 输入: (Batch, 1, 1024) -> 输出: (Batch, 64, 128) -> Flatten
        # self.feature = nn.Sequential(
        #     nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=1),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        #
        #     nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        #
        #     nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2)
        # )

        # self.feature = nn.Sequential(
        #     # 第一层卷积核稍微改小一点，适应 512 的长度
        #     nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=1),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        #
        #     nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        #
        #     nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2)
        # )

        # 特征提取器：把所有的 BatchNorm1d 换成 InstanceNorm1d
        self.feature = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=1),
            nn.InstanceNorm1d(16, affine=True),  # 【修改点】BN -> IN
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm1d(32, affine=True),  # 【修改点】BN -> IN
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(64, affine=True),  # 【修改点】BN -> IN
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # 计算 Flatten 后的大小：经过多次下采样，需要根据输入维度自动计算或预估
        # 这里假设输入长度 1024，经过层层下采样，最后大概剩 64 * 31 左右
        # 为了通用性，我们在 forward 里加一个 Global Average Pooling 把它变成固定维度

        # 2. 故障分类器 (Label Classifier) - 也就是常规的任务
        self.class_classifier = nn.Sequential(
            nn.Linear(64, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, num_classes)  # 输出 4 类 (Normal, IR, OR, B)
        )

        # 3. 域判别器 (Domain Classifier) - 判断数据是源域还是目标域
        self.domain_classifier = nn.Sequential(
            nn.Linear(64, 100),
            nn.ReLU(),
            nn.Linear(100, 2)  # 输出 2 类 (Source=0, Target=1)
        )

    def forward(self, x, alpha=1.0):
        # x shape: [batch, 1024] -> [batch, 1, 1024]
        x = x.unsqueeze(1)

        # 提取特征
        features = self.feature(x)
        # 使用自适应池化，强制变成 (Batch, 64, 1) 然后展平 -> (Batch, 64)
        features = nn.functional.adaptive_avg_pool1d(features, 1)
        features = features.view(features.size(0), -1)

        # --- 分支 1：故障分类 ---
        class_output = self.class_classifier(features)

        # --- 分支 2：域判别 (关键点) ---
        # 经过梯度反转层
        reverse_features = GradientReverseLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)

        return class_output, domain_output, features