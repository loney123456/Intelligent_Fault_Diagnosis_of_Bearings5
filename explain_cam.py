import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model import DANN_Model
import os

# ================= 配置 =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'dann_model_final.pth'  # 你的模型路径
FONT_PATH = 'SimHei'  # 字体，防止中文乱码


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子 (Hook) 以获取中间层的输出和梯度
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # 1. 前向传播
        self.model.eval()
        # 注意：这里需要根据你的 model.forward 修改
        # 你的 forward 返回 class_output, domain_output, features
        # 我们只需要 class_output
        logits, _, _ = self.model(x, alpha=0)

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)

        # 2. 反向传播
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # 3. 获取梯度和特征图
        gradients = self.gradients.data.cpu().numpy()[0]  # (C, L)
        activations = self.activations.data.cpu().numpy()[0]  # (C, L)

        # 4. 全局平均池化 (GAP) 获取权重
        weights = np.mean(gradients, axis=1)  # (C,)

        # 5. 加权求和
        cam = np.zeros(activations.shape[1], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # 6. ReLU (只保留正向影响)
        cam = np.maximum(cam, 0)

        # 7. 归一化并缩放到输入尺寸
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)  # 防止除零

        return cam


def find_last_conv_layer(model):
    """
    自动查找模型中最后一个 Conv1d 层
    """
    layers = []
    # 遍历 model.feature (假设它是 nn.Sequential)
    for name, module in model.feature.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            layers.append(module)

    if not layers:
        raise ValueError("未在模型中找到 Conv1d 层")

    return layers[-1]  # 返回最后一个卷积层


def visualize_cam(model_path, data_sample, true_label_name, pred_label_name):
    # 加载模型
    model = DANN_Model(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # 找到目标层 (最后一个卷积层)
    target_layer = find_last_conv_layer(model)
    grad_cam = GradCAM(model, target_layer)

    # 准备数据
    x = torch.from_numpy(data_sample).float().unsqueeze(0).to(DEVICE)  # (1, Length) -> (1, 1, Length)
    # 你的模型 forward 会自动 unsqueeze，所以这里如果传入 (1, 512)
    # model 内部变成 (1, 1, 512)，没问题。
    # 但根据你的 model.py: x = x.unsqueeze(1)，所以传入 (Batch, Length) 即可。

    # 获取 CAM 热力图
    # x shape: (1, 512)
    cam_map = grad_cam(x)

    # 因为卷积层输出长度变小了 (MaxPool)，我们需要把 heatmap 插值放大回原始长度
    input_len = data_sample.shape[0]
    x_axis = np.linspace(0, input_len, len(cam_map))
    x_axis_new = np.linspace(0, input_len, input_len)

    # 线性插值放大
    cam_resized = np.interp(x_axis_new, x_axis, cam_map)

    # ================== 绘图 ==================
    plt.figure(figsize=(12, 6))
    plt.rcParams['font.sans-serif'] = [FONT_PATH]
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制原始频谱 (蓝色)
    plt.plot(data_sample, color='blue', alpha=0.6, label='原始频谱信号 (Input Spectrum)')

    # 绘制 CAM 热力图 (红色覆盖，表示关注度)
    # 使用 fill_between 让关注区域更明显
    plt.fill_between(range(len(data_sample)), 0, cam_resized * np.max(data_sample),
                     color='red', alpha=0.3, label='模型关注区域 (Attention Map)')

    plt.plot(cam_resized * np.max(data_sample), color='red', linewidth=2)

    plt.title(f'可解释性分析: 真实={true_label_name} | 预测={pred_label_name}', fontsize=16)
    plt.xlabel('频率索引 (Frequency Index)', fontsize=14)
    plt.ylabel('幅值 (Amplitude)', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    save_name = f'cam_result_{true_label_name}.png'
    plt.savefig(save_name, dpi=300)
    print(f"结果已保存: {save_name}")
    plt.show()


def main():
    # 1. 加载数据
    print("正在加载数据...")
    source_x = np.load('source_x.npy')
    source_y = np.load('source_y.npy')

    # 简单的归一化 (必须和训练时一致)
    mean = source_x.mean()
    std = source_x.std()

    # 2. 挑选一个特定的样本进行分析
    # 比如我们想看一个 "外圈故障 (Label=2)" 的样本
    target_class = 2
    indices = np.where(source_y == target_class)[0]

    if len(indices) > 0:
        idx = indices[10]  # 随便挑第10个
        sample_raw = source_x[idx]
        sample_norm = (sample_raw - mean) / (std + 1e-5)  # 归一化后送入模型

        label_map = {0: '正常', 1: '内圈故障', 2: '外圈故障', 3: '滚动体故障'}
        print(f"正在分析样本 ID: {idx}, 类别: {label_map[target_class]}")

        visualize_cam(MODEL_PATH, sample_norm, label_map[target_class], label_map[target_class])
    else:
        print("未找到该类别的样本")


if __name__ == '__main__':
    main()