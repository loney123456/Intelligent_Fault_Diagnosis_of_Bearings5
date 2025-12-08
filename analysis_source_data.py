# -*- coding: utf-8 -*-
"""
analysis_source_data.py

用途：
1. 读取 source_x.npy / source_y.npy 源域样本
2. 对 N / IR / OR / B 各选若干条典型样本，画时域波形 + 频谱
3. 在频谱上叠加理论 BPFO / BPFI / BSF / FTF 及其倍频竖线
4. 计算每条样本的统计特征（均值、标准差、RMS、峭度、峰值因子），
   画箱线图比较四类故障的差异，并在命令行输出每类的统计汇总

使用方法：
    python analysis_source_data.py

注意：
- 需要已安装 numpy、matplotlib、scipy（没有的话：pip install numpy matplotlib scipy）
- 默认假设：source_x.npy 形状为 (num_samples, signal_len) 或 (num_samples, 1, signal_len)
- 类别标签映射 CLASS_MAP 可能与你项目不同，请按实际修改。
"""

import os
import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis


# =========================
# 1. 基本配置（按需要修改）
# =========================

# 采样频率（Hz）：如果你的源域主要用 48kHz，就写 48000；
# 如果用 12kHz，就改成 12000
FS = 48000.0

# 轴转速（RPM）：E 题 / CWRU 常见工况之一是 1797 rpm 左右
SHAFT_SPEED_RPM = 1797.0

# 轴承几何参数（可以改为你自己实际的参数）
# 这里给出的是 CWRU 6205-2RS JEM SKF 驱动端轴承的大致几何参数，
# 单位只要保持一致（英寸或毫米都可以），因为公式只用到 d/D 比值
N_BALLS = 9                  # 滚动体个数 Z
BALL_DIAMETER = 0.3126       # 球直径 d（英寸）
PITCH_DIAMETER = 1.537       # 节圆直径 D（英寸）
CONTACT_ANGLE_DEG = 0.0      # 接触角（大多数工况可近似为 0°）

# 类别标签映射：请确认与你生成 source_y.npy 时的编码一致
# 比如：0=正常，1=内圈故障，2=外圈故障，3=滚动体故障
CLASS_MAP: Dict[int, str] = {
    0: "Normal",
    1: "Inner race fault",
    2: "Outer race fault",
    3: "Ball fault",
}

# 每个类别挑多少条样本来画时域+频域图
EXAMPLES_PER_CLASS = 3

# 图像输出目录
OUTPUT_DIR = "analysis_plots"


# =========================
# 2. 计算轴承故障特征频率
# =========================

def compute_fault_frequencies(
    rpm: float = SHAFT_SPEED_RPM,
    n_balls: int = N_BALLS,
    ball_diameter: float = BALL_DIAMETER,
    pitch_diameter: float = PITCH_DIAMETER,
    contact_angle_deg: float = CONTACT_ANGLE_DEG,
) -> Dict[str, float]:
    """
    根据轴承几何参数计算 BPFO/BPFI/BSF/FTF 四种故障特征频率（Hz）。

    公式参考：
        - BPFI = f * (Z/2) * (1 + d/D * cos(α))
        - BPFO = f * (Z/2) * (1 - d/D * cos(α))
        - BSF  = f * (D/(2d)) * (1 - (d/D * cos(α))^2)
        - FTF  = f * (1/2) * (1 - d/D * cos(α))
    其中 f 为轴转速频率 (Hz)，Z 为滚动体个数，d 为滚动体直径，D 为节圆直径。
    """

    f = rpm / 60.0  # 轴转速频率（Hz）
    Z = float(n_balls)
    d = float(ball_diameter)
    D = float(pitch_diameter)
    alpha = math.radians(contact_angle_deg)

    ratio = (d / D) * math.cos(alpha)

    bpfi = f * 0.5 * Z * (1.0 + ratio)
    bpfo = f * 0.5 * Z * (1.0 - ratio)
    bsf = f * (D / (2.0 * d)) * (1.0 - ratio ** 2)
    ftf = f * 0.5 * (1.0 - ratio)

    return {
        "BPFI": bpfi,
        "BPFO": bpfo,
        "BSF": bsf,
        "FTF": ftf,
    }


# =========================
# 3. 数据读取与预处理
# =========================

def load_source_data(
    x_path: str = "source_x.npy",
    y_path: str = "source_y.npy",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 npy 文件加载源域数据。

    返回：
        signals: shape (N, L) 的二维数组，每行是一条时间序列
        labels:  shape (N,) 的整数标签
    """
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"找不到 {x_path} 或 {y_path}，请确认文件路径是否正确。"
        )

    X = np.load(x_path)
    y = np.load(y_path)

    # 若形状为 (N, 1, L) 或 (N, C, L)，只取第一个通道
    if X.ndim == 3:
        X = X[:, 0, :]
    elif X.ndim == 2:
        pass
    else:
        raise ValueError(
            f"不支持的 source_x 形状：{X.shape}，期望 (N, L) 或 (N, C, L)"
        )

    y = y.astype(int).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X 和 y 样本数不一致：X={X.shape[0]}，y={y.shape[0]}"
        )

    print(f"加载数据成功：X 形状 = {X.shape}，y 形状 = {y.shape}")
    return X, y


def select_example_indices(
    labels: np.ndarray,
    examples_per_class: int = EXAMPLES_PER_CLASS,
) -> Dict[int, List[int]]:
    """
    为每个类别选出若干条样本的索引，用于画时域+频域图。
    """
    selected: Dict[int, List[int]] = {}
    for cls, name in CLASS_MAP.items():
        idx = np.where(labels == cls)[0]
        if idx.size == 0:
            print(f"[警告] 标签 {cls} ({name}) 在数据中不存在。")
            continue
        # 只取前若干条
        selected[cls] = idx[:examples_per_class].tolist()
        print(f"类别 {cls} ({name}) 选取了 {len(selected[cls])} 条样本用于示例。")
    return selected


# =========================
# 4. 绘图函数
# =========================

def add_fault_frequency_lines(
    ax: plt.Axes,
    fault_freqs_hz: Dict[str, float],
    fs: float,
    max_harmonic: int = 5,
):
    """
    在频谱图上叠加 BPFO/BPFI/BSF/FTF 及其若干倍频的竖线。
    """
    nyquist = fs / 2.0
    for name, f0 in fault_freqs_hz.items():
        if f0 <= 0:
            continue
        k = 1
        while True:
            f = f0 * k
            if f >= nyquist:
                break
            ax.axvline(f, linestyle="--", linewidth=0.8, alpha=0.7, label=name if k == 1 else None)
            k += 1

    # 去重 legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)


def plot_time_and_spectrum_examples(
    signals: np.ndarray,
    labels: np.ndarray,
    example_indices: Dict[int, List[int]],
    fs: float = FS,
    output_dir: str = OUTPUT_DIR,
):
    """
    对每个类别的若干条样本画时域+频域图，并保存到 output_dir。
    """
    os.makedirs(output_dir, exist_ok=True)

    fault_freqs = compute_fault_frequencies()

    for cls, indices in example_indices.items():
        class_name = CLASS_MAP.get(cls, f"class_{cls}")

        for k, idx in enumerate(indices):
            x = signals[idx]
            n = x.shape[0]
            t = np.arange(n) / fs

            # 频谱（只取正频率部分）
            fft_vals = np.fft.rfft(x)
            fft_amp = np.abs(fft_vals) / n
            freqs = np.fft.rfftfreq(n, d=1.0 / fs)

            fig, axes = plt.subplots(2, 1, figsize=(10, 6))
            fig.suptitle(f"{class_name} | 样本 idx={idx}", fontproperties="SimHei")

            # 时域波形
            axes[0].plot(t, x)
            axes[0].set_xlabel("时间 (s)", fontproperties="SimHei")
            axes[0].set_ylabel("加速度幅值", fontproperties="SimHei")
            axes[0].grid(True, linestyle="--", alpha=0.4)

            # 频谱
            axes[1].plot(freqs, fft_amp)
            axes[1].set_xlim(0, fs / 2.0)
            axes[1].set_xlabel("频率 (Hz)", fontproperties="SimHei")
            axes[1].set_ylabel("幅值", fontproperties="SimHei")
            axes[1].grid(True, linestyle="--", alpha=0.4)

            add_fault_frequency_lines(axes[1], fault_freqs, fs)

            fname = os.path.join(output_dir, f"{cls}_{class_name}_idx{idx}.png")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(fname, dpi=200)
            plt.close(fig)
            print(f"已保存示例图：{fname}")


# =========================
# 5. 统计特征计算与箱线图
# =========================

def compute_signal_features(
    signals: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    对每条样本计算统计特征：
        - mean: 均值
        - std: 标准差
        - rms: 均方根
        - kurtosis: 峭度
        - peak_factor: 峰值因子 = max(|x|) / rms
    返回一个字典，每个 key 对应 shape (N,) 的数组。
    """
    N = signals.shape[0]
    features = {
        "mean": np.zeros(N),
        "std": np.zeros(N),
        "rms": np.zeros(N),
        "kurtosis": np.zeros(N),
        "peak_factor": np.zeros(N),
    }

    for i in range(N):
        x = signals[i]
        features["mean"][i] = np.mean(x)
        features["std"][i] = np.std(x, ddof=1)
        rms_val = np.sqrt(np.mean(x ** 2))
        features["rms"][i] = rms_val
        # Fisher=False 表示使用“普通”峭度（不是减 3 的那种）
        features["kurtosis"][i] = kurtosis(x, fisher=False, bias=False)
        if rms_val > 0:
            features["peak_factor"][i] = np.max(np.abs(x)) / rms_val
        else:
            features["peak_factor"][i] = 0.0

    return features


def plot_feature_boxplots(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    output_dir: str = OUTPUT_DIR,
):
    """
    针对每个特征画箱线图，比较不同类别的统计特性。
    """
    os.makedirs(output_dir, exist_ok=True)

    label_ids = sorted(CLASS_MAP.keys())
    class_names = [CLASS_MAP[c] for c in label_ids]

    for feat_name, feat_values in features.items():
        data_per_class = []
        for cls in label_ids:
            mask = labels == cls
            vals = feat_values[mask]
            if vals.size == 0:
                data_per_class.append([])
                print(f"[警告] 特征 {feat_name} 中，类别 {cls} 没有样本。")
            else:
                data_per_class.append(vals)

        plt.figure(figsize=(8, 5))
        plt.boxplot(data_per_class, tick_labels=class_names, showfliers=False)
        plt.ylabel(feat_name, fontproperties="SimHei")
        plt.title(f"{feat_name} 各故障类型分布", fontproperties="SimHei")
        plt.grid(True, linestyle="--", alpha=0.4)
        fname = os.path.join(output_dir, f"boxplot_{feat_name}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"已保存箱线图：{fname}")


def print_feature_summary(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
):
    """
    在命令行打印每个类别、每个特征的均值和标准差，便于写论文时引用。
    """
    label_ids = sorted(CLASS_MAP.keys())

    for cls in label_ids:
        mask = labels == cls
        if not np.any(mask):
            continue
        name = CLASS_MAP[cls]
        print("\n" + "=" * 60)
        print(f"类别 {cls} ({name})，样本数 = {mask.sum()}")
        for feat_name, feat_values in features.items():
            vals = feat_values[mask]
            mean_val = np.mean(vals)
            std_val = np.std(vals, ddof=1)
            print(f"{feat_name:>12s} : mean = {mean_val:8.4f}, std = {std_val:8.4f}")


# =========================
# 6. 主流程
# =========================

def main():
    # 1) 加载数据
    signals, labels = load_source_data()

    # 2) 为每个类别挑若干示例样本，并画时域 + 频域图
    example_indices = select_example_indices(labels)
    plot_time_and_spectrum_examples(signals, labels, example_indices)

    # 3) 计算统计特征，并画箱线图
    features = compute_signal_features(signals)
    plot_feature_boxplots(features, labels)

    # 4) 打印每个类别的特征统计汇总
    print_feature_summary(features, labels)

    print("\n分析完成！图片已保存在目录：", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
