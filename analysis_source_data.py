# -*- coding: utf-8 -*-
"""
analysis_source_data.py

修复版本 - 解决频谱图显示异常问题

主要修复：
1. 频谱图故障特征频率线颜色区分
2. 限制频谱图显示范围到有意义的低频段
3. 箱线图添加颜色区分
4. 采样率根据CWRU数据集实际情况调整
"""

import os
import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 1. 基本配置
# =========================

# ============================================================
# 重要说明：根据 data_load_final.py 的设置
# 源域和目标域参数不同，但最终都重采样到 12kHz
# ============================================================

# 源域参数 (CWRU 数据集)
SOURCE_FS = 12000.0  # 统一采样率（12kHz DE数据，48kHz Normal降采样后）
SOURCE_RPM = 1800.0  # 源域转速 (RPM)

# 目标域参数 (高铁轴承数据)
TARGET_FS = 12000.0  # 统一采样率（32kHz 重采样到 12kHz）
TARGET_RPM = 600.0  # 目标域转速 (RPM)

# 当前分析使用的参数（分析源域时使用源域参数）
FS = SOURCE_FS
SHAFT_SPEED_RPM = SOURCE_RPM

# SKF 6205-2RS 轴承几何参数（源域和目标域假设相同）
N_BALLS = 9  # 滚动体个数
BALL_DIAMETER = 0.3126  # 球直径（英寸）
PITCH_DIAMETER = 1.537  # 节圆直径（英寸）
CONTACT_ANGLE_DEG = 0.0  # 接触角

# 类别标签映射
CLASS_MAP: Dict[int, str] = {
    0: "Normal",
    1: "Inner race fault",
    2: "Outer race fault",
    3: "Ball fault",
}

# 类别颜色（用于美化图表）
CLASS_COLORS = {
    0: '#2ecc71',  # 绿色 - Normal
    1: '#e74c3c',  # 红色 - IR
    2: '#3498db',  # 蓝色 - OR
    3: '#9b59b6',  # 紫色 - Ball
}

# 故障特征频率线颜色
FAULT_FREQ_COLORS = {
    'BPFI': '#e74c3c',  # 红色
    'BPFO': '#3498db',  # 蓝色
    'BSF': '#9b59b6',  # 紫色
    'FTF': '#f39c12',  # 橙色
}

EXAMPLES_PER_CLASS = 3
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
    """计算 BPFO/BPFI/BSF/FTF 四种故障特征频率（Hz）"""

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


def print_fault_frequencies_comparison():
    """
    打印源域和目标域的故障特征频率对比
    """
    source_freqs = compute_fault_frequencies(rpm=SOURCE_RPM)
    target_freqs = compute_fault_frequencies(rpm=TARGET_RPM)

    print("\n" + "=" * 70)
    print("故障特征频率对比 (源域 vs 目标域)")
    print("=" * 70)
    print(f"{'参数':<15} {'源域 (CWRU)':<20} {'目标域 (高铁)':<20} {'比值':<10}")
    print("-" * 70)
    print(f"{'采样率 (Hz)':<15} {SOURCE_FS:<20.0f} {TARGET_FS:<20.0f} {'-':<10}")
    print(f"{'转速 (RPM)':<15} {SOURCE_RPM:<20.0f} {TARGET_RPM:<20.0f} {SOURCE_RPM / TARGET_RPM:<10.2f}")
    print(
        f"{'转频 fr (Hz)':<15} {SOURCE_RPM / 60:<20.2f} {TARGET_RPM / 60:<20.2f} {(SOURCE_RPM / 60) / (TARGET_RPM / 60):<10.2f}")
    print("-" * 70)

    for name in ['BPFI', 'BPFO', 'BSF', 'FTF']:
        src = source_freqs[name]
        tgt = target_freqs[name]
        ratio = src / tgt if tgt > 0 else 0
        print(f"{name:<15} {src:<20.2f} {tgt:<20.2f} {ratio:<10.2f}")

    print("=" * 70)
    print("注意：源域转速是目标域的3倍，故障特征频率也是3倍关系")
    print("=" * 70 + "\n")

    return source_freqs, target_freqs


# =========================
# 3. 数据读取
# =========================

def load_source_data(
        x_path: str = "source_x.npy",
        y_path: str = "source_y.npy",
) -> Tuple[np.ndarray, np.ndarray]:
    """从 npy 文件加载源域数据"""

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"找不到 {x_path} 或 {y_path}")

    X = np.load(x_path)
    y = np.load(y_path)

    if X.ndim == 3:
        X = X[:, 0, :]

    y = y.astype(int).reshape(-1)

    print(f"加载数据成功：X 形状 = {X.shape}，y 形状 = {y.shape}")
    return X, y


def select_example_indices(
        labels: np.ndarray,
        examples_per_class: int = EXAMPLES_PER_CLASS,
) -> Dict[int, List[int]]:
    """为每个类别选出若干条样本的索引"""

    selected: Dict[int, List[int]] = {}
    for cls, name in CLASS_MAP.items():
        idx = np.where(labels == cls)[0]
        if idx.size == 0:
            print(f"[警告] 标签 {cls} ({name}) 在数据中不存在。")
            continue
        selected[cls] = idx[:examples_per_class].tolist()
        print(f"类别 {cls} ({name}) 选取了 {len(selected[cls])} 条样本")
    return selected


# =========================
# 4. 绘图函数（修复版）
# =========================

def add_fault_frequency_lines(
        ax: plt.Axes,
        fault_freqs_hz: Dict[str, float],
        fs: float,
        max_freq: float = None,
        max_harmonic: int = 3,
):
    """
    在频谱图上叠加故障特征频率及其倍频的竖线

    修复：
    1. 使用不同颜色区分不同故障类型
    2. 限制倍频数量避免过多竖线
    3. 添加透明度和线宽控制
    """
    if max_freq is None:
        max_freq = fs / 2.0

    legend_handles = []
    legend_labels = []

    for name, f0 in fault_freqs_hz.items():
        if f0 <= 0:
            continue

        color = FAULT_FREQ_COLORS.get(name, 'gray')

        for k in range(1, max_harmonic + 1):
            f = f0 * k
            if f >= max_freq:
                break

            # 基频用实线，倍频用虚线
            linestyle = '-' if k == 1 else '--'
            linewidth = 1.5 if k == 1 else 0.8
            alpha = 0.8 if k == 1 else 0.5

            line = ax.axvline(f, color=color, linestyle=linestyle,
                              linewidth=linewidth, alpha=alpha)

            # 只为基频添加图例
            if k == 1:
                legend_handles.append(line)
                legend_labels.append(f'{name} ({f0:.1f} Hz)')

    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='upper right',
                  fontsize=8, framealpha=0.9)


def plot_time_and_spectrum_examples(
        signals: np.ndarray,
        labels: np.ndarray,
        example_indices: Dict[int, List[int]],
        fs: float = FS,
        output_dir: str = OUTPUT_DIR,
):
    """
    对每个类别的若干条样本画时域+频域图

    修复：
    1. 频谱图限制显示范围到1000Hz（更好地显示故障频率）
    2. 使用包络谱代替原始频谱（可选）
    3. 改进图表美观度
    """
    os.makedirs(output_dir, exist_ok=True)
    fault_freqs = compute_fault_frequencies()

    for cls, indices in example_indices.items():
        class_name = CLASS_MAP.get(cls, f"class_{cls}")
        color = CLASS_COLORS.get(cls, 'blue')

        for k, idx in enumerate(indices):
            x = signals[idx]
            n = x.shape[0]
            t = np.arange(n) / fs

            # FFT频谱
            fft_vals = np.fft.rfft(x)
            fft_amp = np.abs(fft_vals) * 2 / n  # 单边谱幅值修正
            freqs = np.fft.rfftfreq(n, d=1.0 / fs)

            fig, axes = plt.subplots(2, 1, figsize=(12, 7))
            fig.suptitle(f"{class_name} | 样本 idx={idx}", fontsize=14, fontweight='bold')

            # === 时域波形 ===
            axes[0].plot(t, x, color=color, linewidth=0.5)
            axes[0].set_xlabel("时间 (s)", fontsize=11)
            axes[0].set_ylabel("加速度幅值", fontsize=11)
            axes[0].set_title("时域波形", fontsize=12)
            axes[0].grid(True, linestyle='--', alpha=0.4)
            axes[0].set_xlim(0, t[-1])

            # === 频谱图 ===
            # 限制显示范围到1000Hz，更清楚地看到故障特征频率
            max_display_freq = 1000  # Hz
            freq_mask = freqs <= max_display_freq

            axes[1].plot(freqs[freq_mask], fft_amp[freq_mask],
                         color=color, linewidth=0.8)
            axes[1].set_xlabel("频率 (Hz)", fontsize=11)
            axes[1].set_ylabel("幅值", fontsize=11)
            axes[1].set_title("频谱 (0-1000 Hz)", fontsize=12)
            axes[1].grid(True, linestyle='--', alpha=0.4)
            axes[1].set_xlim(0, max_display_freq)

            # 添加故障特征频率线
            add_fault_frequency_lines(axes[1], fault_freqs, fs,
                                      max_freq=max_display_freq, max_harmonic=3)

            # 使用下划线替换空格，保持文件名一致性
            safe_class_name = class_name.replace(' ', '_')
            fname = os.path.join(output_dir, f"{cls}_{safe_class_name}_idx{idx}.png")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(fname, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"已保存示例图：{fname}")


def plot_envelope_spectrum_examples(
        signals: np.ndarray,
        labels: np.ndarray,
        example_indices: Dict[int, List[int]],
        fs: float = FS,
        output_dir: str = OUTPUT_DIR,
):
    """
    绘制包络谱图（更适合观察故障特征频率）
    """
    from scipy.signal import hilbert

    os.makedirs(output_dir, exist_ok=True)
    fault_freqs = compute_fault_frequencies()

    for cls, indices in example_indices.items():
        class_name = CLASS_MAP.get(cls, f"class_{cls}")
        color = CLASS_COLORS.get(cls, 'blue')

        for k, idx in enumerate(indices):
            x = signals[idx]
            n = x.shape[0]

            # 计算包络
            analytic_signal = hilbert(x)
            envelope = np.abs(analytic_signal)

            # 包络谱
            env_fft = np.fft.rfft(envelope - np.mean(envelope))
            env_amp = np.abs(env_fft) * 2 / n
            freqs = np.fft.rfftfreq(n, d=1.0 / fs)

            fig, ax = plt.subplots(figsize=(12, 5))

            # 限制显示范围
            max_display_freq = 500  # Hz
            freq_mask = freqs <= max_display_freq

            ax.plot(freqs[freq_mask], env_amp[freq_mask],
                    color=color, linewidth=0.8)
            ax.set_xlabel("频率 (Hz)", fontsize=11)
            ax.set_ylabel("幅值", fontsize=11)
            ax.set_title(f"{class_name} | 样本 idx={idx} | 包络谱",
                         fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_xlim(0, max_display_freq)

            add_fault_frequency_lines(ax, fault_freqs, fs,
                                      max_freq=max_display_freq, max_harmonic=3)

            fname = os.path.join(output_dir, f"{cls}_{class_name.replace(' ', '_')}_envelope_idx{idx}.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"已保存包络谱图：{fname}")


# =========================
# 5. 统计特征与箱线图（美化版）
# =========================

def compute_signal_features(signals: np.ndarray) -> Dict[str, np.ndarray]:
    """计算统计特征"""

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
    绘制带颜色的箱线图
    """
    os.makedirs(output_dir, exist_ok=True)

    label_ids = sorted(CLASS_MAP.keys())
    class_names = [CLASS_MAP[c] for c in label_ids]
    colors = [CLASS_COLORS[c] for c in label_ids]

    # 预先计算每个类别的样本数量
    sample_counts = [np.sum(labels == cls) for cls in label_ids]

    # 创建带样本数量的X轴标签
    class_labels_with_n = [f"{name}\n(n={count})" for name, count in zip(class_names, sample_counts)]

    # 特征中文名映射
    feature_names_cn = {
        'mean': '均值',
        'std': '标准差',
        'rms': '均方根值',
        'kurtosis': '峭度',
        'peak_factor': '峰值因子'
    }

    for feat_name, feat_values in features.items():
        data_per_class = []
        for cls in label_ids:
            mask = labels == cls
            vals = feat_values[mask]
            data_per_class.append(vals if vals.size > 0 else [])

        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制带颜色的箱线图
        bp = ax.boxplot(data_per_class, tick_labels=class_labels_with_n,
                        patch_artist=True, showfliers=False)

        # 设置颜色
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # 设置中位线颜色
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        # 设置须线颜色
        for whisker in bp['whiskers']:
            whisker.set_color('gray')
            whisker.set_linewidth(1.5)

        for cap in bp['caps']:
            cap.set_color('gray')
            cap.set_linewidth(1.5)

        feat_cn = feature_names_cn.get(feat_name, feat_name)
        ax.set_ylabel(feat_cn, fontsize=12)
        ax.set_title(f"{feat_cn} 各故障类型分布对比", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4, axis='y')

        fname = os.path.join(output_dir, f"boxplot_{feat_name}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"已保存箱线图：{fname}")


def plot_all_features_combined(
        features: Dict[str, np.ndarray],
        labels: np.ndarray,
        output_dir: str = OUTPUT_DIR,
):
    """
    绘制所有特征的组合箱线图（2x3布局）
    """
    os.makedirs(output_dir, exist_ok=True)

    label_ids = sorted(CLASS_MAP.keys())
    class_names = [CLASS_MAP[c] for c in label_ids]
    colors = [CLASS_COLORS[c] for c in label_ids]

    # 预先计算每个类别的样本数量，创建带n的标签
    sample_counts = [np.sum(labels == cls) for cls in label_ids]
    class_labels_with_n = [f"{name}\n(n={count})" for name, count in zip(class_names, sample_counts)]

    feature_names_cn = {
        'mean': '均值',
        'std': '标准差',
        'rms': '均方根值',
        'kurtosis': '峭度',
        'peak_factor': '峰值因子'
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (feat_name, feat_values) in enumerate(features.items()):
        if idx >= 6:
            break

        ax = axes[idx]
        data_per_class = []
        for cls in label_ids:
            mask = labels == cls
            vals = feat_values[mask]
            data_per_class.append(vals if vals.size > 0 else [])

        bp = ax.boxplot(data_per_class, tick_labels=class_labels_with_n,
                        patch_artist=True, showfliers=False)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        feat_cn = feature_names_cn.get(feat_name, feat_name)
        ax.set_title(feat_cn, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4, axis='y')
        ax.tick_params(axis='x', rotation=15, labelsize=9)

    # 隐藏多余的子图
    for idx in range(len(features), 6):
        axes[idx].set_visible(False)

    fig.suptitle("源域数据统计特征分析", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fname = os.path.join(output_dir, "all_features_boxplot.png")
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"已保存组合箱线图：{fname}")


def print_feature_summary(
        features: Dict[str, np.ndarray],
        labels: np.ndarray,
):
    """打印特征统计汇总"""

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
    print("=" * 60)
    print("源域数据分析 - 修复版")
    print("=" * 60)

    # 0) 打印源域和目标域参数对比
    print_fault_frequencies_comparison()

    # 1) 加载数据
    signals, labels = load_source_data()

    # 2) 选取示例样本
    example_indices = select_example_indices(labels)

    # 3) 画时域 + 频域图（修复版）
    print("\n正在生成时域+频谱图...")
    plot_time_and_spectrum_examples(signals, labels, example_indices)

    # 4) 画包络谱图（可选，更适合观察故障频率）
    print("\n正在生成包络谱图...")
    plot_envelope_spectrum_examples(signals, labels, example_indices)

    # 5) 计算统计特征
    print("\n正在计算统计特征...")
    features = compute_signal_features(signals)

    # 6) 画箱线图（美化版）
    print("\n正在生成箱线图...")
    plot_feature_boxplots(features, labels)

    # 7) 画组合箱线图
    plot_all_features_combined(features, labels)

    # 8) 打印统计汇总
    print_feature_summary(features, labels)

    print("\n" + "=" * 60)
    print(f"分析完成！图片已保存在目录：{os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()