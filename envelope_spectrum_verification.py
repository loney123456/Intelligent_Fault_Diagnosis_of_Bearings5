# envelope_spectrum_verification.py
# 功能：对低置信度文件(D、E、F)进行包络谱分析，验证故障类型
# ====================================================================

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert, butter, filtfilt
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =========================================
# 目标域轴承参数（根据题目信息估算）
# =========================================
# 题目信息：列车速度约90km/h，轴承转速约600rpm，采样频率32kHz

FS = 32000  # 采样频率 32kHz
RPM = 600  # 转速约600rpm
FR = RPM / 60  # 转频 = 10 Hz

# 列车轴承典型参数（估算值）
# 参考：列车轴承一般为圆柱滚子轴承，滚动体数量较多
# 这里使用典型参数进行估算

# 估算方法1：使用经验公式
# 对于典型轴承：BPFO ≈ 0.4 * n * fr, BPFI ≈ 0.6 * n * fr
# 假设滚动体数量 n ≈ 13-17

# 我们计算多种可能的参数组合
BEARING_PARAMS = {
    '参数组1 (n=13)': {'n': 13, 'd': 25, 'D': 120},  # 典型小型轴承
    '参数组2 (n=15)': {'n': 15, 'd': 28, 'D': 130},  # 典型中型轴承
    '参数组3 (n=17)': {'n': 17, 'd': 30, 'D': 140},  # 典型大型轴承
}


def calculate_fault_frequencies(n, d, D, fr):
    """
    计算轴承故障特征频率

    参数:
        n: 滚动体数量
        d: 滚动体直径 (mm)
        D: 轴承节径 (mm)
        fr: 转频 (Hz)

    返回:
        BPFO: 外圈故障特征频率
        BPFI: 内圈故障特征频率
        BSF: 滚动体故障特征频率
        FTF: 保持架故障特征频率
    """
    ratio = d / D

    BPFO = n * fr / 2 * (1 - ratio)  # 外圈故障频率
    BPFI = n * fr / 2 * (1 + ratio)  # 内圈故障频率
    BSF = D / d * fr / 2 * (1 - ratio ** 2)  # 滚动体故障频率
    FTF = fr / 2 * (1 - ratio)  # 保持架故障频率

    return BPFO, BPFI, BSF, FTF


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # 确保频率在有效范围内
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))

    if low >= high:
        high = min(low + 0.1, 0.999)

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def envelope_spectrum(data, fs, lowcut=1000, highcut=5000):
    """
    计算包络谱

    步骤:
    1. 带通滤波（选择共振频带）
    2. 希尔伯特变换提取包络
    3. FFT计算包络谱
    """
    # 1. 带通滤波
    filtered = bandpass_filter(data, lowcut, highcut, fs)

    # 2. 希尔伯特变换提取包络
    analytic_signal = hilbert(filtered)
    envelope = np.abs(analytic_signal)

    # 去除直流分量
    envelope = envelope - np.mean(envelope)

    # 3. FFT计算包络谱
    n = len(envelope)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    spectrum = np.abs(np.fft.rfft(envelope)) / n

    return freqs, spectrum, envelope


def find_peaks_near_frequency(freqs, spectrum, target_freq, tolerance=3):
    """在目标频率附近寻找峰值"""
    mask = (freqs >= target_freq - tolerance) & (freqs <= target_freq + tolerance)
    if mask.sum() == 0:
        return None, 0

    local_freqs = freqs[mask]
    local_spectrum = spectrum[mask]

    peak_idx = np.argmax(local_spectrum)
    return local_freqs[peak_idx], local_spectrum[peak_idx]


def analyze_file(file_data, file_id, fs=FS, fr=FR):
    """分析单个文件的包络谱"""

    print(f"\n{'=' * 70}")
    print(f"文件 {file_id} 包络谱分析")
    print(f"{'=' * 70}")

    # 取所有样本的平均或拼接
    # 这里我们分析多个样本，取有代表性的结果
    n_samples = len(file_data)
    sample_length = file_data.shape[1]

    # 拼接前10个样本作为长信号进行分析
    n_use = min(10, n_samples)
    long_signal = file_data[:n_use].flatten()

    print(f"信号长度: {len(long_signal)} 点 ({len(long_signal) / fs:.2f} 秒)")
    print(f"转频: {fr:.2f} Hz (转速: {RPM} rpm)")

    # 计算不同参数下的故障频率
    print(f"\n估算的故障特征频率:")
    print("-" * 50)

    fault_freqs_list = []
    for name, params in BEARING_PARAMS.items():
        BPFO, BPFI, BSF, FTF = calculate_fault_frequencies(
            params['n'], params['d'], params['D'], fr
        )
        fault_freqs_list.append({
            'name': name,
            'BPFO': BPFO,
            'BPFI': BPFI,
            'BSF': BSF,
            'FTF': FTF
        })
        print(f"{name}: BPFO={BPFO:.2f}Hz, BPFI={BPFI:.2f}Hz, BSF={BSF:.2f}Hz")

    # 使用中间参数组作为参考
    ref_params = BEARING_PARAMS['参数组2 (n=15)']
    BPFO, BPFI, BSF, FTF = calculate_fault_frequencies(
        ref_params['n'], ref_params['d'], ref_params['D'], fr
    )

    # 尝试不同的滤波频带
    filter_bands = [
        (500, 3000, "低频带 500-3000Hz"),
        (1000, 5000, "中频带 1000-5000Hz"),
        (2000, 8000, "高频带 2000-8000Hz"),
        (3000, 12000, "超高频带 3000-12000Hz"),
    ]

    results = []

    for lowcut, highcut, band_name in filter_bands:
        try:
            freqs, spectrum, envelope = envelope_spectrum(long_signal, fs, lowcut, highcut)

            # 在故障频率附近寻找峰值
            _, bpfo_amp = find_peaks_near_frequency(freqs, spectrum, BPFO, tolerance=5)
            _, bpfi_amp = find_peaks_near_frequency(freqs, spectrum, BPFI, tolerance=5)
            _, bsf_amp = find_peaks_near_frequency(freqs, spectrum, BSF, tolerance=5)

            # 也检查谐波
            _, bpfo_2x = find_peaks_near_frequency(freqs, spectrum, 2 * BPFO, tolerance=5)
            _, bpfi_2x = find_peaks_near_frequency(freqs, spectrum, 2 * BPFI, tolerance=5)

            results.append({
                'band': band_name,
                'lowcut': lowcut,
                'highcut': highcut,
                'freqs': freqs,
                'spectrum': spectrum,
                'BPFO_amp': bpfo_amp,
                'BPFI_amp': bpfi_amp,
                'BSF_amp': bsf_amp,
                'BPFO_2x': bpfo_2x,
                'BPFI_2x': bpfi_2x,
            })
        except Exception as e:
            print(f"  {band_name}: 处理失败 - {e}")

    # 找出最佳频带（故障特征最明显的）
    best_result = max(results, key=lambda x: max(x['BPFO_amp'], x['BPFI_amp'], x['BSF_amp']))

    print(f"\n各频带故障特征强度分析:")
    print("-" * 70)
    print(f"{'频带':<25} {'BPFO强度':<12} {'BPFI强度':<12} {'BSF强度':<12} {'判断'}")
    print("-" * 70)

    for r in results:
        # 判断最可能的故障类型
        amps = {'OR': r['BPFO_amp'], 'IR': r['BPFI_amp'], 'Ball': r['BSF_amp']}
        max_type = max(amps, key=amps.get)
        max_amp = amps[max_type]

        # 计算相对强度
        total = sum(amps.values()) + 1e-10
        ratio = max_amp / total * 100

        if ratio > 50:
            judgment = f"→ {max_type} ({ratio:.0f}%)"
        else:
            judgment = "不确定"

        print(f"{r['band']:<25} {r['BPFO_amp']:<12.6f} {r['BPFI_amp']:<12.6f} "
              f"{r['BSF_amp']:<12.6f} {judgment}")

    # 综合判断
    print(f"\n综合分析:")
    print("-" * 50)

    total_bpfo = sum(r['BPFO_amp'] + r.get('BPFO_2x', 0) for r in results)
    total_bpfi = sum(r['BPFI_amp'] + r.get('BPFI_2x', 0) for r in results)
    total_bsf = sum(r['BSF_amp'] for r in results)

    total = total_bpfo + total_bpfi + total_bsf + 1e-10

    print(f"  外圈故障(OR)特征强度: {total_bpfo:.6f} ({total_bpfo / total * 100:.1f}%)")
    print(f"  内圈故障(IR)特征强度: {total_bpfi:.6f} ({total_bpfi / total * 100:.1f}%)")
    print(f"  滚动体(Ball)特征强度: {total_bsf:.6f} ({total_bsf / total * 100:.1f}%)")

    # 最终判断
    fault_scores = {'OR': total_bpfo, 'IR': total_bpfi, 'Ball': total_bsf}
    predicted_fault = max(fault_scores, key=fault_scores.get)
    confidence = fault_scores[predicted_fault] / total * 100

    print(f"\n  📊 包络谱诊断结果: {predicted_fault} (特征占比: {confidence:.1f}%)")

    return {
        'file_id': file_id,
        'envelope_prediction': predicted_fault,
        'envelope_confidence': confidence,
        'BPFO_score': total_bpfo / total * 100,
        'BPFI_score': total_bpfi / total * 100,
        'BSF_score': total_bsf / total * 100,
        'best_result': best_result,
        'all_results': results,
        'fault_freqs': {'BPFO': BPFO, 'BPFI': BPFI, 'BSF': BSF}
    }


def plot_envelope_spectrum(analysis_result, file_data, fs=FS):
    """绘制包络谱分析图"""

    file_id = analysis_result['file_id']
    best_result = analysis_result['best_result']
    fault_freqs = analysis_result['fault_freqs']

    # 准备数据
    n_use = min(10, len(file_data))
    long_signal = file_data[:n_use].flatten()

    # 重新计算用于绘图
    freqs, spectrum, envelope = envelope_spectrum(
        long_signal, fs,
        best_result['lowcut'],
        best_result['highcut']
    )

    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 1. 原始信号（一小段）
    ax1 = axes[0]
    t = np.arange(len(long_signal[:8000])) / fs * 1000  # ms
    ax1.plot(t, long_signal[:8000], 'b-', linewidth=0.5)
    ax1.set_xlabel('时间 (ms)')
    ax1.set_ylabel('幅值')
    ax1.set_title(f'文件{file_id}: 原始振动信号 (前250ms)')
    ax1.grid(True, alpha=0.3)

    # 2. 包络信号
    ax2 = axes[1]
    t_env = np.arange(len(envelope[:8000])) / fs * 1000
    ax2.plot(t_env, envelope[:8000], 'g-', linewidth=0.5)
    ax2.set_xlabel('时间 (ms)')
    ax2.set_ylabel('包络幅值')
    ax2.set_title(f'文件{file_id}: 包络信号 (滤波频带: {best_result["lowcut"]}-{best_result["highcut"]}Hz)')
    ax2.grid(True, alpha=0.3)

    # 3. 包络谱
    ax3 = axes[2]

    # 只显示0-300Hz范围
    freq_mask = freqs <= 300
    ax3.plot(freqs[freq_mask], spectrum[freq_mask], 'b-', linewidth=1)

    # 标注故障特征频率
    colors = {'BPFO': 'red', 'BPFI': 'orange', 'BSF': 'purple'}
    labels = {'BPFO': '外圈故障频率', 'BPFI': '内圈故障频率', 'BSF': '滚动体故障频率'}

    ymax = spectrum[freq_mask].max() * 1.2

    for fault_type, freq in fault_freqs.items():
        color = colors[fault_type]
        label = labels[fault_type]

        # 基频
        ax3.axvline(freq, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
        ax3.text(freq, ymax * 0.95, f'{label}\n{freq:.1f}Hz',
                 ha='center', va='top', fontsize=8, color=color,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 二倍频
        if 2 * freq <= 300:
            ax3.axvline(2 * freq, color=color, linestyle=':', alpha=0.5, linewidth=1)
            ax3.text(2 * freq, ymax * 0.7, f'2×{fault_type}\n{2 * freq:.1f}Hz',
                     ha='center', va='top', fontsize=7, color=color)

    # 标注转频
    ax3.axvline(FR, color='green', linestyle='-', alpha=0.5, linewidth=2)
    ax3.text(FR, ymax * 0.5, f'转频\n{FR:.1f}Hz', ha='center', fontsize=8, color='green')

    ax3.set_xlabel('频率 (Hz)')
    ax3.set_ylabel('幅值')
    ax3.set_title(f'文件{file_id}: 包络谱 (诊断结果: {analysis_result["envelope_prediction"]}, '
                  f'置信度: {analysis_result["envelope_confidence"]:.1f}%)')
    ax3.set_xlim(0, 300)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("包络谱验证分析 - 针对低置信度文件 D、E、F")
    print("=" * 70)

    # 加载数据
    target_dict = np.load("target_data.npy", allow_pickle=True).item()
    print(f"✅ 加载目标域数据: {len(target_dict)} 个文件")

    # 要分析的文件
    files_to_analyze = ['D', 'E', 'F']

    # 模型预测结果（之前的结果）
    model_predictions = {
        'D': {'pred': 'OR', 'vote_ratio': 54.8, 'confidence': 0.6943},
        'E': {'pred': 'OR', 'vote_ratio': 52.1, 'confidence': 0.9640},
        'F': {'pred': 'OR', 'vote_ratio': 41.4, 'confidence': 0.6578},
    }

    # 分析每个文件
    analysis_results = {}

    for file_id in files_to_analyze:
        result = analyze_file(target_dict[file_id], file_id)
        analysis_results[file_id] = result

    # =========================================
    # 绘制包络谱图
    # =========================================
    print("\n" + "=" * 70)
    print("生成包络谱可视化图")
    print("=" * 70)

    for file_id in files_to_analyze:
        fig = plot_envelope_spectrum(analysis_results[file_id], target_dict[file_id])
        filename = f'envelope_spectrum_{file_id}.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ 已保存: {filename}")
        plt.close(fig)

    # =========================================
    # 生成综合对比图
    # =========================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, file_id in enumerate(files_to_analyze):
        ax = axes[idx]
        result = analysis_results[file_id]

        # 绘制故障类型得分条形图
        fault_types = ['OR\n(外圈)', 'IR\n(内圈)', 'Ball\n(滚动体)']
        scores = [result['BPFO_score'], result['BPFI_score'], result['BSF_score']]
        colors = ['red' if s == max(scores) else 'steelblue' for s in scores]

        bars = ax.bar(fault_types, scores, color=colors, edgecolor='black')

        # 添加数值标签
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{score:.1f}%', ha='center', va='bottom', fontsize=10)

        # 模型预测信息
        model_pred = model_predictions[file_id]
        ax.set_title(f'文件{file_id}\n模型预测: {model_pred["pred"]} (投票{model_pred["vote_ratio"]:.1f}%)\n'
                     f'包络谱诊断: {result["envelope_prediction"]}', fontsize=11)
        ax.set_ylabel('特征频率能量占比 (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('包络谱验证分析 - 故障特征频率能量对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('envelope_verification_summary.png', dpi=150, bbox_inches='tight')
    print("✅ 已保存: envelope_verification_summary.png")
    plt.close()

    # =========================================
    # 最终对比总结
    # =========================================
    print("\n" + "=" * 70)
    print("最终对比总结：模型预测 vs 包络谱验证")
    print("=" * 70)

    print(
        f"\n{'文件':<6} {'模型预测':<10} {'投票%':<10} {'置信度':<10} {'包络谱诊断':<12} {'包络谱置信度':<12} {'是否一致'}")
    print("-" * 80)

    consistent_count = 0
    for file_id in files_to_analyze:
        model_pred = model_predictions[file_id]
        envelope_result = analysis_results[file_id]

        is_consistent = model_pred['pred'] == envelope_result['envelope_prediction']
        consistent_count += is_consistent

        print(f"{file_id:<6} {model_pred['pred']:<10} {model_pred['vote_ratio']:<10.1f} "
              f"{model_pred['confidence']:<10.4f} {envelope_result['envelope_prediction']:<12} "
              f"{envelope_result['envelope_confidence']:<12.1f} {'✅ 一致' if is_consistent else '❌ 不一致'}")

    print("-" * 80)
    print(
        f"\n一致性统计: {consistent_count}/{len(files_to_analyze)} ({consistent_count / len(files_to_analyze) * 100:.0f}%)")

    # =========================================
    # 详细分析说明
    # =========================================
    print("\n" + "=" * 70)
    print("详细分析说明")
    print("=" * 70)

    for file_id in files_to_analyze:
        model_pred = model_predictions[file_id]
        envelope_result = analysis_results[file_id]

        print(f"\n【文件 {file_id}】")
        print("-" * 50)
        print(
            f"  模型预测: {model_pred['pred']} (投票比例: {model_pred['vote_ratio']:.1f}%, 置信度: {model_pred['confidence']:.4f})")
        print(
            f"  包络谱诊断: {envelope_result['envelope_prediction']} (特征占比: {envelope_result['envelope_confidence']:.1f}%)")
        print(f"  各故障类型特征能量:")
        print(f"    - 外圈故障(BPFO): {envelope_result['BPFO_score']:.1f}%")
        print(f"    - 内圈故障(BPFI): {envelope_result['BPFI_score']:.1f}%")
        print(f"    - 滚动体故障(BSF): {envelope_result['BSF_score']:.1f}%")

        if model_pred['pred'] == envelope_result['envelope_prediction']:
            print(f"  ✅ 结论: 包络谱验证支持模型预测结果，该文件确为{model_pred['pred']}故障")
        else:
            # 分析差异原因
            print(f"  ⚠️ 结论: 模型预测与包络谱结果不一致，需要进一步分析")
            print(f"     可能原因:")
            print(f"     1. 故障特征不够典型，两种方法侧重点不同")
            print(f"     2. 可能存在多种故障特征的混合")
            print(f"     3. 信号质量或噪声影响")

    # =========================================
    # 最终建议
    # =========================================
    print("\n" + "=" * 70)
    print("最终诊断建议")
    print("=" * 70)

    print("""
基于模型预测和包络谱验证的综合分析：

1. 如果两种方法结果一致 → 可以高可信度确认故障类型
2. 如果两种方法结果不一致 → 建议：
   a) 以模型预测为主（因为模型学习了更多特征）
   b) 在报告中注明存在不确定性
   c) 可能是轻微故障或复合故障的早期阶段

最终标签建议（用于提交）：
""")

    for file_id in files_to_analyze:
        model_pred = model_predictions[file_id]
        envelope_result = analysis_results[file_id]

        # 综合判断
        if model_pred['pred'] == envelope_result['envelope_prediction']:
            final_label = model_pred['pred']
            confidence_level = "高"
        else:
            # 如果不一致，分析哪个更可信
            if model_pred['confidence'] > 0.8:
                final_label = model_pred['pred']
                confidence_level = "中"
            elif envelope_result['envelope_confidence'] > 60:
                final_label = envelope_result['envelope_prediction']
                confidence_level = "中"
            else:
                final_label = model_pred['pred']  # 默认用模型结果
                confidence_level = "低"

        print(f"  文件{file_id}: {final_label} (可信度: {confidence_level})")

    print("\n✅ 包络谱验证分析完成!")

    return analysis_results


if __name__ == "__main__":
    results = main()
