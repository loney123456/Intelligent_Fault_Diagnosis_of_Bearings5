# import os
# import scipy.io
# import numpy as np
# from scipy import signal as scipy_signal
# from scipy import fft
#
# ROOT_DIR = r"datasets"
# SAMPLE_LEN = 2048  # 增加到2048以获得更好的频率分辨率
# STRIDE = 1024
#
# LABEL_MAP = {
#     'Normal': 0,
#     'IR': 1,
#     'OR': 2,
#     'B': 3
# }
#
# # 轴承参数
# SOURCE_BEARING = {
#     'n_balls': 9,
#     'd': 0.3126,  # 滚动体直径（英寸）
#     'D': 1.537,  # 节径（英寸）
#     'rpm': 1800,
#     'fs': 12000  # 采样率
# }
#
# TARGET_BEARING = {
#     'n_balls': 9,  # 假设相同
#     'd': 0.3126,  # 假设相同
#     'D': 1.537,  # 假设相同
#     'rpm': 600,
#     'fs': 32000
# }
#
#
# def calculate_fault_freqs(bearing_params):
#     """计算故障特征频率"""
#     fr = bearing_params['rpm'] / 60  # 转频
#     n = bearing_params['n_balls']
#     d = bearing_params['d']
#     D = bearing_params['D']
#
#     # 外圈故障频率
#     BPFO = (n / 2) * fr * (1 - d / D)
#     # 内圈故障频率
#     BPFI = (n / 2) * fr * (1 + d / D)
#     # 滚动体故障频率
#     BSF = (D / d) * fr * (1 - (d / D) ** 2)
#
#     return {
#         'fr': fr,
#         'BPFO': BPFO,
#         'BPFI': BPFI,
#         'BSF': BSF
#     }
#
#
# def extract_frequency_features(signal, fs, fault_freqs, target_fault_freqs):
#     """
#     提取频域特征，并做物理频率对齐
#     """
#     # FFT
#     n = len(signal)
#     fft_vals = fft.fft(signal)
#     fft_amp = np.abs(fft_vals[:n // 2])
#     freqs = fft.fftfreq(n, 1 / fs)[:n // 2]
#
#     # 特征：在故障特征频率附近提取能量
#     features = []
#
#     # 为了对齐，我们提取"相对于转频的倍数"
#     # 例如：BPFO = 3.5 * fr（源域）和 BPFO = 3.5 * fr（目标域）
#     # 虽然绝对频率不同，但相对倍数相同
#
#     # 方法：提取多个频带的能量
#     # 频带1: 0-2倍转频
#     # 频带2: 2-5倍转频
#     # 频带3: 5-10倍转频
#     # 频带4: 10-20倍转频
#
#     fr = fault_freqs['fr']
#     bands = [
#         (0, 2 * fr),
#         (2 * fr, 5 * fr),
#         (5 * fr, 10 * fr),
#         (10 * fr, 20 * fr),
#         (20 * fr, 50 * fr)
#     ]
#
#     for low, high in bands:
#         mask = (freqs >= low) & (freqs < high)
#         if mask.any():
#             band_energy = np.sum(fft_amp[mask])
#             features.append(band_energy)
#         else:
#             features.append(0.0)
#
#     # 添加时域统计特征
#     features.append(np.mean(signal))
#     features.append(np.std(signal))
#     features.append(np.max(np.abs(signal)))
#     features.append(np.sqrt(np.mean(signal ** 2)))  # RMS
#
#     return np.array(features, dtype=np.float32)
#
#
# def load_mat_data(filepath, key_filter='DE_time'):
#     """读取mat文件"""
#     try:
#         mat = scipy.io.loadmat(filepath)
#         for key in mat.keys():
#             if key.startswith('__'):
#                 continue
#             if key_filter in key:
#                 return mat[key].flatten()
#
#         sorted_keys = sorted(mat.keys(),
#                              key=lambda k: len(mat[k]) if hasattr(mat[k], '__len__') else 0,
#                              reverse=True)
#         for key in sorted_keys:
#             if not key.startswith('__'):
#                 return mat[key].flatten()
#         return None
#     except Exception as e:
#         print(f"Error loading {filepath}: {e}")
#         return None
#
#
# def slice_data(signal, label, fs, fault_freqs, target_fault_freqs=None):
#     """
#     切片并提取特征
#     """
#     data = []
#     labels = []
#
#     n_samples = (len(signal) - SAMPLE_LEN) // STRIDE + 1
#
#     for i in range(n_samples):
#         start = i * STRIDE
#         segment = signal[start: start + SAMPLE_LEN]
#
#         # 提取特征
#         if target_fault_freqs is None:
#             # 源域
#             features = extract_frequency_features(segment, fs, fault_freqs, fault_freqs)
#         else:
#             # 目标域 - 使用目标域的故障频率但传入源域的作为参考
#             features = extract_frequency_features(segment, fs, target_fault_freqs, fault_freqs)
#
#         data.append(features)
#         labels.append(label)
#
#     return data, labels
#
#
# def process_source_domain(root_path):
#     """处理源域数据"""
#     all_data = []
#     all_labels = []
#
#     source_fault_freqs = calculate_fault_freqs(SOURCE_BEARING)
#     print(f"源域故障特征频率: {source_fault_freqs}")
#
#     # 12kHz DE数据
#     de_path = os.path.join(root_path, 'source_domain_datasets', '12kHz_DE_data')
#     print(f"正在扫描: {de_path} ...")
#
#     if os.path.exists(de_path):
#         for fault_type in os.listdir(de_path):
#             type_path = os.path.join(de_path, fault_type)
#             if not os.path.isdir(type_path):
#                 continue
#
#             current_label = LABEL_MAP.get(fault_type, -1)
#             if current_label == -1:
#                 continue
#
#             if fault_type == 'OR':
#                 for position in os.listdir(type_path):
#                     pos_path = os.path.join(type_path, position)
#                     for diameter in os.listdir(pos_path):
#                         dia_path = os.path.join(pos_path, diameter)
#                         for file in os.listdir(dia_path):
#                             if file.endswith('.mat'):
#                                 sig = load_mat_data(os.path.join(dia_path, file))
#                                 if sig is not None:
#                                     d, l = slice_data(sig, current_label,
#                                                       SOURCE_BEARING['fs'],
#                                                       source_fault_freqs)
#                                     all_data.extend(d)
#                                     all_labels.extend(l)
#             else:
#                 for diameter in os.listdir(type_path):
#                     dia_path = os.path.join(type_path, diameter)
#                     if not os.path.isdir(dia_path):
#                         continue
#                     for file in os.listdir(dia_path):
#                         if file.endswith('.mat'):
#                             sig = load_mat_data(os.path.join(dia_path, file))
#                             if sig is not None:
#                                 d, l = slice_data(sig, current_label,
#                                                   SOURCE_BEARING['fs'],
#                                                   source_fault_freqs)
#                                 all_data.extend(d)
#                                 all_labels.extend(l)
#
#     # 48kHz正常数据
#     normal_path = os.path.join(root_path, 'source_domain_datasets', '48kHz_Normal_data')
#     print(f"正在扫描: {normal_path} ...")
#
#     if os.path.exists(normal_path):
#         for file in os.listdir(normal_path):
#             if file.endswith('.mat'):
#                 sig = load_mat_data(os.path.join(normal_path, file))
#                 if sig is not None:
#                     # 48k数据，故障频率x4
#                     sig = sig[::4]  # 降采样到12k
#                     d, l = slice_data(sig, LABEL_MAP['Normal'],
#                                       SOURCE_BEARING['fs'],
#                                       source_fault_freqs)
#                     all_data.extend(d)
#                     all_labels.extend(l)
#
#     return np.array(all_data), np.array(all_labels)
#
#
# def process_target_domain(root_path):
#     """处理目标域数据"""
#     target_dict = {}
#     target_path = os.path.join(root_path, 'target_domain_datasets')
#
#     target_fault_freqs = calculate_fault_freqs(TARGET_BEARING)
#     source_fault_freqs = calculate_fault_freqs(SOURCE_BEARING)
#     print(f"目标域故障特征频率: {target_fault_freqs}")
#
#     if os.path.exists(target_path):
#         for file in sorted(os.listdir(target_path)):
#             if file.endswith('.mat'):
#                 file_id = os.path.splitext(file)[0]
#                 sig = load_mat_data(os.path.join(target_path, file), key_filter='time')
#
#                 if sig is not None:
#                     d, _ = slice_data(sig, -1,
#                                       TARGET_BEARING['fs'],
#                                       target_fault_freqs,
#                                       source_fault_freqs)
#                     target_dict[file_id] = np.array(d)
#
#     return target_dict
#
#
# if __name__ == '__main__':
#     print("开始处理源域数据...")
#     source_x, source_y = process_source_domain(ROOT_DIR)
#     print(f"源域处理完成: Shape X={source_x.shape}, Y={source_y.shape}")
#
#     print("\n开始处理目标域数据...")
#     target_data = process_target_domain(ROOT_DIR)
#     print(f"目标域处理完成: 共 {len(target_data)} 个文件")
#
#     # 归一化特征
#     from sklearn.preprocessing import StandardScaler
#
#     scaler = StandardScaler()
#     source_x = scaler.fit_transform(source_x)
#
#     # 目标域也用同样的scaler
#     for key in target_data:
#         target_data[key] = scaler.transform(target_data[key])
#
#     np.save('source_x.npy', source_x)
#     np.save('source_y.npy', source_y)
#     np.save('target_data.npy', target_data)
#     print("\n所有数据已保存为 .npy 文件!")


import os
import scipy.io
import numpy as np

ROOT_DIR = r"datasets"
SAMPLE_LEN = 512  # 每个样本的时间序列长度
STRIDE = 256  # 滑动步长

LABEL_MAP = {
    'Normal': 0,
    'IR': 1,
    'OR': 2,
    'B': 3
}

# 采样率配置
SOURCE_FS_12K = 12000
SOURCE_FS_48K = 48000
TARGET_FS = 32000

# 统一的目标采样率（重采样到这个频率）
UNIFIED_FS = 12000


def resample_signal(signal, orig_fs, target_fs):
    """简单的重采样（通过插值或抽取）"""
    if orig_fs == target_fs:
        return signal

    ratio = target_fs / orig_fs
    new_len = int(len(signal) * ratio)

    # 使用线性插值重采样
    old_indices = np.arange(len(signal))
    new_indices = np.linspace(0, len(signal) - 1, new_len)
    resampled = np.interp(new_indices, old_indices, signal)

    return resampled


def load_mat_data(filepath, key_filter='DE_time'):
    """读取mat文件"""
    try:
        mat = scipy.io.loadmat(filepath)
        for key in mat.keys():
            if key.startswith('__'):
                continue
            if key_filter in key:
                return mat[key].flatten()

        # 如果没找到指定key，取最长的数组
        sorted_keys = sorted(mat.keys(),
                             key=lambda k: len(mat[k]) if hasattr(mat[k], '__len__') else 0,
                             reverse=True)
        for key in sorted_keys:
            if not key.startswith('__'):
                data = mat[key]
                if hasattr(data, 'flatten'):
                    return data.flatten()
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def slice_signal(signal, sample_len=SAMPLE_LEN, stride=STRIDE):
    """
    将长信号切成多个固定长度的样本
    返回: shape = (num_samples, sample_len)
    """
    samples = []
    n_samples = (len(signal) - sample_len) // stride + 1

    for i in range(n_samples):
        start = i * stride
        segment = signal[start: start + sample_len]
        samples.append(segment)

    return samples


def process_source_domain(root_path):
    """处理源域数据，返回原始时间序列"""
    all_data = []
    all_labels = []

    # ========== 12kHz DE数据 ==========
    de_path = os.path.join(root_path, 'source_domain_datasets', '12kHz_DE_data')
    print(f"正在扫描: {de_path} ...")

    if os.path.exists(de_path):
        for fault_type in os.listdir(de_path):
            type_path = os.path.join(de_path, fault_type)
            if not os.path.isdir(type_path):
                continue

            current_label = LABEL_MAP.get(fault_type, -1)
            if current_label == -1:
                print(f"  跳过未知类型: {fault_type}")
                continue

            print(f"  处理故障类型: {fault_type} (label={current_label})")

            if fault_type == 'OR':
                # OR有子目录（不同位置）
                for position in os.listdir(type_path):
                    pos_path = os.path.join(type_path, position)
                    if not os.path.isdir(pos_path):
                        continue
                    for diameter in os.listdir(pos_path):
                        dia_path = os.path.join(pos_path, diameter)
                        if not os.path.isdir(dia_path):
                            continue
                        for file in os.listdir(dia_path):
                            if file.endswith('.mat'):
                                sig = load_mat_data(os.path.join(dia_path, file))
                                if sig is not None:
                                    # 12kHz 数据，不需要重采样
                                    samples = slice_signal(sig)
                                    all_data.extend(samples)
                                    all_labels.extend([current_label] * len(samples))
            else:
                # IR, B, Normal 的目录结构
                for diameter in os.listdir(type_path):
                    dia_path = os.path.join(type_path, diameter)
                    if not os.path.isdir(dia_path):
                        # 可能直接是文件
                        if diameter.endswith('.mat'):
                            sig = load_mat_data(os.path.join(type_path, diameter))
                            if sig is not None:
                                samples = slice_signal(sig)
                                all_data.extend(samples)
                                all_labels.extend([current_label] * len(samples))
                        continue
                    for file in os.listdir(dia_path):
                        if file.endswith('.mat'):
                            sig = load_mat_data(os.path.join(dia_path, file))
                            if sig is not None:
                                samples = slice_signal(sig)
                                all_data.extend(samples)
                                all_labels.extend([current_label] * len(samples))

    # ========== 48kHz Normal 数据 ==========
    normal_path = os.path.join(root_path, 'source_domain_datasets', '48kHz_Normal_data')
    print(f"正在扫描: {normal_path} ...")

    if os.path.exists(normal_path):
        for file in os.listdir(normal_path):
            if file.endswith('.mat'):
                sig = load_mat_data(os.path.join(normal_path, file))
                if sig is not None:
                    # 48kHz -> 12kHz，降采样（取1/4）
                    sig_resampled = resample_signal(sig, SOURCE_FS_48K, UNIFIED_FS)
                    samples = slice_signal(sig_resampled)
                    all_data.extend(samples)
                    all_labels.extend([LABEL_MAP['Normal']] * len(samples))
                    print(f"    {file}: 48kHz降采样后切出 {len(samples)} 个样本")

    return np.array(all_data, dtype=np.float32), np.array(all_labels, dtype=np.int64)


def process_target_domain(root_path):
    """处理目标域数据，返回原始时间序列（无标签）"""
    target_dict = {}
    target_path = os.path.join(root_path, 'target_domain_datasets')

    print(f"正在扫描目标域: {target_path} ...")

    if os.path.exists(target_path):
        for file in sorted(os.listdir(target_path)):
            if file.endswith('.mat'):
                file_id = os.path.splitext(file)[0]
                sig = load_mat_data(os.path.join(target_path, file), key_filter='time')

                if sig is not None:
                    # 目标域是32kHz，重采样到12kHz
                    sig_resampled = resample_signal(sig, TARGET_FS, UNIFIED_FS)
                    samples = slice_signal(sig_resampled)
                    target_dict[file_id] = np.array(samples, dtype=np.float32)
                    print(f"  {file_id}: 切出 {len(samples)} 个样本")

    return target_dict


if __name__ == '__main__':
    print("=" * 60)
    print("开始处理源域数据...")
    print("=" * 60)
    source_x, source_y = process_source_domain(ROOT_DIR)
    print(f"\n源域处理完成:")
    print(f"  X shape: {source_x.shape}")
    print(f"  Y shape: {source_y.shape}")

    # 打印各类别样本数
    unique, counts = np.unique(source_y, return_counts=True)
    print("  各类别样本数:")
    for u, c in zip(unique, counts):
        label_name = [k for k, v in LABEL_MAP.items() if v == u][0]
        print(f"    {label_name} (label={u}): {c}")

    print("\n" + "=" * 60)
    print("开始处理目标域数据...")
    print("=" * 60)
    target_data = process_target_domain(ROOT_DIR)
    print(f"\n目标域处理完成: 共 {len(target_data)} 个文件")
    total_target_samples = sum(len(v) for v in target_data.values())
    print(f"  目标域总样本数: {total_target_samples}")

    # 保存
    np.save('source_x.npy', source_x)
    np.save('source_y.npy', source_y)
    np.save('target_data.npy', target_data)

    print("\n" + "=" * 60)
    print("所有数据已保存:")
    print("  source_x.npy")
    print("  source_y.npy")
    print("  target_data.npy")
    print("=" * 60)
