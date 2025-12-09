# import os
# import scipy.io
# import numpy as np
# from scipy import signal as scipy_signal
#
# # ================= 配置区域 =================
# # 根目录：根据你的截图，这里填 'datasets' 的绝对路径
# # 例如: r"E:\Project\Intelligent Fault Diagnosis of Bearings\datasets"
# ROOT_DIR = r"datasets"
#
# # 定义样本参数
# SAMPLE_LEN = 1024
# STRIDE = 1024
#
# # 标签映射
# LABEL_MAP = {
#     'Normal': 0,  # 正常
#     'IR': 1,  # 内圈
#     'OR': 2,  # 外圈
#     'B': 3  # 滚动体
# }
#
#
# # ===========================================
#
# def load_mat_data(filepath, key_filter='DE_time'):
#     """通用读取函数：自动寻找包含特定关键词(如DE_time)的变量"""
#     try:
#         mat = scipy.io.loadmat(filepath)
#         for key in mat.keys():
#             # 过滤掉系统自带的 __header__ 等key
#             if key.startswith('__'): continue
#
#             # 策略：优先找 DE_time，如果找不到且是正常数据，可能找别的
#             if key_filter in key:
#                 return mat[key].flatten()
#
#         # 如果通过 key_filter 没找到，尝试找最长的数据（兜底策略）
#         # 针对部分 Normal 数据可能命名不一样的情况
#         sorted_keys = sorted(mat.keys(), key=lambda k: len(mat[k]) if hasattr(mat[k], '__len__') else 0, reverse=True)
#         for key in sorted_keys:
#             if not key.startswith('__'):
#                 return mat[key].flatten()
#
#         return None
#     except Exception as e:
#         print(f"Error loading {filepath}: {e}")
#         return None
#
#
# # def slice_data(signal, label):
# #     """将长信号切片成样本"""
# #     data = []
# #     labels = []
# #     n_samples = (len(signal) - SAMPLE_LEN) // STRIDE + 1
# #     for i in range(n_samples):
# #         start = i * STRIDE
# #         segment = signal[start: start + SAMPLE_LEN]
# #         data.append(segment)
# #         labels.append(label)
# #     return data, labels
#
#
# # def slice_data(signal, label):
# #     """将长信号切片，并进行 FFT 变换"""
# #     data = []
# #     labels = []
# #     n_samples = (len(signal) - SAMPLE_LEN) // STRIDE + 1
# #
# #     for i in range(n_samples):
# #         start = i * STRIDE
# #         segment = signal[start: start + SAMPLE_LEN]
# #
# #         # ============ 【核心修改：加入 FFT】 ============
# #         # 1. 傅里叶变换
# #         fft_res = np.fft.fft(segment)
# #         # 2. 取模（求幅值）
# #         fft_amp = np.abs(fft_res)
# #         # 3. 归一化 (让数据在 0-1 之间，这一步对迁移学习极其重要！)
# #         fft_amp = fft_amp / (np.max(fft_amp) + 1e-5)
# #         # 4. 取一半 (FFT是对称的，只取前一半有效数据)
# #         fft_amp = fft_amp[:SAMPLE_LEN // 2]
# #         # ===============================================
# #
# #         data.append(fft_amp)
# #         labels.append(label)
# #
# #     return data, labels
#
# # def slice_data(signal, label):
# #     data = []
# #     labels = []
# #     n_samples = (len(signal) - SAMPLE_LEN) // STRIDE + 1
# #
# #     for i in range(n_samples):
# #         start = i * STRIDE
# #         segment = signal[start: start + SAMPLE_LEN]
# #
# #         # 1. FFT
# #         fft_res = np.fft.fft(segment)
# #         fft_amp = np.abs(fft_res)
# #         fft_amp = fft_amp[:SAMPLE_LEN // 2]
# #
# #         # ============ 【核心修改：Log 变换】 ============
# #         # 使用 Log1p 把数据压缩，这对于跨工况迁移极其重要！
# #         fft_amp = np.log1p(fft_amp)
# #         # ==============================================
# #
# #         # 2. 归一化 (Min-Max)
# #         # 将数据严格限制在 0-1 之间
# #         _min = np.min(fft_amp)
# #         _max = np.max(fft_amp)
# #         if _max - _min > 1e-5:
# #             fft_amp = (fft_amp - _min) / (_max - _min)
# #         else:
# #             fft_amp = np.zeros_like(fft_amp)
# #
# #         data.append(fft_amp)
# #         labels.append(label)
# #
# #     return data, labels
#
# def slice_data(signal, label):
#     data = []
#     labels = []
#     n_samples = (len(signal) - SAMPLE_LEN) // STRIDE + 1
#
#     for i in range(n_samples):
#         start = i * STRIDE
#         segment = signal[start: start + SAMPLE_LEN]
#
#         # ============ 【新增核心操作：源域加噪】 ============
#         # 只有当处理源域数据时（label != -1），我们才加噪音
#         # 目标域（label == -1）本身就很脏，不用加
#         if label != -1:
#             # 生成高斯噪声：均值为0，标准差为0.5（这个值可以调整，越大越脏）
#             # 你可以尝试 0.5 到 1.0 之间的值
#             noise = np.random.normal(0, 0.5, segment.shape)
#             segment = segment + noise
#         # =================================================
#
#         # 下面继续是之前的 FFT 和 Log 处理 (保持不变)
#         fft_res = np.fft.fft(segment)
#         fft_amp = np.abs(fft_res)
#         fft_amp = fft_amp[:SAMPLE_LEN // 2]
#
#         fft_amp = np.log1p(fft_amp)
#
#         _min = np.min(fft_amp)
#         _max = np.max(fft_amp)
#         if _max - _min > 1e-5:
#             fft_amp = (fft_amp - _min) / (_max - _min)
#         else:
#             fft_amp = np.zeros_like(fft_amp)
#
#         data.append(fft_amp)
#         labels.append(label)
#
#     return data, labels
#
# def process_source_domain(root_path):
#     all_data = []
#     all_labels = []
#
#     # --- 1. 处理 12kHz_DE_data (包含 B, IR, OR) ---
#     de_path = os.path.join(root_path, 'source_domain_datasets', '12kHz_DE_data')
#     print(f"正在扫描: {de_path} ...")
#
#     if os.path.exists(de_path):
#         for fault_type in os.listdir(de_path):  # B, IR, OR
#             type_path = os.path.join(de_path, fault_type)
#             if not os.path.isdir(type_path): continue
#
#             current_label = LABEL_MAP.get(fault_type, -1)
#             if current_label == -1: continue
#
#             # 特殊处理 OR (外圈)，因为它多了一层 (Centered/Opposite/Orthogonal)
#             if fault_type == 'OR':
#                 for position in os.listdir(type_path):  # Centered, Opposite...
#                     pos_path = os.path.join(type_path, position)
#                     for diameter in os.listdir(pos_path):  # 0007, 0014...
#                         dia_path = os.path.join(pos_path, diameter)
#                         for file in os.listdir(dia_path):
#                             if file.endswith('.mat'):
#                                 sig = load_mat_data(os.path.join(dia_path, file))
#                                 if sig is not None:
#                                     d, l = slice_data(sig, current_label)
#                                     all_data.extend(d)
#                                     all_labels.extend(l)
#
#             # 处理 B 和 IR (结构较浅: B -> 0007 -> file)
#             else:
#                 for diameter in os.listdir(type_path):  # 0007, 0014...
#                     dia_path = os.path.join(type_path, diameter)
#                     if not os.path.isdir(dia_path): continue
#                     for file in os.listdir(dia_path):
#                         if file.endswith('.mat'):
#                             sig = load_mat_data(os.path.join(dia_path, file))
#                             if sig is not None:
#                                 d, l = slice_data(sig, current_label)
#                                 all_data.extend(d)
#                                 all_labels.extend(l)
#
#     # --- 2. 处理 48kHz_Normal_data (正常数据) ---
#     # 注意：通常做对比实验时，采样率最好一致。
#     # 但根据目录树，正常数据只在48k文件夹里，或者你需要确认12k_DE里是否有正常数据。
#     # 这里我们加载 48kHz_Normal_data 作为 0 类。
#     normal_path = os.path.join(root_path, 'source_domain_datasets', '48kHz_Normal_data')
#     print(f"正在扫描: {normal_path} ...")
#
#     if os.path.exists(normal_path):
#         for file in os.listdir(normal_path):
#             if file.endswith('.mat'):
#                 sig = load_mat_data(os.path.join(normal_path, file))
#                 if sig is not None:
#                     # 48k数据可能需要降采样到12k，这里为了代码跑通先不处理，直接切片
#                     # 如果需要严谨，每隔4个点取1个点: sig = sig[::4]
#                     d, l = slice_data(sig, LABEL_MAP['Normal'])
#                     all_data.extend(d)
#                     all_labels.extend(l)
#
#     return np.array(all_data), np.array(all_labels)
#
#
# # def process_target_domain(root_path):
# #     target_dict = {}
# #     target_path = os.path.join(root_path, 'target_domain_datasets')
# #     print(f"正在扫描: {target_path} ...")
# #
# #     if os.path.exists(target_path):
# #         for file in sorted(os.listdir(target_path)):
# #             if file.endswith('.mat'):
# #                 # 提取文件名作为ID (如 'A')
# #                 file_id = os.path.splitext(file)[0]
# #                 sig = load_mat_data(os.path.join(target_path, file), key_filter='time')  # 目标域可能没有DE_time，通常找time
# #                 if sig is not None:
# #                     # 目标域不做重叠切片，或者根据需求做
# #                     d, _ = slice_data(sig, -1)  # label -1 表示未知
# #                     target_dict[file_id] = np.array(d)
# #
# #     return target_dict
#
# # def process_target_domain(root_path):
# #     target_dict = {}
# #     target_path = os.path.join(root_path, 'target_domain_datasets')
# #     print(f"正在扫描: {target_path} ...")
# #
# #     if os.path.exists(target_path):
# #         for file in sorted(os.listdir(target_path)):
# #             if file.endswith('.mat'):
# #                 file_id = os.path.splitext(file)[0]
# #                 sig = load_mat_data(os.path.join(target_path, file), key_filter='time')
# #
# #                 if sig is not None:
# #                     # ============ 【新增核心代码：降采样】 ============
# #                     # 目标是 32kHz -> 12kHz (比例 12/32 = 0.375)
# #                     # 计算新的点数
# #                     new_len = int(len(sig) * 12000 / 32000)
# #                     # 强制重采样
# #                     sig = scipy_signal.resample(sig, new_len)
# #                     # ===============================================
# #
# #                     d, _ = slice_data(sig, -1)
# #                     target_dict[file_id] = np.array(d)
# #
# #     return target_dict
#
# def process_target_domain(root_path):
#     target_dict = {}
#     target_path = os.path.join(root_path, 'target_domain_datasets')
#
#     if os.path.exists(target_path):
#         for file in sorted(os.listdir(target_path)):
#             if file.endswith('.mat'):
#                 file_id = os.path.splitext(file)[0]
#                 sig = load_mat_data(os.path.join(target_path, file), key_filter='time')
#
#                 if sig is not None:
#                     # ============ 【核心修正：物理对齐】 ============
#                     # 原采样率 32000 Hz。
#                     # 我们需要目标采样率为 4000 Hz (因为源域是12k/1800rpm, 目标是600rpm, 12k/3 = 4k)
#                     target_sample_rate = 4000
#
#                     new_len = int(len(sig) * target_sample_rate / 32000)
#                     sig = scipy_signal.resample(sig, new_len)
#                     # ==============================================
#
#                     # 注意：这里继续使用 slice_data (它里面有 FFT 和 Log 处理，保持不变)
#                     # 虽然采样率变了，但我们只关心相对的频谱形状
#                     d, _ = slice_data(sig, -1)
#                     target_dict[file_id] = np.array(d)
#
#     return target_dict
#
# if __name__ == '__main__':
#     # 1. 加载源域
#     print("开始处理源域数据...")
#     source_x, source_y = process_source_domain(ROOT_DIR)
#     print(f"源域处理完成: Shape X={source_x.shape}, Y={source_y.shape}")
#
#     # 2. 加载目标域
#     print("\n开始处理目标域数据...")
#     target_data = process_target_domain(ROOT_DIR)
#     print(f"目标域处理完成: 共 {len(target_data)} 个文件")
#
#     # 3. 保存
#     np.save('source_x.npy', source_x)
#     np.save('source_y.npy', source_y)
#     np.save('target_data.npy', target_data)
#     print("\n所有数据已保存为 .npy 文件!")

import os
import scipy.io
import numpy as np
from scipy import signal as scipy_signal

ROOT_DIR = r"datasets"
SAMPLE_LEN = 1024
STRIDE = 1024

LABEL_MAP = {
    'Normal': 0,
    'IR': 1,
    'OR': 2,
    'B': 3
}

# 【关键】统一物理参数
TARGET_RPM = 600  # 目标域转速
SOURCE_RPM = 1800  # 源域转速
RPM_RATIO = SOURCE_RPM / TARGET_RPM  # 3倍

# 【关键】统一采样率到12kHz
UNIFIED_FS = 12000


def load_mat_data(filepath, key_filter='DE_time'):
    """读取mat文件"""
    try:
        mat = scipy.io.loadmat(filepath)
        for key in mat.keys():
            if key.startswith('__'):
                continue
            if key_filter in key:
                return mat[key].flatten()

        sorted_keys = sorted(mat.keys(),
                             key=lambda k: len(mat[k]) if hasattr(mat[k], '__len__') else 0,
                             reverse=True)
        for key in sorted_keys:
            if not key.startswith('__'):
                return mat[key].flatten()
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def slice_data(signal, label, is_target=False):
    """
    关键改进：
    1. 不做FFT，保留时域信号
    2. 目标域通过重采样对齐转速
    """
    data = []
    labels = []

    # 【新增】如果是目标域，先做时间伸缩对齐转速
    if is_target:
        # 32kHz -> 12kHz 物理对齐
        # 600rpm -> 1800rpm 等效（通过时间压缩3倍）
        new_len = int(len(signal) * 12000 / 32000)
        signal = scipy_signal.resample(signal, new_len)

        # 【关键】压缩信号使其转速等效
        # 通过插值让信号变"快"3倍
        signal_len = len(signal)
        new_signal_len = int(signal_len / RPM_RATIO)
        signal = scipy_signal.resample(signal, new_signal_len)

    n_samples = (len(signal) - SAMPLE_LEN) // STRIDE + 1

    for i in range(n_samples):
        start = i * STRIDE
        segment = signal[start: start + SAMPLE_LEN]

        # 【改进】只做简单归一化，不做FFT
        # 保留时域信息更有利于CNN学习
        segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)

        data.append(segment)
        labels.append(label)

    return data, labels


def process_source_domain(root_path):
    """处理源域数据"""
    all_data = []
    all_labels = []

    # 12kHz DE数据
    de_path = os.path.join(root_path, 'source_domain_datasets', '12kHz_DE_data')
    print(f"正在扫描: {de_path} ...")

    if os.path.exists(de_path):
        for fault_type in os.listdir(de_path):
            type_path = os.path.join(de_path, fault_type)
            if not os.path.isdir(type_path):
                continue

            current_label = LABEL_MAP.get(fault_type, -1)
            if current_label == -1:
                continue

            if fault_type == 'OR':
                for position in os.listdir(type_path):
                    pos_path = os.path.join(type_path, position)
                    for diameter in os.listdir(pos_path):
                        dia_path = os.path.join(pos_path, diameter)
                        for file in os.listdir(dia_path):
                            if file.endswith('.mat'):
                                sig = load_mat_data(os.path.join(dia_path, file))
                                if sig is not None:
                                    d, l = slice_data(sig, current_label, is_target=False)
                                    all_data.extend(d)
                                    all_labels.extend(l)
            else:
                for diameter in os.listdir(type_path):
                    dia_path = os.path.join(type_path, diameter)
                    if not os.path.isdir(dia_path):
                        continue
                    for file in os.listdir(dia_path):
                        if file.endswith('.mat'):
                            sig = load_mat_data(os.path.join(dia_path, file))
                            if sig is not None:
                                d, l = slice_data(sig, current_label, is_target=False)
                                all_data.extend(d)
                                all_labels.extend(l)

    # 48kHz正常数据（降采样到12kHz）
    normal_path = os.path.join(root_path, 'source_domain_datasets', '48kHz_Normal_data')
    print(f"正在扫描: {normal_path} ...")

    if os.path.exists(normal_path):
        for file in os.listdir(normal_path):
            if file.endswith('.mat'):
                sig = load_mat_data(os.path.join(normal_path, file))
                if sig is not None:
                    # 48k -> 12k 降采样
                    sig = sig[::4]
                    d, l = slice_data(sig, LABEL_MAP['Normal'], is_target=False)
                    all_data.extend(d)
                    all_labels.extend(l)

    return np.array(all_data), np.array(all_labels)


def process_target_domain(root_path):
    """处理目标域数据"""
    target_dict = {}
    target_path = os.path.join(root_path, 'target_domain_datasets')

    if os.path.exists(target_path):
        for file in sorted(os.listdir(target_path)):
            if file.endswith('.mat'):
                file_id = os.path.splitext(file)[0]
                sig = load_mat_data(os.path.join(target_path, file), key_filter='time')

                if sig is not None:
                    # 【关键】is_target=True 触发转速对齐
                    d, _ = slice_data(sig, -1, is_target=True)
                    target_dict[file_id] = np.array(d)

    return target_dict


if __name__ == '__main__':
    print("开始处理源域数据...")
    source_x, source_y = process_source_domain(ROOT_DIR)
    print(f"源域处理完成: Shape X={source_x.shape}, Y={source_y.shape}")

    print("\n开始处理目标域数据...")
    target_data = process_target_domain(ROOT_DIR)
    print(f"目标域处理完成: 共 {len(target_data)} 个文件")

    np.save('source_x.npy', source_x)
    np.save('source_y.npy', source_y)
    np.save('target_data.npy', target_data)
    print("\n所有数据已保存为 .npy 文件!")