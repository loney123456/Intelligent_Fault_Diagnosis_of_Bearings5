"""
Data loading utilities for the "高速列车轴承智能故障诊断" problem.

目录假定为：

datasets/
├── source_domain_datasets/
│   ├── 12kHz_DE_data/
│   ├── 12kHz_FE_data/
│   ├── 48kHz_DE_data/
│   └── 48kHz_Normal_data/
├── target_domain_datasets/
│   ├── A.mat ... P.mat
└── data_load.py  (本文件)

本文件提供：
- 扫描源域 / 目标域 .mat 文件元信息
- 从 .mat 中读取振动信号与转速
- 将长信号切分成固定长度的片段
- 构造用于训练 / 迁移的 numpy 数据集

可以通过开关选择使用：
- 12kHz_DE_data
- 12kHz_FE_data
- 48kHz_DE_data
- 48kHz_Normal_data (作为 Normal 类)
"""

import os
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import scipy.io as sio

# 四类工作状态到整数标签的映射
FAULT_TYPE_MAP: Dict[str, int] = {
    "N": 0,   # Normal
    "B": 1,   # Ball fault
    "IR": 2,  # Inner race fault
    "OR": 3,  # Outer race fault
}


def _walk_mat_files(root: str) -> List[str]:
    """
    递归遍历 root 目录下所有 .mat 文件，返回绝对路径列表。
    """
    mat_paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".mat"):
                mat_paths.append(os.path.join(dirpath, name))
    return sorted(mat_paths)


def parse_source_meta(path: str) -> Dict:
    """
    根据路径和文件名解析“源域” .mat 文件的元信息。

    返回字典字段：
        path:       文件绝对路径
        fs:         采样频率 (Hz)，12k 或 48k
        sensor:     'DE' / 'FE' / None
        fault_type: 'B' / 'IR' / 'OR' / 'N' / None
        fault_size: 故障直径（英寸，float）或 None
        load_id:    载荷等级 0-3 或 None
        location:   外圈故障位置 'Centered'/'Opposite'/'Orthogonal'/None
        raw_freq_folder: 如 '12kHz_DE_data'
    """
    norm_path = os.path.abspath(path).replace("\\", "/")
    parts = norm_path.split("/")

    # 找到 source_domain_datasets 的位置
    try:
        idx = parts.index("source_domain_datasets")
    except ValueError as exc:
        raise ValueError(f"Not a source-domain path: {path}") from exc

    freq_folder = parts[idx + 1]  # e.g. '12kHz_DE_data', '48kHz_Normal_data'
    raw_freq_folder = freq_folder

    # 采样频率
    if freq_folder.startswith("12kHz"):
        fs = 12000
    elif freq_folder.startswith("48kHz"):
        fs = 48000
    else:
        fs = None

    # 通道信息（严格意义上 DE/FE 只对 *_DE_data / *_FE_data 有意义）
    if "DE" in freq_folder:
        sensor = "DE"
    elif "FE" in freq_folder:
        sensor = "FE"
    else:
        sensor = None

    filename = parts[-1]

    # 正常数据：48kHz_Normal_data
    if "Normal_data" in freq_folder:
        fault_type = "N"
        fault_size = None
        load_id = None
        location = None
        return dict(
            path=path,
            fs=fs,
            sensor=sensor,
            fault_type=fault_type,
            fault_size=fault_size,
            load_id=load_id,
            location=location,
            raw_freq_folder=raw_freq_folder,
        )

    # 故障数据在 B / IR / OR 子目录下
    fault_folder = parts[idx + 2]  # 'B', 'IR', 'OR'

    if fault_folder in ("B", "IR"):
        fault_type = fault_folder
        diam_folder = parts[idx + 3]  # '0007', '0014', '0021', '0028'
        try:
            fault_size = int(diam_folder) / 1000.0  # 0007 -> 0.007 (inch)
        except ValueError:
            fault_size = None

        # 文件名可能是 B007_0.mat 或 B028_0_(1797rpm).mat / IR014_3.mat 等
        m = re.search(r"(B|IR)(\d{3})_(\d)", filename)
        if m:
            load_id = int(m.group(3))
        else:
            load_id = None

        location = None

    elif fault_folder == "OR":
        fault_type = "OR"
        # 外圈故障位置子目录：Centered / Opposite / Orthogonal
        location = parts[idx + 3]
        diam_folder = parts[idx + 4]  # '0007', '0014', '0021', '0028'
        try:
            fault_size = int(diam_folder) / 1000.0
        except ValueError:
            fault_size = None

        # 文件名形式：OR007@6_0.mat 或 OR007@6_0_(1797rpm).mat 等
        m = re.search(r"OR(\d{3})@(\d+)_([0-3])", filename)
        if m:
            load_id = int(m.group(3))
        else:
            load_id = None
    else:
        # 无法识别的情况（理论上不会出现）
        fault_type = None
        fault_size = None
        load_id = None
        location = None

    return dict(
        path=path,
        fs=fs,
        sensor=sensor,
        fault_type=fault_type,
        fault_size=fault_size,
        load_id=load_id,
        location=location,
        raw_freq_folder=raw_freq_folder,
    )


def scan_source_meta(datasets_root: str) -> List[Dict]:
    """
    扫描 datasets_root/source_domain_datasets 下所有 .mat，返回元信息列表。
    """
    src_root = os.path.join(datasets_root, "source_domain_datasets")
    mat_paths = _walk_mat_files(src_root)
    metas = [parse_source_meta(p) for p in mat_paths]
    return metas


def scan_target_meta(datasets_root: str) -> List[Dict]:
    """
    扫描 datasets_root/target_domain_datasets 下 A.mat ~ P.mat，返回元信息列表。

    返回字典字段：
        path:    文件绝对路径
        file_id: 'A' ~ 'P'
        fs:      采样频率 (题目说明为 32000Hz)
    """
    tgt_root = os.path.join(datasets_root, "target_domain_datasets")
    mat_paths = _walk_mat_files(tgt_root)
    metas: List[Dict] = []
    for p in mat_paths:
        filename = os.path.basename(p)
        file_id = os.path.splitext(filename)[0]
        metas.append(
            dict(
                path=p,
                file_id=file_id,
                fs=32000,  # 题目说明：目标域采样频率 32kHz
            )
        )
    return metas


def _extract_rpm_from_mat(mat_dict: Dict) -> Optional[float]:
    """
    尝试从 .mat 字典中提取 RPM（转速）。
    """
    for k, v in mat_dict.items():
        k_upper = k.upper()
        if "RPM" in k_upper:
            arr = np.asarray(v).squeeze()
            if arr.size > 0:
                return float(arr.flat[0])
    return None


def _extract_signal_from_mat(
    mat_dict: Dict,
    sensor_preference: Tuple[str, ...] = ("DE", "FE", "BA"),
) -> np.ndarray:
    """
    从 .mat 字典中提取一条一维振动信号：

    1. 优先查找 *_DE_time / *_FE_time / *_BA_time（按 sensor_preference 顺序）
    2. 若找不到，退化为：从所有 1D 数组中，选择长度最大的非 RPM 信号。
    """
    keys = list(mat_dict.keys())

    # Step 1: 变量名中带 _DE_time / _FE_time / _BA_time
    for sensor in sensor_preference:
        for k in keys:
            if k.startswith("__"):
                continue
            if f"_{sensor}_time" in k:
                arr = np.asarray(mat_dict[k]).squeeze()
                if arr.ndim == 1 and arr.size > 0:
                    return arr.astype(np.float32)

    # Step 2: 兜底策略——在所有 1D 数组中挑一个最长的（且变量名不含 RPM）
    candidate_key = None
    candidate_len = 0
    for k in keys:
        if k.startswith("__"):
            continue
        if "RPM" in k.upper():
            continue
        arr = np.asarray(mat_dict[k]).squeeze()
        if arr.ndim == 1 and arr.size > candidate_len:
            candidate_len = arr.size
            candidate_key = k

    if candidate_key is None:
        raise RuntimeError("No suitable 1D vibration signal found in .mat file.")

    return np.asarray(mat_dict[candidate_key]).squeeze().astype(np.float32)


def load_signal(
    path: str,
    sensor_preference: Tuple[str, ...] = ("DE", "FE", "BA"),
) -> Tuple[np.ndarray, Optional[float]]:
    """
    读取指定 .mat 文件，返回 (signal_1d, rpm)：

    - 优先从 *_DE_time / *_FE_time / *_BA_time 中按顺序选择；
    - 若不存在，则自动从 1D 数组中选择长度最大的非 RPM 信号；
    - rpm 若无法获得则返回 None。
    """
    mat = sio.loadmat(path)
    rpm = _extract_rpm_from_mat(mat)
    signal = _extract_signal_from_mat(mat, sensor_preference=sensor_preference)
    return signal, rpm


def segment_signal(
    signal: np.ndarray,
    seg_len: int = 2048,
    step: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    将一条长时间序列切分为若干片段。

    参数：
        signal:    一维 numpy 数组
        seg_len:   每个片段长度
        step:      滑动步长，None 表示与 seg_len 相同（无重叠）
        normalize: 是否对每段做零均值 + 单位方差标准化

    返回：
        segments: (num_segments, seg_len) 的二维数组
    """
    if step is None:
        step = seg_len

    signal = np.asarray(signal).astype(np.float32).ravel()
    n = signal.shape[0]
    segments: List[np.ndarray] = []

    for start in range(0, n - seg_len + 1, step):
        seg = signal[start:start + seg_len]
        if normalize:
            seg = seg - float(np.mean(seg))
            std = float(np.std(seg)) + 1e-8
            seg = seg / std
        segments.append(seg.astype(np.float32))

    if not segments:
        # 长度不足一个 seg_len
        return np.empty((0, seg_len), dtype=np.float32)

    return np.stack(segments, axis=0)


def build_source_dataset(
    datasets_root: str,
    seg_len: int = 2048,
    step: Optional[int] = None,
    use_12k_DE: bool = True,
    use_12k_FE: bool = False,
    use_48k_DE: bool = False,
    use_48k_normal: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    构建“源域”分段数据集。

    参数：
        datasets_root:   datasets 目录路径
        seg_len:         每个时间片段长度
        step:            滑动步长
        use_12k_DE:      是否使用 12kHz_DE_data
        use_12k_FE:      是否使用 12kHz_FE_data
        use_48k_DE:      是否使用 48kHz_DE_data
        use_48k_normal:  是否使用 48kHz_Normal_data (Normal 类)

    返回：
        X:          (num_segments, seg_len) 源域时间片段
        y:          (num_segments,)        对应标签（0=N,1=B,2=IR,3=OR）
        metas_seg:  每个片段对应的元信息列表（包含文件路径、载荷、故障尺寸等）
    """
    metas = scan_source_meta(datasets_root)

    allowed_folders = set()
    if use_12k_DE:
        allowed_folders.add("12kHz_DE_data")
    if use_12k_FE:
        allowed_folders.add("12kHz_FE_data")
    if use_48k_DE:
        allowed_folders.add("48kHz_DE_data")
    if use_48k_normal:
        allowed_folders.add("48kHz_Normal_data")

    if not allowed_folders:
        raise ValueError(
            "At least one of use_12k_DE/use_12k_FE/use_48k_DE/use_48k_normal must be True."
        )

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    metas_seg: List[Dict] = []

    for m in metas:
        if m["raw_freq_folder"] not in allowed_folders:
            continue

        fault_type = m["fault_type"]
        if fault_type is None:
            continue
        if fault_type not in FAULT_TYPE_MAP:
            continue

        label = FAULT_TYPE_MAP[fault_type]

        signal, rpm = load_signal(m["path"], sensor_preference=("DE", "FE", "BA"))
        segs = segment_signal(signal, seg_len=seg_len, step=step, normalize=True)
        if segs.shape[0] == 0:
            continue

        X_list.append(segs)
        y_list.append(np.full((segs.shape[0],), label, dtype=np.int64))

        # 给每个 segment 记录一份 meta 信息
        for _ in range(segs.shape[0]):
            meta_seg = dict(m)
            meta_seg["rpm"] = rpm
            metas_seg.append(meta_seg)

    if not X_list:
        raise RuntimeError(
            "No source segments collected. "
            "Please check datasets path or filtering conditions."
        )

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y, metas_seg


def build_target_dataset(
    datasets_root: str,
    seg_len: int = 2048,
    step: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    构建“目标域”分段数据集（无标签）。

    参数：
        datasets_root:  datasets 目录路径
        seg_len:        每个时间片段长度
        step:           滑动步长

    返回：
        X_tgt:    (num_segments, seg_len) 目标域时间片段
        file_ids: 每个片段所属的文件编号（'A' ~ 'P'）
    """
    metas = scan_target_meta(datasets_root)

    X_list: List[np.ndarray] = []
    file_ids: List[str] = []

    for m in metas:
        signal, rpm = load_signal(m["path"], sensor_preference=("DE", "FE", "BA"))
        segs = segment_signal(signal, seg_len=seg_len, step=step, normalize=True)
        if segs.shape[0] == 0:
            continue

        X_list.append(segs)
        file_ids.extend([m["file_id"]] * segs.shape[0])

    if not X_list:
        raise RuntimeError(
            "No target segments collected. "
            "Please check datasets path."
        )

    X = np.concatenate(X_list, axis=0)
    return X, file_ids


# 自检：可选，用于验证数据读取得当
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick self-check for data_load.py")
    parser.add_argument("--datasets_root", type=str, default="datasets",
                        help="Path to the datasets directory.")
    parser.add_argument("--seg_len", type=int, default=2048)
    parser.add_argument("--step", type=int, default=2048)
    parser.add_argument("--use_12k_DE", type=int, default=1)
    parser.add_argument("--use_12k_FE", type=int, default=0)
    parser.add_argument("--use_48k_DE", type=int, default=0)
    parser.add_argument("--use_48k_normal", type=int, default=0)
    args = parser.parse_args()

    print("Checking source-domain dataset ...")
    X_src, y_src, metas_src = build_source_dataset(
        datasets_root=args.datasets_root,
        seg_len=args.seg_len,
        step=args.step,
        use_12k_DE=bool(args.use_12k_DE),
        use_12k_FE=bool(args.use_12k_FE),
        use_48k_DE=bool(args.use_48k_DE),
        use_48k_normal=bool(args.use_48k_normal),
    )
    print(f"Source segments shape: {X_src.shape}, labels shape: {y_src.shape}")

    print("Checking target-domain dataset ...")
    X_tgt, file_ids = build_target_dataset(
        datasets_root=args.datasets_root,
        seg_len=args.seg_len,
        step=args.step,
    )
    print(f"Target segments shape: {X_tgt.shape}, total segments: {len(file_ids)}")
