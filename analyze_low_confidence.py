# analyze_low_confidence.py
# 功能：深入分析低置信度样本的问题原因
# ====================================================================

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
from scipy import signal

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
LABEL_NAMES = ['Normal', 'IR', 'OR', 'Ball']
LABEL_NAMES_CN = ['正常', '内圈故障', '外圈故障', '滚动体故障']


# =========================================
# 模型定义（和训练时一致）
# =========================================
class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.InstanceNorm1d(out_channels, affine=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.InstanceNorm1d(out_channels, affine=True)
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


class DANN_Model_Ultimate(nn.Module):
    def __init__(self, num_classes=4):
        super(DANN_Model_Ultimate, self).__init__()
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
        self.class_classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x, alpha=1.0):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.global_pool(x)
        features = features.view(features.size(0), -1)
        class_output = self.class_classifier(features)
        reverse_features = GradientReverseLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        return class_output, domain_output, features


def analyze_file_details(model, file_data, file_id, mean, std):
    """详细分析单个文件的预测情况"""

    model.eval()
    data = file_data.astype(np.float32)
    data_norm = (data - mean) / (std + 1e-5)

    with torch.no_grad():
        tensor_data = torch.from_numpy(data_norm).to(DEVICE)
        logits, _, features = model(tensor_data, alpha=0)
        probs = F.softmax(logits, dim=1)
        confidence, preds = probs.max(dim=1)

        preds = preds.cpu().numpy()
        confidence = confidence.cpu().numpy()
        probs = probs.cpu().numpy()
        features = features.cpu().numpy()

    # 统计
    vote_counts = Counter(preds)

    result = {
        'file_id': file_id,
        'predictions': preds,
        'confidence': confidence,
        'probs': probs,
        'features': features,
        'vote_counts': vote_counts,
        'raw_data': data
    }

    return result


def main():
    print("=" * 70)
    print("低置信度样本深入分析")
    print("=" * 70)

    # 加载数据
    source_x = np.load("source_x.npy").astype(np.float32)
    mean = source_x.mean()
    std = source_x.std()

    target_dict = np.load("target_data.npy", allow_pickle=True).item()

    # 加载模型
    model = DANN_Model_Ultimate(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load("dann_model_best.pth", map_location=DEVICE))
    model.eval()
    print("✅ 模型加载成功\n")

    # 重点分析的文件
    problem_files = ['D', 'E', 'F']

    # =========================================
    # 1. 详细分析每个问题文件
    # =========================================
    print("=" * 70)
    print("1. 问题文件详细预测分布")
    print("=" * 70)

    file_results = {}

    for file_id in problem_files:
        result = analyze_file_details(model, target_dict[file_id], file_id, mean, std)
        file_results[file_id] = result

        print(f"\n【文件 {file_id}】")
        print("-" * 50)

        # 各类别预测数量
        print("各类别预测样本数：")
        total = len(result['predictions'])
        for i in range(NUM_CLASSES):
            count = result['vote_counts'].get(i, 0)
            pct = count / total * 100
            print(f"  {LABEL_NAMES_CN[i]:>8s}: {count:4d} ({pct:5.1f}%)")

        # 置信度分布
        print(f"\n置信度统计：")
        print(f"  平均: {result['confidence'].mean():.4f}")
        print(f"  最小: {result['confidence'].min():.4f}")
        print(f"  最大: {result['confidence'].max():.4f}")
        print(f"  标准差: {result['confidence'].std():.4f}")

        # 高置信度 vs 低置信度样本的预测差异
        high_conf_mask = result['confidence'] >= 0.8
        low_conf_mask = result['confidence'] < 0.6

        high_conf_preds = result['predictions'][high_conf_mask]
        low_conf_preds = result['predictions'][low_conf_mask]

        print(f"\n高置信度(>=0.8)样本预测分布:")
        if len(high_conf_preds) > 0:
            high_counts = Counter(high_conf_preds)
            for i in range(NUM_CLASSES):
                count = high_counts.get(i, 0)
                print(f"  {LABEL_NAMES_CN[i]:>8s}: {count:4d}")
        else:
            print("  无")

        print(f"\n低置信度(<0.6)样本预测分布:")
        if len(low_conf_preds) > 0:
            low_counts = Counter(low_conf_preds)
            for i in range(NUM_CLASSES):
                count = low_counts.get(i, 0)
                print(f"  {LABEL_NAMES_CN[i]:>8s}: {count:4d}")
        else:
            print("  无")

        # 平均概率分布
        avg_probs = result['probs'].mean(axis=0)
        print(f"\n平均概率分布：")
        for i in range(NUM_CLASSES):
            print(f"  {LABEL_NAMES_CN[i]:>8s}: {avg_probs[i]:.4f}")

    # =========================================
    # 2. 可视化分析
    # =========================================
    print("\n" + "=" * 70)
    print("2. 生成可视化分析图")
    print("=" * 70)

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    for row, file_id in enumerate(problem_files):
        result = file_results[file_id]

        # 列1：置信度分布直方图
        ax = axes[row, 0]
        ax.hist(result['confidence'], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0.7, color='r', linestyle='--', label='阈值0.7')
        ax.set_title(f'文件{file_id}: 置信度分布', fontsize=12)
        ax.set_xlabel('置信度')
        ax.set_ylabel('样本数')
        ax.legend()

        # 列2：各类别概率箱线图
        ax = axes[row, 1]
        box_data = [result['probs'][:, i] for i in range(NUM_CLASSES)]
        bp = ax.boxplot(box_data, labels=LABEL_NAMES_CN)
        ax.set_title(f'文件{file_id}: 各类别概率分布', fontsize=12)
        ax.set_ylabel('概率')

        # 列3：按预测类别分组的置信度
        ax = axes[row, 2]
        for i in range(NUM_CLASSES):
            mask = result['predictions'] == i
            if mask.sum() > 0:
                conf_i = result['confidence'][mask]
                ax.scatter([i] * len(conf_i), conf_i, alpha=0.3, s=10, label=LABEL_NAMES_CN[i])
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_xticklabels(LABEL_NAMES_CN)
        ax.set_title(f'文件{file_id}: 各预测类别的置信度', fontsize=12)
        ax.set_ylabel('置信度')
        ax.axhline(0.7, color='r', linestyle='--', alpha=0.5)

        # 列4：典型样本的时域波形
        ax = axes[row, 3]
        # 选取3个不同类别预测的样本
        colors = ['green', 'orange', 'purple', 'brown']
        for i in range(NUM_CLASSES):
            mask = result['predictions'] == i
            if mask.sum() > 0:
                idx = np.where(mask)[0][0]  # 取第一个
                sample = result['raw_data'][idx]
                ax.plot(sample[:200], alpha=0.7, label=f'预测:{LABEL_NAMES_CN[i]}', color=colors[i])
        ax.set_title(f'文件{file_id}: 不同预测的典型波形', fontsize=12)
        ax.set_xlabel('采样点')
        ax.set_ylabel('幅值')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('low_confidence_analysis.png', dpi=200)
    print("✅ 已保存: low_confidence_analysis.png")
    plt.close()

    # =========================================
    # 3. 对比高置信度文件和低置信度文件
    # =========================================
    print("\n" + "=" * 70)
    print("3. 高置信度 vs 低置信度文件对比")
    print("=" * 70)

    high_conf_files = ['G', 'H', 'K', 'M', 'O']  # 投票比例>98%

    print("\n【高置信度文件特征】")
    for file_id in high_conf_files:
        result = analyze_file_details(model, target_dict[file_id], file_id, mean, std)
        dominant_class = result['vote_counts'].most_common(1)[0]
        vote_ratio = dominant_class[1] / len(result['predictions']) * 100
        avg_conf = result['confidence'].mean()

        # 信号统计特征
        data = target_dict[file_id]
        data_std = data.std(axis=1).mean()
        data_max = np.abs(data).max(axis=1).mean()

        print(f"  {file_id}: 预测={LABEL_NAMES_CN[dominant_class[0]]}, "
              f"投票={vote_ratio:.1f}%, 置信度={avg_conf:.4f}, "
              f"信号std={data_std:.4f}, max={data_max:.4f}")

    print("\n【低置信度文件特征】")
    for file_id in problem_files:
        result = file_results[file_id]
        dominant_class = result['vote_counts'].most_common(1)[0]
        vote_ratio = dominant_class[1] / len(result['predictions']) * 100
        avg_conf = result['confidence'].mean()

        # 信号统计特征
        data = target_dict[file_id]
        data_std = data.std(axis=1).mean()
        data_max = np.abs(data).max(axis=1).mean()

        print(f"  {file_id}: 预测={LABEL_NAMES_CN[dominant_class[0]]}, "
              f"投票={vote_ratio:.1f}%, 置信度={avg_conf:.4f}, "
              f"信号std={data_std:.4f}, max={data_max:.4f}")

    # =========================================
    # 4. 频谱分析
    # =========================================
    print("\n" + "=" * 70)
    print("4. 问题文件频谱分析")
    print("=" * 70)

    fs = 12000  # 采样率

    fig, axes = plt.subplots(len(problem_files), 2, figsize=(16, 4 * len(problem_files)))

    for row, file_id in enumerate(problem_files):
        result = file_results[file_id]

        # 选择高置信度和低置信度样本各一个
        high_conf_idx = np.argmax(result['confidence'])
        low_conf_idx = np.argmin(result['confidence'])

        for col, (idx, title_suffix) in enumerate([
            (high_conf_idx, f"高置信度样本 (conf={result['confidence'][idx]:.4f})"),
            (low_conf_idx, f"低置信度样本 (conf={result['confidence'][idx]:.4f})")
        ]):
            ax = axes[row, col]
            sample = result['raw_data'][idx]

            # 计算频谱
            freqs, psd = signal.welch(sample, fs=fs, nperseg=256)

            ax.semilogy(freqs, psd)
            ax.set_title(f'文件{file_id}: {title_suffix}', fontsize=11)
            ax.set_xlabel('频率 (Hz)')
            ax.set_ylabel('功率谱密度')
            ax.set_xlim(0, 2000)
            ax.grid(True, alpha=0.3)

            # 标注预测类别
            pred_class = result['predictions'][idx]
            ax.text(0.98, 0.95, f'预测: {LABEL_NAMES_CN[pred_class]}',
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('low_confidence_spectrum.png', dpi=200)
    print("✅ 已保存: low_confidence_spectrum.png")
    plt.close()

    # =========================================
    # 5. 问题诊断与建议
    # =========================================
    print("\n" + "=" * 70)
    print("5. 问题诊断与改进建议")
    print("=" * 70)

    print("""
【问题诊断】

1. 文件D、E、F 的共同特点：
   - 都被预测为OR（外圈故障）
   - 但同时有相当比例的样本被预测为其他类别
   - 这表明这些文件可能存在：
     a) 轻微故障或早期故障（特征不明显）
     b) 混合故障（同时存在多种故障特征）
     c) 与源域数据分布差异较大

2. 具体分析：
   - 文件F投票最分散（Normal=127, IR=68, OR=155, Ball=24）
     → 可能是轻微外圈故障，部分特征接近正常
   - 文件E置信度高但投票分散（IR=176, OR=195）
     → 模型对每个样本都很确定，但不同样本预测不同
     → 可能是内圈+外圈的复合故障
   - 文件D（Normal=131, OR=205）
     → 也可能是轻微外圈故障

【改进方案】

方案1: 置信度加权投票（推荐）
   - 不仅看数量，还要考虑置信度权重
   - 高置信度的预测应该有更大话语权

方案2: 概率平均法
   - 对所有样本的概率分布取平均
   - 选择平均概率最高的类别

方案3: 排除低置信度样本
   - 只统计置信度>0.7的样本
   - 减少噪声样本的干扰

方案4: 二次确认机制
   - 对低置信度文件进行人工复核
   - 或使用其他方法（如频谱分析）辅助判断
""")

    # =========================================
    # 6. 应用改进方案并输出新结果
    # =========================================
    print("\n" + "=" * 70)
    print("6. 应用改进方案对比")
    print("=" * 70)

    print("\n各种投票策略的结果对比：")
    print("-" * 70)
    print(f"{'文件':<6} {'原始投票':<12} {'加权投票':<12} {'概率平均':<12} {'过滤投票':<12}")
    print("-" * 70)

    improved_results = []

    for file_id in sorted(target_dict.keys()):
        result = analyze_file_details(model, target_dict[file_id], file_id, mean, std)

        # 原始投票
        orig_pred = result['vote_counts'].most_common(1)[0][0]
        orig_ratio = result['vote_counts'][orig_pred] / len(result['predictions']) * 100

        # 方案1：置信度加权投票
        weighted_votes = np.zeros(NUM_CLASSES)
        for i, (pred, conf) in enumerate(zip(result['predictions'], result['confidence'])):
            weighted_votes[pred] += conf
        weighted_pred = np.argmax(weighted_votes)
        weighted_conf = weighted_votes[weighted_pred] / weighted_votes.sum() * 100

        # 方案2：概率平均
        avg_probs = result['probs'].mean(axis=0)
        prob_pred = np.argmax(avg_probs)
        prob_conf = avg_probs[prob_pred] * 100

        # 方案3：过滤低置信度
        high_conf_mask = result['confidence'] >= 0.7
        if high_conf_mask.sum() > 0:
            filtered_preds = result['predictions'][high_conf_mask]
            filtered_counts = Counter(filtered_preds)
            filtered_pred = filtered_counts.most_common(1)[0][0]
            filtered_ratio = filtered_counts[filtered_pred] / len(filtered_preds) * 100
        else:
            filtered_pred = orig_pred
            filtered_ratio = 0

        print(f"{file_id:<6} "
              f"{LABEL_NAMES[orig_pred]:<6}({orig_ratio:4.1f}%)  "
              f"{LABEL_NAMES[weighted_pred]:<6}({weighted_conf:4.1f}%)  "
              f"{LABEL_NAMES[prob_pred]:<6}({prob_conf:4.1f}%)  "
              f"{LABEL_NAMES[filtered_pred]:<6}({filtered_ratio:4.1f}%)")

        improved_results.append({
            'file': file_id,
            'original': LABEL_NAMES[orig_pred],
            'weighted': LABEL_NAMES[weighted_pred],
            'prob_avg': LABEL_NAMES[prob_pred],
            'filtered': LABEL_NAMES[filtered_pred],
            'weighted_conf': weighted_conf,
            'prob_conf': prob_conf
        })

    # 统计改进效果
    print("\n" + "-" * 70)
    print("【改进建议】")

    # 检查D, E, F是否因不同策略而改变预测
    changes = []
    for r in improved_results:
        if r['file'] in ['D', 'E', 'F']:
            methods = [r['original'], r['weighted'], r['prob_avg'], r['filtered']]
            if len(set(methods)) > 1:
                changes.append(r)

    if changes:
        print("部分文件在不同策略下预测结果不一致，建议：")
        for c in changes:
            print(f"  文件{c['file']}: 原始={c['original']}, 加权={c['weighted']}, "
                  f"概率平均={c['prob_avg']}, 过滤={c['filtered']}")
    else:
        print("所有策略的预测结果一致，说明即使置信度较低，预测方向是正确的。")

    print("\n✅ 分析完成!")
    print("\n【最终建议】")
    print("1. 使用'置信度加权投票'或'概率平均'方法作为最终预测")
    print("2. 对D、E、F文件在报告中标注'预测置信度较低，可能为轻微故障'")
    print("3. 这些文件预测为OR（外圈故障）是合理的，只是故障特征不够显著")


if __name__ == "__main__":
    main()
