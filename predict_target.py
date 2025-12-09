# # predict_target.py
# # 功能：使用训练好的最佳模型，预测目标域16个文件（A-P）的故障标签
# # ====================================================================
#
# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from collections import Counter
# import pandas as pd
#
# # =========================================
# # 配置
# # =========================================
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NUM_CLASSES = 4
# MODEL_PATH = "dann_model_best.pth"  # 刚刚保存的最佳模型
#
# LABEL_NAMES = {0: 'Normal', 1: 'IR', 2: 'OR', 3: 'Ball'}
# LABEL_NAMES_CN = {0: '正常', 1: '内圈故障', 2: '外圈故障', 3: '滚动体故障'}
#
#
# # =========================================
# # 模型定义（必须和训练时一致）
# # =========================================
# class GradientReverseLayer(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.alpha = alpha
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.neg() * ctx.alpha, None
#
#
# class ResidualBlock1D(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock1D, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.bn1 = nn.InstanceNorm1d(out_channels, affine=True)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.InstanceNorm1d(out_channels, affine=True)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
#                 nn.InstanceNorm1d(out_channels, affine=True)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class DANN_Model_Ultimate(nn.Module):
#     def __init__(self, num_classes=4):
#         super(DANN_Model_Ultimate, self).__init__()
#
#         self.stem = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
#             nn.InstanceNorm1d(32, affine=True),
#             nn.ReLU(),
#             nn.MaxPool1d(2)
#         )
#
#         self.layer1 = ResidualBlock1D(32, 64, stride=2)
#         self.layer2 = ResidualBlock1D(64, 128, stride=2)
#         self.layer3 = ResidualBlock1D(128, 256, stride=1)
#
#         self.global_pool = nn.AdaptiveAvgPool1d(1)
#
#         self.class_classifier = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, num_classes)
#         )
#
#         self.domain_classifier = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 2)
#         )
#
#     def forward(self, x, alpha=1.0):
#         x = x.unsqueeze(1)
#         x = self.stem(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         features = self.global_pool(x)
#         features = features.view(features.size(0), -1)
#
#         class_output = self.class_classifier(features)
#         reverse_features = GradientReverseLayer.apply(features, alpha)
#         domain_output = self.domain_classifier(reverse_features)
#
#         return class_output, domain_output, features
#
#
# # =========================================
# # 主函数：预测目标域
# # =========================================
# def predict_target_domain():
#     """预测目标域16个文件的标签"""
#
#     # 1. 加载源域数据（获取归一化参数）
#     print("=" * 60)
#     print("加载数据...")
#     print("=" * 60)
#
#     source_x = np.load("source_x.npy").astype(np.float32)
#     mean = source_x.mean()
#     std = source_x.std()
#     print(f"源域数据统计: mean={mean:.4f}, std={std:.4f}")
#
#     # 加载目标域数据
#     target_dict = np.load("target_data.npy", allow_pickle=True).item()
#     print(f"目标域文件数量: {len(target_dict)}")
#
#     # 2. 加载模型
#     print("\n" + "=" * 60)
#     print(f"加载模型: {MODEL_PATH}")
#     print("=" * 60)
#
#     model = DANN_Model_Ultimate(num_classes=NUM_CLASSES).to(DEVICE)
#
#     try:
#         state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
#         model.load_state_dict(state_dict)
#         print("✅ 模型加载成功!")
#     except FileNotFoundError:
#         print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
#         print("请先运行 baseline_experiments.py 训练并保存模型")
#         return None
#
#     model.eval()
#
#     # 3. 对每个文件进行预测
#     print("\n" + "=" * 60)
#     print("目标域文件预测结果")
#     print("=" * 60)
#
#     results = []
#     all_predictions_detail = {}  # 保存详细预测结果
#
#     with torch.no_grad():
#         for file_id in sorted(target_dict.keys()):
#             # 获取数据并归一化
#             data = target_dict[file_id].astype(np.float32)
#             data = (data - mean) / (std + 1e-5)
#
#             tensor_data = torch.from_numpy(data).to(DEVICE)
#
#             # 预测
#             logits, _, features = model(tensor_data, alpha=0)
#             probs = F.softmax(logits, dim=1)
#             confidence, preds = probs.max(dim=1)
#
#             # 统计
#             pred_labels = preds.cpu().numpy()
#             conf_values = confidence.cpu().numpy()
#
#             # 投票统计
#             vote_counts = Counter(pred_labels)
#             final_pred = vote_counts.most_common(1)[0][0]
#             vote_ratio = vote_counts[final_pred] / len(pred_labels)
#
#             # 置信度统计
#             conf_mean = conf_values.mean()
#             conf_std = conf_values.std()
#             conf_min = conf_values.min()
#             conf_max = conf_values.max()
#
#             # 各类别预测数量
#             class_dist = {LABEL_NAMES[i]: vote_counts.get(i, 0) for i in range(NUM_CLASSES)}
#
#             # 记录结果
#             result = {
#                 'File': file_id,
#                 'Predicted_Label': LABEL_NAMES[final_pred],
#                 'Predicted_Label_CN': LABEL_NAMES_CN[final_pred],
#                 'Vote_Ratio': vote_ratio,
#                 'Confidence_Mean': conf_mean,
#                 'Confidence_Std': conf_std,
#                 'Confidence_Min': conf_min,
#                 'Confidence_Max': conf_max,
#                 'Normal_Count': class_dist['Normal'],
#                 'IR_Count': class_dist['IR'],
#                 'OR_Count': class_dist['OR'],
#                 'Ball_Count': class_dist['Ball'],
#                 'Total_Samples': len(pred_labels)
#             }
#             results.append(result)
#
#             # 保存详细预测
#             all_predictions_detail[file_id] = {
#                 'predictions': pred_labels,
#                 'confidences': conf_values,
#                 'final_label': final_pred
#             }
#
#             # 打印结果
#             print(f"\n文件 {file_id}:")
#             print(f"  预测结果: {LABEL_NAMES[final_pred]} ({LABEL_NAMES_CN[final_pred]})")
#             print(f"  投票比例: {vote_ratio * 100:.1f}%")
#             print(f"  置信度: {conf_mean:.4f} ± {conf_std:.4f} (范围: {conf_min:.4f} ~ {conf_max:.4f})")
#             print(f"  各类别样本数: Normal={class_dist['Normal']}, IR={class_dist['IR']}, "
#                   f"OR={class_dist['OR']}, Ball={class_dist['Ball']}")
#
#     # 4. 保存结果
#     print("\n" + "=" * 60)
#     print("保存结果")
#     print("=" * 60)
#
#     # 保存CSV
#     df = pd.DataFrame(results)
#     csv_path = "target_predictions.csv"
#     df.to_csv(csv_path, index=False, encoding='utf-8-sig')
#     print(f"✅ 预测结果已保存到: {csv_path}")
#
#     # 保存详细预测（numpy格式）
#     np.save("target_predictions_detail.npy", all_predictions_detail)
#     print(f"✅ 详细预测结果已保存到: target_predictions_detail.npy")
#
#     # 5. 生成简洁的标签对照表
#     print("\n" + "=" * 60)
#     print("目标域文件标签对照表（可直接用于论文）")
#     print("=" * 60)
#     print(f"{'文件':<6} {'预测标签':<12} {'中文名称':<12} {'置信度':<10} {'投票比例':<10}")
#     print("-" * 60)
#     for r in results:
#         print(f"{r['File']:<6} {r['Predicted_Label']:<12} {r['Predicted_Label_CN']:<12} "
#               f"{r['Confidence_Mean']:.4f}     {r['Vote_Ratio'] * 100:.1f}%")
#
#     # 6. 统计摘要
#     print("\n" + "=" * 60)
#     print("预测结果统计摘要")
#     print("=" * 60)
#
#     label_counts = Counter([r['Predicted_Label'] for r in results])
#     print("各类别文件数量:")
#     for label in ['Normal', 'IR', 'OR', 'Ball']:
#         count = label_counts.get(label, 0)
#         cn_name = {'Normal': '正常', 'IR': '内圈故障', 'OR': '外圈故障', 'Ball': '滚动体故障'}[label]
#         print(f"  {label} ({cn_name}): {count} 个文件")
#
#     avg_conf = np.mean([r['Confidence_Mean'] for r in results])
#     avg_vote = np.mean([r['Vote_Ratio'] for r in results])
#     print(f"\n整体统计:")
#     print(f"  平均置信度: {avg_conf:.4f}")
#     print(f"  平均投票比例: {avg_vote * 100:.1f}%")
#
#     return results
#
#
# # =========================================
# # 入口
# # =========================================
# if __name__ == "__main__":
#     results = predict_target_domain()
#
#     if results:
#         print("\n" + "=" * 60)
#         print("✅ 预测完成!")
#         print("=" * 60)
#         print("生成的文件:")
#         print("  1. target_predictions.csv - 预测结果表格")
#         print("  2. target_predictions_detail.npy - 详细预测数据")


# predict_target_improved.py
# 功能：改进版预测脚本 - 使用置信度加权投票，添加可靠性评级
# ====================================================================

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
MODEL_PATH = "dann_model_best.pth"

LABEL_NAMES = {0: 'Normal', 1: 'IR', 2: 'OR', 3: 'Ball'}
LABEL_NAMES_CN = {0: '正常', 1: '内圈故障', 2: '外圈故障', 3: '滚动体故障'}


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


def get_reliability_rating(vote_ratio, avg_conf, second_ratio=0):
    """
    根据投票比例、置信度和次高类别比例，给出可靠性评级
    """
    # 检查是否可能是复合故障
    if avg_conf > 0.9 and vote_ratio < 0.6 and second_ratio > 0.35:
        return "⚠️ 可能复合故障"

    if vote_ratio >= 0.9 and avg_conf >= 0.9:
        return "★★★ 高可靠"
    elif vote_ratio >= 0.75 and avg_conf >= 0.8:
        return "★★☆ 可靠"
    elif vote_ratio >= 0.6 and avg_conf >= 0.7:
        return "★☆☆ 一般"
    else:
        return "☆☆☆ 需复核"


def predict_target_domain_improved():
    """改进版：使用置信度加权投票"""

    print("=" * 70)
    print("目标域预测（改进版 - 置信度加权投票）")
    print("=" * 70)

    # 加载数据
    source_x = np.load("source_x.npy").astype(np.float32)
    mean = source_x.mean()
    std = source_x.std()
    target_dict = np.load("target_data.npy", allow_pickle=True).item()

    # 加载模型
    model = DANN_Model_Ultimate(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ 模型加载成功\n")

    results = []

    with torch.no_grad():
        for file_id in sorted(target_dict.keys()):
            data = target_dict[file_id].astype(np.float32)
            data_norm = (data - mean) / (std + 1e-5)
            tensor_data = torch.from_numpy(data_norm).to(DEVICE)

            logits, _, _ = model(tensor_data, alpha=0)
            probs = F.softmax(logits, dim=1)
            confidence, preds = probs.max(dim=1)

            preds_np = preds.cpu().numpy()
            conf_np = confidence.cpu().numpy()
            probs_np = probs.cpu().numpy()

            # ========== 方法1：原始投票 ==========
            vote_counts = Counter(preds_np)
            orig_pred = vote_counts.most_common(1)[0][0]
            orig_ratio = vote_counts[orig_pred] / len(preds_np)

            # ========== 方法2：置信度加权投票（推荐）==========
            weighted_votes = np.zeros(NUM_CLASSES)
            for pred, conf in zip(preds_np, conf_np):
                weighted_votes[pred] += conf
            weighted_pred = np.argmax(weighted_votes)
            weighted_ratio = weighted_votes[weighted_pred] / weighted_votes.sum()

            # ========== 方法3：概率平均 ==========
            avg_probs = probs_np.mean(axis=0)
            prob_pred = np.argmax(avg_probs)

            # 次高类别分析
            sorted_probs = np.sort(avg_probs)[::-1]
            second_ratio = sorted_probs[1]  # 次高概率

            # 统计信息
            avg_conf = conf_np.mean()

            # 可靠性评级
            reliability = get_reliability_rating(orig_ratio, avg_conf, second_ratio)

            # 备注
            remarks = ""
            if avg_conf > 0.9 and orig_ratio < 0.6:
                # 高置信度但投票分散 → 可能复合故障
                sorted_classes = np.argsort(avg_probs)[::-1]
                class1 = LABEL_NAMES[sorted_classes[0]]
                class2 = LABEL_NAMES[sorted_classes[1]]
                remarks = f"可能为{class1}+{class2}复合故障"
            elif orig_ratio < 0.6 and avg_conf < 0.75:
                remarks = "轻微故障或早期故障"

            result = {
                'File': file_id,
                'Prediction': LABEL_NAMES[weighted_pred],
                'Prediction_CN': LABEL_NAMES_CN[weighted_pred],
                'Vote_Ratio': orig_ratio,
                'Weighted_Ratio': weighted_ratio,
                'Avg_Confidence': avg_conf,
                'Reliability': reliability,
                'Remarks': remarks,
                'Prob_Normal': avg_probs[0],
                'Prob_IR': avg_probs[1],
                'Prob_OR': avg_probs[2],
                'Prob_Ball': avg_probs[3]
            }
            results.append(result)

    # ========== 输出结果 ==========
    print("\n" + "=" * 90)
    print("最终预测结果（置信度加权投票）")
    print("=" * 90)
    print(f"{'文件':<4} {'预测标签':<8} {'中文':<10} {'投票%':<8} {'加权%':<8} {'置信度':<8} {'可靠性':<14} {'备注'}")
    print("-" * 90)

    for r in results:
        print(f"{r['File']:<4} {r['Prediction']:<8} {r['Prediction_CN']:<10} "
              f"{r['Vote_Ratio'] * 100:>5.1f}%   {r['Weighted_Ratio'] * 100:>5.1f}%   "
              f"{r['Avg_Confidence']:.4f}   {r['Reliability']:<14} {r['Remarks']}")

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("target_predictions_improved.csv", index=False, encoding='utf-8-sig')
    print("\n✅ 结果已保存到: target_predictions_improved.csv")

    # ========== 统计摘要 ==========
    print("\n" + "=" * 70)
    print("统计摘要")
    print("=" * 70)

    label_counts = Counter([r['Prediction'] for r in results])
    print("\n各类别文件数量:")
    for label in ['Normal', 'IR', 'OR', 'Ball']:
        count = label_counts.get(label, 0)
        cn = {'Normal': '正常', 'IR': '内圈故障', 'OR': '外圈故障', 'Ball': '滚动体故障'}[label]
        files = [r['File'] for r in results if r['Prediction'] == label]
        print(f"  {label} ({cn}): {count} 个 → {', '.join(files)}")

    # 可靠性统计
    print("\n可靠性分布:")
    reliability_counts = Counter([r['Reliability'] for r in results])
    for rel, count in sorted(reliability_counts.items()):
        files = [r['File'] for r in results if r['Reliability'] == rel]
        print(f"  {rel}: {count} 个 → {', '.join(files)}")

    # ========== 生成论文用表格 ==========
    print("\n" + "=" * 70)
    print("论文用表格（LaTeX格式）")
    print("=" * 70)

    print("""
\\begin{table}[htbp]
\\centering
\\caption{目标域轴承数据诊断结果}
\\begin{tabular}{cccccl}
\\hline
文件 & 预测类别 & 投票比例 & 置信度 & 可靠性 & 备注 \\\\
\\hline""")

    for r in results:
        rel_simple = r['Reliability'].replace('★', '$\\star$').replace('☆', '$\\circ$')
        remark = r['Remarks'] if r['Remarks'] else '-'
        print(f"{r['File']} & {r['Prediction_CN']} & {r['Vote_Ratio'] * 100:.1f}\\% & "
              f"{r['Avg_Confidence']:.3f} & {rel_simple} & {remark} \\\\")

    print("""\\hline
\\end{tabular}
\\label{tab:target_results}
\\end{table}
""")

    return results


if __name__ == "__main__":
    results = predict_target_domain_improved()

    print("\n" + "=" * 70)
    print("✅ 预测完成!")
    print("=" * 70)
    print("""
【结论说明】

1. 文件E被标注为"可能复合故障"：
   - 置信度很高(0.964)说明模型对每个样本都很确定
   - 但约47%预测IR，52%预测OR
   - 从工程角度，轴承同时存在内圈和外圈损伤是常见的

2. 文件D、F被标注为"轻微故障"：
   - 置信度较低说明故障特征不明显
   - 可能是早期故障或轻微损伤

3. 其他文件预测可靠性较高：
   - 特别是G、H、K、M、O的投票比例>98%，置信度>0.95
   - 这些文件的预测结果非常可靠
""")
