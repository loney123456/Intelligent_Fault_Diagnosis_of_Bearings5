# -*- coding: utf-8 -*-
"""
train_new.py

DANN 模型训练脚本 - 极致版

功能：
1. 源域数据训练（带类别权重平衡）
2. 域对抗训练（GRL + 域判别器）
3. MMD 损失对齐特征分布
4. 渐进式伪标签训练
5. Label Smoothing 防止过拟合
6. 余弦退火学习率调度
7. 早停机制

使用方法：
    python train_new.py

输出：
    - dann_model_best.pth: 最佳模型权重
    - dann_model_best_full.pth: 完整模型信息
    - training_metrics.png: 训练曲线图
"""

import os
import random
import copy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 导入模型
from model import DANN_Model_Ultimate

# =========================================
# 配置参数
# =========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据参数
NUM_CLASSES = 4
TEST_SIZE = 0.2
RANDOM_SEED = 42

# 训练参数
BATCH_SIZE = 128
LR_DANN = 1e-3
DANN_EPOCHS = 200

# 域对抗参数
WARMUP_EPOCHS = 60  # 预热阶段（只训练分类器）
DOMAIN_WEIGHT = 0.012  # 域损失权重
MMD_WEIGHT = 0.005  # MMD 损失权重

# 伪标签参数
PSEUDO_START_EPOCH = 80  # 开始伪标签的 epoch
PSEUDO_BASE_THRESHOLD = 0.65  # 基础置信度阈值
PSEUDO_WEIGHT_MAX = 0.30  # 伪标签损失最大权重
PSEUDO_UPDATE_FREQ = 6  # 伪标签更新频率
MAX_PER_CLASS = 700  # 每类最大伪标签数
MIN_PER_CLASS = 60  # 每类最小伪标签数

# 早停参数
PATIENCE = 45

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =========================================
# 工具函数
# =========================================
def set_seed(seed):
    """设置随机种子保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================
# Label Smoothing 交叉熵损失
# =========================================
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing 交叉熵损失
    防止模型过于自信，提升泛化能力
    """

    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_classes = pred.size(-1)

        # 创建 one-hot 标签
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)

        # Label smoothing
        smooth_labels = one_hot * (1 - self.smoothing) + self.smoothing / n_classes

        # 计算损失
        log_probs = F.log_softmax(pred, dim=-1)

        if self.weight is not None:
            weight = self.weight[target]
            loss = -(smooth_labels * log_probs).sum(dim=-1) * weight
        else:
            loss = -(smooth_labels * log_probs).sum(dim=-1)

        return loss.mean()


# =========================================
# MMD 损失（最大均值差异）
# =========================================
def compute_mmd(source_features, target_features, kernel='rbf'):
    """
    计算 MMD 损失，用于对齐源域和目标域特征分布
    使用多核 RBF 核
    """
    batch_size = min(source_features.size(0), target_features.size(0))
    source_features = source_features[:batch_size]
    target_features = target_features[:batch_size]

    def rbf_kernel(x, y, sigma=1.0):
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))

    # 多核 MMD
    sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]
    mmd = 0
    for sigma in sigmas:
        k_ss = rbf_kernel(source_features, source_features, sigma)
        k_tt = rbf_kernel(target_features, target_features, sigma)
        k_st = rbf_kernel(source_features, target_features, sigma)
        mmd += k_ss.mean() + k_tt.mean() - 2 * k_st.mean()

    return mmd / len(sigmas)


# =========================================
# 渐进式伪标签生成
# =========================================
def generate_progressive_pseudo_labels(model, target_x, epoch, max_epoch,
                                       base_threshold=0.65, max_per_class=700,
                                       min_per_class=50):
    """
    增强版伪标签生成
    - 渐进式提高阈值
    - 基于熵的不确定性筛选
    - 类别平衡采样
    """
    model.eval()

    # 渐进式阈值
    progress = min(1.0, (epoch - PSEUDO_START_EPOCH) / (max_epoch - PSEUDO_START_EPOCH))
    current_threshold = base_threshold + 0.20 * progress

    target_tensor = torch.from_numpy(target_x).to(DEVICE)

    batch_size = 256
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(target_tensor), batch_size):
            batch = target_tensor[i:i + batch_size]
            logits, _, _ = model(batch, alpha=0)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    confidence, predictions = all_probs.max(dim=1)

    # 计算熵作为不确定性度量
    entropy = -(all_probs * torch.log(all_probs + 1e-8)).sum(dim=1)
    max_entropy = np.log(NUM_CLASSES)
    normalized_entropy = entropy / max_entropy

    pseudo_data_list = []
    pseudo_labels_list = []
    num_per_class = np.zeros(NUM_CLASSES, dtype=int)

    for cls in range(NUM_CLASSES):
        cls_mask = predictions == cls
        cls_indices = torch.where(cls_mask)[0]

        if len(cls_indices) == 0:
            continue

        cls_confidence = confidence[cls_indices]
        cls_entropy = normalized_entropy[cls_indices]

        # 综合得分：高置信度 + 低熵
        cls_score = cls_confidence - 0.3 * cls_entropy

        # 自适应阈值
        cls_conf_mean = cls_confidence.mean().item()
        adaptive_threshold = max(
            current_threshold - 0.15,
            min(current_threshold, cls_conf_mean - 0.2)
        )

        # 筛选高置信度样本
        high_conf_mask = (cls_confidence > adaptive_threshold) & (cls_entropy < 0.5)
        high_conf_indices = cls_indices[high_conf_mask]

        # 保底机制
        if len(high_conf_indices) < min_per_class and len(cls_indices) >= min_per_class:
            _, top_indices = cls_score.topk(min(min_per_class, len(cls_indices)))
            high_conf_indices = cls_indices[top_indices]
        elif len(high_conf_indices) < min_per_class and len(cls_indices) > 0:
            _, top_indices = cls_score.topk(len(cls_indices))
            high_conf_indices = cls_indices[top_indices]

        # 限制最大数量
        if len(high_conf_indices) > max_per_class:
            cls_conf_selected = confidence[high_conf_indices]
            _, top_indices = cls_conf_selected.topk(max_per_class)
            high_conf_indices = high_conf_indices[top_indices]

        if len(high_conf_indices) > 0:
            pseudo_data_list.append(target_x[high_conf_indices.numpy()])
            pseudo_labels_list.append(np.full(len(high_conf_indices), cls, dtype=np.int64))
            num_per_class[cls] = len(high_conf_indices)

    if len(pseudo_data_list) > 0:
        pseudo_data = np.concatenate(pseudo_data_list, axis=0)
        pseudo_labels = np.concatenate(pseudo_labels_list, axis=0)
    else:
        pseudo_data = np.array([])
        pseudo_labels = np.array([])

    return pseudo_data, pseudo_labels, num_per_class, current_threshold


# =========================================
# DANN 训练函数
# =========================================
def train_dann(X_train, y_train, X_val, y_val, target_x, save_path="dann_model_best.pth"):
    """
    训练 DANN 模型 - 极致版

    Args:
        X_train: 源域训练数据
        y_train: 源域训练标签
        X_val: 源域验证数据
        y_val: 源域验证标签
        target_x: 目标域数据（无标签）
        save_path: 模型保存路径

    Returns:
        best_acc: 最佳验证准确率
        best_f1: 最佳验证 F1 分数
        history: 训练历史
    """
    set_seed(RANDOM_SEED)

    print(f"\n{'=' * 60}")
    print(f"开始训练 DANN 模型（极致版）")
    print(f"{'=' * 60}")
    print(f"设备: {DEVICE}")
    print(f"源域训练样本: {len(X_train)}, 验证样本: {len(X_val)}")
    print(f"目标域样本: {len(target_x)}")

    # 计算类别权重
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    print(f"类别权重: {class_weights.cpu().numpy().round(2)}")

    # 创建数据加载器
    src_train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    src_val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    src_train_loader = DataLoader(src_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    src_val_loader = DataLoader(src_val_ds, batch_size=BATCH_SIZE, shuffle=False)

    tgt_ds = TensorDataset(torch.from_numpy(target_x), torch.zeros(len(target_x)).long())
    tgt_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 创建模型
    model = DANN_Model_Ultimate(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=LR_DANN, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2)

    # 损失函数
    criterion_class = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights)
    criterion_domain = nn.CrossEntropyLoss()
    criterion_pseudo = nn.CrossEntropyLoss()

    # 训练记录
    history = {
        'train_loss': [], 'train_acc': [],
        'val_acc': [], 'val_f1': [],
        'domain_loss': [], 'mmd_loss': [], 'pseudo_loss': []
    }

    best_f1 = 0.0
    best_acc = 0.0
    best_state = None
    no_improve_count = 0

    current_pseudo_data = None
    current_pseudo_labels = None

    for epoch in range(1, DANN_EPOCHS + 1):
        model.train()
        total_cls_loss = 0.0
        total_dom_loss = 0.0
        total_mmd_loss = 0.0
        total_pseudo_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 伪标签更新
        if epoch >= PSEUDO_START_EPOCH and (epoch - PSEUDO_START_EPOCH) % PSEUDO_UPDATE_FREQ == 0:
            pseudo_data, pseudo_labels, num_per_class, curr_thresh = generate_progressive_pseudo_labels(
                model, target_x, epoch, DANN_EPOCHS,
                base_threshold=PSEUDO_BASE_THRESHOLD,
                max_per_class=MAX_PER_CLASS,
                min_per_class=MIN_PER_CLASS
            )

            if len(pseudo_labels) > 0:
                current_pseudo_data = pseudo_data.astype(np.float32)
                current_pseudo_labels = pseudo_labels.astype(np.int64)
                print(f"  [伪标签] Epoch {epoch}: {len(pseudo_labels)} 个样本 (阈值={curr_thresh:.2f})")
                print(
                    f"    分布: N={num_per_class[0]}, IR={num_per_class[1]}, OR={num_per_class[2]}, B={num_per_class[3]}")

        len_dataloader = min(len(src_train_loader), len(tgt_loader))
        src_iter = iter(src_train_loader)
        tgt_iter = iter(tgt_loader)

        for i in range(len_dataloader):
            try:
                s_data, s_label = next(src_iter)
                t_data, _ = next(tgt_iter)
            except StopIteration:
                break

            s_data, s_label = s_data.to(DEVICE), s_label.to(DEVICE)
            t_data = t_data.to(DEVICE)

            bs_src = s_data.size(0)
            bs_tgt = t_data.size(0)

            domain_label_s = torch.zeros(bs_src, dtype=torch.long, device=DEVICE)
            domain_label_t = torch.ones(bs_tgt, dtype=torch.long, device=DEVICE)

            # 计算 alpha（域对抗强度）
            p = float(i + (epoch - 1) * len_dataloader) / (DANN_EPOCHS * len_dataloader)
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

            # 预热阶段
            if epoch <= WARMUP_EPOCHS:
                alpha = 0.0
                domain_weight = 0.0
                mmd_weight = 0.0
            else:
                progress = (epoch - WARMUP_EPOCHS) / (DANN_EPOCHS - WARMUP_EPOCHS)
                domain_weight = DOMAIN_WEIGHT * min(progress * 2, 1.0)
                mmd_weight = MMD_WEIGHT * min(progress * 2, 1.0)

            # 前向传播
            class_out_s, domain_out_s, feat_s = model(s_data, alpha)
            class_out_t, domain_out_t, feat_t = model(t_data, alpha)

            # 分类损失
            err_s_label = criterion_class(class_out_s, s_label)

            # 域损失和 MMD 损失
            if epoch <= WARMUP_EPOCHS:
                loss = err_s_label
                domain_loss_val = 0.0
                mmd_loss_val = 0.0
            else:
                err_s_domain = criterion_domain(domain_out_s, domain_label_s)
                err_t_domain = criterion_domain(domain_out_t, domain_label_t)
                domain_loss_val = (err_s_domain + err_t_domain).item()

                mmd_loss = compute_mmd(feat_s, feat_t)
                mmd_loss_val = mmd_loss.item()

                loss = err_s_label + (err_s_domain + err_t_domain) * domain_weight + mmd_loss * mmd_weight

            # 伪标签损失
            pseudo_loss_val = 0.0
            if epoch >= PSEUDO_START_EPOCH and current_pseudo_data is not None and len(current_pseudo_data) > 0:
                pseudo_batch_size = min(BATCH_SIZE // 2, len(current_pseudo_data))
                pseudo_indices = np.random.choice(len(current_pseudo_data), pseudo_batch_size, replace=False)

                pseudo_batch_x = torch.from_numpy(current_pseudo_data[pseudo_indices]).to(DEVICE)
                pseudo_batch_y = torch.from_numpy(current_pseudo_labels[pseudo_indices]).to(DEVICE)

                pseudo_out, _, _ = model(pseudo_batch_x, alpha=0)
                pseudo_loss = criterion_pseudo(pseudo_out, pseudo_batch_y)

                pseudo_progress = (epoch - PSEUDO_START_EPOCH) / (DANN_EPOCHS - PSEUDO_START_EPOCH)
                current_pseudo_weight = PSEUDO_WEIGHT_MAX * min(pseudo_progress * 1.5, 1.0)

                loss = loss + pseudo_loss * current_pseudo_weight
                pseudo_loss_val = pseudo_loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 统计
            total_cls_loss += err_s_label.item()
            total_dom_loss += domain_loss_val
            total_mmd_loss += mmd_loss_val if epoch > WARMUP_EPOCHS else 0
            total_pseudo_loss += pseudo_loss_val

            preds = class_out_s.argmax(dim=1)
            total_correct += (preds == s_label).sum().item()
            total_samples += bs_src

        scheduler.step()

        # 验证
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in src_val_loader:
                xb = xb.to(DEVICE)
                class_out, _, _ = model(xb, alpha=0.0)
                preds = class_out.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(yb.numpy())

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        # 记录历史
        avg_cls = total_cls_loss / max(len_dataloader, 1)
        train_acc = total_correct / max(total_samples, 1)

        history['train_loss'].append(avg_cls)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(acc)
        history['val_f1'].append(f1)
        history['domain_loss'].append(total_dom_loss / max(len_dataloader, 1))
        history['mmd_loss'].append(total_mmd_loss / max(len_dataloader, 1))
        history['pseudo_loss'].append(total_pseudo_loss / max(len_dataloader, 1))

        # 打印进度
        if epoch % 10 == 0 or epoch == 1:
            avg_dom = total_dom_loss / max(len_dataloader, 1)
            avg_mmd = total_mmd_loss / max(len_dataloader, 1)
            avg_pse = total_pseudo_loss / max(len_dataloader, 1)
            pred_dist = np.bincount(y_pred, minlength=NUM_CLASSES)
            print(
                f"  [Epoch {epoch:03d}/{DANN_EPOCHS}] Cls={avg_cls:.4f} Dom={avg_dom:.4f} MMD={avg_mmd:.4f} Pse={avg_pse:.4f}")
            print(f"    TrainAcc={train_acc:.4f} ValAcc={acc:.4f} ValF1={f1:.4f} | {pred_dist}")

        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 早停
        if no_improve_count >= PATIENCE and epoch > PSEUDO_START_EPOCH + 50:
            print(f"  [早停] Epoch {epoch}")
            break

    # 加载最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # 最终报告
    print(f"\n{'=' * 60}")
    print("最终分类报告")
    print(f"{'=' * 60}")

    model.eval()
    y_true_final, y_pred_final = [], []
    with torch.no_grad():
        for xb, yb in src_val_loader:
            xb = xb.to(DEVICE)
            class_out, _, _ = model(xb, alpha=0.0)
            preds = class_out.argmax(dim=1).cpu().numpy()
            y_pred_final.extend(preds)
            y_true_final.extend(yb.numpy())

    print(classification_report(y_true_final, y_pred_final,
                                target_names=['Normal', 'IR', 'OR', 'Ball'],
                                digits=4, zero_division=0))

    # 保存模型
    torch.save(best_state, save_path)
    print(f"\n✅ 最佳模型已保存到: {save_path}")

    # 保存完整模型信息
    model_info = {
        'state_dict': best_state,
        'best_f1': best_f1,
        'best_acc': best_acc,
        'num_classes': NUM_CLASSES,
        'model_type': 'DANN_Model_Ultimate'
    }
    torch.save(model_info, save_path.replace('.pth', '_full.pth'))
    print(f"✅ 完整模型信息已保存到: {save_path.replace('.pth', '_full.pth')}")

    return best_acc, best_f1, history


# =========================================
# 绘制训练曲线
# =========================================
def plot_training_history(history, save_path="training_metrics.png"):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # 损失曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='分类损失')
    axes[0, 0].plot(epochs, history['domain_loss'], 'r--', label='域损失')
    axes[0, 0].plot(epochs, history['mmd_loss'], 'g--', label='MMD损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='训练准确率')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='验证准确率')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('准确率曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 曲线
    axes[1, 0].plot(epochs, history['val_f1'], 'g-', label='验证 Macro F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 分数曲线')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 伪标签损失
    axes[1, 1].plot(epochs, history['pseudo_loss'], 'm-', label='伪标签损失')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('伪标签损失曲线')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('DANN 训练过程监控', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线已保存到: {save_path}")


# =========================================
# 主函数
# =========================================
def main():
    print("=" * 60)
    print("DANN 模型训练 - 极致版")
    print("=" * 60)

    # 加载数据
    print("\n正在加载数据...")
    X = np.load("source_x.npy").astype(np.float32)
    y = np.load("source_y.npy").astype(np.int64)
    target_x = np.load("target_data.npy", allow_pickle=True).item()

    print(f"源域数据: X = {X.shape}, y = {y.shape}")

    # 合并目标域数据
    target_all = []
    for key in sorted(target_x.keys()):
        target_all.append(target_x[key])
    target_all = np.concatenate(target_all, axis=0).astype(np.float32)
    print(f"目标域数据: {target_all.shape}")

    # 类别统计
    unique, counts = np.unique(y, return_counts=True)
    print("源域类别分布:")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples")

    # 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\n训练集: {len(X_train)}, 验证集: {len(X_val)}")

    # 训练模型
    best_acc, best_f1, history = train_dann(
        X_train, y_train, X_val, y_val, target_all,
        save_path="dann_model_best.pth"
    )

    print(f"\n{'=' * 60}")
    print(f"训练完成！")
    print(f"最佳验证准确率: {best_acc:.4f}")
    print(f"最佳验证 F1 分数: {best_f1:.4f}")
    print(f"{'=' * 60}")

    # 绘制训练曲线
    plot_training_history(history)


if __name__ == "__main__":
    main()