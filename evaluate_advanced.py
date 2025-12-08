import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model import DANN_Model

# ================= 配置区域 =================
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MODEL_PATH = 'dann_model.pth'  # 确保你已经训练好并有这个文件
MODEL_PATH = 'dann_model_final.pth'  # 确保你已经训练好并有这个文件
FONT_PATH = 'SimHei'  # 用于显示中文，如果报错改成 'Arial'

# 标签映射
LABEL_NAMES = ['正常 (Normal)', '内圈 (Inner)', '外圈 (Outer)', '滚动体 (Ball)']


def load_and_split_data():
    """
    加载源域数据，并按 8:2 切分为 训练集 和 验证集。
    比赛标准：必须在验证集上算指标，不能在训练集上算！
    """
    print("正在加载源域数据...")
    source_x = np.load('source_x.npy').astype(np.float32)
    source_y = np.load('source_y.npy').astype(np.int64)

    # 归一化 (使用整体的均值方差)
    mean = source_x.mean()
    std = source_x.std()
    source_x = (source_x - mean) / (std + 1e-5)

    # 【核心】：切分验证集 (Test Size = 20%)
    # random_state=42 保证每次切分都一样，具有可复现性（比赛很重要）
    X_train, X_val, y_train, y_val = train_test_split(
        source_x, source_y, test_size=0.2, random_state=42, stratify=source_y
    )

    print(f"数据切分完成: 训练集 {len(X_train)}样本, 验证集 {len(X_val)}样本")

    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return val_loader


def evaluate_model():
    # 1. 准备数据
    val_loader = load_and_split_data()

    # 2. 加载模型
    print(f"加载模型权重: {MODEL_PATH}")
    model = DANN_Model(num_classes=4).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("错误：找不到模型文件，请先运行 train.py！")
        return

    model.eval()

    # 3. 预测
    y_true = []
    y_pred = []

    print("正在进行推理评估...")
    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(DEVICE)
            # alpha=0 表示关闭对抗层，只用特征提取+分类
            class_out, _, _ = model(data, alpha=0)
            preds = class_out.max(1)[1].cpu().numpy()

            y_true.extend(label.numpy())
            y_pred.extend(preds)

    # 4. 计算指标
    acc = accuracy_score(y_true, y_pred)
    # macro: 每一类权重相同（关注小样本类），weighted: 按样本数加权
    f1 = f1_score(y_true, y_pred, average='macro')

    print("\n" + "=" * 30)
    print(f"【验证集最终评估报告】")
    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"F1分数 (Macro F1): {f1:.4f}")
    print("=" * 30)
    print("\n详细分类报告:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))

    # 5. 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # 归一化混淆矩阵 (显示百分比而不是具体个数，看起来更直观)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = [FONT_PATH]
    plt.rcParams['axes.unicode_minus'] = False

    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                annot_kws={"size": 14})  # 字体大小

    plt.title('验证集混淆矩阵 (Confusion Matrix)', fontsize=16)
    plt.ylabel('真实标签 (True Label)', fontsize=14)
    plt.xlabel('预测标签 (Predicted Label)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = 'confusion_matrix.png'
    plt.savefig(save_path, dpi=300)
    print(f"混淆矩阵已保存为: {save_path}")
    plt.show()


if __name__ == '__main__':
    evaluate_model()