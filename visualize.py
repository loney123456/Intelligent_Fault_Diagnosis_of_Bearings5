import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns

# 导入你的模型和数据加载函数
from model import DANN_Model
from train import load_datasets  # 复用train.py里的数据加载逻辑

# ================= 配置 =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MODEL_PATH = 'dann_model.pth'  # 确保你已经运行过 train.py 并有了这个文件
MODEL_PATH = 'dann_model_final.pth'  # 确保你已经运行过 train.py 并有了这个文件
SAMPLES_TO_PLOT = 2000  # 为了图表清晰且运行快，只随机取2000个点来画


def get_features(model, dataloader, domain_label):
    """提取特征向量、标签和域标记"""
    model.eval()
    features_list = []
    labels_list = []
    domains_list = []

    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(DEVICE)
            # alpha=0 因为此时不需要梯度反转，只需要提取特征
            # model 返回: class_output, domain_output, features
            _, _, feat = model(data, alpha=0)

            features_list.append(feat.cpu().numpy())
            labels_list.append(label.numpy())
            # 标记数据来源 (0=源域, 1=目标域)
            domains_list.append(np.full(label.shape, domain_label))

    return np.concatenate(features_list), np.concatenate(labels_list), np.concatenate(domains_list)


def plot_tsne(features, labels, domains):
    print("正在进行 t-SNE 降维计算 (可能需要几分钟)...")
    # t-SNE 将高维特征 (64维) 降维到 2维
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    X_embedded = tsne.fit_transform(features)

    # 设置绘图风格
    sns.set(style='whitegrid')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # --- 图1: 域分布可视化 (Domain Adaptation Effect) ---
    # 目的: 展示源域和目标域是否重合。如果重合得好，说明迁移成功。
    print("正在绘制域分布图...")
    domain_names = ['源域 (实验室)', '目标域 (列车)']
    scatter1 = sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=[domain_names[d] for d in domains],
        style=[domain_names[d] for d in domains],
        palette=['blue', 'red'],
        alpha=0.6, s=40, ax=axes[0]
    )
    axes[0].set_title('特征分布可视化：按数据来源', fontsize=16)
    axes[0].legend(title='Domain')

    # --- 图2: 故障类别可视化 (Class Separability) ---
    # 目的: 展示不同故障是否分得开。
    print("正在绘制类别分布图...")
    label_map = {0: 'Normal', 1: 'Inner Race', 2: 'Outer Race', 3: 'Ball'}
    # 这里的 labels 对于源域是真实标签，对于目标域可以暂且不画或者画预测标签
    # 为了展示清晰，这里我们画出所有的点，看聚类效果
    scatter2 = sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=[label_map[l] for l in labels],
        palette='Set1',
        alpha=0.7, s=40, ax=axes[1]
    )
    axes[1].set_title('特征分布可视化：按故障类别', fontsize=16)
    axes[1].legend(title='Fault Type')

    plt.tight_layout()
    plt.savefig('tsne_result.png', dpi=300)
    print("图片已保存为 tsne_result.png")
    plt.show()


def main():
    # 1. 加载数据
    src_dataset, tgt_dataset, _, _, _ = load_datasets()

    # 随机采样以避免图表过于密集 (可选)
    # 这里我们简单创建 Loader
    src_loader = DataLoader(src_dataset, batch_size=64, shuffle=True)
    tgt_loader = DataLoader(tgt_dataset, batch_size=64, shuffle=True)

    # 2. 加载模型
    print(f"加载模型: {MODEL_PATH}")
    model = DANN_Model(num_classes=4).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("错误: 未找到模型文件。请先运行 python train.py 进行训练！")
        return

    # 3. 提取特征
    print("提取源域特征...")
    src_feats, src_labels, src_domains = get_features(model, src_loader, domain_label=0)

    print("提取目标域特征...")
    # 注意：目标域这里的 label 全是 0 (因为无标签)，所以在"按类别绘图"时，
    # 理想做法是让模型预测一次目标域的 Pseudo Label 填进去，为了代码简单，
    # 这里我们只取前1000个点，并让模型现场预测一下Label用于画图
    tgt_feats, _, tgt_domains = get_features(model, tgt_loader, domain_label=1)

    # 现场预测目标域标签 (为了画第二张图好看)
    model.eval()
    tgt_pred_labels = []
    # 将 numpy 转回 tensor 预测
    tgt_tensor = torch.from_numpy(tgt_feats).to(DEVICE)
    # 分批预测防止显存爆炸
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(tgt_tensor), batch_size):
            batch = tgt_tensor[i:i + batch_size]
            # 这里我们要把 feature 再次送入 class_classifier
            # 但 model.forward 是从头跑的。我们可以手动调用 classifier
            # 修改: 最简单的是重新在 get_features 里预测。
            # 这里为了简单，我们直接用分类器部分
            class_out = model.class_classifier(batch)
            pred = class_out.max(1)[1].cpu().numpy()
            tgt_pred_labels.extend(pred)

    tgt_labels = np.array(tgt_pred_labels)

    # 4. 混合数据并采样 (防止数据量太大跑不动 t-SNE)
    # 简单拼接
    X = np.concatenate((src_feats, tgt_feats))
    y = np.concatenate((src_labels, tgt_labels))
    d = np.concatenate((src_domains, tgt_domains))

    # 随机打乱并取样
    indices = np.random.choice(len(X), min(SAMPLES_TO_PLOT, len(X)), replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    d_sample = d[indices]

    # 5. 绘图
    plot_tsne(X_sample, y_sample, d_sample)


if __name__ == '__main__':
    main()