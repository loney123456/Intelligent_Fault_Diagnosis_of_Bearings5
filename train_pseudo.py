import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
from model import DANN_Model

# ================= 配置 =================
BATCH_SIZE = 64
LR = 0.0001  # 也就是微调，学习率要非常小
EPOCHS = 100  # 再训练50轮足够了
CONFIDENCE_THRESHOLD = 0.95  # 只信任非常确定的样本
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_pseudo_labels(model, target_dict, mean, std):
    """
    用现有模型预测目标域，提取高置信度的伪标签数据
    """
    model.eval()
    pseudo_data = []
    pseudo_labels = []

    print(f"正在筛选置信度 > {CONFIDENCE_THRESHOLD} 的目标域样本...")

    with torch.no_grad():
        for file_id, data in target_dict.items():
            # 预处理
            data = (data - mean) / (std + 1e-5)
            tensor_data = torch.from_numpy(data).float().to(DEVICE)

            # 预测
            class_out, _, _ = model(tensor_data, alpha=0)

            # 计算概率 (Softmax)
            probs = torch.softmax(class_out, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            # 筛选高置信度样本
            mask = max_probs > CONFIDENCE_THRESHOLD

            if mask.sum() > 0:
                selected_data = tensor_data[mask].cpu().numpy()
                selected_labels = preds[mask].cpu().numpy()

                pseudo_data.append(selected_data)
                pseudo_labels.append(selected_labels)

    if len(pseudo_data) > 0:
        pseudo_data = np.concatenate(pseudo_data)
        pseudo_labels = np.concatenate(pseudo_labels)
        print(f"成功提取伪标签样本: {len(pseudo_data)} 个")
        return pseudo_data, pseudo_labels
    else:
        print("未提取到高置信度样本，跳过伪标签训练。")
        return None, None


def train_with_pseudo():
    # 1. 加载基础数据
    source_x = np.load('source_x.npy').astype(np.float32)
    source_y = np.load('source_y.npy').astype(np.int64)
    target_dict = np.load('target_data.npy', allow_pickle=True).item()

    # 归一化参数
    mean = source_x.mean()
    std = source_x.std()
    source_x = (source_x - mean) / (std + 1e-5)

    # 2. 加载已经训练好的模型
    print("加载预训练模型...")
    model = DANN_Model(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load('dann_model.pth'))  # 加载上一阶段的模型

    # 3. 生成伪标签
    p_data, p_labels = get_pseudo_labels(model, target_dict, mean, std)

    if p_data is None:
        return

    # 4. 合并数据集 (源域 + 伪标签目标域)
    # 源域数据
    src_dataset = TensorDataset(torch.from_numpy(source_x), torch.from_numpy(source_y))
    # 伪标签数据
    pseudo_dataset = TensorDataset(torch.from_numpy(p_data), torch.from_numpy(p_labels).long())

    # 合并！
    combined_dataset = ConcatDataset([src_dataset, pseudo_dataset])
    train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. 开始微调 (Fine-tuning)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # criterion = nn.CrossEntropyLoss()
    class_weights = torch.FloatTensor([1.0, 1.0, 1.0, 2.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\n开始伪标签微调 (Semi-supervised Fine-tuning)...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        total_acc = 0
        count = 0

        for data, label in train_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)

            # 微调时只训练分类器，不再需要对抗了，因为我们认为这些目标域数据已经是"自己人"了
            output, _, _ = model(data, alpha=0)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.max(1)[1]
            total_acc += pred.eq(label).sum().item()
            count += len(label)

        avg_acc = 100. * total_acc / count
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {total_loss / len(train_loader):.4f}  Combined Acc: {avg_acc:.2f}%")

    # 6. 保存最终模型
    torch.save(model.state_dict(), 'dann_model_final.pth')
    print("最终模型已保存: dann_model_final.pth")


if __name__ == '__main__':
    train_with_pseudo()