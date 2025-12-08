# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# from model import DANN_Model  # 导入刚才定义的模型
#
# # ================= 配置 =================
# BATCH_SIZE = 64
# LR = 0.001
# EPOCHS = 50  # 训练轮数
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# def load_datasets():
#     # 读取 numpy 数据
#     print("正在加载数据...")
#     source_x = np.load('source_x.npy').astype(np.float32)
#     source_y = np.load('source_y.npy').astype(np.int64)
#     target_dict = np.load('target_data.npy', allow_pickle=True).item()
#
#     # 简单的归一化 (Z-score normalization)
#     mean = source_x.mean()
#     std = source_x.std()
#     source_x = (source_x - mean) / (std + 1e-5)
#
#     # 将字典形式的目标域数据合并成一个大数组用于训练 (Unlabeled)
#     target_x_list = []
#     for key in target_dict:
#         t_data = target_dict[key].astype(np.float32)
#         # 同样归一化
#         t_data = (t_data - mean) / (std + 1e-5)
#         target_x_list.append(t_data)
#     target_x = np.concatenate(target_x_list, axis=0)
#
#     # 转为 Tensor
#     src_dataset = TensorDataset(torch.from_numpy(source_x), torch.from_numpy(source_y))
#     # 目标域没有标签，随便填个0，训练时不用的
#     tgt_dataset = TensorDataset(torch.from_numpy(target_x), torch.zeros(len(target_x)).long())
#
#     return src_dataset, tgt_dataset, target_dict, mean, std
#
#
# def train():
#     # 1. 准备数据
#     src_dataset, tgt_dataset, raw_target_dict, data_mean, data_std = load_datasets()
#
#     src_loader = DataLoader(src_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#     tgt_loader = DataLoader(tgt_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#
#     # 2. 初始化模型
#     model = DANN_Model(num_classes=4).to(DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=LR)
#     criterion_class = nn.CrossEntropyLoss()
#     criterion_domain = nn.CrossEntropyLoss()
#
#     print(f"开始训练... 设备: {DEVICE}")
#
#     for epoch in range(EPOCHS):
#         model.train()
#         len_dataloader = min(len(src_loader), len(tgt_loader))
#         total_loss = 0
#         total_acc = 0
#
#         # 必须同时遍历源域和目标域
#         src_iter = iter(src_loader)
#         tgt_iter = iter(tgt_loader)
#
#         for i in range(len_dataloader):
#             try:
#                 s_data, s_label = next(src_iter)
#                 t_data, _ = next(tgt_iter)
#             except StopIteration:
#                 break
#
#             s_data, s_label = s_data.to(DEVICE), s_label.to(DEVICE)
#             t_data = t_data.to(DEVICE)
#
#             # --- 构建域标签 ---
#             # 源域标签为 0, 目标域标签为 1
#             domain_label_s = torch.zeros(BATCH_SIZE).long().to(DEVICE)
#             domain_label_t = torch.ones(BATCH_SIZE).long().to(DEVICE)
#
#             # --- 前向传播 ---
#             # 动态调整 alpha (训练初期 alpha 小，后期大，让模型先学分类再学对抗)
#             p = float(i + epoch * len_dataloader) / EPOCHS / len_dataloader
#             alpha = 2. / (1. + np.exp(-10 * p)) - 1
#
#             # 源域前向
#             class_out_s, domain_out_s, _ = model(s_data, alpha)
#             # 目标域前向 (不需要分类结果)
#             _, domain_out_t, _ = model(t_data, alpha)
#
#             # --- 计算 Loss ---
#             # 1. 分类 Loss (只在源域计算)
#             err_s_label = criterion_class(class_out_s, s_label)
#             # 2. 域 Loss (源域 + 目标域)
#             err_s_domain = criterion_domain(domain_out_s, domain_label_s)
#             err_t_domain = criterion_domain(domain_out_t, domain_label_t)
#
#             # 总 Loss
#             loss = err_s_label + (err_s_domain + err_t_domain) * 0.5
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#
#             # 计算源域准确率
#             pred = class_out_s.data.max(1, keepdim=True)[1]
#             total_acc += pred.eq(s_label.data.view_as(pred)).cpu().sum()
#
#         print(
#             f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {total_loss / len_dataloader:.4f}  Source Acc: {100. * total_acc / (len_dataloader * BATCH_SIZE):.2f}%")
#
#     # 3. 保存模型
#     torch.save(model.state_dict(), 'dann_model.pth')
#     print("模型已保存!")
#
#     # 4. 生成目标域结果 (预测 A-P 的标签)
#     predict_target(model, raw_target_dict, data_mean, data_std)
#
#
# def predict_target(model, target_dict, mean, std):
#     model.eval()
#     results = {}
#     label_names = {0: 'Normal', 1: 'Inner Race', 2: 'Outer Race', 3: 'Ball'}
#
#     print("\n=== 目标域 (列车数据) 诊断结果 ===")
#     with torch.no_grad():
#         for file_id, data in target_dict.items():
#             # 预处理
#             data = (data - mean) / (std + 1e-5)
#             tensor_data = torch.from_numpy(data).float().to(DEVICE)
#
#             # 预测
#             class_out, _, _ = model(tensor_data, alpha=0)
#             preds = class_out.max(1, keepdim=True)[1].cpu().numpy().flatten()
#
#             # 投票机制：因为一个文件切成了很多片，我们看哪种故障投票最多
#             counts = np.bincount(preds)
#             final_pred = np.argmax(counts)
#
#             print(f"文件 {file_id}.mat -> 预测结果: {label_names[final_pred]}")
#             results[file_id] = label_names[final_pred]
#
#
# if __name__ == '__main__':
#     train()


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt  # 引入画图库
from model import DANN_Model

# ================= 配置 =================
BATCH_SIZE = 128
# LR = 0.001
LR = 0.0005
EPOCHS = 1500  # 【修改】增加到100轮
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_datasets():
    # 保持原样，省略...
    print("正在加载数据...")
    source_x = np.load('source_x.npy').astype(np.float32)
    source_y = np.load('source_y.npy').astype(np.int64)
    target_dict = np.load('target_data.npy', allow_pickle=True).item()

    mean = source_x.mean()
    std = source_x.std()
    source_x = (source_x - mean) / (std + 1e-5)

    target_x_list = []
    for key in target_dict:
        t_data = target_dict[key].astype(np.float32)
        t_data = (t_data - mean) / (std + 1e-5)
        target_x_list.append(t_data)
    target_x = np.concatenate(target_x_list, axis=0)

    src_dataset = TensorDataset(torch.from_numpy(source_x), torch.from_numpy(source_y))
    tgt_dataset = TensorDataset(torch.from_numpy(target_x), torch.zeros(len(target_x)).long())

    return src_dataset, tgt_dataset, target_dict, mean, std


def train():
    src_dataset, tgt_dataset, raw_target_dict, data_mean, data_std = load_datasets()

    src_loader = DataLoader(src_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = DANN_Model(num_classes=4).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # 【新增】学习率调整策略，让后期训练更稳定
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    # 【新增】用于记录画图数据
    history = {'loss': [], 'source_acc': []}

    print(f"开始训练... 设备: {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()
        len_dataloader = min(len(src_loader), len(tgt_loader))
        total_loss = 0
        total_acc = 0

        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)

        for i in range(len_dataloader):
            try:
                s_data, s_label = next(src_iter)
                t_data, _ = next(tgt_iter)
            except StopIteration:
                break

            s_data, s_label = s_data.to(DEVICE), s_label.to(DEVICE)
            t_data = t_data.to(DEVICE)

            domain_label_s = torch.zeros(BATCH_SIZE).long().to(DEVICE)
            domain_label_t = torch.ones(BATCH_SIZE).long().to(DEVICE)

            p = float(i + epoch * len_dataloader) / EPOCHS / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            class_out_s, domain_out_s, _ = model(s_data, alpha)
            _, domain_out_t, _ = model(t_data, alpha)

            err_s_label = criterion_class(class_out_s, s_label)
            err_s_domain = criterion_domain(domain_out_s, domain_label_s)
            err_t_domain = criterion_domain(domain_out_t, domain_label_t)

            # loss = err_s_label + (err_s_domain + err_t_domain) * 0.5  # 如果想加强迁移，可以把 0.5 改成 1.0
            loss = err_s_label + (err_s_domain + err_t_domain) * 1.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = class_out_s.data.max(1, keepdim=True)[1]
            total_acc += pred.eq(s_label.data.view_as(pred)).cpu().sum()

        scheduler.step()

        avg_loss = total_loss / len_dataloader
        avg_acc = 100. * total_acc / (len_dataloader * BATCH_SIZE)

        # 记录数据
        history['loss'].append(avg_loss)
        history['source_acc'].append(avg_acc.item())

        print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f}  Source Acc: {avg_acc:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), 'dann_model.pth')

    # 【新增】绘制训练曲线图
    plot_training_curve(history)

    # 预测
    predict_target(model, raw_target_dict, data_mean, data_std)


def plot_training_curve(history):
    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(12, 5))

    # 左图：Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'b-', label='Total Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # 右图：Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['source_acc'], 'r-', label='Source Accuracy')
    plt.title('Source Domain Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("训练曲线图已保存为 training_metrics.png")


def predict_target(model, target_dict, mean, std):
    # 保持原样，省略...
    # (把原来 train.py 里的 predict_target 复制过来)
    model.eval()
    results = {}
    label_names = {0: 'Normal', 1: 'Inner Race', 2: 'Outer Race', 3: 'Ball'}
    print("\n=== 目标域 (列车数据) 诊断结果 ===")
    with torch.no_grad():
        for file_id, data in target_dict.items():
            data = (data - mean) / (std + 1e-5)
            tensor_data = torch.from_numpy(data).float().to(DEVICE)
            class_out, _, _ = model(tensor_data, alpha=0)
            preds = class_out.max(1, keepdim=True)[1].cpu().numpy().flatten()
            counts = np.bincount(preds)
            final_pred = np.argmax(counts)
            print(f"文件 {file_id}.mat -> 预测结果: {label_names[final_pred]}")
            results[file_id] = label_names[final_pred]


if __name__ == '__main__':
    train()