# # baseline_experiments.py
# # ---------------------------------------------------------
# # 功能：
# #   1）只用源域数据做基线实验（不涉及目标域）
# #   2）基线1：纯 CNN 分类器（结构与 DANN 的特征提取 + 分类头一致，但没有域判别器）
# #   3）基线2：简单统计特征（mean/std/rms/kurtosis/peak_factor） + SVM
# #   4）对每个模型做 5 次随机划分，统计 Accuracy / Macro F1 的 均值 ± 标准差
# # ---------------------------------------------------------
#
# import os
# import random
# import copy
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
#
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
#
# from model import DANN_Model  # 复用你现有的特征提取结构
#
# # =================== 一些超参数 ===================
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BATCH_SIZE = 128
# LR_CNN = 1e-3
# EPOCHS_CNN = 60         # 你机器快的话可以改大一点，比如 80/100
# NUM_RUNS = 5            # 随机划分次数
# TEST_SIZE = 0.2         # 8:2 划分
#
# # -------------------------------------------------
# # 工具函数：固定随机种子，保证可复现
# # -------------------------------------------------
# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#
#
# # -------------------------------------------------
# # 统计特征：mean / std / rms / kurtosis / peak_factor
# # X: (N, L)
# # -------------------------------------------------
# def compute_stat_features(X: np.ndarray) -> np.ndarray:
#     # 均值
#     mean = np.mean(X, axis=1)
#     # 标准差
#     std = np.std(X, axis=1) + 1e-8
#     # 均方根 RMS
#     rms = np.sqrt(np.mean(X ** 2, axis=1)) + 1e-8
#     # 峰度（用标准化 4 阶矩，正态分布约为 3）
#     centered = X - np.mean(X, axis=1, keepdims=True)
#     std_expand = np.std(X, axis=1, keepdims=True) + 1e-8
#     kurt = np.mean((centered / std_expand) ** 4, axis=1)
#     # 峰值因子 peak factor = 峰值 / RMS
#     peak = np.max(np.abs(X), axis=1)
#     peak_factor = peak / rms
#
#     feats = np.stack([mean, std, rms, kurt, peak_factor], axis=1)
#     return feats.astype(np.float32)
#
#
# # -------------------------------------------------
# # 纯 CNN 分类器：复用 DANN_Model 的 feature + class_classifier
# # 但不包含 GRL 和域判别器
# # -------------------------------------------------
# class CNNClassifier(nn.Module):
#     def __init__(self, num_classes: int = 4):
#         super().__init__()
#         base = DANN_Model(num_classes=num_classes)
#         # 直接拿你 DANN 里的特征提取器和分类头
#         self.feature = base.feature
#         self.classifier = base.class_classifier
#
#     def forward(self, x):
#         # x: [batch, 512]
#         x = x.unsqueeze(1)  # -> [batch, 1, 512]
#         feat = self.feature(x)
#         # 和 DANN 一样做全局平均池化
#         feat = nn.functional.adaptive_avg_pool1d(feat, 1)
#         feat = feat.view(feat.size(0), -1)  # [batch, 64]
#         out = self.classifier(feat)         # [batch, num_classes]
#         return out
#
#
# # -------------------------------------------------
# # 单次：训练 + 验证 纯 CNN 基线
# # -------------------------------------------------
# def train_cnn_once(X_train, y_train, X_val, y_val, seed: int):
#     print(f"\n[Run seed={seed}] 训练 CNN 基线模型 ...")
#     set_seed(seed)
#
#     train_dataset = TensorDataset(
#         torch.from_numpy(X_train), torch.from_numpy(y_train)
#     )
#     val_dataset = TensorDataset(
#         torch.from_numpy(X_val), torch.from_numpy(y_val)
#     )
#
#     train_loader = DataLoader(
#         train_dataset, batch_size=BATCH_SIZE, shuffle=True
#     )
#     val_loader = DataLoader(
#         val_dataset, batch_size=BATCH_SIZE, shuffle=False
#     )
#
#     model = CNNClassifier(num_classes=4).to(DEVICE)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR_CNN)
#
#     best_acc = 0.0
#     best_f1 = 0.0
#     best_state = copy.deepcopy(model.state_dict())
#
#     for epoch in range(1, EPOCHS_CNN + 1):
#         model.train()
#         total_loss = 0.0
#
#         for data, label in train_loader:
#             data = data.to(DEVICE)
#             label = label.to(DEVICE)
#
#             logits = model(data)
#             loss = criterion(logits, label)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#
#         # ---- 验证 ----
#         model.eval()
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for data, label in val_loader:
#                 data = data.to(DEVICE)
#                 logits = model(data)
#                 preds = logits.argmax(dim=1).cpu().numpy()
#                 y_true.extend(label.numpy())
#                 y_pred.extend(preds)
#
#         acc = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred, average="macro")
#
#         if acc > best_acc:
#             best_acc = acc
#             best_f1 = f1
#             best_state = copy.deepcopy(model.state_dict())
#
#         if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS_CNN:
#             avg_loss = total_loss / len(train_loader)
#             print(
#                 f"  Epoch [{epoch:03d}/{EPOCHS_CNN}] "
#                 f"Loss={avg_loss:.4f}  ValAcc={acc:.4f}  ValF1={f1:.4f}"
#             )
#
#     # 恢复最好模型（如果你想之后保存，可以在这里另存）
#     model.load_state_dict(best_state)
#     return best_acc, best_f1
#
#
# # -------------------------------------------------
# # 单次：训练 + 验证 统计特征 + SVM 基线
# # -------------------------------------------------
# def train_svm_once(X_train, y_train, X_val, y_val, seed: int):
#     print(f"[Run seed={seed}] 训练 SVM 基线模型 ...")
#     set_seed(seed)
#
#     # 提取统计特征
#     X_train_feat = compute_stat_features(X_train)
#     X_val_feat = compute_stat_features(X_val)
#
#     # 标准化
#     scaler = StandardScaler()
#     X_train_norm = scaler.fit_transform(X_train_feat)
#     X_val_norm = scaler.transform(X_val_feat)
#
#     clf = SVC(kernel="rbf", C=10, gamma="scale")
#     clf.fit(X_train_norm, y_train)
#
#     y_pred = clf.predict(X_val_norm)
#     acc = accuracy_score(y_val, y_pred)
#     f1 = f1_score(y_val, y_pred, average="macro")
#
#     print(
#         f"  SVM ValAcc={acc:.4f}  ValF1={f1:.4f}"
#     )
#     return acc, f1
#
#
# # -------------------------------------------------
# # 主流程：多次随机划分 + 统计均值 ± 标准差
# # -------------------------------------------------
# def main():
#     # 1. 加载 + 归一化源域数据（和你训练 DANN 时保持一致）
#     print("正在加载源域数据 source_x.npy / source_y.npy ...")
#     source_x = np.load("source_x.npy").astype(np.float32)
#     source_y = np.load("source_y.npy").astype(np.int64)
#
#     mean = source_x.mean()
#     std = source_x.std()
#     source_x = (source_x - mean) / (std + 1e-5)
#     print(f"数据形状：X = {source_x.shape}, y = {source_y.shape}")
#
#     seeds = [0, 1, 2, 3, 4]
#
#     cnn_results = []
#     svm_results = []
#
#     for i, seed in enumerate(seeds, start=1):
#         print("\n" + "=" * 60)
#         print(f"  第 {i} 次随机划分 (random_state={seed})")
#         print("=" * 60)
#
#         X_train, X_val, y_train, y_val = train_test_split(
#             source_x,
#             source_y,
#             test_size=TEST_SIZE,
#             stratify=source_y,
#             random_state=seed,
#         )
#
#         print(
#             f"划分完成：训练集 {len(X_train)}，验证集 {len(X_val)}"
#         )
#
#         # ---- 纯 CNN 基线 ----
#         acc_cnn, f1_cnn = train_cnn_once(
#             X_train, y_train, X_val, y_val, seed
#         )
#         cnn_results.append((acc_cnn, f1_cnn))
#
#         # ---- 统计特征 + SVM 基线 ----
#         acc_svm, f1_svm = train_svm_once(
#             X_train, y_train, X_val, y_val, seed
#         )
#         svm_results.append((acc_svm, f1_svm))
#
#     # 2. 汇总统计
#     def summarize(name, results):
#         accs = np.array([x[0] for x in results])
#         f1s = np.array([x[1] for x in results])
#
#         print("\n" + "#" * 60)
#         print(f"{name} 在 {NUM_RUNS} 次随机划分上的结果：")
#         for i, (acc, f1) in enumerate(results, start=1):
#             print(
#                 f"  Run{i}: Acc={acc:.4f}, F1={f1:.4f}"
#             )
#         print("-" * 60)
#         print(
#             f"  Accuracy: mean={accs.mean():.4f}, std={accs.std():.4f}"
#         )
#         print(
#             f"  Macro F1: mean={f1s.mean():.4f}, std={f1s.std():.4f}"
#         )
#         print("#" * 60)
#
#     summarize("CNN 基线模型", cnn_results)
#     summarize("统计特征 + SVM 基线模型", svm_results)
#
#
# if __name__ == "__main__":
#     main()



# baseline_experiments.py
# 说明：
#   1）同一份 source_x/source_y，做 5 次随机划分（random_state = 0~4）
#   2）每次划分在同一套 训练/验证 集上，同时训练：
#        - 统计特征 + SVM
#        - 简单 CNN
#        - DANN（使用 target_data 做无监督域自适应）
#   3）每种方法统计 5 次的 Acc、Macro-F1 的 均值 ± 标准差
#
# 运行方式：
#   python baseline_experiments.py

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
#
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.model_selection import train_test_split
#
# from model import DANN_Model   # 使用你项目里已经写好的 DANN 模型
#
# # y = np.load("source_y.npy")
# # unique, counts = np.unique(y, return_counts=True)
# # print("label 分布：", dict(zip(unique, counts)))
#
# # =========================================
# # 一些通用配置
# # =========================================
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NUM_CLASSES = 4
# CNN_EPOCHS = 60
# DANN_EPOCHS = 60          # 可以后面自己改大一点，例如 100、150
# BATCH_SIZE = 128
# LR_CNN = 1e-3
# LR_DANN = 5e-4            # 和 train.py 里差不多
#
#
# # =========================================
# # 统计特征提取函数（和之前思路一致）
# # =========================================
# def extract_stat_features(X: np.ndarray) -> np.ndarray:
#     """
#     X: shape = [N, L]，每一行是一条时间序列
#     返回: shape = [N, 5] -> [mean, std, rms, kurtosis, peak_factor]
#     """
#     N, L = X.shape
#     feats = np.zeros((N, 5), dtype=np.float32)
#
#     for i in range(N):
#         x = X[i]
#         mean = np.mean(x)
#         std = np.std(x)
#         rms = np.sqrt(np.mean(x ** 2))
#         # 避免除零
#         var = np.var(x) + 1e-12
#         kurtosis = np.mean((x - mean) ** 4) / (var ** 2)
#         peak_factor = np.max(np.abs(x)) / (rms + 1e-12)
#
#         feats[i] = [mean, std, rms, kurtosis, peak_factor]
#
#     return feats
#
#
# # =========================================
# # 简单 CNN（只做源域分类）
# # =========================================
# # class SimpleCNN(nn.Module):
# #     def __init__(self, input_length=512, num_classes=4):
# #         super(SimpleCNN, self).__init__()
# #         self.feature = nn.Sequential(
# #             nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
# #             nn.BatchNorm1d(16),
# #             nn.ReLU(),
# #             nn.MaxPool1d(2),
# #
# #             nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
# #             nn.BatchNorm1d(32),
# #             nn.ReLU(),
# #             nn.MaxPool1d(2),
# #
# #             nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
# #             nn.BatchNorm1d(64),
# #             nn.ReLU(),
# #             nn.AdaptiveAvgPool1d(1)   # -> (B, 64, 1)
# #         )
# #         self.classifier = nn.Sequential(
# #             nn.Flatten(),             # -> (B, 64)
# #             nn.Linear(64, 64),
# #             nn.ReLU(),
# #             nn.Dropout(0.5),
# #             nn.Linear(64, num_classes)
# #         )
# #
# #     def forward(self, x):
# #         # x: (B, L) -> (B, 1, L)
# #         x = x.unsqueeze(1)
# #         x = self.feature(x)
# #         x = self.classifier(x)
# #         return x
#
# class SimpleCNN(nn.Module):
#     def __init__(self, input_dim=9, num_classes=4):
#         super(SimpleCNN, self).__init__()
#         self.feature = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#
#             nn.Linear(128, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU()
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.feature(x)
#         x = self.classifier(x)
#         return x
#
#
# # =========================================
# # 训练一次 CNN（给定一次划分）
# # =========================================
# def train_one_cnn_run(X_train, y_train, X_val, y_val, random_state):
#     torch.manual_seed(random_state)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(random_state)
#
#     train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
#     val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
#
#     input_dim = X_train.shape[1]  # 自动获取特征维度
#     # model = SimpleCNN(input_length=X_train.shape[1], num_classes=NUM_CLASSES).to(DEVICE)
#     model = SimpleCNN(input_dim=X_train.shape[1], num_classes=NUM_CLASSES).to(DEVICE)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LR_CNN)
#
#     best_f1 = 0.0
#     best_acc = 0.0
#     best_state = None
#
#     print(f"\n[Run seed={random_state}] 训练 CNN 基线模型 ...")
#     for epoch in range(1, CNN_EPOCHS + 1):
#         # -------- 训练 --------
#         model.train()
#         total_loss = 0.0
#         for xb, yb in train_loader:
#             xb = xb.to(DEVICE)
#             yb = yb.to(DEVICE)
#
#             optimizer.zero_grad()
#             logits = model(xb)
#             loss = criterion(logits, yb)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         avg_loss = total_loss / len(train_loader)
#
#         # -------- 验证 --------
#         model.eval()
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 xb = xb.to(DEVICE)
#                 logits = model(xb)
#                 preds = logits.argmax(dim=1).cpu().numpy()
#                 y_pred.extend(preds)
#                 y_true.extend(yb.numpy())
#
#         acc = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred, average='macro')
#
#         print(f"  Epoch [{epoch:03d}/{CNN_EPOCHS}] Loss={avg_loss:.4f}  ValAcc={acc:.4f}  ValF1={f1:.4f}")
#
#         if f1 > best_f1:
#             best_f1 = f1
#             best_acc = acc
#             best_state = model.state_dict()
#
#     # 用最佳权重再评估一次（其实 best_acc/best_f1 已经记下来了）
#     if best_state is not None:
#         model.load_state_dict(best_state)
#
#     return best_acc, best_f1
#
#
# # =========================================
# # 训练一次 SVM（统计特征）
# # =========================================
# def train_one_svm_run(feats_train, y_train, feats_val, y_val, random_state):
#     print(f"[Run seed={random_state}] 训练 SVM 基线模型 ...")
#     clf = Pipeline([
#         ("scaler", StandardScaler()),
#         ("svc", SVC(kernel='rbf', C=10.0, gamma='scale'))
#     ])
#
#     clf.fit(feats_train, y_train)
#     y_pred = clf.predict(feats_val)
#
#     acc = accuracy_score(y_val, y_pred)
#     f1 = f1_score(y_val, y_pred, average='macro')
#     print(f"  SVM ValAcc={acc:.4f}  ValF1={f1:.4f}")
#     return acc, f1
#
#
# # =========================================
# # 训练一次 DANN（利用 target_data 做域自适应）
# # =========================================
# # def train_one_dann_run(X_train, y_train, X_val, y_val, target_x, random_state):
# #     torch.manual_seed(random_state)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(random_state)
# #
# #     # 源域训练集 / 验证集
# #     src_train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
# #     src_val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
# #
# #     src_train_loader = DataLoader(src_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
# #     src_val_loader = DataLoader(src_val_ds, batch_size=BATCH_SIZE, shuffle=False)
# #
# #     # 目标域：全部当作无标签数据参与训练
# #     tgt_ds = TensorDataset(torch.from_numpy(target_x), torch.zeros(len(target_x)).long())
# #     tgt_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
# #
# #     model = DANN_Model(num_classes=NUM_CLASSES).to(DEVICE)
# #     optimizer = optim.Adam(model.parameters(), lr=LR_DANN)
# #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DANN_EPOCHS)
# #
# #     criterion_class = nn.CrossEntropyLoss()
# #     criterion_domain = nn.CrossEntropyLoss()
# #
# #     best_f1 = 0.0
# #     best_acc = 0.0
# #     best_state = None
# #
# #     print(f"\n[Run seed={random_state}] 训练 DANN 模型 ...")
# #     for epoch in range(1, DANN_EPOCHS + 1):
# #         # # ----- debug: 看一下输出形状 -----
# #         # model.eval()
# #         # with torch.no_grad():
# #         #     xb, yb = next(iter(src_train_loader))
# #         #     xb = xb.to(DEVICE)
# #         #     co, do, feat = model(xb, alpha=0.0)
# #         #     print("class_out shape:", co.shape)
# #         #     print("domain_out shape:", do.shape)
# #         # model.train()
# #
# #         model.train()
# #         total_loss = 0.0
# #         total_cls_loss = 0.0
# #         total_dom_loss = 0.0
# #         total_correct = 0
# #         total_samples = 0
# #
# #         len_dataloader = min(len(src_train_loader), len(tgt_loader))
# #         src_iter = iter(src_train_loader)
# #         tgt_iter = iter(tgt_loader)
# #
# #         for i in range(len_dataloader):
# #             try:
# #                 s_data, s_label = next(src_iter)
# #                 t_data, _ = next(tgt_iter)
# #             except StopIteration:
# #                 break
# #
# #             s_data, s_label = s_data.to(DEVICE), s_label.to(DEVICE)
# #             t_data = t_data.to(DEVICE)
# #
# #             bs_src = s_data.size(0)
# #             bs_tgt = t_data.size(0)
# #
# #             domain_label_s = torch.zeros(bs_src, dtype=torch.long, device=DEVICE)
# #             domain_label_t = torch.ones(bs_tgt, dtype=torch.long, device=DEVICE)
# #
# #             # 计算 alpha（和 train.py 一致）
# #             p = float(i + (epoch - 1) * len_dataloader) / (DANN_EPOCHS * len_dataloader)
# #             alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
# #
# #             # 【关键修改2】前期几乎不用域对抗
# #             if epoch <= 10:
# #                 alpha = alpha * 0.01  # 前10个epoch几乎不对抗
# #             elif epoch <= 30:
# #                 alpha = alpha * 0.1  # 11-30 epoch 慢慢增加
# #
# #             # 源域前向
# #             class_out_s, domain_out_s, _ = model(s_data, alpha)
# #             # 目标域前向（只算域损失）
# #             _, domain_out_t, _ = model(t_data, alpha)
# #
# #             # # 分类损失（源域）
# #             # err_s_label = criterion_class(class_out_s, s_label)
# #             # # 域损失（源 + 目标）
# #             # err_s_domain = criterion_domain(domain_out_s, domain_label_s)
# #             # err_t_domain = criterion_domain(domain_out_t, domain_label_t)
# #
# #             # 【关键修复】完整的损失函数
# #             err_s_label = criterion_class(class_out_s, s_label)
# #             err_s_domain = criterion_domain(domain_out_s, domain_label_s)
# #             err_t_domain = criterion_domain(domain_out_t, domain_label_t)
# #
# #
# #             # loss = err_s_label + (err_s_domain + err_t_domain) * 1.0
# #             loss = err_s_label + (err_s_domain + err_t_domain) * 0.1
# #             # loss = err_s_label
# #
# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()
# #
# #             total_loss += loss.item()
# #             preds = class_out_s.argmax(dim=1)
# #             total_correct += (preds == s_label).sum().item()
# #             total_samples += bs_src
# #
# #         scheduler.step()
# #         avg_loss = total_loss / max(len_dataloader, 1)
# #         avg_cls_loss = total_cls_loss / max(len_dataloader, 1)
# #         avg_dom_loss = total_dom_loss / max(len_dataloader, 1)
# #         train_acc = total_correct / max(total_samples, 1)
# #
# #
# #         # ---------- 在源域验证集上评估 ----------
# #         model.eval()
# #         y_true, y_pred = [], []
# #         with torch.no_grad():
# #             for xb, yb in src_val_loader:
# #                 xb = xb.to(DEVICE)
# #                 # 验证时 alpha=0，只做分类
# #                 class_out, _, _ = model(xb, alpha=0.0)
# #                 preds = class_out.argmax(dim=1).cpu().numpy()
# #                 y_pred.extend(preds)
# #                 y_true.extend(yb.numpy())
# #
# #         acc = accuracy_score(y_true, y_pred)
# #         f1 = f1_score(y_true, y_pred, average='macro')
# #
# #         pred_dist = np.bincount(y_pred, minlength=NUM_CLASSES)
# #
# #         # print(f"  [DANN] Epoch [{epoch:03d}/{DANN_EPOCHS}] "
# #         #       f"Loss={avg_loss:.4f}  TrainAcc={train_acc:.4f}  ValAcc={acc:.4f}  ValF1={f1:.4f}")
# #         # print("DANN 验证集预测标签分布：", np.bincount(y_pred))
# #
# #         if epoch % 10 == 0 or epoch == 1:
# #             print(f"  [DANN] Epoch [{epoch:03d}/{DANN_EPOCHS}]")
# #             print(f"    TotalLoss={avg_loss:.4f} ClsLoss={avg_cls_loss:.4f} DomLoss={avg_dom_loss:.4f}")
# #             print(f"    TrainAcc={train_acc:.4f} ValAcc={acc:.4f} ValF1={f1:.4f}")
# #             print(f"    ValPredDist: {pred_dist} Alpha={alpha:.4f}")
# #
# #         if f1 > best_f1:
# #             best_f1 = f1
# #             best_acc = acc
# #             best_state = model.state_dict()
# #
# #     if best_state is not None:
# #         model.load_state_dict(best_state)
# #
# #     return best_acc, best_f1
#
# # def train_one_dann_run(X_train, y_train, X_val, y_val, target_x, random_state):
# #     torch.manual_seed(random_state)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(random_state)
# #
# #     # 源域训练集 / 验证集
# #     src_train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
# #     src_val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
# #
# #     src_train_loader = DataLoader(src_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
# #     src_val_loader = DataLoader(src_val_ds, batch_size=BATCH_SIZE, shuffle=False)
# #
# #     # 目标域：全部当作无标签数据参与训练
# #     tgt_ds = TensorDataset(torch.from_numpy(target_x), torch.zeros(len(target_x)).long())
# #     tgt_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
# #
# #     model = DANN_Model(num_classes=NUM_CLASSES).to(DEVICE)
# #     optimizer = optim.Adam(model.parameters(), lr=LR_DANN)
# #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DANN_EPOCHS)
# #
# #     criterion_class = nn.CrossEntropyLoss()
# #     criterion_domain = nn.CrossEntropyLoss()
# #
# #     best_f1 = 0.0
# #     best_acc = 0.0
# #     best_state = None
# #
# #     WARMUP_EPOCHES = 20
# #
# #     print(f"\n[Run seed={random_state}] 训练 DANN 模型 ...")
# #     for epoch in range(1, DANN_EPOCHS + 1):
# #         # # ----- debug: 看一下输出形状 -----
# #         # model.eval()
# #         # with torch.no_grad():
# #         #     xb, yb = next(iter(src_train_loader))
# #         #     xb = xb.to(DEVICE)
# #         #     co, do, feat = model(xb, alpha=0.0)
# #         #     print("class_out shape:", co.shape)
# #         #     print("domain_out shape:", do.shape)
# #         # model.train()
# #
# #         model.train()
# #         total_loss = 0.0
# #         total_cls_loss = 0.0
# #         total_dom_loss = 0.0
# #         total_correct = 0
# #         total_samples = 0
# #
# #         len_dataloader = min(len(src_train_loader), len(tgt_loader))
# #         src_iter = iter(src_train_loader)
# #         tgt_iter = iter(tgt_loader)
# #
# #         for i in range(len_dataloader):
# #             try:
# #                 s_data, s_label = next(src_iter)
# #                 t_data, _ = next(tgt_iter)
# #             except StopIteration:
# #                 break
# #
# #             s_data, s_label = s_data.to(DEVICE), s_label.to(DEVICE)
# #             t_data = t_data.to(DEVICE)
# #
# #             bs_src = s_data.size(0)
# #             bs_tgt = t_data.size(0)
# #
# #             domain_label_s = torch.zeros(bs_src, dtype=torch.long, device=DEVICE)
# #             domain_label_t = torch.ones(bs_tgt, dtype=torch.long, device=DEVICE)
# #
# #             # 计算 alpha（和 train.py 一致）
# #             p = float(i + (epoch - 1) * len_dataloader) / (DANN_EPOCHS * len_dataloader)
# #             alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
# #
# #             # 源域前向
# #             class_out_s, domain_out_s, _ = model(s_data, alpha)
# #             # 目标域前向（只算域损失）
# #             _, domain_out_t, _ = model(t_data, alpha)
# #
# #             err_s_label = criterion_class(class_out_s, s_label)
# #
# #             if epoch <= WARMUP_EPOCHES:
# #                 loss = err_s_label
# #             else:
# #                 err_s_domain = criterion_domain(domain_out_s, domain_label_s)
# #                 err_t_domain = criterion_domain(domain_out_t, domain_label_t)
# #                 loss = err_s_label + (err_s_domain + err_t_domain) * 0.1
# #
# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()
# #
# #             total_loss += loss.item()
# #             preds = class_out_s.argmax(dim=1)
# #             total_correct += (preds == s_label).sum().item()
# #             total_samples += bs_src
# #
# #         scheduler.step()
# #         avg_loss = total_loss / max(len_dataloader, 1)
# #         avg_cls_loss = total_cls_loss / max(len_dataloader, 1)
# #         avg_dom_loss = total_dom_loss / max(len_dataloader, 1)
# #         train_acc = total_correct / max(total_samples, 1)
# #
# #
# #         # ---------- 在源域验证集上评估 ----------
# #         model.eval()
# #         y_true, y_pred = [], []
# #         with torch.no_grad():
# #             for xb, yb in src_val_loader:
# #                 xb = xb.to(DEVICE)
# #                 # 验证时 alpha=0，只做分类
# #                 class_out, _, _ = model(xb, alpha=0.0)
# #                 preds = class_out.argmax(dim=1).cpu().numpy()
# #                 y_pred.extend(preds)
# #                 y_true.extend(yb.numpy())
# #
# #         acc = accuracy_score(y_true, y_pred)
# #         f1 = f1_score(y_true, y_pred, average='macro')
# #
# #         pred_dist = np.bincount(y_pred, minlength=NUM_CLASSES)
# #
# #         # print(f"  [DANN] Epoch [{epoch:03d}/{DANN_EPOCHS}] "
# #         #       f"Loss={avg_loss:.4f}  TrainAcc={train_acc:.4f}  ValAcc={acc:.4f}  ValF1={f1:.4f}")
# #         # print("DANN 验证集预测标签分布：", np.bincount(y_pred))
# #
# #         if epoch % 10 == 0 or epoch == 1:
# #             print(f"  [DANN] Epoch [{epoch:03d}/{DANN_EPOCHS}]")
# #             print(f"    TotalLoss={avg_loss:.4f} ClsLoss={avg_cls_loss:.4f} DomLoss={avg_dom_loss:.4f}")
# #             print(f"    TrainAcc={train_acc:.4f} ValAcc={acc:.4f} ValF1={f1:.4f}")
# #             print(f"    ValPredDist: {pred_dist} Alpha={alpha:.4f}")
# #
# #         if f1 > best_f1:
# #             best_f1 = f1
# #             best_acc = acc
# #             best_state = model.state_dict()
# #
# #     if best_state is not None:
# #         model.load_state_dict(best_state)
# #
# #     return best_acc, best_f1
#
# def train_one_dann_run(X_train, y_train, X_val, y_val, target_x, random_state):
#     torch.manual_seed(random_state)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(random_state)
#
#     src_train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
#     src_val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
#
#     src_train_loader = DataLoader(src_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#     src_val_loader = DataLoader(src_val_ds, batch_size=BATCH_SIZE, shuffle=False)
#
#     tgt_ds = TensorDataset(torch.from_numpy(target_x), torch.zeros(len(target_x)).long())
#     tgt_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#
#     model = DANN_Model(num_classes=NUM_CLASSES).to(DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=LR_DANN)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DANN_EPOCHS)
#
#     criterion_class = nn.CrossEntropyLoss()
#     criterion_domain = nn.CrossEntropyLoss()
#
#     best_f1 = 0.0
#     best_acc = 0.0
#     best_state = None
#
#     # 【新增】前20轮只训练分类器
#     WARMUP_EPOCHS = 20
#
#     print(f"\n[Run seed={random_state}] 训练 DANN 模型 ...")
#     for epoch in range(1, DANN_EPOCHS + 1):
#         model.train()
#         total_loss = 0.0
#         total_cls_loss = 0.0
#         total_dom_loss = 0.0
#         total_correct = 0
#         total_samples = 0
#
#         len_dataloader = min(len(src_train_loader), len(tgt_loader))
#         src_iter = iter(src_train_loader)
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
#             bs_src = s_data.size(0)
#             bs_tgt = t_data.size(0)
#
#             domain_label_s = torch.zeros(bs_src, dtype=torch.long, device=DEVICE)
#             domain_label_t = torch.ones(bs_tgt, dtype=torch.long, device=DEVICE)
#
#             # 计算alpha
#             p = float(i + (epoch - 1) * len_dataloader) / (DANN_EPOCHS * len_dataloader)
#             alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
#
#             # 【关键】前20轮不用域对抗
#             if epoch <= WARMUP_EPOCHS:
#                 alpha = 0.0
#
#             class_out_s, domain_out_s, _ = model(s_data, alpha)
#             _, domain_out_t, _ = model(t_data, alpha)
#
#             err_s_label = criterion_class(class_out_s, s_label)
#
#             # 【关键修复】前20轮只用分类损失
#             if epoch <= WARMUP_EPOCHS:
#                 loss = err_s_label
#                 domain_loss_val = 0.0
#             else:
#                 err_s_domain = criterion_domain(domain_out_s, domain_label_s)
#                 err_t_domain = criterion_domain(domain_out_t, domain_label_t)
#                 domain_loss_val = (err_s_domain + err_t_domain).item()
#                 loss = err_s_label + (err_s_domain + err_t_domain) * 0.1  # 域权重降到0.1
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#             total_cls_loss += err_s_label.item()
#             total_dom_loss += domain_loss_val
#
#             preds = class_out_s.argmax(dim=1)
#             total_correct += (preds == s_label).sum().item()
#             total_samples += bs_src
#
#         scheduler.step()
#         avg_loss = total_loss / max(len_dataloader, 1)
#         avg_cls_loss = total_cls_loss / max(len_dataloader, 1)
#         avg_dom_loss = total_dom_loss / max(len_dataloader, 1)
#         train_acc = total_correct / max(total_samples, 1)
#
#         # 验证
#         model.eval()
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for xb, yb in src_val_loader:
#                 xb = xb.to(DEVICE)
#                 class_out, _, _ = model(xb, alpha=0.0)
#                 preds = class_out.argmax(dim=1).cpu().numpy()
#                 y_pred.extend(preds)
#                 y_true.extend(yb.numpy())
#
#         acc = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred, average='macro')
#
#         pred_dist = np.bincount(y_pred, minlength=NUM_CLASSES)
#
#         if epoch % 10 == 0 or epoch == 1:
#             print(f"  [DANN] Epoch [{epoch:03d}/{DANN_EPOCHS}]")
#             print(f"    ClsLoss={avg_cls_loss:.4f} DomLoss={avg_dom_loss:.4f}")
#             print(f"    TrainAcc={train_acc:.4f} ValAcc={acc:.4f} ValF1={f1:.4f}")
#             print(f"    ValPredDist: {pred_dist}")
#
#         if f1 > best_f1:
#             best_f1 = f1
#             best_acc = acc
#             best_state = model.state_dict()
#
#     if best_state is not None:
#         model.load_state_dict(best_state)
#
#     return best_acc, best_f1
#
# # =========================================
# # 结果汇总打印
# # =========================================
# def summarize_results(name, results):
#     accs = np.array([r[0] for r in results])
#     f1s = np.array([r[1] for r in results])
#
#     print("\n############################################################")
#     print(f"{name} 在 5 次随机划分上的结果：")
#     for i, (acc, f1) in enumerate(results, 1):
#         print(f"  Run{i}: Acc={acc:.4f}, F1={f1:.4f}")
#     print("------------------------------------------------------------")
#     print(f"  Accuracy: mean={accs.mean():.4f}, std={accs.std(ddof=0):.4f}")
#     print(f"  Macro F1: mean={f1s.mean():.4f}, std={f1s.std(ddof=0):.4f}")
#     print("############################################################")
#
#
# # =========================================
# # 主函数
# # =========================================
# def main():
#     print("正在加载源域数据 source_x.npy / source_y.npy ...")
#     X = np.load("source_x.npy").astype(np.float32)
#     y = np.load("source_y.npy").astype(np.int64)
#     print(f"数据形状：X = {X.shape}, y = {y.shape}")
#
#     # 整体归一化（和你其它脚本保持一致）
#     mean = X.mean()
#     std = X.std()
#     X = (X - mean) / (std + 1e-5)
#
#     # 预先计算全体统计特征
#     all_feats = extract_stat_features(X)
#
#     # 载入目标域数据，并用同样均值方差归一化
#     print("正在加载目标域数据 target_data.npy ...")
#     target_dict = np.load("target_data.npy", allow_pickle=True).item()
#     tgt_list = []
#     for k, data in target_dict.items():
#         data = data.astype(np.float32)
#         data = (data - mean) / (std + 1e-5)
#         tgt_list.append(data)
#     target_x = np.concatenate(tgt_list, axis=0)
#     print(f"目标域数据合并后形状：{target_x.shape}")
#
#     seeds = [0, 1, 2, 3, 4]
#     cnn_results = []
#     svm_results = []
#     dann_results = []
#
#     indices = np.arange(len(y))
#
#     for i, seed in enumerate(seeds, 1):
#         print("\n============================================================")
#         print(f"  第 {i} 次随机划分 (random_state={seed})")
#         print("============================================================")
#
#         train_idx, val_idx = train_test_split(
#             indices,
#             test_size=0.2,
#             random_state=seed,
#             stratify=y
#         )
#
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]
#         feats_train, feats_val = all_feats[train_idx], all_feats[val_idx]
#
#         print(f"划分完成：训练集 {len(train_idx)}，验证集 {len(val_idx)}")
#
#         # ---------- CNN ----------
#         acc_cnn, f1_cnn = train_one_cnn_run(X_train, y_train, X_val, y_val, seed)
#         cnn_results.append((acc_cnn, f1_cnn))
#
#         # ---------- SVM ----------
#         acc_svm, f1_svm = train_one_svm_run(feats_train, y_train, feats_val, y_val, seed)
#         svm_results.append((acc_svm, f1_svm))
#
#         # ---------- DANN ----------
#         acc_dann, f1_dann = train_one_dann_run(X_train, y_train, X_val, y_val, target_x, seed)
#         dann_results.append((acc_dann, f1_dann))
#
#     # 最后统一汇总
#     summarize_results("CNN 基线模型", cnn_results)
#     summarize_results("统计特征 + SVM 基线模型", svm_results)
#     summarize_results("DANN 域自适应模型", dann_results)
#
#
# if __name__ == "__main__":
#     main()


# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
#
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.model_selection import train_test_split
#
# # =========================================
# # 通用配置
# # =========================================
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NUM_CLASSES = 4
# CNN_EPOCHS = 60
# DANN_EPOCHS = 80
# BATCH_SIZE = 128
# LR_CNN = 1e-3
# LR_DANN = 5e-4
#
#
# # =========================================
# # 统计特征提取（用于 SVM）
# # =========================================
# def extract_stat_features(X: np.ndarray) -> np.ndarray:
#     """
#     X: shape = [N, L]，每一行是一条时间序列
#     返回: shape = [N, 5] -> [mean, std, rms, kurtosis, peak_factor]
#     """
#     N = X.shape[0]
#     feats = np.zeros((N, 5), dtype=np.float32)
#
#     for i in range(N):
#         x = X[i]
#         mean = np.mean(x)
#         std = np.std(x)
#         rms = np.sqrt(np.mean(x ** 2))
#         var = np.var(x) + 1e-12
#         kurtosis = np.mean((x - mean) ** 4) / (var ** 2)
#         peak_factor = np.max(np.abs(x)) / (rms + 1e-12)
#         feats[i] = [mean, std, rms, kurtosis, peak_factor]
#
#     return feats
#
#
# # =========================================
# # 简单 1D-CNN（用于时间序列分类）
# # =========================================
# class SimpleCNN(nn.Module):
#     def __init__(self, input_length=512, num_classes=4):
#         super(SimpleCNN, self).__init__()
#         self.feature = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#
#             nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#
#             nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1)
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, num_classes)
#         )
#
#     def forward(self, x):
#         # x: (B, L) -> (B, 1, L)
#         x = x.unsqueeze(1)
#         x = self.feature(x)
#         x = self.classifier(x)
#         return x
#
#
# # =========================================
# # 梯度反转层
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
# # =========================================
# # DANN 模型（1D-CNN 版本）
# # =========================================
# class DANN_Model(nn.Module):
#     def __init__(self, num_classes=4):
#         super(DANN_Model, self).__init__()
#
#         self.feature = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#
#             nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#
#             nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1)
#         )
#
#         self.class_classifier = nn.Sequential(
#             nn.Linear(64, 100),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(100, num_classes)
#         )
#
#         self.domain_classifier = nn.Sequential(
#             nn.Linear(64, 100),
#             nn.ReLU(),
#             nn.Linear(100, 2)
#         )
#
#     def forward(self, x, alpha=1.0):
#         # x: (B, L) -> (B, 1, L)
#         x = x.unsqueeze(1)
#         features = self.feature(x)
#         features = features.view(features.size(0), -1)  # (B, 64)
#
#         class_output = self.class_classifier(features)
#
#         reverse_features = GradientReverseLayer.apply(features, alpha)
#         domain_output = self.domain_classifier(reverse_features)
#
#         return class_output, domain_output, features
#
#
# # =========================================
# # 训练函数
# # =========================================
# def train_one_cnn_run(X_train, y_train, X_val, y_val, random_state):
#     torch.manual_seed(random_state)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(random_state)
#
#     train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
#     val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
#
#     model = SimpleCNN(input_length=X_train.shape[1], num_classes=NUM_CLASSES).to(DEVICE)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LR_CNN)
#
#     best_f1 = 0.0
#     best_acc = 0.0
#     best_state = None
#
#     print(f"\n[Run seed={random_state}] 训练 CNN 基线模型 ...")
#     for epoch in range(1, CNN_EPOCHS + 1):
#         model.train()
#         total_loss = 0.0
#         for xb, yb in train_loader:
#             xb = xb.to(DEVICE)
#             yb = yb.to(DEVICE)
#
#             optimizer.zero_grad()
#             logits = model(xb)
#             loss = criterion(logits, yb)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         avg_loss = total_loss / len(train_loader)
#
#         model.eval()
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 xb = xb.to(DEVICE)
#                 logits = model(xb)
#                 preds = logits.argmax(dim=1).cpu().numpy()
#                 y_pred.extend(preds)
#                 y_true.extend(yb.numpy())
#
#         acc = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred, average='macro')
#
#         if epoch % 10 == 0 or epoch == 1:
#             print(f"  Epoch [{epoch:03d}/{CNN_EPOCHS}] Loss={avg_loss:.4f}  ValAcc={acc:.4f}  ValF1={f1:.4f}")
#
#         if f1 > best_f1:
#             best_f1 = f1
#             best_acc = acc
#             best_state = model.state_dict()
#
#     if best_state is not None:
#         model.load_state_dict(best_state)
#
#     return best_acc, best_f1
#
#
# def train_one_svm_run(feats_train, y_train, feats_val, y_val, random_state):
#     print(f"[Run seed={random_state}] 训练 SVM 基线模型 ...")
#     clf = Pipeline([
#         ("scaler", StandardScaler()),
#         ("svc", SVC(kernel='rbf', C=10.0, gamma='scale'))
#     ])
#
#     clf.fit(feats_train, y_train)
#     y_pred = clf.predict(feats_val)
#
#     acc = accuracy_score(y_val, y_pred)
#     f1 = f1_score(y_val, y_pred, average='macro')
#     print(f"  SVM ValAcc={acc:.4f}  ValF1={f1:.4f}")
#     return acc, f1
#
#
# def train_one_dann_run(X_train, y_train, X_val, y_val, target_x, random_state):
#     torch.manual_seed(random_state)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(random_state)
#
#     src_train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
#     src_val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
#
#     src_train_loader = DataLoader(src_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#     src_val_loader = DataLoader(src_val_ds, batch_size=BATCH_SIZE, shuffle=False)
#
#     tgt_ds = TensorDataset(torch.from_numpy(target_x), torch.zeros(len(target_x)).long())
#     tgt_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#
#     model = DANN_Model(num_classes=NUM_CLASSES).to(DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=LR_DANN)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DANN_EPOCHS)
#
#     criterion_class = nn.CrossEntropyLoss()
#     criterion_domain = nn.CrossEntropyLoss()
#
#     best_f1 = 0.0
#     best_acc = 0.0
#     best_state = None
#
#     WARMUP_EPOCHS = 20
#
#     print(f"\n[Run seed={random_state}] 训练 DANN 模型 ...")
#     for epoch in range(1, DANN_EPOCHS + 1):
#         model.train()
#         total_loss = 0.0
#         total_cls_loss = 0.0
#         total_dom_loss = 0.0
#         total_correct = 0
#         total_samples = 0
#
#         len_dataloader = min(len(src_train_loader), len(tgt_loader))
#         src_iter = iter(src_train_loader)
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
#             bs_src = s_data.size(0)
#             bs_tgt = t_data.size(0)
#
#             domain_label_s = torch.zeros(bs_src, dtype=torch.long, device=DEVICE)
#             domain_label_t = torch.ones(bs_tgt, dtype=torch.long, device=DEVICE)
#
#             p = float(i + (epoch - 1) * len_dataloader) / (DANN_EPOCHS * len_dataloader)
#             alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
#
#             if epoch <= WARMUP_EPOCHS:
#                 alpha = 0.0
#
#             class_out_s, domain_out_s, _ = model(s_data, alpha)
#             _, domain_out_t, _ = model(t_data, alpha)
#
#             err_s_label = criterion_class(class_out_s, s_label)
#
#             if epoch <= WARMUP_EPOCHS:
#                 loss = err_s_label
#                 domain_loss_val = 0.0
#             else:
#                 err_s_domain = criterion_domain(domain_out_s, domain_label_s)
#                 err_t_domain = criterion_domain(domain_out_t, domain_label_t)
#                 domain_loss_val = (err_s_domain + err_t_domain).item()
#                 loss = err_s_label + (err_s_domain + err_t_domain) * 0.1
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#             total_cls_loss += err_s_label.item()
#             total_dom_loss += domain_loss_val
#
#             preds = class_out_s.argmax(dim=1)
#             total_correct += (preds == s_label).sum().item()
#             total_samples += bs_src
#
#         scheduler.step()
#         avg_loss = total_loss / max(len_dataloader, 1)
#         avg_cls_loss = total_cls_loss / max(len_dataloader, 1)
#         avg_dom_loss = total_dom_loss / max(len_dataloader, 1)
#         train_acc = total_correct / max(total_samples, 1)
#
#         model.eval()
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for xb, yb in src_val_loader:
#                 xb = xb.to(DEVICE)
#                 class_out, _, _ = model(xb, alpha=0.0)
#                 preds = class_out.argmax(dim=1).cpu().numpy()
#                 y_pred.extend(preds)
#                 y_true.extend(yb.numpy())
#
#         acc = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred, average='macro')
#
#         if epoch % 10 == 0 or epoch == 1:
#             print(f"  [DANN] Epoch [{epoch:03d}/{DANN_EPOCHS}]")
#             print(f"    ClsLoss={avg_cls_loss:.4f} DomLoss={avg_dom_loss:.4f}")
#             print(f"    TrainAcc={train_acc:.4f} ValAcc={acc:.4f} ValF1={f1:.4f}")
#
#         if f1 > best_f1:
#             best_f1 = f1
#             best_acc = acc
#             best_state = model.state_dict()
#
#     if best_state is not None:
#         model.load_state_dict(best_state)
#
#     return best_acc, best_f1
#
#
# def summarize_results(name, results):
#     accs = np.array([r[0] for r in results])
#     f1s = np.array([r[1] for r in results])
#
#     print("\n" + "#" * 60)
#     print(f"{name} 在 5 次随机划分上的结果：")
#     for i, (acc, f1) in enumerate(results, 1):
#         print(f"  Run{i}: Acc={acc:.4f}, F1={f1:.4f}")
#     print("-" * 60)
#     print(f"  Accuracy: mean={accs.mean():.4f}, std={accs.std(ddof=0):.4f}")
#     print(f"  Macro F1: mean={f1s.mean():.4f}, std={f1s.std(ddof=0):.4f}")
#     print("#" * 60)
#
#
# def main():
#     print("正在加载源域数据 source_x.npy / source_y.npy ...")
#     X = np.load("source_x.npy").astype(np.float32)
#     y = np.load("source_y.npy").astype(np.int64)
#     print(f"数据形状：X = {X.shape}, y = {y.shape}")
#
#     # 检查数据维度
#     if X.ndim == 2 and X.shape[1] < 50:
#         print(f"\n[警告] 检测到数据是低维特征 ({X.shape[1]}维)，而不是原始时间序列！")
#         print("[建议] 请重新运行 data_load_final.py 生成原始时间序列数据。")
#         print("[继续] 使用 MLP 代替 CNN 进行实验...\n")
#         USE_MLP = True
#     else:
#         USE_MLP = False
#
#     # 归一化
#     mean = X.mean()
#     std = X.std()
#     X = (X - mean) / (std + 1e-5)
#
#     # 统计特征（用于 SVM）
#     if USE_MLP:
#         all_feats = X  # 已经是特征了
#     else:
#         all_feats = extract_stat_features(X)
#
#     # 加载目标域
#     print("正在加载目标域数据 target_data.npy ...")
#     target_dict = np.load("target_data.npy", allow_pickle=True).item()
#     tgt_list = []
#     for k, data in target_dict.items():
#         data = data.astype(np.float32)
#         data = (data - mean) / (std + 1e-5)
#         tgt_list.append(data)
#     target_x = np.concatenate(tgt_list, axis=0)
#     print(f"目标域数据合并后形状：{target_x.shape}")
#
#     seeds = [0, 1, 2, 3, 4]
#     cnn_results = []
#     svm_results = []
#     dann_results = []
#
#     indices = np.arange(len(y))
#
#     for i, seed in enumerate(seeds, 1):
#         print("\n" + "=" * 60)
#         print(f"  第 {i} 次随机划分 (random_state={seed})")
#         print("=" * 60)
#
#         train_idx, val_idx = train_test_split(
#             indices,
#             test_size=0.2,
#             random_state=seed,
#             stratify=y
#         )
#
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]
#         feats_train, feats_val = all_feats[train_idx], all_feats[val_idx]
#
#         print(f"划分完成：训练集 {len(train_idx)}，验证集 {len(val_idx)}")
#
#         # CNN
#         acc_cnn, f1_cnn = train_one_cnn_run(X_train, y_train, X_val, y_val, seed)
#         cnn_results.append((acc_cnn, f1_cnn))
#
#         # SVM
#         acc_svm, f1_svm = train_one_svm_run(feats_train, y_train, feats_val, y_val, seed)
#         svm_results.append((acc_svm, f1_svm))
#
#         # DANN
#         acc_dann, f1_dann = train_one_dann_run(X_train, y_train, X_val, y_val, target_x, seed)
#         dann_results.append((acc_dann, f1_dann))
#
#     summarize_results("CNN 基线模型", cnn_results)
#     summarize_results("统计特征 + SVM 基线模型", svm_results)
#     summarize_results("DANN 域自适应模型", dann_results)
#
#
# if __name__ == "__main__":
#     main()


# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
#
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.model_selection import train_test_split
#
# # =========================================
# # 通用配置
# # =========================================
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NUM_CLASSES = 4
# CNN_EPOCHS = 60
# DANN_EPOCHS = 100  # 增加到100轮
# BATCH_SIZE = 128
# LR_CNN = 1e-3
# LR_DANN = 3e-4  # 稍微降低学习率
#
#
# # =========================================
# # 统计特征提取（用于 SVM）
# # =========================================
# def extract_stat_features(X: np.ndarray) -> np.ndarray:
#     N = X.shape[0]
#     feats = np.zeros((N, 5), dtype=np.float32)
#
#     for i in range(N):
#         x = X[i]
#         mean = np.mean(x)
#         std = np.std(x)
#         rms = np.sqrt(np.mean(x ** 2))
#         var = np.var(x) + 1e-12
#         kurtosis = np.mean((x - mean) ** 4) / (var ** 2)
#         peak_factor = np.max(np.abs(x)) / (rms + 1e-12)
#         feats[i] = [mean, std, rms, kurtosis, peak_factor]
#
#     return feats
#
#
# # =========================================
# # 简单 1D-CNN
# # =========================================
# class SimpleCNN(nn.Module):
#     def __init__(self, input_length=512, num_classes=4):
#         super(SimpleCNN, self).__init__()
#         self.feature = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#
#             nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#
#             nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1)
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, num_classes)
#         )
#
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.feature(x)
#         x = self.classifier(x)
#         return x
#
#
# # =========================================
# # 梯度反转层
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
# # =========================================
# # 改进版 DANN 模型
# # =========================================
# class DANN_Model_Improved(nn.Module):
#     def __init__(self, num_classes=4):
#         super(DANN_Model_Improved, self).__init__()
#
#         self.feature = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
#             nn.InstanceNorm1d(16, affine=True),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#
#             nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
#             nn.InstanceNorm1d(32, affine=True),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#
#             nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm1d(64, affine=True),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1)
#         )
#
#         self.class_classifier = nn.Sequential(
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, num_classes)
#         )
#
#         self.domain_classifier = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 2)
#         )
#
#     def forward(self, x, alpha=1.0):
#         x = x.unsqueeze(1)
#         features = self.feature(x)
#         features = features.view(features.size(0), -1)
#
#         class_output = self.class_classifier(features)
#
#         reverse_features = GradientReverseLayer.apply(features, alpha)
#         domain_output = self.domain_classifier(reverse_features)
#
#         return class_output, domain_output, features
#
#
# # =========================================
# # CNN 训练函数
# # =========================================
# def train_one_cnn_run(X_train, y_train, X_val, y_val, random_state):
#     torch.manual_seed(random_state)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(random_state)
#
#     train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
#     val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
#
#     model = SimpleCNN(input_length=X_train.shape[1], num_classes=NUM_CLASSES).to(DEVICE)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LR_CNN)
#
#     best_f1 = 0.0
#     best_acc = 0.0
#     best_state = None
#
#     print(f"\n[Run seed={random_state}] 训练 CNN 基线模型 ...")
#     for epoch in range(1, CNN_EPOCHS + 1):
#         model.train()
#         total_loss = 0.0
#         for xb, yb in train_loader:
#             xb = xb.to(DEVICE)
#             yb = yb.to(DEVICE)
#
#             optimizer.zero_grad()
#             logits = model(xb)
#             loss = criterion(logits, yb)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         avg_loss = total_loss / len(train_loader)
#
#         model.eval()
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 xb = xb.to(DEVICE)
#                 logits = model(xb)
#                 preds = logits.argmax(dim=1).cpu().numpy()
#                 y_pred.extend(preds)
#                 y_true.extend(yb.numpy())
#
#         acc = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred, average='macro')
#
#         if epoch % 10 == 0 or epoch == 1:
#             print(f"  Epoch [{epoch:03d}/{CNN_EPOCHS}] Loss={avg_loss:.4f}  ValAcc={acc:.4f}  ValF1={f1:.4f}")
#
#         if f1 > best_f1:
#             best_f1 = f1
#             best_acc = acc
#             best_state = model.state_dict()
#
#     if best_state is not None:
#         model.load_state_dict(best_state)
#
#     return best_acc, best_f1
#
#
# # =========================================
# # SVM 训练函数
# # =========================================
# def train_one_svm_run(feats_train, y_train, feats_val, y_val, random_state):
#     print(f"[Run seed={random_state}] 训练 SVM 基线模型 ...")
#     clf = Pipeline([
#         ("scaler", StandardScaler()),
#         ("svc", SVC(kernel='rbf', C=10.0, gamma='scale'))
#     ])
#
#     clf.fit(feats_train, y_train)
#     y_pred = clf.predict(feats_val)
#
#     acc = accuracy_score(y_val, y_pred)
#     f1 = f1_score(y_val, y_pred, average='macro')
#     print(f"  SVM ValAcc={acc:.4f}  ValF1={f1:.4f}")
#     return acc, f1
#
#
# # =========================================
# # 改进版 DANN 训练函数
# # =========================================
# def train_one_dann_run_improved(X_train, y_train, X_val, y_val, target_x, random_state):
#     torch.manual_seed(random_state)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(random_state)
#
#     src_train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
#     src_val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
#
#     src_train_loader = DataLoader(src_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#     src_val_loader = DataLoader(src_val_ds, batch_size=BATCH_SIZE, shuffle=False)
#
#     tgt_ds = TensorDataset(torch.from_numpy(target_x), torch.zeros(len(target_x)).long())
#     tgt_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#
#     model = DANN_Model_Improved(num_classes=NUM_CLASSES).to(DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=LR_DANN, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DANN_EPOCHS)
#
#     criterion_class = nn.CrossEntropyLoss()
#     criterion_domain = nn.CrossEntropyLoss()
#
#     best_f1 = 0.0
#     best_acc = 0.0
#     best_state = None
#
#     WARMUP_EPOCHS = 40
#     DOMAIN_WEIGHT = 0.02
#     PATIENCE = 20
#     no_improve_count = 0
#
#     print(f"\n[Run seed={random_state}] 训练改进版 DANN 模型 ...")
#
#     for epoch in range(1, DANN_EPOCHS + 1):
#         model.train()
#         total_loss = 0.0
#         total_cls_loss = 0.0
#         total_dom_loss = 0.0
#         total_correct = 0
#         total_samples = 0
#
#         len_dataloader = min(len(src_train_loader), len(tgt_loader))
#         src_iter = iter(src_train_loader)
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
#             bs_src = s_data.size(0)
#             bs_tgt = t_data.size(0)
#
#             domain_label_s = torch.zeros(bs_src, dtype=torch.long, device=DEVICE)
#             domain_label_t = torch.ones(bs_tgt, dtype=torch.long, device=DEVICE)
#
#             p = float(i + (epoch - 1) * len_dataloader) / (DANN_EPOCHS * len_dataloader)
#             alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
#
#             if epoch <= WARMUP_EPOCHS:
#                 alpha = 0.0
#                 domain_weight = 0.0
#             else:
#                 progress = (epoch - WARMUP_EPOCHS) / (DANN_EPOCHS - WARMUP_EPOCHS)
#                 domain_weight = DOMAIN_WEIGHT * min(progress * 2, 1.0)
#
#             class_out_s, domain_out_s, _ = model(s_data, alpha)
#             _, domain_out_t, _ = model(t_data, alpha)
#
#             err_s_label = criterion_class(class_out_s, s_label)
#
#             if epoch <= WARMUP_EPOCHS:
#                 loss = err_s_label
#                 domain_loss_val = 0.0
#             else:
#                 err_s_domain = criterion_domain(domain_out_s, domain_label_s)
#                 err_t_domain = criterion_domain(domain_out_t, domain_label_t)
#                 domain_loss_val = (err_s_domain + err_t_domain).item()
#                 loss = err_s_label + (err_s_domain + err_t_domain) * domain_weight
#
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#
#             total_loss += loss.item()
#             total_cls_loss += err_s_label.item()
#             total_dom_loss += domain_loss_val
#
#             preds = class_out_s.argmax(dim=1)
#             total_correct += (preds == s_label).sum().item()
#             total_samples += bs_src
#
#         scheduler.step()
#
#         avg_cls_loss = total_cls_loss / max(len_dataloader, 1)
#         avg_dom_loss = total_dom_loss / max(len_dataloader, 1)
#         train_acc = total_correct / max(total_samples, 1)
#
#         model.eval()
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for xb, yb in src_val_loader:
#                 xb = xb.to(DEVICE)
#                 class_out, _, _ = model(xb, alpha=0.0)
#                 preds = class_out.argmax(dim=1).cpu().numpy()
#                 y_pred.extend(preds)
#                 y_true.extend(yb.numpy())
#
#         acc = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred, average='macro')
#
#         if epoch % 10 == 0 or epoch == 1:
#             print(f"  [DANN] Epoch [{epoch:03d}/{DANN_EPOCHS}]")
#             print(f"    ClsLoss={avg_cls_loss:.4f} DomLoss={avg_dom_loss:.4f}")
#             print(f"    TrainAcc={train_acc:.4f} ValAcc={acc:.4f} ValF1={f1:.4f}")
#
#         if f1 > best_f1:
#             best_f1 = f1
#             best_acc = acc
#             best_state = model.state_dict()
#             no_improve_count = 0
#         else:
#             no_improve_count += 1
#
#         if no_improve_count >= PATIENCE and epoch > WARMUP_EPOCHS + 10:
#             print(f"  [早停] 连续 {PATIENCE} 轮无改善，停止训练")
#             break
#
#     if best_state is not None:
#         model.load_state_dict(best_state)
#
#     return best_acc, best_f1
#
#
# # =========================================
# # 结果汇总
# # =========================================
# def summarize_results(name, results):
#     accs = np.array([r[0] for r in results])
#     f1s = np.array([r[1] for r in results])
#
#     print("\n" + "#" * 60)
#     print(f"{name} 在 5 次随机划分上的结果：")
#     for i, (acc, f1) in enumerate(results, 1):
#         print(f"  Run{i}: Acc={acc:.4f}, F1={f1:.4f}")
#     print("-" * 60)
#     print(f"  Accuracy: mean={accs.mean():.4f}, std={accs.std(ddof=0):.4f}")
#     print(f"  Macro F1: mean={f1s.mean():.4f}, std={f1s.std(ddof=0):.4f}")
#     print("#" * 60)
#
#
# # =========================================
# # 主函数
# # =========================================
# def main():
#     print("正在加载源域数据 source_x.npy / source_y.npy ...")
#     X = np.load("source_x.npy").astype(np.float32)
#     y = np.load("source_y.npy").astype(np.int64)
#     print(f"数据形状：X = {X.shape}, y = {y.shape}")
#
#     unique, counts = np.unique(y, return_counts=True)
#     print("类别分布:", dict(zip(unique, counts)))
#
#     mean = X.mean()
#     std = X.std()
#     X = (X - mean) / (std + 1e-5)
#
#     all_feats = extract_stat_features(X)
#
#     print("正在加载目标域数据 target_data.npy ...")
#     target_dict = np.load("target_data.npy", allow_pickle=True).item()
#     tgt_list = []
#     for k, data in target_dict.items():
#         data = data.astype(np.float32)
#         data = (data - mean) / (std + 1e-5)
#         tgt_list.append(data)
#     target_x = np.concatenate(tgt_list, axis=0)
#     print(f"目标域数据合并后形状：{target_x.shape}")
#
#     seeds = [0, 1, 2, 3, 4]
#     cnn_results = []
#     svm_results = []
#     dann_results = []
#
#     indices = np.arange(len(y))
#
#     for i, seed in enumerate(seeds, 1):
#         print("\n" + "=" * 60)
#         print(f"  第 {i} 次随机划分 (random_state={seed})")
#         print("=" * 60)
#
#         train_idx, val_idx = train_test_split(
#             indices,
#             test_size=0.2,
#             random_state=seed,
#             stratify=y
#         )
#
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]
#         feats_train, feats_val = all_feats[train_idx], all_feats[val_idx]
#
#         print(f"划分完成：训练集 {len(train_idx)}，验证集 {len(val_idx)}")
#
#         acc_cnn, f1_cnn = train_one_cnn_run(X_train, y_train, X_val, y_val, seed)
#         cnn_results.append((acc_cnn, f1_cnn))
#
#         acc_svm, f1_svm = train_one_svm_run(feats_train, y_train, feats_val, y_val, seed)
#         svm_results.append((acc_svm, f1_svm))
#
#         acc_dann, f1_dann = train_one_dann_run_improved(X_train, y_train, X_val, y_val, target_x, seed)
#         dann_results.append((acc_dann, f1_dann))
#
#     summarize_results("CNN 基线模型", cnn_results)
#     summarize_results("统计特征 + SVM 基线模型", svm_results)
#     summarize_results("DANN 域自适应模型（改进版）", dann_results)
#
#
# if __name__ == "__main__":
#     main()


# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
#
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, f1_score, classification_report
# from sklearn.model_selection import train_test_split
#
# # =========================================
# # 通用配置
# # =========================================
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NUM_CLASSES = 4
# CNN_EPOCHS = 60
# DANN_EPOCHS = 100
# BATCH_SIZE = 128
# LR_CNN = 1e-3
# LR_DANN = 3e-4
#
#
# # =========================================
# # 统计特征提取
# # =========================================
# def extract_stat_features(X: np.ndarray) -> np.ndarray:
#     N = X.shape[0]
#     feats = np.zeros((N, 5), dtype=np.float32)
#     for i in range(N):
#         x = X[i]
#         mean = np.mean(x)
#         std = np.std(x)
#         rms = np.sqrt(np.mean(x ** 2))
#         var = np.var(x) + 1e-12
#         kurtosis = np.mean((x - mean) ** 4) / (var ** 2)
#         peak_factor = np.max(np.abs(x)) / (rms + 1e-12)
#         feats[i] = [mean, std, rms, kurtosis, peak_factor]
#     return feats
#
#
# # =========================================
# # 简单 1D-CNN
# # =========================================
# class SimpleCNN(nn.Module):
#     def __init__(self, input_length=512, num_classes=4):
#         super(SimpleCNN, self).__init__()
#         self.feature = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1)
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, num_classes)
#         )
#
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.feature(x)
#         x = self.classifier(x)
#         return x
#
#
# # =========================================
# # 梯度反转层
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
# # =========================================
# # 改进版 DANN 模型
# # =========================================
# class DANN_Model_Improved(nn.Module):
#     def __init__(self, num_classes=4):
#         super(DANN_Model_Improved, self).__init__()
#         self.feature = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
#             nn.InstanceNorm1d(16, affine=True),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
#             nn.InstanceNorm1d(32, affine=True),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm1d(64, affine=True),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1)
#         )
#         self.class_classifier = nn.Sequential(
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, num_classes)
#         )
#         self.domain_classifier = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 2)
#         )
#
#     def forward(self, x, alpha=1.0):
#         x = x.unsqueeze(1)
#         features = self.feature(x)
#         features = features.view(features.size(0), -1)
#         class_output = self.class_classifier(features)
#         reverse_features = GradientReverseLayer.apply(features, alpha)
#         domain_output = self.domain_classifier(reverse_features)
#         return class_output, domain_output, features
#
#
# # =========================================
# # CNN 训练函数（带类别平衡）
# # =========================================
# def train_one_cnn_run(X_train, y_train, X_val, y_val, random_state):
#     torch.manual_seed(random_state)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(random_state)
#
#     # 计算类别权重
#     class_counts = np.bincount(y_train)
#     class_weights = len(y_train) / (len(class_counts) * class_counts)
#     class_weights = torch.FloatTensor(class_weights).to(DEVICE)
#
#     train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
#     val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
#
#     model = SimpleCNN(input_length=X_train.shape[1], num_classes=NUM_CLASSES).to(DEVICE)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#     optimizer = optim.Adam(model.parameters(), lr=LR_CNN)
#
#     best_f1 = 0.0
#     best_acc = 0.0
#     best_state = None
#
#     print(f"\n[Run seed={random_state}] 训练 CNN 基线模型 ...")
#     for epoch in range(1, CNN_EPOCHS + 1):
#         model.train()
#         total_loss = 0.0
#         for xb, yb in train_loader:
#             xb = xb.to(DEVICE)
#             yb = yb.to(DEVICE)
#             optimizer.zero_grad()
#             logits = model(xb)
#             loss = criterion(logits, yb)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         model.eval()
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 xb = xb.to(DEVICE)
#                 logits = model(xb)
#                 preds = logits.argmax(dim=1).cpu().numpy()
#                 y_pred.extend(preds)
#                 y_true.extend(yb.numpy())
#
#         acc = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred, average='macro')
#
#         if epoch % 10 == 0 or epoch == 1:
#             avg_loss = total_loss / len(train_loader)
#             print(f"  Epoch [{epoch:03d}/{CNN_EPOCHS}] Loss={avg_loss:.4f}  ValAcc={acc:.4f}  ValF1={f1:.4f}")
#
#         if f1 > best_f1:
#             best_f1 = f1
#             best_acc = acc
#             best_state = model.state_dict()
#
#     if best_state is not None:
#         model.load_state_dict(best_state)
#
#     return best_acc, best_f1
#
#
# # =========================================
# # SVM 训练函数（带类别平衡）
# # =========================================
# def train_one_svm_run(feats_train, y_train, feats_val, y_val, random_state):
#     print(f"[Run seed={random_state}] 训练 SVM 基线模型 ...")
#     clf = Pipeline([
#         ("scaler", StandardScaler()),
#         ("svc", SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced'))
#     ])
#     clf.fit(feats_train, y_train)
#     y_pred = clf.predict(feats_val)
#     acc = accuracy_score(y_val, y_pred)
#     f1 = f1_score(y_val, y_pred, average='macro')
#     print(f"  SVM ValAcc={acc:.4f}  ValF1={f1:.4f}")
#     return acc, f1
#
#
# # =========================================
# # DANN 训练函数（带类别平衡）
# # =========================================
# def train_one_dann_run_balanced(X_train, y_train, X_val, y_val, target_x, random_state):
#     torch.manual_seed(random_state)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(random_state)
#
#     # 计算类别权重
#     class_counts = np.bincount(y_train)
#     class_weights = len(y_train) / (len(class_counts) * class_counts)
#     class_weights = torch.FloatTensor(class_weights).to(DEVICE)
#     print(f"  类别权重: {class_weights.cpu().numpy().round(2)}")
#
#     src_train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
#     src_val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
#     src_train_loader = DataLoader(src_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#     src_val_loader = DataLoader(src_val_ds, batch_size=BATCH_SIZE, shuffle=False)
#
#     tgt_ds = TensorDataset(torch.from_numpy(target_x), torch.zeros(len(target_x)).long())
#     tgt_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#
#     model = DANN_Model_Improved(num_classes=NUM_CLASSES).to(DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=LR_DANN, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DANN_EPOCHS)
#
#     # 使用加权交叉熵
#     criterion_class = nn.CrossEntropyLoss(weight=class_weights)
#     criterion_domain = nn.CrossEntropyLoss()
#
#     best_f1 = 0.0
#     best_acc = 0.0
#     best_state = None
#
#     WARMUP_EPOCHS = 40
#     DOMAIN_WEIGHT = 0.02
#     PATIENCE = 20
#     no_improve_count = 0
#
#     print(f"\n[Run seed={random_state}] 训练类别平衡版 DANN 模型 ...")
#
#     for epoch in range(1, DANN_EPOCHS + 1):
#         model.train()
#         total_cls_loss = 0.0
#         total_dom_loss = 0.0
#         total_correct = 0
#         total_samples = 0
#
#         len_dataloader = min(len(src_train_loader), len(tgt_loader))
#         src_iter = iter(src_train_loader)
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
#             bs_src = s_data.size(0)
#             bs_tgt = t_data.size(0)
#
#             domain_label_s = torch.zeros(bs_src, dtype=torch.long, device=DEVICE)
#             domain_label_t = torch.ones(bs_tgt, dtype=torch.long, device=DEVICE)
#
#             p = float(i + (epoch - 1) * len_dataloader) / (DANN_EPOCHS * len_dataloader)
#             alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
#
#             if epoch <= WARMUP_EPOCHS:
#                 alpha = 0.0
#                 domain_weight = 0.0
#             else:
#                 progress = (epoch - WARMUP_EPOCHS) / (DANN_EPOCHS - WARMUP_EPOCHS)
#                 domain_weight = DOMAIN_WEIGHT * min(progress * 2, 1.0)
#
#             class_out_s, domain_out_s, _ = model(s_data, alpha)
#             _, domain_out_t, _ = model(t_data, alpha)
#
#             err_s_label = criterion_class(class_out_s, s_label)
#
#             if epoch <= WARMUP_EPOCHS:
#                 loss = err_s_label
#                 domain_loss_val = 0.0
#             else:
#                 err_s_domain = criterion_domain(domain_out_s, domain_label_s)
#                 err_t_domain = criterion_domain(domain_out_t, domain_label_t)
#                 domain_loss_val = (err_s_domain + err_t_domain).item()
#                 loss = err_s_label + (err_s_domain + err_t_domain) * domain_weight
#
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#
#             total_cls_loss += err_s_label.item()
#             total_dom_loss += domain_loss_val
#
#             preds = class_out_s.argmax(dim=1)
#             total_correct += (preds == s_label).sum().item()
#             total_samples += bs_src
#
#         scheduler.step()
#
#         avg_cls_loss = total_cls_loss / max(len_dataloader, 1)
#         avg_dom_loss = total_dom_loss / max(len_dataloader, 1)
#         train_acc = total_correct / max(total_samples, 1)
#
#         model.eval()
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for xb, yb in src_val_loader:
#                 xb = xb.to(DEVICE)
#                 class_out, _, _ = model(xb, alpha=0.0)
#                 preds = class_out.argmax(dim=1).cpu().numpy()
#                 y_pred.extend(preds)
#                 y_true.extend(yb.numpy())
#
#         acc = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred, average='macro')
#
#         if epoch % 10 == 0 or epoch == 1:
#             pred_dist = np.bincount(y_pred, minlength=NUM_CLASSES)
#             print(f"  [DANN] Epoch [{epoch:03d}/{DANN_EPOCHS}]")
#             print(f"    ClsLoss={avg_cls_loss:.4f} DomLoss={avg_dom_loss:.4f}")
#             print(f"    TrainAcc={train_acc:.4f} ValAcc={acc:.4f} ValF1={f1:.4f}")
#             print(f"    预测分布: {pred_dist}")
#
#         if f1 > best_f1:
#             best_f1 = f1
#             best_acc = acc
#             best_state = model.state_dict()
#             no_improve_count = 0
#         else:
#             no_improve_count += 1
#
#         if no_improve_count >= PATIENCE and epoch > WARMUP_EPOCHS + 10:
#             print(f"  [早停] 连续 {PATIENCE} 轮无改善")
#             break
#
#     if best_state is not None:
#         model.load_state_dict(best_state)
#
#     # 打印最终分类报告
#     model.eval()
#     y_true_final, y_pred_final = [], []
#     with torch.no_grad():
#         for xb, yb in src_val_loader:
#             xb = xb.to(DEVICE)
#             class_out, _, _ = model(xb, alpha=0.0)
#             preds = class_out.argmax(dim=1).cpu().numpy()
#             y_pred_final.extend(preds)
#             y_true_final.extend(yb.numpy())
#
#     print("\n  [最终分类报告]")
#     print(classification_report(y_true_final, y_pred_final,
#                                 target_names=['Normal', 'IR', 'OR', 'Ball'], digits=4))
#
#     return best_acc, best_f1
#
#
# # =========================================
# # 结果汇总
# # =========================================
# def summarize_results(name, results):
#     accs = np.array([r[0] for r in results])
#     f1s = np.array([r[1] for r in results])
#     print("\n" + "#" * 60)
#     print(f"{name} 在 5 次随机划分上的结果：")
#     for i, (acc, f1) in enumerate(results, 1):
#         print(f"  Run{i}: Acc={acc:.4f}, F1={f1:.4f}")
#     print("-" * 60)
#     print(f"  Accuracy: mean={accs.mean():.4f}, std={accs.std(ddof=0):.4f}")
#     print(f"  Macro F1: mean={f1s.mean():.4f}, std={f1s.std(ddof=0):.4f}")
#     print("#" * 60)
#
#
# # =========================================
# # 主函数
# # =========================================
# def main():
#     print("正在加载源域数据 source_x.npy / source_y.npy ...")
#     X = np.load("source_x.npy").astype(np.float32)
#     y = np.load("source_y.npy").astype(np.int64)
#     print(f"数据形状：X = {X.shape}, y = {y.shape}")
#
#     unique, counts = np.unique(y, return_counts=True)
#     print("类别分布:", dict(zip(unique, counts)))
#     print("类别比例:", {k: f"{v / len(y) * 100:.1f}%" for k, v in zip(unique, counts)})
#
#     mean = X.mean()
#     std = X.std()
#     X = (X - mean) / (std + 1e-5)
#
#     all_feats = extract_stat_features(X)
#
#     print("正在加载目标域数据 target_data.npy ...")
#     target_dict = np.load("target_data.npy", allow_pickle=True).item()
#     tgt_list = []
#     for k, data in target_dict.items():
#         data = data.astype(np.float32)
#         data = (data - mean) / (std + 1e-5)
#         tgt_list.append(data)
#     target_x = np.concatenate(tgt_list, axis=0)
#     print(f"目标域数据合并后形状：{target_x.shape}")
#
#     seeds = [0, 1, 2, 3, 4]
#     cnn_results = []
#     svm_results = []
#     dann_results = []
#
#     indices = np.arange(len(y))
#
#     for i, seed in enumerate(seeds, 1):
#         print("\n" + "=" * 60)
#         print(f"  第 {i} 次随机划分 (random_state={seed})")
#         print("=" * 60)
#
#         train_idx, val_idx = train_test_split(
#             indices, test_size=0.2, random_state=seed, stratify=y
#         )
#
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]
#         feats_train, feats_val = all_feats[train_idx], all_feats[val_idx]
#
#         print(f"划分完成：训练集 {len(train_idx)}，验证集 {len(val_idx)}")
#
#         acc_cnn, f1_cnn = train_one_cnn_run(X_train, y_train, X_val, y_val, seed)
#         cnn_results.append((acc_cnn, f1_cnn))
#
#         acc_svm, f1_svm = train_one_svm_run(feats_train, y_train, feats_val, y_val, seed)
#         svm_results.append((acc_svm, f1_svm))
#
#         acc_dann, f1_dann = train_one_dann_run_balanced(X_train, y_train, X_val, y_val, target_x, seed)
#         dann_results.append((acc_dann, f1_dann))
#
#     summarize_results("CNN 基线模型", cnn_results)
#     summarize_results("统计特征 + SVM 基线模型", svm_results)
#     summarize_results("DANN 域自适应模型（类别平衡版）", dann_results)
#
#
# if __name__ == "__main__":
#     main()


# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
#
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, f1_score, classification_report
# from sklearn.model_selection import train_test_split
#
# # =========================================
# # 通用配置
# # =========================================
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NUM_CLASSES = 4
# CNN_EPOCHS = 60
# DANN_EPOCHS = 100
# BATCH_SIZE = 128
# LR_CNN = 1e-3
# LR_DANN = 3e-4
#
#
# # =========================================
# # 统计特征提取
# # =========================================
# def extract_stat_features(X: np.ndarray) -> np.ndarray:
#     N = X.shape[0]
#     feats = np.zeros((N, 5), dtype=np.float32)
#     for i in range(N):
#         x = X[i]
#         mean = np.mean(x)
#         std = np.std(x)
#         rms = np.sqrt(np.mean(x ** 2))
#         var = np.var(x) + 1e-12
#         kurtosis = np.mean((x - mean) ** 4) / (var ** 2)
#         peak_factor = np.max(np.abs(x)) / (rms + 1e-12)
#         feats[i] = [mean, std, rms, kurtosis, peak_factor]
#     return feats
#
#
# # =========================================
# # 简单 1D-CNN
# # =========================================
# class SimpleCNN(nn.Module):
#     def __init__(self, input_length=512, num_classes=4):
#         super(SimpleCNN, self).__init__()
#         self.feature = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1)
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, num_classes)
#         )
#
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.feature(x)
#         x = self.classifier(x)
#         return x
#
#
# # =========================================
# # 梯度反转层
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
# # =========================================
# # 改进版 DANN 模型
# # =========================================
# class DANN_Model_Improved(nn.Module):
#     def __init__(self, num_classes=4):
#         super(DANN_Model_Improved, self).__init__()
#         self.feature = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
#             nn.InstanceNorm1d(16, affine=True),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
#             nn.InstanceNorm1d(32, affine=True),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm1d(64, affine=True),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1)
#         )
#         self.class_classifier = nn.Sequential(
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, num_classes)
#         )
#         self.domain_classifier = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 2)
#         )
#
#     def forward(self, x, alpha=1.0):
#         x = x.unsqueeze(1)
#         features = self.feature(x)
#         features = features.view(features.size(0), -1)
#         class_output = self.class_classifier(features)
#         reverse_features = GradientReverseLayer.apply(features, alpha)
#         domain_output = self.domain_classifier(reverse_features)
#         return class_output, domain_output, features
#
#
# # =========================================
# # CNN 训练函数
# # =========================================
# def train_one_cnn_run(X_train, y_train, X_val, y_val, random_state):
#     torch.manual_seed(random_state)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(random_state)
#
#     class_counts = np.bincount(y_train)
#     class_weights = len(y_train) / (len(class_counts) * class_counts)
#     class_weights = torch.FloatTensor(class_weights).to(DEVICE)
#
#     train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
#     val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
#
#     model = SimpleCNN(input_length=X_train.shape[1], num_classes=NUM_CLASSES).to(DEVICE)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#     optimizer = optim.Adam(model.parameters(), lr=LR_CNN)
#
#     best_f1 = 0.0
#     best_acc = 0.0
#     best_state = None
#
#     print(f"\n[Run seed={random_state}] 训练 CNN 基线模型 ...")
#     for epoch in range(1, CNN_EPOCHS + 1):
#         model.train()
#         total_loss = 0.0
#         for xb, yb in train_loader:
#             xb = xb.to(DEVICE)
#             yb = yb.to(DEVICE)
#             optimizer.zero_grad()
#             logits = model(xb)
#             loss = criterion(logits, yb)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         model.eval()
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 xb = xb.to(DEVICE)
#                 logits = model(xb)
#                 preds = logits.argmax(dim=1).cpu().numpy()
#                 y_pred.extend(preds)
#                 y_true.extend(yb.numpy())
#
#         acc = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred, average='macro')
#
#         if epoch % 10 == 0 or epoch == 1:
#             avg_loss = total_loss / len(train_loader)
#             print(f"  Epoch [{epoch:03d}/{CNN_EPOCHS}] Loss={avg_loss:.4f}  ValAcc={acc:.4f}  ValF1={f1:.4f}")
#
#         if f1 > best_f1:
#             best_f1 = f1
#             best_acc = acc
#             best_state = model.state_dict()
#
#     if best_state is not None:
#         model.load_state_dict(best_state)
#
#     return best_acc, best_f1
#
#
# # =========================================
# # SVM 训练函数
# # =========================================
# def train_one_svm_run(feats_train, y_train, feats_val, y_val, random_state):
#     print(f"[Run seed={random_state}] 训练 SVM 基线模型 ...")
#     clf = Pipeline([
#         ("scaler", StandardScaler()),
#         ("svc", SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced'))
#     ])
#     clf.fit(feats_train, y_train)
#     y_pred = clf.predict(feats_val)
#     acc = accuracy_score(y_val, y_pred)
#     f1 = f1_score(y_val, y_pred, average='macro')
#     print(f"  SVM ValAcc={acc:.4f}  ValF1={f1:.4f}")
#     return acc, f1
#
#
# # =========================================
# # 【核心改进】生成伪标签的函数
# # =========================================
# def generate_pseudo_labels(model, target_x, confidence_threshold=0.85):
#     """
#     对目标域数据生成伪标签
#
#     参数:
#         model: 训练好的模型
#         target_x: 目标域数据 (numpy array)
#         confidence_threshold: 置信度阈值，只有预测置信度高于此值的样本才会被使用
#
#     返回:
#         pseudo_data: 高置信度样本的数据
#         pseudo_labels: 对应的伪标签
#         num_per_class: 每个类别的伪标签数量
#     """
#     model.eval()
#
#     # 将目标域数据转换为 tensor
#     target_tensor = torch.from_numpy(target_x).to(DEVICE)
#
#     # 分批处理，避免显存溢出
#     batch_size = 256
#     all_probs = []
#
#     with torch.no_grad():
#         for i in range(0, len(target_tensor), batch_size):
#             batch = target_tensor[i:i + batch_size]
#             logits, _, _ = model(batch, alpha=0)
#             probs = F.softmax(logits, dim=1)
#             all_probs.append(probs.cpu())
#
#     all_probs = torch.cat(all_probs, dim=0)
#
#     # 获取每个样本的最大置信度和对应的预测类别
#     confidence, pseudo_labels = all_probs.max(dim=1)
#
#     # 筛选高置信度样本
#     high_conf_mask = confidence > confidence_threshold
#
#     # 获取高置信度样本的数据和标签
#     pseudo_data = target_x[high_conf_mask.numpy()]
#     pseudo_labels = pseudo_labels[high_conf_mask].numpy()
#
#     # 统计每个类别的伪标签数量
#     if len(pseudo_labels) > 0:
#         num_per_class = np.bincount(pseudo_labels, minlength=NUM_CLASSES)
#     else:
#         num_per_class = np.zeros(NUM_CLASSES, dtype=int)
#
#     return pseudo_data, pseudo_labels, num_per_class
#
#
# # =========================================
# # 【核心改进】带伪标签的 DANN 训练函数
# # =========================================
# def train_one_dann_run_with_pseudo(X_train, y_train, X_val, y_val, target_x, random_state):
#     """带伪标签策略的 DANN 训练"""
#     torch.manual_seed(random_state)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(random_state)
#
#     # 计算类别权重
#     class_counts = np.bincount(y_train)
#     class_weights = len(y_train) / (len(class_counts) * class_counts)
#     class_weights = torch.FloatTensor(class_weights).to(DEVICE)
#     print(f"  类别权重: {class_weights.cpu().numpy().round(2)}")
#
#     src_train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
#     src_val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
#     src_train_loader = DataLoader(src_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#     src_val_loader = DataLoader(src_val_ds, batch_size=BATCH_SIZE, shuffle=False)
#
#     tgt_ds = TensorDataset(torch.from_numpy(target_x), torch.zeros(len(target_x)).long())
#     tgt_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#
#     model = DANN_Model_Improved(num_classes=NUM_CLASSES).to(DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=LR_DANN, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DANN_EPOCHS)
#
#     criterion_class = nn.CrossEntropyLoss(weight=class_weights)
#     criterion_domain = nn.CrossEntropyLoss()
#     # 伪标签损失不使用类别权重，因为伪标签本身可能有偏差
#     criterion_pseudo = nn.CrossEntropyLoss()
#
#     best_f1 = 0.0
#     best_acc = 0.0
#     best_state = None
#
#     WARMUP_EPOCHS = 40
#     DOMAIN_WEIGHT = 0.02
#     PATIENCE = 20
#     no_improve_count = 0
#
#     # 【伪标签配置】
#     PSEUDO_START_EPOCH = 60  # 从第60轮开始使用伪标签
#     PSEUDO_THRESHOLD = 0.85  # 置信度阈值
#     PSEUDO_WEIGHT = 0.1  # 伪标签损失权重（较小，避免错误标签影响太大）
#     PSEUDO_UPDATE_FREQ = 10  # 每10轮更新一次伪标签
#
#     # 存储当前的伪标签数据
#     current_pseudo_data = None
#     current_pseudo_labels = None
#
#     print(f"\n[Run seed={random_state}] 训练 DANN 模型（带伪标签） ...")
#
#     for epoch in range(1, DANN_EPOCHS + 1):
#         model.train()
#         total_cls_loss = 0.0
#         total_dom_loss = 0.0
#         total_pseudo_loss = 0.0
#         total_correct = 0
#         total_samples = 0
#
#         # 【伪标签更新】在指定轮次更新伪标签
#         if epoch >= PSEUDO_START_EPOCH and (epoch - PSEUDO_START_EPOCH) % PSEUDO_UPDATE_FREQ == 0:
#             pseudo_data, pseudo_labels, num_per_class = generate_pseudo_labels(
#                 model, target_x, confidence_threshold=PSEUDO_THRESHOLD
#             )
#
#             if len(pseudo_labels) > 0:
#                 current_pseudo_data = pseudo_data.astype(np.float32)
#                 current_pseudo_labels = pseudo_labels.astype(np.int64)
#                 print(f"  [伪标签更新] Epoch {epoch}: 生成 {len(pseudo_labels)} 个伪标签")
#                 print(f"    各类别数量: Normal={num_per_class[0]}, IR={num_per_class[1]}, "
#                       f"OR={num_per_class[2]}, Ball={num_per_class[3]}")
#             else:
#                 current_pseudo_data = None
#                 current_pseudo_labels = None
#                 print(f"  [伪标签更新] Epoch {epoch}: 无高置信度样本")
#
#         len_dataloader = min(len(src_train_loader), len(tgt_loader))
#         src_iter = iter(src_train_loader)
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
#             bs_src = s_data.size(0)
#             bs_tgt = t_data.size(0)
#
#             domain_label_s = torch.zeros(bs_src, dtype=torch.long, device=DEVICE)
#             domain_label_t = torch.ones(bs_tgt, dtype=torch.long, device=DEVICE)
#
#             p = float(i + (epoch - 1) * len_dataloader) / (DANN_EPOCHS * len_dataloader)
#             alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
#
#             if epoch <= WARMUP_EPOCHS:
#                 alpha = 0.0
#                 domain_weight = 0.0
#             else:
#                 progress = (epoch - WARMUP_EPOCHS) / (DANN_EPOCHS - WARMUP_EPOCHS)
#                 domain_weight = DOMAIN_WEIGHT * min(progress * 2, 1.0)
#
#             class_out_s, domain_out_s, _ = model(s_data, alpha)
#             _, domain_out_t, _ = model(t_data, alpha)
#
#             # 源域分类损失
#             err_s_label = criterion_class(class_out_s, s_label)
#
#             # 域对抗损失
#             if epoch <= WARMUP_EPOCHS:
#                 loss = err_s_label
#                 domain_loss_val = 0.0
#             else:
#                 err_s_domain = criterion_domain(domain_out_s, domain_label_s)
#                 err_t_domain = criterion_domain(domain_out_t, domain_label_t)
#                 domain_loss_val = (err_s_domain + err_t_domain).item()
#                 loss = err_s_label + (err_s_domain + err_t_domain) * domain_weight
#
#             # 【伪标签损失】
#             pseudo_loss_val = 0.0
#             if epoch >= PSEUDO_START_EPOCH and current_pseudo_data is not None and len(current_pseudo_data) > 0:
#                 # 随机采样一个 batch 的伪标签样本
#                 pseudo_batch_size = min(BATCH_SIZE // 2, len(current_pseudo_data))
#                 pseudo_indices = np.random.choice(len(current_pseudo_data), pseudo_batch_size, replace=False)
#
#                 pseudo_batch_x = torch.from_numpy(current_pseudo_data[pseudo_indices]).to(DEVICE)
#                 pseudo_batch_y = torch.from_numpy(current_pseudo_labels[pseudo_indices]).to(DEVICE)
#
#                 # 计算伪标签损失
#                 pseudo_out, _, _ = model(pseudo_batch_x, alpha=0)
#                 pseudo_loss = criterion_pseudo(pseudo_out, pseudo_batch_y)
#
#                 # 动态调整伪标签权重：随着训练进行逐渐增加
#                 pseudo_progress = (epoch - PSEUDO_START_EPOCH) / (DANN_EPOCHS - PSEUDO_START_EPOCH)
#                 current_pseudo_weight = PSEUDO_WEIGHT * min(pseudo_progress * 2, 1.0)
#
#                 loss = loss + pseudo_loss * current_pseudo_weight
#                 pseudo_loss_val = pseudo_loss.item()
#
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#
#             total_cls_loss += err_s_label.item()
#             total_dom_loss += domain_loss_val
#             total_pseudo_loss += pseudo_loss_val
#
#             preds = class_out_s.argmax(dim=1)
#             total_correct += (preds == s_label).sum().item()
#             total_samples += bs_src
#
#         scheduler.step()
#
#         avg_cls_loss = total_cls_loss / max(len_dataloader, 1)
#         avg_dom_loss = total_dom_loss / max(len_dataloader, 1)
#         avg_pseudo_loss = total_pseudo_loss / max(len_dataloader, 1)
#         train_acc = total_correct / max(total_samples, 1)
#
#         model.eval()
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for xb, yb in src_val_loader:
#                 xb = xb.to(DEVICE)
#                 class_out, _, _ = model(xb, alpha=0.0)
#                 preds = class_out.argmax(dim=1).cpu().numpy()
#                 y_pred.extend(preds)
#                 y_true.extend(yb.numpy())
#
#         acc = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred, average='macro')
#
#         if epoch % 10 == 0 or epoch == 1:
#             pred_dist = np.bincount(y_pred, minlength=NUM_CLASSES)
#             print(f"  [DANN] Epoch [{epoch:03d}/{DANN_EPOCHS}]")
#             print(f"    ClsLoss={avg_cls_loss:.4f} DomLoss={avg_dom_loss:.4f} PseudoLoss={avg_pseudo_loss:.4f}")
#             print(f"    TrainAcc={train_acc:.4f} ValAcc={acc:.4f} ValF1={f1:.4f}")
#             print(f"    预测分布: {pred_dist}")
#
#         if f1 > best_f1:
#             best_f1 = f1
#             best_acc = acc
#             best_state = model.state_dict()
#             no_improve_count = 0
#         else:
#             no_improve_count += 1
#
#         if no_improve_count >= PATIENCE and epoch > WARMUP_EPOCHS + 10:
#             print(f"  [早停] 连续 {PATIENCE} 轮无改善")
#             break
#
#     if best_state is not None:
#         model.load_state_dict(best_state)
#
#     # 打印最终分类报告
#     model.eval()
#     y_true_final, y_pred_final = [], []
#     with torch.no_grad():
#         for xb, yb in src_val_loader:
#             xb = xb.to(DEVICE)
#             class_out, _, _ = model(xb, alpha=0.0)
#             preds = class_out.argmax(dim=1).cpu().numpy()
#             y_pred_final.extend(preds)
#             y_true_final.extend(yb.numpy())
#
#     print("\n  [最终分类报告]")
#     print(classification_report(y_true_final, y_pred_final,
#                                 target_names=['Normal', 'IR', 'OR', 'Ball'],
#                                 digits=4, zero_division=0))
#
#     return best_acc, best_f1
#
#
# # =========================================
# # 结果汇总
# # =========================================
# def summarize_results(name, results):
#     accs = np.array([r[0] for r in results])
#     f1s = np.array([r[1] for r in results])
#     print("\n" + "#" * 60)
#     print(f"{name} 在 5 次随机划分上的结果：")
#     for i, (acc, f1) in enumerate(results, 1):
#         print(f"  Run{i}: Acc={acc:.4f}, F1={f1:.4f}")
#     print("-" * 60)
#     print(f"  Accuracy: mean={accs.mean():.4f}, std={accs.std(ddof=0):.4f}")
#     print(f"  Macro F1: mean={f1s.mean():.4f}, std={f1s.std(ddof=0):.4f}")
#     print("#" * 60)
#
#
# # =========================================
# # 主函数
# # =========================================
# def main():
#     print("正在加载源域数据 source_x.npy / source_y.npy ...")
#     X = np.load("source_x.npy").astype(np.float32)
#     y = np.load("source_y.npy").astype(np.int64)
#     print(f"数据形状：X = {X.shape}, y = {y.shape}")
#
#     unique, counts = np.unique(y, return_counts=True)
#     print("类别分布:", dict(zip(unique, counts)))
#     print("类别比例:", {k: f"{v / len(y) * 100:.1f}%" for k, v in zip(unique, counts)})
#
#     mean = X.mean()
#     std = X.std()
#     X = (X - mean) / (std + 1e-5)
#
#     all_feats = extract_stat_features(X)
#
#     print("正在加载目标域数据 target_data.npy ...")
#     target_dict = np.load("target_data.npy", allow_pickle=True).item()
#     tgt_list = []
#     for k, data in target_dict.items():
#         data = data.astype(np.float32)
#         data = (data - mean) / (std + 1e-5)
#         tgt_list.append(data)
#     target_x = np.concatenate(tgt_list, axis=0)
#     print(f"目标域数据合并后形状：{target_x.shape}")
#
#     seeds = [0, 1, 2, 3, 4]
#     cnn_results = []
#     svm_results = []
#     dann_results = []
#
#     indices = np.arange(len(y))
#
#     for i, seed in enumerate(seeds, 1):
#         print("\n" + "=" * 60)
#         print(f"  第 {i} 次随机划分 (random_state={seed})")
#         print("=" * 60)
#
#         train_idx, val_idx = train_test_split(
#             indices, test_size=0.2, random_state=seed, stratify=y
#         )
#
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]
#         feats_train, feats_val = all_feats[train_idx], all_feats[val_idx]
#
#         print(f"划分完成：训练集 {len(train_idx)}，验证集 {len(val_idx)}")
#
#         acc_cnn, f1_cnn = train_one_cnn_run(X_train, y_train, X_val, y_val, seed)
#         cnn_results.append((acc_cnn, f1_cnn))
#
#         acc_svm, f1_svm = train_one_svm_run(feats_train, y_train, feats_val, y_val, seed)
#         svm_results.append((acc_svm, f1_svm))
#
#         # 使用带伪标签的 DANN
#         acc_dann, f1_dann = train_one_dann_run_with_pseudo(X_train, y_train, X_val, y_val, target_x, seed)
#         dann_results.append((acc_dann, f1_dann))
#
#     summarize_results("CNN 基线模型", cnn_results)
#     summarize_results("统计特征 + SVM 基线模型", svm_results)
#     summarize_results("DANN 域自适应模型（带伪标签）", dann_results)
#
#
# if __name__ == "__main__":
#     main()


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# =========================================
# 通用配置（增加轮次）
# =========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
CNN_EPOCHS = 60
DANN_EPOCHS = 150  # 增加到150轮
BATCH_SIZE = 128
LR_CNN = 1e-3
LR_DANN = 3e-4


# =========================================
# 统计特征提取
# =========================================
def extract_stat_features(X: np.ndarray) -> np.ndarray:
    N = X.shape[0]
    feats = np.zeros((N, 5), dtype=np.float32)
    for i in range(N):
        x = X[i]
        mean = np.mean(x)
        std = np.std(x)
        rms = np.sqrt(np.mean(x ** 2))
        var = np.var(x) + 1e-12
        kurtosis = np.mean((x - mean) ** 4) / (var ** 2)
        peak_factor = np.max(np.abs(x)) / (rms + 1e-12)
        feats[i] = [mean, std, rms, kurtosis, peak_factor]
    return feats


# =========================================
# 简单 1D-CNN
# =========================================
class SimpleCNN(nn.Module):
    def __init__(self, input_length=512, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.feature(x)
        x = self.classifier(x)
        return x


# =========================================
# 梯度反转层
# =========================================
class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# =========================================
# 改进版 DANN 模型
# =========================================
class DANN_Model_Improved(nn.Module):
    def __init__(self, num_classes=4):
        super(DANN_Model_Improved, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(16, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(32, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(64, affine=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x, alpha=1.0):
        x = x.unsqueeze(1)
        features = self.feature(x)
        features = features.view(features.size(0), -1)
        class_output = self.class_classifier(features)
        reverse_features = GradientReverseLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        return class_output, domain_output, features


# =========================================
# CNN 训练函数
# =========================================
def train_one_cnn_run(X_train, y_train, X_val, y_val, random_state):
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(input_length=X_train.shape[1], num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR_CNN)

    best_f1 = 0.0
    best_acc = 0.0
    best_state = None

    print(f"\n[Run seed={random_state}] 训练 CNN 基线模型 ...")
    for epoch in range(1, CNN_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                preds = logits.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(yb.numpy())

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch [{epoch:03d}/{CNN_EPOCHS}] Loss={avg_loss:.4f}  ValAcc={acc:.4f}  ValF1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_acc, best_f1


# =========================================
# SVM 训练函数
# =========================================
def train_one_svm_run(feats_train, y_train, feats_val, y_val, random_state):
    print(f"[Run seed={random_state}] 训练 SVM 基线模型 ...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced'))
    ])
    clf.fit(feats_train, y_train)
    y_pred = clf.predict(feats_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f"  SVM ValAcc={acc:.4f}  ValF1={f1:.4f}")
    return acc, f1


# =========================================
# 【改进】生成类别平衡的伪标签
# =========================================
def generate_balanced_pseudo_labels(model, target_x, confidence_threshold=0.80, max_per_class=500):
    """
    生成类别平衡的伪标签

    改进点：
    1. 降低置信度阈值到0.80，获取更多样本
    2. 对每个类别分别筛选，确保类别平衡
    3. 限制每个类别的最大样本数，避免某类主导
    """
    model.eval()

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

    # 对每个类别分别筛选高置信度样本
    pseudo_data_list = []
    pseudo_labels_list = []
    num_per_class = np.zeros(NUM_CLASSES, dtype=int)

    for cls in range(NUM_CLASSES):
        # 找到预测为该类别的样本
        cls_mask = predictions == cls
        cls_indices = torch.where(cls_mask)[0]

        if len(cls_indices) == 0:
            continue

        # 获取这些样本的置信度
        cls_confidence = confidence[cls_indices]

        # 筛选高置信度样本
        high_conf_mask = cls_confidence > confidence_threshold
        high_conf_indices = cls_indices[high_conf_mask]

        if len(high_conf_indices) == 0:
            # 如果没有超过阈值的，取置信度最高的几个（至少取一些）
            if len(cls_indices) > 0:
                # 取置信度最高的 min(10, len) 个
                num_to_take = min(10, len(cls_indices))
                _, top_indices = cls_confidence.topk(num_to_take)
                high_conf_indices = cls_indices[top_indices]

        # 限制每个类别的最大数量
        if len(high_conf_indices) > max_per_class:
            # 按置信度排序，取最高的
            cls_conf_selected = confidence[high_conf_indices]
            _, top_indices = cls_conf_selected.topk(max_per_class)
            high_conf_indices = high_conf_indices[top_indices]

        # 添加到列表
        if len(high_conf_indices) > 0:
            pseudo_data_list.append(target_x[high_conf_indices.numpy()])
            pseudo_labels_list.append(np.full(len(high_conf_indices), cls, dtype=np.int64))
            num_per_class[cls] = len(high_conf_indices)

    # 合并所有类别的伪标签
    if len(pseudo_data_list) > 0:
        pseudo_data = np.concatenate(pseudo_data_list, axis=0)
        pseudo_labels = np.concatenate(pseudo_labels_list, axis=0)
    else:
        pseudo_data = np.array([])
        pseudo_labels = np.array([])

    return pseudo_data, pseudo_labels, num_per_class


# =========================================
# 【优化版】带伪标签的 DANN 训练函数
# =========================================
def train_one_dann_run_with_pseudo_v2(X_train, y_train, X_val, y_val, target_x, random_state):
    """优化版：增加轮次 + 类别平衡伪标签"""
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    print(f"  类别权重: {class_weights.cpu().numpy().round(2)}")

    src_train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    src_val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    src_train_loader = DataLoader(src_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    src_val_loader = DataLoader(src_val_ds, batch_size=BATCH_SIZE, shuffle=False)

    tgt_ds = TensorDataset(torch.from_numpy(target_x), torch.zeros(len(target_x)).long())
    tgt_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = DANN_Model_Improved(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_DANN, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DANN_EPOCHS)

    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    criterion_domain = nn.CrossEntropyLoss()
    criterion_pseudo = nn.CrossEntropyLoss()  # 伪标签不加权

    best_f1 = 0.0
    best_acc = 0.0
    best_state = None

    # 【调整后的配置】
    WARMUP_EPOCHS = 50  # 延长预热期
    DOMAIN_WEIGHT = 0.02
    PATIENCE = 30  # 增加耐心值
    no_improve_count = 0

    # 【伪标签配置】
    PSEUDO_START_EPOCH = 70  # 延后开始
    PSEUDO_THRESHOLD = 0.80  # 降低阈值
    PSEUDO_WEIGHT = 0.15  # 稍微增加权重
    PSEUDO_UPDATE_FREQ = 10
    MAX_PER_CLASS = 400  # 每类最多400个

    current_pseudo_data = None
    current_pseudo_labels = None

    print(f"\n[Run seed={random_state}] 训练 DANN 模型（优化版伪标签） ...")

    for epoch in range(1, DANN_EPOCHS + 1):
        model.train()
        total_cls_loss = 0.0
        total_dom_loss = 0.0
        total_pseudo_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 【伪标签更新】使用类别平衡版本
        if epoch >= PSEUDO_START_EPOCH and (epoch - PSEUDO_START_EPOCH) % PSEUDO_UPDATE_FREQ == 0:
            pseudo_data, pseudo_labels, num_per_class = generate_balanced_pseudo_labels(
                model, target_x,
                confidence_threshold=PSEUDO_THRESHOLD,
                max_per_class=MAX_PER_CLASS
            )

            if len(pseudo_labels) > 0:
                current_pseudo_data = pseudo_data.astype(np.float32)
                current_pseudo_labels = pseudo_labels.astype(np.int64)
                print(f"  [伪标签更新] Epoch {epoch}: 生成 {len(pseudo_labels)} 个伪标签")
                print(f"    各类别数量: Normal={num_per_class[0]}, IR={num_per_class[1]}, "
                      f"OR={num_per_class[2]}, Ball={num_per_class[3]}")
            else:
                current_pseudo_data = None
                current_pseudo_labels = None

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

            p = float(i + (epoch - 1) * len_dataloader) / (DANN_EPOCHS * len_dataloader)
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

            if epoch <= WARMUP_EPOCHS:
                alpha = 0.0
                domain_weight = 0.0
            else:
                progress = (epoch - WARMUP_EPOCHS) / (DANN_EPOCHS - WARMUP_EPOCHS)
                domain_weight = DOMAIN_WEIGHT * min(progress * 2, 1.0)

            class_out_s, domain_out_s, _ = model(s_data, alpha)
            _, domain_out_t, _ = model(t_data, alpha)

            err_s_label = criterion_class(class_out_s, s_label)

            if epoch <= WARMUP_EPOCHS:
                loss = err_s_label
                domain_loss_val = 0.0
            else:
                err_s_domain = criterion_domain(domain_out_s, domain_label_s)
                err_t_domain = criterion_domain(domain_out_t, domain_label_t)
                domain_loss_val = (err_s_domain + err_t_domain).item()
                loss = err_s_label + (err_s_domain + err_t_domain) * domain_weight

            # 【伪标签损失】
            pseudo_loss_val = 0.0
            if epoch >= PSEUDO_START_EPOCH and current_pseudo_data is not None and len(current_pseudo_data) > 0:
                pseudo_batch_size = min(BATCH_SIZE // 2, len(current_pseudo_data))
                pseudo_indices = np.random.choice(len(current_pseudo_data), pseudo_batch_size, replace=False)

                pseudo_batch_x = torch.from_numpy(current_pseudo_data[pseudo_indices]).to(DEVICE)
                pseudo_batch_y = torch.from_numpy(current_pseudo_labels[pseudo_indices]).to(DEVICE)

                pseudo_out, _, _ = model(pseudo_batch_x, alpha=0)
                pseudo_loss = criterion_pseudo(pseudo_out, pseudo_batch_y)

                pseudo_progress = (epoch - PSEUDO_START_EPOCH) / (DANN_EPOCHS - PSEUDO_START_EPOCH)
                current_pseudo_weight = PSEUDO_WEIGHT * min(pseudo_progress * 2, 1.0)

                loss = loss + pseudo_loss * current_pseudo_weight
                pseudo_loss_val = pseudo_loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_cls_loss += err_s_label.item()
            total_dom_loss += domain_loss_val
            total_pseudo_loss += pseudo_loss_val

            preds = class_out_s.argmax(dim=1)
            total_correct += (preds == s_label).sum().item()
            total_samples += bs_src

        scheduler.step()

        avg_cls_loss = total_cls_loss / max(len_dataloader, 1)
        avg_dom_loss = total_dom_loss / max(len_dataloader, 1)
        avg_pseudo_loss = total_pseudo_loss / max(len_dataloader, 1)
        train_acc = total_correct / max(total_samples, 1)

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

        if epoch % 10 == 0 or epoch == 1:
            pred_dist = np.bincount(y_pred, minlength=NUM_CLASSES)
            print(f"  [DANN] Epoch [{epoch:03d}/{DANN_EPOCHS}]")
            print(f"    ClsLoss={avg_cls_loss:.4f} DomLoss={avg_dom_loss:.4f} PseudoLoss={avg_pseudo_loss:.4f}")
            print(f"    TrainAcc={train_acc:.4f} ValAcc={acc:.4f} ValF1={f1:.4f}")
            print(f"    预测分布: {pred_dist}")

        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_state = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 延后早停判断
        if no_improve_count >= PATIENCE and epoch > PSEUDO_START_EPOCH + 20:
            print(f"  [早停] 连续 {PATIENCE} 轮无改善，在 Epoch {epoch} 停止")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 打印最终分类报告
    model.eval()
    y_true_final, y_pred_final = [], []
    with torch.no_grad():
        for xb, yb in src_val_loader:
            xb = xb.to(DEVICE)
            class_out, _, _ = model(xb, alpha=0.0)
            preds = class_out.argmax(dim=1).cpu().numpy()
            y_pred_final.extend(preds)
            y_true_final.extend(yb.numpy())

    print("\n  [最终分类报告]")
    print(classification_report(y_true_final, y_pred_final,
                                target_names=['Normal', 'IR', 'OR', 'Ball'],
                                digits=4, zero_division=0))

    return best_acc, best_f1


# =========================================
# 结果汇总
# =========================================
def summarize_results(name, results):
    accs = np.array([r[0] for r in results])
    f1s = np.array([r[1] for r in results])
    print("\n" + "#" * 60)
    print(f"{name} 在 5 次随机划分上的结果：")
    for i, (acc, f1) in enumerate(results, 1):
        print(f"  Run{i}: Acc={acc:.4f}, F1={f1:.4f}")
    print("-" * 60)
    print(f"  Accuracy: mean={accs.mean():.4f}, std={accs.std(ddof=0):.4f}")
    print(f"  Macro F1: mean={f1s.mean():.4f}, std={f1s.std(ddof=0):.4f}")
    print("#" * 60)


# =========================================
# 主函数
# =========================================
def main():
    print("正在加载源域数据 source_x.npy / source_y.npy ...")
    X = np.load("source_x.npy").astype(np.float32)
    y = np.load("source_y.npy").astype(np.int64)
    print(f"数据形状：X = {X.shape}, y = {y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    print("类别分布:", dict(zip(unique, counts)))
    print("类别比例:", {k: f"{v / len(y) * 100:.1f}%" for k, v in zip(unique, counts)})

    mean = X.mean()
    std = X.std()
    X = (X - mean) / (std + 1e-5)

    all_feats = extract_stat_features(X)

    print("正在加载目标域数据 target_data.npy ...")
    target_dict = np.load("target_data.npy", allow_pickle=True).item()
    tgt_list = []
    for k, data in target_dict.items():
        data = data.astype(np.float32)
        data = (data - mean) / (std + 1e-5)
        tgt_list.append(data)
    target_x = np.concatenate(tgt_list, axis=0)
    print(f"目标域数据合并后形状：{target_x.shape}")

    seeds = [0, 1, 2, 3, 4]
    cnn_results = []
    svm_results = []
    dann_results = []

    indices = np.arange(len(y))

    for i, seed in enumerate(seeds, 1):
        print("\n" + "=" * 60)
        print(f"  第 {i} 次随机划分 (random_state={seed})")
        print("=" * 60)

        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=y
        )

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        feats_train, feats_val = all_feats[train_idx], all_feats[val_idx]

        print(f"划分完成：训练集 {len(train_idx)}，验证集 {len(val_idx)}")

        acc_cnn, f1_cnn = train_one_cnn_run(X_train, y_train, X_val, y_val, seed)
        cnn_results.append((acc_cnn, f1_cnn))

        acc_svm, f1_svm = train_one_svm_run(feats_train, y_train, feats_val, y_val, seed)
        svm_results.append((acc_svm, f1_svm))

        # 使用优化版伪标签 DANN
        acc_dann, f1_dann = train_one_dann_run_with_pseudo_v2(X_train, y_train, X_val, y_val, target_x, seed)
        dann_results.append((acc_dann, f1_dann))

    summarize_results("CNN 基线模型", cnn_results)
    summarize_results("统计特征 + SVM 基线模型", svm_results)
    summarize_results("DANN 域自适应模型（优化版伪标签）", dann_results)


if __name__ == "__main__":
    main()
