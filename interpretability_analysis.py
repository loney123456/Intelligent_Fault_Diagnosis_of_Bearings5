# # interpretability_analysis.py
# # 功能：任务4 - DANN模型可解释性分析
# # 包括：Grad-CAM、特征重要性、时频分析(STFT)、包络谱分析
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
# import matplotlib.pyplot as plt
# from scipy.signal import hilbert, butter, filtfilt, stft
# from scipy.fft import fft, fftfreq
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
#
#
# # ===================== 模型定义（与model.py一致）=====================
# class GradientReversalLayer(torch.autograd.Function):
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
# class FeatureExtractor(nn.Module):
#     def __init__(self, input_size=1024):
#         super(FeatureExtractor, self).__init__()
#         self.conv1 = nn.Conv1d(1, 32, kernel_size=64, stride=8, padding=28)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.pool1 = nn.MaxPool1d(2)
#
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=32, stride=4, padding=14)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.pool2 = nn.MaxPool1d(2)
#
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=7)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.pool3 = nn.AdaptiveAvgPool1d(8)
#
#         self.feature_dim = 128 * 8
#
#     def forward(self, x):
#         if x.dim() == 2:
#             x = x.unsqueeze(1)
#
#         x = self.pool1(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool2(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool3(F.relu(self.bn3(self.conv3(x))))
#         x = x.view(x.size(0), -1)
#         return x
#
#
# class Classifier(nn.Module):
#     def __init__(self, feature_dim=1024, num_classes=4):
#         super(Classifier, self).__init__()
#         self.fc1 = nn.Linear(feature_dim, 256)
#         self.bn1 = nn.BatchNorm1d(256)
#         self.dropout1 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(256, 64)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.dropout2 = nn.Dropout(0.3)
#         self.fc3 = nn.Linear(64, num_classes)
#
#     def forward(self, x):
#         x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
#         x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
#         x = self.fc3(x)
#         return x
#
#
# class DANN(nn.Module):
#     def __init__(self, input_size=1024, num_classes=4):
#         super(DANN, self).__init__()
#         self.feature_extractor = FeatureExtractor(input_size)
#         self.classifier = Classifier(self.feature_extractor.feature_dim, num_classes)
#
#     def forward(self, x, alpha=1.0):
#         features = self.feature_extractor(x)
#         class_output = self.classifier(features)
#         return class_output, features
#
#
# # ===================== Grad-CAM实现 =====================
# class GradCAM:
#     """Grad-CAM for 1D CNN"""
#
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
#
#         # 注册钩子
#         target_layer.register_forward_hook(self.save_activation)
#         target_layer.register_full_backward_hook(self.save_gradient)
#
#     def save_activation(self, module, input, output):
#         self.activations = output.detach()
#
#     def save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].detach()
#
#     def generate_cam(self, input_tensor, target_class=None):
#         """生成CAM热力图"""
#         self.model.eval()
#
#         # 前向传播
#         output, _ = self.model(input_tensor)
#
#         if target_class is None:
#             target_class = output.argmax(dim=1)
#
#         # 反向传播
#         self.model.zero_grad()
#         one_hot = torch.zeros_like(output)
#         one_hot[0, target_class] = 1
#         output.backward(gradient=one_hot, retain_graph=True)
#
#         # 计算权重
#         weights = self.gradients.mean(dim=2, keepdim=True)
#
#         # 加权求和
#         cam = (weights * self.activations).sum(dim=1, keepdim=True)
#         cam = F.relu(cam)
#
#         # 归一化
#         cam = cam - cam.min()
#         if cam.max() > 0:
#             cam = cam / cam.max()
#
#         return cam.squeeze().cpu().numpy()
#
#
# # ===================== 时频分析(STFT) =====================
# def compute_stft(signal, fs=12000, nperseg=256, noverlap=None):
#     """计算短时傅里叶变换"""
#     if noverlap is None:
#         noverlap = nperseg // 2
#
#     f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
#     return f, t, np.abs(Zxx)
#
#
# def plot_stft_analysis(signals, labels, predictions, fs=12000, save_path='stft_analysis.png'):
#     """绘制STFT时频分析图"""
#
#     # 选择每类一个典型样本
#     class_names = ['Normal', 'IR', 'OR', 'Ball']
#     unique_preds = np.unique(predictions)
#
#     n_classes = len(unique_preds)
#     fig, axes = plt.subplots(n_classes, 3, figsize=(15, 4 * n_classes))
#
#     if n_classes == 1:
#         axes = axes.reshape(1, -1)
#
#     for idx, pred_class in enumerate(unique_preds):
#         # 找到该类的第一个样本
#         sample_idx = np.where(predictions == pred_class)[0][0]
#         signal = signals[sample_idx]
#
#         # 时域波形
#         ax1 = axes[idx, 0]
#         t_signal = np.arange(len(signal)) / fs * 1000
#         ax1.plot(t_signal, signal, 'b-', linewidth=0.5)
#         ax1.set_title(f'{class_names[pred_class]} - 时域波形', fontsize=11, fontweight='bold')
#         ax1.set_xlabel('时间 (ms)')
#         ax1.set_ylabel('幅值')
#         ax1.grid(True, alpha=0.3)
#
#         # STFT时频图
#         ax2 = axes[idx, 1]
#         f, t, Sxx = compute_stft(signal, fs=fs, nperseg=128)
#         im = ax2.pcolormesh(t * 1000, f, 20 * np.log10(Sxx + 1e-10),
#                             shading='gouraud', cmap='jet')
#         ax2.set_title(f'{class_names[pred_class]} - STFT时频图', fontsize=11, fontweight='bold')
#         ax2.set_xlabel('时间 (ms)')
#         ax2.set_ylabel('频率 (Hz)')
#         ax2.set_ylim(0, fs // 2)
#         plt.colorbar(im, ax=ax2, label='dB')
#
#         # 频谱
#         ax3 = axes[idx, 2]
#         freqs = fftfreq(len(signal), 1 / fs)[:len(signal) // 2]
#         spectrum = np.abs(fft(signal))[:len(signal) // 2] / len(signal) * 2
#         ax3.plot(freqs, spectrum, 'g-', linewidth=0.7)
#         ax3.set_title(f'{class_names[pred_class]} - 频谱', fontsize=11, fontweight='bold')
#         ax3.set_xlabel('频率 (Hz)')
#         ax3.set_ylabel('幅值')
#         ax3.set_xlim(0, 3000)
#         ax3.grid(True, alpha=0.3)
#
#     plt.suptitle('目标域样本时频分析 (STFT)', fontsize=14, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     print(f"已保存: {save_path}")
#     plt.close()
#
#
# # ===================== 包络谱分析 =====================
# def compute_envelope_spectrum(signal, fs=12000, band=(500, 4000)):
#     """计算包络谱"""
#     # 带通滤波
#     nyq = fs / 2
#     low = max(band[0] / nyq, 0.01)
#     high = min(band[1] / nyq, 0.99)
#
#     b, a = butter(4, [low, high], btype='band')
#     filtered = filtfilt(b, a, signal)
#
#     # Hilbert变换求包络
#     analytic = hilbert(filtered)
#     envelope = np.abs(analytic)
#     envelope = envelope - np.mean(envelope)
#
#     # 包络谱
#     n = len(envelope)
#     freqs = fftfreq(n, 1 / fs)[:n // 2]
#     spectrum = np.abs(fft(envelope))[:n // 2] / n * 2
#
#     return freqs, spectrum
#
#
# def plot_envelope_spectrum_analysis(signals, labels, predictions, fs=12000,
#                                     save_path='envelope_spectrum_analysis.png'):
#     """绘制包络谱分析图"""
#
#     class_names = ['Normal', 'IR', 'OR', 'Ball']
#     unique_preds = np.unique(predictions)
#
#     # 故障特征频率（假设转速约1750rpm）
#     fr = 29.17  # 转频
#     BPFO = 107.0
#     BPFI = 162.0
#     BSF = 70.0
#
#     n_classes = len(unique_preds)
#     fig, axes = plt.subplots(n_classes, 2, figsize=(14, 4 * n_classes))
#
#     if n_classes == 1:
#         axes = axes.reshape(1, -1)
#
#     for idx, pred_class in enumerate(unique_preds):
#         sample_idx = np.where(predictions == pred_class)[0][0]
#         signal = signals[sample_idx]
#
#         # 时域包络
#         ax1 = axes[idx, 0]
#
#         # 计算包络
#         nyq = fs / 2
#         b, a = butter(4, [500 / nyq, 4000 / nyq], btype='band')
#         filtered = filtfilt(b, a, signal)
#         envelope = np.abs(hilbert(filtered))
#
#         t = np.arange(len(signal)) / fs * 1000
#         ax1.plot(t, signal, 'b-', linewidth=0.5, alpha=0.5, label='原始信号')
#         ax1.plot(t, envelope, 'r-', linewidth=1, label='包络')
#         ax1.set_title(f'{class_names[pred_class]} - 信号与包络', fontsize=11, fontweight='bold')
#         ax1.set_xlabel('时间 (ms)')
#         ax1.set_ylabel('幅值')
#         ax1.legend(loc='upper right', fontsize=8)
#         ax1.grid(True, alpha=0.3)
#
#         # 包络谱
#         ax2 = axes[idx, 1]
#         freqs, spectrum = compute_envelope_spectrum(signal, fs)
#
#         ax2.plot(freqs, spectrum, 'r-', linewidth=0.7)
#
#         # 标注故障频率
#         ax2.axvline(BPFO, color='blue', linestyle='--', alpha=0.7, label=f'BPFO={BPFO:.0f}Hz')
#         ax2.axvline(BPFI, color='orange', linestyle='--', alpha=0.7, label=f'BPFI={BPFI:.0f}Hz')
#         ax2.axvline(2 * BSF, color='purple', linestyle='--', alpha=0.7, label=f'2BSF={2 * BSF:.0f}Hz')
#
#         ax2.set_title(f'{class_names[pred_class]} - 包络谱', fontsize=11, fontweight='bold')
#         ax2.set_xlabel('频率 (Hz)')
#         ax2.set_ylabel('幅值')
#         ax2.set_xlim(0, 400)
#         ax2.legend(loc='upper right', fontsize=7)
#         ax2.grid(True, alpha=0.3)
#
#     plt.suptitle('目标域样本包络谱分析', fontsize=14, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     print(f"已保存: {save_path}")
#     plt.close()
#
#
# # ===================== Grad-CAM可视化 =====================
# def plot_gradcam_analysis(model, signals, predictions, device,
#                           save_path='gradcam_analysis.png'):
#     """绘制Grad-CAM可解释性分析图"""
#
#     class_names = ['Normal', 'IR', 'OR', 'Ball']
#     unique_preds = np.unique(predictions)
#
#     # 获取目标层（最后一个卷积层）
#     target_layer = model.feature_extractor.conv3
#     gradcam = GradCAM(model, target_layer)
#
#     n_classes = len(unique_preds)
#     fig, axes = plt.subplots(n_classes, 2, figsize=(14, 4 * n_classes))
#
#     if n_classes == 1:
#         axes = axes.reshape(1, -1)
#
#     for idx, pred_class in enumerate(unique_preds):
#         sample_idx = np.where(predictions == pred_class)[0][0]
#         signal = signals[sample_idx]
#
#         # 准备输入
#         input_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0).to(device)
#
#         # 生成CAM
#         try:
#             cam = gradcam.generate_cam(input_tensor, pred_class)
#
#             # 将CAM插值到原始信号长度
#             cam_interp = np.interp(
#                 np.linspace(0, 1, len(signal)),
#                 np.linspace(0, 1, len(cam)),
#                 cam
#             )
#         except Exception as e:
#             print(f"Grad-CAM计算失败: {e}")
#             cam_interp = np.zeros(len(signal))
#
#         # 绘制原始信号
#         ax1 = axes[idx, 0]
#         t = np.arange(len(signal))
#         ax1.plot(t, signal, 'b-', linewidth=0.5)
#         ax1.set_title(f'{class_names[pred_class]} - 原始信号', fontsize=11, fontweight='bold')
#         ax1.set_xlabel('采样点')
#         ax1.set_ylabel('幅值')
#         ax1.grid(True, alpha=0.3)
#
#         # 绘制带CAM热力图的信号
#         ax2 = axes[idx, 1]
#
#         # 创建颜色映射
#         colors = plt.cm.jet(cam_interp)
#
#         # 绘制彩色信号
#         for i in range(len(signal) - 1):
#             ax2.plot([t[i], t[i + 1]], [signal[i], signal[i + 1]],
#                      color=colors[i], linewidth=1)
#
#         # 添加颜色条
#         sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
#         sm.set_array([])
#         cbar = plt.colorbar(sm, ax=ax2)
#         cbar.set_label('重要性', fontsize=9)
#
#         ax2.set_title(f'{class_names[pred_class]} - Grad-CAM热力图', fontsize=11, fontweight='bold')
#         ax2.set_xlabel('采样点')
#         ax2.set_ylabel('幅值')
#         ax2.grid(True, alpha=0.3)
#
#     plt.suptitle('Grad-CAM可解释性分析\n(颜色越红表示对分类越重要)', fontsize=14, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     print(f"已保存: {save_path}")
#     plt.close()
#
#
# # ===================== 特征重要性分析 =====================
# def plot_feature_importance(model, signals, predictions, device,
#                             save_path='feature_importance_analysis.png'):
#     """通过扰动分析特征重要性"""
#
#     model.eval()
#     class_names = ['Normal', 'IR', 'OR', 'Ball']
#
#     # 选择一个样本进行分析
#     sample_idx = 0
#     signal = signals[sample_idx]
#     pred_class = predictions[sample_idx]
#
#     # 将信号分成多个区间
#     n_segments = 16
#     segment_len = len(signal) // n_segments
#
#     # 计算原始预测概率
#     input_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output, _ = model(input_tensor)
#         orig_prob = F.softmax(output, dim=1)[0, pred_class].item()
#
#     # 扰动每个区间，计算重要性
#     importance = []
#     for i in range(n_segments):
#         perturbed = signal.copy()
#         start = i * segment_len
#         end = min((i + 1) * segment_len, len(signal))
#
#         # 用噪声替换该区间
#         perturbed[start:end] = np.random.randn(end - start) * np.std(signal)
#
#         input_tensor = torch.FloatTensor(perturbed).unsqueeze(0).unsqueeze(0).to(device)
#         with torch.no_grad():
#             output, _ = model(input_tensor)
#             new_prob = F.softmax(output, dim=1)[0, pred_class].item()
#
#         # 重要性 = 概率下降程度
#         importance.append(orig_prob - new_prob)
#
#     importance = np.array(importance)
#     importance = np.maximum(importance, 0)  # 只保留正向影响
#     if importance.max() > 0:
#         importance = importance / importance.max()
#
#     # 绘图
#     fig, axes = plt.subplots(2, 1, figsize=(14, 8))
#
#     # 原始信号
#     ax1 = axes[0]
#     t = np.arange(len(signal))
#     ax1.plot(t, signal, 'b-', linewidth=0.5)
#
#     # 标注区间重要性
#     for i in range(n_segments):
#         start = i * segment_len
#         end = min((i + 1) * segment_len, len(signal))
#         color = plt.cm.Reds(importance[i])
#         ax1.axvspan(start, end, alpha=0.3, color=color)
#
#     ax1.set_title(f'样本信号与区间重要性 (预测: {class_names[pred_class]})',
#                   fontsize=12, fontweight='bold')
#     ax1.set_xlabel('采样点')
#     ax1.set_ylabel('幅值')
#     ax1.grid(True, alpha=0.3)
#
#     # 重要性柱状图
#     ax2 = axes[1]
#     x = np.arange(n_segments)
#     colors = plt.cm.Reds(importance)
#     bars = ax2.bar(x, importance, color=colors, edgecolor='black', linewidth=0.5)
#     ax2.set_title('各区间对分类的重要性', fontsize=12, fontweight='bold')
#     ax2.set_xlabel('区间编号')
#     ax2.set_ylabel('重要性得分')
#     ax2.set_xticks(x)
#     ax2.set_xticklabels([f'{i + 1}' for i in x])
#     ax2.grid(True, alpha=0.3, axis='y')
#
#     # 添加颜色条
#     sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(0, 1))
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
#     cbar.set_label('重要性', fontsize=10)
#
#     plt.suptitle('特征重要性分析 (扰动法)', fontsize=14, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 0.95, 0.96])
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     print(f"已保存: {save_path}")
#     plt.close()
#
#
# # ===================== 综合可解释性报告 =====================
# def plot_interpretability_summary(signals, predictions, confidences,
#                                   save_path='interpretability_summary.png'):
#     """生成综合可解释性报告"""
#
#     class_names = ['Normal', 'IR', 'OR', 'Ball']
#
#     fig = plt.figure(figsize=(16, 12))
#
#     # 1. 预测分布饼图
#     ax1 = fig.add_subplot(2, 3, 1)
#     unique, counts = np.unique(predictions, return_counts=True)
#     colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
#     labels = [class_names[i] for i in unique]
#     ax1.pie(counts, labels=labels, colors=[colors[i] for i in unique],
#             autopct='%1.1f%%', startangle=90)
#     ax1.set_title('预测类别分布', fontsize=12, fontweight='bold')
#
#     # 2. 置信度分布直方图
#     ax2 = fig.add_subplot(2, 3, 2)
#     ax2.hist(confidences, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
#     ax2.axvline(np.mean(confidences), color='red', linestyle='--',
#                 label=f'均值: {np.mean(confidences):.3f}')
#     ax2.set_title('预测置信度分布', fontsize=12, fontweight='bold')
#     ax2.set_xlabel('置信度')
#     ax2.set_ylabel('样本数')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
#
#     # 3. 各类别置信度箱线图
#     ax3 = fig.add_subplot(2, 3, 3)
#     conf_by_class = [confidences[predictions == i] for i in range(4) if np.sum(predictions == i) > 0]
#     labels_box = [class_names[i] for i in range(4) if np.sum(predictions == i) > 0]
#     bp = ax3.boxplot(conf_by_class, labels=labels_box, patch_artist=True)
#     for patch, color in zip(bp['boxes'], [colors[i] for i in range(4) if np.sum(predictions == i) > 0]):
#         patch.set_facecolor(color)
#         patch.set_alpha(0.6)
#     ax3.set_title('各类别置信度分布', fontsize=12, fontweight='bold')
#     ax3.set_ylabel('置信度')
#     ax3.grid(True, alpha=0.3, axis='y')
#
#     # 4-6. 各类别典型信号
#     for idx, pred_class in enumerate(np.unique(predictions)[:3]):
#         ax = fig.add_subplot(2, 3, 4 + idx)
#
#         # 找到该类最高置信度的样本
#         class_mask = predictions == pred_class
#         class_conf = confidences[class_mask]
#         class_signals = signals[class_mask]
#
#         if len(class_conf) > 0:
#             best_idx = np.argmax(class_conf)
#             signal = class_signals[best_idx]
#             conf = class_conf[best_idx]
#
#             t = np.arange(len(signal))
#             ax.plot(t, signal, color=colors[pred_class], linewidth=0.5)
#             ax.set_title(f'{class_names[pred_class]} 典型样本\n置信度: {conf:.3f}',
#                          fontsize=11, fontweight='bold')
#             ax.set_xlabel('采样点')
#             ax.set_ylabel('幅值')
#             ax.grid(True, alpha=0.3)
#
#     plt.suptitle('DANN模型可解释性分析报告', fontsize=16, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     print(f"已保存: {save_path}")
#     plt.close()
#
#
# # ===================== 主函数 =====================
# def main():
#     print("=" * 70)
#     print("任务4 - DANN模型可解释性分析")
#     print("=" * 70)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"使用设备: {device}")
#
#     # 1. 加载模型
#     print("\n[1/6] 加载模型...")
#     model = DANN(input_size=1024, num_classes=4).to(device)
#
#     model_path = 'dann_model_best_full.pth'
#     if os.path.exists(model_path):
#         checkpoint = torch.load(model_path, map_location=device)
#         if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['model_state_dict'])
#         else:
#             model.load_state_dict(checkpoint)
#         print(f"已加载模型: {model_path}")
#     else:
#         print(f"警告: 未找到模型文件 {model_path}，使用随机初始化")
#
#     model.eval()
#
#     # 2. 加载目标域数据
#     print("\n[2/6] 加载目标域数据...")
#     target_data = np.load('target_data.npy')
#     print(f"目标域数据形状: {target_data.shape}")
#
#     # 处理数据（分段）
#     sample_len = 1024
#     n_files = target_data.shape[0]
#
#     all_signals = []
#     all_predictions = []
#     all_confidences = []
#
#     for i in range(n_files):
#         file_data = target_data[i]
#         n_samples = len(file_data) // sample_len
#
#         for j in range(min(n_samples, 10)):  # 每个文件最多取10段
#             segment = file_data[j * sample_len: (j + 1) * sample_len]
#             if len(segment) == sample_len:
#                 all_signals.append(segment)
#
#     all_signals = np.array(all_signals)
#     print(f"处理后样本数: {len(all_signals)}")
#
#     # 3. 模型预测
#     print("\n[3/6] 模型预测...")
#     with torch.no_grad():
#         for i in range(0, len(all_signals), 32):
#             batch = all_signals[i:i + 32]
#             inputs = torch.FloatTensor(batch).unsqueeze(1).to(device)
#             outputs, _ = model(inputs)
#             probs = F.softmax(outputs, dim=1)
#             preds = outputs.argmax(dim=1).cpu().numpy()
#             confs = probs.max(dim=1)[0].cpu().numpy()
#
#             all_predictions.extend(preds)
#             all_confidences.extend(confs)
#
#     all_predictions = np.array(all_predictions)
#     all_confidences = np.array(all_confidences)
#
#     class_names = ['Normal', 'IR', 'OR', 'Ball']
#     print("\n预测结果统计:")
#     for i in range(4):
#         count = np.sum(all_predictions == i)
#         if count > 0:
#             mean_conf = np.mean(all_confidences[all_predictions == i])
#             print(f"  {class_names[i]}: {count} 样本, 平均置信度: {mean_conf:.3f}")
#
#     # 4. 时频分析(STFT)
#     print("\n[4/6] 生成STFT时频分析图...")
#     plot_stft_analysis(all_signals, None, all_predictions, fs=12000,
#                        save_path='stft_analysis.png')
#
#     # 5. 包络谱分析
#     print("\n[5/6] 生成包络谱分析图...")
#     plot_envelope_spectrum_analysis(all_signals, None, all_predictions, fs=12000,
#                                     save_path='envelope_spectrum_analysis.png')
#
#     # 6. Grad-CAM分析
#     print("\n[6/6] 生成Grad-CAM可解释性分析...")
#     plot_gradcam_analysis(model, all_signals, all_predictions, device,
#                           save_path='gradcam_analysis.png')
#
#     # 7. 特征重要性分析
#     print("\n[7/7] 生成特征重要性分析...")
#     plot_feature_importance(model, all_signals, all_predictions, device,
#                             save_path='feature_importance_analysis.png')
#
#     # 8. 综合报告
#     print("\n[8/8] 生成综合可解释性报告...")
#     plot_interpretability_summary(all_signals, all_predictions, all_confidences,
#                                   save_path='interpretability_summary.png')
#
#     # 总结
#     print("\n" + "=" * 70)
#     print("可解释性分析完成!")
#     print("=" * 70)
#     print("""
# 生成的文件:
#   1. stft_analysis.png              - STFT时频分析图
#   2. envelope_spectrum_analysis.png - 包络谱分析图
#   3. gradcam_analysis.png           - Grad-CAM热力图
#   4. feature_importance_analysis.png - 特征重要性分析
#   5. interpretability_summary.png   - 综合可解释性报告
# """)
#
#
# if __name__ == "__main__":
#     main()


# interpretability_analysis.py
# 功能：任务4 - DANN模型可解释性分析（修复版）
# ====================================================================

# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from scipy.signal import hilbert, butter, filtfilt, stft
# from scipy.fft import fft, fftfreq
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
#
#
# def check_model_file():
#     """检查模型文件的格式"""
#     model_files = ['dann_model_best_full.pth', 'dann_model_best.pth',
#                    'dann_model_final.pth', 'dann_model.pth']
#
#     for model_file in model_files:
#         if os.path.exists(model_file):
#             print(f"\n检查模型文件: {model_file}")
#             checkpoint = torch.load(model_file, map_location='cpu')
#
#             if isinstance(checkpoint, dict):
#                 print(f"  类型: 字典")
#                 print(f"  键: {list(checkpoint.keys())}")
#
#                 if 'state_dict' in checkpoint:
#                     state_dict = checkpoint['state_dict']
#                     print(f"  state_dict键数量: {len(state_dict)}")
#                     print(f"  前5个键: {list(state_dict.keys())[:5]}")
#                 elif 'model_state_dict' in checkpoint:
#                     state_dict = checkpoint['model_state_dict']
#                     print(f"  model_state_dict键数量: {len(state_dict)}")
#                     print(f"  前5个键: {list(state_dict.keys())[:5]}")
#                 else:
#                     # 可能直接就是state_dict
#                     print(f"  直接state_dict键数量: {len(checkpoint)}")
#                     print(f"  前5个键: {list(checkpoint.keys())[:5]}")
#             else:
#                 print(f"  类型: {type(checkpoint)}")
#
#             return model_file, checkpoint
#
#     print("未找到任何模型文件!")
#     return None, None
#
#
# def build_model_from_state_dict(state_dict):
#     """根据state_dict的键推断并构建模型"""
#
#     # 分析state_dict的结构
#     keys = list(state_dict.keys())
#     print(f"\n分析模型结构，共 {len(keys)} 个参数")
#
#     # 检测是否有特定前缀
#     has_feature_extractor = any('feature_extractor' in k for k in keys)
#     has_encoder = any('encoder' in k for k in keys)
#     has_conv = any('conv' in k for k in keys)
#
#     print(f"  has_feature_extractor: {has_feature_extractor}")
#     print(f"  has_encoder: {has_encoder}")
#     print(f"  has_conv: {has_conv}")
#
#     # 打印所有键以便调试
#     print("\n所有参数键:")
#     for k in keys:
#         print(f"  {k}: {state_dict[k].shape}")
#
#     return keys
#
#
# # ===================== 尝试从model.py导入 =====================
# try:
#     from model import DANN, FeatureExtractor, Classifier
#
#     print("成功从 model.py 导入模型定义")
#     USE_LOCAL_MODEL = True
# except ImportError:
#     print("无法从 model.py 导入，将使用内置模型定义")
#     USE_LOCAL_MODEL = False
#
# # ===================== 内置模型定义（备用）=====================
# if not USE_LOCAL_MODEL:
#     class GradientReversalFunction(torch.autograd.Function):
#         @staticmethod
#         def forward(ctx, x, alpha):
#             ctx.alpha = alpha
#             return x.view_as(x)
#
#         @staticmethod
#         def backward(ctx, grad_output):
#             return grad_output.neg() * ctx.alpha, None
#
#
#     class FeatureExtractor(nn.Module):
#         def __init__(self, input_size=1024):
#             super(FeatureExtractor, self).__init__()
#             self.conv1 = nn.Conv1d(1, 32, kernel_size=64, stride=8, padding=28)
#             self.bn1 = nn.BatchNorm1d(32)
#             self.pool1 = nn.MaxPool1d(2)
#
#             self.conv2 = nn.Conv1d(32, 64, kernel_size=32, stride=4, padding=14)
#             self.bn2 = nn.BatchNorm1d(64)
#             self.pool2 = nn.MaxPool1d(2)
#
#             self.conv3 = nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=7)
#             self.bn3 = nn.BatchNorm1d(128)
#             self.pool3 = nn.AdaptiveAvgPool1d(8)
#
#             self.feature_dim = 128 * 8
#
#         def forward(self, x):
#             if x.dim() == 2:
#                 x = x.unsqueeze(1)
#             x = self.pool1(F.relu(self.bn1(self.conv1(x))))
#             x = self.pool2(F.relu(self.bn2(self.conv2(x))))
#             x = self.pool3(F.relu(self.bn3(self.conv3(x))))
#             x = x.view(x.size(0), -1)
#             return x
#
#
#     class Classifier(nn.Module):
#         def __init__(self, feature_dim=1024, num_classes=4):
#             super(Classifier, self).__init__()
#             self.fc1 = nn.Linear(feature_dim, 256)
#             self.bn1 = nn.BatchNorm1d(256)
#             self.dropout1 = nn.Dropout(0.5)
#             self.fc2 = nn.Linear(256, 64)
#             self.bn2 = nn.BatchNorm1d(64)
#             self.dropout2 = nn.Dropout(0.3)
#             self.fc3 = nn.Linear(64, num_classes)
#
#         def forward(self, x):
#             x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
#             x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
#             x = self.fc3(x)
#             return x
#
#
#     class DANN(nn.Module):
#         def __init__(self, input_size=1024, num_classes=4):
#             super(DANN, self).__init__()
#             self.feature_extractor = FeatureExtractor(input_size)
#             self.classifier = Classifier(self.feature_extractor.feature_dim, num_classes)
#
#         def forward(self, x, alpha=1.0):
#             features = self.feature_extractor(x)
#             class_output = self.classifier(features)
#             return class_output, features
#
#
# def load_model_smart(device):
#     """智能加载模型，处理各种保存格式"""
#
#     model_files = ['dann_model_best_full.pth', 'dann_model_best.pth',
#                    'dann_model_final.pth', 'dann_model.pth']
#
#     for model_file in model_files:
#         if not os.path.exists(model_file):
#             continue
#
#         print(f"\n尝试加载: {model_file}")
#         checkpoint = torch.load(model_file, map_location=device)
#
#         # 解析checkpoint
#         if isinstance(checkpoint, dict):
#             if 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#                 num_classes = checkpoint.get('num_classes', 4)
#                 print(f"  从 'state_dict' 键加载，num_classes={num_classes}")
#             elif 'model_state_dict' in checkpoint:
#                 state_dict = checkpoint['model_state_dict']
#                 num_classes = checkpoint.get('num_classes', 4)
#                 print(f"  从 'model_state_dict' 键加载，num_classes={num_classes}")
#             else:
#                 # 检查是否直接是state_dict
#                 if any('conv' in k or 'fc' in k or 'weight' in k for k in checkpoint.keys()):
#                     state_dict = checkpoint
#                     num_classes = 4
#                     print(f"  直接作为state_dict加载")
#                 else:
#                     print(f"  未知格式，跳过")
#                     continue
#         else:
#             print(f"  非字典格式，跳过")
#             continue
#
#         # 分析state_dict结构来确定模型架构
#         keys = list(state_dict.keys())
#         print(f"  参数数量: {len(keys)}")
#         print(f"  前3个键: {keys[:3]}")
#
#         # 创建模型并加载权重
#         try:
#             model = DANN(input_size=1024, num_classes=num_classes).to(device)
#             model.load_state_dict(state_dict, strict=False)
#             print(f"  ✓ 模型加载成功!")
#             return model, True
#         except Exception as e:
#             print(f"  加载失败: {e}")
#
#             # 尝试修改键名后加载
#             try:
#                 new_state_dict = {}
#                 for k, v in state_dict.items():
#                     # 移除可能的前缀
#                     new_key = k.replace('module.', '')
#                     new_state_dict[new_key] = v
#
#                 model = DANN(input_size=1024, num_classes=num_classes).to(device)
#                 model.load_state_dict(new_state_dict, strict=False)
#                 print(f"  ✓ 修改键名后加载成功!")
#                 return model, True
#             except Exception as e2:
#                 print(f"  修改键名后仍失败: {e2}")
#
#     # 如果都失败，返回未训练的模型
#     print("\n警告: 所有模型文件加载失败，使用随机初始化的模型")
#     model = DANN(input_size=1024, num_classes=4).to(device)
#     return model, False
#
#
# # ===================== Grad-CAM =====================
# class GradCAM1D:
#     """1D卷积网络的Grad-CAM实现"""
#
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
#
#         self._register_hooks()
#
#     def _register_hooks(self):
#         def forward_hook(module, input, output):
#             self.activations = output.detach()
#
#         def backward_hook(module, grad_input, grad_output):
#             self.gradients = grad_output[0].detach()
#
#         self.target_layer.register_forward_hook(forward_hook)
#         self.target_layer.register_full_backward_hook(backward_hook)
#
#     def generate(self, input_tensor, target_class=None):
#         self.model.eval()
#         self.model.zero_grad()
#
#         output, _ = self.model(input_tensor)
#
#         if target_class is None:
#             target_class = output.argmax(dim=1).item()
#
#         one_hot = torch.zeros_like(output)
#         one_hot[0, target_class] = 1
#         output.backward(gradient=one_hot)
#
#         if self.gradients is None or self.activations is None:
#             return np.zeros(input_tensor.shape[-1])
#
#         weights = self.gradients.mean(dim=2, keepdim=True)
#         cam = (weights * self.activations).sum(dim=1, keepdim=True)
#         cam = F.relu(cam)
#
#         cam = cam.squeeze().cpu().numpy()
#         if cam.max() > 0:
#             cam = (cam - cam.min()) / (cam.max() - cam.min())
#
#         return cam
#
#
# # ===================== STFT时频分析 =====================
# def compute_stft(signal, fs=12000, nperseg=128, noverlap=None):
#     """计算STFT"""
#     if noverlap is None:
#         noverlap = nperseg // 2
#     f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
#     return f, t, np.abs(Zxx)
#
#
# def plot_stft_analysis(signals, predictions, fs=12000, save_path='stft_analysis.png'):
#     """STFT时频分析图"""
#     class_names = ['Normal', 'IR', 'OR', 'Ball']
#     unique_preds = np.unique(predictions)
#
#     n_classes = len(unique_preds)
#     fig, axes = plt.subplots(n_classes, 3, figsize=(15, 4 * n_classes))
#
#     if n_classes == 1:
#         axes = axes.reshape(1, -1)
#
#     for idx, pred_class in enumerate(unique_preds):
#         sample_idx = np.where(predictions == pred_class)[0][0]
#         signal = signals[sample_idx]
#
#         # 时域波形
#         ax1 = axes[idx, 0]
#         t_signal = np.arange(len(signal)) / fs * 1000
#         ax1.plot(t_signal, signal, 'b-', linewidth=0.5)
#         ax1.set_title(f'{class_names[pred_class]} - 时域波形', fontsize=11, fontweight='bold')
#         ax1.set_xlabel('时间 (ms)')
#         ax1.set_ylabel('幅值')
#         ax1.grid(True, alpha=0.3)
#
#         # STFT
#         ax2 = axes[idx, 1]
#         f, t, Sxx = compute_stft(signal, fs=fs)
#         im = ax2.pcolormesh(t * 1000, f, 20 * np.log10(Sxx + 1e-10),
#                             shading='gouraud', cmap='jet')
#         ax2.set_title(f'{class_names[pred_class]} - STFT时频图', fontsize=11, fontweight='bold')
#         ax2.set_xlabel('时间 (ms)')
#         ax2.set_ylabel('频率 (Hz)')
#         ax2.set_ylim(0, fs // 2)
#         plt.colorbar(im, ax=ax2, label='dB')
#
#         # 频谱
#         ax3 = axes[idx, 2]
#         freqs = fftfreq(len(signal), 1 / fs)[:len(signal) // 2]
#         spectrum = np.abs(fft(signal))[:len(signal) // 2] / len(signal) * 2
#         ax3.plot(freqs, spectrum, 'g-', linewidth=0.7)
#         ax3.set_title(f'{class_names[pred_class]} - 频谱', fontsize=11, fontweight='bold')
#         ax3.set_xlabel('频率 (Hz)')
#         ax3.set_ylabel('幅值')
#         ax3.set_xlim(0, 3000)
#         ax3.grid(True, alpha=0.3)
#
#     plt.suptitle('目标域样本时频分析 (STFT)', fontsize=14, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     print(f"已保存: {save_path}")
#     plt.close()
#
#
# # ===================== 包络谱分析 =====================
# def compute_envelope_spectrum(signal, fs=12000, band=(500, 4000)):
#     """计算包络谱"""
#     nyq = fs / 2
#     low = max(band[0] / nyq, 0.01)
#     high = min(band[1] / nyq, 0.99)
#
#     b, a = butter(4, [low, high], btype='band')
#     filtered = filtfilt(b, a, signal)
#
#     envelope = np.abs(hilbert(filtered))
#     envelope = envelope - np.mean(envelope)
#
#     n = len(envelope)
#     freqs = fftfreq(n, 1 / fs)[:n // 2]
#     spectrum = np.abs(fft(envelope))[:n // 2] / n * 2
#
#     return freqs, spectrum
#
#
# def plot_envelope_analysis(signals, predictions, fs=12000,
#                            save_path='envelope_spectrum_analysis.png'):
#     """包络谱分析图"""
#     class_names = ['Normal', 'IR', 'OR', 'Ball']
#     unique_preds = np.unique(predictions)
#
#     # 故障特征频率
#     BPFO, BPFI, BSF = 107.0, 162.0, 70.0
#
#     n_classes = len(unique_preds)
#     fig, axes = plt.subplots(n_classes, 2, figsize=(14, 4 * n_classes))
#
#     if n_classes == 1:
#         axes = axes.reshape(1, -1)
#
#     for idx, pred_class in enumerate(unique_preds):
#         sample_idx = np.where(predictions == pred_class)[0][0]
#         signal = signals[sample_idx]
#
#         # 信号与包络
#         ax1 = axes[idx, 0]
#         nyq = fs / 2
#         b, a = butter(4, [500 / nyq, 4000 / nyq], btype='band')
#         filtered = filtfilt(b, a, signal)
#         envelope = np.abs(hilbert(filtered))
#
#         t = np.arange(len(signal)) / fs * 1000
#         ax1.plot(t, signal, 'b-', linewidth=0.5, alpha=0.5, label='原始信号')
#         ax1.plot(t, envelope, 'r-', linewidth=1, label='包络')
#         ax1.set_title(f'{class_names[pred_class]} - 信号与包络', fontsize=11, fontweight='bold')
#         ax1.set_xlabel('时间 (ms)')
#         ax1.set_ylabel('幅值')
#         ax1.legend(loc='upper right', fontsize=8)
#         ax1.grid(True, alpha=0.3)
#
#         # 包络谱
#         ax2 = axes[idx, 1]
#         freqs, spectrum = compute_envelope_spectrum(signal, fs)
#         ax2.plot(freqs, spectrum, 'r-', linewidth=0.7)
#         ax2.axvline(BPFO, color='blue', linestyle='--', alpha=0.7, label=f'BPFO={BPFO:.0f}Hz')
#         ax2.axvline(BPFI, color='orange', linestyle='--', alpha=0.7, label=f'BPFI={BPFI:.0f}Hz')
#         ax2.axvline(2 * BSF, color='purple', linestyle='--', alpha=0.7, label=f'2BSF={2 * BSF:.0f}Hz')
#         ax2.set_title(f'{class_names[pred_class]} - 包络谱', fontsize=11, fontweight='bold')
#         ax2.set_xlabel('频率 (Hz)')
#         ax2.set_ylabel('幅值')
#         ax2.set_xlim(0, 400)
#         ax2.legend(loc='upper right', fontsize=7)
#         ax2.grid(True, alpha=0.3)
#
#     plt.suptitle('目标域样本包络谱分析', fontsize=14, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     print(f"已保存: {save_path}")
#     plt.close()
#
#
# # ===================== Grad-CAM可视化 =====================
# def plot_gradcam(model, signals, predictions, device, save_path='gradcam_analysis.png'):
#     """Grad-CAM热力图"""
#     class_names = ['Normal', 'IR', 'OR', 'Ball']
#     unique_preds = np.unique(predictions)
#
#     # 获取目标层
#     try:
#         if hasattr(model, 'feature_extractor'):
#             if hasattr(model.feature_extractor, 'conv3'):
#                 target_layer = model.feature_extractor.conv3
#             elif hasattr(model.feature_extractor, 'conv_layers'):
#                 target_layer = model.feature_extractor.conv_layers[-1]
#             else:
#                 # 找最后一个卷积层
#                 conv_layers = [m for m in model.feature_extractor.modules() if isinstance(m, nn.Conv1d)]
#                 target_layer = conv_layers[-1] if conv_layers else None
#         else:
#             conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
#             target_layer = conv_layers[-1] if conv_layers else None
#     except:
#         target_layer = None
#
#     if target_layer is None:
#         print("警告: 无法找到卷积层，跳过Grad-CAM")
#         return
#
#     gradcam = GradCAM1D(model, target_layer)
#
#     n_classes = len(unique_preds)
#     fig, axes = plt.subplots(n_classes, 2, figsize=(14, 4 * n_classes))
#
#     if n_classes == 1:
#         axes = axes.reshape(1, -1)
#
#     for idx, pred_class in enumerate(unique_preds):
#         sample_idx = np.where(predictions == pred_class)[0][0]
#         signal = signals[sample_idx]
#
#         input_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0).to(device)
#
#         try:
#             cam = gradcam.generate(input_tensor, pred_class)
#             cam_interp = np.interp(np.linspace(0, 1, len(signal)),
#                                    np.linspace(0, 1, len(cam)), cam)
#         except:
#             cam_interp = np.zeros(len(signal))
#
#         # 原始信号
#         ax1 = axes[idx, 0]
#         ax1.plot(signal, 'b-', linewidth=0.5)
#         ax1.set_title(f'{class_names[pred_class]} - 原始信号', fontsize=11, fontweight='bold')
#         ax1.set_xlabel('采样点')
#         ax1.set_ylabel('幅值')
#         ax1.grid(True, alpha=0.3)
#
#         # CAM热力图
#         ax2 = axes[idx, 1]
#         colors = plt.cm.jet(cam_interp)
#         for i in range(len(signal) - 1):
#             ax2.plot([i, i + 1], [signal[i], signal[i + 1]], color=colors[i], linewidth=1)
#
#         sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
#         plt.colorbar(sm, ax=ax2, label='重要性')
#         ax2.set_title(f'{class_names[pred_class]} - Grad-CAM', fontsize=11, fontweight='bold')
#         ax2.set_xlabel('采样点')
#         ax2.set_ylabel('幅值')
#         ax2.grid(True, alpha=0.3)
#
#     plt.suptitle('Grad-CAM可解释性分析', fontsize=14, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     print(f"已保存: {save_path}")
#     plt.close()
#
#
# # ===================== 特征重要性（扰动法）=====================
# def plot_feature_importance(model, signals, predictions, device,
#                             save_path='feature_importance_analysis.png'):
#     """扰动法特征重要性"""
#     model.eval()
#     class_names = ['Normal', 'IR', 'OR', 'Ball']
#
#     sample_idx = 0
#     signal = signals[sample_idx]
#     pred_class = predictions[sample_idx]
#
#     n_segments = 16
#     segment_len = len(signal) // n_segments
#
#     input_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output, _ = model(input_tensor)
#         orig_prob = F.softmax(output, dim=1)[0, pred_class].item()
#
#     importance = []
#     for i in range(n_segments):
#         perturbed = signal.copy()
#         start, end = i * segment_len, min((i + 1) * segment_len, len(signal))
#         perturbed[start:end] = np.random.randn(end - start) * np.std(signal)
#
#         input_tensor = torch.FloatTensor(perturbed).unsqueeze(0).unsqueeze(0).to(device)
#         with torch.no_grad():
#             output, _ = model(input_tensor)
#             new_prob = F.softmax(output, dim=1)[0, pred_class].item()
#
#         importance.append(max(orig_prob - new_prob, 0))
#
#     importance = np.array(importance)
#     if importance.max() > 0:
#         importance = importance / importance.max()
#
#     fig, axes = plt.subplots(2, 1, figsize=(14, 8))
#
#     ax1 = axes[0]
#     ax1.plot(signal, 'b-', linewidth=0.5)
#     for i in range(n_segments):
#         start, end = i * segment_len, min((i + 1) * segment_len, len(signal))
#         ax1.axvspan(start, end, alpha=0.3, color=plt.cm.Reds(importance[i]))
#     ax1.set_title(f'样本信号与区间重要性 (预测: {class_names[pred_class]})', fontsize=12, fontweight='bold')
#     ax1.set_xlabel('采样点')
#     ax1.set_ylabel('幅值')
#     ax1.grid(True, alpha=0.3)
#
#     ax2 = axes[1]
#     bars = ax2.bar(range(n_segments), importance, color=[plt.cm.Reds(v) for v in importance],
#                    edgecolor='black', linewidth=0.5)
#     ax2.set_title('各区间对分类的重要性', fontsize=12, fontweight='bold')
#     ax2.set_xlabel('区间编号')
#     ax2.set_ylabel('重要性得分')
#     ax2.set_xticks(range(n_segments))
#     ax2.grid(True, alpha=0.3, axis='y')
#
#     plt.suptitle('特征重要性分析 (扰动法)', fontsize=14, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     print(f"已保存: {save_path}")
#     plt.close()
#
#
# # ===================== 综合报告 =====================
# def plot_summary(signals, predictions, confidences, save_path='interpretability_summary.png'):
#     """综合可解释性报告"""
#     class_names = ['Normal', 'IR', 'OR', 'Ball']
#     colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
#
#     fig = plt.figure(figsize=(16, 12))
#
#     # 1. 饼图
#     ax1 = fig.add_subplot(2, 3, 1)
#     unique, counts = np.unique(predictions, return_counts=True)
#     ax1.pie(counts, labels=[class_names[i] for i in unique],
#             colors=[colors[i] for i in unique], autopct='%1.1f%%')
#     ax1.set_title('预测类别分布', fontsize=12, fontweight='bold')
#
#     # 2. 置信度分布
#     ax2 = fig.add_subplot(2, 3, 2)
#     ax2.hist(confidences, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
#     ax2.axvline(np.mean(confidences), color='red', linestyle='--',
#                 label=f'均值: {np.mean(confidences):.3f}')
#     ax2.set_title('预测置信度分布', fontsize=12, fontweight='bold')
#     ax2.set_xlabel('置信度')
#     ax2.set_ylabel('样本数')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
#
#     # 3. 箱线图
#     ax3 = fig.add_subplot(2, 3, 3)
#     conf_by_class = [confidences[predictions == i] for i in unique]
#     bp = ax3.boxplot(conf_by_class, labels=[class_names[i] for i in unique], patch_artist=True)
#     for patch, i in zip(bp['boxes'], unique):
#         patch.set_facecolor(colors[i])
#         patch.set_alpha(0.6)
#     ax3.set_title('各类别置信度', fontsize=12, fontweight='bold')
#     ax3.set_ylabel('置信度')
#     ax3.grid(True, alpha=0.3, axis='y')
#
#     # 4-6. 典型样本
#     for idx, pred_class in enumerate(unique[:3]):
#         ax = fig.add_subplot(2, 3, 4 + idx)
#         class_mask = predictions == pred_class
#         class_conf = confidences[class_mask]
#         class_signals = signals[class_mask]
#
#         if len(class_conf) > 0:
#             best_idx = np.argmax(class_conf)
#             ax.plot(class_signals[best_idx], color=colors[pred_class], linewidth=0.5)
#             ax.set_title(f'{class_names[pred_class]} 典型样本\n置信度: {class_conf[best_idx]:.3f}',
#                          fontsize=11, fontweight='bold')
#         ax.set_xlabel('采样点')
#         ax.set_ylabel('幅值')
#         ax.grid(True, alpha=0.3)
#
#     plt.suptitle('DANN模型可解释性分析报告', fontsize=16, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     print(f"已保存: {save_path}")
#     plt.close()
#
#
# # ===================== 主函数 =====================
# def main():
#     print("=" * 70)
#     print("任务4 - DANN模型可解释性分析")
#     print("=" * 70)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"使用设备: {device}")
#
#     # 0. 检查模型文件
#     print("\n" + "=" * 50)
#     print("检查模型文件...")
#     print("=" * 50)
#     check_model_file()
#
#     # 1. 加载模型
#     print("\n" + "=" * 50)
#     print("[1/7] 智能加载模型...")
#     print("=" * 50)
#     model, load_success = load_model_smart(device)
#     model.eval()
#
#     # 2. 加载目标域数据
#     print("\n" + "=" * 50)
#     print("[2/7] 加载目标域数据...")
#     print("=" * 50)
#
#     if os.path.exists('target_data.npy'):
#         target_data = np.load('target_data.npy')
#         print(f"目标域数据形状: {target_data.shape}")
#     else:
#         print("未找到target_data.npy，使用模拟数据")
#         target_data = np.random.randn(16, 32000) * 0.1
#
#     # 处理数据
#     sample_len = 1024
#     all_signals = []
#
#     for i in range(target_data.shape[0]):
#         file_data = target_data[i]
#         n_samples = len(file_data) // sample_len
#
#         for j in range(min(n_samples, 10)):
#             segment = file_data[j * sample_len: (j + 1) * sample_len]
#             if len(segment) == sample_len:
#                 # 归一化
#                 segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
#                 all_signals.append(segment)
#
#     all_signals = np.array(all_signals)
#     print(f"处理后样本数: {len(all_signals)}")
#
#     # 3. 模型预测
#     print("\n" + "=" * 50)
#     print("[3/7] 模型预测...")
#     print("=" * 50)
#
#     all_predictions = []
#     all_confidences = []
#
#     with torch.no_grad():
#         for i in range(0, len(all_signals), 32):
#             batch = all_signals[i:i + 32]
#             inputs = torch.FloatTensor(batch).unsqueeze(1).to(device)
#             outputs, _ = model(inputs)
#             probs = F.softmax(outputs, dim=1)
#             preds = outputs.argmax(dim=1).cpu().numpy()
#             confs = probs.max(dim=1)[0].cpu().numpy()
#
#             all_predictions.extend(preds)
#             all_confidences.extend(confs)
#
#     all_predictions = np.array(all_predictions)
#     all_confidences = np.array(all_confidences)
#
#     class_names = ['Normal', 'IR', 'OR', 'Ball']
#     print("\n预测结果统计:")
#     for i in range(4):
#         count = np.sum(all_predictions == i)
#         if count > 0:
#             mean_conf = np.mean(all_confidences[all_predictions == i])
#             print(f"  {class_names[i]}: {count} 样本, 平均置信度: {mean_conf:.3f}")
#
#     # 4. STFT分析
#     print("\n" + "=" * 50)
#     print("[4/7] 生成STFT时频分析图...")
#     print("=" * 50)
#     plot_stft_analysis(all_signals, all_predictions, fs=12000,
#                        save_path='stft_analysis.png')
#
#     # 5. 包络谱分析
#     print("\n" + "=" * 50)
#     print("[5/7] 生成包络谱分析图...")
#     print("=" * 50)
#     plot_envelope_analysis(all_signals, all_predictions, fs=12000,
#                            save_path='envelope_spectrum_analysis.png')
#
#     # 6. Grad-CAM
#     print("\n" + "=" * 50)
#     print("[6/7] 生成Grad-CAM分析...")
#     print("=" * 50)
#     if load_success:
#         plot_gradcam(model, all_signals, all_predictions, device,
#                      save_path='gradcam_analysis.png')
#     else:
#         print("模型未成功加载，跳过Grad-CAM")
#
#     # 7. 特征重要性
#     print("\n" + "=" * 50)
#     print("[7/7] 生成特征重要性分析...")
#     print("=" * 50)
#     plot_feature_importance(model, all_signals, all_predictions, device,
#                             save_path='feature_importance_analysis.png')
#
#     # 8. 综合报告
#     print("\n" + "=" * 50)
#     print("[8/8] 生成综合报告...")
#     print("=" * 50)
#     plot_summary(all_signals, all_predictions, all_confidences,
#                  save_path='interpretability_summary.png')
#
#     # 总结
#     print("\n" + "=" * 70)
#     print("可解释性分析完成!")
#     print("=" * 70)
#     print("""
# 生成的文件:
#   1. stft_analysis.png              - STFT时频分析
#   2. envelope_spectrum_analysis.png - 包络谱分析
#   3. gradcam_analysis.png           - Grad-CAM热力图
#   4. feature_importance_analysis.png - 特征重要性
#   5. interpretability_summary.png   - 综合报告
# """)
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
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, stft
from scipy.fft import fft, fftfreq
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def check_model_file():
    """检查模型文件的格式"""
    model_files = ['dann_model_best_full.pth', 'dann_model_best.pth',
                   'dann_model_final.pth', 'dann_model.pth']

    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"\n检查模型文件: {model_file}")
            checkpoint = torch.load(model_file, map_location='cpu')

            if isinstance(checkpoint, dict):
                print(f"  类型: 字典")
                print(f"  键: {list(checkpoint.keys())}")

                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print(f"  state_dict键数量: {len(state_dict)}")
                    print(f"  前5个键: {list(state_dict.keys())[:5]}")
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"  model_state_dict键数量: {len(state_dict)}")
                    print(f"  前5个键: {list(state_dict.keys())[:5]}")
                else:
                    # 可能直接就是state_dict
                    print(f"  直接state_dict键数量: {len(checkpoint)}")
                    print(f"  前5个键: {list(checkpoint.keys())[:5]}")
            else:
                print(f"  类型: {type(checkpoint)}")

            return model_file, checkpoint

    print("未找到任何模型文件!")
    return None, None


def build_model_from_state_dict(state_dict):
    """根据state_dict的键推断并构建模型"""

    # 分析state_dict的结构
    keys = list(state_dict.keys())
    print(f"\n分析模型结构，共 {len(keys)} 个参数")

    # 检测是否有特定前缀
    has_feature_extractor = any('feature_extractor' in k for k in keys)
    has_encoder = any('encoder' in k for k in keys)
    has_conv = any('conv' in k for k in keys)

    print(f"  has_feature_extractor: {has_feature_extractor}")
    print(f"  has_encoder: {has_encoder}")
    print(f"  has_conv: {has_conv}")

    # 打印所有键以便调试
    print("\n所有参数键:")
    for k in keys:
        print(f"  {k}: {state_dict[k].shape}")

    return keys


# ===================== 尝试从model.py导入 =====================
try:
    from model import DANN, FeatureExtractor, Classifier

    print("成功从 model.py 导入模型定义")
    USE_LOCAL_MODEL = True
except ImportError:
    print("无法从 model.py 导入，将使用内置模型定义")
    USE_LOCAL_MODEL = False

# ===================== 内置模型定义（备用）=====================
if not USE_LOCAL_MODEL:
    class GradientReversalFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.alpha, None


    class FeatureExtractor(nn.Module):
        def __init__(self, input_size=1024):
            super(FeatureExtractor, self).__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=64, stride=8, padding=28)
            self.bn1 = nn.BatchNorm1d(32)
            self.pool1 = nn.MaxPool1d(2)

            self.conv2 = nn.Conv1d(32, 64, kernel_size=32, stride=4, padding=14)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool2 = nn.MaxPool1d(2)

            self.conv3 = nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=7)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool3 = nn.AdaptiveAvgPool1d(8)

            self.feature_dim = 128 * 8

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0), -1)
            return x


    class Classifier(nn.Module):
        def __init__(self, feature_dim=1024, num_classes=4):
            super(Classifier, self).__init__()
            self.fc1 = nn.Linear(feature_dim, 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.dropout2 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
            x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)
            return x


    class DANN(nn.Module):
        def __init__(self, input_size=1024, num_classes=4):
            super(DANN, self).__init__()
            self.feature_extractor = FeatureExtractor(input_size)
            self.classifier = Classifier(self.feature_extractor.feature_dim, num_classes)

        def forward(self, x, alpha=1.0):
            features = self.feature_extractor(x)
            class_output = self.classifier(features)
            return class_output, features


def load_model_smart(device):
    """智能加载模型，处理各种保存格式"""

    model_files = ['dann_model_best_full.pth', 'dann_model_best.pth',
                   'dann_model_final.pth', 'dann_model.pth']

    for model_file in model_files:
        if not os.path.exists(model_file):
            continue

        print(f"\n尝试加载: {model_file}")
        checkpoint = torch.load(model_file, map_location=device)

        # 解析checkpoint
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                num_classes = checkpoint.get('num_classes', 4)
                print(f"  从 'state_dict' 键加载，num_classes={num_classes}")
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                num_classes = checkpoint.get('num_classes', 4)
                print(f"  从 'model_state_dict' 键加载，num_classes={num_classes}")
            else:
                # 检查是否直接是state_dict
                if any('conv' in k or 'fc' in k or 'weight' in k for k in checkpoint.keys()):
                    state_dict = checkpoint
                    num_classes = 4
                    print(f"  直接作为state_dict加载")
                else:
                    print(f"  未知格式，跳过")
                    continue
        else:
            print(f"  非字典格式，跳过")
            continue

        # 分析state_dict结构来确定模型架构
        keys = list(state_dict.keys())
        print(f"  参数数量: {len(keys)}")
        print(f"  前3个键: {keys[:3]}")

        # 创建模型并加载权重
        try:
            model = DANN(input_size=1024, num_classes=num_classes).to(device)
            model.load_state_dict(state_dict, strict=False)
            print(f"  ✓ 模型加载成功!")
            return model, True
        except Exception as e:
            print(f"  加载失败: {e}")

            # 尝试修改键名后加载
            try:
                new_state_dict = {}
                for k, v in state_dict.items():
                    # 移除可能的前缀
                    new_key = k.replace('module.', '')
                    new_state_dict[new_key] = v

                model = DANN(input_size=1024, num_classes=num_classes).to(device)
                model.load_state_dict(new_state_dict, strict=False)
                print(f"  ✓ 修改键名后加载成功!")
                return model, True
            except Exception as e2:
                print(f"  修改键名后仍失败: {e2}")

    # 如果都失败，返回未训练的模型
    print("\n警告: 所有模型文件加载失败，使用随机初始化的模型")
    model = DANN(input_size=1024, num_classes=4).to(device)
    return model, False


# ===================== Grad-CAM =====================
class GradCAM1D:
    """1D卷积网络的Grad-CAM实现"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        output, _ = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        if self.gradients is None or self.activations is None:
            return np.zeros(input_tensor.shape[-1])

        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


# ===================== STFT时频分析 =====================
def compute_stft(signal, fs=12000, nperseg=128, noverlap=None):
    """计算STFT"""
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, np.abs(Zxx)


def plot_stft_analysis(signals, predictions, fs=12000, save_path='stft_analysis.png'):
    """STFT时频分析图"""
    class_names = ['Normal', 'IR', 'OR', 'Ball']
    unique_preds = np.unique(predictions)

    n_classes = len(unique_preds)
    fig, axes = plt.subplots(n_classes, 3, figsize=(15, 4 * n_classes))

    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for idx, pred_class in enumerate(unique_preds):
        sample_idx = np.where(predictions == pred_class)[0][0]
        signal = signals[sample_idx]

        # 时域波形
        ax1 = axes[idx, 0]
        t_signal = np.arange(len(signal)) / fs * 1000
        ax1.plot(t_signal, signal, 'b-', linewidth=0.5)
        ax1.set_title(f'{class_names[pred_class]} - 时域波形', fontsize=11, fontweight='bold')
        ax1.set_xlabel('时间 (ms)')
        ax1.set_ylabel('幅值')
        ax1.grid(True, alpha=0.3)

        # STFT
        ax2 = axes[idx, 1]
        f, t, Sxx = compute_stft(signal, fs=fs)
        im = ax2.pcolormesh(t * 1000, f, 20 * np.log10(Sxx + 1e-10),
                            shading='gouraud', cmap='jet')
        ax2.set_title(f'{class_names[pred_class]} - STFT时频图', fontsize=11, fontweight='bold')
        ax2.set_xlabel('时间 (ms)')
        ax2.set_ylabel('频率 (Hz)')
        ax2.set_ylim(0, fs // 2)
        plt.colorbar(im, ax=ax2, label='dB')

        # 频谱
        ax3 = axes[idx, 2]
        freqs = fftfreq(len(signal), 1 / fs)[:len(signal) // 2]
        spectrum = np.abs(fft(signal))[:len(signal) // 2] / len(signal) * 2
        ax3.plot(freqs, spectrum, 'g-', linewidth=0.7)
        ax3.set_title(f'{class_names[pred_class]} - 频谱', fontsize=11, fontweight='bold')
        ax3.set_xlabel('频率 (Hz)')
        ax3.set_ylabel('幅值')
        ax3.set_xlim(0, 3000)
        ax3.grid(True, alpha=0.3)

    plt.suptitle('目标域样本时频分析 (STFT)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()


# ===================== 包络谱分析 =====================
def compute_envelope_spectrum(signal, fs=12000, band=(500, 4000)):
    """计算包络谱"""
    nyq = fs / 2
    low = max(band[0] / nyq, 0.01)
    high = min(band[1] / nyq, 0.99)

    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)

    envelope = np.abs(hilbert(filtered))
    envelope = envelope - np.mean(envelope)

    n = len(envelope)
    freqs = fftfreq(n, 1 / fs)[:n // 2]
    spectrum = np.abs(fft(envelope))[:n // 2] / n * 2

    return freqs, spectrum


def plot_envelope_analysis(signals, predictions, fs=12000,
                           save_path='envelope_spectrum_analysis.png'):
    """包络谱分析图"""
    class_names = ['Normal', 'IR', 'OR', 'Ball']
    unique_preds = np.unique(predictions)

    # 故障特征频率
    BPFO, BPFI, BSF = 107.0, 162.0, 70.0

    n_classes = len(unique_preds)
    fig, axes = plt.subplots(n_classes, 2, figsize=(14, 4 * n_classes))

    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for idx, pred_class in enumerate(unique_preds):
        sample_idx = np.where(predictions == pred_class)[0][0]
        signal = signals[sample_idx]

        # 信号与包络
        ax1 = axes[idx, 0]
        nyq = fs / 2
        b, a = butter(4, [500 / nyq, 4000 / nyq], btype='band')
        filtered = filtfilt(b, a, signal)
        envelope = np.abs(hilbert(filtered))

        t = np.arange(len(signal)) / fs * 1000
        ax1.plot(t, signal, 'b-', linewidth=0.5, alpha=0.5, label='原始信号')
        ax1.plot(t, envelope, 'r-', linewidth=1, label='包络')
        ax1.set_title(f'{class_names[pred_class]} - 信号与包络', fontsize=11, fontweight='bold')
        ax1.set_xlabel('时间 (ms)')
        ax1.set_ylabel('幅值')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 包络谱
        ax2 = axes[idx, 1]
        freqs, spectrum = compute_envelope_spectrum(signal, fs)
        ax2.plot(freqs, spectrum, 'r-', linewidth=0.7)
        ax2.axvline(BPFO, color='blue', linestyle='--', alpha=0.7, label=f'BPFO={BPFO:.0f}Hz')
        ax2.axvline(BPFI, color='orange', linestyle='--', alpha=0.7, label=f'BPFI={BPFI:.0f}Hz')
        ax2.axvline(2 * BSF, color='purple', linestyle='--', alpha=0.7, label=f'2BSF={2 * BSF:.0f}Hz')
        ax2.set_title(f'{class_names[pred_class]} - 包络谱', fontsize=11, fontweight='bold')
        ax2.set_xlabel('频率 (Hz)')
        ax2.set_ylabel('幅值')
        ax2.set_xlim(0, 400)
        ax2.legend(loc='upper right', fontsize=7)
        ax2.grid(True, alpha=0.3)

    plt.suptitle('目标域样本包络谱分析', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()


# ===================== Grad-CAM可视化 =====================
def plot_gradcam(model, signals, predictions, device, save_path='gradcam_analysis.png'):
    """Grad-CAM热力图"""
    class_names = ['Normal', 'IR', 'OR', 'Ball']
    unique_preds = np.unique(predictions)

    # 获取目标层
    try:
        if hasattr(model, 'feature_extractor'):
            if hasattr(model.feature_extractor, 'conv3'):
                target_layer = model.feature_extractor.conv3
            elif hasattr(model.feature_extractor, 'conv_layers'):
                target_layer = model.feature_extractor.conv_layers[-1]
            else:
                # 找最后一个卷积层
                conv_layers = [m for m in model.feature_extractor.modules() if isinstance(m, nn.Conv1d)]
                target_layer = conv_layers[-1] if conv_layers else None
        else:
            conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
            target_layer = conv_layers[-1] if conv_layers else None
    except:
        target_layer = None

    if target_layer is None:
        print("警告: 无法找到卷积层，跳过Grad-CAM")
        return

    gradcam = GradCAM1D(model, target_layer)

    n_classes = len(unique_preds)
    fig, axes = plt.subplots(n_classes, 2, figsize=(14, 4 * n_classes))

    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for idx, pred_class in enumerate(unique_preds):
        sample_idx = np.where(predictions == pred_class)[0][0]
        signal = signals[sample_idx]

        input_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0).to(device)

        try:
            cam = gradcam.generate(input_tensor, pred_class)
            cam_interp = np.interp(np.linspace(0, 1, len(signal)),
                                   np.linspace(0, 1, len(cam)), cam)
        except:
            cam_interp = np.zeros(len(signal))

        # 原始信号
        ax1 = axes[idx, 0]
        ax1.plot(signal, 'b-', linewidth=0.5)
        ax1.set_title(f'{class_names[pred_class]} - 原始信号', fontsize=11, fontweight='bold')
        ax1.set_xlabel('采样点')
        ax1.set_ylabel('幅值')
        ax1.grid(True, alpha=0.3)

        # CAM热力图
        ax2 = axes[idx, 1]
        colors = plt.cm.jet(cam_interp)
        for i in range(len(signal) - 1):
            ax2.plot([i, i + 1], [signal[i], signal[i + 1]], color=colors[i], linewidth=1)

        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
        plt.colorbar(sm, ax=ax2, label='重要性')
        ax2.set_title(f'{class_names[pred_class]} - Grad-CAM', fontsize=11, fontweight='bold')
        ax2.set_xlabel('采样点')
        ax2.set_ylabel('幅值')
        ax2.grid(True, alpha=0.3)

    plt.suptitle('Grad-CAM可解释性分析', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()


# ===================== 特征重要性（扰动法）=====================
def plot_feature_importance(model, signals, predictions, device,
                            save_path='feature_importance_analysis.png'):
    """扰动法特征重要性"""
    model.eval()
    class_names = ['Normal', 'IR', 'OR', 'Ball']

    sample_idx = 0
    signal = signals[sample_idx]
    pred_class = predictions[sample_idx]

    n_segments = 16
    segment_len = len(signal) // n_segments

    input_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output, _ = model(input_tensor)
        orig_prob = F.softmax(output, dim=1)[0, pred_class].item()

    importance = []
    for i in range(n_segments):
        perturbed = signal.copy()
        start, end = i * segment_len, min((i + 1) * segment_len, len(signal))
        perturbed[start:end] = np.random.randn(end - start) * np.std(signal)

        input_tensor = torch.FloatTensor(perturbed).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output, _ = model(input_tensor)
            new_prob = F.softmax(output, dim=1)[0, pred_class].item()

        importance.append(max(orig_prob - new_prob, 0))

    importance = np.array(importance)
    if importance.max() > 0:
        importance = importance / importance.max()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax1 = axes[0]
    ax1.plot(signal, 'b-', linewidth=0.5)
    for i in range(n_segments):
        start, end = i * segment_len, min((i + 1) * segment_len, len(signal))
        ax1.axvspan(start, end, alpha=0.3, color=plt.cm.Reds(importance[i]))
    ax1.set_title(f'样本信号与区间重要性 (预测: {class_names[pred_class]})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('采样点')
    ax1.set_ylabel('幅值')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    bars = ax2.bar(range(n_segments), importance, color=[plt.cm.Reds(v) for v in importance],
                   edgecolor='black', linewidth=0.5)
    ax2.set_title('各区间对分类的重要性', fontsize=12, fontweight='bold')
    ax2.set_xlabel('区间编号')
    ax2.set_ylabel('重要性得分')
    ax2.set_xticks(range(n_segments))
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('特征重要性分析 (扰动法)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()


# ===================== 综合报告 =====================
def plot_summary(signals, predictions, confidences, save_path='interpretability_summary.png'):
    """综合可解释性报告"""
    class_names = ['Normal', 'IR', 'OR', 'Ball']
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

    fig = plt.figure(figsize=(16, 12))

    # 1. 饼图
    ax1 = fig.add_subplot(2, 3, 1)
    unique, counts = np.unique(predictions, return_counts=True)
    ax1.pie(counts, labels=[class_names[i] for i in unique],
            colors=[colors[i] for i in unique], autopct='%1.1f%%')
    ax1.set_title('预测类别分布', fontsize=12, fontweight='bold')

    # 2. 置信度分布
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(confidences, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(confidences), color='red', linestyle='--',
                label=f'均值: {np.mean(confidences):.3f}')
    ax2.set_title('预测置信度分布', fontsize=12, fontweight='bold')
    ax2.set_xlabel('置信度')
    ax2.set_ylabel('样本数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 箱线图
    ax3 = fig.add_subplot(2, 3, 3)
    conf_by_class = [confidences[predictions == i] for i in unique]
    bp = ax3.boxplot(conf_by_class, labels=[class_names[i] for i in unique], patch_artist=True)
    for patch, i in zip(bp['boxes'], unique):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.6)
    ax3.set_title('各类别置信度', fontsize=12, fontweight='bold')
    ax3.set_ylabel('置信度')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4-6. 典型样本
    for idx, pred_class in enumerate(unique[:3]):
        ax = fig.add_subplot(2, 3, 4 + idx)
        class_mask = predictions == pred_class
        class_conf = confidences[class_mask]
        class_signals = signals[class_mask]

        if len(class_conf) > 0:
            best_idx = np.argmax(class_conf)
            ax.plot(class_signals[best_idx], color=colors[pred_class], linewidth=0.5)
            ax.set_title(f'{class_names[pred_class]} 典型样本\n置信度: {class_conf[best_idx]:.3f}',
                         fontsize=11, fontweight='bold')
        ax.set_xlabel('采样点')
        ax.set_ylabel('幅值')
        ax.grid(True, alpha=0.3)

    plt.suptitle('DANN模型可解释性分析报告', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()


# ===================== 主函数 =====================
def main():
    print("=" * 70)
    print("任务4 - DANN模型可解释性分析")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 0. 检查模型文件
    print("\n" + "=" * 50)
    print("检查模型文件...")
    print("=" * 50)
    check_model_file()

    # 1. 加载模型
    print("\n" + "=" * 50)
    print("[1/7] 智能加载模型...")
    print("=" * 50)
    model, load_success = load_model_smart(device)
    model.eval()

    # 2. 加载目标域数据
    print("\n" + "=" * 50)
    print("[2/7] 加载目标域数据...")
    print("=" * 50)

    if os.path.exists('target_data.npy'):
        # 【修复】添加 allow_pickle=True，因为 target_data.npy 是字典格式
        target_data_raw = np.load('target_data.npy', allow_pickle=True)

        # 【修复】处理字典格式的数据
        if isinstance(target_data_raw, np.ndarray) and target_data_raw.ndim == 0:
            # numpy保存的字典会变成0维数组，需要用.item()提取
            target_data_dict = target_data_raw.item()
        elif isinstance(target_data_raw, dict):
            target_data_dict = target_data_raw
        else:
            # 如果是普通数组格式
            target_data_dict = None
            target_data_array = target_data_raw

        if target_data_dict is not None:
            print(f"目标域数据为字典格式，包含 {len(target_data_dict)} 个文件")
            print(f"文件列表: {list(target_data_dict.keys())}")
        else:
            print(f"目标域数据形状: {target_data_array.shape}")
    else:
        print("未找到target_data.npy，使用模拟数据")
        target_data_dict = None
        target_data_array = np.random.randn(16, 32000) * 0.1

    # 处理数据
    sample_len = 1024
    all_signals = []

    # 【修复】根据数据格式选择不同的处理方式
    try:
        if target_data_dict is not None:
            # 字典格式：遍历每个文件
            for file_id, file_data in target_data_dict.items():
                file_data = file_data.astype(np.float32).flatten()
                n_samples = len(file_data) // sample_len

                for j in range(min(n_samples, 10)):
                    segment = file_data[j * sample_len: (j + 1) * sample_len]
                    if len(segment) == sample_len:
                        # 归一化
                        segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
                        all_signals.append(segment)
        else:
            # 数组格式
            for i in range(target_data_array.shape[0]):
                file_data = target_data_array[i]
                n_samples = len(file_data) // sample_len

                for j in range(min(n_samples, 10)):
                    segment = file_data[j * sample_len: (j + 1) * sample_len]
                    if len(segment) == sample_len:
                        # 归一化
                        segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
                        all_signals.append(segment)
    except NameError:
        # 如果target_data_dict未定义，使用target_data_array
        for i in range(target_data_array.shape[0]):
            file_data = target_data_array[i]
            n_samples = len(file_data) // sample_len

            for j in range(min(n_samples, 10)):
                segment = file_data[j * sample_len: (j + 1) * sample_len]
                if len(segment) == sample_len:
                    segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
                    all_signals.append(segment)

    all_signals = np.array(all_signals)
    print(f"处理后样本数: {len(all_signals)}")

    # 3. 模型预测
    print("\n" + "=" * 50)
    print("[3/7] 模型预测...")
    print("=" * 50)

    all_predictions = []
    all_confidences = []

    with torch.no_grad():
        for i in range(0, len(all_signals), 32):
            batch = all_signals[i:i + 32]
            inputs = torch.FloatTensor(batch).unsqueeze(1).to(device)
            outputs, _ = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).cpu().numpy()
            confs = probs.max(dim=1)[0].cpu().numpy()

            all_predictions.extend(preds)
            all_confidences.extend(confs)

    all_predictions = np.array(all_predictions)
    all_confidences = np.array(all_confidences)

    class_names = ['Normal', 'IR', 'OR', 'Ball']
    print("\n预测结果统计:")
    for i in range(4):
        count = np.sum(all_predictions == i)
        if count > 0:
            mean_conf = np.mean(all_confidences[all_predictions == i])
            print(f"  {class_names[i]}: {count} 样本, 平均置信度: {mean_conf:.3f}")

    # 4. STFT分析
    print("\n" + "=" * 50)
    print("[4/7] 生成STFT时频分析图...")
    print("=" * 50)
    plot_stft_analysis(all_signals, all_predictions, fs=12000,
                       save_path='stft_analysis.png')

    # 5. 包络谱分析
    print("\n" + "=" * 50)
    print("[5/7] 生成包络谱分析图...")
    print("=" * 50)
    plot_envelope_analysis(all_signals, all_predictions, fs=12000,
                           save_path='envelope_spectrum_analysis.png')

    # 6. Grad-CAM
    print("\n" + "=" * 50)
    print("[6/7] 生成Grad-CAM分析...")
    print("=" * 50)
    if load_success:
        plot_gradcam(model, all_signals, all_predictions, device,
                     save_path='gradcam_analysis.png')
    else:
        print("模型未成功加载，跳过Grad-CAM")

    # 7. 特征重要性
    print("\n" + "=" * 50)
    print("[7/7] 生成特征重要性分析...")
    print("=" * 50)
    plot_feature_importance(model, all_signals, all_predictions, device,
                            save_path='feature_importance_analysis.png')

    # 8. 综合报告
    print("\n" + "=" * 50)
    print("[8/8] 生成综合报告...")
    print("=" * 50)
    plot_summary(all_signals, all_predictions, all_confidences,
                 save_path='interpretability_summary.png')

    # 总结
    print("\n" + "=" * 70)
    print("可解释性分析完成!")
    print("=" * 70)
    print("""
生成的文件:
  1. stft_analysis.png              - STFT时频分析
  2. envelope_spectrum_analysis.png - 包络谱分析
  3. gradcam_analysis.png           - Grad-CAM热力图
  4. feature_importance_analysis.png - 特征重要性
  5. interpretability_summary.png   - 综合报告
""")


if __name__ == "__main__":
    main()














