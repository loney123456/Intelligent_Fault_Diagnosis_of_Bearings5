# # task1_complete_analysis.py
# # 功能：任务1补充 - 源域数据筛选表格 + 故障机理可视化
# # ====================================================================
#
# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import hilbert, butter, filtfilt
# import pandas as pd
#
# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
#
#
# def generate_data_selection_table():
#     """生成源域数据筛选表格"""
#
#     print("=" * 70)
#     print("源域数据筛选策略")
#     print("=" * 70)
#
#     # 筛选策略说明
#     print("""
# 【筛选原则】
# 1. 选择DE（驱动端）通道数据，信号质量更好
# 2. 选择0.007"和0.014"故障直径（特征明显且不过于严重）
# 3. 外圈故障选择6点钟位置（承载区，特征最典型）
# 4. 包含所有转速条件，增加数据多样性
# 5. 各类故障样本数量均衡
#
# 【筛选结果】
# 信号通道: DE (驱动端加速度信号)
# 采样频率: 12kHz
# 转速条件: 1730-1797 rpm (负载0-3 hp)
# 故障直径: 0.007" + 0.014"
# """)
#
#     # 创建数据表格
#     data_files = [
#         # Normal
#         {'文件名': '97.mat', '故障类型': 'Normal', '故障直径': '-', '转速': 1797, '负载': 0},
#         {'文件名': '98.mat', '故障类型': 'Normal', '故障直径': '-', '转速': 1772, '负载': 1},
#         {'文件名': '99.mat', '故障类型': 'Normal', '故障直径': '-', '转速': 1750, '负载': 2},
#         {'文件名': '100.mat', '故障类型': 'Normal', '故障直径': '-', '转速': 1730, '负载': 3},
#         # IR 0.007
#         {'文件名': '105.mat', '故障类型': 'IR', '故障直径': '0.007"', '转速': 1797, '负载': 0},
#         {'文件名': '106.mat', '故障类型': 'IR', '故障直径': '0.007"', '转速': 1772, '负载': 1},
#         {'文件名': '107.mat', '故障类型': 'IR', '故障直径': '0.007"', '转速': 1750, '负载': 2},
#         {'文件名': '108.mat', '故障类型': 'IR', '故障直径': '0.007"', '转速': 1730, '负载': 3},
#         # IR 0.014
#         {'文件名': '169.mat', '故障类型': 'IR', '故障直径': '0.014"', '转速': 1797, '负载': 0},
#         {'文件名': '170.mat', '故障类型': 'IR', '故障直径': '0.014"', '转速': 1772, '负载': 1},
#         {'文件名': '171.mat', '故障类型': 'IR', '故障直径': '0.014"', '转速': 1750, '负载': 2},
#         {'文件名': '172.mat', '故障类型': 'IR', '故障直径': '0.014"', '转速': 1730, '负载': 3},
#         # OR 0.007 @6
#         {'文件名': '130.mat', '故障类型': 'OR', '故障直径': '0.007"@6点', '转速': 1797, '负载': 0},
#         {'文件名': '131.mat', '故障类型': 'OR', '故障直径': '0.007"@6点', '转速': 1772, '负载': 1},
#         {'文件名': '132.mat', '故障类型': 'OR', '故障直径': '0.007"@6点', '转速': 1750, '负载': 2},
#         {'文件名': '133.mat', '故障类型': 'OR', '故障直径': '0.007"@6点', '转速': 1730, '负载': 3},
#         # OR 0.014 @6
#         {'文件名': '197.mat', '故障类型': 'OR', '故障直径': '0.014"@6点', '转速': 1797, '负载': 0},
#         {'文件名': '198.mat', '故障类型': 'OR', '故障直径': '0.014"@6点', '转速': 1772, '负载': 1},
#         {'文件名': '199.mat', '故障类型': 'OR', '故障直径': '0.014"@6点', '转速': 1750, '负载': 2},
#         {'文件名': '200.mat', '故障类型': 'OR', '故障直径': '0.014"@6点', '转速': 1730, '负载': 3},
#         # Ball 0.007
#         {'文件名': '118.mat', '故障类型': 'Ball', '故障直径': '0.007"', '转速': 1797, '负载': 0},
#         {'文件名': '119.mat', '故障类型': 'Ball', '故障直径': '0.007"', '转速': 1772, '负载': 1},
#         {'文件名': '120.mat', '故障类型': 'Ball', '故障直径': '0.007"', '转速': 1750, '负载': 2},
#         {'文件名': '121.mat', '故障类型': 'Ball', '故障直径': '0.007"', '转速': 1730, '负载': 3},
#         # Ball 0.014
#         {'文件名': '185.mat', '故障类型': 'Ball', '故障直径': '0.014"', '转速': 1797, '负载': 0},
#         {'文件名': '186.mat', '故障类型': 'Ball', '故障直径': '0.014"', '转速': 1772, '负载': 1},
#         {'文件名': '187.mat', '故障类型': 'Ball', '故障直径': '0.014"', '转速': 1750, '负载': 2},
#         {'文件名': '188.mat', '故障类型': 'Ball', '故障直径': '0.014"', '转速': 1730, '负载': 3},
#     ]
#
#     df = pd.DataFrame(data_files)
#
#     print("\n源域数据筛选明细表：")
#     print("-" * 70)
#     print(df.to_string(index=False))
#
#     # 统计
#     print("\n" + "-" * 70)
#     print("筛选统计：")
#     for fault_type in ['Normal', 'IR', 'OR', 'Ball']:
#         count = len([d for d in data_files if d['故障类型'] == fault_type])
#         print(f"  {fault_type}: {count} 个文件")
#     print(f"  总计: {len(data_files)} 个文件")
#
#     # 保存表格
#     df.to_csv('source_data_selection_table.csv', index=False, encoding='utf-8-sig')
#     print("\n已保存: source_data_selection_table.csv")
#
#     return df
#
#
# def generate_fault_mechanism_figure():
#     """生成故障机理对比可视化图"""
#
#     print("\n" + "=" * 70)
#     print("生成故障机理可视化图")
#     print("=" * 70)
#
#     fig, axes = plt.subplots(4, 3, figsize=(14, 11))
#
#     fs = 12000  # 采样频率
#     fr = 29.95  # 转频 (1797rpm/60)
#     N = 512  # 样本长度
#     t = np.arange(N) / fs * 1000  # 时间(ms)
#
#     # CWRU轴承6205-2RS参数
#     n = 9  # 滚动体数量
#     d = 7.94  # 滚动体直径 mm
#     D = 39.04  # 节径 mm
#
#     # 计算故障频率
#     BPFO = n / 2 * fr * (1 - d / D)  # ~107 Hz
#     BPFI = n / 2 * fr * (1 + d / D)  # ~162 Hz
#     BSF = D / (2 * d) * fr * (1 - (d / D) ** 2)  # ~70 Hz
#
#     print(f"故障特征频率: BPFO={BPFO:.1f}Hz, BPFI={BPFI:.1f}Hz, BSF={BSF:.1f}Hz")
#
#     fault_configs = [
#         ('Normal', '正常状态', None),
#         ('OR', '外圈故障', BPFO),
#         ('IR', '内圈故障', BPFI),
#         ('Ball', '滚动体故障', 2 * BSF),
#     ]
#
#     for i, (fault_type, fault_name, fault_freq) in enumerate(fault_configs):
#
#         # 生成模拟信号
#         np.random.seed(42 + i)
#
#         if fault_type == 'Normal':
#             # 正常：随机噪声
#             signal_data = np.random.randn(N) * 0.1
#         elif fault_type == 'OR':
#             # 外圈：等幅等间隔冲击
#             signal_data = np.zeros(N)
#             impulse_interval = int(fs / BPFO)
#             for j in range(0, N, impulse_interval):
#                 if j < N:
#                     decay = np.exp(-np.arange(min(30, N - j)) / 6)
#                     carrier = np.sin(2 * np.pi * 3000 * np.arange(min(30, N - j)) / fs)
#                     signal_data[j:j + len(decay)] += 0.8 * decay * carrier
#             signal_data += np.random.randn(N) * 0.05
#         elif fault_type == 'IR':
#             # 内圈：幅值调制冲击
#             signal_data = np.zeros(N)
#             impulse_interval = int(fs / BPFI)
#             for j in range(0, N, impulse_interval):
#                 if j < N:
#                     # 转频调制
#                     mod = 0.4 + 0.6 * np.sin(2 * np.pi * fr * j / fs)
#                     decay = np.exp(-np.arange(min(25, N - j)) / 5)
#                     carrier = np.sin(2 * np.pi * 3200 * np.arange(min(25, N - j)) / fs)
#                     signal_data[j:j + len(decay)] += mod * decay * carrier
#             signal_data += np.random.randn(N) * 0.05
#         elif fault_type == 'Ball':
#             # 滚动体：2×BSF频率冲击
#             signal_data = np.zeros(N)
#             impulse_interval = int(fs / (2 * BSF))
#             for j in range(0, N, impulse_interval):
#                 if j < N:
#                     decay = np.exp(-np.arange(min(20, N - j)) / 4)
#                     carrier = np.sin(2 * np.pi * 2800 * np.arange(min(20, N - j)) / fs)
#                     signal_data[j:j + len(decay)] += 0.6 * decay * carrier
#             signal_data += np.random.randn(N) * 0.05
#
#         # 第1列：时域波形
#         ax1 = axes[i, 0]
#         ax1.plot(t, signal_data, 'b-', linewidth=0.6)
#         ax1.set_title(f'{fault_name} - 时域波形', fontsize=11, fontweight='bold')
#         ax1.set_xlabel('时间 (ms)', fontsize=9)
#         ax1.set_ylabel('幅值', fontsize=9)
#         ax1.grid(True, alpha=0.3)
#         ax1.set_xlim(0, t[-1])
#
#         # 添加特征说明
#         feature_text = {
#             'Normal': '无周期性冲击',
#             'OR': '等幅等间隔冲击',
#             'IR': '幅值调制冲击',
#             'Ball': '复杂调制冲击'
#         }
#         ax1.text(0.98, 0.95, feature_text[fault_type], transform=ax1.transAxes,
#                  fontsize=9, ha='right', va='top', color='red',
#                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
#
#         # 第2列：频谱
#         ax2 = axes[i, 1]
#         freqs = np.fft.rfftfreq(N, 1 / fs)
#         spectrum = np.abs(np.fft.rfft(signal_data)) / N * 2
#         ax2.plot(freqs, spectrum, 'g-', linewidth=0.7)
#         ax2.set_title(f'{fault_name} - 频谱', fontsize=11, fontweight='bold')
#         ax2.set_xlabel('频率 (Hz)', fontsize=9)
#         ax2.set_ylabel('幅值', fontsize=9)
#         ax2.set_xlim(0, 2000)
#         ax2.grid(True, alpha=0.3)
#
#         # 第3列：包络谱
#         ax3 = axes[i, 2]
#
#         # 带通滤波 + 希尔伯特包络
#         try:
#             nyq = fs / 2
#             b, a = butter(4, [500 / nyq, 4000 / nyq], btype='band')
#             filtered = filtfilt(b, a, signal_data)
#             envelope = np.abs(hilbert(filtered))
#             envelope = envelope - np.mean(envelope)
#
#             env_freqs = np.fft.rfftfreq(N, 1 / fs)
#             env_spectrum = np.abs(np.fft.rfft(envelope)) / N * 2
#
#             ax3.plot(env_freqs, env_spectrum, 'r-', linewidth=0.7)
#
#             # 标注故障特征频率
#             ymax = ax3.get_ylim()[1]
#             ax3.axvline(BPFO, color='blue', linestyle='--', alpha=0.6, linewidth=1.2)
#             ax3.axvline(BPFI, color='orange', linestyle='--', alpha=0.6, linewidth=1.2)
#             ax3.axvline(2 * BSF, color='purple', linestyle='--', alpha=0.6, linewidth=1.2)
#
#             # 只在第一行添加频率标签
#             if i == 0:
#                 ax3.text(BPFO, ymax * 0.9, f'BPFO\n{BPFO:.0f}Hz', fontsize=7,
#                          ha='center', color='blue')
#                 ax3.text(BPFI, ymax * 0.7, f'BPFI\n{BPFI:.0f}Hz', fontsize=7,
#                          ha='center', color='orange')
#                 ax3.text(2 * BSF, ymax * 0.5, f'2BSF\n{2 * BSF:.0f}Hz', fontsize=7,
#                          ha='center', color='purple')
#
#         except Exception as e:
#             ax3.text(0.5, 0.5, f'处理失败: {e}', transform=ax3.transAxes, ha='center')
#
#         ax3.set_title(f'{fault_name} - 包络谱', fontsize=11, fontweight='bold')
#         ax3.set_xlabel('频率 (Hz)', fontsize=9)
#         ax3.set_ylabel('幅值', fontsize=9)
#         ax3.set_xlim(0, 400)
#         ax3.grid(True, alpha=0.3)
#
#     plt.suptitle('轴承故障振动机理与特征对比\n(CWRU轴承6205-2RS, 转速1797rpm, 采样率12kHz)',
#                  fontsize=13, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig('fault_mechanism_comparison.png', dpi=200, bbox_inches='tight')
#     print("已保存: fault_mechanism_comparison.png")
#     plt.close()
#
#
# def generate_domain_comparison_figure():
#     """生成源域与目标域对比图"""
#
#     print("\n" + "=" * 70)
#     print("生成源域与目标域对比图")
#     print("=" * 70)
#
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
#     # 1. 参数对比柱状图
#     ax1 = axes[0]
#
#     params = ['转速\n(rpm)', 'BPFO\n(Hz)', 'BPFI\n(Hz)', 'BSF\n(Hz)', '采样率\n(kHz)']
#     source_vals = [1750, 107, 162, 70, 12]
#     target_vals = [600, 60, 90, 22, 32]
#
#     x = np.arange(len(params))
#     width = 0.35
#
#     bars1 = ax1.bar(x - width / 2, source_vals, width, label='源域(CWRU)',
#                     color='steelblue', edgecolor='black')
#     bars2 = ax1.bar(x + width / 2, target_vals, width, label='目标域(列车)',
#                     color='coral', edgecolor='black')
#
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(params, fontsize=10)
#     ax1.set_ylabel('数值', fontsize=11)
#     ax1.set_title('源域与目标域参数对比', fontsize=13, fontweight='bold')
#     ax1.legend(fontsize=10)
#     ax1.grid(True, alpha=0.3, axis='y')
#
#     # 添加数值标签
#     for bar in bars1:
#         height = bar.get_height()
#         ax1.text(bar.get_x() + bar.get_width() / 2, height + 5,
#                  f'{int(height)}', ha='center', va='bottom', fontsize=9)
#     for bar in bars2:
#         height = bar.get_height()
#         ax1.text(bar.get_x() + bar.get_width() / 2, height + 5,
#                  f'{int(height)}', ha='center', va='bottom', fontsize=9)
#
#     # 2. 域迁移挑战说明
#     ax2 = axes[1]
#     ax2.axis('off')
#
#     challenges = [
#         ['挑战', '源域', '目标域', '解决方案'],
#         ['转速差异', '~1750 rpm', '~600 rpm', '学习与转速无关的形态特征'],
#         ['采样率', '12 kHz', '32 kHz', '重采样统一时间尺度'],
#         ['轴承类型', '深沟球轴承', '圆柱滚子轴承', '学习通用故障模式'],
#         ['环境噪声', '实验室(低)', '实际运行(高)', '域对抗训练提取鲁棒特征'],
#         ['数据标签', '有标签', '无标签', 'DANN无监督域适应'],
#     ]
#
#     table = ax2.table(cellText=challenges[1:], colLabels=challenges[0],
#                       loc='center', cellLoc='center')
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1.3, 2.0)
#
#     # 设置表头样式
#     for i in range(4):
#         table[(0, i)].set_facecolor('#34495E')
#         table[(0, i)].set_text_props(color='white', fontweight='bold')
#
#     # 设置交替行颜色
#     for i in range(1, 6):
#         color = '#EBF5FB' if i % 2 == 1 else '#FDEBD0'
#         for j in range(4):
#             table[(i, j)].set_facecolor(color)
#
#     ax2.set_title('跨域迁移的关键挑战与解决方案', fontsize=13, fontweight='bold', pad=20)
#
#     plt.tight_layout()
#     plt.savefig('domain_comparison_analysis.png', dpi=200, bbox_inches='tight')
#     print("已保存: domain_comparison_analysis.png")
#     plt.close()
#
#
# def generate_fault_frequency_table():
#     """生成故障频率计算表"""
#
#     print("\n" + "=" * 70)
#     print("故障特征频率计算")
#     print("=" * 70)
#
#     # CWRU轴承参数
#     print("\nCWRU轴承 6205-2RS 参数:")
#     print("  滚动体数量 n = 9")
#     print("  滚动体直径 d = 7.94 mm")
#     print("  节径 D = 39.04 mm")
#     print("  接触角 θ = 0°")
#
#     # 不同转速下的故障频率
#     rpms = [1797, 1772, 1750, 1730]
#
#     freq_data = []
#     for rpm in rpms:
#         fr = rpm / 60
#         n, d, D = 9, 7.94, 39.04
#
#         BPFO = n / 2 * fr * (1 - d / D)
#         BPFI = n / 2 * fr * (1 + d / D)
#         BSF = D / (2 * d) * fr * (1 - (d / D) ** 2)
#         FTF = fr / 2 * (1 - d / D)
#
#         freq_data.append({
#             '转速(rpm)': rpm,
#             '转频fr(Hz)': f'{fr:.2f}',
#             'BPFO(Hz)': f'{BPFO:.2f}',
#             'BPFI(Hz)': f'{BPFI:.2f}',
#             'BSF(Hz)': f'{BSF:.2f}',
#             'FTF(Hz)': f'{FTF:.2f}'
#         })
#
#     df = pd.DataFrame(freq_data)
#     print("\n不同转速下的故障特征频率:")
#     print(df.to_string(index=False))
#
#     df.to_csv('fault_characteristic_frequencies.csv', index=False, encoding='utf-8-sig')
#     print("\n已保存: fault_characteristic_frequencies.csv")
#
#     return df
#
#
# def main():
#     """主函数"""
#
#     print("=" * 70)
#     print("任务1补充 - 源域数据筛选 + 故障机理可视化")
#     print("=" * 70)
#
#     # 1. 生成数据筛选表格
#     generate_data_selection_table()
#
#     # 2. 生成故障频率表
#     generate_fault_frequency_table()
#
#     # 3. 生成故障机理对比图
#     generate_fault_mechanism_figure()
#
#     # 4. 生成域对比分析图
#     generate_domain_comparison_figure()
#
#     # 总结
#     print("\n" + "=" * 70)
#     print("任务1补充完成!")
#     print("=" * 70)
#     print("""
# 生成的文件:
#   1. source_data_selection_table.csv     - 源域数据筛选表格
#   2. fault_characteristic_frequencies.csv - 故障特征频率表
#   3. fault_mechanism_comparison.png      - 故障机理对比图
#   4. domain_comparison_analysis.png      - 源域与目标域对比图
# """)
#
#
# if __name__ == "__main__":
#     main()

# task1_complete_analysis.py
# 功能：任务1补充 - 源域数据筛选表格 + 故障机理可视化（修复版）
# ====================================================================

# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from scipy.signal import hilbert, butter, filtfilt
# import pandas as pd
#
# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
#
#
# def generate_data_selection_table():
#     """生成源域数据筛选表格（修复引号问题）"""
#
#     print("=" * 70)
#     print("源域数据筛选策略")
#     print("=" * 70)
#
#     # 筛选策略说明
#     print("""
# 【筛选原则】
# 1. 选择DE（驱动端）通道数据，信号质量更好
# 2. 选择0.007英寸和0.014英寸故障直径（特征明显且不过于严重）
# 3. 外圈故障选择6点钟位置（承载区，特征最典型）
# 4. 包含所有转速条件，增加数据多样性
# 5. 各类故障样本数量均衡
#
# 【筛选结果】
# 信号通道: DE (驱动端加速度信号)
# 采样频率: 12kHz
# 转速条件: 1730-1797 rpm (负载0-3 hp)
# 故障直径: 0.007英寸 + 0.014英寸
# """)
#
#     # 创建数据表格（避免使用英寸符号"）
#     data_files = [
#         # Normal
#         {'文件名': '97.mat', '故障类型': 'Normal', '故障直径(英寸)': '-', '转速(rpm)': 1797, '负载(hp)': 0,
#          '通道': 'DE'},
#         {'文件名': '98.mat', '故障类型': 'Normal', '故障直径(英寸)': '-', '转速(rpm)': 1772, '负载(hp)': 1,
#          '通道': 'DE'},
#         {'文件名': '99.mat', '故障类型': 'Normal', '故障直径(英寸)': '-', '转速(rpm)': 1750, '负载(hp)': 2,
#          '通道': 'DE'},
#         {'文件名': '100.mat', '故障类型': 'Normal', '故障直径(英寸)': '-', '转速(rpm)': 1730, '负载(hp)': 3,
#          '通道': 'DE'},
#         # IR 0.007
#         {'文件名': '105.mat', '故障类型': 'IR', '故障直径(英寸)': '0.007', '转速(rpm)': 1797, '负载(hp)': 0,
#          '通道': 'DE'},
#         {'文件名': '106.mat', '故障类型': 'IR', '故障直径(英寸)': '0.007', '转速(rpm)': 1772, '负载(hp)': 1,
#          '通道': 'DE'},
#         {'文件名': '107.mat', '故障类型': 'IR', '故障直径(英寸)': '0.007', '转速(rpm)': 1750, '负载(hp)': 2,
#          '通道': 'DE'},
#         {'文件名': '108.mat', '故障类型': 'IR', '故障直径(英寸)': '0.007', '转速(rpm)': 1730, '负载(hp)': 3,
#          '通道': 'DE'},
#         # IR 0.014
#         {'文件名': '169.mat', '故障类型': 'IR', '故障直径(英寸)': '0.014', '转速(rpm)': 1797, '负载(hp)': 0,
#          '通道': 'DE'},
#         {'文件名': '170.mat', '故障类型': 'IR', '故障直径(英寸)': '0.014', '转速(rpm)': 1772, '负载(hp)': 1,
#          '通道': 'DE'},
#         {'文件名': '171.mat', '故障类型': 'IR', '故障直径(英寸)': '0.014', '转速(rpm)': 1750, '负载(hp)': 2,
#          '通道': 'DE'},
#         {'文件名': '172.mat', '故障类型': 'IR', '故障直径(英寸)': '0.014', '转速(rpm)': 1730, '负载(hp)': 3,
#          '通道': 'DE'},
#         # OR 0.007 @6
#         {'文件名': '130.mat', '故障类型': 'OR', '故障直径(英寸)': '0.007 @6点钟', '转速(rpm)': 1797, '负载(hp)': 0,
#          '通道': 'DE'},
#         {'文件名': '131.mat', '故障类型': 'OR', '故障直径(英寸)': '0.007 @6点钟', '转速(rpm)': 1772, '负载(hp)': 1,
#          '通道': 'DE'},
#         {'文件名': '132.mat', '故障类型': 'OR', '故障直径(英寸)': '0.007 @6点钟', '转速(rpm)': 1750, '负载(hp)': 2,
#          '通道': 'DE'},
#         {'文件名': '133.mat', '故障类型': 'OR', '故障直径(英寸)': '0.007 @6点钟', '转速(rpm)': 1730, '负载(hp)': 3,
#          '通道': 'DE'},
#         # OR 0.014 @6
#         {'文件名': '197.mat', '故障类型': 'OR', '故障直径(英寸)': '0.014 @6点钟', '转速(rpm)': 1797, '负载(hp)': 0,
#          '通道': 'DE'},
#         {'文件名': '198.mat', '故障类型': 'OR', '故障直径(英寸)': '0.014 @6点钟', '转速(rpm)': 1772, '负载(hp)': 1,
#          '通道': 'DE'},
#         {'文件名': '199.mat', '故障类型': 'OR', '故障直径(英寸)': '0.014 @6点钟', '转速(rpm)': 1750, '负载(hp)': 2,
#          '通道': 'DE'},
#         {'文件名': '200.mat', '故障类型': 'OR', '故障直径(英寸)': '0.014 @6点钟', '转速(rpm)': 1730, '负载(hp)': 3,
#          '通道': 'DE'},
#         # Ball 0.007
#         {'文件名': '118.mat', '故障类型': 'Ball', '故障直径(英寸)': '0.007', '转速(rpm)': 1797, '负载(hp)': 0,
#          '通道': 'DE'},
#         {'文件名': '119.mat', '故障类型': 'Ball', '故障直径(英寸)': '0.007', '转速(rpm)': 1772, '负载(hp)': 1,
#          '通道': 'DE'},
#         {'文件名': '120.mat', '故障类型': 'Ball', '故障直径(英寸)': '0.007', '转速(rpm)': 1750, '负载(hp)': 2,
#          '通道': 'DE'},
#         {'文件名': '121.mat', '故障类型': 'Ball', '故障直径(英寸)': '0.007', '转速(rpm)': 1730, '负载(hp)': 3,
#          '通道': 'DE'},
#         # Ball 0.014
#         {'文件名': '185.mat', '故障类型': 'Ball', '故障直径(英寸)': '0.014', '转速(rpm)': 1797, '负载(hp)': 0,
#          '通道': 'DE'},
#         {'文件名': '186.mat', '故障类型': 'Ball', '故障直径(英寸)': '0.014', '转速(rpm)': 1772, '负载(hp)': 1,
#          '通道': 'DE'},
#         {'文件名': '187.mat', '故障类型': 'Ball', '故障直径(英寸)': '0.014', '转速(rpm)': 1750, '负载(hp)': 2,
#          '通道': 'DE'},
#         {'文件名': '188.mat', '故障类型': 'Ball', '故障直径(英寸)': '0.014', '转速(rpm)': 1730, '负载(hp)': 3,
#          '通道': 'DE'},
#     ]
#
#     df = pd.DataFrame(data_files)
#
#     print("\n源域数据筛选明细表：")
#     print("-" * 90)
#     print(df.to_string(index=False))
#
#     # 统计
#     print("\n" + "-" * 90)
#     print("筛选统计：")
#     for fault_type in ['Normal', 'IR', 'OR', 'Ball']:
#         count = len([d for d in data_files if d['故障类型'] == fault_type])
#         print(f"  {fault_type}: {count} 个文件")
#     print(f"  总计: {len(data_files)} 个文件")
#
#     # 保存表格
#     df.to_csv('source_data_selection_table.csv', index=False, encoding='utf-8-sig')
#     print("\n已保存: source_data_selection_table.csv")
#
#     return df
#
#
# def generate_fault_mechanism_figure():
#     """生成故障机理对比可视化图（修复图例问题）"""
#
#     print("\n" + "=" * 70)
#     print("生成故障机理可视化图")
#     print("=" * 70)
#
#     fig, axes = plt.subplots(4, 3, figsize=(15, 12))
#
#     fs = 12000  # 采样频率
#     fr = 29.95  # 转频 (1797rpm/60)
#     N = 512  # 样本长度
#     t = np.arange(N) / fs * 1000  # 时间(ms)
#
#     # CWRU轴承6205-2RS参数
#     n = 9  # 滚动体数量
#     d = 7.94  # 滚动体直径 mm
#     D = 39.04  # 节径 mm
#
#     # 计算故障频率
#     BPFO = n / 2 * fr * (1 - d / D)  # ~107 Hz
#     BPFI = n / 2 * fr * (1 + d / D)  # ~162 Hz
#     BSF = D / (2 * d) * fr * (1 - (d / D) ** 2)  # ~70 Hz
#
#     print(f"故障特征频率: BPFO={BPFO:.1f}Hz, BPFI={BPFI:.1f}Hz, 2xBSF={2 * BSF:.1f}Hz")
#
#     fault_configs = [
#         ('Normal', '正常状态', None),
#         ('OR', '外圈故障', BPFO),
#         ('IR', '内圈故障', BPFI),
#         ('Ball', '滚动体故障', 2 * BSF),
#     ]
#
#     for i, (fault_type, fault_name, fault_freq) in enumerate(fault_configs):
#
#         # 生成模拟信号
#         np.random.seed(42 + i)
#
#         if fault_type == 'Normal':
#             signal_data = np.random.randn(N) * 0.1
#         elif fault_type == 'OR':
#             signal_data = np.zeros(N)
#             impulse_interval = int(fs / BPFO)
#             for j in range(0, N, impulse_interval):
#                 if j < N:
#                     decay = np.exp(-np.arange(min(30, N - j)) / 6)
#                     carrier = np.sin(2 * np.pi * 3000 * np.arange(min(30, N - j)) / fs)
#                     signal_data[j:j + len(decay)] += 0.8 * decay * carrier
#             signal_data += np.random.randn(N) * 0.05
#         elif fault_type == 'IR':
#             signal_data = np.zeros(N)
#             impulse_interval = int(fs / BPFI)
#             for j in range(0, N, impulse_interval):
#                 if j < N:
#                     mod = 0.4 + 0.6 * np.sin(2 * np.pi * fr * j / fs)
#                     decay = np.exp(-np.arange(min(25, N - j)) / 5)
#                     carrier = np.sin(2 * np.pi * 3200 * np.arange(min(25, N - j)) / fs)
#                     signal_data[j:j + len(decay)] += mod * decay * carrier
#             signal_data += np.random.randn(N) * 0.05
#         elif fault_type == 'Ball':
#             signal_data = np.zeros(N)
#             impulse_interval = int(fs / (2 * BSF))
#             for j in range(0, N, impulse_interval):
#                 if j < N:
#                     decay = np.exp(-np.arange(min(20, N - j)) / 4)
#                     carrier = np.sin(2 * np.pi * 2800 * np.arange(min(20, N - j)) / fs)
#                     signal_data[j:j + len(decay)] += 0.6 * decay * carrier
#             signal_data += np.random.randn(N) * 0.05
#
#         # 第1列：时域波形
#         ax1 = axes[i, 0]
#         ax1.plot(t, signal_data, 'b-', linewidth=0.6)
#         ax1.set_title(f'{fault_name} - 时域波形', fontsize=11, fontweight='bold')
#         ax1.set_xlabel('时间 (ms)', fontsize=9)
#         ax1.set_ylabel('幅值', fontsize=9)
#         ax1.grid(True, alpha=0.3)
#         ax1.set_xlim(0, t[-1])
#
#         feature_text = {
#             'Normal': '无周期性冲击',
#             'OR': '等幅等间隔冲击',
#             'IR': '幅值调制冲击',
#             'Ball': '复杂调制冲击'
#         }
#         ax1.text(0.98, 0.95, feature_text[fault_type], transform=ax1.transAxes,
#                  fontsize=9, ha='right', va='top', color='red',
#                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
#
#         # 第2列：频谱
#         ax2 = axes[i, 1]
#         freqs = np.fft.rfftfreq(N, 1 / fs)
#         spectrum = np.abs(np.fft.rfft(signal_data)) / N * 2
#         ax2.plot(freqs, spectrum, 'g-', linewidth=0.7)
#         ax2.set_title(f'{fault_name} - 频谱', fontsize=11, fontweight='bold')
#         ax2.set_xlabel('频率 (Hz)', fontsize=9)
#         ax2.set_ylabel('幅值', fontsize=9)
#         ax2.set_xlim(0, 2000)
#         ax2.grid(True, alpha=0.3)
#
#         # 第3列：包络谱
#         ax3 = axes[i, 2]
#
#         try:
#             nyq = fs / 2
#             b, a = butter(4, [500 / nyq, 4000 / nyq], btype='band')
#             filtered = filtfilt(b, a, signal_data)
#             envelope = np.abs(hilbert(filtered))
#             envelope = envelope - np.mean(envelope)
#
#             env_freqs = np.fft.rfftfreq(N, 1 / fs)
#             env_spectrum = np.abs(np.fft.rfft(envelope)) / N * 2
#
#             ax3.plot(env_freqs, env_spectrum, 'r-', linewidth=0.7, label='包络谱')
#
#             # 绘制故障特征频率线
#             ax3.axvline(BPFO, color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
#             ax3.axvline(BPFI, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
#             ax3.axvline(2 * BSF, color='purple', linestyle='--', alpha=0.7, linewidth=1.5)
#
#         except Exception as e:
#             ax3.text(0.5, 0.5, f'处理失败: {e}', transform=ax3.transAxes, ha='center')
#
#         ax3.set_title(f'{fault_name} - 包络谱', fontsize=11, fontweight='bold')
#         ax3.set_xlabel('频率 (Hz)', fontsize=9)
#         ax3.set_ylabel('幅值', fontsize=9)
#         ax3.set_xlim(0, 400)
#         ax3.grid(True, alpha=0.3)
#
#     # 创建统一图例（放在图的底部）
#     legend_elements = [
#         Line2D([0], [0], color='blue', linestyle='--', linewidth=2,
#                label=f'BPFO (外圈故障频率) = {BPFO:.1f} Hz'),
#         Line2D([0], [0], color='orange', linestyle='--', linewidth=2,
#                label=f'BPFI (内圈故障频率) = {BPFI:.1f} Hz'),
#         Line2D([0], [0], color='purple', linestyle='--', linewidth=2,
#                label=f'2×BSF (滚动体故障频率) = {2 * BSF:.1f} Hz'),
#     ]
#
#     fig.legend(handles=legend_elements, loc='lower center', ncol=3,
#                fontsize=10, frameon=True, fancybox=True, shadow=True,
#                bbox_to_anchor=(0.5, -0.02))
#
#     plt.suptitle('轴承故障振动机理与特征对比\n(CWRU轴承6205-2RS, 转速1797rpm, 采样率12kHz)',
#                  fontsize=14, fontweight='bold')
#     plt.tight_layout(rect=[0, 0.05, 1, 0.96])
#     plt.savefig('fault_mechanism_comparison.png', dpi=200, bbox_inches='tight')
#     print("已保存: fault_mechanism_comparison.png")
#     plt.close()
#
#
# def generate_domain_comparison_figure():
#     """生成源域与目标域对比图（分成两张图）"""
#
#     print("\n" + "=" * 70)
#     print("生成源域与目标域对比图")
#     print("=" * 70)
#
#     # ========== 图1: 参数对比柱状图 ==========
#     fig1, ax1 = plt.subplots(figsize=(10, 6))
#
#     params = ['转速\n(rpm)', 'BPFO\n(Hz)', 'BPFI\n(Hz)', 'BSF\n(Hz)', '采样率\n(kHz)']
#     source_vals = [1750, 107, 162, 70, 12]
#     target_vals = [600, 60, 90, 22, 32]
#
#     x = np.arange(len(params))
#     width = 0.35
#
#     bars1 = ax1.bar(x - width / 2, source_vals, width, label='源域(CWRU)',
#                     color='steelblue', edgecolor='black', linewidth=1.2)
#     bars2 = ax1.bar(x + width / 2, target_vals, width, label='目标域(列车)',
#                     color='coral', edgecolor='black', linewidth=1.2)
#
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(params, fontsize=11)
#     ax1.set_ylabel('数值', fontsize=12)
#     ax1.set_title('源域与目标域参数对比', fontsize=14, fontweight='bold')
#     ax1.legend(fontsize=11, loc='upper right')
#     ax1.grid(True, alpha=0.3, axis='y')
#
#     # 添加数值标签
#     for bar in bars1:
#         height = bar.get_height()
#         ax1.text(bar.get_x() + bar.get_width() / 2, height + 20,
#                  f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
#     for bar in bars2:
#         height = bar.get_height()
#         ax1.text(bar.get_x() + bar.get_width() / 2, height + 20,
#                  f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
#
#     ax1.set_ylim(0, max(source_vals) * 1.15)
#
#     plt.tight_layout()
#     plt.savefig('domain_parameter_comparison.png', dpi=200, bbox_inches='tight')
#     print("已保存: domain_parameter_comparison.png")
#     plt.close()
#
#     # ========== 图2: 跨域迁移挑战表格 ==========
#     fig2, ax2 = plt.subplots(figsize=(12, 6))
#     ax2.axis('off')
#
#     challenges = [
#         ['挑战', '源域', '目标域', '解决方案'],
#         ['转速差异', '~1750 rpm', '~600 rpm', '学习与转速无关的形态特征'],
#         ['采样率', '12 kHz', '32 kHz', '重采样统一时间尺度'],
#         ['轴承类型', '深沟球轴承', '圆柱滚子轴承', '学习通用故障模式'],
#         ['环境噪声', '实验室(低)', '实际运行(高)', '域对抗训练提取鲁棒特征'],
#         ['数据标签', '有标签', '无标签', 'DANN无监督域适应'],
#     ]
#
#     table = ax2.table(cellText=challenges[1:], colLabels=challenges[0],
#                       loc='center', cellLoc='center')
#     table.auto_set_font_size(False)
#     table.set_fontsize(11)
#     table.scale(1.4, 2.2)
#
#     # 设置表头样式
#     for i in range(4):
#         table[(0, i)].set_facecolor('#2C3E50')
#         table[(0, i)].set_text_props(color='white', fontweight='bold')
#
#     # 设置交替行颜色
#     for i in range(1, 6):
#         color = '#EBF5FB' if i % 2 == 1 else '#FEF9E7'
#         for j in range(4):
#             table[(i, j)].set_facecolor(color)
#
#     ax2.set_title('跨域迁移的关键挑战与解决方案', fontsize=14, fontweight='bold', pad=30)
#
#     plt.tight_layout()
#     plt.savefig('domain_transfer_challenges.png', dpi=200, bbox_inches='tight')
#     print("已保存: domain_transfer_challenges.png")
#     plt.close()
#
#
# def generate_fault_frequency_table():
#     """生成故障频率计算表"""
#
#     print("\n" + "=" * 70)
#     print("故障特征频率计算")
#     print("=" * 70)
#
#     print("\nCWRU轴承 6205-2RS 参数:")
#     print("  滚动体数量 n = 9")
#     print("  滚动体直径 d = 7.94 mm")
#     print("  节径 D = 39.04 mm")
#     print("  接触角 θ = 0°")
#
#     rpms = [1797, 1772, 1750, 1730]
#
#     freq_data = []
#     for rpm in rpms:
#         fr = rpm / 60
#         n, d, D = 9, 7.94, 39.04
#
#         BPFO = n / 2 * fr * (1 - d / D)
#         BPFI = n / 2 * fr * (1 + d / D)
#         BSF = D / (2 * d) * fr * (1 - (d / D) ** 2)
#         FTF = fr / 2 * (1 - d / D)
#
#         freq_data.append({
#             '转速(rpm)': rpm,
#             '转频fr(Hz)': round(fr, 2),
#             'BPFO(Hz)': round(BPFO, 2),
#             'BPFI(Hz)': round(BPFI, 2),
#             'BSF(Hz)': round(BSF, 2),
#             'FTF(Hz)': round(FTF, 2)
#         })
#
#     df = pd.DataFrame(freq_data)
#     print("\n不同转速下的故障特征频率:")
#     print(df.to_string(index=False))
#
#     df.to_csv('fault_characteristic_frequencies.csv', index=False, encoding='utf-8-sig')
#     print("\n已保存: fault_characteristic_frequencies.csv")
#
#     return df
#
#
# def main():
#     """主函数"""
#
#     print("=" * 70)
#     print("任务1补充 - 源域数据筛选 + 故障机理可视化（修复版）")
#     print("=" * 70)
#
#     # 1. 生成数据筛选表格
#     generate_data_selection_table()
#
#     # 2. 生成故障频率表
#     generate_fault_frequency_table()
#
#     # 3. 生成故障机理对比图
#     generate_fault_mechanism_figure()
#
#     # 4. 生成域对比分析图（分成两张）
#     generate_domain_comparison_figure()
#
#     # 总结
#     print("\n" + "=" * 70)
#     print("任务1补充完成!")
#     print("=" * 70)
#     print("""
# 生成的文件:
#   1. source_data_selection_table.csv      - 源域数据筛选表格（修复引号问题）
#   2. fault_characteristic_frequencies.csv - 故障特征频率表
#   3. fault_mechanism_comparison.png       - 故障机理对比图（添加统一图例）
#   4. domain_parameter_comparison.png      - 源域与目标域参数对比（独立图片）
#   5. domain_transfer_challenges.png       - 跨域迁移挑战表格（独立图片）
# """)
#
#
# if __name__ == "__main__":
#     main()


# task1_complete_analysis.py
# 功能：任务1补充 - 源域数据筛选表格 + 故障机理可视化（修复版v2）
# ====================================================================

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import hilbert, butter, filtfilt
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_data_selection_table():
    """生成源域数据筛选表格（修复：故障位置单独一列）"""

    print("=" * 70)
    print("源域数据筛选策略")
    print("=" * 70)

    print("""
【筛选原则】
1. 选择DE（驱动端）通道数据，信号质量更好
2. 选择0.007英寸和0.014英寸故障直径（特征明显且不过于严重）
3. 外圈故障选择6点钟位置（承载区，特征最典型）
4. 包含所有转速条件，增加数据多样性
5. 各类故障样本数量均衡
""")

    # 创建数据表格（故障位置单独一列）
    data_files = [
        # Normal
        {'文件名': '97.mat', '故障类型': 'Normal', '故障直径': '-', '故障位置': '-', '转速(rpm)': 1797, '负载(hp)': 0,
         '通道': 'DE'},
        {'文件名': '98.mat', '故障类型': 'Normal', '故障直径': '-', '故障位置': '-', '转速(rpm)': 1772, '负载(hp)': 1,
         '通道': 'DE'},
        {'文件名': '99.mat', '故障类型': 'Normal', '故障直径': '-', '故障位置': '-', '转速(rpm)': 1750, '负载(hp)': 2,
         '通道': 'DE'},
        {'文件名': '100.mat', '故障类型': 'Normal', '故障直径': '-', '故障位置': '-', '转速(rpm)': 1730, '负载(hp)': 3,
         '通道': 'DE'},
        # IR 0.007
        {'文件名': '105.mat', '故障类型': 'IR', '故障直径': '0.007', '故障位置': '-', '转速(rpm)': 1797, '负载(hp)': 0,
         '通道': 'DE'},
        {'文件名': '106.mat', '故障类型': 'IR', '故障直径': '0.007', '故障位置': '-', '转速(rpm)': 1772, '负载(hp)': 1,
         '通道': 'DE'},
        {'文件名': '107.mat', '故障类型': 'IR', '故障直径': '0.007', '故障位置': '-', '转速(rpm)': 1750, '负载(hp)': 2,
         '通道': 'DE'},
        {'文件名': '108.mat', '故障类型': 'IR', '故障直径': '0.007', '故障位置': '-', '转速(rpm)': 1730, '负载(hp)': 3,
         '通道': 'DE'},
        # IR 0.014
        {'文件名': '169.mat', '故障类型': 'IR', '故障直径': '0.014', '故障位置': '-', '转速(rpm)': 1797, '负载(hp)': 0,
         '通道': 'DE'},
        {'文件名': '170.mat', '故障类型': 'IR', '故障直径': '0.014', '故障位置': '-', '转速(rpm)': 1772, '负载(hp)': 1,
         '通道': 'DE'},
        {'文件名': '171.mat', '故障类型': 'IR', '故障直径': '0.014', '故障位置': '-', '转速(rpm)': 1750, '负载(hp)': 2,
         '通道': 'DE'},
        {'文件名': '172.mat', '故障类型': 'IR', '故障直径': '0.014', '故障位置': '-', '转速(rpm)': 1730, '负载(hp)': 3,
         '通道': 'DE'},
        # OR 0.007 @6点钟
        {'文件名': '130.mat', '故障类型': 'OR', '故障直径': '0.007', '故障位置': '6点钟', '转速(rpm)': 1797,
         '负载(hp)': 0, '通道': 'DE'},
        {'文件名': '131.mat', '故障类型': 'OR', '故障直径': '0.007', '故障位置': '6点钟', '转速(rpm)': 1772,
         '负载(hp)': 1, '通道': 'DE'},
        {'文件名': '132.mat', '故障类型': 'OR', '故障直径': '0.007', '故障位置': '6点钟', '转速(rpm)': 1750,
         '负载(hp)': 2, '通道': 'DE'},
        {'文件名': '133.mat', '故障类型': 'OR', '故障直径': '0.007', '故障位置': '6点钟', '转速(rpm)': 1730,
         '负载(hp)': 3, '通道': 'DE'},
        # OR 0.014 @6点钟
        {'文件名': '197.mat', '故障类型': 'OR', '故障直径': '0.014', '故障位置': '6点钟', '转速(rpm)': 1797,
         '负载(hp)': 0, '通道': 'DE'},
        {'文件名': '198.mat', '故障类型': 'OR', '故障直径': '0.014', '故障位置': '6点钟', '转速(rpm)': 1772,
         '负载(hp)': 1, '通道': 'DE'},
        {'文件名': '199.mat', '故障类型': 'OR', '故障直径': '0.014', '故障位置': '6点钟', '转速(rpm)': 1750,
         '负载(hp)': 2, '通道': 'DE'},
        {'文件名': '200.mat', '故障类型': 'OR', '故障直径': '0.014', '故障位置': '6点钟', '转速(rpm)': 1730,
         '负载(hp)': 3, '通道': 'DE'},
        # Ball 0.007
        {'文件名': '118.mat', '故障类型': 'Ball', '故障直径': '0.007', '故障位置': '-', '转速(rpm)': 1797,
         '负载(hp)': 0, '通道': 'DE'},
        {'文件名': '119.mat', '故障类型': 'Ball', '故障直径': '0.007', '故障位置': '-', '转速(rpm)': 1772,
         '负载(hp)': 1, '通道': 'DE'},
        {'文件名': '120.mat', '故障类型': 'Ball', '故障直径': '0.007', '故障位置': '-', '转速(rpm)': 1750,
         '负载(hp)': 2, '通道': 'DE'},
        {'文件名': '121.mat', '故障类型': 'Ball', '故障直径': '0.007', '故障位置': '-', '转速(rpm)': 1730,
         '负载(hp)': 3, '通道': 'DE'},
        # Ball 0.014
        {'文件名': '185.mat', '故障类型': 'Ball', '故障直径': '0.014', '故障位置': '-', '转速(rpm)': 1797,
         '负载(hp)': 0, '通道': 'DE'},
        {'文件名': '186.mat', '故障类型': 'Ball', '故障直径': '0.014', '故障位置': '-', '转速(rpm)': 1772,
         '负载(hp)': 1, '通道': 'DE'},
        {'文件名': '187.mat', '故障类型': 'Ball', '故障直径': '0.014', '故障位置': '-', '转速(rpm)': 1750,
         '负载(hp)': 2, '通道': 'DE'},
        {'文件名': '188.mat', '故障类型': 'Ball', '故障直径': '0.014', '故障位置': '-', '转速(rpm)': 1730,
         '负载(hp)': 3, '通道': 'DE'},
    ]

    df = pd.DataFrame(data_files)

    print("\n源域数据筛选明细表：")
    print("-" * 100)
    print(df.to_string(index=False))

    # 统计
    print("\n" + "-" * 100)
    print("筛选统计：")
    for fault_type in ['Normal', 'IR', 'OR', 'Ball']:
        count = len([d for d in data_files if d['故障类型'] == fault_type])
        print(f"  {fault_type}: {count} 个文件")
    print(f"  总计: {len(data_files)} 个文件")

    # 保存表格
    df.to_csv('source_data_selection_table.csv', index=False, encoding='utf-8-sig')
    print("\n已保存: source_data_selection_table.csv")

    return df


def generate_fault_mechanism_figure():
    """生成故障机理对比可视化图（修复：图例在每个包络谱子图内）"""

    print("\n" + "=" * 70)
    print("生成故障机理可视化图")
    print("=" * 70)

    fig, axes = plt.subplots(4, 3, figsize=(15, 13))

    fs = 12000
    fr = 29.95
    N = 512
    t = np.arange(N) / fs * 1000

    # CWRU轴承6205-2RS参数
    n = 9
    d = 7.94
    D = 39.04

    # 计算故障频率
    BPFO = n / 2 * fr * (1 - d / D)
    BPFI = n / 2 * fr * (1 + d / D)
    BSF = D / (2 * d) * fr * (1 - (d / D) ** 2)

    print(f"故障特征频率: BPFO={BPFO:.1f}Hz, BPFI={BPFI:.1f}Hz, 2xBSF={2 * BSF:.1f}Hz")

    fault_configs = [
        ('Normal', '正常状态', None),
        ('OR', '外圈故障', BPFO),
        ('IR', '内圈故障', BPFI),
        ('Ball', '滚动体故障', 2 * BSF),
    ]

    for i, (fault_type, fault_name, fault_freq) in enumerate(fault_configs):

        np.random.seed(42 + i)

        if fault_type == 'Normal':
            signal_data = np.random.randn(N) * 0.1
        elif fault_type == 'OR':
            signal_data = np.zeros(N)
            impulse_interval = int(fs / BPFO)
            for j in range(0, N, impulse_interval):
                if j < N:
                    decay = np.exp(-np.arange(min(30, N - j)) / 6)
                    carrier = np.sin(2 * np.pi * 3000 * np.arange(min(30, N - j)) / fs)
                    signal_data[j:j + len(decay)] += 0.8 * decay * carrier
            signal_data += np.random.randn(N) * 0.05
        elif fault_type == 'IR':
            signal_data = np.zeros(N)
            impulse_interval = int(fs / BPFI)
            for j in range(0, N, impulse_interval):
                if j < N:
                    mod = 0.4 + 0.6 * np.sin(2 * np.pi * fr * j / fs)
                    decay = np.exp(-np.arange(min(25, N - j)) / 5)
                    carrier = np.sin(2 * np.pi * 3200 * np.arange(min(25, N - j)) / fs)
                    signal_data[j:j + len(decay)] += mod * decay * carrier
            signal_data += np.random.randn(N) * 0.05
        elif fault_type == 'Ball':
            signal_data = np.zeros(N)
            impulse_interval = int(fs / (2 * BSF))
            for j in range(0, N, impulse_interval):
                if j < N:
                    decay = np.exp(-np.arange(min(20, N - j)) / 4)
                    carrier = np.sin(2 * np.pi * 2800 * np.arange(min(20, N - j)) / fs)
                    signal_data[j:j + len(decay)] += 0.6 * decay * carrier
            signal_data += np.random.randn(N) * 0.05

        # 第1列：时域波形
        ax1 = axes[i, 0]
        ax1.plot(t, signal_data, 'b-', linewidth=0.6)
        ax1.set_title(f'{fault_name} - 时域波形', fontsize=11, fontweight='bold')
        ax1.set_xlabel('时间 (ms)', fontsize=9)
        ax1.set_ylabel('幅值', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, t[-1])

        feature_text = {
            'Normal': '无周期性冲击',
            'OR': '等幅等间隔冲击',
            'IR': '幅值调制冲击',
            'Ball': '复杂调制冲击'
        }
        ax1.text(0.98, 0.95, feature_text[fault_type], transform=ax1.transAxes,
                 fontsize=9, ha='right', va='top', color='red',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # 第2列：频谱
        ax2 = axes[i, 1]
        freqs = np.fft.rfftfreq(N, 1 / fs)
        spectrum = np.abs(np.fft.rfft(signal_data)) / N * 2
        ax2.plot(freqs, spectrum, 'g-', linewidth=0.7)
        ax2.set_title(f'{fault_name} - 频谱', fontsize=11, fontweight='bold')
        ax2.set_xlabel('频率 (Hz)', fontsize=9)
        ax2.set_ylabel('幅值', fontsize=9)
        ax2.set_xlim(0, 2000)
        ax2.grid(True, alpha=0.3)

        # 第3列：包络谱（每个子图内添加图例）
        ax3 = axes[i, 2]

        try:
            nyq = fs / 2
            b, a = butter(4, [500 / nyq, 4000 / nyq], btype='band')
            filtered = filtfilt(b, a, signal_data)
            envelope = np.abs(hilbert(filtered))
            envelope = envelope - np.mean(envelope)

            env_freqs = np.fft.rfftfreq(N, 1 / fs)
            env_spectrum = np.abs(np.fft.rfft(envelope)) / N * 2

            ax3.plot(env_freqs, env_spectrum, 'r-', linewidth=0.7)

            # 绘制故障特征频率线（带标签用于图例）
            line1 = ax3.axvline(BPFO, color='blue', linestyle='--', alpha=0.8, linewidth=1.5,
                                label=f'BPFO={BPFO:.0f}Hz')
            line2 = ax3.axvline(BPFI, color='orange', linestyle='--', alpha=0.8, linewidth=1.5,
                                label=f'BPFI={BPFI:.0f}Hz')
            line3 = ax3.axvline(2 * BSF, color='purple', linestyle='--', alpha=0.8, linewidth=1.5,
                                label=f'2BSF={2 * BSF:.0f}Hz')

            # 在每个子图内添加图例
            ax3.legend(loc='upper right', fontsize=7, framealpha=0.9,
                       handlelength=1.5, handletextpad=0.3)

        except Exception as e:
            ax3.text(0.5, 0.5, f'处理失败: {e}', transform=ax3.transAxes, ha='center')

        ax3.set_title(f'{fault_name} - 包络谱', fontsize=11, fontweight='bold')
        ax3.set_xlabel('频率 (Hz)', fontsize=9)
        ax3.set_ylabel('幅值', fontsize=9)
        ax3.set_xlim(0, 400)
        ax3.grid(True, alpha=0.3)

    plt.suptitle('轴承故障振动机理与特征对比\n(CWRU轴承6205-2RS, 转速1797rpm, 采样率12kHz)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('fault_mechanism_comparison.png', dpi=200, bbox_inches='tight')
    print("已保存: fault_mechanism_comparison.png")
    plt.close()


def generate_domain_comparison_figure():
    """生成源域与目标域对比图"""

    print("\n" + "=" * 70)
    print("生成源域与目标域对比图")
    print("=" * 70)

    # ========== 图1: 参数对比柱状图 ==========
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    params = ['转速\n(rpm)', 'BPFO\n(Hz)', 'BPFI\n(Hz)', 'BSF\n(Hz)', '采样率\n(kHz)']
    source_vals = [1750, 107, 162, 70, 12]
    target_vals = [600, 60, 90, 22, 32]

    x = np.arange(len(params))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, source_vals, width, label='源域(CWRU)',
                    color='steelblue', edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width / 2, target_vals, width, label='目标域(列车)',
                    color='coral', edgecolor='black', linewidth=1.2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(params, fontsize=11)
    ax1.set_ylabel('数值', fontsize=12)
    ax1.set_title('源域与目标域参数对比', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 20,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 20,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylim(0, max(source_vals) * 1.15)

    plt.tight_layout()
    plt.savefig('domain_parameter_comparison.png', dpi=200, bbox_inches='tight')
    print("已保存: domain_parameter_comparison.png")
    plt.close()

    # ========== 图2: 跨域迁移挑战表格（修复标题距离）==========
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.axis('off')

    challenges = [
        ['挑战', '源域', '目标域', '解决方案'],
        ['转速差异', '~1750 rpm', '~600 rpm', '学习与转速无关的形态特征'],
        ['采样率', '12 kHz', '32 kHz', '重采样统一时间尺度'],
        ['轴承类型', '深沟球轴承', '圆柱滚子轴承', '学习通用故障模式'],
        ['环境噪声', '实验室(低)', '实际运行(高)', '域对抗训练提取鲁棒特征'],
        ['数据标签', '有标签', '无标签', 'DANN无监督域适应'],
    ]

    # 创建表格，位置靠上
    table = ax2.table(cellText=challenges[1:], colLabels=challenges[0],
                      loc='upper center', cellLoc='center',
                      bbox=[0.05, 0.1, 0.9, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
        table[(0, i)].set_height(0.15)

    # 设置数据行样式
    for i in range(1, 6):
        color = '#EBF5FB' if i % 2 == 1 else '#FEF9E7'
        for j in range(4):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_height(0.12)

    # 标题紧贴表格上方
    ax2.set_title('跨域迁移的关键挑战与解决方案', fontsize=14, fontweight='bold',
                  pad=5, y=0.98)

    plt.savefig('domain_transfer_challenges.png', dpi=200, bbox_inches='tight',
                pad_inches=0.1)
    print("已保存: domain_transfer_challenges.png")
    plt.close()


def generate_fault_frequency_table():
    """生成故障频率计算表"""

    print("\n" + "=" * 70)
    print("故障特征频率计算")
    print("=" * 70)

    print("\nCWRU轴承 6205-2RS 参数:")
    print("  滚动体数量 n = 9")
    print("  滚动体直径 d = 7.94 mm")
    print("  节径 D = 39.04 mm")
    print("  接触角 θ = 0°")

    rpms = [1797, 1772, 1750, 1730]

    freq_data = []
    for rpm in rpms:
        fr = rpm / 60
        n, d, D = 9, 7.94, 39.04

        BPFO = n / 2 * fr * (1 - d / D)
        BPFI = n / 2 * fr * (1 + d / D)
        BSF = D / (2 * d) * fr * (1 - (d / D) ** 2)
        FTF = fr / 2 * (1 - d / D)

        freq_data.append({
            '转速(rpm)': rpm,
            '转频fr(Hz)': round(fr, 2),
            'BPFO(Hz)': round(BPFO, 2),
            'BPFI(Hz)': round(BPFI, 2),
            'BSF(Hz)': round(BSF, 2),
            'FTF(Hz)': round(FTF, 2)
        })

    df = pd.DataFrame(freq_data)
    print("\n不同转速下的故障特征频率:")
    print(df.to_string(index=False))

    df.to_csv('fault_characteristic_frequencies.csv', index=False, encoding='utf-8-sig')
    print("\n已保存: fault_characteristic_frequencies.csv")

    return df


def main():
    """主函数"""

    print("=" * 70)
    print("任务1补充 - 源域数据筛选 + 故障机理可视化（修复版v2）")
    print("=" * 70)

    # 1. 生成数据筛选表格
    generate_data_selection_table()

    # 2. 生成故障频率表
    generate_fault_frequency_table()

    # 3. 生成故障机理对比图
    generate_fault_mechanism_figure()

    # 4. 生成域对比分析图
    generate_domain_comparison_figure()

    print("\n" + "=" * 70)
    print("任务1补充完成!")
    print("=" * 70)
    print("""
生成的文件:
  1. source_data_selection_table.csv      - 源域数据筛选表格（故障位置单独一列）
  2. fault_characteristic_frequencies.csv - 故障特征频率表
  3. fault_mechanism_comparison.png       - 故障机理对比图（每个包络谱有图例）
  4. domain_parameter_comparison.png      - 源域与目标域参数对比
  5. domain_transfer_challenges.png       - 跨域迁移挑战表格（标题贴近表格）
""")


if __name__ == "__main__":
    main()
