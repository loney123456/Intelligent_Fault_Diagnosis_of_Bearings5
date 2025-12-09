# generate_final_report.py
# 功能：生成最终的诊断报告和论文用图表
# ====================================================================

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_final_report():
    """生成最终诊断报告"""

    # 完整的诊断结果
    results = [
        {'File': 'A', 'Label': 'IR', 'Label_CN': '内圈故障', 'Vote': 79.7, 'Conf': 0.824, 'Reliability': '★★☆',
         'Rel_Level': '可靠', 'Note': ''},
        {'File': 'B', 'Label': 'Ball', 'Label_CN': '滚动体故障', 'Vote': 89.0, 'Conf': 0.923, 'Reliability': '★★☆',
         'Rel_Level': '可靠', 'Note': ''},
        {'File': 'C', 'Label': 'OR', 'Label_CN': '外圈故障', 'Vote': 78.1, 'Conf': 0.898, 'Reliability': '★★☆',
         'Rel_Level': '可靠', 'Note': ''},
        {'File': 'D', 'Label': 'OR', 'Label_CN': '外圈故障', 'Vote': 54.8, 'Conf': 0.694, 'Reliability': '☆☆☆',
         'Rel_Level': '需复核', 'Note': '轻微/早期故障'},
        {'File': 'E', 'Label': 'OR', 'Label_CN': '外圈故障', 'Vote': 52.1, 'Conf': 0.964, 'Reliability': '★☆☆',
         'Rel_Level': '一般', 'Note': '存在强冲击脉冲'},
        {'File': 'F', 'Label': 'OR', 'Label_CN': '外圈故障', 'Vote': 41.4, 'Conf': 0.658, 'Reliability': '☆☆☆',
         'Rel_Level': '需复核', 'Note': '特征分散'},
        {'File': 'G', 'Label': 'Ball', 'Label_CN': '滚动体故障', 'Vote': 98.9, 'Conf': 0.968, 'Reliability': '★★★',
         'Rel_Level': '高可靠', 'Note': ''},
        {'File': 'H', 'Label': 'IR', 'Label_CN': '内圈故障', 'Vote': 99.2, 'Conf': 0.953, 'Reliability': '★★★',
         'Rel_Level': '高可靠', 'Note': ''},
        {'File': 'I', 'Label': 'OR', 'Label_CN': '外圈故障', 'Vote': 85.0, 'Conf': 0.900, 'Reliability': '★★☆',
         'Rel_Level': '可靠', 'Note': ''},
        {'File': 'J', 'Label': 'OR', 'Label_CN': '外圈故障', 'Vote': 77.0, 'Conf': 0.882, 'Reliability': '★★☆',
         'Rel_Level': '可靠', 'Note': ''},
        {'File': 'K', 'Label': 'IR', 'Label_CN': '内圈故障', 'Vote': 100.0, 'Conf': 0.980, 'Reliability': '★★★',
         'Rel_Level': '高可靠', 'Note': ''},
        {'File': 'L', 'Label': 'Ball', 'Label_CN': '滚动体故障', 'Vote': 75.9, 'Conf': 0.826, 'Reliability': '★★☆',
         'Rel_Level': '可靠', 'Note': ''},
        {'File': 'M', 'Label': 'IR', 'Label_CN': '内圈故障', 'Vote': 100.0, 'Conf': 0.984, 'Reliability': '★★★',
         'Rel_Level': '高可靠', 'Note': ''},
        {'File': 'N', 'Label': 'IR', 'Label_CN': '内圈故障', 'Vote': 77.3, 'Conf': 0.884, 'Reliability': '★★☆',
         'Rel_Level': '可靠', 'Note': ''},
        {'File': 'O', 'Label': 'Ball', 'Label_CN': '滚动体故障', 'Vote': 99.5, 'Conf': 0.969, 'Reliability': '★★★',
         'Rel_Level': '高可靠', 'Note': ''},
        {'File': 'P', 'Label': 'OR', 'Label_CN': '外圈故障', 'Vote': 66.3, 'Conf': 0.704, 'Reliability': '★☆☆',
         'Rel_Level': '一般', 'Note': ''},
    ]

    df = pd.DataFrame(results)

    # 保存CSV
    df.to_csv('final_diagnosis_results.csv', index=False, encoding='utf-8-sig')
    print("已保存: final_diagnosis_results.csv")

    # =========================================
    # 生成最终可视化图
    # =========================================

    fig = plt.figure(figsize=(16, 12))

    # 1. 诊断结果分布饼图
    ax1 = fig.add_subplot(2, 2, 1)
    label_counts = df['Label'].value_counts()
    colors = {'IR': '#FF6B6B', 'OR': '#4ECDC4', 'Ball': '#45B7D1', 'Normal': '#96CEB4'}
    pie_colors = [colors[label] for label in label_counts.index]

    # 添加中文标签
    label_names = {'IR': 'IR(内圈)', 'OR': 'OR(外圈)', 'Ball': 'Ball(滚动体)', 'Normal': 'Normal(正常)'}
    pie_labels = [f"{label_names.get(l, l)}\n{label_counts[l]}个" for l in label_counts.index]

    wedges, texts, autotexts = ax1.pie(label_counts.values, labels=pie_labels,
                                       autopct='%1.0f%%', colors=pie_colors,
                                       explode=[0.05] * len(label_counts),
                                       textprops={'fontsize': 10})
    ax1.set_title('目标域诊断结果分布\n(共16个文件)', fontsize=14, fontweight='bold')

    # 2. 投票比例和置信度条形图
    ax2 = fig.add_subplot(2, 2, 2)
    x = np.arange(len(df))
    width = 0.35
    bars1 = ax2.bar(x - width / 2, df['Vote'], width, label='投票比例(%)', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x + width / 2, df['Conf'] * 100, width, label='置信度(%)', color='coral', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['File'])
    ax2.set_ylabel('百分比 (%)')
    ax2.set_xlabel('文件')
    ax2.set_title('各文件投票比例与置信度对比', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.text(15.5, 72, '70%阈值', ha='right', va='bottom', color='red', fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.grid(True, alpha=0.3, axis='y')

    # 标注低置信度文件（使用文字而非emoji）
    for i, (v, c, note) in enumerate(zip(df['Vote'], df['Conf'], df['Note'])):
        if note:  # 有备注的文件
            ax2.annotate('!', (i, max(v, c * 100) + 3), ha='center', fontsize=14,
                         fontweight='bold', color='red')

    # 3. 可靠性分布
    ax3 = fig.add_subplot(2, 2, 3)
    rel_counts = df['Rel_Level'].value_counts()
    rel_order = ['高可靠', '可靠', '一般', '需复核']
    rel_counts = rel_counts.reindex([r for r in rel_order if r in rel_counts.index])

    rel_colors = {'高可靠': '#2ECC71', '可靠': '#3498DB', '一般': '#F39C12', '需复核': '#E74C3C'}
    bar_colors = [rel_colors[r] for r in rel_counts.index]
    bars = ax3.bar(rel_counts.index, rel_counts.values, color=bar_colors, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('文件数量')
    ax3.set_title('诊断可靠性分布', fontsize=14, fontweight='bold')

    # 在条形上标注文件名
    for bar, (rel, count) in zip(bars, rel_counts.items()):
        files = df[df['Rel_Level'] == rel]['File'].tolist()
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{', '.join(files)}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        # 在条形内部标注数量
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                 f"{count}个", ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    ax3.set_ylim(0, max(rel_counts.values) + 2)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 详细结果表格
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    # 创建表格数据
    table_data = []
    for _, row in df.iterrows():
        note = row['Note'] if row['Note'] else '-'
        table_data.append([row['File'], row['Label_CN'], f"{row['Vote']:.1f}%",
                           f"{row['Conf']:.3f}", row['Rel_Level'], note])

    table = ax4.table(cellText=table_data,
                      colLabels=['文件', '诊断标签', '投票比例', '置信度', '可靠性', '备注'],
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # 设置表头样式
    for i in range(6):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # 根据可靠性设置行颜色
    rel_color_map = {'高可靠': '#D5F4E6', '可靠': '#D6EAF8', '一般': '#FCF3CF', '需复核': '#FADBD8'}
    for i, row in enumerate(df.itertuples(), 1):
        color = rel_color_map.get(row.Rel_Level, 'white')
        for j in range(6):
            table[(i, j)].set_facecolor(color)

    ax4.set_title('完整诊断结果表', fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('高速列车轴承智能故障诊断 - 目标域预测结果总览',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('final_diagnosis_summary.png', dpi=150, bbox_inches='tight')
    print("已保存: final_diagnosis_summary.png")
    plt.close()

    # =========================================
    # 生成论文用的专业图表
    # =========================================

    generate_paper_figures(df)

    # =========================================
    # 生成简洁的提交用标签文件
    # =========================================

    submission = df[['File', 'Label']].copy()
    submission.columns = ['文件', '故障类型']
    submission.to_csv('target_labels_submission.csv', index=False, encoding='utf-8-sig')
    print("已保存: target_labels_submission.csv (提交用)")

    # 打印最终结果
    print("\n" + "=" * 70)
    print("最终诊断结果（用于提交）")
    print("=" * 70)

    # 使用ASCII字符代替emoji
    rel_icon = {'高可靠': '[+++]', '可靠': '[++ ]', '一般': '[+  ]', '需复核': '[!  ]'}

    for _, row in df.iterrows():
        icon = rel_icon.get(row['Rel_Level'], '[   ]')
        note = f" ({row['Note']})" if row['Note'] else ""
        print(f"  {icon} 文件{row['File']}: {row['Label']} ({row['Label_CN']}){note}")

    print("\n" + "-" * 70)

    # 统计
    print("\n统计摘要:")
    for label in ['Normal', 'IR', 'OR', 'Ball']:
        count = (df['Label'] == label).sum()
        files = df[df['Label'] == label]['File'].tolist()
        cn = {'Normal': '正常', 'IR': '内圈故障', 'OR': '外圈故障', 'Ball': '滚动体故障'}[label]
        if count > 0:
            print(f"  {label} ({cn}): {count}个 --> {', '.join(files)}")

    print("\n可靠性统计:")
    for rel in ['高可靠', '可靠', '一般', '需复核']:
        count = (df['Rel_Level'] == rel).sum()
        files = df[df['Rel_Level'] == rel]['File'].tolist()
        if count > 0:
            print(f"  {rel}: {count}个 --> {', '.join(files)}")

    return df


def generate_paper_figures(df):
    """生成论文专用的高质量图表"""

    # =========================================
    # 图1: 诊断结果与可靠性综合图
    # =========================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：各文件诊断结果
    ax1 = axes[0]

    # 按故障类型分组
    label_colors = {'IR': '#E74C3C', 'OR': '#3498DB', 'Ball': '#2ECC71', 'Normal': '#95A5A6'}

    files = df['File'].tolist()
    votes = df['Vote'].tolist()
    confs = [c * 100 for c in df['Conf'].tolist()]
    labels = df['Label'].tolist()
    colors = [label_colors[l] for l in labels]

    x = np.arange(len(files))
    bars = ax1.bar(x, votes, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)

    # 在条形上添加置信度点
    ax1.scatter(x, confs, color='darkred', s=50, zorder=5, label='置信度', marker='D')

    # 添加阈值线
    ax1.axhline(y=70, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(-0.5, 72, '70%', ha='left', va='bottom', fontsize=9, color='gray')

    ax1.set_xticks(x)
    ax1.set_xticklabels(files, fontsize=10)
    ax1.set_xlabel('文件编号', fontsize=12)
    ax1.set_ylabel('百分比 (%)', fontsize=12)
    ax1.set_title('(a) 各文件诊断投票比例与置信度', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3, axis='y')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', edgecolor='black', label='IR(内圈故障)'),
        Patch(facecolor='#3498DB', edgecolor='black', label='OR(外圈故障)'),
        Patch(facecolor='#2ECC71', edgecolor='black', label='Ball(滚动体故障)'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='darkred',
                   markersize=8, label='置信度')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # 右图：故障类型分布
    ax2 = axes[1]

    # 统计各类型数量
    label_counts = df['Label'].value_counts()
    label_order = ['IR', 'OR', 'Ball']
    counts = [label_counts.get(l, 0) for l in label_order]
    label_names = ['IR\n(内圈故障)', 'OR\n(外圈故障)', 'Ball\n(滚动体故障)']
    bar_colors = [label_colors[l] for l in label_order]

    bars = ax2.bar(label_names, counts, color=bar_colors, edgecolor='black', linewidth=1.5)

    # 在条形上标注数量和文件
    for i, (bar, label) in enumerate(zip(bars, label_order)):
        files_in_cat = df[df['Label'] == label]['File'].tolist()
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{len(files_in_cat)}个\n({', '.join(files_in_cat)})",
                 ha='center', va='bottom', fontsize=10)
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                 f"{len(files_in_cat)}", ha='center', va='center',
                 fontsize=16, fontweight='bold', color='white')

    ax2.set_ylabel('文件数量', fontsize=12)
    ax2.set_title('(b) 故障类型分布统计', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(counts) + 2.5)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('paper_figure_diagnosis_results.png', dpi=300, bbox_inches='tight')
    print("已保存: paper_figure_diagnosis_results.png (论文用)")
    plt.close()

    # =========================================
    # 图2: 可靠性分析热力图
    # =========================================

    fig, ax = plt.subplots(figsize=(12, 6))

    # 创建数据矩阵
    # x轴：文件，y轴：指标（投票比例、置信度）
    files = df['File'].tolist()
    data_matrix = np.array([df['Vote'].values, df['Conf'].values * 100])

    # 绘制热力图
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=30, vmax=100)

    # 设置刻度
    ax.set_xticks(np.arange(len(files)))
    ax.set_xticklabels(files, fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['投票比例(%)', '置信度(%)'], fontsize=11)

    # 添加数值标注
    for i in range(2):
        for j in range(len(files)):
            value = data_matrix[i, j]
            color = 'white' if value < 60 else 'black'
            ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label('百分比 (%)', fontsize=11)

    # 在文件名下方标注故障类型
    for j, (file, label) in enumerate(zip(files, df['Label'].tolist())):
        ax.text(j, 2.1, label, ha='center', va='top', fontsize=10,
                color=label_colors[label], fontweight='bold')

    ax.set_title('目标域诊断结果可靠性热力图', fontsize=14, fontweight='bold')
    ax.set_xlabel('文件编号（下方为诊断标签）', fontsize=12)

    plt.tight_layout()
    plt.savefig('paper_figure_reliability_heatmap.png', dpi=300, bbox_inches='tight')
    print("已保存: paper_figure_reliability_heatmap.png (论文用)")
    plt.close()

    # =========================================
    # 图3: 包络谱验证对比图
    # =========================================

    fig, ax = plt.subplots(figsize=(10, 6))

    # D、E、F三个文件的对比数据
    verification_data = {
        'D': {'model': 'OR', 'model_conf': 54.8, 'envelope': 'Ball', 'env_conf': 43.2, 'consistent': False},
        'E': {'model': 'OR', 'model_conf': 52.1, 'envelope': 'Ball', 'env_conf': 45.3, 'consistent': False},
        'F': {'model': 'OR', 'model_conf': 41.4, 'envelope': 'OR', 'env_conf': 35.3, 'consistent': True},
    }

    x = np.arange(3)
    width = 0.35

    model_confs = [verification_data[f]['model_conf'] for f in ['D', 'E', 'F']]
    env_confs = [verification_data[f]['env_conf'] for f in ['D', 'E', 'F']]

    bars1 = ax.bar(x - width / 2, model_confs, width, label='模型预测置信度', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width / 2, env_confs, width, label='包络谱诊断置信度', color='coral', edgecolor='black')

    # 在条形上标注诊断结果
    for i, f in enumerate(['D', 'E', 'F']):
        data = verification_data[f]
        ax.text(i - width / 2, model_confs[i] + 2, data['model'], ha='center', fontsize=11, fontweight='bold')
        ax.text(i + width / 2, env_confs[i] + 2, data['envelope'], ha='center', fontsize=11, fontweight='bold')

        # 标注是否一致
        if data['consistent']:
            ax.annotate('一致', (i, 60), ha='center', fontsize=12, color='green', fontweight='bold')
        else:
            ax.annotate('不一致', (i, 60), ha='center', fontsize=12, color='red', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['文件D', '文件E', '文件F'], fontsize=12)
    ax.set_ylabel('置信度 (%)', fontsize=12)
    ax.set_title('低置信度文件的包络谱验证对比', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 75)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('paper_figure_envelope_verification.png', dpi=300, bbox_inches='tight')
    print("已保存: paper_figure_envelope_verification.png (论文用)")
    plt.close()


if __name__ == "__main__":
    df = generate_final_report()
    print("\n" + "=" * 70)
    print("最终报告生成完成!")
    print("=" * 70)
    print("""
生成的文件列表:
  1. final_diagnosis_results.csv       - 完整诊断结果表
  2. target_labels_submission.csv      - 提交用标签文件
  3. final_diagnosis_summary.png       - 综合结果总览图
  4. paper_figure_diagnosis_results.png - 论文用诊断结果图
  5. paper_figure_reliability_heatmap.png - 论文用可靠性热力图
  6. paper_figure_envelope_verification.png - 论文用包络谱验证图
""")
