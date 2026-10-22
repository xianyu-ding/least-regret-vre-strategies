# -*- coding: utf-8 -*-
"""
多视角SCCA结构关键性分析可视化 - Nature期刊风格 (双面板版本)
Panel A: 准则关键性多视角散点图 (以comprehensive为参考)
Panel B: 视角一致性与分歧度综合分析
"""

from pipeline_common import (SCORE_DIR, RESULTS_DIR, WEIGHTS_DIR, INDICATOR_PATH, rank_scores, winner_shares, draw_dirichlet)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from pathlib import Path
import seaborn as sns
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

# ========== PATHS ==========
BASE_DIR = RESULTS_DIR
SCC_DIR = RESULTS_DIR / "scc"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========== COLORS - Nature期刊配色方案 ==========
PERSPECTIVE_COLORS = {
   'comprehensive': '#1f77b4',    # Classic blue
   'government': '#d62728',       # Red
   'news': '#ff7f0e',            # Orange  
   'research': '#2ca02c',        # Green
   'social_media': '#9467bd'     # Purple
}

CRITERIA_COLORS = {
   "Cost": "#1f77b4",
   "LOLE": "#d62728", 
   "Environment": "#ff7f0e",
   "CO2": "#2ca02c",
   "land_occupation": "#ff7f0e",
   "freshwater_ecotoxicity": "#ff7f0e",
   "human_toxicity": "#ff7f0e",
   "ionising_radiation": "#ff7f0e",
   "freshwater_eutrophication": "#ff7f0e",
   "marine_eutrophication": "#ff7f0e",
   "mineral_resource_depletion": "#ff7f0e"
}

# ========== NATURE期刊风格设置 ==========
def set_nature_style():
   """Nature期刊严谨风格设置"""
   # 重置matplotlib
   mpl.rcdefaults()
   
   # Nature期刊标准设置
   mpl.rcParams.update({
       # 图像质量
       "figure.dpi": 150,
       "savefig.dpi": 600,
       "savefig.bbox": "tight",
       "savefig.pad_inches": 0.1,
       "savefig.facecolor": "white",
       "savefig.edgecolor": "none",
       
       # 字体设置 - Nature期刊标准
       "font.family": "sans-serif",
       "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
       "font.size": 10,
       "font.weight": "normal",
       
       # 坐标轴
       "axes.linewidth": 0.8,
       "axes.edgecolor": "black",
       "axes.facecolor": "white",
       "axes.titlesize": 12,
       "axes.titleweight": "bold",
       "axes.titlepad": 15,
       "axes.labelsize": 11,
       "axes.labelweight": "bold",
       "axes.labelpad": 4,
       "axes.spines.top": False,
       "axes.spines.right": False,
       "axes.axisbelow": True,
       
       # 刻度
       "xtick.labelsize": 9,
       "ytick.labelsize": 9,
       "xtick.major.size": 4,
       "ytick.major.size": 4,
       "xtick.minor.size": 2,
       "ytick.minor.size": 2,
       "xtick.major.width": 0.8,
       "ytick.major.width": 0.8,
       "xtick.color": "black",
       "ytick.color": "black",
       
       # 图例
       "legend.fontsize": 9,
       "legend.frameon": True,
       "legend.framealpha": 1.0,
       "legend.facecolor": "white",
       "legend.edgecolor": "black",
       "legend.fancybox": False,
       "legend.shadow": False,
       "legend.numpoints": 1,
       "legend.handlelength": 1.5,
       "legend.handletextpad": 0.5,
       "legend.columnspacing": 1.0,
       
       # 网格
       "grid.color": "#b3b3b3",
       "grid.linestyle": "-",
       "grid.linewidth": 0.4,
       "grid.alpha": 0.8,
       
       # 线条
       "lines.linewidth": 1.5,
       "lines.markersize": 6,
       "lines.markeredgewidth": 0.5,
       
       # 图形
       "figure.facecolor": "white",
       "figure.edgecolor": "white",
       "figure.frameon": False,
       
       # 其他
       "patch.linewidth": 0.5,
       "patch.facecolor": "#1f77b4",
       "patch.edgecolor": "black",
       "patch.force_edgecolor": True,
   })

# ========== DATA LOADING ==========
def load_scca_data():
   """加载SCCA多视角数据"""
   try:
       # 读取SCCA视角对比矩阵
       scca_file = SCC_DIR / "scca_perspective_comparison_matrix.csv"
       if not scca_file.exists():
           raise FileNotFoundError(f"SCCA数据文件不存在: {scca_file}")
       
       scca_matrix = pd.read_csv(scca_file, index_col=0)
       print(f"成功加载SCCA数据: {scca_file}")
       print(f"数据维度: {scca_matrix.shape}")
       print(f"准则: {list(scca_matrix.index)}")
       print(f"视角: {list(scca_matrix.columns)}")
       
       return scca_matrix
       
   except Exception as e:
       raise RuntimeError(f"加载SCCA数据失败: {e}")

# ========== PANEL A: 多维散点图 ==========
def plot_multi_perspective_scatter(ax, scca_matrix):
   """Panel A: 准则关键性多视角散点图 (以comprehensive为参考)"""
   
   # 选择主要准则进行可视化
   main_criteria = ['Cost', 'LOLE', 'Environment', 'CO2']
   available_criteria = [c for c in main_criteria if c in scca_matrix.index]
   
   if not available_criteria:
       raise ValueError("没有找到主要准则数据")
   
   plot_data = scca_matrix.loc[available_criteria].copy()
   
   # 确保comprehensive视角存在
   if 'comprehensive' not in plot_data.columns:
       raise ValueError("没有找到comprehensive视角数据")
   
   # 使用comprehensive作为参考（x轴）
   comprehensive_values = plot_data['comprehensive']
   
   perspectives = [p for p in list(scca_matrix.columns) if p != 'comprehensive']
   
   # 使用不同的标记样式对应不同视角
   perspective_markers = {
       'comprehensive': 'o',
       'government': 's', 
       'news': '^',
       'research': 'D',
       'social_media': 'v'
   }
   
   # 为每个准则创建图例
   criterion_legend_elements = []
   
   # 按准则分组绘制，每个准则用相同颜色但不同形状
   for j, criterion in enumerate(available_criteria):
       criterion_color = CRITERIA_COLORS[criterion]
       x_comprehensive = comprehensive_values[criterion]
       
       # 为该准则添加图例元素（只需要一次）
       criterion_legend_elements.append(
           Line2D([0], [0], marker='o', color='w', markerfacecolor=criterion_color,
                 markersize=8, markeredgecolor='black', linewidth=0,
                 label=criterion if criterion != 'CO2' else 'CO2')
       )
       
       for i, perspective in enumerate(perspectives):
           if perspective in plot_data.columns:
               y_val = plot_data.loc[criterion, perspective]
               
               scatter = ax.scatter(
                   x_comprehensive, y_val,
                   c=criterion_color,
                   marker=perspective_markers.get(perspective, 'o'),
                   s=80,
                   alpha=0.8,
                   edgecolors='black',
                   linewidths=1,
                   zorder=5
               )
   
   # 创建视角图例元素（空心形状）
   perspective_legend_elements = []
   for perspective in perspectives:
       if perspective in perspective_markers:
           perspective_legend_elements.append(
               Line2D([0], [0], marker=perspective_markers[perspective], color='w', 
                     markerfacecolor='white', markersize=8, markeredgecolor='black', 
                     linewidth=1.5, label=perspective.replace('_', ' ').title())
           )
   
   # 添加对角线参考线 (完全一致线)
   min_val = min(comprehensive_values.min(), 
                 plot_data[perspectives].min().min())
   max_val = max(comprehensive_values.max(), 
                 plot_data[perspectives].max().max())
   
   ax.plot([min_val, max_val], [min_val, max_val], 
           'k--', alpha=0.5, linewidth=1, zorder=1,
           label='Perfect Consensus')
   
   # 添加置信区间带
   x_range = np.linspace(min_val, max_val, 100)
   for confidence in [0.1, 0.2]:
       ax.fill_between(x_range, x_range*(1-confidence), x_range*(1+confidence),
                      alpha=0.1, color='gray', zorder=1)
   
   # 设置坐标轴
   ax.set_xlabel('Comprehensive Perspective Criticality Index', fontweight='bold', fontsize=11)
   ax.set_ylabel('Other Perspective Criticality Index', fontweight='bold', fontsize=11)
   ax.set_title('Multi-Perspective Criteria Criticality Analysis', 
               fontweight='bold', fontsize=12)
   
   # 网格和样式
   ax.grid(True, alpha=0.3, zorder=0)
   ax.set_axisbelow(True)
   
   # 创建双图例：准则颜色 + 视角形状
   # 准则图例
   criterion_legend = ax.legend(handles=criterion_legend_elements, 
                               title="Criteria", loc='upper left',
                               frameon=True, fancybox=False, 
                               edgecolor='black', framealpha=1.0)
   criterion_legend.get_title().set_fontweight('bold')
   
   # 视角图例
   perspective_legend = ax.legend(handles=perspective_legend_elements,
                                 title="Perspectives", loc='lower right',
                                 frameon=True, fancybox=False,
                                 edgecolor='black', framealpha=1.0)
   perspective_legend.get_title().set_fontweight('bold')
   
   # 添加回准则图例（matplotlib只显示最后一个图例）
   ax.add_artist(criterion_legend)
   
   # 设置相等的坐标轴比例
   ax.set_aspect('equal', adjustable='box')
   
   # 添加统计信息文本框 - 改为以comprehensive为参考的相关性
   corr_summary = []
   comprehensive_vals = comprehensive_values.values
   
   for perspective in perspectives:
       if perspective in plot_data.columns:
           # 计算该视角与comprehensive的相关性
           perspective_vals = plot_data[perspective].values
           
           # 确保有足够的数据点计算相关性
           if len(comprehensive_vals) > 1 and len(perspective_vals) > 1:
               r = np.corrcoef(comprehensive_vals, perspective_vals)[0, 1]
               corr_summary.append(f"{perspective.capitalize()}: r={r:.3f}")
           else:
               corr_summary.append(f"{perspective.capitalize()}: r=N/A")
   
   stats_text = "Consensus with Comprehensive:\n" + "\n".join(corr_summary[:3]) + "\n" + "\n".join(corr_summary[3:])
   ax.text(0.7, 0.99, stats_text, transform=ax.transAxes,
          verticalalignment='top', horizontalalignment='center', fontsize=8, fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.9))
   
   return ax

# ========== PANEL B: 综合一致性与分歧分析 ==========
def plot_consensus_divergence_analysis(ax, scca_matrix):
   """Panel B: 视角一致性与分歧度综合分析"""
   
   # 计算每个准则的统计指标
   analysis_data = []
   
   for criterion in scca_matrix.index:
       values = scca_matrix.loc[criterion].values
       
       mean_val = np.mean(values)
       std_val = np.std(values)
       cv = std_val / mean_val if mean_val > 0 else 0
       range_val = np.max(values) - np.min(values)
       
       # 计算该准则在各视角间的一致性（标准化方差）
       normalized_values = values / np.mean(values) if np.mean(values) > 0 else values
       consistency_score = 1 / (1 + np.var(normalized_values))  # 一致性分数，范围0-1
       avg_correlation = consistency_score
       
       # 分类
       if criterion in ['Cost', 'LOLE', 'CO2', 'Environment']:
           category = 'Primary'
           size = 100
       else:
           category = 'Environmental'
           size = 60
       
       analysis_data.append({
           'criterion': criterion,
           'category': category,
           'mean_criticality': mean_val,
           'coefficient_variation': cv,
           'range_criticality': range_val,
           'avg_correlation': avg_correlation,
           'size': size
       })
   
   df = pd.DataFrame(analysis_data)
   
   # 创建散点图：x轴为平均关键性，y轴为变异系数
   for category in df['category'].unique():
       subset = df[df['category'] == category]
       
       if category == 'Primary':
           colors = [CRITERIA_COLORS.get(c, '#666666') for c in subset['criterion']]
           marker = 'o'
           edgecolors = 'black'
           linewidths = 1.5
           alpha = 0.9
       else:
           colors = ['#ff7f0e'] * len(subset)  # 统一的环境准则颜色
           marker = 's'
           edgecolors = 'darkgray'
           linewidths = 1
           alpha = 0.7
       
       scatter = ax.scatter(
           subset['mean_criticality'], 
           subset['coefficient_variation'],
           c=colors,
           s=subset['size'],
           marker=marker,
           alpha=alpha,
           edgecolors=edgecolors,
           linewidths=linewidths,
           label=f'{category} Criteria',
           zorder=5
       )
   
   # 智能标签定位：所有标签都放在右边
   def get_label_positions(df):
       """计算不重叠的标签位置 - 全部放在右边"""
       positions = {}
       
       # 按Y坐标排序，从上到下分配标签位置
       df_sorted = df.sort_values('coefficient_variation', ascending=False)
       
       # 预定义右侧标签偏移策略 - 确保垂直分布
       y_positions = np.linspace(15, -15, len(df_sorted))  # 从上到下分布
       
       for i, (_, row) in enumerate(df_sorted.iterrows()):
           criterion = row['criterion']
           
           # 所有标签都放在右边，使用不同的水平和垂直偏移
           base_offset_x = 20  # 基础右偏移
           
           # 根据准则类型调整水平距离
           if criterion in ['Cost', 'LOLE', 'CO2', 'Environment']:
               offset_x = base_offset_x  # 主要准则离得近一些
           else:
               offset_x = base_offset_x + 30  # 环境准则离得远一些，避免与主要准则重叠
           
           offset_y = y_positions[i]
           
           positions[criterion] = (offset_x, offset_y)
           
           if 'land_occupation' in positions:
               current_x, current_y = positions['land_occupation']
               positions['land_occupation'] = (current_x, current_y + 8)  # 向下移动8个点
       
       return positions
   
   # 获取标签位置
   label_positions = get_label_positions(df)
   
   # 添加标签
   for _, row in df.iterrows():
       criterion = row['criterion']
       x_pos = row['mean_criticality']
       y_pos = row['coefficient_variation']
       
       # 标签简化
       label_map = {
           'LOLE': 'LOLE',
           'CO2': 'CO2',
           'Cost': 'Cost',
           'Environment': 'Environment',
           'freshwater_ecotoxicity': 'Fresh. Ecotox.',
           'freshwater_eutrophication': 'Fresh. Eutro.',
           'marine_eutrophication': 'Marine Eutro.',
           'mineral_resource_depletion': 'Mineral Depl.',
           'human_toxicity': 'Human Tox.',
           'ionising_radiation': 'Ionising Rad.',
           'land_occupation': 'Land Occup.'
       }
       
       label = label_map.get(criterion, criterion.replace('_', ' ').title())
       offset_x, offset_y = label_positions[criterion]
       
       # 所有标签都是左对齐（因为都在右边）
       ha = 'left'
       va = 'center'
       
       ax.annotate(
           label,
           (x_pos, y_pos),
           xytext=(offset_x, offset_y), 
           textcoords='offset points',
           fontsize=7, 
           fontweight='bold',
           ha=ha,
           va=va,
           bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='white', 
                    edgecolor='gray', 
                    alpha=0.9,
                    linewidth=0.5),
           arrowprops=dict(arrowstyle='->', 
                         connectionstyle='arc3,rad=0.15',
                         color='gray',
                         alpha=0.6,
                         linewidth=1),
           zorder=6
       )
   
   # 添加区域划分线
   mean_cv = df['coefficient_variation'].mean()
   mean_crit = df['mean_criticality'].mean()
   
   # 垂直线（高/低关键性分界）
   ax.axvline(mean_crit, color='gray', linestyle=':', alpha=0.7, linewidth=1)
   # 水平线（高/低分歧分界）
   ax.axhline(mean_cv, color='gray', linestyle=':', alpha=0.7, linewidth=1)
   
   # 添加象限标签 - 调整位置避免与数据标签重叠，放在图内
   ax.text(0.85, 0.92, 'High Criticality\nHigh Divergence', 
          transform=ax.transAxes, ha='center', va='top',
          fontsize=8, fontweight='bold', style='italic',
          bbox=dict(boxstyle='round,pad=0.25', facecolor='lightcoral', alpha=0.7))
   
   ax.text(0.15, 0.92, 'Low Criticality\nHigh Divergence', 
          transform=ax.transAxes, ha='center', va='top',
          fontsize=8, fontweight='bold', style='italic',
          bbox=dict(boxstyle='round,pad=0.25', facecolor='lightyellow', alpha=0.7))
   
   ax.text(0.85, 0.05, 'High Criticality\nLow Divergence', 
          transform=ax.transAxes, ha='center', va='bottom',
          fontsize=8, fontweight='bold', style='italic',
          bbox=dict(boxstyle='round,pad=0.25', facecolor='lightgreen', alpha=0.7))
   
   ax.text(0.15, 0.05, 'Low Criticality\nLow Divergence', 
          transform=ax.transAxes, ha='center', va='bottom',
          fontsize=8, fontweight='bold', style='italic',
          bbox=dict(boxstyle='round,pad=0.25', facecolor='lightblue', alpha=0.7))
   
   # 坐标轴设置
   ax.set_xlabel('Mean Structural Criticality Index', fontweight='bold', fontsize=11)
   ax.set_ylabel('Coefficient of Variation (Divergence)', fontweight='bold', fontsize=11)
   ax.set_title('Consensus-Divergence Analysis Matrix', fontweight='bold', fontsize=12)
   
   # 网格和图例
   ax.grid(True, alpha=0.3, zorder=0)
   
   # 创建自定义图例，Primary Criteria分两行显示
   # Primary Criteria - 分成两行
   primary_handles = []
   primary_labels = []
   for criterion in ['Cost', 'LOLE', 'Environment', 'CO2']:
       if criterion in df['criterion'].values:
           color = CRITERIA_COLORS[criterion]
           label = criterion if criterion != 'CO2' else 'CO2'
           primary_handles.append(
               Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                     markersize=7, markeredgecolor='black', linewidth=1)
           )
           primary_labels.append(label)
   
   primary_legend = ax.legend(
       primary_handles, primary_labels,
       title='Primary\nCriteria',
       loc='center', bbox_to_anchor=(1.25, 0.75),
       frameon=True, fancybox=False, edgecolor='black',
       ncol=1, fontsize=8
   )
   primary_legend.get_title().set_fontweight('bold')
   primary_legend.get_title().set_fontsize(9)
   
   # Environmental Criteria - 显示所有环境准则
   env_criteria = [c for c in df['criterion'].values if c not in ['Cost', 'LOLE', 'Environment', 'CO2']]
   if env_criteria:
       env_handles = []
       env_labels = []
       # 显示所有环境准则
       for criterion in env_criteria:
           color = CRITERIA_COLORS.get(criterion, '#ff7f0e')
           label = criterion.replace('_', ' ').title()
           if criterion == 'freshwater_ecotoxicity':
               label = 'Fresh. Ecotox.'
           elif criterion == 'freshwater_eutrophication':
               label = 'Fresh. Eutro.'
           elif criterion == 'marine_eutrophication':
               label = 'Marine Eutro.'
           elif criterion == 'mineral_resource_depletion':
               label = 'Mineral Depl.'
           elif criterion == 'human_toxicity':
               label = 'Human Tox.'
           elif criterion == 'ionising_radiation':
               label = 'Ionising Rad.'
           elif criterion == 'land_occupation':
               label = 'Land Occup.'
           elif criterion == 'CO2':
               label = 'CO2'
           else:
               label = criterion.replace('_', ' ').title()
               
           env_handles.append(
               Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
                     markersize=6, markeredgecolor='black', linewidth=1)
           )
           env_labels.append(label)
       
       env_legend = ax.legend(
           env_handles, env_labels,
           title='Environmental\nCriteria',
           loc='center', bbox_to_anchor=(1.25, 0.35),
           frameon=True, fancybox=False, edgecolor='black',
           ncol=1, fontsize=7
       )
       env_legend.get_title().set_fontweight('bold')
       env_legend.get_title().set_fontsize(8)
       
       # 添加回第一个图例
       ax.add_artist(primary_legend)
   
   # 设置坐标轴范围
   ax.set_xlim(df['mean_criticality'].min() * 0.9, df['mean_criticality'].max() * 1.1)
   ax.set_ylim(-0.02, df['coefficient_variation'].max() * 1.1)

# ========== 主可视化函数 ==========
def create_scca_nature_visualization():
   """创建Nature风格的SCCA多视角分析双面板可视化"""
   
   print("加载SCCA多视角数据...")
   scca_matrix = load_scca_data()
   
   # 设置Nature期刊风格
   set_nature_style()
   
   # 创建双面板布局 (1x2) - 左边更长，右边更短
   fig = plt.figure(figsize=(14, 6))
   
   # 为面板标签留出空间 - 减少面板间距
   gs = GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.01, 
                 top=0.85, bottom=0.1, left=0.08, right=0.7)
   # Panel A: 多视角散点图
   ax1 = fig.add_subplot(gs[0, 0])
   plot_multi_perspective_scatter(ax1, scca_matrix)
   
   # Panel B: 一致性与分歧分析
   ax2 = fig.add_subplot(gs[0, 1])
   plot_consensus_divergence_analysis(ax2, scca_matrix)
   
   # 添加面板标签在图片正上方 (Nature期刊风格) - 根据新间距调整位置
   fig.text(0.28, 0.93, 'a', fontsize=18, fontweight='bold', 
            transform=fig.transFigure, ha='center')
   fig.text(0.59, 0.93, 'b', fontsize=18, fontweight='bold', 
            transform=fig.transFigure, ha='center')
   
   # 保存图形
   save_path = FIG_DIR / "fig_scca_sci.png"
   plt.savefig(save_path, dpi=600, bbox_inches='tight', 
              facecolor='white', edgecolor='none')
   plt.close()
   
   print(f"\nNature风格SCCA双面板分析已保存到: {save_path}")
   print("\n图形组件:")
   print("  • Panel A: 多视角准则关键性散点图 (以comprehensive为参考)")
   print("  • Panel B: 一致性-分歧度矩阵分析")
   
   # 输出关键发现
   print(f"\n=== 关键发现 ===")
   mean_criticality = scca_matrix.mean(axis=1).sort_values(ascending=False)
   print(f"最关键准则: {mean_criticality.index[0]} (SCI = {mean_criticality.iloc[0]:.4f})")
   
   cv_data = scca_matrix.std(axis=1) / scca_matrix.mean(axis=1)
   most_divergent = cv_data.sort_values(ascending=False).index[0]
   print(f"分歧最大准则: {most_divergent} (CV = {cv_data[most_divergent]:.4f})")
   
   # 计算与comprehensive的相关性
   if 'comprehensive' in scca_matrix.columns:
       comprehensive_values = scca_matrix['comprehensive']
       other_perspectives = [p for p in scca_matrix.columns if p != 'comprehensive']
       
       correlations = []
       for perspective in other_perspectives:
           if perspective in scca_matrix.columns:
               r = np.corrcoef(comprehensive_values.values, scca_matrix[perspective].values)[0, 1]
               correlations.append(r)
               print(f"与comprehensive一致性 ({perspective}): {r:.3f}")
       
       if correlations:
           avg_correlation = np.mean(correlations)
           print(f"平均与comprehensive一致性: {avg_correlation:.3f}")

if __name__ == "__main__":
   print("=== Nature期刊风格SCCA多视角分析可视化 (以comprehensive为参考) ===")
   create_scca_nature_visualization()
