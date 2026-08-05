# -*- coding: utf-8 -*-
"""
ARSI备选方案等级位移指数直观条形图分析 - Nature期刊风格
Panel A: 情景排名变化条形图 (各准则对每个情景的具体影响)
Panel B: 准则影响力对比条形图 (各准则的总体影响强度)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from pathlib import Path
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# ========== PATHS ==========
BASE_DIR = Path(r"C:/Users/vanessa/OneDrive - University College London/research/data/paper3/MCDM_FINAL")
SCC_DIR = BASE_DIR / "scc_out"
FIG_DIR = BASE_DIR / "eda_out" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========== COLORS ==========
# Nature期刊风格配色方案
ALTERNATIVE_COLORS = {
    'Netzero': '#2E86AB',      # 深蓝色
    'ndc-ltp': '#2E86AB',      # 深蓝色 (备用名称)
    'NDC-LTP': '#2E86AB',      # 深蓝色 (备用名称)
    'CurrentPolicy': '#A23B72', # 紫红色
    'BAU': '#F18F01',          # 橙色
    'Delayed': '#C73E1D',      # 红色
    'TechOptimistic': '#1A9850', # 绿色
    'TechPessimistic': '#8C564B', # 棕色
    'VRE-70%': '#9467BD',      # 紫色
    'VRE-75%': '#17BECF',      # 青色
    'VRE-80%': '#BCBD22',      # 橄榄色
    'VRE-85%': '#FF7F0E'       # 橙色
}

# Core category colors
CATEGORY_COLORS = {
    "Cost": "#2E86AB",           # Economic
    "LOLD": "#A23B72",           # Technical (changed from LOLE)
    "LOLE": "#A23B72",           # Technical (backup)
    "Environment": "#F18F01",    # Local Environmental  
    "CO2": "#C73E1D"             # Global Environmental
}

# Environmental sub-criteria colors (按照类别分组)
ENV_COLORS = {
    "Land Occupation": "#F18F01",                    # Local Environmental
    "Freshwater Ecotoxicity": "#F18F01",           # Local Environmental
    "Human Toxicity": "#F18F01",                    # Local Environmental
    "Ionising Radiation": "#F18F01",               # Local Environmental
    "Freshwater Eutrophication": "#F18F01",        # Local Environmental
    "Marine Eutrophication": "#F18F01",            # Local Environmental
    "Mineral Resource Depletion": "#F18F01",       # Local Environmental
    # Original names (backup)
    "land_occupation": "#F18F01",
    "freshwater_ecotoxicity": "#F18F01",
    "human_toxicity": "#F18F01",
    "ionising_radiation": "#F18F01",
    "freshwater_eutrophication": "#F18F01",
    "marine_eutrophication": "#F18F01",
    "mineral_resource_depletion": "#F18F01"
}

# Combined criteria colors
CRITERIA_COLORS = {**CATEGORY_COLORS, **ENV_COLORS}

# ========== STYLE SETUP ==========
def set_nature_style():
    """Nature期刊风格设置"""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.style.use('default')
    
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 14,  # Increased from 11
        "font.weight": "normal",
        "axes.linewidth": 1.5,
        "axes.titlesize": 18,  # Increased from 14
        "axes.titleweight": "bold",
        "axes.labelsize": 20,  # Increased from 12
        "axes.labelweight": "bold",
        "xtick.labelsize": 16.5,  # Increased from 10
        "ytick.labelsize": 17,  # Increased from 10
        "legend.fontsize": 16,  # Increased from 12
        "legend.title_fontsize": 18,  # Increased from 13
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "black",
        "legend.fancybox": True,
        "grid.alpha": 0.4,
        "grid.linewidth": 0.8,
    })

# ========== DATA LOADING ==========
def load_arsi_data():
    """加载ARSI分析数据"""
    try:
        # 加载综合ARSI数据
        arsi_all = pd.read_csv(SCC_DIR / "arsi_all_perspectives_detailed.csv")
        
        # 加载共识分析数据
        consensus_data = pd.read_csv(SCC_DIR / "arsi_consensus_analysis.csv")
        
        # 加载热力图数据
        heatmap_data = pd.read_csv(SCC_DIR / "arsi_heatmap_data.csv", index_col=0)
        
        # 统一名称处理
        name_mapping = {'ndc-ltp': 'Netzero', 'NDC-LTP': 'Netzero'}
        
        for df in [arsi_all, consensus_data]:
            if 'alternative' in df.columns:
                df['alternative'] = df['alternative'].replace(name_mapping)
        
        heatmap_data = heatmap_data.rename(index=name_mapping)
        
        # 将 LOLE 改为 LOLD
        lole_mapping = {'LOLE': 'LOLD'}
        for df in [arsi_all, consensus_data]:
            if 'criterion' in df.columns:
                df['criterion'] = df['criterion'].replace(lole_mapping)
        
        heatmap_data = heatmap_data.rename(columns=lole_mapping)
        
        # 规范化环境准则名称
        env_name_mapping = {
            'land_occupation': 'Land Occupation',
            'freshwater_ecotoxicity': 'Freshwater Ecotoxicity',
            'human_toxicity': 'Human Toxicity',
            'ionising_radiation': 'Ionising Radiation',
            'freshwater_eutrophication': 'Freshwater Eutrophication',
            'marine_eutrophication': 'Marine Eutrophication',
            'mineral_resource_depletion': 'Mineral Resource Depletion'
        }
        
        for df in [arsi_all, consensus_data]:
            if 'criterion' in df.columns:
                df['criterion'] = df['criterion'].replace(env_name_mapping)
        
        heatmap_data = heatmap_data.rename(columns=env_name_mapping)
        
        print("数据加载成功!")
        print(f"方案数量: {len(arsi_all['alternative'].unique())}")
        print(f"准则数量: {len(consensus_data['criterion'].unique())}")
        
        return arsi_all, consensus_data, heatmap_data
        
    except Exception as e:
        raise FileNotFoundError(f"无法加载ARSI数据: {SCC_DIR}: {e}")

# ========== PANEL A: 情景排名变化条形图 ==========
def plot_scenario_ranking_changes(ax, consensus_data):
    """Panel A: 情景排名变化条形图 - 显示各准则对每个情景的具体影响"""
    
    # 准备数据矩阵
    pivot_data = consensus_data.pivot(index='alternative', columns='criterion', values='mean_arsi')
    
    # 只保留有足够数据的情景和准则
    scenario_completeness = pivot_data.notna().sum(axis=1) / len(pivot_data.columns)
    criteria_completeness = pivot_data.notna().sum(axis=0) / len(pivot_data.index)
    
    valid_scenarios = scenario_completeness[scenario_completeness > 0.5].index.tolist()
    valid_criteria = criteria_completeness[criteria_completeness > 0.5].index.tolist()
    
    # 选择主要准则（优先选择关键准则）
    preferred_criteria = ['Cost', 'LOLD', 'Environment', 'CO2', 'Land Occupation', 
                         'Human Toxicity', 'Freshwater Ecotoxicity']
    main_criteria = [c for c in preferred_criteria if c in valid_criteria]
    
    # 如果主要准则不够，添加其他准则
    if len(main_criteria) < 6:
        other_criteria = [c for c in valid_criteria if c not in main_criteria]
        main_criteria.extend(other_criteria[:6-len(main_criteria)])
    
    # 筛选数据
    plot_data = pivot_data.loc[valid_scenarios, main_criteria[:6]]  # 限制为6个准则以保持可读性
    
    # 创建分组条形图
    n_scenarios = len(valid_scenarios)
    n_criteria = len(main_criteria[:6])
    
    # 设置条形位置
    x_positions = np.arange(n_scenarios)
    bar_width = 0.12
    
    # 为每个准则绘制条形
    for i, criterion in enumerate(main_criteria[:6]):
        criterion_data = plot_data[criterion].values
        x_pos = x_positions + (i - n_criteria/2 + 0.5) * bar_width
        
        # 获取颜色
        color = CRITERIA_COLORS.get(criterion, '#666666')
        
        # 绘制条形，根据正负值设置不同的透明度和样式
        bars = ax.bar(x_pos, criterion_data, bar_width, 
                     color=color, alpha=0.8, 
                     edgecolor='black', linewidth=0.8,
                     label=criterion)
        
        # 为显著影响添加数值标签
        for j, (bar, value) in enumerate(zip(bars, criterion_data)):
            if abs(value) > 0.05:  # 只标注显著影响
                height = bar.get_height()
                label_y = height + 0.02 if height >= 0 else height - 0.02
                ax.text(bar.get_x() + bar.get_width()/2, label_y,
                       f'{value:.2f}', ha='center', 
                       va='bottom' if height >= 0 else 'top',
                       fontsize=15, fontweight='bold',  # Increased from 8
                       bbox=dict(boxstyle='round,pad=0.2', 
                               facecolor='white', alpha=0.8,
                               edgecolor=color, linewidth=0.5))
    
    # 添加零基线
    ax.axhline(0, color='black', linewidth=1.5, alpha=0.8)
    
    # 设置坐标轴
    ax.set_xticks(x_positions)
    ax.set_xticklabels(valid_scenarios, rotation=0, ha='center', fontweight='bold')
    ax.set_ylabel('ARSI Value (Ranking Change)\n← Worse Ranking | Better Ranking →', 
                 fontweight='bold', fontsize=16)  # Increased from 11
    ax.set_title('Scenario Ranking Changes by Criteria\n(How Each Criterion Affects Each Scenario)', 
                fontweight='bold', fontsize=18)  # Increased from 14
    
    # 添加网格
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 添加图例
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
             fontsize=16, ncol=2, framealpha=0.9,  # Increased from 12
             prop={'weight': 'bold'})
    
    # 添加影响区域标注
    ylim = ax.get_ylim()
    ax.fill_between([ax.get_xlim()[0], ax.get_xlim()[1]], 0, ylim[1], 
                   alpha=0.05, color='green', label='Better Ranking')
    ax.fill_between([ax.get_xlim()[0], ax.get_xlim()[1]], ylim[0], 0, 
                   alpha=0.05, color='red', label='Worse Ranking')
    
    return plot_data, valid_scenarios, main_criteria[:6]

# ========== PANEL B: 准则影响力对比条形图 ==========
def plot_criteria_impact_comparison(ax, consensus_data, main_criteria):
    """Panel B: 准则影响力对比条形图 - 显示各准则的总体影响强度"""
    
    # 计算每个准则的影响力指标
    criteria_stats = []
    
    for criterion in main_criteria:
        crit_data = consensus_data[consensus_data['criterion'] == criterion]['mean_arsi']
        
        if len(crit_data) == 0:
            continue
            
        # 计算统计指标
        max_positive = crit_data.max()
        max_negative = crit_data.min()
        mean_abs_impact = crit_data.abs().mean()
        impact_range = max_positive - max_negative
        std_impact = crit_data.std()
        
        # 正面和负面影响的平均值
        positive_impacts = crit_data[crit_data > 0]
        negative_impacts = crit_data[crit_data < 0]
        
        avg_positive = positive_impacts.mean() if len(positive_impacts) > 0 else 0
        avg_negative = negative_impacts.mean() if len(negative_impacts) > 0 else 0
        
        criteria_stats.append({
            'criterion': criterion,
            'max_positive': max_positive,
            'max_negative': max_negative,
            'avg_positive': avg_positive,
            'avg_negative': avg_negative,
            'mean_abs_impact': mean_abs_impact,
            'impact_range': impact_range,
            'std_impact': std_impact
        })
    
    stats_df = pd.DataFrame(criteria_stats)
    
    if len(stats_df) == 0:
        ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=18)
        return
    
    # 按影响强度排序
    stats_df = stats_df.sort_values('mean_abs_impact', ascending=True)
    
    # 创建水平堆叠条形图
    y_pos = np.arange(len(stats_df))
    
    # 绘制正面影响（右侧）
    positive_bars = ax.barh(y_pos, stats_df['avg_positive'], 
                           color='#1a9850', alpha=0.8, height=0.6,
                           label='Average Positive Impact')
    
    # 绘制负面影响（左侧）
    negative_bars = ax.barh(y_pos, stats_df['avg_negative'], 
                           color='#d73027', alpha=0.8, height=0.6,
                           label='Average Negative Impact')
    
    # 添加最大值标记
    for i, (_, row) in enumerate(stats_df.iterrows()):
        # 最大正面影响标记
        if row['max_positive'] > 0:
            ax.plot(row['max_positive'], i, marker='|', color='darkgreen', 
                   markersize=12, markeredgewidth=3)
            if row['max_positive'] > 0.1:
                ax.text(row['max_positive'] + 0.05, i, f'{row["max_positive"]:.2f}',
                       va='center', ha='left', fontsize=18, fontweight='bold',
                       color='darkgreen')
        
        # 最大负面影响标记
        if row['max_negative'] < 0:
            ax.plot(row['max_negative'], i, marker='|', color='darkred', 
                   markersize=12, markeredgewidth=3)
            if row['max_negative'] < -0.1:
                # 调整负值标签位置以避免重叠
                ax.text(row['max_negative'] - 0.05, i, f'{row["max_negative"]:.2f}',
                       va='center', ha='right', fontsize=14, fontweight='bold',
                       color='darkred')
        
        # 添加准则标签背景色
        criterion = row['criterion']
        color = CRITERIA_COLORS.get(criterion, '#666666')
        ax.add_patch(Rectangle((-0.02, i-0.4), 0.01, 0.8, 
                              facecolor=color, alpha=0.7, 
                              transform=ax.get_yaxis_transform()))
    
    # 添加零基线
    ax.axvline(0, color='black', linewidth=2, alpha=0.8)
    
    # 设置坐标轴
    ax.set_yticks(y_pos)
    y_labels = []
    for criterion in stats_df['criterion']:
        color = CRITERIA_COLORS.get(criterion, '#666666')
        y_labels.append(criterion)
    
    ax.set_yticklabels(y_labels, fontweight='bold')
    
    # 为每个Y轴标签设置对应颜色
    for i, (tick, criterion) in enumerate(zip(ax.get_yticklabels(), stats_df['criterion'])):
        color = CRITERIA_COLORS.get(criterion, '#666666')
        tick.set_color(color)
        tick.set_fontweight('bold')
        tick.set_fontsize(16)
    
    ax.set_xlabel('ARSI Impact Value\n← Negative Impact | Positive Impact →', 
                 fontweight='bold', fontsize=16)
    # 增大并加粗x轴刻度标注
    ax.tick_params(axis='x', labelsize=16, width=2, labelcolor='black', labelrotation=0)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    
    ax.set_title('Criteria Impact Strength Comparison\n(Average and Maximum Effects)', 
                fontweight='bold', fontsize=18)
    
    # 添加网格
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 创建详细图例
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor='#1a9850', alpha=0.8, 
                          label='Average Positive Impact'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='#d73027', alpha=0.8, 
                          label='Average Negative Impact'),
        plt.Line2D([0], [0], marker='|', color='darkgreen', linewidth=0,
                   markersize=10, markeredgewidth=3, label='Maximum Positive'),
        plt.Line2D([0], [0], marker='|', color='darkred', linewidth=0,
                   markersize=10, markeredgewidth=3, label='Maximum Negative')
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', 
             bbox_to_anchor=(0.98, 0.02), fontsize=16,
             prop={'weight': 'bold'})
    
    # 调整图形边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return stats_df

# ========== MAIN VISUALIZATION FUNCTION ==========
def create_arsi_bar_chart_visualization():
    """创建ARSI条形图分析可视化"""
    
    print("加载ARSI分析数据...")
    arsi_all, consensus_data, heatmap_data = load_arsi_data()
    
    # 设置样式
    set_nature_style()
    
    # 增大图形宽度并调整Panel B比例
    fig = plt.figure(figsize=(25, 8))  # Increased width from 22 to 24
    gs = GridSpec(1, 2, width_ratios=[1.3, 1.3], wspace=0.15)  # Adjusted ratio for more balanced space
    
    print("生成Panel A: 情景排名变化条形图...")
    ax1 = fig.add_subplot(gs[0, 0])
    plot_data, valid_scenarios, main_criteria = plot_scenario_ranking_changes(ax1, consensus_data)
    
    print("生成Panel B: 准则影响力对比条形图...")
    ax2 = fig.add_subplot(gs[0, 1])
    stats_df = plot_criteria_impact_comparison(ax2, consensus_data, main_criteria)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    save_path = FIG_DIR / "figI_arsi_bar_chart_analysis_updated.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nARSI条形图分析可视化已保存到: {save_path}")
    print("图形组件:")
    print("  • Panel A: 情景排名变化条形图 - 直接显示各准则对每个情景的具体影响")
    print("  • Panel B: 准则影响力对比条形图 - 展示各准则的平均和最大影响")
    
    # 打印关键发现
    print(f"\n=== 关键发现摘要 ===")
    
    if len(consensus_data) > 0:
        max_impact_row = consensus_data.loc[consensus_data['mean_arsi'].abs().idxmax()]
        print(f"最强影响: {max_impact_row['criterion']} 对 {max_impact_row['alternative']} "
              f"(ARSI = {max_impact_row['mean_arsi']:.4f})")
        
        scenario_impacts = consensus_data.groupby('alternative')['mean_arsi'].agg(['mean', 'std', 'min', 'max'])
        most_volatile = scenario_impacts['std'].idxmax()
        print(f"受影响最大的情景: {most_volatile} (标准差 = {scenario_impacts.loc[most_volatile, 'std']:.4f})")
        
        best_performer = scenario_impacts['mean'].idxmax()
        print(f"总体表现最好的情景: {best_performer} (平均ARSI = {scenario_impacts.loc[best_performer, 'mean']:.4f})")

if __name__ == "__main__":
    print("=== ARSI条形图影响分析可视化 - Nature期刊风格 ===")
    create_arsi_bar_chart_visualization()