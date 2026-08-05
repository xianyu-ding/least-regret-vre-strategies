# -*- coding: utf-8 -*-
"""
SCC敲除分析可视化 - Nature期刊风格
Panel A: 指标重要性排名 (翻转概率)
Panel B: 胜率变化冲击图 (替代热力图的精美方案)
Panel C: 敲除影响网络图
Panel D: 稳健性分析散点图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from pathlib import Path
import networkx as nx
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches

# ========== PATHS ==========
BASE_DIR = Path(r"C:/Users/vanessa/OneDrive - University College London/research/data/paper3/MCDM_FINAL")
SCC_DIR = BASE_DIR / "scc_out"
FIG_DIR = BASE_DIR / "eda_out" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========== COLORS ==========
CATEGORY_COLORS = {
    "Cost": "#2E86AB",           # Economic
    "LOLE": "#A23B72",           # Technical
    "Environment": "#F18F01",    # Local Environmental  
    "CO2": "#C73E1D"             # Global Environmental (mapped from Global (CO2))
}

# Environmental sub-criteria colors (按照类别分组)
ENV_COLORS = {
    "land_occupation": "#F18F01",           # Local Environmental
    "freshwater_ecotoxicity": "#F18F01",   # Local Environmental
    "human_toxicity": "#F18F01",           # Local Environmental
    "ionising_radiation": "#F18F01",       # Local Environmental
    "freshwater_eutrophication": "#F18F01", # Local Environmental
    "marine_eutrophication": "#F18F01",    # Local Environmental
    "mineral_resource_depletion": "#F18F01" # Local Environmental
}

IMPACT_COLORS = {
    "positive": "#1a9850",    # 绿色表示正面影响
    "negative": "#d73027",    # 红色表示负面影响
    "neutral": "#fdae61"      # 黄色表示中性影响
}

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
        "font.size": 11,
        "font.weight": "bold",
        "axes.linewidth": 1.5,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",
        "grid.alpha": 0.3,
    })

# ========== DATA LOADING ==========
def load_knockout_data():
    """加载敲除分析数据"""
    try:
        # 加载翻转概率数据
        top_flip = pd.read_csv(SCC_DIR / "scc_knockout_top_flip.csv")
        env_flip = pd.read_csv(SCC_DIR / "scc_knockout_env_flip.csv")
        
        # 加载胜率变化数据
        top_winrate_delta = pd.read_csv(SCC_DIR / "scc_knockout_top_winrate_delta.csv")
        env_winrate_delta = pd.read_csv(SCC_DIR / "scc_knockout_env_winrate_delta.csv")
        
        # 加载胜率数据用于稳健性分析
        winrates = pd.read_csv(SCC_DIR / "scc_winrates.csv")
        
        # 替换标签
        for df in [top_winrate_delta, env_winrate_delta, winrates]:
            if 'alternative' in df.columns:
                df['alternative'] = df['alternative'].str.replace('ndc-ltp', 'Netzero', case=False)
        
        return top_flip, env_flip, top_winrate_delta, env_winrate_delta, winrates
        
    except Exception as e:
        raise FileNotFoundError(f"无法加载敲除分析数据: {SCC_DIR}: {e}")

# ========== PANEL A: 指标重要性排名 ==========
def plot_indicator_importance(ax, top_flip, env_flip):
    """Panel A: 指标重要性排名（翻转概率条形图）"""
    
    # 合并顶层和环保指标数据
    top_data = top_flip.copy()
    top_data['category'] = 'Top-level'
    top_data['indicator'] = top_data['group']
    
    env_data = env_flip.copy()
    env_data['category'] = 'Environmental'
    env_data['group'] = env_data['indicator']
    
    # 合并并排序
    all_data = pd.concat([
        top_data[['indicator', 'flip_prob', 'category']],
        env_data[['indicator', 'flip_prob', 'category']]
    ]).sort_values('flip_prob', ascending=True)
    
    # 准备绘图数据
    y_pos = np.arange(len(all_data))
    flip_probs = all_data['flip_prob'].values
    indicators = all_data['indicator'].values
    categories = all_data['category'].values
    
    # 为每个指标分配颜色
    colors = []
    for i, (indicator, category) in enumerate(zip(indicators, categories)):
        if category == 'Top-level':
            colors.append(CATEGORY_COLORS.get(indicator, '#666666'))
        else:
            colors.append(ENV_COLORS.get(indicator, '#F18F01'))
    
    # 创建水平条形图
    bars = ax.barh(y_pos, flip_probs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 添加数值标签
    for i, (bar, prob) in enumerate(zip(bars, flip_probs)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{prob:.2%}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    # 设置坐标轴
    ax.set_yticks(y_pos)
    
    # 创建带颜色的y轴标签
    y_labels = []
    label_colors = []
    for ind, cat in zip(indicators, categories):
        if ind == 'LOLE':
            display_name = 'LOLD'
        elif ind == 'CO2':
            display_name = 'CO2 Emission'  
        else:
            display_name = ind.replace('_', ' ').title()
        
        y_labels.append(display_name)
        
        # 根据类别设置标签颜色
        if cat == 'Top-level':
            if ind in CATEGORY_COLORS:
                label_colors.append(CATEGORY_COLORS[ind])
            else:
                label_colors.append('#666666')
        else:
            label_colors.append(ENV_COLORS.get(ind, '#F18F01'))
    
    ax.set_yticklabels(y_labels, fontweight='bold', fontsize=12)  # 确保加粗
    
    # 为每个y轴标签设置颜色和加粗
    for i, (tick_label, color) in enumerate(zip(ax.get_yticklabels(), label_colors)):
        tick_label.set_color(color)
        tick_label.set_fontweight('bold')  # 添加加粗
    ax.set_xlabel('Decision Flip Probability', fontweight='bold', fontsize=14)
    ax.set_title('A. Indicator Importance Ranking', fontweight='bold', fontsize=16)
    
    # 添加类别分隔线和标签
    top_count = sum(categories == 'Top-level')
    if top_count > 0:
        ax.axhline(y=top_count - 0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)
        
        # 添加每个指标的类别标识（右侧标签）- 增大字体和尺寸
        for i, (indicator, category) in enumerate(zip(indicators, categories)):
            if category == 'Top-level':
                category_label = 'Top'
                bg_color = 'lightblue'
            else:
                category_label = 'Env'
                bg_color = 'lightgreen'
            
            ax.text(max(flip_probs) * 1.18, i, category_label, 
                   ha='center', va='center', fontweight='bold', fontsize=11,  # 增大字体
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=bg_color, alpha=0.8, 
                            edgecolor='black', linewidth=1))  # 增大padding和边框
    
    ax.set_xlim(0, max(flip_probs) * 1.35)  # 为类别标签留出更多空间
    ax.grid(True, axis='x', alpha=0.3)

# ========== PANEL B: 多情景表现对比图 ==========
def plot_scenario_performance_comparison(ax, top_winrate_delta, env_winrate_delta, winrates):
    """Panel B: 多情景表现对比 - 原始权重 vs 蒙特卡洛 vs 敲除"""
    
    # 获取top 6方案
    top_alternatives = winrates.nlargest(6, 'win_rate')['alternative'].tolist()
    
    # 加载基准排名数据
    try:
        baseline = pd.read_csv(SCC_DIR / "baseline_ranking.csv")
        baseline['alternative'] = baseline['alternative'].str.replace('ndc-ltp', 'Netzero', case=False)
    except:
        print("警告: 无法加载基准排名数据，使用胜率作为替代")
        baseline = winrates.copy()
        baseline['rank'] = baseline['win_rate'].rank(method='min', ascending=False)
    
    # 准备数据
    scenario_data = []
    
    for alt in top_alternatives:
        # 1. 原始权重下的排名 (基准排名)
        baseline_rank = baseline[baseline['alternative'] == alt]['rank'].iloc[0]
        baseline_performance = 1 / baseline_rank  # 转换为性能指标 (排名越高，值越大)
        
        # 2. 蒙特卡洛权重下的胜率
        mc_winrate = winrates[winrates['alternative'] == alt]['win_rate'].iloc[0]
        
        # 3. 敲除分析下的稳健性 (计算平均胜率损失)
        all_deltas = pd.concat([
            top_winrate_delta[top_winrate_delta['alternative'] == alt].rename(columns={'group': 'indicator'}),
            env_winrate_delta[env_winrate_delta['alternative'] == alt]
        ])
        
        # 计算敲除后的平均胜率 (原胜率 - 平均胜率损失)
        avg_loss = all_deltas['delta_winrate'].mean()
        knockout_performance = max(0, mc_winrate - avg_loss)  # 敲除后的预期胜率
        
        scenario_data.append({
            'alternative': alt,
            'baseline_performance': baseline_performance,
            'mc_winrate': mc_winrate,
            'knockout_performance': knockout_performance,
            'vulnerability': avg_loss  # 脆弱性指标
        })
    
    df = pd.DataFrame(scenario_data)
    
    # 设置颜色方案 - 使用与Panel A/B一致的颜色
    method_colors = {
        'baseline': '#1f77b4',      # Blue
        'monte_carlo': '#ff7f0e',   # Orange  
        'knockout': '#d62728'       # Red
    }
    
    # 方案颜色 - 使用与收敛图一致的颜色
    alt_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b'   # Brown
    ]
    
    # 创建连线图：每个方案在三种情景下的表现
    x_positions = [1, 2, 3]  # 三个情景的x坐标
    x_labels = ['Fixed Weights\n(Baseline)', 'Monte Carlo\n(Uncertainty)', 'Knockout\n(Robustness)']
    
    # 绘制每个方案的表现轨迹
    for i, (_, row) in enumerate(df.iterrows()):
        alt = row['alternative']
        y_values = [row['baseline_performance'], row['mc_winrate'], row['knockout_performance']]
        
        # 绘制连线
        ax.plot(x_positions, y_values, 'o-', color=alt_colors[i], 
               linewidth=3, markersize=8, alpha=0.8, label=alt)
        
        # 添加脆弱性指示器 (在knockout点) - 只对显著脆弱性显示
        vulnerability = abs(row['vulnerability'])
        if vulnerability > 0.05:  # 只有脆弱性>5%才显示误差棒
            ax.errorbar(3, row['knockout_performance'], yerr=vulnerability*1.5, 
                       color=alt_colors[i], capsize=4, capthick=1.5, alpha=0.5,
                       linewidth=1.5)
        
        # 添加数值标签 - 特别处理VRE-75%避免遮挡，隐藏0%标签
        for j, (x, y) in enumerate(zip(x_positions, y_values)):
            if j == 0:  # baseline显示排名
                label_text = f"#{int(1/y)}"
            else:  # 其他显示百分比
                label_text = f"{y:.0%}"
            
            # 隐藏Monte Carlo和Knockout位置的0%标签
            if label_text == "0%" and (j == 1 or j == 2):
                continue  # 跳过0%标签的显示
            
            # 特别处理VRE-75% (i=0)，在前两个位置大幅下移
            if i == 0 and j <= 1:  # VRE-75%在Fixed Weights和Monte Carlo位置
                xytext_offset = (0, -60)  # 大幅下移
                va_align = 'top'
            elif y > 0.85:  # 其他高位置的点，标签放在下方
                xytext_offset = (0, -25 - i*8)
                va_align = 'top'
            elif y < 0.15:  # 低位置的点，标签放在上方
                xytext_offset = (0, 25 + i*8)
                va_align = 'bottom'
            else:  # 中间位置，根据i值交替放置
                if i < 3:  # 前3个方案放上方
                    xytext_offset = (0, 25 + i*8)
                    va_align = 'bottom'
                else:  # 后3个方案放下方
                    xytext_offset = (0, -25 - (i-3)*8)
                    va_align = 'top'
            
            # 只有在偏移较大时才显示箭头
            show_arrow = abs(xytext_offset[1]) > 30
            
            ax.annotate(label_text, (x, y), xytext=xytext_offset, 
                       textcoords='offset points', ha='center', va=va_align,
                       fontweight='bold', fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=alt_colors[i], alpha=0.95, linewidth=2),
                       arrowprops=dict(arrowstyle='->', color=alt_colors[i], lw=1.5, alpha=0.7) if show_arrow else None)
    
    # 添加情景区域背景
    for i, (x, color, label) in enumerate(zip(x_positions, method_colors.values(), x_labels)):
        ax.axvspan(x-0.3, x+0.3, alpha=0.1, color=color)
        
        # 添加情景说明文本
        ax.text(x, ax.get_ylim()[1]*0.95, label.split('\n')[0], 
               ha='center', va='top', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
        ax.text(x, ax.get_ylim()[1]*0.85, label.split('\n')[1], 
               ha='center', va='top', fontweight='bold', fontsize=10,
               style='italic')
    
    # 添加表现水平参考线 - 增大标签和改善可见性
    performance_levels = [0.2, 0.4, 0.6, 0.8]
    level_labels = ['Poor', 'Average', 'Good', 'Excellent']
    level_colors = ['#d73027', '#fdae61', '#abd9e9', '#2c7bb6']
    
    for level, label, color in zip(performance_levels, level_labels, level_colors):
        ax.axhline(y=level, color=color, linestyle='--', alpha=0.5, linewidth=1)
        ax.text(3.85, level, label, va='center', fontweight='bold', 
               color=color, fontsize=14,  # 增大字体
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor=color, alpha=0.9, linewidth=2))  # 添加背景框
    
    # 设置坐标轴
    ax.set_xlim(0.5, 4.2)  # 进一步增加右边距为性能水平标签留出空间
    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Baseline\nRanking', 'Monte Carlo\nWin Rate', 'Post-Knockout\nRobustness'], 
                      fontweight='bold', fontsize=12)
    ax.set_ylabel('Performance Score', fontweight='bold', fontsize=14)
    ax.set_title('B. Multi-Scenario Performance Comparison', fontweight='bold', fontsize=16)
    
    # 图例
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.6), fontsize=10, 
             title='Alternatives', title_fontsize=12)
    
    # 网格
    ax.grid(True, alpha=0.3, axis='y')

# ========== PANEL C: 敲除影响网络图 ==========
def plot_knockout_network(ax, top_winrate_delta, env_winrate_delta, winrates):
    """Panel C: 敲除影响网络图"""
    
    # 获取top 6方案
    top_alternatives = winrates.nlargest(6, 'win_rate')['alternative'].tolist()
    
    # 创建网络图
    G = nx.Graph()
    
    # 合并敲除数据
    all_deltas = pd.concat([
        top_winrate_delta.rename(columns={'group': 'indicator'}),
        env_winrate_delta
    ])
    
    # 筛选显著影响（阈值可调）
    significant_deltas = all_deltas[
        (all_deltas['alternative'].isin(top_alternatives)) & 
        (abs(all_deltas['delta_winrate']) > 0.02)
    ]
    
    # 添加节点
    indicators = significant_deltas['indicator'].unique()
    alternatives = significant_deltas['alternative'].unique()
    
    # 添加指标节点
    for indicator in indicators:
        G.add_node(indicator, node_type='indicator')
    
    # 添加方案节点
    for alt in alternatives:
        G.add_node(alt, node_type='alternative')
    
    # 添加边（基于影响强度）
    for _, row in significant_deltas.iterrows():
        weight = abs(row['delta_winrate'])
        G.add_edge(row['indicator'], row['alternative'], 
                  weight=weight, delta=row['delta_winrate'])
    
    # 设置布局
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # 绘制节点
    indicator_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'indicator']
    alternative_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'alternative']
    
    # 绘制指标节点（方形）
    for node in indicator_nodes:
        x, y = pos[node]
        color = INDICATOR_COLORS.get(node, '#666666')
        rect = FancyBboxPatch((x-0.08, y-0.04), 0.16, 0.08, 
                             boxstyle="round,pad=0.01", 
                             facecolor=color, edgecolor='black', 
                             linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, node.replace('_', '\n'), ha='center', va='center', 
               fontweight='bold', fontsize=8)
    
    # 绘制方案节点（圆形）
    alt_colors = plt.cm.Set1(np.linspace(0, 1, len(alternative_nodes)))
    for i, node in enumerate(alternative_nodes):
        x, y = pos[node]
        circle = Circle((x, y), 0.06, color=alt_colors[i], alpha=0.8,
                       edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, node, ha='center', va='center', 
               fontweight='bold', fontsize=9)
    
    # 绘制边
    for edge in G.edges():
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        weight = G[edge[0]][edge[1]]['weight']
        delta = G[edge[0]][edge[1]]['delta']
        
        # 边的颜色和粗细
        color = IMPACT_COLORS['positive'] if delta > 0 else IMPACT_COLORS['negative']
        linewidth = 5 * weight / significant_deltas['delta_winrate'].abs().max()
        
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, 
               alpha=0.7, zorder=1)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('C. Knockout Impact Network', fontweight='bold', fontsize=16)

# ========== PANEL D: 稳健性分析 ==========
def plot_robustness_analysis(ax, top_winrate_delta, env_winrate_delta, winrates):
    """Panel D: 稳健性分析散点图"""
    
    # 计算每个方案的稳健性指标
    all_deltas = pd.concat([
        top_winrate_delta.rename(columns={'group': 'indicator'}),
        env_winrate_delta
    ])
    
    robustness_data = []
    for alt in winrates['alternative']:
        alt_deltas = all_deltas[all_deltas['alternative'] == alt]['delta_winrate']
        base_winrate = winrates[winrates['alternative'] == alt]['win_rate'].iloc[0]
        
        # 稳健性指标
        volatility = alt_deltas.std()  # 波动性
        max_loss = -alt_deltas.min()   # 最大损失
        mean_impact = abs(alt_deltas).mean()  # 平均影响
        
        robustness_data.append({
            'alternative': alt,
            'base_winrate': base_winrate,
            'volatility': volatility,
            'max_loss': max_loss,
            'mean_impact': mean_impact
        })
    
    rob_df = pd.DataFrame(robustness_data)
    
    # 创建散点图：X轴=基准胜率，Y轴=波动性
    scatter = ax.scatter(rob_df['base_winrate'], rob_df['volatility'], 
                        s=rob_df['max_loss']*3000,  # 气泡大小表示最大损失
                        c=rob_df['mean_impact'], cmap='RdYlBu_r', 
                        alpha=0.7, edgecolors='black', linewidth=2)
    
    # 添加方案标签
    for _, row in rob_df.iterrows():
        ax.annotate(row['alternative'], 
                   (row['base_winrate'], row['volatility']),
                   xytext=(5, 15), textcoords='offset points',
                   fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.8, edgecolor='gray'))
    
    # 添加象限分析线
    median_winrate = rob_df['base_winrate'].median()
    median_volatility = rob_df['volatility'].median()
    
    ax.axvline(median_winrate, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(median_volatility, color='gray', linestyle='--', alpha=0.7)
    
    # 象限标签
    ax.text(0.95*ax.get_xlim()[1], 0.95*ax.get_ylim()[1], 
           'High Risk\nHigh Return', ha='right', va='top', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    ax.text(0.95*ax.get_xlim()[1], 0.05*ax.get_ylim()[1], 
           'Low Risk\nHigh Return', ha='right', va='bottom', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    # 设置坐标轴
    ax.set_xlabel('Baseline Win Rate', fontweight='bold', fontsize=14)
    ax.set_ylabel('Knockout Volatility (Std Dev)', fontweight='bold', fontsize=14)
    ax.set_title('D. Robustness Analysis: Risk vs Return', fontweight='bold', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Mean Impact Magnitude', fontweight='bold')

# ========== MAIN VISUALIZATION FUNCTION ==========
def create_knockout_analysis_visualization():
    """创建完整的敲除分析可视化"""
    
    print("加载敲除分析数据...")
    top_flip, env_flip, top_winrate_delta, env_winrate_delta, winrates = load_knockout_data()
    
    # 设置样式
    set_nature_style()
    
    # 创建1x2布局 - 恢复原始高度
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))  # 恢复到8
    fig.subplots_adjust(wspace=0.1)  # 减少间距从0.4到0.2
    
    # Panel A: 指标重要性排名
    plot_indicator_importance(ax1, top_flip, env_flip)
    
    # Panel B: 多情景表现对比
    plot_scenario_performance_comparison(ax2, top_winrate_delta, env_winrate_delta, winrates)
    
    # 整体标题 - 移除大标题
    # plt.suptitle('SCC Analysis: Indicator Importance and Multi-Scenario Performance', 
    #              fontsize=20, fontweight='bold', y=0.95)
    
    # 保存图形
    save_path = FIG_DIR / "figG_scc_knockout_analysis.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n敲除分析可视化已保存到: {save_path}")
    print("图形组件:")
    print("  • Panel A: 指标重要性排名（翻转概率）")
    print("  • Panel B: 多情景表现对比（固定权重 vs 蒙特卡洛 vs 敲除）")

if __name__ == "__main__":
    print("=== SCC敲除分析可视化 ===")
    create_knockout_analysis_visualization()