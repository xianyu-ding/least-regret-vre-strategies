# -*- coding: utf-8 -*-
"""
真实SCC结果可视化 - 使用计算过程中保存的真实数据
Panel A: 真实的胜率收敛过程
Panel B: 真实的权重场景下得分分布
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ========== PATHS ==========
BASE_DIR = Path(r"C:/Users/vanessa/OneDrive - University College London/research/data/paper3/MCDM_FINAL")
SCC_DIR = BASE_DIR / "scc_out"
FIG_DIR = BASE_DIR / "eda_out" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========== COLORS ==========
CATEGORY_COLORS = {
    "Economic": "#2E86AB",
    "Technical": "#A23B72", 
    "Local Environmental": "#F18F01",
    "Global Environmental": "#C73E1D"
}

PERFORMANCE_COLORS = {
    "excellent": "#1a9850",
    "good": "#91bfdb", 
    "average": "#fee08b",
    "poor": "#d73027"
}

# ========== STYLE SETUP ==========
def set_nature_advanced_style():
    """Advanced Nature style for dynamic visualization"""
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
        "font.size": 12,
        "font.weight": "bold",
        "axes.linewidth": 1.5,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "axes.labelweight": "bold",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",
        "grid.alpha": 0.3,
    })

# ========== DATA LOADING ==========
def load_real_scc_data():
    """加载真实的SCC计算数据"""
    try:
        # 加载最终结果
        winrates = pd.read_csv(SCC_DIR / "scc_winrates.csv")
        rank_stats = pd.read_csv(SCC_DIR / "scc_rank_stats.csv") 
        baseline = pd.read_csv(SCC_DIR / "baseline_ranking.csv")
        
        # 加载真实的演化数据
        evolution_real = pd.read_csv(SCC_DIR / "scc_evolution_real.csv")
        
        # 加载权重场景数据
        try:
            scenarios_real = pd.read_csv(SCC_DIR / "scc_weight_scenarios_real.csv")
        except:
            scenarios_real = None
            print("警告: 未找到权重场景数据，将使用备用方案")
        
        try:
            normalized = pd.read_csv(SCC_DIR / "normalized_scores.csv")
        except:
            normalized = None
        
        # 替换标签
        for df in [winrates, rank_stats, baseline, evolution_real]:
            if df is not None and 'alternative' in df.columns:
                df['alternative'] = df['alternative'].str.replace('ndc-ltp', 'Netzero', case=False)
        
        if scenarios_real is not None and 'alternative' in scenarios_real.columns:
            scenarios_real['alternative'] = scenarios_real['alternative'].str.replace('ndc-ltp', 'Netzero', case=False)
            
        if normalized is not None and 'alternative' in normalized.columns:
            normalized['alternative'] = normalized['alternative'].str.replace('ndc-ltp', 'Netzero', case=False)
            
        return winrates, rank_stats, baseline, evolution_real, scenarios_real, normalized
    except Exception as e:
        raise FileNotFoundError(f"无法加载SCC结果数据: {SCC_DIR}: {e}")

def prepare_panel_a_data(evolution_real, winrates):
    """准备Panel A的真实收敛数据"""
    # 获取胜率最高的6个方案用于可视化
    top_alternatives = winrates.nlargest(6, 'win_rate')['alternative'].tolist()
    
    # 筛选这些方案的演化数据
    panel_a_data = evolution_real[evolution_real['alternative'].isin(top_alternatives)].copy()
    
    print(f"Panel A: 使用 {len(top_alternatives)} 个顶级方案的真实收敛数据")
    print(f"数据点总数: {len(panel_a_data)}")
    
    return panel_a_data, top_alternatives

def prepare_panel_b_data(scenarios_real, baseline, winrates):
    """准备Panel B的真实权重场景数据"""
    if scenarios_real is None:
        print("使用基准数据创建Panel B的备用可视化")
        # 使用基准排名前6的方案
        top_alts = baseline.head(6)['alternative'].tolist()
        return None, top_alts
    
    # 获取与Panel A相同的方案
    top_alternatives = winrates.nlargest(6, 'win_rate')['alternative'].tolist()
    
    # 筛选这些方案的权重场景数据
    panel_b_data = scenarios_real[scenarios_real['alternative'].isin(top_alternatives)].copy()
    
    print(f"Panel B: 使用 {len(top_alternatives)} 个方案的真实权重场景数据")
    print(f"权重场景数据点: {len(panel_b_data)}")
    
    return panel_b_data, top_alternatives

# ========== CUSTOM FORMATTER FOR LOG AXIS ==========
def format_log_axis(ax):
    """自定义格式化对数坐标轴，显示常规数字而非科学计数法"""
    from matplotlib.ticker import FuncFormatter, LogLocator
    
    def log_formatter(x, pos):
        """将对数坐标的标签格式化为常规数字"""
        if x >= 1000:
            return f'{int(x/1000)}K'
        else:
            return f'{int(x)}'
    
    ax.xaxis.set_major_formatter(FuncFormatter(log_formatter))
    # 设置主要刻度位置
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))

# ========== VISUALIZATION ==========
def create_real_scc_visualization():
    """基于真实SCC数据创建可视化"""
    
    # 加载真实数据
    print("加载真实SCC数据...")
    winrates, rank_stats, baseline, evolution_real, scenarios_real, normalized = load_real_scc_data()
    
    # 准备面板数据
    panel_a_data, top_alts_a = prepare_panel_a_data(evolution_real, winrates)
    panel_b_data, top_alts_b = prepare_panel_b_data(scenarios_real, baseline, winrates)
    
    # 设置图形样式
    set_nature_advanced_style()
    fig, (ax_evol, ax_dist) = plt.subplots(1, 2, figsize=(16, 8))
    fig.subplots_adjust(wspace=0.3)
    
    # 颜色方案 - 确保Panel A和B使用相同的颜色
    colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b'   # Brown
    ]
    
    # ========== Panel A: 真实的胜率收敛过程 ==========
    print("绘制Panel A: 真实胜率收敛曲线...")
    
    for i, alt in enumerate(top_alts_a):
        alt_data = panel_a_data[panel_a_data['alternative'] == alt].sort_values('iteration')
        
        if len(alt_data) == 0:
            continue
        
        # 对数据进行平滑处理，减少尖锐变化
        if len(alt_data) > 3:
            # 使用移动平均进行平滑
            window_size = min(5, len(alt_data) // 10 + 1)
            smoothed_rates = alt_data['cumulative_win_rate'].rolling(window=window_size, center=True, min_periods=1).mean()
            smoothed_lower = alt_data['confidence_lower'].rolling(window=window_size, center=True, min_periods=1).mean()
            smoothed_upper = alt_data['confidence_upper'].rolling(window=window_size, center=True, min_periods=1).mean()
        else:
            smoothed_rates = alt_data['cumulative_win_rate']
            smoothed_lower = alt_data['confidence_lower']
            smoothed_upper = alt_data['confidence_upper']
            
        # 绘制主收敛曲线 - 使用平滑后的数据
        ax_evol.plot(alt_data['iteration'], smoothed_rates, 
                    color=colors[i], linewidth=3, alpha=0.8, label=alt,
                    linestyle='-', marker='', markersize=0)
        
        # 绘制置信带 - 使用平滑后的数据
        ax_evol.fill_between(alt_data['iteration'], 
                           smoothed_lower, 
                           smoothed_upper,
                           color=colors[i], alpha=0.2)
        
        # 添加最终性能箭头 (对数坐标下的优化)
        if len(alt_data) > 10:
            final_performance = smoothed_rates.iloc[-1]
            final_iter = alt_data['iteration'].iloc[-1]
            
            # 在对数坐标下，箭头起点需要特殊计算
            log_final = np.log10(final_iter)
            log_start = log_final - 0.3  # 在对数空间中后退0.3个单位
            arrow_start = 10**log_start
            
            ax_evol.annotate('', xy=(final_iter, final_performance), 
                            xytext=(arrow_start, final_performance),
                            arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
    
    ax_evol.set_ylim(0, 1)
    ax_evol.set_xscale('log')  # 设置x轴为对数坐标
    ax_evol.set_xlim(9, None)  # 设置x轴从9开始
    ax_evol.set_xlabel('Monte Carlo Iteration (log scale)', fontweight='bold', fontsize=15)
    ax_evol.set_ylabel('Cumulative Win Rate', fontweight='bold', fontsize=15)
    ax_evol.set_title('A. Real Performance Evolution: Win Rate Convergence', 
                     fontweight='bold', fontsize=17)
    ax_evol.legend(fontsize=11, loc='center left', bbox_to_anchor=(0.7, 0.5))
    ax_evol.grid(True, alpha=0.3, which='both')  # 显示主要和次要网格线
    
    # 应用自定义格式化器
    format_log_axis(ax_evol)
    
    # 样式化标签
    for label in ax_evol.get_xticklabels() + ax_evol.get_yticklabels():
        label.set_fontweight('bold')
    
    # ========== Panel B: 真实权重场景下的得分分布 ==========
    print("绘制Panel B: 真实得分分布...")
    
    if panel_b_data is not None:
        # 使用真实的权重场景数据创建violin图
        violin_data = []
        positions = []
        colors_violin = []
        performance_levels = []
        
        for i, alt in enumerate(top_alts_b):
            alt_scores = panel_b_data[panel_b_data['alternative'] == alt]['total_score'].values
            
            if len(alt_scores) == 0:
                continue
                
            violin_data.append(alt_scores)
            positions.append(i)
            colors_violin.append(colors[i])
            
            # 基于平均得分确定性能等级 (得分越低越好)
            mean_score = np.mean(alt_scores)
            all_means = []
            for a in top_alts_b:
                alt_s = panel_b_data[panel_b_data['alternative'] == a]['total_score'].values
                if len(alt_s) > 0:
                    all_means.append(np.mean(alt_s))
            
            if len(all_means) > 0:
                if mean_score < np.percentile(all_means, 25):
                    performance_levels.append('excellent')
                elif mean_score < np.percentile(all_means, 50):
                    performance_levels.append('good')
                elif mean_score < np.percentile(all_means, 75):
                    performance_levels.append('average')
                else:
                    performance_levels.append('poor')
            else:
                performance_levels.append('average')
        
        # 创建violin图
        if len(violin_data) > 0:
            parts = ax_dist.violinplot(violin_data, positions=positions, widths=0.7, showmeans=True)
            
            # 为violin着色
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors_violin[i])
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.2)
            
            # 增强violin图元素
            parts['cmeans'].set_color('red')
            parts['cmeans'].set_linewidth(2)
            
            ax_dist.set_xticks(positions)
            ax_dist.set_xticklabels([top_alts_b[i] for i in positions], rotation=45, ha='right', fontweight='bold')
            
            # 固定标签重叠问题 - 使用更智能的标签位置策略
            y_max = ax_dist.get_ylim()[1]
            y_min = ax_dist.get_ylim()[0]
            y_range = y_max - y_min
            
            # 创建标签位置的垂直偏移数组，避免重叠
            label_offsets = []
            excellent_count = sum(1 for level in performance_levels if level == 'excellent')
            
            if excellent_count >= 2:
                # 如果有多个excellent标签，使用交替的高度
                excellent_positions = [0.92, 0.85, 0.78, 0.92, 0.85, 0.78]  # 循环使用不同高度
            else:
                excellent_positions = [0.92] * 6
            
            excellent_counter = 0
            
            for i, (pos, level) in enumerate(zip(positions, performance_levels)):
                if level == 'excellent':
                    y_offset = excellent_positions[excellent_counter % len(excellent_positions)]
                    excellent_counter += 1
                elif level == 'good':
                    y_offset = 0.88
                elif level == 'average':
                    y_offset = 0.84
                else:  # poor
                    y_offset = 0.92
                
                # 根据性能等级调整字体大小
                font_size = 12 if level == 'excellent' and excellent_count >= 2 else 14
                
                ax_dist.text(pos, y_max * y_offset, level.title(), 
                            ha='center', va='top', fontweight='bold', fontsize=font_size,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=PERFORMANCE_COLORS[level], 
                                     alpha=0.8, edgecolor='black', linewidth=1.2))
        
        ax_dist.set_ylabel('Total Score Distribution (Real Data)', fontweight='bold', fontsize=15)
        ax_dist.set_title('B. Real Score Variability Across Weight Scenarios', 
                         fontweight='bold', fontsize=17)
        
    else:
        # 备用方案：显示基准排名
        print("使用备用可视化方案...")
        baseline_top = baseline.head(6)
        
        bars = ax_dist.bar(range(len(baseline_top)), baseline_top['U_total'], 
                          color=colors[:len(baseline_top)], alpha=0.7, edgecolor='black')
        
        ax_dist.set_xticks(range(len(baseline_top)))
        ax_dist.set_xticklabels(baseline_top['alternative'], rotation=45, ha='right', fontweight='bold')
        ax_dist.set_ylabel('Baseline Total Score', fontweight='bold', fontsize=15)
        ax_dist.set_title('B. Baseline Performance Comparison', fontweight='bold', fontsize=17)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax_dist.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax_dist.grid(True, alpha=0.3, axis='y')
    
    for label in ax_dist.get_yticklabels():
        label.set_fontweight('bold')
    
    # 总标题
    #plt.suptitle('Real SCC Analysis: Evolution Curves and Score Distributions', 
             #    fontsize=18, fontweight='bold', y=0.95)
    
    # 保存图形
    save_path = FIG_DIR / "figF_scc_real_data_panels_AB.png"
    fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\n真实SCC可视化已保存到: {save_path}")
    print("图形组件:")
    print("  • Panel A: 真实的胜率收敛曲线 (基于实际MC迭代)")
    if panel_b_data is not None:
        print("  • Panel B: 真实权重场景下的得分分布")
    else:
        print("  • Panel B: 基准性能比较 (备用方案)")
    print(f"  • 可视化方案数量: {len(top_alts_a)}")

def print_data_summary():
    """打印数据摘要信息"""
    try:
        winrates, rank_stats, baseline, evolution_real, scenarios_real, normalized = load_real_scc_data()
        
        print("\n=== 真实SCC数据摘要 ===")
        print(f"替代方案总数: {len(winrates)}")
        print(f"演化数据点: {len(evolution_real)}")
        if scenarios_real is not None:
            print(f"权重场景数据点: {len(scenarios_real)}")
        
        print(f"\n胜率排名前5:")
        top5 = winrates.head(5)
        for i, row in top5.iterrows():
            print(f"  {row['alternative']}: {row['win_rate']:.3f}")
            
        print(f"\n演化数据覆盖的迭代范围:")
        print(f"  最小迭代: {evolution_real['iteration'].min()}")
        print(f"  最大迭代: {evolution_real['iteration'].max()}")
        print(f"  数据点密度: {len(evolution_real) / evolution_real['iteration'].max():.2f} 点/迭代")
        
    except Exception as e:
        print(f"无法加载数据摘要: {e}")

if __name__ == "__main__":
    print("=== 基于真实SCC数据的可视化 ===")
    print_data_summary()
    create_real_scc_visualization()