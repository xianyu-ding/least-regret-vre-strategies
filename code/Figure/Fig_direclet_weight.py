# -*- coding: utf-8 -*-
"""
Monte Carlo权重抽样分布可视化 - 基于实际SCCA分析数据
单一面板：真实Monte Carlo权重抽样分布散点图
"""

from pipeline_common import (SCORE_DIR, RESULTS_DIR, WEIGHTS_DIR, INDICATOR_PATH, rank_scores, winner_shares, draw_dirichlet)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import matplotlib.patches as mpatches

# ========== PATHS ==========
BASE_DIR = RESULTS_DIR
SCC_DIR = RESULTS_DIR / "scc"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========== COLOR SCHEME ==========
PERSPECTIVE_COLORS = {
    'comprehensive': '#1f4e79',     # Deep Blue - Professional
    'government': '#8B0000',        # Dark Red - Authority
    'news': '#FF8C00',              # Dark Orange - Media
    'research': '#2F4F2F',          # Dark Green - Academic
    'social_media': '#4B0082'       # Indigo - Social
}

CRITERIA_COLORS = {
    "Cost": "#1f77b4",
    "LOLE": "#ff7f0e", 
    "Environment": "#2ca02c",
    "CO2": "#d62728"
}

# ========== STYLE SETUP ==========
def set_nature_style():
    """Nature journal style settings"""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "font.size": 10,
        "font.weight": "normal",
        "axes.linewidth": 1.2,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black",
        "axes.grid": True,
        "axes.axisbelow": True,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.8,
        "grid.color": "gray",
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1,
    })

# ========== DATA LOADING ==========
def load_real_weight_scenario_data():
    """加载真实的Monte Carlo权重场景数据"""
    weight_files = []
    
    perspectives = ['comprehensive', 'government', 'news', 'research', 'social_media']
    
    for perspective in perspectives:
        weight_file = SCC_DIR / f"scc_weight_scenarios_{perspective}.csv"
        if weight_file.exists():
            try:
                df = pd.read_csv(weight_file)
                weight_files.append(df)
                print(f"✓ 成功加载 {perspective} 权重场景数据: {len(df)} 条记录")
            except Exception as e:
                print(f"✗ 无法加载 {perspective} 权重数据: {e}")
        else:
            print(f"✗ 权重文件不存在: {weight_file}")
    
    if not weight_files:
        raise FileNotFoundError(f"无法找到任何权重场景数据文件在 {SCC_DIR}")
    
    # 合并所有权重数据
    all_weights = pd.concat(weight_files, ignore_index=True)
    print(f"📊 总共加载了 {len(all_weights)} 条权重记录")
    
    # 显示数据结构信息
    print(f"📋 数据包含列: {list(all_weights.columns)}")
    print(f"🔍 视角数量: {all_weights['perspective'].nunique()}")
    print(f"🔢 迭代数量: {all_weights['iteration'].nunique()}")
    
    return all_weights

def process_real_weight_data(weight_data):
    """处理真实权重数据用于可视化"""
    processed_data = []
    
    for _, row in weight_data.iterrows():
        # 提取各准则权重（基于实际数据列名）
        criteria_weights = {
            'Cost': row.get('w_cost', 0),
            'LOLE': row.get('w_lole', 0), 
            'Environment': row.get('w_environment', 0),
            'CO2': row.get('w_co2', 0)
        }
        
        for criterion, weight in criteria_weights.items():
            processed_data.append({
                'iteration': row['iteration'],
                'perspective': row['perspective'],
                'alternative': row.get('alternative', 'unknown'),
                'criterion': criterion,
                'weight': weight
            })
    
    return pd.DataFrame(processed_data)

# ========== VISUALIZATION ==========
def plot_real_weight_distribution(ax, weight_data):
    """绘制真实Monte Carlo权重分布散点图"""
    
    if len(weight_data) == 0:
        ax.text(0.5, 0.5, 'No weight data available', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, fontweight='bold')
        return
    
    # 准则顺序
    criteria = ['Cost', 'LOLE', 'Environment', 'CO2']
    
    # 过滤存在的准则
    available_criteria = [c for c in criteria if c in weight_data['criterion'].values]
    if not available_criteria:
        ax.text(0.5, 0.5, 'No matching criteria found', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, fontweight='bold')
        return
    
    # 设置绘图位置
    x_positions = np.arange(len(available_criteria))
    width = 0.15  # 每个视角的宽度
    
    # 获取所有视角
    perspectives = weight_data['perspective'].unique()
    perspectives = [p for p in perspectives if p in PERSPECTIVE_COLORS]
    
    print(f"🎨 绘制视角: {perspectives}")
    print(f"📊 绘制准则: {available_criteria}")
    
    # 为每个视角绘制散点图
    for i, perspective in enumerate(perspectives):
        # 获取该视角的数据
        persp_data = weight_data[weight_data['perspective'] == perspective]
        
        if len(persp_data) == 0:
            continue
        
        # 计算x位置偏移
        x_offset = (i - len(perspectives)/2 + 0.5) * width
        color = PERSPECTIVE_COLORS[perspective]
        
        # 为每个准则绘制散点和统计
        for j, criterion in enumerate(available_criteria):
            criterion_data = persp_data[persp_data['criterion'] == criterion]
            
            if len(criterion_data) == 0:
                continue
                
            x_pos = x_positions[j] + x_offset
            weights = criterion_data['weight'].values
            
            # 先绘制箱线图（使用全部数据）
            if len(weights) > 10:  # 只有足够数据时才绘制箱线图
                try:
                    bp = ax.boxplot([weights], positions=[x_pos], 
                                  widths=width*0.5, patch_artist=True, 
                                  showfliers=False, showcaps=True,
                                  medianprops=dict(color='black', linewidth=2),
                                  whiskerprops=dict(color=color, linewidth=1.5, alpha=0.8),
                                  capprops=dict(color=color, linewidth=1.5, alpha=0.8))
                    
                    # 设置箱体样式
                    bp['boxes'][0].set_facecolor(color)
                    bp['boxes'][0].set_alpha(0.25)
                    bp['boxes'][0].set_edgecolor(color)
                    bp['boxes'][0].set_linewidth(1.5)
                    
                    # 获取箱线图的上下限，用于过滤散点数据
                    q1 = np.percentile(weights, 25)
                    q3 = np.percentile(weights, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # 只显示箱线图范围内的散点（避免异常值超出箱线图）
                    mask = (weights >= lower_bound) & (weights <= upper_bound)
                    weights_filtered = weights[mask]
                    
                except:
                    # 如果箱线图绘制失败，使用全部数据
                    weights_filtered = weights
            else:
                weights_filtered = weights
            
            # 数据抽样（基于过滤后的数据，避免过于密集）
            sample_size = min(2000, len(weights_filtered))
            if len(weights_filtered) > sample_size:
                indices = np.random.choice(len(weights_filtered), sample_size, replace=False)
                weights_sample = weights_filtered[indices]
            else:
                weights_sample = weights_filtered
            
            # 添加随机抖动避免重叠
            jitter_strength = width * 0.08
            np.random.seed(42 + j + i*10)  # 不同位置使用不同种子避免模式重复
            x_jitter = np.random.normal(0, jitter_strength, len(weights_sample))
            x_positions_jittered = np.full(len(weights_sample), x_pos) + x_jitter
            
            # 绘制散点（在箱线图之上）
            scatter = ax.scatter(x_positions_jittered, weights_sample, 
                               s=15, c=[color], alpha=0.6,
                               edgecolors='white', linewidth=0.3,
                               marker='o', zorder=4)  # 提高zorder确保在箱线图之上
        
        # 为每个视角创建图例标签
        legend_label = perspective.replace("_", " ").title()
        ax.scatter([], [], s=40, c=[color], alpha=0.8,
                  edgecolors='white', linewidth=0.5, marker='o',
                  label=legend_label)
    
    # 添加准则平均线（跨视角平均）
    for j, criterion in enumerate(available_criteria):
        criterion_data = weight_data[weight_data['criterion'] == criterion]
        if len(criterion_data) > 0:
            overall_mean = criterion_data['weight'].mean()
            ax.axhline(y=overall_mean, 
                      xmin=(j-0.45)/len(available_criteria), 
                      xmax=(j+0.45)/len(available_criteria),
                      color=CRITERIA_COLORS.get(criterion, 'gray'), 
                      linewidth=3, alpha=0.7, linestyle=':',
                      zorder=1)
    
    # 设置轴标签和标题 - 更大更醒目的字体
    ax.set_xticks(x_positions)
    ax.set_xticklabels(available_criteria, fontweight='bold', fontsize=14)  # 增大x轴标签
    ax.set_ylabel('Criterion Weight', fontweight='bold', fontsize=14)  # 增大y轴标签
    #ax.set_title('Multi-Perspective Monte Carlo Weight Distribution\n(Real SCCA Analysis Data)', 
                #fontweight='bold', fontsize=12, pad=15)
    
    # 增大刻度数字
    ax.tick_params(axis='both', which='major', labelsize=15)  # 增大刻度标签
    
    # 添加网格和样式
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_ylim(0, min(1.0, weight_data['weight'].max() * 1.1))
    
    # 图例 - 更大更醒目，分两行显示
    legend_elements = []
    
    # 添加视角图例（散点）
    for perspective in perspectives:
        if perspective in PERSPECTIVE_COLORS:
            color = PERSPECTIVE_COLORS[perspective]
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color, markersize=12, alpha=0.8,
                          markeredgecolor='white', markeredgewidth=0.5,
                          label=perspective.replace("_", " ").title())
            )
    
    # 添加虚线含义说明
    legend_elements.append(
        plt.Line2D([0], [0], color='gray', linewidth=3, linestyle=':',
                  alpha=0.7, label='Cross-Perspective Average')
    )
    
    ax.legend(handles=legend_elements, title='Perspectives & Elements', 
             loc='upper left', bbox_to_anchor=(0.02, 0.98), 
             fontsize=11, title_fontsize=12, framealpha=0.95,
             markerscale=1.3, handletextpad=0.6, ncol=3)  # ncol=3 使图例分布在2-3行

def create_real_weight_visualization():
    """创建真实权重数据可视化"""
    
    print("=== 加载真实SCCA权重场景数据 ===")
    
    # 加载真实权重场景数据
    try:
        weight_data = load_real_weight_scenario_data()
        processed_weights = process_real_weight_data(weight_data)
        print(f"✅ 数据处理完成，共 {len(processed_weights)} 条权重记录")
    except Exception as e:
        print(f"❌ 错误: 无法加载权重数据: {e}")
        return
    
    # 设置样式
    set_nature_style()
    
    # 创建单一图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    print("🎨 绘制权重分布图...")
    plot_real_weight_distribution(ax, processed_weights)
    
    # 最终调整
    plt.tight_layout()
    
    # 保存图形
    save_path = FIG_DIR / "real_monte_carlo_weight_distribution.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ 权重分布图已保存到: {save_path}")
    
    # 保存处理后的数据
    analysis_save_path = SCC_DIR / "processed_real_weight_data.csv"
    processed_weights.to_csv(analysis_save_path, index=False)
    print(f"📄 处理后数据已保存到: {analysis_save_path}")
    
    # 生成详细统计摘要
    print(f"\n=== 真实权重分布统计摘要 ===")
    
    # 整体统计
    print(f"📊 总体统计:")
    print(f"   • 总迭代数: {processed_weights['iteration'].nunique():,}")
    print(f"   • 总视角数: {processed_weights['perspective'].nunique()}")
    print(f"   • 总记录数: {len(processed_weights):,}")
    
    # 按视角和准则统计
    print(f"\n📋 按视角-准则统计 (均值 ± 标准差):")
    for perspective in processed_weights['perspective'].unique():
        print(f"\n   {perspective.upper()}:")
        persp_data = processed_weights[processed_weights['perspective'] == perspective]
        for criterion in ['Cost', 'LOLE', 'Environment', 'CO2']:
            crit_data = persp_data[persp_data['criterion'] == criterion]
            if len(crit_data) > 0:
                mean_val = crit_data['weight'].mean()
                std_val = crit_data['weight'].std()
                min_val = crit_data['weight'].min()
                max_val = crit_data['weight'].max()
                print(f"     {criterion:12s}: {mean_val:.3f} ± {std_val:.3f} [{min_val:.3f}, {max_val:.3f}]")
    
    # 准则间差异分析
    print(f"\n🔍 准则间权重变异性分析:")
    for criterion in ['Cost', 'LOLE', 'Environment', 'CO2']:
        criterion_data = processed_weights[processed_weights['criterion'] == criterion]
        if len(criterion_data) > 0:
            # 计算视角间差异
            perspective_means = criterion_data.groupby('perspective')['weight'].mean()
            inter_perspective_std = perspective_means.std()
            
            # 计算总体变异
            total_std = criterion_data['weight'].std()
            
            print(f"   {criterion:12s}: 视角间差异={inter_perspective_std:.4f}, 总体变异={total_std:.4f}")

if __name__ == "__main__":
    print("=== 真实Monte Carlo权重分布可视化 ===")
    create_real_weight_visualization()
