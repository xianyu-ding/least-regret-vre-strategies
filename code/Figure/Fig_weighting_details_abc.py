# -*- coding: utf-8 -*-
"""
Source-Specific Weight Variation Analysis - Nature Style
Shows how different sources (Government, News, Research, Social media) 
weight different criteria differently
"""

from pipeline_common import (SCORE_DIR, RESULTS_DIR, WEIGHTS_DIR, INDICATOR_PATH, rank_scores, winner_shares, draw_dirichlet)
import os, json, re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import seaborn as sns

# ========== PATHS ==========
BASE_DIR = str(SCORE_DIR)
OUT_DIR  = str(RESULTS_DIR)
FIG_DIR  = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FILES = {
    "govern":   os.path.join(BASE_DIR, "govern_scores_only.jsonl"),
    "news":     os.path.join(BASE_DIR, "news_scores_only.jsonl"),
    "research": os.path.join(BASE_DIR, "research_scores_only.jsonl"),
    "zhihu":    os.path.join(BASE_DIR, "zhihu_scores_only.jsonl"),
}

# ========== HYPERS ==========
LAMBDA = 0.5
BETA = 1.0
EPS = 1e-12

# ========== INDICATORS ==========
INDICATORS = [
    "total_transition_cost","lole","land_occupation",
    "freshwater_ecotoxicity","human_toxicity","ionising_radiation",
    "freshwater_eutrophication","marine_eutrophication",
    "mineral_resource_depletion","co2_emission",
]

INDICATOR_ALIASES = {
    "total_transition_cost": "Transition Cost",
    "lole": "Reliability (LOLE)",  # 修改：从 "Reliability (LOLE)" 改为 "LOLE"
    "land_occupation": "Land Use",
    "freshwater_ecotoxicity": "Freshwater Toxicity",
    "human_toxicity": "Human Toxicity",
    "ionising_radiation": "Radiation",
    "freshwater_eutrophication": "Freshwater Eutroph.",
    "marine_eutrophication": "Marine Eutroph.",
    "mineral_resource_depletion": "Mineral Depletion",
    "co2_emission": "CO2 Emissions",
}

SOURCE_ALIASES = {
    "govern": "Government",
    "news": "News",
    "research": "Research",
    "zhihu": "Social Media",
}

ENV_SET = [
    "land_occupation","freshwater_ecotoxicity","human_toxicity",
    "ionising_radiation","freshwater_eutrophication",
    "marine_eutrophication","mineral_resource_depletion",
]

# ========== ENHANCED COLORS ==========
SOURCE_COLORS = {
    "Government": "#2E86AB",
    "News": "#A23B72", 
    "Research": "#F18F01",
    "Social Media": "#C73E1D"
}

CATEGORY_COLORS = {
    "Cost": "#2E86AB",           # Economic
    "LOLE": "#A23B72",           # 修改：从 "LOLE" 改为 "LOLE"
    "Environment": "#F18F01",    # Local Environmental  
    "Global (CO2)": "#C73E1D"    # Global Environmental
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

# All indicators colors (for Panel B) - 按照新的分类体系
INDICATOR_COLORS = {
    "total_transition_cost": "#2E86AB",  # Economic (Cost)
    "lole": "#A23B72",                   # Technical (LOLE)
    "co2_emission": "#C73E1D",           # Global Environmental
    # Local Environmental indicators (所有环境指标)
    "land_occupation": "#F18F01",
    "freshwater_ecotoxicity": "#F18F01", 
    "human_toxicity": "#F18F01",
    "ionising_radiation": "#F18F01",
    "freshwater_eutrophication": "#F18F01",
    "marine_eutrophication": "#F18F01",
    "mineral_resource_depletion": "#F18F01"
}

# ========== STYLE SETUP ==========
def set_nature_source_style():
    """Enhanced Nature style for source variation visualization"""
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

# ========== DATA PROCESSING ==========
def parse_score_value(v):
    from pipeline_common import parse_score_value as parse
    return parse(v)

def normalize_scores_dict(d):
    from pipeline_common import normalize_scores_dict as validate
    return validate(d)

def flatten_assessment(ass):
    from pipeline_common import flatten_assessment as flatten
    return flatten(ass)

def read_jsonl_scores(path, source_tag):
    from pipeline_common import read_jsonl_scores as read
    return read(path, source_tag, include_length=False)

def collapse_to_tiers(df):
    long = df.melt(id_vars=["source"], value_vars=INDICATORS,
                   var_name="indicator", value_name="score")
    long["tier"] = np.select(
        [long["score"]==1, long["score"]==2, long["score"]>=3],
        [1, 2, 3]
    )
    return long

def per_source_tier_props(tri_long):
    counts = tri_long.pivot_table(index=["source","indicator"], columns="tier",
                                  values="score", aggfunc="count", fill_value=0)
    for t in [1,2,3]:
        if t not in counts.columns:
            counts[t] = 0
    counts = counts[[1,2,3]].sort_index()
    denom = counts.sum(axis=1).replace(0, np.nan)
    props = counts.div(denom, axis=0).reset_index().rename(columns={1:"q1",2:"q2",3:"q3"})
    props["n"] = counts.sum(axis=1).values
    return props

def calculate_source_specific_weights(per_src):
    from Data_weight import (calculate_top_level_weights, calculate_env_sub_weights,
                             calculate_overall_weights, INDICATOR_ALIASES)
    reverse = {alias: canonical for canonical, alias in INDICATOR_ALIASES.items()}
    output = {}
    for source, frame in per_src.groupby('source'):
        Q = frame.copy()
        Q['z_k'] = Q['q3'] + LAMBDA * Q['q2']
        overall = calculate_overall_weights(calculate_top_level_weights(Q), calculate_env_sub_weights(Q))
        output[source] = {reverse[key]: value for key, value in overall.items()}
    return output

def calculate_top_level_by_source(source_weights):
    return {source: {'Cost': weights['total_transition_cost'], 'LOLE': weights['lole'],
                     'Environment': sum(weights[key] for key in ENV_SET),
                     'Global (CO2)': weights['co2_emission']}
            for source, weights in source_weights.items()}

def calculate_environmental_by_source(source_weights):
    """Calculate environmental sub-weights for each source (normalized within environment)"""
    env_by_source = {}
    
    for source, weights in source_weights.items():
        env_weights = {ind: weights.get(ind, 0) for ind in ENV_SET}
        
        # Normalize within environment category
        total_env = sum(env_weights.values())
        if total_env > EPS:
            env_weights = {k: v/total_env for k, v in env_weights.items()}
        else:
            env_weights = {k: 1.0/len(ENV_SET) for k in ENV_SET}
        
        env_by_source[source] = env_weights
    
    return env_by_source

# ========== VISUALIZATION ==========
def create_source_weight_variation():
    """Create comprehensive source weight variation visualization"""
    
    # Load and process data
    frames = []
    for src, path in FILES.items():
        df = read_jsonl_scores(path, src)
        if not df.empty:
            frames.append(df)
    
    if not frames:
        print("[ERROR] no valid input rows.")
        return
    
    df_all = pd.concat(frames, ignore_index=True)
    tri = collapse_to_tiers(df_all)
    per_src = per_source_tier_props(tri)
    
    # Calculate source-specific weights
    source_weights = calculate_source_specific_weights(per_src)
    top_level_by_source = calculate_top_level_by_source(source_weights)
    env_by_source = calculate_environmental_by_source(source_weights)
    
    # Setup figure - FIXED LAYOUT
    set_nature_source_style()
    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(2, 6, height_ratios=[1, 1.2], width_ratios=[1.5, 1.5, 0.2, 1.8, 0.3, 0.2],
                  hspace=0.35, wspace=0.5)
    
    # ========== Panel A: Top-Level Weight Comparison ==========
    ax_top = fig.add_subplot(gs[0, :2])
    
    # Prepare data for grouped bar chart
    categories = list(CATEGORY_COLORS.keys())
    sources = list(SOURCE_COLORS.keys())
    
    x = np.arange(len(categories))
    width = 0.2
    
    for i, source in enumerate(sources):
        source_key = list(SOURCE_ALIASES.keys())[list(SOURCE_ALIASES.values()).index(source)]
        weights = [top_level_by_source[source_key].get(cat, 0) for cat in categories]
        
        bars = ax_top.bar(x + i*width, weights, width, label=source, 
                         color=SOURCE_COLORS[source], alpha=0.8, 
                         edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            if height > 0.01:
                ax_top.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                           f'{weight:.3f}', ha='center', va='bottom',
                           fontweight='bold', fontsize=10)
    
    ax_top.set_xlabel('Top-Level Categories', fontweight='bold', fontsize=16)
    ax_top.set_ylabel('Weight', fontweight='bold', fontsize=16)
    ax_top.set_title('A. Top-Level Criteria Weights by Source', fontweight='bold', fontsize=18)
    ax_top.set_xticks(x + width * 1.5)
    ax_top.set_xticklabels(categories, fontweight='bold', fontsize=14)
    ax_top.legend(fontsize=13, frameon=True, edgecolor='black')
    ax_top.grid(True, alpha=0.3, axis='y')
    
    # Remove top and right spines
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    
    # Style tick labels and color-code x-axis labels by category
    for label in ax_top.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(14)
    
    # Color-code x-axis labels by category colors
    for i, (label, category) in enumerate(zip(ax_top.get_xticklabels(), categories)):
        label.set_color(CATEGORY_COLORS[category])
        label.set_fontweight('bold')
        label.set_fontsize(14)
    
    # ========== Panel B: Individual Indicator Heatmap - FIXED POSITION ==========
    ax_heat = fig.add_subplot(gs[0, 3])
    
    # Prepare heatmap data
    heat_data = []
    indicators_ordered = []
    indicator_colors = []
    
    for indicator in INDICATORS:
        indicator_weights = []
        for source in ['govern', 'news', 'research', 'zhihu']:
            weight = source_weights[source].get(indicator, 0)
            indicator_weights.append(weight)
        heat_data.append(indicator_weights)
        indicators_ordered.append(INDICATOR_ALIASES[indicator])
        indicator_colors.append(INDICATOR_COLORS[indicator])
    
    heat_matrix = np.array(heat_data)
    
    # Create custom colormap for each row
    fig_temp = plt.figure(figsize=(1,1))
    ax_temp = fig_temp.add_subplot(111)
    
    for i in range(len(indicators_ordered)):
        for j in range(len(sources)):
            value = heat_matrix[i, j]
            # Use indicator-specific color with alpha based on value
            alpha = 0.3 + 0.7 * (value / heat_matrix.max())
            color = mpl.colors.to_rgba(indicator_colors[i], alpha=alpha)
            
            rect = Rectangle((j-0.4, i-0.4), 0.8, 0.8, 
                           facecolor=color, edgecolor='white', linewidth=2)
            ax_heat.add_patch(rect)
            
            # Add text annotations
            text_color = 'white' if value > heat_matrix.max()*0.5 else 'black'
            ax_heat.text(j, i, f'{value:.3f}', ha='center', va='center',
                        color=text_color, fontweight='bold', fontsize=11)
    
    ax_heat.set_xlim(-0.5, len(sources)-0.5)
    ax_heat.set_ylim(-0.5, len(indicators_ordered)-0.5)
    ax_heat.set_xticks(range(len(sources)))
    ax_heat.set_xticklabels(sources, fontweight='bold', fontsize=13, rotation=25, ha='right')
    ax_heat.set_yticks(range(len(indicators_ordered)))
    ax_heat.set_yticklabels(indicators_ordered, fontweight='bold', fontsize=11)
    ax_heat.set_title('B. Individual Indicator Weights by Source', fontweight='bold', fontsize=18)
    ax_heat.invert_yaxis()
    
    # Remove top and right spines
    ax_heat.spines['top'].set_visible(False)
    ax_heat.spines['right'].set_visible(False)
    
    # Style tick labels and color-code y-axis labels by indicator category
    for label in ax_heat.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(13)
    
    # Color-code y-axis labels by indicator colors
    for i, (label, indicator) in enumerate(zip(ax_heat.get_yticklabels(), INDICATORS)):
        label.set_color(INDICATOR_COLORS[indicator])
        label.set_fontweight('bold')
        label.set_fontsize(11)
    
    plt.close(fig_temp)
    
    # ========== Panel C: Environmental Sub-criteria Weight Comparison ==========
    ax_env = fig.add_subplot(gs[1, :4])
    
    # Prepare data for environmental sub-criteria grouped bar chart
    env_indicators = ENV_SET
    env_labels = [INDICATOR_ALIASES[ind] for ind in env_indicators]
    sources = ['govern', 'news', 'research', 'zhihu']
    source_labels = [SOURCE_ALIASES[s] for s in sources]
    
    x = np.arange(len(env_indicators))
    width = 0.2
    
    # Create grouped bar chart for environmental indicators
    for i, source in enumerate(sources):
        env_weights = [env_by_source[source].get(ind, 0) for ind in env_indicators]
        
        bars = ax_env.bar(x + i*width, env_weights, width, 
                         label=SOURCE_ALIASES[source],
                         color=SOURCE_COLORS[SOURCE_ALIASES[source]], 
                         alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar, weight in zip(bars, env_weights):
            height = bar.get_height()
            if height > 0.01:
                ax_env.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                           f'{weight:.2f}', ha='center', va='bottom',
                           fontweight='bold', fontsize=13, rotation=0)
    
    # Customize the environmental comparison chart
    ax_env.set_xlabel('Environmental Sub-Criteria', fontweight='bold', fontsize=16)
    ax_env.set_ylabel('Relative Weight within Environment', fontweight='bold', fontsize=16)
    ax_env.set_title('C. Environmental Sub-Criteria Weights by Source', fontweight='bold', fontsize=18)
    ax_env.set_xticks(x + width * 1.5)
    ax_env.set_xticklabels(env_labels, fontweight='bold', fontsize=13, rotation=25, ha='right')
    
    # Enhanced legend - moved further left
    ax_env.legend(fontsize=13, frameon=True, edgecolor='black', 
                 bbox_to_anchor=(0.85, 1), loc='upper left')
    ax_env.grid(True, alpha=0.3, axis='y')
    
    # Remove top and right spines
    ax_env.spines['top'].set_visible(False)
    ax_env.spines['right'].set_visible(False)
    
    # Style tick labels - ENHANCED
    for label in ax_env.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(16)
    
    # Color-code x-axis labels by environmental indicator colors
    for label in ax_env.get_xticklabels():
        label.set_color("#F18F01")
        label.set_fontweight('bold')
        label.set_fontsize(13)
    
    # Overall title
    plt.suptitle('Multi-Source Criteria Weighting: Comparative Analysis Across Information Sources', 
                 fontsize=25, fontweight='bold', y=0.99)
    
    fig.subplots_adjust(bottom=0.20, hspace=0.55)

    # Save figure
    save_path = os.path.join(FIG_DIR, "figE_source_weight_variation.png")
    fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.splitext(save_path)[0] + '.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Source weight variation analysis saved to: {save_path}")
    print("Figure components:")
    print("  • Panel A: Top-level criteria weights comparison (grouped bars)")
    print("  • Panel B: Individual indicator weights heatmap (color-coded by criteria)")  
    print("  • Panel C: Environmental sub-criteria weights comparison (grouped bars)")
    print("Enhanced features:")
    print("  • Consistent color scheme: Economic(Blue), Technical(Purple), Local Environmental(Orange), Global Environmental(Red)")
    print("  • Criteria-specific coloring in all panels")
    print("  • Environmental indicators unified as Local Environmental category")
    print("  • UPDATED: 'lole' displayed as 'LOLE' throughout, 'Reliability(LOLE)' updated")

if __name__ == "__main__":
    create_source_weight_variation()
