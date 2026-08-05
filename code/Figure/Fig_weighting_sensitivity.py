# -*- coding: utf-8 -*-
"""
TELA Weights with Lambda Sensitivity Analysis - Complex Nature Style
Two-panel design: Main weights + Lambda sensitivity analysis
"""

import os, json, re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Wedge
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import seaborn as sns

# ========== PATHS ==========
BASE_DIR = r"D:\llm\results\result_v5"
OUT_DIR  = os.path.join(BASE_DIR, "eda_out")
FIG_DIR  = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FILES = {
    "govern":   os.path.join(BASE_DIR, "govern_scores_only.jsonl"),
    "news":     os.path.join(BASE_DIR, "news_scores_only.jsonl"),
    "research": os.path.join(BASE_DIR, "research_scores_only.jsonl"),
    "zhihu":    os.path.join(BASE_DIR, "zhihu_scores_only.jsonl"),
}

# ========== HYPERS ==========
LAMBDA_DEFAULT = 0.5
BETA = 1.0
EPS = 1e-12

# Lambda sensitivity range
LAMBDA_RANGE = np.linspace(0, 1, 21)  # 0.0, 0.05, 0.1, ..., 1.0

# ========== INDICATORS ==========
INDICATORS = [
    "total_transition_cost","lole","land_occupation",
    "freshwater_ecotoxicity","human_toxicity","ionising_radiation",
    "freshwater_eutrophication","marine_eutrophication",
    "mineral_resource_depletion","co2_emission",
]

INDICATOR_ALIASES = {
    "total_transition_cost": "Transition Cost",
    "lole": "Reliability", 
    "land_occupation": "Land Use",
    "freshwater_ecotoxicity": "Freshwater Toxicity",
    "human_toxicity": "Human Toxicity",
    "ionising_radiation": "Radiation",
    "freshwater_eutrophication": "Freshwater Eutroph.",
    "marine_eutrophication": "Marine Eutroph.",
    "mineral_resource_depletion": "Mineral Depletion",
    "co2_emission": "CO2 Emissions",
}

ENV_SET = [
    "land_occupation","freshwater_ecotoxicity","human_toxicity",
    "ionising_radiation","freshwater_eutrophication",
    "marine_eutrophication","mineral_resource_depletion",
]

# ========== ENHANCED COLORS ==========
CATEGORY_COLORS = {
    "Cost": "#2E86AB",
    "LOLD": "#A23B72", 
    "Environment": "#F18F01",
    "Global (CO2)": "#C73E1D"
}

# Environmental sub-criteria colors
ENV_COLORS = {
    "land_occupation": "#FF6B6B",
    "freshwater_ecotoxicity": "#4ECDC4", 
    "human_toxicity": "#45B7D1",
    "ionising_radiation": "#96CEB4",
    "freshwater_eutrophication": "#FECA57",
    "marine_eutrophication": "#FF9FF3",
    "mineral_resource_depletion": "#54A0FF"
}

# ========== STYLE SETUP ==========
def set_nature_complex_style():
    """Enhanced Nature style for complex visualization"""
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
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",
        "grid.alpha": 0.3,
    })

# ========== DATA PROCESSING ==========
def parse_score_value(v):
    if v is None: return 1
    if isinstance(v, (int, float)):
        x = int(round(v))
    elif isinstance(v, str):
        m = re.search(r'(\d+)', v)
        x = int(m.group(1)) if m else 1
    else:
        x = 1
    return max(1, min(5, x))

def normalize_scores_dict(d):
    return {k: parse_score_value(d.get(k, 1)) for k in INDICATORS}

def flatten_assessment(ass):
    flat = {}
    if isinstance(ass, dict):
        for sec in ["economic","technical","environmental","global_environmental"]:
            sub = ass.get(sec, {})
            if isinstance(sub, dict):
                for k,v in sub.items():
                    flat[k] = parse_score_value(v)
    return normalize_scores_dict(flat)

def read_jsonl_scores(path, source_tag):
    rows = []
    if not os.path.isfile(path):
        print(f"[WARN] missing file: {path}")
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "scores" in obj and isinstance(obj["scores"], dict):
                sc = normalize_scores_dict(obj["scores"])
            elif "assessment" in obj:
                sc = flatten_assessment(obj["assessment"])
            else:
                continue
            row = {"source": source_tag}
            row.update(sc)
            rows.append(row)
    return pd.DataFrame(rows)

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

def aggregate_across_sources_with_lambda(per_src, lambda_val):
    """Calculate weights for a given lambda value"""
    Q = per_src.groupby("indicator")[["q1","q2","q3"]].mean().reset_index()
    Q["z_k"] = Q["q3"] + lambda_val * Q["q2"]
    zpow = np.power(Q["z_k"].values, BETA)
    denom = zpow.sum()
    if denom <= EPS:
        Q["w_k"] = 1.0 / len(Q)
    else:
        Q["w_k"] = zpow / denom
    return Q

def top_level_weights_with_lambda(Q):
    """Calculate top-level weights from aggregated Q"""
    z_cost = float(Q.loc[Q["indicator"]=="total_transition_cost","z_k"].values[0])
    z_lole = float(Q.loc[Q["indicator"]=="lole","z_k"].values[0])
    z_co2  = float(Q.loc[Q["indicator"]=="co2_emission","z_k"].values[0])
    z_envs = Q.loc[Q["indicator"].isin(ENV_SET),"z_k"].values
    z_env  = float(np.mean(z_envs)) if len(z_envs)>0 else 0.0

    Z = {
        "Cost": z_cost,
        "LOLD": z_lole,
        "Environment": z_env,
        "Global (CO2)": z_co2,
    }
    zpow = {k: (v**BETA) for k,v in Z.items()}
    s = sum(zpow.values())
    if s <= EPS:
        W1 = {k: 1.0/len(zpow) for k in zpow}
    else:
        W1 = {k: zpow[k]/s for k in zpow}
    
    return W1

def environment_weights_with_lambda(Q):
    """Calculate environmental sub-weights"""
    env_df = Q[Q["indicator"].isin(ENV_SET)].copy()
    if env_df.empty:
        return {}
    zpow = np.power(env_df["z_k"].values, BETA)
    denom = zpow.sum()
    if denom <= EPS:
        weights = np.ones(len(env_df)) / len(env_df)
    else:
        weights = zpow / denom
    
    return dict(zip(env_df["indicator"], weights))

# ========== SENSITIVITY ANALYSIS ==========
def run_sensitivity_analysis(per_src):
    """Run sensitivity analysis across lambda range"""
    results = {
        'lambda': [],
        'top_level': {cat: [] for cat in CATEGORY_COLORS.keys()},
        'environmental': {ind: [] for ind in ENV_SET}
    }
    
    for lambda_val in LAMBDA_RANGE:
        Q = aggregate_across_sources_with_lambda(per_src, lambda_val)
        top_weights = top_level_weights_with_lambda(Q)
        env_weights = environment_weights_with_lambda(Q)
        
        results['lambda'].append(lambda_val)
        
        for cat in CATEGORY_COLORS.keys():
            results['top_level'][cat].append(top_weights.get(cat, 0))
        
        for ind in ENV_SET:
            results['environmental'][ind].append(env_weights.get(ind, 0))
    
    return results

# ========== VISUALIZATION ==========
def create_complex_tela_visualization():
    """Create complex two-panel TELA visualization with sensitivity analysis"""
    
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
    
    # Calculate baseline weights (lambda = 0.5)
    Q_baseline = aggregate_across_sources_with_lambda(per_src, LAMBDA_DEFAULT)
    top_weights_baseline = top_level_weights_with_lambda(Q_baseline)
    env_weights_baseline = environment_weights_with_lambda(Q_baseline)
    
    # Run sensitivity analysis
    sensitivity_results = run_sensitivity_analysis(per_src)
    
    # Setup figure
    set_nature_complex_style()
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1.2, 1.2, 1, 0.6],
                  hspace=0.35, wspace=0.25)
    
    # ========== LEFT PANEL: Main Weights Structure (Two Bar Charts) ==========
    
    # Top-left: Top-level weights bar chart
    ax_top_weights = fig.add_subplot(gs[0, :2])
    
    categories = list(CATEGORY_COLORS.keys())
    weights = [top_weights_baseline[cat] for cat in categories]
    colors = [CATEGORY_COLORS[cat] for cat in categories]
    
    # Create horizontal bar chart for top-level weights
    y_pos = np.arange(len(categories))
    bars_top = ax_top_weights.barh(y_pos, weights, color=colors, alpha=0.8, 
                                  edgecolor='black', linewidth=2, height=0.6)
    
    # Add weight value labels on bars
    for i, (bar, weight) in enumerate(zip(bars_top, weights)):
        width = bar.get_width()
        ax_top_weights.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                           f'{weight:.3f}', ha='left', va='center', 
                           fontweight='bold', fontsize=12)
    
    # Add percentage labels at the end of bars
    for i, (bar, weight) in enumerate(zip(bars_top, weights)):
        width = bar.get_width()
        ax_top_weights.text(width/2, bar.get_y() + bar.get_height()/2, 
                           f'{weight:.1%}', ha='center', va='center', 
                           fontweight='bold', fontsize=11, color='white')
    
    ax_top_weights.set_yticks(y_pos)
    ax_top_weights.set_yticklabels(categories, fontsize=13)
    ax_top_weights.set_xlabel('Weight', fontweight='bold', fontsize=14)
    ax_top_weights.set_title('A. Top-Level Criteria Weights (λ = 0.5)', 
                            fontweight='bold', fontsize=16, pad=15)
    ax_top_weights.grid(True, alpha=0.3, axis='x')
    ax_top_weights.set_xlim(0, max(weights) * 1.2)
    
    # Style labels
    for label in ax_top_weights.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax_top_weights.get_xticklabels():
        label.set_fontweight('bold')
    
    # Bottom-left: Environmental sub-weights bar chart
    ax_env_weights = fig.add_subplot(gs[1, :2])
    
    env_items = sorted(env_weights_baseline.items(), key=lambda x: x[1], reverse=True)
    env_indicators = [item[0] for item in env_items]
    env_weights_vals = [item[1] for item in env_items]
    env_colors_vals = [ENV_COLORS[ind] for ind in env_indicators]
    env_labels_vals = [INDICATOR_ALIASES[ind] for ind in env_indicators]
    
    # Create horizontal bar chart for environmental weights
    y_pos_env = np.arange(len(env_indicators))
    bars_env = ax_env_weights.barh(y_pos_env, env_weights_vals, color=env_colors_vals, 
                                  alpha=0.8, edgecolor='black', linewidth=1.5, height=0.7)
    
    # Add weight value labels on bars - ENHANCED
    for i, (bar, weight) in enumerate(zip(bars_env, env_weights_vals)):
        width = bar.get_width()
        ax_env_weights.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{weight:.3f}', ha='left', va='center', 
                           fontweight='bold', fontsize=13)  # 增大字体
    
    # Add percentage labels inside bars - ENHANCED
    for i, (bar, weight) in enumerate(zip(bars_env, env_weights_vals)):
        width = bar.get_width()
        if width > 0.05:  # Only show percentage if bar is wide enough
            ax_env_weights.text(width/2, bar.get_y() + bar.get_height()/2, 
                               f'{weight:.1%}', ha='center', va='center', 
                               fontweight='bold', fontsize=12, color='white')  # 增大字体
    
    ax_env_weights.set_yticks(y_pos_env)
    ax_env_weights.set_yticklabels(env_labels_vals, fontsize=15, ha='right')  # 增大字体并右对齐
    ax_env_weights.set_xlabel('Relative Weight within Environment', fontweight='bold', fontsize=16)  # 增大字体
    ax_env_weights.set_title('B. Environmental Sub-Criteria Weights (λ = 0.5)', 
                            fontweight='bold', fontsize=18, pad=15)  # 增大字体
    ax_env_weights.grid(True, alpha=0.3, axis='x')
    ax_env_weights.set_xlim(0, max(env_weights_vals) * 1.25)
    
    # Style labels - ENHANCED
    for label in ax_env_weights.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(15)  # 确保Y轴刻度更大
        label.set_horizontalalignment('right')  # 手动设置右对齐
    for label in ax_env_weights.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(13)  # 确保X轴刻度更大
    
    # Add connection annotation between panels - ENHANCED (靠右显示)
    connection_text = f"Environment weight: {top_weights_baseline['Environment']:.1%} × sub-weights"
    ax_env_weights.text(0.98, 0.95, connection_text, transform=ax_env_weights.transAxes,
                       fontsize=14, fontweight='bold', style='italic',  # 增大字体
                       ha='right', va='top',  # 右对齐，顶部对齐
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', 
                                alpha=0.8, edgecolor='black', linewidth=1.5))
    
    # ========== RIGHT PANEL: Lambda Sensitivity Analysis ==========
    
    # Top-right: Top-level sensitivity
    ax_sens_top = fig.add_subplot(gs[0, 2:])
    
    for cat, color in CATEGORY_COLORS.items():
        ax_sens_top.plot(sensitivity_results['lambda'], sensitivity_results['top_level'][cat], 
                        color=color, linewidth=4, label=cat, alpha=0.8, marker='o', markersize=6)  # 增粗线条，增大标记
    
    # Highlight baseline lambda
    ax_sens_top.axvline(x=LAMBDA_DEFAULT, color='red', linestyle='--', linewidth=3, alpha=0.8, 
                       label=f'λ = {LAMBDA_DEFAULT}')  # 增粗基准线
    
    ax_sens_top.set_xlabel('Lambda (λ)', fontweight='bold', fontsize=16)  # 增大字体
    ax_sens_top.set_ylabel('Top-Level Weight', fontweight='bold', fontsize=16)  # 增大字体
    ax_sens_top.set_title('C. Top-Level Sensitivity to λ', fontweight='bold', fontsize=18)  # 增大字体
    ax_sens_top.legend(fontsize=13, frameon=True, edgecolor='black', framealpha=0.9)  # 增大图例字体
    ax_sens_top.grid(True, alpha=0.3)
    
    # Style tick labels - ENHANCED
    for label in ax_sens_top.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(14)  # 增大X轴刻度
    for label in ax_sens_top.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(14)  # 增大Y轴刻度
    
    # 手动设置图例加粗
    legend = ax_sens_top.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontweight('bold')
    
    # Bottom-right: Environmental sensitivity
    ax_sens_env = fig.add_subplot(gs[1, 2:])
    
    for ind, color in ENV_COLORS.items():
        ax_sens_env.plot(sensitivity_results['lambda'], sensitivity_results['environmental'][ind], 
                        color=color, linewidth=3, label=INDICATOR_ALIASES[ind][:12] + "..." if len(INDICATOR_ALIASES[ind]) > 12 else INDICATOR_ALIASES[ind], 
                        alpha=0.8, marker='s', markersize=4)  # 增粗线条，增大标记
    
    # Highlight baseline lambda
    ax_sens_env.axvline(x=LAMBDA_DEFAULT, color='red', linestyle='--', linewidth=3, alpha=0.8)  # 增粗基准线
    
    ax_sens_env.set_xlabel('Lambda (λ)', fontweight='bold', fontsize=16)  # 增大字体
    ax_sens_env.set_ylabel('Environmental Sub-Weight', fontweight='bold', fontsize=16)  # 增大字体
    ax_sens_env.set_title('D. Environmental Sensitivity to λ', fontweight='bold', fontsize=18)  # 增大字体
    ax_sens_env.legend(fontsize=11, frameon=True, ncol=2, edgecolor='black', framealpha=0.9)  # 增大图例字体
    ax_sens_env.grid(True, alpha=0.3)
    
    # Style tick labels - ENHANCED
    for label in ax_sens_env.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(14)  # 增大X轴刻度
    for label in ax_sens_env.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(14)  # 增大Y轴刻度
    
    # 手动设置图例加粗
    legend = ax_sens_env.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontweight('bold')
    
    # Overall title
    #plt.suptitle('TELA Multi-Criteria Weights with Lambda Sensitivity Analysis', 
                # fontsize=25, fontweight='bold', y=0.95)
    
    # Save figure
    save_path = os.path.join(FIG_DIR, "figD_tela_weights_sensitivity.png")
    fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Complex TELA weights with sensitivity analysis saved to: {save_path}")
    print("Figure components:")
    print("  • Left: Hierarchical weight structure (nested pie charts)")
    print("  • Top-right: Top-level category sensitivity to λ")  
    print("  • Bottom-right: Environmental sub-criteria sensitivity to λ")
    print(f"  • Sensitivity range: λ ∈ [0, 1] with {len(LAMBDA_RANGE)} points")

if __name__ == "__main__":
    create_complex_tela_visualization()