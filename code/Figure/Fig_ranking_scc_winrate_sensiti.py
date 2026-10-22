# -*- coding: utf-8 -*-
"""
Equal-Weights Top-5 Analysis with Real SCC Data
Panel A: Stacked score composition under equal weights (Top-5)
Panel B: Violin plot showing total-score distribution across real weight scenarios
Matches visualization style and data loading approach from real SCC visualization
"""

from pipeline_common import (SCORE_DIR, RESULTS_DIR, WEIGHTS_DIR, INDICATOR_PATH, rank_scores, winner_shares, draw_dirichlet)
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

PERSPECTIVE = os.environ.get("LEAST_REGRET_PERSPECTIVE", "comprehensive")

# ========== PATHS ==========
BASE_DIR = RESULTS_DIR
SCC_DIR = RESULTS_DIR / "scc"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========== DEFINITIONS ==========
ENV_SET = [
    "land_occupation","freshwater_ecotoxicity","human_toxicity",
    "ionising_radiation","freshwater_eutrophication",
    "marine_eutrophication","mineral_resource_depletion"
]

CATEGORY_COLORS = {
    "Cost": "#2E86AB",
    "LOLE": "#A23B72",
    "Environment": "#F18F01",
    "CO2": "#C73E1D",
}

SERIES_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange  
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b'   # Brown
]

PERFORMANCE_COLORS = {
    "excellent": "#1a9850",
    "good": "#91bfdb", 
    "average": "#fee08b",
    "poor": "#d73027"
}

# ========== STYLE SETUP ==========
def set_nature_advanced_style():
    """Advanced Nature style matching code1"""
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

# ========== DATA LOADING (matching code1 approach) ==========
def load_real_scc_data():
    """Load real SCC calculation data - matching code1"""
    try:
        # Load final results
        winrates = pd.read_csv(SCC_DIR / f"scc_winrates_{PERSPECTIVE}.csv")
        rank_stats = pd.read_csv(SCC_DIR / f"scc_rank_stats_{PERSPECTIVE}.csv") 
        baseline = pd.read_csv(SCC_DIR / f"baseline_ranking_{PERSPECTIVE}.csv")
        
        # Load normalized scores
        normalized = pd.read_csv(SCC_DIR / "normalized_scores.csv")
        
        # Load real weight scenarios data
        try:
            scenarios_real = pd.read_csv(SCC_DIR / f"scc_weight_scenarios_{PERSPECTIVE}.csv")
        except:
            scenarios_real = None
            print("Warning: Real weight scenarios data not found, will use fallback")
        
        # Replace labels
        for df in [winrates, rank_stats, baseline, normalized]:
            if df is not None and 'alternative' in df.columns:
                df['alternative'] = df['alternative'].str.replace('ndc-ltp', 'Netzero', case=False)
        
        if scenarios_real is not None and 'alternative' in scenarios_real.columns:
            scenarios_real['alternative'] = scenarios_real['alternative'].str.replace('ndc-ltp', 'Netzero', case=False)
            
        return winrates, rank_stats, baseline, normalized, scenarios_real
    except Exception as e:
        raise FileNotFoundError(f"Cannot load SCC data from: {SCC_DIR}: {e}")

# ========== EQUAL WEIGHTS CALCULATION ==========
def equal_weights():
    """Define equal weights for all categories"""
    w_top = pd.Series({"Cost": 0.25, "LOLE": 0.25, "Environment": 0.25, "CO2": 0.25})
    w_env = pd.Series(1/7, index=ENV_SET)
    return w_top, w_env

def compute_equal_scores(normalized):
    """Compute scores under equal weights"""
    u_df = normalized.copy()
    w_top, w_env = equal_weights()

    # Environment aggregation
    E = (u_df[ENV_SET] * w_env.values).sum(axis=1)

    # Total score (lower is better)
    U = (w_top["Cost"] * u_df["total_transition_cost"] +
         w_top["LOLE"] * u_df["lole"] +
         w_top["Environment"] * E +
         w_top["CO2"] * u_df["co2_emission"])

    out = pd.DataFrame({
        "alternative": u_df["alternative"],
        "total_score": U,
        "cost_score": u_df["total_transition_cost"] * w_top["Cost"],
        "lole_score": u_df["lole"] * w_top["LOLE"],
        "env_score": E * w_top["Environment"],
        "co2_score": u_df["co2_emission"] * w_top["CO2"],
    }).sort_values("total_score").reset_index(drop=True)
    
    out["rank"] = rank_scores(out["total_score"])

    # Save results
    out_dir = BASE_DIR / "sensitivity_equal_weights_fixed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "equal_weights_ranking.csv", index=False)
    
    print(f"\nEqual weights ranking calculated:")
    print(f"  Top 5 alternatives: {out.head(5)['alternative'].tolist()}")
    
    return out

# ========== VISUALIZATION ==========
def create_equal_weights_visualization(equal_df, scenarios_real, baseline, k=5):
    """Create two-panel visualization matching code1 style"""
    
    set_nature_advanced_style()
    topk = equal_df.head(k).copy()
    alts_order = topk["alternative"].tolist()
    
    fig, (ax_comp, ax_dist) = plt.subplots(1, 2, figsize=(16, 8))
    fig.subplots_adjust(wspace=0.3)
    
    # ========== Panel A: Stacked Score Composition ==========
    print("Drawing Panel A: Stacked score composition under equal weights...")
    
    categories = ["Cost", "LOLE", "Environment", "CO2"]
    x = np.arange(len(topk))
    width = 0.65
    
    stacks = [
        topk["cost_score"].values,
        topk["lole_score"].values,
        topk["env_score"].values,
        topk["co2_score"].values,
    ]
    
    bottom = np.zeros(len(topk))
    for cat, vals in zip(categories, stacks):
        ax_comp.bar(x, vals, width, bottom=bottom, label=cat,
                    color=CATEGORY_COLORS[cat], alpha=0.86, 
                    edgecolor="black", linewidth=1.2)
        bottom += vals
    
    ax_comp.set_ylabel("Weighted Score (lower = better)", fontweight='bold', fontsize=15)
    ax_comp.set_title("A. Score Composition (Top-5 under Equal Weights)", 
                      fontweight='bold', fontsize=17)
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(alts_order, rotation=45, ha="right", fontweight='bold')
    ax_comp.legend(loc="upper right", frameon=True, fontsize=12)
    ax_comp.grid(True, axis="y", alpha=0.3)
    
    # Bold axis labels
    for label in ax_comp.get_yticklabels():
        label.set_fontweight('bold')
    
    # ========== Panel B: Violin Plot with Real Scenarios ==========
    print("Drawing Panel B: Real score distribution across weight scenarios...")
    
    if scenarios_real is not None and not scenarios_real.empty and "total_score" in scenarios_real.columns:
        # Filter scenarios for top-5 alternatives
        panel_b_data = scenarios_real[scenarios_real['alternative'].isin(alts_order)].copy()
        
        if len(panel_b_data) > 0:
            print(f"  Using {len(panel_b_data)} real weight scenario data points")
            
            # Collect data for violin plot
            violin_data = []
            positions = []
            colors_violin = []
            performance_levels = []
            
            # Calculate all means first for performance level determination
            all_means = []
            present_alts = []
            
            for i, alt in enumerate(alts_order):
                alt_scores = panel_b_data[panel_b_data['alternative'] == alt]['total_score'].values
                
                if len(alt_scores) == 0:
                    continue
                
                violin_data.append(alt_scores)
                positions.append(i)
                colors_violin.append(SERIES_COLORS[i])
                present_alts.append(alt)
                all_means.append(np.mean(alt_scores))
            
            # Determine performance levels
            if len(all_means) >= 2:
                p25, p50, p75 = np.percentile(all_means, [25, 50, 75])
                for mean_score in all_means:
                    if mean_score < p25:
                        performance_levels.append('excellent')
                    elif mean_score < p50:
                        performance_levels.append('good')
                    elif mean_score < p75:
                        performance_levels.append('average')
                    else:
                        performance_levels.append('poor')
            else:
                performance_levels = ['average'] * len(all_means)
            
            # Create violin plot
            if len(violin_data) > 0:
                parts = ax_dist.violinplot(violin_data, positions=positions, 
                                          widths=0.7, showmeans=True)
                
                # Color the violins
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors_violin[i])
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1.2)
                
                # Enhanced violin elements
                parts['cmeans'].set_color('red')
                parts['cmeans'].set_linewidth(2)
                
                ax_dist.set_xticks(positions)
                ax_dist.set_xticklabels(present_alts, rotation=45, ha='right', fontweight='bold')
                
                # Smart label positioning to avoid overlaps (matching code1)
                y_max = ax_dist.get_ylim()[1]
                y_min = ax_dist.get_ylim()[0]
                
                # Create label position offsets to avoid overlaps
                excellent_count = sum(1 for level in performance_levels if level == 'excellent')
                
                if excellent_count >= 2:
                    # Multiple excellent labels: use alternating heights
                    excellent_positions = [0.92, 0.85, 0.78, 0.92, 0.85, 0.78]
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
                    
                    # Adjust font size based on performance level density
                    font_size = 12 if level == 'excellent' and excellent_count >= 2 else 14
                    
                    ax_dist.text(pos, y_max * y_offset, level.title(), 
                                ha='center', va='top', fontweight='bold', fontsize=font_size,
                                bbox=dict(boxstyle='round,pad=0.3', 
                                         facecolor=PERFORMANCE_COLORS[level], 
                                         alpha=0.8, edgecolor='black', linewidth=1.2))
            
            ax_dist.set_ylabel('Total Score Distribution (Real Data)', fontweight='bold', fontsize=15)
            ax_dist.set_title('B. Real Score Variability Across Weight Scenarios', 
                             fontweight='bold', fontsize=17)
            ax_dist.grid(True, alpha=0.3, axis='y')
            
            # Bold y-axis labels
            for label in ax_dist.get_yticklabels():
                label.set_fontweight('bold')
        else:
            print("  Warning: No matching alternatives found in scenarios data")
            _fallback_baseline(ax_dist, baseline, alts_order)
    else:
        print("  Using fallback baseline visualization...")
        _fallback_baseline(ax_dist, baseline, alts_order)
    
    # Save figure
    save_path = FIG_DIR / "equal_weights_real_data.png"
    fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\n✅ Equal weights visualization saved to: {save_path}")
    print("Figure components:")
    print("  • Panel A: Stacked score composition under equal weights")
    if scenarios_real is not None and not scenarios_real.empty:
        print("  • Panel B: Real score distribution across weight scenarios")
    else:
        print("  • Panel B: Baseline performance comparison (fallback)")
    print(f"  • Visualized alternatives: {len(alts_order)}")

def _fallback_baseline(ax, baseline, alts_order):
    """Fallback visualization using baseline data"""
    ax.set_title("B. Baseline Performance Comparison", fontweight='bold', fontsize=17)
    
    if baseline is None or baseline.empty:
        ax.text(0.5, 0.5, "No real scenarios and no baseline data available.",
                ha="center", va="center", transform=ax.transAxes, 
                fontsize=12, fontweight="bold")
        return
    
    # Support both possible column names
    score_col = "U_total" if "U_total" in baseline.columns else (
        "total_score" if "total_score" in baseline.columns else None)
    
    if score_col is None:
        ax.text(0.5, 0.5, "Baseline file missing total score column.",
                ha="center", va="center", transform=ax.transAxes, 
                fontsize=12, fontweight="bold")
        return
    
    # Filter and reindex baseline data
    base = baseline.set_index("alternative").reindex(alts_order).dropna(subset=[score_col])
    
    if base.empty:
        ax.text(0.5, 0.5, "No matching alternatives in baseline.",
                ha="center", va="center", transform=ax.transAxes, 
                fontsize=12, fontweight="bold")
        return
    
    # Create bar chart
    x = np.arange(len(base))
    bars = ax.bar(x, base[score_col].values,
                  color=[SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(base))],
                  alpha=0.85, edgecolor="black", linewidth=1.2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(base.index.tolist(), rotation=45, ha="right", fontweight="bold")
    ax.set_ylabel("Baseline Total Score (lower = better)", fontweight='bold', fontsize=15)
    
    # Add value labels
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h, f"{h:.3f}", 
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax.grid(True, alpha=0.3, axis="y")
    
    # Bold y-axis labels
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

# ========== SUMMARY ==========
def print_data_summary(equal_df, scenarios_real):
    """Print data summary information"""
    print("\n=== Equal Weights Analysis Summary ===")
    print(f"Total alternatives analyzed: {len(equal_df)}")
    
    if scenarios_real is not None:
        print(f"Real weight scenarios available: {len(scenarios_real)} data points")
        print(f"Unique alternatives in scenarios: {scenarios_real['alternative'].nunique()}")
    else:
        print("Real weight scenarios: Not available")
    
    print(f"\nTop 5 alternatives under equal weights:")
    top5 = equal_df.head(5)
    for i, row in top5.iterrows():
        print(f"  {row['rank']}. {row['alternative']}: {row['total_score']:.4f}")
        print(f"     Cost: {row['cost_score']:.4f}, LOLE: {row['lole_score']:.4f}, " +
              f"Env: {row['env_score']:.4f}, CO2: {row['co2_score']:.4f}")

# ========== MAIN ==========
def main():
    print("=== Equal Weights Analysis with Real SCC Data ===\n")
    
    # Load data
    print("Loading real SCC data...")
    winrates, rank_stats, baseline, normalized, scenarios_real = load_real_scc_data()
    
    # Compute equal weights scores
    print("\nComputing scores under equal weights...")
    equal_df = compute_equal_scores(normalized)
    
    # Print summary
    print_data_summary(equal_df, scenarios_real)
    
    # Create visualization
    print("\nCreating visualization...")
    create_equal_weights_visualization(equal_df, scenarios_real, baseline, k=5)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot a saved ranking perspective")
    parser.add_argument("--perspective", default=PERSPECTIVE, choices=["comprehensive", "government", "news", "research", "social_media"])
    PERSPECTIVE = parser.parse_args().perspective
    main()
