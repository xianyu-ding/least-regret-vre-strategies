# -*- coding: utf-8 -*-
"""
标准统计学不确定性分析 - 分面图方案
按指标类别分别显示，解决标注混乱问题
"""

import os, re, json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from scipy import stats

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy clustering functions not available. Clustering will be skipped.")

# ====== PATH ======
BASE_DIR = r"D:\llm\results\result_v5"  # <-- 改成你的目录
OUT_DIR = os.path.join(BASE_DIR, "eda_out")
TBL_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# ====== CONFIG ======
INDICATORS = [
    "total_transition_cost","lole","land_occupation",
    "freshwater_ecotoxicity","human_toxicity","ionising_radiation",
    "freshwater_eutrophication","marine_eutrophication",
    "mineral_resource_depletion","co2_emission",
]

INDICATOR_ALIASES = {
    "total_transition_cost": "Transition cost",
    "lole": "Reliability (LOLD)",
    "land_occupation": "Land occupation",
    "freshwater_ecotoxicity": "Freshwater ecotox.",
    "human_toxicity": "Human toxicity",
    "ionising_radiation": "Ionising radiation",
    "freshwater_eutrophication": "Freshwater eutroph.",
    "marine_eutrophication": "Marine eutroph.",
    "mineral_resource_depletion": "Mineral depletion",
    "co2_emission": "Cumulative CO2",
}

SOURCE_ALIASES = {
    "govern": "Government",
    "news": "News",
    "research": "Research",
    "zhihu": "Social media",
}

CATEGORY_COLORS = {
    "Economic": "#2E86AB",
    "Technical": "#A23B72",
    "Local Environmental": "#F18F01",
    "Global Environmental": "#C73E1D"
}

TIER_COLORS = {
    1: "#d73027",  # Red for high concern (tier 1)
    2: "#fee08b",  # Yellow for medium concern (tier 2)
    3: "#1a9850"   # Green for low concern (tier 3)
}

EPS = 1e-9

# ====== ENHANCED STYLE ======
def set_publication_style():
    """Set publication-quality style parameters"""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.style.use('default')
    
    mpl.rcParams.update({
        # Figure settings
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        
        # Font settings
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 16,
        "font.weight": "bold",
        
        # Axes settings
        "axes.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "black",
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.titlepad": 15,
        "axes.labelsize": 20,
        "axes.labelweight": "bold",
        "axes.labelpad": 8,
        
        # Tick settings
        "xtick.labelsize": 16,
        "ytick.labelsize": 18,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        
        # Legend settings
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fontsize": 20,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        
        # Grid settings
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        "axes.grid": True,
        "axes.axisbelow": True,
        
        # Other
        "axes.unicode_minus": False,
        "mathtext.default": "regular",
    })

def get_indicator_category(indicator):
    """Categorize indicators for better organization"""
    categories = {
        "total_transition_cost": "Economic",
        "lole": "Technical",
        "land_occupation": "Local Environmental",
        "freshwater_ecotoxicity": "Local Environmental",
        "human_toxicity": "Local Environmental",
        "ionising_radiation": "Local Environmental",
        "freshwater_eutrophication": "Local Environmental",
        "marine_eutrophication": "Local Environmental",
        "mineral_resource_depletion": "Local Environmental",
        "co2_emission": "Global Environmental"
    }
    return categories.get(indicator, "Other")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_publication_figure(fig, base, dpi=600):
    """Save figures in publication quality - PNG only"""
    fig.savefig(f"{base}.png", dpi=dpi, bbox_inches='tight', facecolor='white')

# ====== IO FUNCTIONS ======
def parse_score_value(v):
    if v is None:
        return 1
    if isinstance(v, (int,float)):
        x = int(round(v))
    elif isinstance(v, str):
        m = re.search(r'(\d+)', v); x = int(m.group(1)) if m else 1
    else:
        x = 1
    return min(5, max(1, x))

def normalize_scores_dict(d):
    return {k: parse_score_value(d.get(k, 1)) for k in INDICATORS}

def flatten_assessment(ass):
    flat = {}
    if isinstance(ass, dict):
        for sec in ["economic","technical","environmental","global_environmental"]:
            s = ass.get(sec, {})
            if isinstance(s, dict):
                for k,v in s.items():
                    flat[k] = parse_score_value(v)
    return normalize_scores_dict(flat)

def read_jsonl_scores(path, source_tag):
    rows = []
    if not os.path.isfile(path):
        print(f"[WARN] missing: {path}")
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
            
            text = obj.get("input_text") or obj.get("question") or obj.get("answer") or ""
            row = {"source": source_tag, "length": len(str(text))}
            row.update(sc); rows.append(row)
    
    return pd.DataFrame(rows)

# ====== CORE FUNCTIONS ======
def collapse_to_tiers(df):
    """Return long table with tier mapping"""
    long = df.melt(id_vars=["source","length"], value_vars=INDICATORS,
                   var_name="indicator", value_name="score")
    # 1->1, 2->2, 3/4/5->3
    long["tier"] = np.select(
        [long["score"]==1, long["score"]==2, long["score"]>=3],
        [1, 2, 3]
    )
    # Add categories
    long["category"] = long["indicator"].apply(get_indicator_category)
    return long

def per_source_tier_props(tri_long):
    """Calculate tier proportions with confidence intervals using Wilson method"""
    counts = tri_long.pivot_table(index=["source","indicator"], columns="tier", 
                                  values="score", aggfunc="count", fill_value=0)
    for t in [1,2,3]:
        if t not in counts.columns:
            counts[t] = 0
    counts = counts[[1,2,3]].sort_index()
    
    total_counts = counts.sum(axis=1)
    props = counts.div(total_counts.replace(0, np.nan), axis=0).reset_index()
    props = props.rename(columns={1:"q1", 2:"q2", 3:"q3"})
    
    def wilson_ci(x, n, alpha=0.05):
        """Wilson score interval - standard method from Wilson (1927)"""
        if n == 0 or np.isnan(x) or np.isnan(n):
            return 0, 0
        z = stats.norm.ppf(1 - alpha/2)
        p = x / n
        denominator = 1 + z**2/n
        centre = (p + z**2/(2*n)) / denominator
        delta = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
        return max(0, centre - delta), min(1, centre + delta)
    
    props = props.merge(total_counts.reset_index(name="n_total"), on=["source", "indicator"])
    
    counts_reset = counts.reset_index()
    for tier_num, tier_col in [(1, "q1"), (2, "q2"), (3, "q3")]:
        if tier_num in counts_reset.columns:
            tier_counts = counts_reset[tier_num].values
        else:
            tier_counts = np.zeros(len(props))
        ci_results = [wilson_ci(x, n) for x, n in zip(tier_counts, props["n_total"])]
        ci_lower, ci_upper = zip(*ci_results)
        props[f"{tier_col}_ci_lower"] = ci_lower
        props[f"{tier_col}_ci_upper"] = ci_upper
    
    mean_tier = tri_long.groupby(["source","indicator"])["tier"].mean().reset_index(name="mean_tier")
    std_tier = tri_long.groupby(["source","indicator"])["tier"].std().reset_index(name="std_tier")
    
    out = props.merge(mean_tier, on=["source","indicator"], how="left")
    out = out.merge(std_tier, on=["source","indicator"], how="left")
    
    out["std_tier"] = out["std_tier"].fillna(0)
    
    out["indicator_label"] = out["indicator"].map(INDICATOR_ALIASES)
    out["source_label"] = out["source"].map(SOURCE_ALIASES)
    out["category"] = out["indicator"].apply(get_indicator_category)
    
    return out

# ====== 分面图方案 ======
def create_faceted_uncertainty_analysis(per_src, tri_long):
    """
    分面图方案：按指标类别分别显示，并为每个 panel 添加小标题
    """
    set_publication_style()
    
    categories = ["Economic", "Technical", "Local Environmental", "Global Environmental"]
    colors_sources = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    source_names = ["govern", "news", "research", "zhihu"]
    
    indicator_shapes = {
        "total_transition_cost": "o",
        "lole": "o",
        "land_occupation": "s",
        "freshwater_ecotoxicity": "D",
        "human_toxicity": "^",
        "ionising_radiation": "v",
        "freshwater_eutrophication": "p",
        "marine_eutrophication": "h",
        "mineral_resource_depletion": "*",
        "co2_emission": "o",
    }
    
    local_env_shapes = {
        "land_occupation": ("s", "Land occupation"),
        "freshwater_ecotoxicity": ("D", "Freshwater ecotox."),
        "human_toxicity": ("^", "Human toxicity"),
        "ionising_radiation": ("v", "Ionising radiation"),
        "freshwater_eutrophication": ("p", "Freshwater eutroph."),
        "marine_eutrophication": ("h", "Marine eutroph."),
        "mineral_resource_depletion": ("*", "Mineral depletion"),
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    global_y_min = per_src['std_tier'].min() - 0.05
    global_y_max = per_src['std_tier'].max() + 0.05
    
    # 固定的 x 轴刻度
    x_ticks = [300, 1000, 3000, 5000]
    from matplotlib.ticker import FixedLocator, FixedFormatter
    
    for i, category in enumerate(categories):
        ax = axes[i]
        category_data = per_src[per_src['category'] == category]
        
        if category_data.empty:
            ax.set_title(category, fontsize=24, fontweight='bold',
                         pad=15, color=CATEGORY_COLORS[category])
            ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=16, fontweight='bold')
            continue
        
        # Debug
        if not category_data['n_total'].empty:
            print(f"Category {category}: n_total range = {category_data['n_total'].min()} to {category_data['n_total'].max()}")
        
        legend_added = set()
        
        for j, source in enumerate(source_names):
            source_data = category_data[category_data['source'] == source]
            if not source_data.empty:
                for _, row in source_data.iterrows():
                    indicator = row['indicator']
                    shape = indicator_shapes.get(indicator, 'o')
                    
                    ax.scatter([row['n_total']], [row['std_tier']], 
                               label=SOURCE_ALIASES[source] if source not in legend_added else "", 
                               alpha=0.8, s=200,
                               color=colors_sources[j], edgecolors='black', 
                               linewidths=2, marker=shape)
                    
                    if source not in legend_added:
                        legend_added.add(source)
                
                # 画趋势线 + 相关系数
                if len(source_data) > 1:
                    x_vals = source_data['n_total'].values
                    y_vals = source_data['std_tier'].values
                    if len(x_vals) >= 2 and np.std(x_vals) > 1e-10:
                        try:
                            z = np.polyfit(x_vals, y_vals, 1)
                            p = np.poly1d(z)
                            x_trend = np.linspace(max(200, x_vals.min()), x_vals.max(), 50)
                            ax.plot(x_trend, p(x_trend), '--', alpha=0.7, 
                                    linewidth=3, color=colors_sources[j])
                            
                            corr = np.corrcoef(x_vals, y_vals)[0, 1]
                            if not np.isnan(corr):
                                ax.text(0.05, 0.95 - j*0.08, 
                                        f'{SOURCE_ALIASES[source][:4]}: r={corr:.3f}',
                                        transform=ax.transAxes, fontsize=18,
                                        color=colors_sources[j], fontweight='bold')
                        except Exception as e:
                            print(f"[TrendLine] {category}-{source}: {e}")
        
        # 轴与网格
        ax.set_xlabel('Sample Size (n)', fontweight='bold', fontsize=22)
        ax.set_ylabel('Uncertainty (Std Dev)', fontweight='bold', fontsize=22)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in x_ticks]))
        
        # x 轴范围
        x_min = min(200, category_data['n_total'].min() * 0.9)
        x_max = max(6000, category_data['n_total'].max() * 1.1)
        ax.set_xlim(x_min, x_max)
        print(f"Category {category}: x-axis limits = {ax.get_xlim()}")
        
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='x', labelsize=16, width=2, which='major')
        ax.tick_params(axis='y', labelsize=18, width=2, which='both')
        ax.set_ylim(global_y_min, global_y_max)
        
        # 🔹 添加每个 panel 的小标题（彩色加粗）
        ax.set_title(category, fontsize=24, fontweight='bold',
                     pad=15, color=CATEGORY_COLORS[category])
        
        # 源图例放在第一个 panel
        if i == 0 and not category_data.empty:
            legend = ax.legend(fontsize=20, loc='upper left', 
                               framealpha=0.9, edgecolor='black',
                               bbox_to_anchor=(0.3, 0.98))
            if legend:
                for text in legend.get_texts():
                    text.set_fontweight('bold')
        
        # Local Environmental 的形状图例
        if category == "Local Environmental" and not category_data.empty:
            from matplotlib.lines import Line2D
            legend_elements, legend_labels = [], []
            for indicator in sorted(category_data['indicator'].unique()):
                if indicator in local_env_shapes:
                    marker, name = local_env_shapes[indicator]
                    legend_elements.append(Line2D([0], [0], marker=marker, color='w', 
                                                  markerfacecolor='#1f77b4', markersize=15,
                                                  markeredgecolor='black', markeredgewidth=2))
                    legend_labels.append(name)
            shape_legend = ax.legend(legend_elements, legend_labels, 
                                     title="Local Environmental Indicators",
                                     loc='center', bbox_to_anchor=(0.53, 0.5),
                                     fontsize=16, title_fontsize=19,
                                     framealpha=0.95, edgecolor='black',
                                     fancybox=False)
            shape_legend.get_title().set_fontweight('bold')
            for text in shape_legend.get_texts():
                text.set_fontweight('bold')
            ax.add_artist(shape_legend)
        
        # 加粗边与刻度
        for spine in ax.spines.values():
            spine.set_linewidth(3)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # 面板内统计角标
        n_indicators = len(category_data['indicator'].unique())
        mean_uncertainty = category_data['std_tier'].mean()
        max_uncertainty = category_data['std_tier'].max()
        stats_text = f'Indicators: {n_indicators}\nMean σ: {mean_uncertainty:.3f}\nMax σ: {max_uncertainty:.3f}'
        if category == "Local Environmental":
            stats_x, stats_y, ha, va = 0.9, 0.98, 'right', 'top'
        else:
            stats_x, stats_y, ha, va = 0.98, 0.02, 'right', 'bottom'
        ax.text(stats_x, stats_y, stats_text, transform=ax.transAxes, 
                fontsize=16, ha=ha, va=va,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                          alpha=0.8, edgecolor='black', linewidth=1),
                fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])  # 给底部注释留点空间
    
    fig.text(0.5, 0.03, 
             'Color = Information Source. Economic/Technical/Global Environmental use circles. Local Environmental uses different shapes (see in-panel legend).',
             ha='center', va='bottom', fontsize=20, fontweight='bold')
    fig.text(0.5, 0.01, 
             'Higher std dev = Greater disagreement. X-axis uses log scale with ticks at 300, 1000, 3000, 5000.',
             ha='center', va='bottom', fontsize=19, fontweight='bold')
    
    save_publication_figure(fig, os.path.join(FIG_DIR, "faceted_uncertainty_analysis"))
    plt.close(fig)


# ====== DATA EXPORT ======
def export_standard_tables(per_src, tri_long):
    """Export tables with standard statistical measures only"""
    standard_table = per_src.copy()
    
    for tier in [1, 2, 3]:
        ci_col = f"ci_width_t{tier}"
        standard_table[ci_col] = (standard_table[f"q{tier}_ci_upper"] - 
                                 standard_table[f"q{tier}_ci_lower"]).round(4)
    
    num_cols = ["q1", "q2", "q3", "mean_tier", "std_tier", 
               "ci_width_t1", "ci_width_t2", "ci_width_t3"]
    for col in num_cols:
        if col in standard_table.columns:
            standard_table[col] = standard_table[col].round(4)
    
    standard_table.to_csv(
        os.path.join(TBL_DIR, "standard_statistical_analysis.csv"), 
        index=False
    )
    
    category_summary = standard_table.groupby(["source_label", "category"]).agg({
        "mean_tier": ["mean", "std", "count"],
        "std_tier": "mean",
        "ci_width_t1": "mean",
        "n_total": "sum"
    }).round(4)
    category_summary.to_csv(
        os.path.join(TBL_DIR, "category_summary_standard.csv")
    )
    
    corr_data = []
    for source in standard_table["source"].unique():
        df_s = standard_table[standard_table["source"] == source]
        if len(df_s) > 3:
            corr = np.corrcoef(df_s["n_total"], df_s["std_tier"])[0, 1]
            if not np.isnan(corr):
                corr_data.append({
                    "source": source,
                    "source_label": SOURCE_ALIASES[source],
                    "correlation_n_vs_uncertainty": corr,
                    "n_indicators": len(df_s)
                })
    
    corr_df = pd.DataFrame(corr_data)
    if not corr_df.empty:
        corr_df.to_csv(
            os.path.join(TBL_DIR, "correlation_analysis.csv"), 
            index=False
        )
    
    print("Standard statistical tables exported:")
    print(f" - {TBL_DIR}/standard_statistical_analysis.csv")
    print(f" - {TBL_DIR}/category041_summary_standard.csv")
    print(f" - {TBL_DIR}/correlation_analysis.csv")

# ====== MAIN EXECUTION ======
def main():
    """Main execution with faceted uncertainty analysis"""
    for p in [OUT_DIR, TBL_DIR, FIG_DIR]:
        ensure_dir(p)
    
    files = {
        "govern": os.path.join(BASE_DIR, "govern_scores_only.jsonl"),
        "news": os.path.join(BASE_DIR, "news_scores_only.jsonl"),
        "research":os.path.join(BASE_DIR, "research_scores_only.jsonl"),
        "zhihu": os.path.join(BASE_DIR, "zhihu_scores_only.jsonl"),
    }
    
    print("Loading data from JSONL files...")
    df_all = pd.concat([
        read_jsonl_scores(files["govern"], "govern"),
        read_jsonl_scores(files["news"], "news"),
        read_jsonl_scores(files["research"], "research"),
        read_jsonl_scores(files["zhihu"], "zhihu"),
    ], ignore_index=True)
    
    if df_all.empty:
        print("[ERROR] No valid rows parsed.")
        return
    
    print(f"Loaded {len(df_all)} total records")
    print(f"Sources: {df_all['source'].value_counts().to_dict()}")
    
    print("Processing tier assignments...")
    tri_long = collapse_to_tiers(df_all)
    
    print("Calculating standard statistical measures...")
    per_src = per_source_tier_props(tri_long)
    
    print("Exporting standard statistical tables...")
    export_standard_tables(per_src, tri_long)
    
    print("Generating faceted uncertainty analysis...")
    create_faceted_uncertainty_analysis(per_src, tri_long)
    
    print("\n" + "="*60)
    print("FACETED UNCERTAINTY ANALYSIS COMPLETE")
    print("="*60)
    print(f"Output directory: {OUT_DIR}")
    print(f"Tables: {TBL_DIR}")
    print(f"Figures: {FIG_DIR}")
    print("\nGenerated enhanced faceted analysis with:")
    print(" • Different shapes for each indicator (especially Local Environmental)")
    print(" • Much larger fonts for maximum readability (22pt axes, 20pt legend)")
    print(" • X-axis uses logarithmic scale with ticks at 300, 1000, 3000, 5000")
    print(" • Color = Source, Shape = Indicator")
    print(" • No cluttered annotations - shapes provide distinction")
    print(" • Enhanced visual hierarchy with bold formatting")
    print(" • Removed titles for cleaner appearance")
    print(" • Source legend moved to upper left in Economic panel")
    print("="*60)

if __name__ == "__main__":
    main()