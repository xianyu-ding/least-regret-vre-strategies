# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 14:58:45 2025

@author: vanessa
"""

# -*- coding: utf-8 -*-
"""
Multi-Perspective Decision Analysis Visualization - Nature Journal Style
Panel A: Criteria Weight Distribution Scatter Plot (5 Perspectives)
Panel B: Alternative Performance Evolution Curves (Win Rate Convergence)
"""

from pipeline_common import (SCORE_DIR, RESULTS_DIR, WEIGHTS_DIR, INDICATOR_PATH, rank_scores, winner_shares, draw_dirichlet)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from pathlib import Path
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.preprocessing import StandardScaler

# ========== PATHS ==========
BASE_DIR = RESULTS_DIR
SCC_DIR = RESULTS_DIR / "scc"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========== ENHANCED COLOR SCHEME ==========
PERSPECTIVE_COLORS = {
    'comprehensive': '#1f4e79',     # Deep Blue - Professional
    'government': '#8B0000',        # Dark Red - Authority
    'news': '#FF8C00',              # Dark Orange - Media
    'research': '#2F4F2F',          # Dark Green - Academic
    'social_media': '#4B0082'       # Indigo - Social
}

ALTERNATIVE_COLORS = {
    'VRE-75%': '#2E86AB',     # Professional Blue
    'VRE-80%': '#A23B72',     # Deep Purple  
    'VRE-85%': '#F18F01',     # Vibrant Orange
    'VRE-90%': '#C73E1D',     # Strong Red
    'VRE-95%': '#1a9850',     # Forest Green
    'Netzero': '#8B4513'      # Saddle Brown
}

CRITERIA_COLORS = {
    "Cost": "#1f77b4",
    "LOLE": "#ff7f0e", 
    "Environment": "#2ca02c",
    "CO2": "#d62728"
}

# ========== ENHANCED STYLE SETUP ==========
def set_nature_style():
    """Enhanced Nature journal style settings"""
    plt.rcParams.update({
        # Figure settings
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        
        # Font settings
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "font.size": 10,
        "font.weight": "normal",
        
        # Axes settings
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
        
        # Tick settings
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        
        # Legend settings
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        
        # Grid settings
        "grid.alpha": 0.3,
        "grid.linewidth": 0.8,
        "grid.color": "gray",
        
        # Line settings
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1,
    })

# ========== DATA LOADING ==========
def load_multi_perspective_data():
    """Load multi-perspective analysis data"""
    try:
        # Load evolution data
        evolution_all = pd.read_csv(SCC_DIR / "scc_evolution_all_perspectives.csv")
        
        # Load weights summary
        weights_summary = pd.read_csv(SCC_DIR / "perspective_weights_summary.csv")
        
        # Load baseline ranking data
        baseline_files = [
            "baseline_ranking_comprehensive.csv",
            "baseline_ranking_government.csv", 
            "baseline_ranking_news.csv",
            "baseline_ranking_research.csv",
            "baseline_ranking_social_media.csv"
        ]
        
        baseline_data = {}
        for file in baseline_files:
            try:
                perspective = file.replace("baseline_ranking_", "").replace(".csv", "")
                df = pd.read_csv(SCC_DIR / file)
                df['alternative'] = df['alternative'].str.replace('ndc-ltp', 'Netzero', case=False)
                baseline_data[perspective] = df
            except:
                print(f"Warning: Cannot load {file}")
        
        # Replace labels
        evolution_all['alternative'] = evolution_all['alternative'].str.replace('ndc-ltp', 'Netzero', case=False)
        
        return evolution_all, weights_summary, baseline_data
        
    except Exception as e:
        raise FileNotFoundError(f"Cannot load multi-perspective data: {SCC_DIR}: {e}")

# ========== PANEL A: CRITERIA WEIGHT SCATTER PLOT ==========
def plot_criteria_weight_scatter(ax, weights_summary):
    """Panel A: Multi-perspective criteria weight distribution scatter plot"""
    
    # Filter top-level criteria weights
    top_weights = weights_summary[
        (weights_summary['level'] == 'top') & 
        (weights_summary['criterion'].isin(['Cost', 'LOLE', 'Environment', 'CO2']))
    ].copy()
    
    if len(top_weights) == 0:
        ax.text(0.5, 0.5, 'No weight data available', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, fontweight='bold')
        return
    
    # Create pivot table
    weight_matrix = top_weights.pivot(index='perspective', columns='criterion', values='weight')
    weight_matrix = weight_matrix.reindex(columns=['Cost', 'LOLE', 'Environment', 'CO2']).fillna(0)
    
    # Create enhanced scatter plot
    criteria = weight_matrix.columns.tolist()
    perspectives = weight_matrix.index.tolist()
    
    # Set up the plot with better spacing
    x_positions = np.arange(len(criteria))
    width = 0.15  # Width for each perspective
    
    # Create grouped scatter plot
    for i, perspective in enumerate(perspectives):
        x_pos = x_positions + (i - len(perspectives)/2 + 0.5) * width
        weights = weight_matrix.loc[perspective].values
        
        color = PERSPECTIVE_COLORS.get(perspective, '#666666')
        
        # Enhanced scatter plot with larger markers
        scatter = ax.scatter(x_pos, weights, 
                           s=120, c=[color], alpha=0.8,
                           edgecolors='black', linewidth=1.5,
                           marker='o', zorder=3,
                           label=perspective.replace('_', ' ').title())
        
        # Add value labels on points (fixed)
        for j, (x, w) in enumerate(zip(x_pos, weights)):
            if w > 0.01:  # Only show non-zero weights
                ax.annotate(f'{w:.3f}', (x, w), 
                           textcoords="offset points", 
                           xytext=(0, 15), ha='center',
                           fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white', 
                                   edgecolor=color,
                                   alpha=0.9, linewidth=1))
    
    # Add connecting lines for each criterion across perspectives (cleaner approach)
    for j, criterion in enumerate(criteria):
        criterion_weights = []
        criterion_x_pos = []
        
        for i, perspective in enumerate(perspectives):
            x_pos = x_positions[j] + (i - len(perspectives)/2 + 0.5) * width
            weight = weight_matrix.loc[perspective, criterion]
            criterion_weights.append(weight)
            criterion_x_pos.append(x_pos)
        
        # Connect points for same criterion across perspectives
        color = CRITERIA_COLORS.get(criterion, 'gray')
        ax.plot(criterion_x_pos, criterion_weights, 
               color=color, alpha=0.5, linewidth=2, zorder=2,
               linestyle='-', marker='none')
    
    # Statistical analysis overlay
    # Add mean and confidence intervals
    for j, criterion in enumerate(criteria):
        criterion_weights = weight_matrix[criterion].values
        mean_weight = np.mean(criterion_weights)
        std_weight = np.std(criterion_weights)
        
        # Add horizontal line for mean
        ax.axhline(mean_weight, xmin=(j-0.4)/len(criteria), xmax=(j+0.4)/len(criteria),
                  color=CRITERIA_COLORS.get(criterion, 'gray'), 
                  linewidth=2, alpha=0.6, linestyle='--')
        
        # Add confidence interval
        ci_lower = mean_weight - 1.96 * std_weight / np.sqrt(len(criterion_weights))
        ci_upper = mean_weight + 1.96 * std_weight / np.sqrt(len(criterion_weights))
        
        ax.fill_between([j-0.4, j+0.4], ci_lower, ci_upper, 
                       color=CRITERIA_COLORS.get(criterion, 'gray'),
                       alpha=0.2, zorder=1)
    
    # Enhance axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(criteria, fontweight='bold', fontsize=11)
    ax.set_ylabel('Criterion Weight', fontweight='bold', fontsize=11)
    ax.set_title('A. Multi-Perspective Criteria Weight Distribution', 
                fontweight='bold', fontsize=12, pad=15)
    
    # Add grid and styling
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_ylim(0, max(weight_matrix.max()) * 1.15)
    
    # Enhanced legend
    legend1 = ax.legend(title='Perspectives', loc='upper left', 
                       bbox_to_anchor=(0.02, 0.98), fontsize=9, 
                       title_fontsize=10, framealpha=0.95)
    
    # Add criterion color legend
    criterion_patches = [mpatches.Patch(color=CRITERIA_COLORS.get(crit, 'gray'), 
                                       label=f'{crit} (μ={weight_matrix[crit].mean():.3f})') 
                        for crit in criteria]
    
    legend2 = ax.legend(handles=criterion_patches, title='Criteria (Mean Weight)', 
                       loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                       fontsize=9, title_fontsize=10, framealpha=0.95)
    
    ax.add_artist(legend1)  # Keep both legends
    
    # Add statistical summary text
    total_variance = weight_matrix.var().sum()
    max_variance_criterion = weight_matrix.var().idxmax()
    
    stats_text = f'Total Variance: {total_variance:.4f}\nMax Variance: {max_variance_criterion}'
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
           fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='lightgray', alpha=0.8),
           verticalalignment='bottom')

# ========== PANEL B: ENHANCED WIN RATE EVOLUTION ==========
def plot_enhanced_win_rate_evolution(ax, evolution_all):
    """Panel B: Enhanced alternative performance evolution with statistical analysis"""
    
    # Get all alternatives and ensure we show all 6 scenarios
    all_alternatives = evolution_all['alternative'].unique()
    
    # If we have comprehensive perspective, use it to rank alternatives, otherwise use first available
    comprehensive_data = evolution_all[evolution_all['perspective'] == 'comprehensive']
    if len(comprehensive_data) > 0:
        comprehensive_final = comprehensive_data[
            comprehensive_data['iteration'] == comprehensive_data['iteration'].max()
        ]
        # Get all alternatives, sorted by performance
        ranked_alternatives = comprehensive_final.nlargest(len(all_alternatives), 'cumulative_win_rate')['alternative'].tolist()
        # Add any missing alternatives
        missing_alts = [alt for alt in all_alternatives if alt not in ranked_alternatives]
        top_alternatives = ranked_alternatives + missing_alts
    else:
        # Use all available alternatives
        top_alternatives = list(all_alternatives)
    
    # Limit to 6 alternatives as mentioned
    top_alternatives = top_alternatives[:6]
    
    # Enhanced plotting with statistical smoothing and smart label positioning
    label_positions = {}  # Track label positions to avoid overlap
    
    for i, alt in enumerate(top_alternatives):
        alt_data = evolution_all[evolution_all['alternative'] == alt].copy()
        
        if len(alt_data) == 0:
            continue
        
        # Group by perspective and plot evolution curves
        perspectives = alt_data['perspective'].unique()
        color = ALTERNATIVE_COLORS.get(alt, '#666666')
        
        # Plot individual perspective curves
        alpha_values = [0.8, 0.6, 0.5, 0.4, 0.3]  # Decreasing alpha for different perspectives
        
        for j, perspective in enumerate(perspectives[:5]):  # Limit to 5 perspectives
            persp_data = alt_data[alt_data['perspective'] == perspective].sort_values('iteration')
            
            if len(persp_data) < 5:  # Skip if insufficient data (reduced threshold)
                continue
            
            # Smooth the curve using rolling average
            window_size = max(5, len(persp_data) // 20)
            smoothed_winrate = persp_data['cumulative_win_rate'].rolling(window=window_size, center=True).mean()
            
            # Plot main curve
            line_alpha = alpha_values[j] if j < len(alpha_values) else 0.3
            linewidth = 2.5 if perspective == 'comprehensive' else 1.5
            linestyle = '-' if perspective == 'comprehensive' else '--'
            
            ax.plot(persp_data['iteration'], smoothed_winrate, 
                   color=color, linewidth=linewidth, alpha=line_alpha,
                   linestyle=linestyle, zorder=3)
            
            # Add confidence bands for the comprehensive perspective
            if perspective == 'comprehensive':
                # Calculate rolling standard deviation
                rolling_std = persp_data['cumulative_win_rate'].rolling(window=window_size, center=True).std()
                upper_bound = smoothed_winrate + rolling_std
                lower_bound = smoothed_winrate - rolling_std
                
                ax.fill_between(persp_data['iteration'], lower_bound, upper_bound,
                              color=color, alpha=0.15, zorder=1)
        
        # Add final point marker and smart label positioning
        final_comprehensive = alt_data[
            (alt_data['perspective'] == 'comprehensive') & 
            (alt_data['iteration'] == alt_data['iteration'].max())
        ]
        
        if len(final_comprehensive) > 0:
            final_winrate = final_comprehensive['cumulative_win_rate'].iloc[0]
            final_iteration = final_comprehensive['iteration'].iloc[0]
            
            ax.scatter([final_iteration], [final_winrate], 
                      s=100, c=[color], edgecolors='black', 
                      linewidth=2, zorder=5, marker='s')
            
            # Smart label positioning to avoid overlap
            base_x_offset = 15
            base_y_offset = 15
            
            # Calculate offset to avoid overlap with existing labels
            x_offset = base_x_offset
            y_offset = base_y_offset
            
            # Check for nearby labels and adjust position
            for pos_key, (existing_x, existing_y) in label_positions.items():
                if (abs(final_iteration - existing_x) < final_iteration * 0.1 and 
                    abs(final_winrate - existing_y) < 0.15):
                    # Adjust offset based on alternative index
                    y_offset = base_y_offset + (i % 3 - 1) * 25
                    x_offset = base_x_offset + (i % 2) * 20
            
            # Store this label position
            label_positions[alt] = (final_iteration, final_winrate)
            
            # Add label with calculated offset
            ax.annotate(f'{alt}\n({final_winrate:.3f})', 
                       (final_iteration, final_winrate),
                       xytext=(x_offset, y_offset), textcoords='offset points',
                       fontweight='bold', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', edgecolor=color,
                               alpha=0.9, linewidth=1.5),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                     color=color, alpha=0.7))
    
    # Enhanced axes and annotations
    max_iteration = evolution_all['iteration'].max()
    ax.set_xlim(0, max_iteration * 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Monte Carlo Iteration', fontweight='bold', fontsize=11)
    ax.set_ylabel('Cumulative Win Rate', fontweight='bold', fontsize=11)
    ax.set_title('B. Alternative Performance Evolution Analysis', 
                fontweight='bold', fontsize=12, pad=15)
    
    # Add performance zones
    performance_zones = [
        (0.8, 1.0, 'Excellent', '#90EE90', 0.1),
        (0.6, 0.8, 'Good', '#FFE4B5', 0.1),
        (0.4, 0.6, 'Moderate', '#F0E68C', 0.1),
        (0.0, 0.4, 'Poor', '#FFB6C1', 0.1)
    ]
    
    for lower, upper, label, color, alpha in performance_zones:
        ax.axhspan(lower, upper, alpha=alpha, color=color, zorder=0)
        ax.text(max_iteration * 0.02, (lower + upper) / 2, label, 
               fontweight='bold', fontsize=9, 
               verticalalignment='center',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                        edgecolor=color, alpha=0.9))
    
    # Add convergence analysis
    convergence_threshold = max_iteration * 0.7  # Assume convergence after 70% of iterations
    ax.axvline(convergence_threshold, color='red', linestyle=':', 
              linewidth=2, alpha=0.7, label='Convergence Zone')
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Add statistical summary
    n_alternatives = len(top_alternatives)
    n_perspectives = len(evolution_all['perspective'].unique())
    max_iteration = evolution_all['iteration'].max()
    
    stats_text = f'Alternatives: {n_alternatives}\nPerspectives: {n_perspectives}\nIterations: {max_iteration:,}'
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
           fontsize=9, ha='right', va='bottom',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
    
    # Legend for line styles
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=2.5, linestyle='-', 
                   label='Comprehensive Perspective'),
        plt.Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', 
                   label='Other Perspectives'),
        plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.15, 
                     label='Confidence Band')
    ]
    
    ax.legend(handles=legend_elements, loc='center right', 
             bbox_to_anchor=(0.98, 0.7), fontsize=9,
             title='Visualization Elements', title_fontsize=10)

# ========== MAIN VISUALIZATION FUNCTION ==========
def create_enhanced_decision_analysis():
    """Create enhanced Nature-style multi-perspective decision analysis"""
    
    print("Loading multi-perspective decision data...")
    evolution_all, weights_summary, baseline_data = load_multi_perspective_data()
    
    # Set enhanced style
    set_nature_style()
    
    # Create figure with improved layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.3)
    
    # Panel A: Criteria weight scatter plot
    print("Creating criteria weight distribution plot...")
    plot_criteria_weight_scatter(ax1, weights_summary)
    
    # Panel B: Enhanced win rate evolution
    print("Creating performance evolution analysis...")
    plot_enhanced_win_rate_evolution(ax2, evolution_all)
    
    # Add panel labels
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, 
            fontsize=16, fontweight='bold', va='bottom', ha='right')
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, 
            fontsize=16, fontweight='bold', va='bottom', ha='right')
    
    # Final adjustments
    plt.tight_layout()
    
    # Save with high quality
    save_path = FIG_DIR / "enhanced_multi_perspective_analysis.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\nEnhanced multi-perspective analysis saved to: {save_path}")
    print("Figure components:")
    print("  • Panel A: Multi-perspective criteria weight distribution (scatter plot)")
    print("  • Panel B: Alternative performance evolution with statistical analysis")
    
    # Generate summary statistics
    print(f"\n=== Analysis Summary ===")
    
    # Weight analysis
    if len(weights_summary) > 0:
        top_weights = weights_summary[
            (weights_summary['level'] == 'top') & 
            (weights_summary['criterion'].isin(['Cost', 'LOLE', 'Environment', 'CO2']))
        ]
        
        if len(top_weights) > 0:
            weight_matrix = top_weights.pivot(index='perspective', columns='criterion', values='weight')
            
            print("Criteria Weight Statistics:")
            for criterion in weight_matrix.columns:
                weights = weight_matrix[criterion].dropna()
                if len(weights) > 0:
                    print(f"  {criterion}: μ={weights.mean():.3f}, σ={weights.std():.3f}, range=[{weights.min():.3f}, {weights.max():.3f}]")
            
            # Cross-perspective correlation
            corr_matrix = weight_matrix.T.corr()
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            print(f"Average inter-perspective correlation: {avg_correlation:.3f}")
    
    # Performance analysis
    if len(evolution_all) > 0:
        final_data = evolution_all[evolution_all['iteration'] == evolution_all['iteration'].max()]
        
        if len(final_data) > 0:
            print("\nFinal Performance Summary:")
            performance_summary = final_data.groupby('alternative')['cumulative_win_rate'].agg(['mean', 'std', 'count'])
            performance_summary = performance_summary.sort_values('mean', ascending=False)
            
            for alt in performance_summary.index[:5]:  # Top 5 alternatives
                mean_perf = performance_summary.loc[alt, 'mean']
                std_perf = performance_summary.loc[alt, 'std']
                count = performance_summary.loc[alt, 'count']
                print(f"  {alt}: {mean_perf:.3f}±{std_perf:.3f} (n={count})")

if __name__ == "__main__":
    print("=== Enhanced Multi-Perspective Decision Analysis ===")
    create_enhanced_decision_analysis()
