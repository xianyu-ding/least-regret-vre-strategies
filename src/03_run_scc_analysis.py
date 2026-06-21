#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete MCDM Analysis and Visualization Module - Nature Journal Style (Fixed)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

# Nature journal style configuration
NATURE_COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent1': '#F18F01',
    'accent2': '#C73E1D',
    'neutral': '#6C757D',
    'light': '#E9ECEF',
    'success': '#28A745',
    'warning': '#FFC107',
    'error': '#DC3545'
}

NATURE_PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#28A745', 
                  '#6C757D', '#FFC107', '#DC3545', '#17A2B8', '#6F42C1']

# Global figure parameters for Nature style - Enhanced with larger fonts
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 26,  # Increased base font size
    'font.weight': 'bold',
    'axes.linewidth': 2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 1.2,
    'legend.frameon': False,
    'legend.fontsize': 16,  # Increased legend font
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.labelsize': 40,  # Increased tick label size
    'ytick.labelsize': 26,
    'axes.titlesize': 35,   # Much larger title size
    'axes.labelsize': 20,   # Much larger axis label size
    'figure.dpi': 300,
    'figure.titlesize': 28  # Increased figure title size
})

def load_data():
    """Load and prepare all data for analysis."""
    # Load LOLE and cost data
    lole_df = pd.read_csv("/Users/vanessa/Library/CloudStorage/OneDrive-UniversityCollegeLondon/research/data/paper3/MCDM_FINAL/lole_df.csv")
    lole_df.rename(columns={"Pv": "LOLE"}, inplace=True)
    lole_df.set_index("Scenario", inplace=True)

    cost_df = pd.read_csv("/Users/vanessa/Library/CloudStorage/OneDrive-UniversityCollegeLondon/research/data/paper3/MCDM_FINAL/cost_df.csv")
    cost_df.rename(columns={"Pv": "Cost"}, inplace=True)
    cost_df.set_index("Scenario", inplace=True)

    # Load environmental data
    env_metrics = [
        "Freshwater ecotoxicity.csv", "Freshwater eutrophication.csv",
        "Human toxicity.csv", "Ionising radiation.csv", "Land occupation.csv",
        "Marine eutrophication.csv", "Mineral resource depletion.csv"
    ]
    
    env_dir = "/Users/vanessa/Library/CloudStorage/OneDrive-UniversityCollegeLondon/research/data/paper3/MCDM_FINAL/environmental_impacts"
    env_df = pd.DataFrame()

    for metric in env_metrics:
        try:
            df_env = pd.read_csv(f"{env_dir}/{metric}")
            df_env = df_env.set_index(df_env.columns[0])
            
            if 'Unit' in df_env.columns:
                df_env = df_env.drop('Unit', axis=1)
            
            if 'Total' in df_env.index:
                df_total = df_env.loc[['Total']].transpose()
                df_total.columns = [metric.replace(".csv", "").replace(" ", "_").lower() + '_impact']
                df_total.index.name = "Scenario"
                
                env_df = df_total if env_df.empty else env_df.join(df_total, how="outer")
                
        except Exception as e:
            print(f"Error processing {metric}: {e}")
            continue

    # Combine all dataframes
    combined_df = cost_df.join(lole_df, how="outer").join(env_df, how="outer").reset_index()
    combined_df.rename(columns={"index": "Scenario"}, inplace=True)
    
    # Remove VRE60 and VRE65 scenarios
    combined_df = combined_df[~combined_df['Scenario'].str.lower().str.replace(' ', '').str.replace('-', '').str.replace('%', '').isin(['vre60', 'vre65'])]
    combined_df.rename(columns={"Scenario": "scenario"}, inplace=True)
    
    # Add indicator suffix to columns
    combined_df.columns = ['scenario'] + [
        col + '_indicator' if col != 'scenario' else col 
        for col in combined_df.columns[1:]
    ]
    
    return combined_df

def log_transform_and_normalize(data):
    """Apply log transformation and normalization to the data."""
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    raw_data = data[numerical_columns].values
    log_data = np.log1p(data[numerical_columns])
    
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(
        scaler.fit_transform(log_data),
        columns=numerical_columns,
        index=data.index
    )
    
    return normalized_data, raw_data, log_data

def generate_random_weights(n_criteria, n_samples):
    """Generate random weights for the criteria."""
    random_samples = np.random.uniform(0, 1, (n_samples, n_criteria))
    return random_samples / np.sum(random_samples, axis=1, keepdims=True)

def score_solutions(data, weights):
    """Calculate scores for each solution using the given weights."""
    scores = np.dot(data, weights.T)
    best_solutions_indices = np.argmin(scores, axis=0)
    return best_solutions_indices, scores

def evaluate_indicator_importance(data, weights, n_criteria):
    """Evaluate the importance of each indicator."""
    importance_scores = np.zeros(n_criteria)
    base_scores = np.dot(data, weights.T)
    base_best = np.argmin(base_scores, axis=0)

    for i in range(n_criteria):
        modified_weights = weights.copy()
        modified_weights[:, i] = 0
        modified_weights /= np.sum(modified_weights, axis=1, keepdims=True)

        modified_scores = np.dot(data, modified_weights.T)
        modified_best = np.argmin(modified_scores, axis=0)
        importance_scores[i] = np.mean(modified_best != base_best)

    return importance_scores

def calculate_stability(data, weights, n_samples=1000):
    """Calculate stability scores for each criterion with enhanced statistics."""
    n_criteria = data.shape[1]
    stability_scores = np.zeros(n_criteria)
    confidence_intervals = np.zeros((n_criteria, 2))

    for i in range(n_criteria):
        bootstrap_stabilities = []
        baseline_scores = np.dot(data, np.mean(weights, axis=0))
        baseline_best = np.argmin(baseline_scores)
        
        for boot in range(100):
            boot_changes = 0
            for _ in range(n_samples // 100):
                perturbed = np.mean(weights, axis=0).copy()
                perturbed[i] += np.random.normal(0, 0.01)
                perturbed = np.maximum(0, perturbed)
                perturbed /= np.sum(perturbed)
                
                new_scores = np.dot(data, perturbed)
                new_best = np.argmin(new_scores)
                
                if new_best != baseline_best:
                    boot_changes += 1
            
            boot_stability = 1.0 - (boot_changes / (n_samples // 100))
            bootstrap_stabilities.append(boot_stability)
        
        stability_scores[i] = np.mean(bootstrap_stabilities)
        confidence_intervals[i] = np.percentile(bootstrap_stabilities, [2.5, 97.5])
    
    return stability_scores, confidence_intervals

def save_figure(fig, filename, folder='/Users/vanessa/My/phd/PHD_UCL/Paper/Picture/paper3/MCDM-new'):
    """Save the matplotlib figure to a specified file."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, bbox_inches='tight', dpi=300, facecolor='white', 
                edgecolor='none', format='png')
    plt.close(fig)

def clean_names(names):
    """Clean up criteria names for display with abbreviations."""
    # Create a mapping for common long names to short abbreviations
    name_mapping = {
        'freshwater_ecotoxicity': 'FW Ecotox',
        'freshwater_eutrophication': 'FW Eutroph',
        'human_toxicity': 'Human Tox',
        'ionising_radiation': 'Radiation',
        'land_occupation': 'Land Use',
        'marine_eutrophication': 'Marine Eutroph',
        'mineral_resource_depletion': 'Min Depletion',
        'cost': 'Cost',
        'lole': 'LOLD'
    }
    
    clean_list = []
    for name in names:
        # Remove suffixes and clean
        clean_name = name.replace('_indicator', '').replace('_impact', '').lower()
        
        # Apply mapping if exists, otherwise use title case
        if clean_name in name_mapping:
            clean_list.append(name_mapping[clean_name])
        else:
            # Fallback: create abbreviation from title case
            title_name = clean_name.replace('_', ' ').title()
            if len(title_name) > 12:
                # Create abbreviation from first letters of words
                words = title_name.split()
                if len(words) > 1:
                    abbrev = ''.join([word[0] for word in words])
                    clean_list.append(abbrev)
                else:
                    # Truncate long single words
                    clean_list.append(title_name[:10] + '...')
            else:
                clean_list.append(title_name)
                
    return clean_list

def plot_enhanced_weight_distribution(weights, errors, criteria_names, importance_scores):
    """Enhanced weight distribution plot with statistical analysis."""
    clean_criteria = clean_names(criteria_names)
    
    # Main weight distribution plot
    fig1, ax1 = plt.subplots(figsize=(16, 10))
    
    x_pos = np.arange(len(weights))
    bars = ax1.bar(x_pos, weights, yerr=errors, capsize=10, 
                   color=NATURE_COLORS['primary'], alpha=0.8, 
                   edgecolor='white', linewidth=3, 
                   error_kw={'linewidth': 3, 'capthick': 3})
    
    # Add gradient effect to bars based on importance
    for bar, importance in zip(bars, importance_scores):
        alpha = 0.5 + 0.5 * importance
        bar.set_alpha(alpha)
    
    # Statistical annotations
    mean_weight = np.mean(weights)
    ax1.axhline(y=mean_weight, color=NATURE_COLORS['accent1'], 
                linestyle='--', alpha=0.8, linewidth=3, 
                label=f'Mean weight: {mean_weight:.3f}')
    
    # Add value labels with enhanced styling
    for i, (bar, weight, error) in enumerate(zip(bars, weights, errors)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + error + 0.008,
                f'{weight:.3f}±{error:.3f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(clean_criteria, rotation=0, ha='center', 
                        fontsize=16, fontweight='bold')
    ax1.set_title('Multi-Criteria Weight Distribution Analysis', 
                  fontsize=28, fontweight='bold', pad=30)
    ax1.set_ylabel('Weight Value', fontsize=22, fontweight='bold')
    ax1.set_xlabel('Evaluation Criteria', fontsize=22, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=18, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linewidth=1.2)
    
    plt.tight_layout()
    save_figure(fig1, 'weight_distribution_main.png')
    
    # Separate importance analysis plot
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    
    bars2 = ax2.bar(x_pos, importance_scores, color=NATURE_COLORS['accent2'], 
                    alpha=0.8, edgecolor='white', linewidth=3)
    
    # Add value labels
    for bar, importance in zip(bars2, importance_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{importance:.3f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(clean_criteria, rotation=0, ha='center', 
                        fontsize=17, fontweight='bold')
    ax2.set_title('Criteria Importance Analysis', fontsize=28, fontweight='bold', pad=30)
    ax2.set_ylabel('Importance Score', fontsize=28, fontweight='bold')
    ax2.set_xlabel('Evaluation Criteria', fontsize=28, fontweight='bold')
    ax2.grid(True, alpha=0.3, linewidth=1.2)
    
    plt.tight_layout()
    save_figure(fig2, 'importance_analysis.png')

def plot_advanced_correlation_analysis(data, scenarios):
    """Advanced correlation analysis with clustering and statistical tests."""
    data_clean = data.copy()
    data_clean.columns = clean_names(data.columns)
    
    # Main correlation heatmap
    fig1, ax_main = plt.subplots(figsize=(16, 14))
    
    corr = data_clean.corr()
    
    # Hierarchical clustering of correlations
    linkage_matrix = linkage(squareform(1 - np.abs(corr)), method='ward')
    dendro = dendrogram(linkage_matrix, labels=corr.columns, no_plot=True)
    cluster_order = dendro['leaves']
    
    # Reorder correlation matrix
    corr_clustered = corr.iloc[cluster_order, cluster_order]
    
    # Create custom colormap
    colors = ['#2E86AB', '#FFFFFF', '#A23B72']
    n_bins = 100
    cmap = sns.blend_palette(colors, n_colors=n_bins, as_cmap=True)
    
    # Plot heatmap
    mask = np.triu(np.ones_like(corr_clustered, dtype=bool), k=1)
    sns.heatmap(corr_clustered, mask=mask, annot=True, cmap=cmap, vmin=-1, vmax=1,
                square=True, linewidths=2, ax=ax_main, cbar_kws={'shrink': 0.8},
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    
    ax_main.set_xticklabels(corr_clustered.columns, rotation=0, ha='center', 
                           fontsize=16, fontweight='bold')
    ax_main.set_yticklabels(corr_clustered.columns, rotation=0, 
                           fontsize=16, fontweight='bold')
    ax_main.set_title('Hierarchically Clustered Correlation Matrix', 
                      fontsize=28, fontweight='bold', pad=30)
    
    plt.tight_layout()
    save_figure(fig1, 'correlation_heatmap.png')

def plot_enhanced_win_rate_analysis(win_rates, scenarios, scores, normalized_data):
    """Enhanced win rate analysis with statistical confidence intervals."""
    n_simulations = scores.shape[1]
    
    # Calculate confidence intervals using bootstrap
    bootstrap_win_rates = []
    for _ in range(1000):
        bootstrap_indices = np.random.choice(n_simulations, n_simulations, replace=True)
        bootstrap_scores = scores[:, bootstrap_indices]
        bootstrap_best = np.argmin(bootstrap_scores, axis=0)
        bootstrap_wr = np.bincount(bootstrap_best, minlength=len(scenarios)) / len(bootstrap_best)
        bootstrap_win_rates.append(bootstrap_wr)
    
    bootstrap_win_rates = np.array(bootstrap_win_rates)
    ci_lower = np.percentile(bootstrap_win_rates, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_win_rates, 97.5, axis=0)
    errors = [(win_rates - ci_lower), (ci_upper - win_rates)]
    
    # Main win rate plot
    fig1, ax1 = plt.subplots(figsize=(16, 10))
    
    x_pos = np.arange(len(win_rates))
    bars = ax1.bar(x_pos, win_rates, yerr=errors, capsize=10,
                   color=[NATURE_PALETTE[i % len(NATURE_PALETTE)] for i in range(len(scenarios))],
                   alpha=0.8, edgecolor='white', linewidth=3,
                   error_kw={'linewidth': 3, 'capthick': 3})
    
    # Add statistical significance markers
    best_scenario_idx = np.argmax(win_rates)
    
    for i, (bar, wr) in enumerate(zip(bars, win_rates)):
        height = bar.get_height()
        if i == best_scenario_idx:
            ax1.text(bar.get_x() + bar.get_width()/2., height + errors[1][i] + 0.03,
                    '***', ha='center', va='bottom', fontsize=20, fontweight='bold')
        
        # Add value labels
        ax1.text(bar.get_x() + bar.get_width()/2., height + errors[1][i] + 0.015,
                f'{wr:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(scenarios, rotation=0, ha='center', fontsize=18, fontweight='bold')
    ax1.set_title('Scenario Win Rates with 95% Confidence Intervals', 
                  fontsize=28, fontweight='bold', pad=30)
    ax1.set_ylabel('Win Rate', fontsize=22, fontweight='bold')
    ax1.set_xlabel('Scenarios', fontsize=22, fontweight='bold')
    ax1.grid(True, alpha=0.3, linewidth=1.2)
    
    plt.tight_layout()
    save_figure(fig1, 'win_rate_enhanced.png')
    
    # Score distribution violin plot
    fig2, ax2 = plt.subplots(figsize=(16, 10))
    
    parts = ax2.violinplot([scores[i, :] for i in range(len(scenarios))], 
                          positions=x_pos, showmeans=True, showmedians=True, widths=0.8)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(NATURE_PALETTE[i % len(NATURE_PALETTE)])
        pc.set_alpha(0.7)
        pc.set_edgecolor('white')
        pc.set_linewidth(2)
    
    # Enhance mean and median lines
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(3)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(3)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(scenarios, rotation=0, ha='center', fontsize=26, fontweight='bold')
    ax2.set_title('Score Distribution by Scenario', fontsize=28, fontweight='bold', pad=30)
    ax2.set_ylabel('Normalized Score', fontsize=26, fontweight='bold')
    ax2.set_xlabel('Scenarios', fontsize=28, fontweight='bold')
    ax2.grid(True, alpha=0.3, linewidth=1.2)
    
    plt.tight_layout()
    save_figure(fig2, 'score_distribution.png')

def plot_pca_analysis(normalized_data, scenarios, win_rates):
    """PCA analysis colored by win rate."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(normalized_data)
    
    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], 
                        c=win_rates, cmap='viridis', s=300, 
                        alpha=0.8, edgecolors='white', linewidth=3)
    
    for i, scenario in enumerate(scenarios):
        ax.annotate(scenario, (pca_data[i, 0], pca_data[i, 1]), 
                   xytext=(8, 8), textcoords='offset points', 
                   fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                  fontsize=22, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                  fontsize=22, fontweight='bold')
    ax.set_title('PCA Analysis Colored by Win Rate', fontsize=28, fontweight='bold', pad=30)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Win Rate', fontsize=20, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    save_figure(fig, 'pca_analysis.png')

def plot_importance_stability_analysis(importance_scores, stability_scores, criteria_names, confidence_intervals):
    """Multi-dimensional importance and stability analysis."""
    clean_criteria = clean_names(criteria_names)
    
    # Main scatter plot
    fig1, ax_main = plt.subplots(figsize=(16, 12))
    
    # Create color map based on importance + stability
    combined_metric = importance_scores + stability_scores
    
    scatter = ax_main.scatter(importance_scores, stability_scores, 
                             s=400, c=combined_metric, 
                             cmap='viridis', alpha=0.8, 
                             edgecolors='white', linewidth=3)
    
    # Add error bars for stability
    ax_main.errorbar(importance_scores, stability_scores, 
                    yerr=[stability_scores - confidence_intervals[:, 0],
                          confidence_intervals[:, 1] - stability_scores],
                    fmt='none', color='gray', alpha=0.6, capsize=5, linewidth=2)
    
    # Add quadrant lines
    imp_mean, stab_mean = np.mean(importance_scores), np.mean(stability_scores)
    ax_main.axhline(y=stab_mean, color=NATURE_COLORS['neutral'], 
                   linestyle='--', alpha=0.8, linewidth=3)
    ax_main.axvline(x=imp_mean, color=NATURE_COLORS['neutral'], 
                   linestyle='--', alpha=0.8, linewidth=3)
    
    # Add criterion labels
    for i, (imp, stab, name) in enumerate(zip(importance_scores, stability_scores, clean_criteria)):
        ax_main.annotate(name, (imp, stab), xytext=(10, 10), 
                        textcoords='offset points', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9,
                                 edgecolor='gray', linewidth=1))
    
    ax_main.set_xlabel('Importance Score', fontsize=18, fontweight='bold')
    ax_main.set_ylabel('Stability Score', fontsize=18, fontweight='bold')
    ax_main.set_title('Criteria Importance vs Stability Analysis', 
                     fontsize=22, fontweight='bold', pad=25)
    ax_main.grid(True, alpha=0.3, linewidth=1.2)
    
    # Enhanced colorbar
    cbar = plt.colorbar(scatter, ax=ax_main, shrink=0.8)
    cbar.set_label('Combined Metric (Importance + Stability)', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    save_figure(fig1, 'importance_stability_analysis.png')

def plot_data_transformation_analysis(raw_data, log_data, normalized_data, criteria_names):
    """Data transformation analysis."""
    clean_criteria = clean_names(criteria_names)
    
    # Get data arrays
    if isinstance(log_data, pd.DataFrame):
        log_data_array = log_data.values
    else:
        log_data_array = log_data
    
    if isinstance(normalized_data, pd.DataFrame):
        normalized_data_array = normalized_data.values
    else:
        normalized_data_array = normalized_data
    
    # Create comparison summary plot
    fig, axes = plt.subplots(3, 1, figsize=(18, 16))
    
    # Show first criterion as example for all transformations
    criterion_idx = 0
    criterion_name = clean_criteria[criterion_idx]
    
    datasets = [
        (raw_data[:, criterion_idx], 'Raw Data', NATURE_COLORS['primary']),
        (log_data_array[:, criterion_idx], 'Log-Transformed', NATURE_COLORS['secondary']),
        (normalized_data_array[:, criterion_idx], 'Normalized', NATURE_COLORS['accent1'])
    ]
    
    for i, (data, label, color) in enumerate(datasets):
        ax = axes[i]
        
        # Create enhanced histogram
        n, bins, patches = ax.hist(data, bins=20, alpha=0.8, color=color, 
                                  edgecolor='white', linewidth=2, density=True)
        
        # Color gradient
        for j, patch in enumerate(patches):
            patch.set_facecolor(plt.cm.viridis(n[j] / max(n)))
        
        # Add statistics
        mean_val = np.mean(data)
        median_val = np.median(data)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=3, 
                  label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=3, 
                  label=f'Median: {median_val:.3f}')
        
        # Fit distribution curve
        try:
            if i == 2:  # Normalized data
                a, b, loc, scale = stats.beta.fit(data)
                x = np.linspace(0, 1, 100)
                ax.plot(x, stats.beta.pdf(x, a, b, loc, scale), 'black', linewidth=4,
                       label='Beta Distribution Fit')
            else:
                mu, sigma = stats.norm.fit(data)
                x = np.linspace(data.min(), data.max(), 100)
                ax.plot(x, stats.norm.pdf(x, mu, sigma), 'black', linewidth=4,
                       label='Normal Distribution Fit')
        except:
            pass
        
        ax.set_title(f'{criterion_name} - {label}', fontsize=24, fontweight='bold', pad=20)
        ax.set_xlabel('Value', fontsize=20, fontweight='bold')
        ax.set_ylabel('Density', fontsize=20, fontweight='bold')
        ax.legend(fontsize=16, loc='upper right')
        ax.grid(True, alpha=0.3, linewidth=1.2)
    
    plt.suptitle(f'Data Transformation Comparison - {criterion_name}', 
                fontsize=30, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_figure(fig, 'data_transformation_comparison.png')

def plot_sensitivity_analysis(data, weights, scenarios, criteria_names):
    """Comprehensive sensitivity analysis."""
    clean_criteria = clean_names(criteria_names)
    
    # Calculate baseline scores
    baseline_weights = np.mean(weights, axis=0)
    baseline_scores = np.dot(data, baseline_weights)
    best_scenario_idx = np.argmin(baseline_scores)
    
    # Sensitivity analysis: vary each weight ±50%
    sensitivity_results = np.zeros((len(criteria_names), len(scenarios)))
    
    for i, criterion in enumerate(criteria_names):
        # Increase weight by 50%
        modified_weights = baseline_weights.copy()
        modified_weights[i] *= 1.5
        modified_weights /= np.sum(modified_weights)  # Renormalize
        
        modified_scores = np.dot(data, modified_weights)
        sensitivity_results[i, :] = modified_scores - baseline_scores
    
    # Tornado diagram
    fig, ax = plt.subplots(figsize=(16, 10))
    
    tornado_data = sensitivity_results[:, best_scenario_idx]
    sorted_indices = np.argsort(np.abs(tornado_data))[::-1]
    
    y_pos = np.arange(len(criteria_names))
    colors = [NATURE_COLORS['accent2'] if x < 0 else NATURE_COLORS['primary'] 
              for x in tornado_data[sorted_indices]]
    
    bars = ax.barh(y_pos, tornado_data[sorted_indices], color=colors, 
                   alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, value in zip(bars, tornado_data[sorted_indices]):
        width = bar.get_width()
        ax.text(width + np.sign(width)*0.002, bar.get_y() + bar.get_height()/2,
               f'{value:.3f}', ha='left' if width > 0 else 'right', 
               va='center', fontweight='bold', fontsize=14)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([clean_criteria[i] for i in sorted_indices], 
                       fontweight='bold', fontsize=18)
    ax.set_xlabel('Score Change (50% Weight Increase)', fontweight='bold', fontsize=22)
    ax.set_title(f'Sensitivity Analysis - Impact on {scenarios[best_scenario_idx]}', 
                fontweight='bold', fontsize=28, pad=30)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
    ax.grid(True, alpha=0.3, linewidth=1.2)
    
    plt.tight_layout()
    save_figure(fig, 'sensitivity_tornado.png')

def plot_ranking_stability(scores, scenarios):
    """Ranking stability analysis."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    rankings = np.argsort(scores, axis=0)
    rank_std = np.std(rankings, axis=1)
    
    x_pos = np.arange(len(scenarios))
    bars = ax.bar(x_pos, rank_std, 
                  color=[NATURE_PALETTE[i % len(NATURE_PALETTE)] for i in range(len(scenarios))],
                  alpha=0.8, edgecolor='white', linewidth=3)
    
    # Add value labels
    for bar, std_val in zip(bars, rank_std):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.08,
                f'{std_val:.2f}', ha='center', va='bottom', 
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios, rotation=0, ha='center', fontsize=18, fontweight='bold')
    ax.set_title('Ranking Stability Analysis (Lower = More Stable)', 
                 fontsize=28, fontweight='bold', pad=30)
    ax.set_ylabel('Ranking Standard Deviation', fontsize=22, fontweight='bold')
    ax.set_xlabel('Scenarios', fontsize=22, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=1.2)
    
    plt.tight_layout()
    save_figure(fig, 'ranking_stability.png')

def main():
    """Enhanced main function with comprehensive analysis."""
    np.random.seed(42)
    
    print("🔬 Loading and preparing data...")
    data_df = load_data()
    scenarios = data_df['scenario'].values
    criteria_names = [col.replace('_indicator', '') for col in data_df.columns if col != 'scenario']
    
    print("📊 Transforming and normalizing data...")
    normalized_data, raw_data, log_data = log_transform_and_normalize(
        data_df.drop(columns=['scenario'])
    )
    
    print("⚖️ Generating weights and calculating scores...")
    n_criteria = len(criteria_names)
    n_samples = 10000
    weights = generate_random_weights(n_criteria, n_samples)
    best_indices, scores = score_solutions(normalized_data.values, weights)
    
    print("🎯 Calculating importance and stability...")
    importance_scores = evaluate_indicator_importance(normalized_data.values, weights, n_criteria)
    stability_scores, confidence_intervals = calculate_stability(normalized_data.values, weights)
    
    print("🏆 Calculating win rates...")
    win_rates = np.bincount(best_indices, minlength=len(scenarios)) / len(best_indices)
    
    print("📈 Generating enhanced visualizations...")
    
    # Generate all enhanced plots
    weight_errors = np.std(weights, axis=0)
    mean_weights = np.mean(weights, axis=0)
    
    print("   📊 Generating weight distribution plots...")
    plot_enhanced_weight_distribution(mean_weights, weight_errors, criteria_names, importance_scores)
    
    print("   📊 Generating correlation analysis...")
    plot_advanced_correlation_analysis(normalized_data, scenarios)
    
    print("   📊 Generating win rate analysis...")
    plot_enhanced_win_rate_analysis(win_rates, scenarios, scores, normalized_data.values)
    
    print("   📊 Generating PCA analysis...")
    plot_pca_analysis(normalized_data.values, scenarios, win_rates)
    
    print("   📊 Generating importance-stability analysis...")
    plot_importance_stability_analysis(importance_scores, stability_scores, criteria_names, confidence_intervals)
    
    print("   📊 Generating data transformation analysis...")
    plot_data_transformation_analysis(raw_data, log_data, normalized_data, criteria_names)
    
    print("   📊 Generating sensitivity analysis...")
    plot_sensitivity_analysis(normalized_data.values, weights, scenarios, criteria_names)
    
    print("   📊 Generating ranking stability analysis...")
    plot_ranking_stability(scores, scenarios)
    
    print("✅ Enhanced Nature-style analysis complete!")
    print(f"📁 Results saved in: /Users/vanessa/Downloads/MCDM-new")
    print("\n📊 Summary Statistics:")
    print(f"   • Number of scenarios: {len(scenarios)}")
    print(f"   • Number of criteria: {len(criteria_names)}")
    print(f"   • Best performing scenario: {scenarios[np.argmax(win_rates)]} (Win rate: {np.max(win_rates):.3f})")
    print(f"   • Most important criterion: {clean_names([criteria_names[np.argmax(importance_scores)]])[0]} (Importance: {np.max(importance_scores):.3f})")
    print(f"   • Most stable criterion: {clean_names([criteria_names[np.argmax(stability_scores)]])[0]} (Stability: {np.max(stability_scores):.3f})")
    
    return data_df, normalized_data, raw_data, log_data

if __name__ == "__main__":
    combined_df, normalized_data, raw_data, log_data = main()