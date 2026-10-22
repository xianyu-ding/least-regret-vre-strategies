# -*- coding: utf-8 -*-
"""
Enhanced Multi-Perspective SCCA Framework with ARSI
整合了多视角情景与随机准则关键性分析(SCCA)的决策框架，新增备选方案等级位移指数(ARSI)

阶段一：问题定义与模型构建 → 阶段二：基于SMAA的情景排序分析 → 
阶段三：SCCA结构性关键性诊断 → 阶段四：洞见综合与政策建议
"""
from pipeline_common import (SCORE_DIR, RESULTS_DIR, WEIGHTS_DIR, INDICATOR_PATH, rank_scores, winner_shares, draw_dirichlet)
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ========= PATHS =========
BASE_DIR = RESULTS_DIR
WEIGHTS_DIR = RESULTS_DIR / "weights"
INDICATOR_FILE = "combined_data.csv"

# ========= MC SETTINGS =========
N_MC = 20000
KAPPA_TOP = 10
KAPPA_ENV = 10
SEED = 42
DO_KNOCKOUT = True

# ========= TRACKING SETTINGS =========
SAVE_EVERY = 100
SAVE_WEIGHT_SCENARIOS = True

# ========= PERSPECTIVE DEFINITIONS =========
PERSPECTIVES = {
    'comprehensive': 'overall_indicator_weights.csv',  # 综合视角（原有）
    'government': 'govern_weights.csv',                 # 政府视角
    'news': 'news_weights.csv',                        # 新闻媒体视角
    'research': 'research_weights.csv',                # 科研视角
    'social_media': 'zhihu_weights.csv'                # 社交媒体视角（知乎）
}

# ========= INDICATOR SCHEMA (STRICT) =========
RAW_COLS = [
    "Scenario","LOLE","Cost","CO2_emission","Freshwater ecotoxicity",
    "Freshwater eutrophication","Human toxicity","Ionising radiation",
    "Land occupation","Marine eutrophication","Mineral resource depletion"
]
RENAME_MAP = {
    "Scenario":"alternative",
    "LOLE":"lole",
    "Cost":"total_transition_cost",
    "CO2_emission":"co2_emission",
    "Freshwater ecotoxicity":"freshwater_ecotoxicity",
    "Freshwater eutrophication":"freshwater_eutrophication",
    "Human toxicity":"human_toxicity",
    "Ionising radiation":"ionising_radiation",
    "Land occupation":"land_occupation",
    "Marine eutrophication":"marine_eutrophication",
    "Mineral resource depletion":"mineral_resource_depletion",
}
INDICATORS = [
    "total_transition_cost","lole","land_occupation","freshwater_ecotoxicity",
    "human_toxicity","ionising_radiation","freshwater_eutrophication",
    "marine_eutrophication","mineral_resource_depletion","co2_emission"
]
ENV_SET = [
    "land_occupation","freshwater_ecotoxicity","human_toxicity",
    "ionising_radiation","freshwater_eutrophication",
    "marine_eutrophication","mineral_resource_depletion"
]
TOP_GROUPS = ["Cost","LOLE","Environment","CO2"]

# ========= WEIGHT FILE SCHEMAS =========
# For overall_indicator_weights.csv format
OVR_COL_CAT = "Indicator"
OVR_COL_W = "Overall_Weight"
OVR_TOP_MAP = {
    "Transition Cost":"Cost",
    "Reliability":"LOLE",
    "CO2 Emissions":"CO2",
}
OVR_ENV_MAP = {
    "Freshwater Toxicity": "freshwater_ecotoxicity",
    "Freshwater Eutroph.": "freshwater_eutrophication",
    "Human Toxicity": "human_toxicity",
    "Radiation": "ionising_radiation",
    "Land Use": "land_occupation",
    "Marine Eutroph.": "marine_eutrophication",
    "Mineral Depletion": "mineral_resource_depletion",
}

# ========= IO & VALIDATION =========
def load_indicators() -> pd.DataFrame:
    fp = INDICATOR_PATH
    if not fp.is_file():
        raise FileNotFoundError(f"Indicator file not found: {fp}")
    df = pd.read_csv(fp, thousands=',')
    miss = [c for c in RAW_COLS if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing columns in {fp.name}: {miss}\nExpected exactly: {RAW_COLS}")
    df = df[RAW_COLS].rename(columns=RENAME_MAP).copy()
    
    # scenario checks
    if df["alternative"].isna().any():
        raise RuntimeError("Scenario/alternative column contains NaN.")
    if df["alternative"].duplicated().any():
        dups = df["alternative"][df["alternative"].duplicated()].unique().tolist()
        raise RuntimeError(f"Scenario names duplicated: {dups}")
    
    # numeric & non-negative
    for k in INDICATORS:
        df[k] = pd.to_numeric(df[k], errors="coerce")
        if not np.isfinite(df[k]).all():
            raise RuntimeError(f"Column '{k}' contains non-numeric/NaN.")
        mn = float(df[k].min())
        if mn < 0:
            raise ValueError(f"Negative values in {k}; choose and document a normalization policy explicitly.")
    return df

def load_perspective_weights(perspective: str) -> tuple[pd.Series, pd.Series]:
    fp = WEIGHTS_DIR / PERSPECTIVES[perspective]
    df = pd.read_csv(fp)
    required = {OVR_COL_CAT, OVR_COL_W}
    if not required.issubset(df.columns):
        raise ValueError(f'{fp}: required columns {required}')
    expected = set(OVR_TOP_MAP) | set(OVR_ENV_MAP)
    if df[OVR_COL_CAT].duplicated().any() or set(df[OVR_COL_CAT]) != expected:
        raise ValueError(f'{fp}: require each of the ten canonical indicator aliases exactly once')
    values = pd.to_numeric(df[OVR_COL_W], errors='coerce')
    if not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1):
        raise ValueError(f'{fp}: weights must be finite, nonnegative and sum to one')
    weights = pd.Series(values.to_numpy(), index=df[OVR_COL_CAT])
    top = pd.Series(0.0, index=TOP_GROUPS)
    for alias, key in OVR_TOP_MAP.items():
        top[key] = weights[alias]
    env = pd.Series({key: weights[alias] for alias, key in OVR_ENV_MAP.items()}).reindex(ENV_SET)
    top['Environment'] = env.sum()
    # An inactive environmental group contributes zero regardless of its internal allocation.
    env = env / env.sum() if env.sum() > 0 else pd.Series(1 / len(ENV_SET), index=ENV_SET)
    return top, env

# ========= CORE FUNCTIONS =========
def l2_normalize_columns(df: pd.DataFrame):
    u = df.copy()
    norms = {}
    for k in INDICATORS:
        vec = u[k].astype(float).values
        L = float(np.sqrt(np.sum(vec*vec)))
        norms[k] = L
        u[k] = vec / L if L > 0 else 0.0
    return u, norms

def aggregate_scores(u_df: pd.DataFrame, W_top: pd.Series, w_env: pd.Series):
    E = (u_df[ENV_SET] * w_env.values).sum(axis=1).values
    U = (W_top["Cost"] * u_df["total_transition_cost"].values
         + W_top["LOLE"] * u_df["lole"].values
         + W_top["Environment"] * E
         + W_top["CO2"] * u_df["co2_emission"].values)
    return E, U

def baseline_ranking(u_df: pd.DataFrame, W_top, w_env, perspective: str, outdir: Path):
    """Calculate baseline ranking for a specific perspective"""
    E, U = aggregate_scores(u_df, W_top, w_env)
    out = pd.DataFrame({
        "alternative": u_df["alternative"].values,
        "U_total": U,
        "E_environment": E
    })
    out["rank"] = rank_scores(out["U_total"])
    out = out.sort_values(["rank", "U_total"], ascending=[True, True])
    out.to_csv(outdir / f"baseline_ranking_{perspective}.csv", index=False)
    return out

def calculate_spearman_correlation(ranks1: np.ndarray, ranks2: np.ndarray) -> float:
    if np.array_equal(ranks1, ranks2):
        return 1.0
    corr, _ = spearmanr(ranks1, ranks2)
    # Complete ties in one ranking have no ordering information.
    return float(corr) if np.isfinite(corr) else 0.0

def scc_with_scca_arsi_analysis(u_df: pd.DataFrame, W_top_centre: pd.Series, w_env_centre: pd.Series, 
                               perspective: str, outdir: Path):
    """
    Enhanced SCC with Stochastic Criteria Criticality Analysis (SCCA) and 
    Alternative Rank Shift Index (ARSI) Analysis
    """
    if N_MC <= 0 or SAVE_EVERY <= 0:
        raise ValueError("n_mc and save_every must be positive")
    rng = np.random.default_rng(SEED)
    alts = u_df["alternative"].values
    M = len(alts)
    X_cost = u_df["total_transition_cost"].values
    X_lole = u_df["lole"].values
    X_co2  = u_df["co2_emission"].values
    X_env  = u_df[ENV_SET].values

    # Initialize tracking arrays
    wins = np.zeros(M, dtype=np.float64)
    ranks_sum = np.zeros(M, dtype=np.float64)
    cumulative_wins = np.zeros(M, dtype=np.float64)
    
    # SCCA: Track structural criticality
    criteria_names = ["Cost", "LOLE", "Environment", "CO2"] + ENV_SET
    n_criteria = len(criteria_names)
    structural_disturbances = np.zeros((N_MC, n_criteria))  # D(t,j) for each iteration and criterion
    
    # ARSI: Track rank shift for each alternative-criterion pair
    rank_shifts = np.zeros((N_MC, M, n_criteria))  # [iteration, alternative, criterion]
    
    # Data tracking
    evolution_data = []
    weight_scenarios = []
    
    # Normalize weights
    if not np.isclose(W_top_centre.sum(), 1.0): 
        W_top_centre = W_top_centre / W_top_centre.sum()
    if not np.isclose(w_env_centre.sum(), 1.0): 
        w_env_centre = w_env_centre / w_env_centre.sum()

    print(f"开始 {perspective} 视角的SCC+SCCA+ARSI分析，共{N_MC}次迭代...")
    
    for iter_num in range(N_MC):
        # Generate random weights
        W_top = draw_dirichlet(rng, W_top_centre.values, KAPPA_TOP)
        w_env = draw_dirichlet(rng, w_env_centre.values, KAPPA_ENV)
        
        # Calculate baseline scores and rankings
        E = X_env.dot(w_env)
        U = W_top[0]*X_cost + W_top[1]*X_lole + W_top[2]*E + W_top[3]*X_co2
        baseline_ranks = rank_scores(U)  # Convert to ranks 1,2,3...
        
        # Find winner
        shares = winner_shares(U)
        wins += shares
        cumulative_wins += shares
        ranks_sum += baseline_ranks

        # === SCCA + ARSI: Criterion Removal Analysis ===
        # Test removal of top-level criteria
        for crit_idx, crit_name in enumerate(["Cost", "LOLE", "Environment", "CO2"]):
            W_temp = W_top.copy()
            if crit_name == "Cost":
                W_temp[0] = 0
            elif crit_name == "LOLE":
                W_temp[1] = 0
            elif crit_name == "Environment":
                W_temp[2] = 0
            elif crit_name == "CO2":
                W_temp[3] = 0
            
            # Renormalize
            if W_temp.sum() > 0:
                W_temp = W_temp / W_temp.sum()
                U_perturbed = W_temp[0]*X_cost + W_temp[1]*X_lole + W_temp[2]*E + W_temp[3]*X_co2
                perturbed_ranks = rank_scores(U_perturbed)
                
                # Calculate Spearman correlation and disturbance (SCCA)
                rho = calculate_spearman_correlation(baseline_ranks, perturbed_ranks)
                structural_disturbances[iter_num, crit_idx] = 1 - rho
                
                # Calculate rank shifts for each alternative (ARSI)
                for alt_idx in range(M):
                    rank_shift = perturbed_ranks[alt_idx] - baseline_ranks[alt_idx]
                    rank_shifts[iter_num, alt_idx, crit_idx] = rank_shift
            else:
                structural_disturbances[iter_num, crit_idx] = 0
                rank_shifts[iter_num, :, crit_idx] = 0
        
        # Test removal of environmental sub-criteria
        for env_idx, env_crit in enumerate(ENV_SET):
            w_temp = w_env.copy()
            env_position = ENV_SET.index(env_crit)
            w_temp[env_position] = 0
            
            # Renormalize
            if w_temp.sum() > 0:
                w_temp = w_temp / w_temp.sum()
                E_perturbed = X_env.dot(w_temp)
                U_perturbed = W_top[0]*X_cost + W_top[1]*X_lole + W_top[2]*E_perturbed + W_top[3]*X_co2
                perturbed_ranks = rank_scores(U_perturbed)
                
                # Calculate Spearman correlation and disturbance (SCCA)
                rho = calculate_spearman_correlation(baseline_ranks, perturbed_ranks)
                structural_disturbances[iter_num, 4 + env_idx] = 1 - rho
                
                # Calculate rank shifts for each alternative (ARSI)
                for alt_idx in range(M):
                    rank_shift = perturbed_ranks[alt_idx] - baseline_ranks[alt_idx]
                    rank_shifts[iter_num, alt_idx, 4 + env_idx] = rank_shift
            else:
                structural_disturbances[iter_num, 4 + env_idx] = 0
                rank_shifts[iter_num, :, 4 + env_idx] = 0
        
        # Save evolution data
        if (iter_num + 1) % SAVE_EVERY == 0 or iter_num < 1000:
            current_winrates = cumulative_wins / (iter_num + 1)
            for alt_idx, alt_name in enumerate(alts):
                p = current_winrates[alt_idx]
                n = iter_num + 1
                std_error = np.sqrt(p * (1-p) / n) if n > 0 else 0
                conf_lower = max(0, p - 1.96 * std_error)
                conf_upper = min(1, p + 1.96 * std_error)
                
                evolution_data.append({
                    'iteration': iter_num + 1,
                    'alternative': alt_name,
                    'perspective': perspective,
                    'cumulative_win_rate': current_winrates[alt_idx],
                    'confidence_lower': conf_lower,
                    'confidence_upper': conf_upper,
                    'current_winner': bool(shares[alt_idx] > 0)
                })

        # Save weight scenarios
        if SAVE_WEIGHT_SCENARIOS and (iter_num < 5000 or iter_num % 10 == 0):
            for alt_idx, alt_name in enumerate(alts):
                weight_scenarios.append({
                    'iteration': iter_num,
                    'alternative': alt_name,
                    'perspective': perspective,
                    'total_score': U[alt_idx],
                    'env_score': E[alt_idx],
                    'w_cost': W_top[0],
                    'w_lole': W_top[1], 
                    'w_environment': W_top[2],
                    'w_co2': W_top[3],
                    'rank_this_iter': baseline_ranks[alt_idx]
                })
        
        # Progress report
        if (iter_num + 1) % 5000 == 0:
            print(f"  完成 {iter_num + 1}/{N_MC} 次迭代")

    # Calculate final results
    winrate = wins / N_MC
    mean_rank = ranks_sum / N_MC
    
    # Calculate Structural Criticality Index (SCI)
    sci_values = np.mean(structural_disturbances, axis=0)
    
    # === Calculate Alternative Rank Shift Index (ARSI) ===
    # ARSI = rank after removal minus rank before removal. Positive: criterion supports the alternative.
    arsi_values = np.mean(rank_shifts, axis=0)  # Shape: [M alternatives, n_criteria criteria]
    
    # Save traditional SCC results
    pd.DataFrame({"alternative": alts, "win_rate": winrate}).sort_values("win_rate", ascending=False)\
        .to_csv(outdir / f"scc_winrates_{perspective}.csv", index=False)
    pd.DataFrame({"alternative": alts, "mean_rank": mean_rank}).sort_values("mean_rank", ascending=True)\
        .to_csv(outdir / f"scc_rank_stats_{perspective}.csv", index=False)

    # === Save SCCA results ===
    scca_results = pd.DataFrame({
        'criterion': criteria_names,
        'perspective': perspective,
        'structural_criticality_index': sci_values,
        'mean_disturbance': sci_values,  # Same as SCI
        'rank_by_criticality': pd.Series(sci_values).rank(ascending=False, method='min').astype(int)
    })
    scca_results = scca_results.sort_values('structural_criticality_index', ascending=False)
    scca_results.to_csv(outdir / f"scca_structural_criticality_{perspective}.csv", index=False)
    
    # === Save ARSI results ===
    # Create comprehensive ARSI matrix
    arsi_data = []
    for alt_idx, alt_name in enumerate(alts):
        for crit_idx, crit_name in enumerate(criteria_names):
            arsi_value = arsi_values[alt_idx, crit_idx]
            
            # Interpret ARSI value
            if arsi_value > 0.1:
                interpretation = "Strong Advantage Criterion"
                impact = "Highly Positive"
            elif arsi_value > 0.05:
                interpretation = "Moderate Advantage Criterion"  
                impact = "Moderately Positive"
            elif arsi_value > -0.05:
                interpretation = "Neutral Criterion"
                impact = "Negligible"
            elif arsi_value > -0.1:
                interpretation = "Moderate Disadvantage Criterion"
                impact = "Moderately Negative"
            else:
                interpretation = "Strong Disadvantage Criterion"
                impact = "Highly Negative"
            
            arsi_data.append({
                'alternative': alt_name,
                'criterion': crit_name,
                'perspective': perspective,
                'arsi_value': arsi_value,
                'interpretation': interpretation,
                'impact_level': impact,
                'abs_arsi': abs(arsi_value)
            })
    
    arsi_df = pd.DataFrame(arsi_data)
    arsi_df.to_csv(outdir / f"arsi_detailed_{perspective}.csv", index=False)
    
    # Create pivot table for easy viewing
    arsi_pivot = arsi_df.pivot(index='alternative', columns='criterion', values='arsi_value')
    arsi_pivot.to_csv(outdir / f"arsi_matrix_{perspective}.csv")
    
    # Summary statistics for each alternative
    arsi_summary = arsi_df.groupby('alternative').agg({
        'arsi_value': ['mean', 'std', 'min', 'max'],
        'abs_arsi': ['mean', 'max']
    }).round(4)
    arsi_summary.columns = ['mean_arsi', 'std_arsi', 'min_arsi', 'max_arsi', 'mean_abs_arsi', 'max_abs_arsi']
    arsi_summary['perspective'] = perspective
    arsi_summary.to_csv(outdir / f"arsi_summary_{perspective}.csv")
    
    # Find most critical criterion-alternative relationships
    top_advantages = arsi_df.nlargest(10, 'arsi_value')[['alternative', 'criterion', 'arsi_value', 'interpretation']]
    top_disadvantages = arsi_df.nsmallest(10, 'arsi_value')[['alternative', 'criterion', 'arsi_value', 'interpretation']]
    
    critical_relationships = pd.concat([
        top_advantages.assign(relationship_type='Top Advantages'),
        top_disadvantages.assign(relationship_type='Top Disadvantages')
    ])
    critical_relationships.to_csv(outdir / f"arsi_critical_relationships_{perspective}.csv", index=False)
    
    # Save detailed rank shift data for further analysis
    # Reshape rank_shifts for saving
    rank_shift_detailed = []
    for iter_num in range(0, N_MC, 100):
        for alt_idx, alt_name in enumerate(alts):
            for crit_idx, crit_name in enumerate(criteria_names):
                rank_shift_detailed.append({
                    'iteration': iter_num,
                    'alternative': alt_name,
                    'criterion': crit_name,
                    'perspective': perspective,
                    'rank_shift': rank_shifts[iter_num, alt_idx, crit_idx]
                })
    
    # Sample detailed data (save only every 100th iteration to reduce file size)
    rank_shift_sample = rank_shift_detailed
    pd.DataFrame(rank_shift_sample).to_csv(outdir / f"arsi_rank_shifts_sample_{perspective}.csv", index=False)
    
    # Save evolution and scenario data
    evolution_df = pd.DataFrame(evolution_data)
    evolution_df.to_csv(outdir / f"scc_evolution_{perspective}.csv", index=False)
    
    if weight_scenarios:
        scenarios_df = pd.DataFrame(weight_scenarios)
        scenarios_df.to_csv(outdir / f"scc_weight_scenarios_{perspective}.csv", index=False)
    
    print(f"  {perspective} 视角分析完成:")
    print(f"    - 传统SCC结果: {len(evolution_df)} 条进化记录")
    print(f"    - SCCA结构关键性: {len(scca_results)} 个准则")
    print(f"    - ARSI分析: {len(arsi_df)} 个方案-准则关系")
    print(f"    - 最关键准则: {scca_results.iloc[0]['criterion']} (SCI = {scca_results.iloc[0]['structural_criticality_index']:.4f})")
    
    # Print top ARSI insights
    print(f"    - 最强优势关系: {top_advantages.iloc[0]['alternative']} - {top_advantages.iloc[0]['criterion']} (ARSI = {top_advantages.iloc[0]['arsi_value']:.4f})")
    print(f"    - 最强劣势关系: {top_disadvantages.iloc[0]['alternative']} - {top_disadvantages.iloc[0]['criterion']} (ARSI = {top_disadvantages.iloc[0]['arsi_value']:.4f})")
    
    return evolution_df, scenarios_df if weight_scenarios else None, scca_results, arsi_df

def multi_perspective_analysis(u_df: pd.DataFrame, outdir: Path):
    """
    Run comprehensive multi-perspective analysis with ARSI
    """
    print("\n=== 开始多视角SCCA+ARSI综合分析 ===")
    
    all_evolution_data = []
    all_scenario_data = []
    all_scca_results = []
    all_arsi_results = []
    perspective_summary = {}
    combined_evolution = combined_scenarios = combined_scca = combined_arsi = None
    # Preflight every requested perspective so missing input cannot yield a partial success.
    validated_weights = {name: load_perspective_weights(name) for name in PERSPECTIVES}
    
    for perspective, weight_file in PERSPECTIVES.items():
        print(f"\n--- 分析视角: {perspective} ---")
        
        try:
            # Load perspective-specific weights
            w_top, w_env = validated_weights[perspective]
            
            # Store weights summary
            perspective_summary[perspective] = {
                'top_weights': w_top.to_dict(),
                'env_weights': w_env.to_dict(),
                'weight_file': weight_file
            }
            
            # Baseline ranking
            baseline_ranking(u_df, w_top, w_env, perspective, outdir)
            
            # SCC + SCCA + ARSI analysis
            evolution, scenarios, scca, arsi = scc_with_scca_arsi_analysis(u_df, w_top, w_env, perspective, outdir)
            
            # Collect results
            all_evolution_data.append(evolution)
            if scenarios is not None:
                all_scenario_data.append(scenarios)
            all_scca_results.append(scca)
            all_arsi_results.append(arsi)
            
        except Exception as e:
            raise RuntimeError(f"Analysis failed for {perspective}: {e}") from e
    
    if not all_evolution_data:
        raise ValueError("No perspectives selected or successfully analyzed")

    # Combine all results
    if all_evolution_data:
        combined_evolution = pd.concat(all_evolution_data, ignore_index=True)
        combined_evolution.to_csv(outdir / "scc_evolution_all_perspectives.csv", index=False)
    
    if all_scenario_data:
        combined_scenarios = pd.concat(all_scenario_data, ignore_index=True)
        combined_scenarios.to_csv(outdir / "scc_scenarios_all_perspectives.csv", index=False)
    
    if all_scca_results:
        combined_scca = pd.concat(all_scca_results, ignore_index=True)
        combined_scca.to_csv(outdir / "scca_all_perspectives_summary.csv", index=False)
        
        # Create perspective comparison matrix
        pivot_scca = combined_scca.pivot(index='criterion', columns='perspective', values='structural_criticality_index')
        pivot_scca.to_csv(outdir / "scca_perspective_comparison_matrix.csv")
    
    if all_arsi_results:
        combined_arsi = pd.concat(all_arsi_results, ignore_index=True)
        combined_arsi.to_csv(outdir / "arsi_all_perspectives_detailed.csv", index=False)
        
        # Create ARSI comparison across perspectives
        arsi_perspective_comparison = combined_arsi.groupby(['alternative', 'criterion'])['arsi_value'].apply(
            lambda x: pd.Series({
                'mean_arsi': x.mean(),
                'std_arsi': x.std(),
                'min_arsi': x.min(),
                'max_arsi': x.max(),
                'arsi_stability': 1 - (x.std() / (abs(x.mean()) + 1e-6))  # Stability measure
            })
        ).unstack(level=1)
        arsi_perspective_comparison.to_csv(outdir / "arsi_perspective_stability_analysis.csv")
        
        # Find consensus insights (criteria that are consistently advantageous/disadvantageous across perspectives)
        consensus_analysis = combined_arsi.groupby(['alternative', 'criterion']).agg({
            'arsi_value': ['mean', 'std', 'count'],
            'impact_level': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Mixed'
        }).round(4)
        consensus_analysis.columns = ['mean_arsi', 'std_arsi', 'n_perspectives', 'dominant_impact']
        consensus_analysis = consensus_analysis.reset_index()
        consensus_analysis['consensus_strength'] = np.where(
            consensus_analysis['std_arsi'] < 0.05, 'Strong Consensus',
            np.where(consensus_analysis['std_arsi'] < 0.1, 'Moderate Consensus', 'Low Consensus')
        )
        consensus_analysis.to_csv(outdir / "arsi_consensus_analysis.csv", index=False)
    
    # Save perspective summary
    summary_rows = []
    for perspective, data in perspective_summary.items():
        for group, weight in data['top_weights'].items():
            summary_rows.append({
                'perspective': perspective,
                'level': 'top',
                'criterion': group,
                'weight': weight,
                'weight_file': data['weight_file']
            })
        for indicator, weight in data['env_weights'].items():
            summary_rows.append({
                'perspective': perspective,
                'level': 'environment',
                'criterion': indicator,
                'weight': weight,
                'weight_file': data['weight_file']
            })
    
    pd.DataFrame(summary_rows).to_csv(outdir / "perspective_weights_summary.csv", index=False)
    
    return combined_evolution, combined_scenarios, combined_scca, combined_arsi, perspective_summary

# ========= ARSI ANALYSIS FUNCTIONS =========
def generate_arsi_insights_report(outdir: Path):
    """
    Generate comprehensive ARSI insights report
    """
    print("\n=== 生成ARSI洞见报告 ===")
    
    try:
        # Load combined ARSI results
        arsi_data = pd.read_csv(outdir / "arsi_all_perspectives_detailed.csv")
        consensus_data = pd.read_csv(outdir / "arsi_consensus_analysis.csv")
        
        insights_report = []
        
        # 1. Cross-perspective stability analysis
        stable_relationships = consensus_data[consensus_data['consensus_strength'] == 'Strong Consensus']
        insights_report.append("=== 跨视角稳定关系分析 ===")
        insights_report.append(f"发现 {len(stable_relationships)} 个强共识关系（标准差 < 0.05）")
        
        # Top stable advantage relationships
        top_stable_advantages = stable_relationships.nlargest(5, 'mean_arsi')
        insights_report.append("\n最稳定的优势关系:")
        for _, row in top_stable_advantages.iterrows():
            insights_report.append(f"  - {row['alternative']} 在 {row['criterion']} 上: ARSI = {row['mean_arsi']:.4f} (±{row['std_arsi']:.4f})")
        
        # Top stable disadvantage relationships
        top_stable_disadvantages = stable_relationships.nsmallest(5, 'mean_arsi')
        insights_report.append("\n最稳定的劣势关系:")
        for _, row in top_stable_disadvantages.iterrows():
            insights_report.append(f"  - {row['alternative']} 在 {row['criterion']} 上: ARSI = {row['mean_arsi']:.4f} (±{row['std_arsi']:.4f})")
        
        # 2. Alternative-specific insights
        insights_report.append("\n=== 方案特性分析 ===")
        for alt in arsi_data['alternative'].unique():
            alt_data = consensus_data[consensus_data['alternative'] == alt]
            
            # Find strongest advantages and disadvantages
            top_advantage = alt_data.loc[alt_data['mean_arsi'].idxmax()] if len(alt_data) > 0 else None
            top_disadvantage = alt_data.loc[alt_data['mean_arsi'].idxmin()] if len(alt_data) > 0 else None
            
            insights_report.append(f"\n{alt}:")
            if top_advantage is not None:
                insights_report.append(f"  最强优势: {top_advantage['criterion']} (ARSI = {top_advantage['mean_arsi']:.4f})")
            if top_disadvantage is not None:
                insights_report.append(f"  最大劣势: {top_disadvantage['criterion']} (ARSI = {top_disadvantage['mean_arsi']:.4f})")
            
            # Count strong consensus relationships
            strong_relationships = alt_data[alt_data['consensus_strength'] == 'Strong Consensus']
            insights_report.append(f"  强共识关系数: {len(strong_relationships)}")
        
        # 3. Criterion-specific insights
        insights_report.append("\n=== 准则影响力分析 ===")
        criterion_impact = consensus_data.groupby('criterion').agg({
            'mean_arsi': ['mean', 'std', lambda x: (x > 0.05).sum(), lambda x: (x < -0.05).sum()],
            'consensus_strength': lambda x: (x == 'Strong Consensus').sum()
        }).round(4)
        criterion_impact.columns = ['avg_impact', 'impact_variability', 'n_strong_advantages', 'n_strong_disadvantages', 'n_strong_consensus']
        
        for criterion in criterion_impact.index:
            row = criterion_impact.loc[criterion]
            insights_report.append(f"\n{criterion}:")
            insights_report.append(f"  平均影响力: {row['avg_impact']:.4f}")
            insights_report.append(f"  影响力方差: {row['impact_variability']:.4f}")
            insights_report.append(f"  强优势关系: {int(row['n_strong_advantages'])} 个")
            insights_report.append(f"  强劣势关系: {int(row['n_strong_disadvantages'])} 个")
            insights_report.append(f"  强共识关系: {int(row['n_strong_consensus'])} 个")
        
        # 4. Perspective divergence analysis
        insights_report.append("\n=== 视角分歧分析 ===")
        high_divergence = consensus_data[consensus_data['consensus_strength'] == 'Low Consensus'].nlargest(10, 'std_arsi')
        insights_report.append(f"发现 {len(consensus_data[consensus_data['consensus_strength'] == 'Low Consensus'])} 个高分歧关系")
        insights_report.append("\n分歧最大的关系:")
        for _, row in high_divergence.iterrows():
            insights_report.append(f"  - {row['alternative']} - {row['criterion']}: 标准差 = {row['std_arsi']:.4f}")
        
        # Save insights report
        with open(outdir / "arsi_insights_report.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(insights_report))
        
        print("ARSI洞见报告已生成")
        return insights_report
        
    except Exception as e:
        print(f"生成ARSI报告时出错: {e}")
        return []

def create_arsi_visualizations_data(outdir: Path):
    """
    Create data files optimized for ARSI visualizations
    """
    print("准备ARSI可视化数据...")
    
    try:
        # Load data
        arsi_data = pd.read_csv(outdir / "arsi_all_perspectives_detailed.csv")
        consensus_data = pd.read_csv(outdir / "arsi_consensus_analysis.csv")
        
        # 1. Heatmap data (alternatives vs criteria, averaged across perspectives)
        heatmap_data = consensus_data.pivot(index='alternative', columns='criterion', values='mean_arsi')
        heatmap_data.to_csv(outdir / "arsi_heatmap_data.csv")
        
        # 2. Network data (for criterion-alternative relationship networks)
        network_edges = []
        for _, row in consensus_data.iterrows():
            if abs(row['mean_arsi']) > 0.05:  # Only significant relationships
                network_edges.append({
                    'source': row['alternative'],
                    'target': row['criterion'],
                    'weight': row['mean_arsi'],
                    'edge_type': 'advantage' if row['mean_arsi'] > 0 else 'disadvantage',
                    'strength': abs(row['mean_arsi']),
                    'consensus': row['consensus_strength']
                })
        pd.DataFrame(network_edges).to_csv(outdir / "arsi_network_data.csv", index=False)
        
        # 3. Perspective comparison data
        perspective_comparison = arsi_data.groupby(['perspective', 'criterion'])['abs_arsi'].mean().unstack(fill_value=0)
        perspective_comparison.to_csv(outdir / "arsi_perspective_comparison.csv")
        
        # 4. Time series style data (showing ARSI evolution patterns)
        arsi_summary_by_perspective = arsi_data.groupby(['alternative', 'perspective']).agg({
            'arsi_value': ['mean', 'std'],
            'abs_arsi': ['mean', 'max']
        }).round(4)
        arsi_summary_by_perspective.columns = ['mean_arsi', 'std_arsi', 'mean_abs_arsi', 'max_abs_arsi']
        arsi_summary_by_perspective.to_csv(outdir / "arsi_perspective_summary.csv")
        
        print("ARSI可视化数据已准备完成")
        
    except Exception as e:
        print(f"准备可视化数据时出错: {e}")

# ========= MAIN =========
def main():
    outdir = RESULTS_DIR / "scc"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== 多视角SCCA+ARSI决策框架 ===")
    print("阶段一：问题定义与模型构建")
    print("阶段二：基于SMAA的情景排序分析")
    print("阶段三：SCCA结构性关键性诊断")
    print("阶段四：ARSI备选方案等级位移分析")
    print("阶段五：洞见综合与政策建议")
    
    # 1) Load indicators
    print("\n[ 阶段一 ] 加载指标数据...")
    data = load_indicators()
    print(f"加载了 {len(data)} 个VRE情景，{len(INDICATORS)} 个评价指标")
    
    # 2) L2 normalization
    print("执行L2标准化...")
    u_df, norms = l2_normalize_columns(data)
    u_df.to_csv(outdir / "normalized_scores.csv", index=False)

    # 3) Multi-perspective analysis with ARSI
    print("\n[ 阶段二-四 ] 执行多视角SMAA+SCCA+ARSI分析...")
    evolution_all, scenarios_all, scca_all, arsi_all, summary = multi_perspective_analysis(u_df, outdir)

    # 4) Generate ARSI insights
    print("\n[ 阶段五 ] 生成深度洞见报告...")
    insights = generate_arsi_insights_report(outdir)
    
    # 5) Prepare visualization data
    create_arsi_visualizations_data(outdir)

    print("\n=== 分析完成 ===")
    print(f"分析视角: {list(PERSPECTIVES.keys())}")
    print(f"输出目录: {outdir}")
    print(f"VRE情景: {list(u_df['alternative'])}")
    
    # Summary of key findings
    if scca_all is not None:
        print("\n=== 各视角最关键准则 (SCCA) ===")
        for perspective in PERSPECTIVES.keys():
            perspective_data = scca_all[scca_all['perspective'] == perspective]
            if not perspective_data.empty:
                top_criterion = perspective_data.loc[perspective_data['structural_criticality_index'].idxmax()]
                print(f"{perspective:15s}: {top_criterion['criterion']} (SCI = {top_criterion['structural_criticality_index']:.4f})")
    
    if arsi_all is not None:
        print("\n=== 最显著的方案-准则关系 (ARSI) ===")
        # Find most extreme relationships across all perspectives
        consensus_data = pd.read_csv(outdir / "arsi_consensus_analysis.csv")
        top_advantage = consensus_data.loc[consensus_data['mean_arsi'].idxmax()]
        top_disadvantage = consensus_data.loc[consensus_data['mean_arsi'].idxmin()]
        
        print(f"最强优势关系: {top_advantage['alternative']} - {top_advantage['criterion']} (ARSI = {top_advantage['mean_arsi']:.4f})")
        print(f"最大劣势关系: {top_disadvantage['alternative']} - {top_disadvantage['criterion']} (ARSI = {top_disadvantage['mean_arsi']:.4f})")
        
        # Count stable relationships
        stable_count = len(consensus_data[consensus_data['consensus_strength'] == 'Strong Consensus'])
        total_count = len(consensus_data)
        print(f"强共识关系: {stable_count}/{total_count} ({100*stable_count/total_count:.1f}%)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Offline ten-criterion ranking; requires complete indicator data')
    parser.add_argument('--indicators', type=Path, default=INDICATOR_PATH)
    parser.add_argument('--weights-dir', type=Path, default=WEIGHTS_DIR)
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--n-mc', type=int, default=N_MC)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--perspective', choices=list(PERSPECTIVES), action='append')
    args = parser.parse_args()
    INDICATOR_PATH, WEIGHTS_DIR, RESULTS_DIR = args.indicators, args.weights_dir, args.results_dir
    N_MC, SEED = args.n_mc, args.seed
    if args.perspective:
        PERSPECTIVES = {key: PERSPECTIVES[key] for key in args.perspective}
    main()
