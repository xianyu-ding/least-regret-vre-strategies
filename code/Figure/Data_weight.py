# -*- coding: utf-8 -*-
"""
TELA Weights Calculator and Exporter (Enhanced Version)

This script calculates three sets of weights based on scores from multiple sources:
1.  Top-Level Weights (Level 1): Weights for the four main categories.
2.  Environmental Sub-Weights (Level 2): Relative weights of environmental indicators within their category.
3.  Overall Indicator Weights: Final, combined weights for all 10 indicators.

Additionally, it now calculates source-specific weights for each perspective:
- Government perspective (govern)
- News perspective (news) 
- Research perspective (research)
- Social media perspective (zhihu)

The results are saved as multiple CSV files to a specified output directory.
"""

import os
import json
import re
import numpy as np
import pandas as pd

# ========== 1. CONFIGURATION ==========

# --- INPUT PATHS (数据源路径) ---
# 注意：这里假设您的原始数据仍然在 result_v5 目录中
BASE_DIR_INPUT = r"D:\llm\results\result_v5"
FILES = {
    "govern":   os.path.join(BASE_DIR_INPUT, "govern_scores_only.jsonl"),
    "news":     os.path.join(BASE_DIR_INPUT, "news_scores_only.jsonl"),
    "research": os.path.join(BASE_DIR_INPUT, "research_scores_only.jsonl"),
    "zhihu":    os.path.join(BASE_DIR_INPUT, "zhihu_scores_only.jsonl"),
}

# --- OUTPUT PATH (您指定的输出路径) ---
OUTPUT_DIR = r"D:\llm\results\result_v5\weights_out"

# --- HYPERPARAMETERS (计算超参数) ---
LAMBDA_DEFAULT = 0.5
BETA = 1.0
EPS = 1e-12

# --- INDICATORS AND CATEGORIES (指标与分类定义) ---
INDICATORS = [
    "total_transition_cost", "lole", "land_occupation",
    "freshwater_ecotoxicity", "human_toxicity", "ionising_radiation",
    "freshwater_eutrophication", "marine_eutrophication",
    "mineral_resource_depletion", "co2_emission",
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

# 环境指标集合，用于分层计算
ENV_SET = [
    "land_occupation", "freshwater_ecotoxicity", "human_toxicity",
    "ionising_radiation", "freshwater_eutrophication",
    "marine_eutrophication", "mineral_resource_depletion",
]

# ========== 2. DATA PROCESSING FUNCTIONS (数据处理函数) ==========
# 这些函数与之前的脚本相同，用于读取和预处理数据

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
        for sec in ["economic", "technical", "environmental", "global_environmental"]:
            sub = ass.get(sec, {})
            if isinstance(sub, dict):
                for k, v in sub.items():
                    flat[k] = parse_score_value(v)
    return normalize_scores_dict(flat)

def read_jsonl_scores(path, source_tag):
    rows = []
    if not os.path.isfile(path):
        print(f"[WARN] Missing file, skipping: {path}")
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
        [long["score"] == 1, long["score"] == 2, long["score"] >= 3],
        [1, 2, 3]
    )
    return long

def per_source_tier_props(tri_long):
    counts = tri_long.pivot_table(index=["source", "indicator"], columns="tier",
                                  values="score", aggfunc="count", fill_value=0)
    for t in [1, 2, 3]:
        if t not in counts.columns:
            counts[t] = 0
    counts = counts[[1, 2, 3]].sort_index()
    denom = counts.sum(axis=1).replace(0, np.nan)
    props = counts.div(denom, axis=0).reset_index().rename(columns={1: "q1", 2: "q2", 3: "q3"})
    return props

# ========== 3. WEIGHT CALCULATION FUNCTIONS (权重计算函数) ==========

def aggregate_across_sources_with_lambda(per_src, lambda_val):
    """聚合所有来源的数据并计算每个指标的"关注度分数" z_k"""
    Q = per_src.groupby("indicator")[["q1", "q2", "q3"]].mean().reset_index()
    # 核心公式: z_k = q3 + lambda * q2
    Q["z_k"] = Q["q3"] + lambda_val * Q["q2"]
    return Q

def aggregate_single_source_with_lambda(per_src, source_name, lambda_val):
    """为单个来源计算"关注度分数" z_k"""
    source_data = per_src[per_src["source"] == source_name].copy()
    if source_data.empty:
        return pd.DataFrame()
    # 核心公式: z_k = q3 + lambda * q2
    source_data["z_k"] = source_data["q3"] + lambda_val * source_data["q2"]
    return source_data[["indicator", "z_k"]]

def calculate_top_level_weights(Q):
    """计算第一层权重 (顶层权重)"""
    z_cost = float(Q.loc[Q["indicator"] == "total_transition_cost", "z_k"].values[0])
    z_lole = float(Q.loc[Q["indicator"] == "lole", "z_k"].values[0])
    z_co2 = float(Q.loc[Q["indicator"] == "co2_emission", "z_k"].values[0])
    z_envs = Q.loc[Q["indicator"].isin(ENV_SET), "z_k"].values
    z_env = float(np.mean(z_envs)) if len(z_envs) > 0 else 0.0

    Z = {
        "Cost": z_cost,
        "LOLE": z_lole,
        "Environment": z_env,
        "Global (CO2)": z_co2,
    }
    zpow = {k: (v ** BETA) for k, v in Z.items()}
    s = sum(zpow.values())
    if s <= EPS:
        W1 = {k: 1.0 / len(zpow) for k in zpow}
    else:
        # 归一化得到权重
        W1 = {k: zpow[k] / s for k in zpow}
    return W1

def calculate_env_sub_weights(Q):
    """计算第二层权重 (环境子指标的相对权重)"""
    env_df = Q[Q["indicator"].isin(ENV_SET)].copy()
    if env_df.empty:
        return {}
    zpow = np.power(env_df["z_k"].values, BETA)
    denom = zpow.sum()
    if denom <= EPS:
        weights = np.ones(len(env_df)) / len(env_df)
    else:
        # 在环境类别内部进行归一化
        weights = zpow / denom
    return dict(zip(env_df["indicator"], weights))

def calculate_overall_weights(top_level_weights, env_sub_weights):
    """计算综合权重"""
    overall_weights = {}
    
    # 获取非环境指标的权重 (直接来自第一层)
    overall_weights[INDICATOR_ALIASES['total_transition_cost']] = top_level_weights['Cost']
    overall_weights[INDICATOR_ALIASES['lole']] = top_level_weights['LOLE']
    overall_weights[INDICATOR_ALIASES['co2_emission']] = top_level_weights['Global (CO2)']

    # 获取环境顶层权重
    w_env_top_level = top_level_weights['Environment']
    
    # 计算每个环境指标的最终权重 (公式：最终权重 = L1权重 * L2权重)
    for indicator, sub_weight in env_sub_weights.items():
        alias = INDICATOR_ALIASES[indicator]
        final_weight = w_env_top_level * sub_weight
        overall_weights[alias] = final_weight

    return overall_weights

# ========== 4. MAIN EXECUTION BLOCK (主执行模块) ==========

def main():
    """主函数，执行所有计算和保存操作"""
    print("--- Starting TELA Weight Calculation and Export (Enhanced) ---")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR}")

    # --- Step 1: Load and process data from all sources ---
    frames = []
    source_frames = {}  # 新增：存储每个来源的数据框
    
    for src, path in FILES.items():
        df = read_jsonl_scores(path, src)
        if not df.empty:
            frames.append(df)
            source_frames[src] = df  # 保存单独的数据框

    if not frames:
        print("[ERROR] No valid input data found. Exiting.")
        return

    df_all = pd.concat(frames, ignore_index=True)
    tri = collapse_to_tiers(df_all)
    per_src = per_source_tier_props(tri)
    print("Step 1: Data loading and preprocessing complete.")

    # --- Step 2: Calculate intermediate scores for combined data ---
    Q = aggregate_across_sources_with_lambda(per_src, LAMBDA_DEFAULT)
    print("Step 2: Intermediate concern scores (z_k) calculated for combined data.")

    # --- Step 3: Calculate hierarchical weights for combined data ---
    top_level_weights = calculate_top_level_weights(Q)
    env_sub_weights = calculate_env_sub_weights(Q)
    overall_weights = calculate_overall_weights(top_level_weights, env_sub_weights)
    print("Step 3: Hierarchical weights calculated for combined data.")

    # --- Step 4: Save original outputs (保持原有文件不变) ---
    # Top-Level Weights (Level 1)
    df_top_level = pd.DataFrame(list(top_level_weights.items()), columns=['Category', 'Weight'])
    top_level_path = os.path.join(OUTPUT_DIR, 'top_level_weights.csv')
    df_top_level.to_csv(top_level_path, index=False, encoding='utf-8-sig')
    print(f"  -> Top-level weights (L1) saved to: {top_level_path}")

    # Environmental Sub-Weights (Level 2)
    env_sub_weights_aliased = {INDICATOR_ALIASES[k]: v for k, v in env_sub_weights.items()}
    df_env_sub = pd.DataFrame(list(env_sub_weights_aliased.items()), columns=['Environmental_Indicator', 'Relative_Weight'])
    env_sub_path = os.path.join(OUTPUT_DIR, 'environmental_sub_weights.csv')
    df_env_sub.to_csv(env_sub_path, index=False, encoding='utf-8-sig')
    print(f"  -> Environmental sub-weights (L2) saved to: {env_sub_path}")

    # Overall Indicator Weights
    df_overall = pd.DataFrame(list(overall_weights.items()), columns=['Indicator', 'Overall_Weight'])
    overall_path = os.path.join(OUTPUT_DIR, 'overall_indicator_weights.csv')
    df_overall.to_csv(overall_path, index=False, encoding='utf-8-sig')
    print(f"  -> Overall indicator weights (Combined) saved to: {overall_path}")

    print("-" * 60)

    # --- Step 5: 新增 - 计算和保存各个来源的权重 ---
    print("Step 5: Calculating source-specific weights...")
    
    source_names = {
        "govern": "Government",
        "news": "News",
        "research": "Research", 
        "zhihu": "Social Media"
    }

    for src_key, src_display_name in source_names.items():
        if src_key not in source_frames or source_frames[src_key].empty:
            print(f"  [WARN] No data for {src_display_name} perspective, skipping...")
            continue

        print(f"  Processing {src_display_name} perspective ({src_key})...")
        
        # 为该来源单独计算分层数据
        src_df = source_frames[src_key]
        src_tri = collapse_to_tiers(src_df)
        src_per_source = per_source_tier_props(src_tri)
        
        # 计算该来源的关注度分数
        src_Q = aggregate_single_source_with_lambda(src_per_source, src_key, LAMBDA_DEFAULT)
        
        if src_Q.empty:
            print(f"    [WARN] Could not calculate weights for {src_display_name}, skipping...")
            continue
        
        # 计算该来源的分层权重
        src_top_level = calculate_top_level_weights(src_Q)
        src_env_sub = calculate_env_sub_weights(src_Q)
        src_overall = calculate_overall_weights(src_top_level, src_env_sub)
        
        # 保存该来源的综合权重
        src_df_overall = pd.DataFrame(list(src_overall.items()), columns=['Indicator', 'Overall_Weight'])
        src_overall_path = os.path.join(OUTPUT_DIR, f'{src_key}_weights.csv')
        src_df_overall.to_csv(src_overall_path, index=False, encoding='utf-8-sig')
        
        print(f"    -> {src_display_name} weights saved to: {src_overall_path}")
        print(f"    -> Verification: Sum of {src_display_name} weights = {src_df_overall['Overall_Weight'].sum():.4f}")

    print("-" * 60)
    print("--- Script finished successfully! All weights have been calculated and saved. ---")
    
    # --- Step 6: 显示综合结果摘要 ---
    print("\n=== SUMMARY OF RESULTS ===")
    print("Files generated:")
    print(f"  1. top_level_weights.csv - Combined top-level category weights")
    print(f"  2. environmental_sub_weights.csv - Environmental indicators relative weights")
    print(f"  3. overall_indicator_weights.csv - Combined overall indicator weights")
    
    for src_key, src_display_name in source_names.items():
        if src_key in source_frames and not source_frames[src_key].empty:
            print(f"  4+. {src_key}_weights.csv - {src_display_name} perspective weights")
    
    print(f"\nAll files saved to: {OUTPUT_DIR}")
    print(f"Combined weights verification: Sum = {df_overall['Overall_Weight'].sum():.4f}")


if __name__ == "__main__":
    main()