#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 18:51:51 2025

@author: vanessa
"""

import pandas as pd
import os
import re

def load_specific_format_data():
    """
    根据实际文件格式特点，专门处理四类数据:
    1. LOLE数据 - lole_df.csv
    2. Cost数据 - cost_df.csv
    3. CO2数据 - co2_df.csv
    4. 环境指标数据 - environmental_impacts 文件夹 (Total 行)
    """

    base_dir = "/Users/vanessa/Library/CloudStorage/OneDrive-UniversityCollegeLondon/research/data/paper3/MCDM_FINAL"

    # ------------------------------------------------------------------
    # 通用函数：解析 "**场景** 数值" 的文本格式
    # ------------------------------------------------------------------
    def parse_text_format(file_path, col_name):
        with open(file_path, 'r') as file:
            content = file.read()
        scenarios = re.findall(r'\*\*([^*]+)\*\*', content)
        values = re.findall(r'\*\*[^*]+\*\*\s*(\d+(?:\.\d+)?)', content)
        if len(scenarios) == len(values) and len(scenarios) > 0:
            return pd.DataFrame({
                "Scenario": scenarios,
                col_name: [float(v) for v in values]
            })
        else:
            print(f"警告: 无法解析 {col_name} 数据，格式不符合预期")
            return pd.DataFrame(columns=["Scenario", col_name])

    # ------------------------------------------------------------------
    # 1. LOLE 数据
    # ------------------------------------------------------------------
    print("处理LOLE数据...")
    lole_path = os.path.join(base_dir, "lole_df.csv")
    try:
        lole_df = pd.read_csv(lole_path)
        if "Scenario" in lole_df.columns and "Pv" in lole_df.columns:
            lole_df.rename(columns={"Pv": "LOLE"}, inplace=True)
        else:
            lole_df = parse_text_format(lole_path, "LOLE")
    except Exception as e:
        print(f"读取LOLE数据出错: {str(e)}")
        lole_df = pd.DataFrame(columns=["Scenario", "LOLE"])
    print(f"LOLE数据处理完成，共{len(lole_df)}行")

    # ------------------------------------------------------------------
    # 2. Cost 数据
    # ------------------------------------------------------------------
    print("\n处理Cost数据...")
    cost_path = os.path.join(base_dir, "cost_df.csv")
    try:
        cost_df = pd.read_csv(cost_path)
        if "Scenario" in cost_df.columns and "Pv" in cost_df.columns:
            cost_df.rename(columns={"Pv": "Cost"}, inplace=True)
        else:
            cost_df = parse_text_format(cost_path, "Cost")
    except Exception as e:
        print(f"读取Cost数据出错: {str(e)}")
        cost_df = pd.DataFrame(columns=["Scenario", "Cost"])
    print(f"Cost数据处理完成，共{len(cost_df)}行")

    # ------------------------------------------------------------------
    # 3. CO2 数据
    # ------------------------------------------------------------------
    print("\n处理CO2数据...")
    co2_path = os.path.join(base_dir, "co2_df.csv")
    try:
        co2_df = pd.read_csv(co2_path)
        if "Scenario" in co2_df.columns and "Pv" in co2_df.columns:
            co2_df.rename(columns={"Pv": "CO2_emission"}, inplace=True)
        else:
            co2_df = parse_text_format(co2_path, "CO2_emission")
    except Exception as e:
        print(f"读取CO2数据出错: {str(e)}")
        co2_df = pd.DataFrame(columns=["Scenario", "CO2_emission"])
    print(f"CO2数据处理完成，共{len(co2_df)}行")

    # ------------------------------------------------------------------
    # 4. 环境指标数据
    # ------------------------------------------------------------------
    print("\n处理环境指标数据...")
    env_dir = os.path.join(base_dir, "environmental_impacts")
    env_metrics = [
        "Freshwater ecotoxicity.csv",
        "Freshwater eutrophication.csv",
        "Human toxicity.csv",
        "Ionising radiation.csv",
        "Land occupation.csv",
        "Marine eutrophication.csv",
        "Mineral resource depletion.csv"
    ]

    env_data = {}
    for metric_file in env_metrics:
        metric_name = metric_file.replace(".csv", "")
        file_path = os.path.join(env_dir, metric_file)
        print(f"\n处理环境指标文件: {metric_name}")
        if not os.path.exists(file_path):
            print(f"  警告: 文件不存在: {file_path}")
            continue
        try:
            df_env = pd.read_csv(file_path)
            df_env.columns = [c.strip() for c in df_env.columns]
            row_mask = df_env.iloc[:, 0].astype(str).str.strip().str.lower().eq('total')
            if not row_mask.any():
                print(f"  警告: 未找到 'Total' 行，跳过 {metric_name}")
                continue
            total_row = df_env[row_mask].iloc[0]
            scenario_cols = df_env.columns[2:]
            unit_value = str(total_row[df_env.columns[1]]) if len(df_env.columns) > 1 else ""
            data_dict = {}
            for col in scenario_cols:
                try:
                    val = float(total_row[col])
                except:
                    val = float('nan')
                data_dict[col] = val
            env_data[metric_name] = {
                "unit": unit_value,
                "data": data_dict
            }
            print(f"  成功读取 {metric_name} 的 Total 行, 共 {len(data_dict)} 个场景")
        except Exception as e:
            print(f"  处理 {metric_name} 时出错: {str(e)}")
    print(f"\n环境指标处理完成，共处理{len(env_data)}个指标")

    # ------------------------------------------------------------------
    # 5. 合并所有数据
    # ------------------------------------------------------------------
    print("\n合并所有数据...")
    all_scenarios = set()
    if not lole_df.empty:
        all_scenarios.update(lole_df["Scenario"].tolist())
    if not cost_df.empty:
        all_scenarios.update(cost_df["Scenario"].tolist())
    if not co2_df.empty:
        all_scenarios.update(co2_df["Scenario"].tolist())
    for metric, env_info in env_data.items():
        all_scenarios.update(env_info["data"].keys())
    all_scenarios = sorted(list(all_scenarios))
    print(f"所有数据共包含{len(all_scenarios)}个唯一场景: {all_scenarios}")

    result_df = pd.DataFrame({"Scenario": all_scenarios})
    if not lole_df.empty:
        result_df = result_df.merge(lole_df, on="Scenario", how="left")
    if not cost_df.empty:
        result_df = result_df.merge(cost_df, on="Scenario", how="left")
    if not co2_df.empty:
        result_df = result_df.merge(co2_df, on="Scenario", how="left")

    for metric_name, env_info in env_data.items():
        data_dict = env_info["data"]
        metric_values = [data_dict.get(s, float('nan')) for s in result_df["Scenario"]]
        result_df[metric_name] = metric_values

    # ✅ 只保留指定的 6 个场景
    valid_scenarios = ["NDC-LTP", "VRE-70%", "VRE-75%", "VRE-80%", "VRE-85%", "VRE-90%"]
    before_count = len(result_df)
    result_df = result_df[result_df["Scenario"].isin(valid_scenarios)]
    after_count = len(result_df)
    print(f"过滤场景: 保留 {after_count} 个, 剔除 {before_count - after_count} 个")

    print(f"数据合并完成，最终数据形状: {result_df.shape}")
    return result_df


if __name__ == "__main__":
    try:
        combined_df = load_specific_format_data()
        print("\n合并后的数据前5行:")
        print(combined_df.head())
        missing_counts = combined_df.isnull().sum()
        print("\n各列缺失值数量:")
        print(missing_counts)
        output_path = "/Users/vanessa/Library/CloudStorage/OneDrive-UniversityCollegeLondon/research/data/paper3/MCDM_FINAL/combined_data.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"\n合并后的数据已保存至: {output_path}")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
