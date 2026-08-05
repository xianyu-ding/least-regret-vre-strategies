# -*- coding: utf-8 -*-
"""
改进版评估数据处理脚本
1. 详细打印每个文件的处理信息（记录数、错误数、包含文本数）
2. 打印总体文本统计信息，方便核对各文件数据
3. 保存最终计算的评估结果
"""

import json
import pandas as pd
import os
import numpy as np
from collections import defaultdict
import datetime

def read_jsonl_files(file_paths, verbose=True):
    """
    读取多个JSONL文件并解析评估数据，区分不同来源和模型。
    同时记录每个文件的详细处理信息：
      - processed：该文件中成功处理的记录数（含 assessment 数据）
      - errors：错误或无 assessment 数据的记录数
      - texts：包含 input_text 的记录数
    """
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_texts = defaultdict(lambda: defaultdict(list))
    all_text_lengths = defaultdict(lambda: defaultdict(list))
    
    # 用于存储每个文件的详细处理信息
    file_details = []
    
    for file_path in file_paths:
        # 判断模型（依据文件所在文件夹）
        model = 'local_deepseek' if 'result_v2' in file_path else 'api_deepseek'
        # 判断来源（依据文件名）
        filename = os.path.basename(file_path)
        if 'news' in filename:
            source = 'news'
        elif 'zhihu' in filename:
            source = 'socialmedia'
        elif 'govern' in filename:
            source = 'government'
        elif 'research' in filename:
            source = 'research'
        else:
            source = 'unknown'
            
        if verbose:
            print(f"Processing {model} - {source}: {file_path}")
        
        processed_count = 0
        error_count = 0
        text_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    assessment = data.get('assessment')
                    if assessment:
                        # 记录各类别各指标的分数
                        for category, metrics in assessment.items():
                            for metric, score in metrics.items():
                                all_data[model][source][f"{category}_{metric}"].append(score)
                        
                        # 如果存在文本信息，则保存
                        text = data.get('input_text', '')
                        if text:
                            text_info = {'text': text, 'source': data.get('source_file', source)}
                            all_texts[model][source].append(text_info)
                            all_text_lengths[model][source].append(len(text))
                            text_count += 1
                        processed_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
        
        # 记录该文件的详细信息
        file_details.append({
            'file_path': file_path,
            'model': model,
            'source': source,
            'processed': processed_count,
            'errors': error_count,
            'texts': text_count
        })
        
        if verbose:
            print(f"  Processed: {processed_count} records, Errors: {error_count}, Texts Found: {text_count}")
    
    return all_data, all_texts, all_text_lengths, file_details

def calculate_weights(model_source_scores):
    """计算权重结构并归一化分数（保留概要信息）"""
    results = {}
    for model, sources in model_source_scores.items():
        results[model] = {}
        for source, scores in sources.items():
            if not scores:
                continue
            try:
                df = pd.DataFrame(scores)
                if df.empty:
                    continue
                
                # 按类别分组（依据列名格式 category_metric）
                category_columns = {}
                for column in df.columns:
                    try:
                        category, metric = column.split('_', 1)
                        category_columns.setdefault(category, []).append(column)
                    except ValueError:
                        continue
                
                if not category_columns:
                    continue
                
                total_scores = df.sum()
                mean_scores = df.mean()
                # 计算每个类别的平均分
                category_means = {cat: np.mean([mean_scores[col] for col in cols])
                                  for cat, cols in category_columns.items()}
                total_category_mean = sum(category_means.values())
                if total_category_mean == 0:
                    continue
                    
                category_weights = {cat: mean / total_category_mean for cat, mean in category_means.items()}
                
                overall_weights = {}
                for category, columns in category_columns.items():
                    category_total = sum(total_scores[col] for col in columns)
                    if category_total == 0:
                        equal_weight = 1.0 / len(columns)
                        for column in columns:
                            overall_weights[column] = category_weights[category] * equal_weight
                    else:
                        for column in columns:
                            overall_weights[column] = category_weights[category] * (total_scores[column] / category_total)
                
                results[model][source] = {
                    'overall_weights': overall_weights,
                    'category_weights': category_weights,
                    'mean_scores': mean_scores.to_dict(),
                    'category_means': category_means
                }
            except Exception as e:
                print(f"Error calculating weights for {model} - {source}: {str(e)}")
                continue
    return results

def print_file_processing_details(file_details):
    """打印每个文件的详细处理信息"""
    print("\n详细文件处理信息：")
    for detail in file_details:
        print(f"文件: {detail['file_path']}")
        print(f"  模型: {detail['model']}  来源: {detail['source']}")
        print(f"  处理记录数: {detail['processed']}，错误数: {detail['errors']}，包含文本的记录数: {detail['texts']}\n")

def print_text_statistics(all_texts, all_text_lengths):
    """打印总体文本统计信息（按模型汇总）"""
    for model, sources in all_texts.items():
        total_texts = sum(len(texts) for texts in sources.values())
        all_lengths = [length for lengths in all_text_lengths[model].values() for length in lengths]
        avg_length = np.mean(all_lengths) if all_lengths else 0
        print(f"\n{model} 模型总体文本统计信息:")
        print(f"  处理的文本总数: {total_texts}")
        print(f"  平均文本长度: {avg_length:.2f} 字符")
        if all_lengths:
            print(f"  最短文本长度: {min(all_lengths)} 字符")
            print(f"  最长文本长度: {max(all_lengths)} 字符")

def save_results_to_json(results, output_path):
    """将计算结果保存为JSON文件"""
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"combined_assessment_results_{timestamp}.json"
    full_path = os.path.join(output_path, filename)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n分析完成，结果已保存至: {full_path}")
    return full_path

def main():
    base_dirs = [
        r"D:\llm\results\result_v2",
        r"D:\llm\results\result_v3"
    ]
    
    file_patterns = [
        "news_results.jsonl",
        "zhihu_results.jsonl", 
        "govern_results.jsonl", 
        "research_results.jsonl"
    ]
    
    input_files = []
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Warning: Directory {base_dir} 不存在，跳过。")
            continue
        for file_name in file_patterns:
            full_path = os.path.join(base_dir, file_name)
            if os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
                input_files.append(full_path)
                print(f"Found file: {full_path}")
            else:
                print(f"文件不存在或为空: {full_path}")
    
    if not input_files:
        print("Error: 未找到输入文件，请检查路径和文件名。")
        return
        
    # 读取并处理所有文件，同时获得详细处理信息
    all_data, all_texts, all_text_lengths, file_details = read_jsonl_files(input_files, verbose=True)
    
    # 打印每个文件的详细处理信息
    print_file_processing_details(file_details)
    
    # 打印总体文本统计信息
    print_text_statistics(all_texts, all_text_lengths)
    
    # 计算评估指标权重
    results = calculate_weights(all_data)
    
    # 保存结果到JSON文件
    output_dir = r"D:\llm\results\analysis"
    os.makedirs(output_dir, exist_ok=True)
    save_results_to_json(results, output_dir)
    
if __name__ == "__main__":
    main()
