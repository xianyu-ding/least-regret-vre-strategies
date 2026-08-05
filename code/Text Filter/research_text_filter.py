# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:40:00 2025

@author: vanessa
"""

import json
import os

# 📂 输入文件夹路径
#INPUT_FOLDER = r"D:\llm\text\news"
#INPUT_FOLDER = r"D:\llm\text\ZWJCYLK2.0-11"
INPUT_FOLDER = r"D:\llm\text\research"
# 📂 输出文件路径（存到 Downloads 文件夹）
DOWNLOADS_FOLDER = r"D:\llm\results"
OUTPUT_JSONL = os.path.join(DOWNLOADS_FOLDER, "filtered_research.jsonl")

# ✅ 严格的能源关键词（去掉容易误匹配的词）



economic_keywords = [
    # 原有经济类关键词
    "能源投资", "能源经济", "能源市场", "电力市场", "绿电交易", "电力交易", 
    "能耗双控", "电价机制", "可再生能源补贴", "可再生能源配额制", "上网电价", 
    "碳交易", "碳定价", "碳市场", "碳税", "碳配额",
    # 补充转型相关经济词
    "能源转型成本", "系统集成成本", "清洁能源成本", "平准化度电成本",
    "电网灵活性价值", "电网波动性成本"
]

technical_keywords = [
    # 原有技术类关键词
    "电网改革", "电力调度", "电力平衡", "电力系统优化", "智能电网", "负荷响应", 
    "电力传输", "储能技术", "抽水蓄能", 
    "热电联产", "负荷预测", "电网可靠性", "分布式能源", "能源微网", 
     "电能质量"
    # 补充VRE技术特性词
    "可再生能源波动性", "间歇性发电", "风光互补", "风光一体化", "多能互补",
    "系统惯量", "爬坡能力", "调频能力", "调峰能力", "源网荷储", 
    "电力系统稳定性", "功率波动", "出力预测", "电力系统系统稳定性","电网适应性"
    # 系统可靠性指标相关
    "供电可靠性", "负荷损失", "电力充裕度"
    "LOLE", "LOLP", "EENS",
    "供电安全", "电力短缺", "负荷切断", 
    "可靠性评估", "可靠性指标", "可靠性标准", "电力不足",
    "容量充足率", "发电充足性", "供需平衡", "电力缺口",
    "可靠性规划", "可靠性水平", "电力系统脆弱性", "供电保障"

]

environmental_keywords = [
    # 原有环境类关键词
    "双碳","碳中和", "碳达峰", "温室气体排放", "低碳能源", "气候变化", "可持续能源", 
    "能源脱碳", "碳捕集与封存", "碳排放交易", "环境影响评估", "空气污染物减排", 
    "可再生能源环境影响", "淡水生态毒性", "水资源消耗", "土地占用", "生态破坏", 
    "电离辐射", "矿产资源消耗",
    # 补充环境评估词
    "生命周期评价", "环境足迹", "生态系统影响", "资源可持续性",
    "累积环境影响", "环境承载力"
]

energy_type_keywords = [
    # 原有能源类型关键词
    "风电", "光伏发电", "太阳能发电", "风能", "海上风电", "陆上风电", "光热发电", 
    "光伏组件", "氢能", "绿氢", "蓝氢", "灰氢", "天然气发电", "核能发电", 
    "生物质能", "地热能", "潮汐能",
    # 补充系统集成词
    "多能互补系统", "风光储一体化", "可再生能源基地",
    "分布式发电系统"
]



all_energy_keywords = (
    economic_keywords + technical_keywords + environmental_keywords + energy_type_keywords
)

def is_energy_related(text):
    """严格匹配能源关键词"""
    text = text.lower()  # 统一小写以防止匹配错误
    matches = [kw for kw in all_energy_keywords if kw in text]  # 计算匹配的关键词数
    word_count = len(text.split())  # 计算总单词数
    keyword_ratio = len(matches) / (word_count + 1)  # 计算关键词占比，避免除零

    # ✅ 只有当：
    # 1. 至少包含 2 个能源关键词
    # 2. 关键词占比超过 2%
    if len(matches) >= 2 and keyword_ratio > 0.02:
        return True
    return False

filtered_count = 0  # 统计筛选出的能源相关文本
total_count = 0  # 统计所有处理的文本

# 🚀 遍历所有 JSONL 文件
with open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:

    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".jsonl"):  # 只处理 JSONL 文件
            input_filepath = os.path.join(INPUT_FOLDER, filename)

            print(f"🔍 处理文件: {filename}")

            with open(input_filepath, "r", encoding="utf-8") as infile:
                for line in infile:
                    total_count += 1
                    try:
                        doc = json.loads(line.strip())  # 解析 JSON
                        text = doc.get("text", "")  # ✅ 读取 "Content" 字段（您的文件使用的是大写 "Content"）

                        if is_energy_related(text):  # 🔍 更严格筛选
                            doc["source_file"] = filename  # 记录原始文件来源
                            outfile.write(json.dumps(doc, ensure_ascii=False) + "\n")
                            filtered_count += 1

                    except json.JSONDecodeError:
                        print(f"[⚠️] {filename} 中出现 JSON 解析错误，跳过该行。")
                        continue

print(f"\n✅ 处理完成，遍历了 {total_count} 条文本，严格筛选出 {filtered_count} 条能源相关文本。")
print(f"📂 结果已保存至: {OUTPUT_JSONL}")
