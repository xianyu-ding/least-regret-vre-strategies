# filter_energy_content.py
# 功能：从JSONL文件中筛选指定年份及之后的能源相关内容

import json
import os

# 📂 文件路径配置
DOWNLOADS_FOLDER = r"D:\llm\results"  # 输入和输出文件夹
if not os.path.exists(DOWNLOADS_FOLDER):
    os.makedirs(DOWNLOADS_FOLDER)

# 输入文件
DEFAULT_INPUT_JSONL = os.path.join(DOWNLOADS_FOLDER, "all_news.jsonl")  # 默认输入文件

# ✅ 能源关键词定义
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
     "电能质量",
    # 补充VRE技术特性词
    "可再生能源波动性", "间歇性发电", "风光互补", "风光一体化", "多能互补",
    "系统惯量", "爬坡能力", "调频能力", "调峰能力", "源网荷储", 
    "电力系统稳定性", "功率波动", "出力预测", "电力系统系统稳定性","电网适应性",
    # 系统可靠性指标相关
    "供电可靠性", "负荷损失", "电力充裕度",
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

# 合并所有关键词
all_energy_keywords = (
    economic_keywords + technical_keywords + environmental_keywords + energy_type_keywords
)

def is_energy_related(text):
    """严格匹配能源关键词"""
    if not isinstance(text, str):
        return False
    
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

def get_user_input():
    """获取用户输入参数"""
    print("\n" + "="*50)
    print("能源内容筛选工具")
    print("="*50)
    
    # 获取筛选年份
    while True:
        try:
            year_input = input("\n请输入筛选年份（默认2013）: ").strip()
            if year_input == "":
                start_year = 2013
                break
            else:
                start_year = int(year_input)
                break
        except ValueError:
            print("❌ 请输入有效的年份数字!")
    
    # 获取输入文件路径
    input_file = input(f"\n请输入JSONL输入文件路径（默认: {DEFAULT_INPUT_JSONL}）: ").strip()
    if input_file == "":
        input_file = DEFAULT_INPUT_JSONL
    
    # 获取输出文件路径
    default_output = os.path.join(DOWNLOADS_FOLDER, f"filtered_news_{start_year}.jsonl")
    output_file = input(f"\n请输入筛选结果保存路径（默认: {default_output}）: ").strip()
    if output_file == "":
        output_file = default_output
    
    return start_year, input_file, output_file

def main():
    """从JSONL文件筛选能源相关内容"""
    # 获取用户输入
    start_year, input_file, output_file = get_user_input()
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n开始筛选处理...")
    print(f"📊 筛选{start_year}年及之后的能源内容")
    print(f"📄 输入文件: {input_file}")
    print(f"📄 输出文件: {output_file}")
    
    if not os.path.exists(input_file):
        print(f"\n❌ 输入文件不存在: {input_file}")
        input("\n按Enter键退出...")
        return
    
    total_count = 0
    year_filtered_count = 0
    energy_filtered_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                total_count += 1
                
                try:
                    record = json.loads(line.strip())
                    
                    # 尝试将年份转换为数字
                    try:
                        year = int(record.get('年份', '0'))
                    except (ValueError, TypeError):
                        year = 0
                    
                    # 筛选指定年份及之后的数据
                    if year >= start_year:
                        year_filtered_count += 1
                        
                        # 筛选能源相关文本
                        text_content = record.get('文本内容', '')
                        if is_energy_related(text_content):
                            # 为筛选后的JSONL创建简化的JSON对象
                            filtered_record = {
                                "content": text_content.strip()
                            }
                            outfile.write(json.dumps(filtered_record, ensure_ascii=False) + '\n')
                            energy_filtered_count += 1
                    
                    # 显示进度
                    if total_count % 100000 == 0:
                        print(f"⏳ 已处理 {total_count} 条记录，筛选出 {energy_filtered_count} 条能源相关内容...")
                
                except Exception as e:
                    print(f"[❌] 处理第{total_count}条记录时出错: {str(e)}")
        
        print(f"\n✅ 筛选完成")
        print(f"📊 总记录数: {total_count}")
        print(f"🔍 {start_year}年及之后记录数: {year_filtered_count}")
        print(f"🔍 最终筛选出能源相关记录数: {energy_filtered_count}")
        print(f"📂 筛选结果已保存至: {output_file}")
    
    except Exception as e:
        print(f"\n❌ 处理过程出错: {str(e)}")
    
    # 等待用户按键退出
    input("\n处理完成，按Enter键退出...")

if __name__ == "__main__":
    main()