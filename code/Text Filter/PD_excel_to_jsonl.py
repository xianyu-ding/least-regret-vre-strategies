# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:25:42 2025

@author: vanessa
"""

# excel_to_jsonl.py
# 功能：将Excel文件转换为JSONL格式

import pandas as pd
import json
import os

# 📂 文件路径配置
EXCEL_FILE = r"D:\llm\text\News\PD\ALL.xlsx"  # 输入Excel文件路径
DOWNLOADS_FOLDER = r"D:\llm\results"  # 输出文件夹
if not os.path.exists(DOWNLOADS_FOLDER):
    os.makedirs(DOWNLOADS_FOLDER)

# 输出文件
OUTPUT_JSONL = os.path.join(DOWNLOADS_FOLDER, "all_news.jsonl")  # 全部数据

def main():
    # 读取Excel文件
    print(f"📊 读取Excel文件: {EXCEL_FILE}")
    try:
        df = pd.read_excel(EXCEL_FILE)
        print(f"✅ 成功读取Excel，共{len(df)}行")
    except Exception as e:
        print(f"❌ 读取Excel失败: {e}")
        return
    
    # 检查列名是否匹配预期
    expected_columns = ['年份', '日期', '报纸版次', '标题', '文本内容']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"⚠️ Excel缺少以下列: {missing_columns}")
        print(f"⚠️ 实际列名: {list(df.columns)}")
        
        # 尝试自动匹配列（如果列数相同）
        if len(df.columns) == len(expected_columns):
            print("📝 尝试自动映射列名...")
            df.columns = expected_columns
        else:
            print("❌ 无法自动修复列名，请检查Excel格式")
            return
    
    # 处理缺失值
    df = df.fillna('')
    
    # 转换为JSONL格式并保存所有数据
    print(f"📝 转换Excel到JSONL: {OUTPUT_JSONL}")
    count = 0
    
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as outfile:
        for _, row in df.iterrows():
            try:
                # 创建JSON对象
                record = {
                    "年份": str(row['年份']),
                    "日期": str(row['日期']),
                    "报纸版次": str(row['报纸版次']),
                    "标题": str(row['标题']),
                    "文本内容": str(row['文本内容'])
                }
                
                # 写入数据
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
                
                # 显示进度
                if count % 1000 == 0:
                    print(f"⏳ 已处理 {count} 条记录...")
                    
            except Exception as e:
                print(f"[❌] 处理第{count+1}行时出错: {str(e)}")
    
    print(f"\n✅ 转换完成")
    print(f"📊 共转换 {count} 条记录")
    print(f"📂 结果已保存至: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()