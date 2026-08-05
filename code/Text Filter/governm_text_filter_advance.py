# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:52:27 2025
@author: vanessa

PDF文档处理脚本 - 将政府文件PDF转换为JSONL格式
"""

import json
import pdfplumber
import re
import os
import logging
from datetime import datetime

def setup_logging(output_folder):
   """配置日志"""
   log_file = os.path.join(output_folder, 
                          f"pdf_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
   
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler(log_file, encoding='utf-8'),
           logging.StreamHandler()
       ]
   )

def has_long_ellipsis(text, min_length=30):
   """
   检查是否包含长串省略号
   min_length: 连续省略号的最小长度
   """
   # 匹配连续的省略号(包括各种类型的省略号)
   patterns = [r'\.{2,}', r'。{2,}', r'…{1,}', r'．{2,}']
   
   ellipsis_count = 0 
   for pattern in patterns:
       matches = re.finditer(pattern, text)
       for match in matches:
           if len(match.group()) >= min_length/3:  # 因为...是3个点
               ellipsis_count += 1
               if ellipsis_count >= 2:  # 发现2段或以上的长省略号
                   return True
   return False

def is_valid_content(text):
   """
   判断内容是否有效
   1. 去除过短的内容(如标题)
   2. 去除含有长串省略号的内容
   3. 去除含有特殊格式的内容
   """
   # 内容太短的过滤掉(比如少于50个字符)
   if len(text) < 50:
       return False
       
   # 检查长串省略号
   if has_long_ellipsis(text):
       return False
       
   # 过滤掉可能的页眉页脚和导航
   if any(x in text for x in ['| | |', '字号：', '打印', '收藏', '目 录', '前 言']):
       return False
       
   # 过滤掉日期、来源等信息
   if re.search(r'\d{4}[-年]\d{1,2}[-月]\d{1,2}[日]?\s*来源：', text):
       return False
       
   return True

def parse_paragraphs(text):
   """
   简单的中文段落解析:
   1. 使用两个或更多换行符分割主段落
   2. 去除每段首尾空白字符
   3. 过滤掉空段落
   """
   if not text:
       return []
       
   # 按两个或更多换行符分割段落
   paragraphs = [p.strip() for p in re.split(r'\n{2,}', text)]
   # 过滤空段落
   paragraphs = [p for p in paragraphs if p]
   
   return paragraphs

def extract_text_from_pdf(file_path):
   """提取PDF文本并分段"""
   text_data = []
   try:
       with pdfplumber.open(file_path) as pdf:
           filename = os.path.basename(file_path)
           logging.info(f"Processing {filename}")
           
           for page in pdf.pages:
               try:
                   text = page.extract_text()
                   if text:
                       paragraphs = parse_paragraphs(text)
                       text_data.extend(paragraphs)
               except Exception as e:
                   logging.error(f"Error processing page in {filename}: {str(e)}")
                   continue
                   
           logging.info(f"Extracted {len(text_data)} paragraphs from {filename}")
           
   except Exception as e:
       logging.error(f"Failed to open PDF {file_path}: {str(e)}")
       return []
       
   return text_data

def save_jsonl(data, output_path):
   """保存JSONL文件"""
   try:
       with open(output_path, "w", encoding="utf-8") as f:
           for entry in data:
               f.write(json.dumps(entry, ensure_ascii=False) + "\n")
       logging.info(f"Successfully saved {len(data)} entries to {output_path}")
   except Exception as e:
       logging.error(f"Error saving JSONL file: {str(e)}")
       raise

def process_pdfs(input_folder, output_folder):
    """处理文件夹中的所有PDF文件"""
    jsonl_data = []
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]
    
    if not pdf_files:
        logging.warning(f"No PDF files found in {input_folder}")
        return []
    
    logging.info(f"Found {len(pdf_files)} PDF files")
    
    for filename in pdf_files:
        file_path = os.path.join(input_folder, filename)
        
        # 提取年份
        match = re.search(r'_(\d{4})', filename)
        year = match.group(1) if match else "Unknown"
        doc_name = filename.replace(".pdf", "")
        
        # 提取文本
        text_paragraphs = extract_text_from_pdf(file_path)
        
        # 过滤并添加到JSONL数据
        filtered_count = 0
        for para in text_paragraphs:
            if is_valid_content(para):
                jsonl_data.append({
                    "time": year,
                    "document": doc_name,
                    "text": para  # 这里改为text
                })
            else:
                filtered_count += 1
                
        logging.info(f"Filtered out {filtered_count} invalid paragraphs from {filename}")
    
    return jsonl_data
def main():
   """主函数"""
   # 文件夹路径
   input_folder = "D:\\llm\\text\\government"
   output_folder = "D:\\llm\\results"
   
   # 确保输出文件夹存在
   os.makedirs(output_folder, exist_ok=True)
   
   # 设置日志
   setup_logging(output_folder)
   
   logging.info("Starting PDF processing pipeline")
   
   try:
       # 处理PDF文件
       jsonl_data = process_pdfs(input_folder, output_folder)
       
       if not jsonl_data:
           logging.warning("No data extracted from PDFs")
           return
           
       # 保存JSONL文件
       output_path = os.path.join(output_folder, "filtered_govern.jsonl")
       save_jsonl(jsonl_data, output_path)
       
       logging.info("Pipeline completed successfully")
       
   except Exception as e:
       logging.error(f"Pipeline failed: {str(e)}")
       raise

if __name__ == "__main__":
   main()