# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 21:33:27 2025

@author: vanessa
"""

import requests
import json
import os
import time
import re
from datetime import datetime
import threading
import queue
from tqdm import tqdm

# Deepseek API configuration
API_URL = "https://api.deepseek.com/chat/completions"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()

def read_jsonl(file_path):
    """Read JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_score(text, indicator):
    """Extract only score for a given indicator from model's text output"""
    # Pattern to match score for a specific indicator - simplified for score only
    pattern = rf"{re.escape(indicator)}：(\d+)分"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        score = int(match.group(1))
        return score
    else:
        # Fallback if pattern not found
        return 1

def process_model_output(text):
    """Process the text output from the model into structured JSON format - scores only"""
    # Create the base structure with scores only
    result = {
        "economic": {
            "total_transition_cost": extract_score(text, "total_transition_cost")
        },
        "technical": {
            "lole": extract_score(text, "lole")
        },
        "environmental": {
            "land_occupation": extract_score(text, "land_occupation"),
            "freshwater_ecotoxicity": extract_score(text, "freshwater_ecotoxicity"),
            "human_toxicity": extract_score(text, "human_toxicity"),
            "ionising_radiation": extract_score(text, "ionising_radiation"),
            "marine_eutrophication": extract_score(text, "marine_eutrophication"),
            "mineral_resource_depletion": extract_score(text, "mineral_resource_depletion")
        }
    }
    
    return result

def process_text(text, retry_count=3):
    """Process a single news text using Deepseek API, provide scores only without reasons"""
    # Optimized prompt focusing only on scores without reasons
    messages = [
        {"role": "system", "content": """你是一个专业的能源新闻评估助手。对以下指标评分（1-5分），无需提供理由：

经济维度：
- total_transition_cost：评估新闻是否讨论能源转型成本

技术维度：
- lole：评估新闻是否讨论电力系统可靠性相关问题

环境维度：
- land_occupation：评估新闻是否讨论土地占用问题
- freshwater_ecotoxicity：评估新闻是否讨论淡水生态毒性问题
- human_toxicity：评估新闻是否讨论人体健康毒性影响
- ionising_radiation：评估新闻是否讨论电离辐射问题
- marine_eutrophication：评估新闻是否讨论海洋富营养化问题
- mineral_resource_depletion：评估新闻是否讨论矿产资源消耗问题

评分标准：
1分：新闻未提供与指标相关的信息
2分：新闻有极少量相关信息
3分：新闻有一定篇幅讨论并提供部分数据
4分：新闻有较多篇幅讨论并提供多个数据
5分：新闻主要围绕该指标展开，提供全面深入的数据

输出格式要求（仅提供分数）：
total_transition_cost：X分
lole：X分
land_occupation：X分
freshwater_ecotoxicity：X分
human_toxicity：X分
ionising_radiation：X分
marine_eutrophication：X分
mineral_resource_depletion：X分"""},
        {"role": "user", "content": f"请评估以下新闻：\n\n{text}"}
    ]
    
    # Prepare the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    data = {
        "model": "deepseek-reasoner",
        "messages": messages,
        "stream": False,
        "temperature": 0
    }
    
    for attempt in range(retry_count):
        try:
            response = requests.post(API_URL, headers=headers, json=data, timeout=60)
            
            # Print response for debugging (only on first attempt)
            if attempt == 0 and response.status_code != 200:
                print(f"API Response Status: {response.status_code}")
                body_preview = response.text[:300] + "..." if len(response.text) > 300 else response.text
                print(f"API Response Body (preview): {body_preview}")
            
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            response_data = response.json()
            content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Process the text output into structured JSON
            structured_result = process_model_output(content)
            
            # Return only the structured data with scores
            return {
                "structured_data": structured_result
            }
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retry_count} failed: {str(e)}")
            if attempt < retry_count - 1:
                time.sleep(2)
            else:
                return None

def save_results(results, output_file, append=False):
    """Save results to file, supports append mode"""
    mode = 'a' if append else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def get_user_choice():
    """Get user choice for processing amount"""
    while True:
        print("\n请选择要处理的数据量：")
        print("1. 处理前2条数据（用于测试）")
        print("2. 处理前100条数据")
        print("3. 处理全部数据")
        print("4. 退出程序")
        
        choice = input("\n请输入选项（1-4）: ").strip()
        
        if choice == '1':
            return 2
        elif choice == '2':
            return 100
        elif choice == '3':
            return -1  # -1 means process all data
        elif choice == '4':
            return 0   # 0 means exit
        else:
            print("无效的选择，请重新输入")

def get_concurrency_limit():
    """Get concurrency limit from user"""
    while True:
        try:
            limit = int(input("请输入并发请求数（建议5-20，取决于API限制）: ").strip())
            if 1 <= limit <= 100:
                return limit
            else:
                print("请输入1-100之间的数字")
        except ValueError:
            print("请输入有效的数字")

def get_api_config():
    """Get API configuration from user"""
    global API_URL, API_KEY
    API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not API_KEY:
        raise RuntimeError("Set the DEEPSEEK_API_KEY environment variable before running API scoring.")
    
    default_url = API_URL
    url = input(f"请输入Deepseek API URL (默认: {default_url}): ").strip()
    if url:
        API_URL = url
    
    model = input("请输入模型名称 (默认: deepseek-reasoner): ").strip()
    if not model:
        model = "deepseek-reasoner"
        
    return model

def worker(worker_id, task_queue, result_queue, progress, lock, model):
    """Worker thread to process tasks"""
    while True:
        try:
            # Get a task from the queue
            task = task_queue.get(block=False)
            if task is None:  # None is our signal to stop
                task_queue.task_done()
                break
                
            idx, item = task
            
            # Get content field, adapt to different input formats
            if 'content' in item:
                text = item['content']
            elif 'text' in item:
                text = item['text']
            else:
                print(f"Warning: Item {idx+1} has no content field, skipping")
                with lock:
                    progress.update(1)
                task_queue.task_done()
                continue
            
            # Process the text
            response = process_text(text)
            
            if response is not None:
                result_item = {
                    'input_text': text,
                    'assessment': response["structured_data"],  # Only save the structured data without reasons
                }
                result_queue.put(result_item)
            
            # Update progress
            with lock:
                progress.update(1)
                
            # Mark task as done
            task_queue.task_done()
            
        except queue.Empty:
            # No more tasks
            break
        except Exception as e:
            print(f"Worker {worker_id} error: {str(e)}")
            with lock:
                progress.update(1)
            task_queue.task_done()

def process_batch_with_threads(data_to_process, output_file, concurrency_limit, model):
    """Process a batch of data with thread-based concurrency"""
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    lock = threading.Lock()
    
    # Create a progress bar
    progress = tqdm(total=len(data_to_process), desc="Processing")
    
    # Add tasks to the queue
    for idx, item in enumerate(data_to_process):
        task_queue.put((idx, item))
    
    # Add termination signals
    for _ in range(concurrency_limit):
        task_queue.put(None)
    
    # Create and start worker threads
    threads = []
    for i in range(concurrency_limit):
        t = threading.Thread(target=worker, args=(i, task_queue, result_queue, progress, lock, model))
        t.daemon = True
        threads.append(t)
        t.start()
    
    # Initialize variables for saving results
    results = []
    save_interval = 500  # Save every 500 records (changed from 50)
    save_count = 0
    interrupted = False
    last_save_time = time.time()
    auto_save_time_interval = 1200  # Auto save every 20 minutes (1200 seconds)
    
    try:
        while task_queue.qsize() > concurrency_limit or not task_queue.empty():
            try:
                # Try to get a result (timeout to allow checking queue size)
                result = result_queue.get(block=True, timeout=0.5)
                results.append(result)
                result_queue.task_done()
                
                # Save results when reaching the interval
                current_time = time.time()
                if len(results) >= save_interval or (current_time - last_save_time) >= auto_save_time_interval:
                    save_results(results, output_file, append=(save_count > 0))
                    print(f"\n自动保存了 {len(results)} 条结果")
                    results.clear()
                    save_count += 1
                    last_save_time = current_time
            except queue.Empty:
                # No results available yet
                continue
    except KeyboardInterrupt:
        print("\n检测到中断信号，正在保存当前结果并等待正在处理的任务完成...")
        interrupted = True
    
    # If interrupted, save current results immediately
    if interrupted and results:
        save_results(results, output_file, append=(save_count > 0))
        print(f"\n已保存 {len(results)} 条中断时的结果")
        results.clear()
        save_count += 1
    
    # Wait for all tasks to complete - fixed the timeout issue
    try:
        # Set a flag for giving up after timeout
        wait_start = time.time()
        wait_timeout = 30 if interrupted else 300  # shorter timeout if interrupted
        
        # Keep checking until queue is empty or timeout
        while not task_queue.empty():
            if time.time() - wait_start > wait_timeout:
                print(f"\n等待任务完成超时 ({wait_timeout}秒)，放弃等待。")
                break
            time.sleep(0.5)  # Small sleep to prevent CPU hogging
            
    except KeyboardInterrupt:
        print("\n再次检测到中断，强制终止。部分任务可能未完成。")
    
    # Get any remaining results
    try:
        while True:
            try:
                result = result_queue.get(block=False)
                results.append(result)
                result_queue.task_done()
            except queue.Empty:
                break
    except Exception as e:
        print(f"获取剩余结果时出错: {str(e)}")
    
    # Save any remaining results
    if results:
        save_results(results, output_file, append=(save_count > 0))
        print(f"\n已保存剩余的 {len(results)} 条结果")
    
    progress.close()
    
    # Calculate successful completions
    processed_count = sum(
        [save_interval] * save_count  # Full batches of 500
    )
    if results:  # Add the last partial batch
        processed_count += len(results)
    
    # Make sure we don't report more successes than attempts
    if processed_count > len(data_to_process):
        processed_count = len(data_to_process)
        
    failed_count = len(data_to_process) - processed_count
    
    return failed_count

def main():
    """Main function to run the multithreaded program"""
    print("能源新闻多线程评估工具 - DeepSeek Reasoner API (优化版)")
    print("=" * 60)
    
    # Get API configuration
    model = get_api_config()
    
    # Get input file path
    input_file = input("请输入学术JSONL文件路径 (默认: D:\\llm\\results\\filtered_research.jsonl): ").strip()
    if not input_file:
        input_file = r"D:\llm\results\filtered_research.jsonl"
    
    # Use fixed output directory
    output_dir = r"D:\llm\results\result_v3"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"research_scores_{timestamp}.jsonl")
    
    print(f"\n输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"API URL: {API_URL}")
    print(f"模型: {model}")
    
    print("\n正在读取输入文件...")
    data = read_jsonl(input_file)
    print(f"总共读取到 {len(data)} 条数据")
    
    while True:
        process_count = get_user_choice()
        
        if process_count == 0:
            print("程序已退出")
            break
            
        if process_count > 0:
            data_to_process = data[:process_count]
        else:
            data_to_process = data
        
        # Get concurrency limit from user
        concurrency_limit = get_concurrency_limit()
            
        print(f"\n将处理 {len(data_to_process)} 条数据，并发数：{concurrency_limit}")
        print(f"每500条记录或每20分钟将自动保存一次")
        confirm = input("确认继续？(y/n): ").strip().lower()
        if confirm != 'y':
            continue
            
        try:
            start_time = time.time()
            print("\n开始处理数据...")
            
            failed_count = process_batch_with_threads(data_to_process, output_file, concurrency_limit, model)
            
            elapsed_time = time.time() - start_time
            processed_count = len(data_to_process) - failed_count
            
            print(f"\n处理完成！")
            print(f"共尝试处理 {len(data_to_process)} 条数据")
            print(f"成功: {processed_count} 条")
            print(f"失败: {failed_count} 条")
            print(f"耗时: {elapsed_time:.2f} 秒")
            if elapsed_time > 0:
                print(f"平均速度: {processed_count / elapsed_time:.2f} 条/秒")
            print(f"结果已保存到: {output_file}")
            
            print("\n是否继续处理其他数据？")
                
        except KeyboardInterrupt:
            print("\n检测到用户中断，程序终止。")

if __name__ == "__main__":
    main()