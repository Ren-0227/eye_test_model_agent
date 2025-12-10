#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理华佗医学数据集并转换为微调格式
Process Huatuo medical datasets and convert to fine-tuning format
"""

import os
import json
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm

def process_knowledge_graph_data(dataset):
    """
    处理知识图谱问答数据
    Process knowledge graph QA data
    """
    processed_data = []
    
    for item in tqdm(dataset, desc="Processing Knowledge Graph Data"):
        # 构造对话格式
        conversation = {
            "system": "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。",
            "instruction": item.get("question", ""),
            "output": item.get("answer", "")
        }
        processed_data.append(conversation)
    
    return processed_data

def process_encyclopedia_data(dataset):
    """
    处理百科全书问答数据
    Process encyclopedia QA data
    """
    processed_data = []
    
    for item in tqdm(dataset, desc="Processing Encyclopedia Data"):
        # 构造对话格式
        conversation = {
            "system": "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。",
            "instruction": item.get("question", ""),
            "output": item.get("answer", "")
        }
        processed_data.append(conversation)
    
    return processed_data

def process_consultation_data(dataset):
    """
    处理咨询问答数据
    Process consultation QA data
    """
    processed_data = []
    
    for item in tqdm(dataset, desc="Processing Consultation Data"):
        # 构造对话格式
        conversation = {
            "system": "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。",
            "instruction": item.get("question", ""),
            "output": item.get("answer", "")
        }
        processed_data.append(conversation)
    
    return processed_data

def process_test_datasets(dataset):
    """
    处理测试数据集
    Process test datasets
    """
    processed_data = []
    
    for item in tqdm(dataset, desc="Processing Test Datasets"):
        # 构造对话格式
        conversation = {
            "system": "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。",
            "instruction": item.get("question", ""),
            "output": item.get("answer", "")
        }
        processed_data.append(conversation)
    
    return processed_data

def save_to_jsonl(data: List[Dict], file_path: str):
    """
    保存数据为JSONL格式
    Save data to JSONL format
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(data)} items to {file_path}")

def main():
    """
    主函数：仅使用 huatuo_encyclopedia_qa 数据集
    Main function: only process huatuo_encyclopedia_qa
    """
    print("开始处理 Huatuo 百科问答数据...")

    # 创建数据目录
    data_dir = "./data/huatuo"
    os.makedirs(data_dir, exist_ok=True)

    all_processed_data = []

    # 处理百科全书数据集（单一数据源）
    print("加载并处理百科全书问答数据...")
    try:
        encyclopedia_dataset = load_dataset(
            "FreedomIntelligence/huatuo_encyclopedia_qa",
            cache_dir="./hf_cache"
        )
        train_data = encyclopedia_dataset.get("train", [])
        if train_data:
            enc_data = process_encyclopedia_data(train_data)
            all_processed_data.extend(enc_data)
            save_to_jsonl(enc_data, os.path.join(data_dir, "encyclopedia_train.jsonl"))
    except Exception as e:
        print(f"处理百科全书数据时出错: {e}")

    # 保存全部数据
    if all_processed_data:
        save_to_jsonl(all_processed_data, os.path.join(data_dir, "all_medical_data.jsonl"))

        # 划分训练集和验证集
        split_index = int(len(all_processed_data) * 0.9)
        train_data = all_processed_data[:split_index]
        val_data = all_processed_data[split_index:]

        save_to_jsonl(train_data, "./data/train_data.jsonl")
        save_to_jsonl(val_data, "./data/eval_data.jsonl")

        print(f"\n总计处理 {len(all_processed_data)} 条数据")
        print(f"训练集: {len(train_data)} 条")
        print(f"验证集: {len(val_data)} 条")
    else:
        print("没有成功处理任何数据")

    print("\n数据处理完成!")

if __name__ == "__main__":
    main()