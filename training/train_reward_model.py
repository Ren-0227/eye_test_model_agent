#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练奖励模型用于强化学习微调
Train reward model for reinforcement learning fine-tuning
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset
import argparse
from typing import Dict, List

class RewardModelConfig:
    """奖励模型配置类"""
    def __init__(self):
        # 修改为本地模型路径
        self.model_name = "./Qwen-1_8B-Chat"
        self.output_dir = "./reward_model"
        self.cache_dir = "./model_cache"
        
        # LoRA配置 (针对Qwen模型调整)
        self.use_lora = True
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        # Qwen模型使用c_attn合并了q, k, v投影
        self.target_modules = ["c_attn"]
        
        # 训练配置
        self.num_epochs = 3
        self.batch_size = 4
        self.learning_rate = 2e-5
        self.warmup_steps = 100
        self.max_length = 2048
        self.save_steps = 500
        self.eval_steps = 500
        self.logging_steps = 100
        
        # 量化配置 - 关闭4位量化以避免Qwen模型的兼容性问题
        self.use_4bit = False
        self.bnb_4bit_compute_dtype = "float16"
        self.bnb_4bit_quant_type = "nf4"
        self.use_nested_quant = False
        
        # 数据配置
        self.train_data_path = "./data/reward_data.jsonl"
        self.eval_data_path = "./data/reward_eval_data.jsonl"


def load_jsonl(file_path):
    """加载 JSONL 格式的训练数据"""
    data = []
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_reward_data(data_item):
    """格式化奖励数据"""
    # 奖励数据应该包含以下字段:
    # - prompt: 用户的输入
    # - chosen: 优质回答
    # - rejected: 劣质回答
    # - chosen_score: 优质回答得分
    # - rejected_score: 劣质回答得分
    
    return {
        "prompt": data_item.get("prompt", ""),
        "chosen": data_item.get("chosen", ""),
        "rejected": data_item.get("rejected", ""),
        "chosen_score": data_item.get("chosen_score", 1.0),
        "rejected_score": data_item.get("rejected_score", 0.0)
    }


def prepare_dataset(config):
    """准备训练和验证数据集"""
    print("正在加载训练数据...")
    train_data = load_jsonl(config.train_data_path)
    eval_data = load_jsonl(config.eval_data_path) if os.path.exists(config.eval_data_path) else []
    
    if not train_data:
        print("未找到训练数据，请先准备数据文件")
        return None, None
    
    # 格式化数据
    train_formatted = [format_reward_data(item) for item in train_data]
    eval_formatted = [format_reward_data(item) for item in eval_data] if eval_data else None
    
    # 创建 Dataset
    train_dataset = Dataset.from_list(train_formatted)
    eval_dataset = Dataset.from_list(eval_formatted) if eval_formatted else None
    
    return train_dataset, eval_dataset


def load_model_and_tokenizer(config):
    """加载模型和分词器"""
    print(f"正在加载模型: {config.model_name}")
    
    # 尝试多种方式加载分词器
    tokenizer = None
    
    # 方式1: 直接使用AutoTokenizer
    try:
        print("尝试使用AutoTokenizer加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            cache_dir=config.cache_dir
        )
    except Exception as e:
        print(f"AutoTokenizer加载失败: {e}")
    
    # 尝试使用QWenTokenizer加载分词器 (正确类名)
    if tokenizer is None:
        try:
            print("尝试使用QWenTokenizer加载分词器...")
            from transformers import QWenTokenizer
            tokenizer = QWenTokenizer.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                cache_dir=config.cache_dir
            )
        except Exception as e:
            print(f"QWenTokenizer加载失败: {e}")
    
    # 如果仍然没有成功加载分词器
    if tokenizer is None:
        raise RuntimeError("无法加载分词器，请检查模型路径和环境配置")
    
    # 设置 pad_token - 改进的处理方式
    if tokenizer.pad_token is None:
        # 对于Qwen模型，我们使用特殊方法设置pad_token
        print("正在为分词器设置pad_token...")
        # 使用vocab中的特殊token或者添加新token
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif hasattr(tokenizer, 'unk_token') and tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            # 最后手段：使用vocab中的第一个token作为pad_token
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
            tokenizer.pad_token_id = 0
    
    print(f"成功加载分词器: {type(tokenizer).__name__}")
    print(f"分词器特殊标记: pad_token={tokenizer.pad_token}, eos_token={tokenizer.eos_token}")
    
    # 加载因果语言模型（用于奖励模型）
    model_kwargs = {
        "trust_remote_code": True,
        "cache_dir": config.cache_dir,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    
    # 检查是否需要量化配置
    if config.use_4bit and config.use_lora:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=config.use_4bit,
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=config.use_nested_quant,
            )
            model_kwargs["quantization_config"] = bnb_config
        except Exception as e:
            print(f"警告: 量化配置加载失败: {e}")
            print("将继续使用非量化配置...")
    
    model = None
    # 尝试多种方式加载模型
    try:
        print("尝试使用AutoModelForCausalLM加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
    except Exception as e:
        print(f"AutoModelForCausalLM加载失败: {e}")
    
    # 如果AutoModel加载失败，尝试直接使用Qwen模型类
    if model is None:
        try:
            print("尝试使用QWenLMHeadModel加载模型...")
            from transformers import QWenLMHeadModel
            model = QWenLMHeadModel.from_pretrained(
                config.model_name,
                **model_kwargs
            )
        except Exception as e:
            print(f"QWenLMHeadModel加载失败: {e}")
    
    if model is None:
        raise RuntimeError("无法加载模型，请检查模型路径和环境配置")
    
    print(f"成功加载模型: {type(model).__name__}")
    
    # 应用 LoRA
    if config.use_lora:
        print("正在应用 LoRA 配置...")
        try:
            if config.use_4bit:
                model = prepare_model_for_kbit_training(model)
        except Exception as e:
            print(f"警告: prepare_model_for_kbit_training 失败: {e}")
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def train_model(config):
    """训练奖励模型"""
    # 准备数据集
    train_dataset, eval_dataset = prepare_dataset(config)
    if train_dataset is None:
        return
    
    # 加载模型和分词器
    try:
        model, tokenizer = load_model_and_tokenizer(config)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 数据整理器
    def tokenize_function(examples):
        # 对prompt和response进行编码
        chosen_texts = [f"{prompt}\n{chosen}" for prompt, chosen in zip(examples["prompt"], examples["chosen"])]
        rejected_texts = [f"{prompt}\n{rejected}" for prompt, rejected in zip(examples["prompt"], examples["rejected"])]
        
        # 确保分词器有pad_token
        if tokenizer.pad_token is None:
            # 对于Qwen模型，我们使用特殊方法设置pad_token
            print("正在为分词器设置pad_token...")
            # 使用vocab中的特殊token或者添加新token
            if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            elif hasattr(tokenizer, 'unk_token') and tokenizer.unk_token:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
            else:
                # 最后手段：使用vocab中的第一个token作为pad_token
                tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
                tokenizer.pad_token_id = 0
        
        chosen_encodings = tokenizer(
            chosen_texts,
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
            return_tensors="pt"
        )
        
        rejected_encodings = tokenizer(
            rejected_texts,
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids_chosen": chosen_encodings["input_ids"],
            "attention_mask_chosen": chosen_encodings["attention_mask"],
            "input_ids_rejected": rejected_encodings["input_ids"],
            "attention_mask_rejected": rejected_encodings["attention_mask"],
            "margin": [c - r for c, r in zip(examples["chosen_score"], examples["rejected_score"])]
        }
    
    # 应用数据预处理
    try:
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        if eval_dataset:
            eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    except Exception as e:
        print(f"数据预处理失败: {e}")
        return
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if eval_dataset else None,
        logging_steps=config.logging_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        save_total_limit=3,
        report_to=None,
        remove_unused_columns=False,
    )
    
    # 自定义训练器用于奖励模型
    class RewardTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # 分别计算chosen和rejected的奖励分数
            rewards_chosen = model(
                input_ids=inputs["input_ids_chosen"],
                attention_mask=inputs["attention_mask_chosen"],
            ).logits.mean(dim=-1)  # 使用logits的平均值作为奖励分数
            
            rewards_rejected = model(
                input_ids=inputs["input_ids_rejected"],
                attention_mask=inputs["attention_mask_rejected"],
            ).logits.mean(dim=-1)  # 使用logits的平均值作为奖励分数
            
            # 计算损失：确保chosen的奖励高于rejected的奖励至少margin
            loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"].unsqueeze(1)).mean()
            
            return (loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}) if return_outputs else loss
    
    # 创建训练器
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("开始训练奖励模型...")
    trainer.train()
    
    # 保存模型
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"奖励模型已保存到: {config.output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练奖励模型用于强化学习微调")
    parser.add_argument("--model_name", type=str, default="./Qwen-1_8B-Chat", help="基础模型名称或路径")
    parser.add_argument("--output_dir", type=str, default="./reward_model", help="输出目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    
    args = parser.parse_args()
    
    # 创建配置对象
    config = RewardModelConfig()
    config.model_name = args.model_name
    config.output_dir = args.output_dir
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    
    # 检查模型路径是否存在
    if not os.path.exists(config.model_name):
        print(f"错误: 模型路径 {config.model_name} 不存在")
        return
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 开始训练
    train_model(config)


if __name__ == "__main__":
    main()