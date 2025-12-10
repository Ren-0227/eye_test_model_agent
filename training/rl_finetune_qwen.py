#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用强化学习(ppo)微调千问模型
Fine-tune Qwen model using Reinforcement Learning (PPO)
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from datasets import Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import argparse

class RLFinetuneConfig:
    """强化学习微调配置类"""
    def __init__(self):
        # 基础模型配置，修改为本地模型路径
        self.model_name = "./Qwen-1_8B-Chat"
        self.output_dir = "./qwen_rl_finetuned_model"
        self.cache_dir = "./model_cache"
        
        # LoRA配置 (针对Qwen模型调整)
        # 为兼容 TRL PPO 的 value head 包装，默认关闭 LoRA（否则会生成 PeftModelForCausalLM，不被 PPOTrainer 接受）
        self.use_lora = False
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        # Qwen模型的注意力模块名称
        self.target_modules = ["c_attn"]  # Qwen使用c_attn而不是q_proj, k_proj, v_proj
        
        # PPO配置
        self.ppo_config = PPOConfig(
            model_name=self.model_name,
            learning_rate=1.41e-5,
            mini_batch_size=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            early_stopping=False,
            optimize_cuda_cache=True,
        )
        
        # 训练配置
        self.num_epochs = 3
        self.batch_size = 1
        self.max_length = 2048
        self.save_steps = 500
        
        # 量化配置（QLoRA）
        self.use_4bit = False  # 关闭量化，避免兼容性问题
        self.bnb_4bit_compute_dtype = "float16"
        self.bnb_4bit_quant_type = "nf4"
        self.use_nested_quant = False
        
        # 数据配置
        self.train_data_path = "./data/train_data.jsonl"
        self.eval_data_path = "./data/eval_data.jsonl"
        
        # 奖励模型（可选）路径，若存在则用于奖励打分
        self.reward_model_path = "./reward_model"


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


def format_instruction_data(data_item, tokenizer):
    """使用官方 chat template 生成 Qwen 对话格式"""
    system_msg = data_item.get(
        "system",
        "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。"
    )
    user_msg = data_item.get("instruction", data_item.get("input", ""))
    assistant_msg = data_item.get("output", data_item.get("response", ""))

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": formatted}


def prepare_dataset(config, tokenizer):
    """准备训练和验证数据集"""
    print("正在加载训练数据...")
    train_data = load_jsonl(config.train_data_path)
    
    if not train_data:
        print("未找到训练数据，请先准备数据文件")
        return None
    
    # 格式化数据
    train_formatted = [format_instruction_data(item, tokenizer) for item in train_data]
    
    # 创建 Dataset
    train_dataset = Dataset.from_list(train_formatted)
    
    return train_dataset


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
    
    # 方式2: 尝试专用的QWenTokenizer
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
        pad_token_id = None
        pad_token = None

        if getattr(tokenizer, "eos_token_id", None) is not None:
            pad_token_id = tokenizer.eos_token_id
            if isinstance(getattr(tokenizer, "eos_token", None), str):
                pad_token = tokenizer.eos_token
            else:
                token_str = tokenizer.convert_ids_to_tokens(pad_token_id)
                if isinstance(token_str, str):
                    pad_token = token_str

        if pad_token is None and isinstance(getattr(tokenizer, "unk_token", None), str):
            pad_token = tokenizer.unk_token
            pad_token_id = tokenizer.unk_token_id

        if pad_token is None and pad_token_id is not None:
            pad_token = str(pad_token_id)

        if pad_token is None:
            pad_token_id = 0
            token_str = tokenizer.convert_ids_to_tokens(pad_token_id)
            pad_token = token_str if isinstance(token_str, str) else "<|pad|>"

        tokenizer.pad_token = pad_token
        tokenizer.pad_token_id = pad_token_id
    
    tokenizer.padding_side = "right"

    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = """{% for message in messages -%}
{% if message['role'] == 'system' -%}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' -%}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' -%}<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif -%}
{% endfor -%}{% if add_generation_prompt -%}<|im_start|>assistant
{% endif -%}"""

    print(f"成功加载分词器: {type(tokenizer).__name__}")
    print(f"分词器特殊标记: pad_token={tokenizer.pad_token}, eos_token={tokenizer.eos_token}")
    
    # 加载模型
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
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
        # 包装为带有价值头的模型用于PPO训练
        model = AutoModelForCausalLMWithValueHead(base_model)
    except Exception as e:
        print(f"AutoModelForCausalLM加载失败: {e}")
    
    # 如果AutoModel加载失败，尝试直接使用Qwen模型类
    if model is None:
        try:
            print("尝试使用QWenLMHeadModel加载模型...")
            from transformers import QWenLMHeadModel
            base_model = QWenLMHeadModel.from_pretrained(
                config.model_name,
                **model_kwargs
            )
            # 包装为带有价值头的模型用于PPO训练
            model = AutoModelForCausalLMWithValueHead(base_model)
        except Exception as e:
            print(f"QWenLMHeadModel加载失败: {e}")
    
    if model is None:
        raise RuntimeError("无法加载模型，请检查模型路径和环境配置")
    
    print(f"成功加载模型: {type(model).__name__}")
    
    return model, tokenizer


def load_reward_model(config):
    """加载奖励模型（可选）。未找到则返回 (None, None)"""
    path = getattr(config, "reward_model_path", None)
    if not path or not os.path.isdir(path):
        print(f"未找到奖励模型目录: {path}，将使用固定奖励。")
        return None, None

    try:
        print(f"正在加载奖励模型分词器: {path}")
        reward_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if reward_tokenizer.pad_token is None:
            reward_tokenizer.pad_token = reward_tokenizer.eos_token
            reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id
        reward_tokenizer.padding_side = "right"

        adapter_cfg = os.path.join(path, "adapter_config.json")
        print(f"正在加载奖励模型: {path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if os.path.exists(adapter_cfg):
            print("检测到奖励模型 LoRA 适配器，正在合并...")
            reward_model = PeftModel.from_pretrained(base_model, path)
        else:
            reward_model = base_model

        reward_model.eval()
        return reward_model, reward_tokenizer
    except Exception as e:
        print(f"奖励模型加载失败，将使用固定奖励。原因: {e}")
        return None, None


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def train_model(config):
    """训练模型"""
    # 加载模型和分词器
    try:
        model, tokenizer = load_model_and_tokenizer(config)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 准备数据集（依赖 tokenizer 的 chat template）
    train_dataset = prepare_dataset(config, tokenizer)
    if train_dataset is None:
        return
    
    # 加载奖励模型（可选）
    reward_model, reward_tokenizer = load_reward_model(config)
    reward_device = next(reward_model.parameters()).device if reward_model is not None else None
    
    # 初始化PPO训练器
    ppo_trainer = PPOTrainer(
        config=config.ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
    )
    
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
    
    # 设置生成参数
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
    }
    
    # 训练循环
    for epoch in range(config.num_epochs):
        print(f"开始第 {epoch + 1} 轮训练...")
        
        for step, batch in enumerate(ppo_trainer.dataloader):
            try:
                # 编码查询
                encoded = tokenizer(
                    batch["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                query_tensors = encoded.input_ids.to(model.device)
                attention_mask = encoded.attention_mask.to(model.device)

                # 获取模型响应
                response_tensors = ppo_trainer.generate(
                    query_tensors,
                    attention_mask=attention_mask,
                    return_prompt=False,
                    **generation_kwargs,
                )

                # 解码响应
                batch_responses = tokenizer.batch_decode(
                    response_tensors,
                    skip_special_tokens=True,
                )

                # 计算奖励
                if reward_model is not None and reward_tokenizer is not None:
                    reward_texts = [
                        f"{prompt}{resp}" for prompt, resp in zip(batch["text"], batch_responses)
                    ]
                    with torch.no_grad():
                        reward_inputs = reward_tokenizer(
                            reward_texts,
                            padding=True,
                            truncation=True,
                            max_length=config.max_length,
                            return_tensors="pt",
                        ).to(reward_device)
                        reward_logits = reward_model(**reward_inputs).logits  # [B, T, V]
                        reward_vals = reward_logits.mean(dim=-1).mean(dim=-1)  # [B]
                    rewards = [r.detach() if torch.is_tensor(r) else torch.tensor(r) for r in reward_vals]
                else:
                    rewards = [torch.tensor(1.0) for _ in range(len(batch_responses))]

                # 运行PPO步
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                ppo_trainer.log_stats(stats, batch, rewards)

                # 保存模型
                if step % config.save_steps == 0:
                    save_path = os.path.join(config.output_dir, f"checkpoint-{step}")
                    ppo_trainer.save_pretrained(save_path)
                    print(f"模型已保存到: {save_path}")
            except Exception as e:
                print(f"训练过程中发生错误: {e}")
                continue
    
    # 保存最终模型
    ppo_trainer.save_pretrained(config.output_dir)
    print(f"最终模型已保存到: {config.output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用强化学习微调千问模型")
    parser.add_argument("--model_name", type=str, default="./Qwen-1_8B-Chat", help="基础模型名称或路径")
    parser.add_argument("--output_dir", type=str, default="./qwen_rl_finetuned_model", help="输出目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    
    args = parser.parse_args()
    
    # 创建配置对象
    config = RLFinetuneConfig()
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