#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查模型结构以确定合适的LoRA目标模块
Check model structure to determine appropriate LoRA target modules
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def check_model_structure(model_path="./Qwen-1_8B-Chat"):
    """检查模型结构并输出详细信息"""
    print(f"正在检查模型结构: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型路径 {model_path} 不存在")
        return
    
    try:
        # 加载分词器
        print("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False  # 避免快速分词器可能的兼容性问题
        )
        print(f"成功加载分词器: {type(tokenizer).__name__}")
        
        # 处理分词器特殊令牌（解决添加未知令牌问题）
        print("\n处理分词器特殊令牌...")
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                print(f"将 pad_token 设置为 eos_token: {tokenizer.eos_token}")
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                print("警告: 模型未定义 eos_token，无法设置 pad_token")
        
        # 输出关键令牌信息
        print("分词器关键令牌信息:")
        key_tokens = ["bos_token", "eos_token", "pad_token", "unk_token"]
        for token in key_tokens:
            token_value = getattr(tokenizer, token, None)
            token_id = getattr(tokenizer, f"{token}_id", None)
            print(f"  {token}: {token_value} (ID: {token_id})")
        
        # 加载模型
        print("\n加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True  # 减少CPU内存占用
        )
        print(f"成功加载模型: {type(model).__name__}")
        print(f"模型设备: {next(model.parameters()).device}")
        
        # 打印模型结构摘要（简化输出，避免过长）
        print("\n模型结构摘要 (简化):")
        print(model.__class__.__name__)
        print("  主要子模块:")
        for name, module in list(model.named_children())[:5]:  # 显示前5个子模块
            print(f"    {name}: {type(module).__name__}")
        if len(list(model.named_children())) > 5:
            print(f"    ... 还有 {len(list(model.named_children())) - 5} 个子模块")
        
        # 查找注意力相关模块
        print("\n查找注意力相关模块...")
        target_modules = set()
        
        def find_attention_modules(module, prefix=""):
            for name, submodule in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                # 检查模块名称是否包含注意力相关关键词
                if any(keyword in name.lower() for keyword in ["attn", "attention", "proj", "qkv", "c_attn"]):
                    target_modules.add(full_name)
                    print(f"  找到可能的目标模块: {full_name} ({type(submodule).__name__})")
                # 递归查找子模块（限制深度避免过深）
                if len(full_name.split('.')) < 6:  # 限制最大深度为5
                    find_attention_modules(submodule, full_name)
        
        find_attention_modules(model)
        
        print(f"\n建议的LoRA目标模块 ({len(target_modules)}个):")
        for i, module in enumerate(sorted(target_modules), 1):
            print(f"  {i}. {module}")
        
        # 针对Qwen模型的特别提示
        if "QWen" in type(model).__name__:
            print("\n注意: Qwen系列模型通常推荐使用 'c_attn' 作为LoRA目标模块")
            
        # 显示模型的基本信息
        print(f"\n模型基本信息:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数量: {total_params:,} ({total_params / 1e9:.2f}B)")
        print(f"  可训练参数量: {trainable_params:,} ({trainable_params / total_params:.2%})")
        
        # 检查模型配置
        if hasattr(model, 'config'):
            print(f"  模型类型: {getattr(model.config, 'model_type', 'unknown')}")
            print(f"  隐藏层大小: {getattr(model.config, 'hidden_size', 'unknown')}")
            print(f"  注意力头数: {getattr(model.config, 'num_attention_heads', 'unknown')}")
            print(f"  层数: {getattr(model.config, 'num_hidden_layers', 'unknown')}")
            print(f"  最大序列长度: {getattr(model.config, 'max_sequence_length', getattr(model.config, 'max_position_embeddings', 'unknown'))}")
            
    except Exception as e:
        print(f"检查模型结构时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 支持命令行参数指定模型路径
    import argparse
    parser = argparse.ArgumentParser(description="检查模型结构以确定LoRA目标模块")
    parser.add_argument("--model_path", type=str, default="./Qwen-1_8B-Chat", help="模型路径或名称")
    args = parser.parse_args()
    check_model_structure(args.model_path)