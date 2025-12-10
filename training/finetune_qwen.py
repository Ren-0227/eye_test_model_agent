def build_data_collator(tokenizer, max_length):
    """构建可 pickling 的 collator，避免多进程下的本地函数问题"""
    def collate(features):
        texts = [f["text"] for f in features]
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = encodings["input_ids"].clone()
        encodings["labels"] = labels
        return encodings

    return collate

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
from tqdm import tqdm
import argparse
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
class QwenFinetuneConfig:
    """千问微调配置类"""
    def __init__(self):
        # 修改为本地模型路径
        self.model_name = "./Qwen-1_8B-Chat"  
        self.output_dir = "./qwen_finetuned_model"
        self.cache_dir = "./model_cache"
        self.use_lora = True
        self.lora_r = 16  # LoRA rank
        self.lora_alpha = 32  # LoRA alpha
        self.lora_dropout = 0.05
        self.target_modules = ["c_attn"]  # Qwen-1.8B 使用 c_attn 来统一处理查询、键、值的投影
        
        # 训练配置
        self.num_epochs = 3
        self.batch_size = 4
        self.gradient_accumulation_steps = 4
        self.learning_rate = 2e-4
        self.warmup_steps = 100
        self.max_length = 2048
        self.save_steps = 500
        self.eval_steps = 500
        self.logging_steps = 100
        
        # 量化配置（QLoRA）
        self.use_4bit = False  # 关闭量化，避免兼容性问题
        self.bnb_4bit_compute_dtype = "float16"
        self.bnb_4bit_quant_type = "nf4"
        self.use_nested_quant = False
        
        # 数据配置
        self.train_data_path = "./data/train_data.jsonl"
        self.eval_data_path = "./data/eval_data.jsonl"


# ==================== 数据加载 ====================
def load_jsonl(file_path):
    """加载 JSONL 格式的训练数据"""
    data = []
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在，将创建示例数据")
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
    # add_generation_prompt=False -> 包含完整标签，适合监督微调
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
    eval_data = load_jsonl(config.eval_data_path) if os.path.exists(config.eval_data_path) else []
    
    if not train_data:
        print("未找到训练数据，请先准备数据文件")
        return None, None
    
    # 格式化数据
    train_formatted = [format_instruction_data(item, tokenizer) for item in train_data]
    eval_formatted = [format_instruction_data(item, tokenizer) for item in eval_data] if eval_data else None
    
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
    
    # 设置 pad_token - 对齐 Qwen Chat 模板
    if tokenizer.pad_token is None:
        print("正在为分词器设置pad_token...")
        # 优先使用已有特殊符号，避免添加新 token（Qwen 老版本可能不支持）
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

        # 如果仍然没有字符串形式，回退为基于 id 的字符串表示
        if pad_token is None and pad_token_id is not None:
            pad_token = str(pad_token_id)

        # 最后兜底：直接用 id 0 的 token 字符串
        if pad_token is None:
            pad_token_id = 0
            token_str = tokenizer.convert_ids_to_tokens(pad_token_id)
            pad_token = token_str if isinstance(token_str, str) else "<|pad|>"

        tokenizer.pad_token = pad_token
        tokenizer.pad_token_id = pad_token_id
    else:
        # 如果已有 pad_token 但缺少 pad_token_id，兜底设为 0
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0
    
    # 统一确保 pad_token_id 有效且非负，pad_token 为字符串
    pid = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if pid is None or (isinstance(pid, int) and pid < 0):
        # 优先用 eos/eod，再兜底 0
        pid = getattr(tokenizer, "eod_id", None)
        if pid is None or (isinstance(pid, int) and pid < 0):
            pid = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    if not isinstance(pid, int):
        pid = 0
    if pid < 0:
        pid = 0
    tokenizer.pad_token_id = pid

    if not isinstance(tokenizer.pad_token, str):
        token_str = tokenizer.convert_ids_to_tokens(pid)
        tokenizer.pad_token = token_str if isinstance(token_str, str) else str(tokenizer.pad_token)

    # 确保右侧 padding，适配 causal LM
    tokenizer.padding_side = "right"

    # 若无 chat_template，则设置一个适用于 Qwen-Chat 格式的模板
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


class LossPlotCallback(TrainerCallback):
    """训练loss记录和可视化回调"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.train_losses = []
        self.eval_losses = []
        self.log_steps = []
        self.eval_steps = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录训练loss"""
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.log_steps.append(state.global_step)
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                self.eval_steps.append(state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时绘制loss图"""
        if len(self.train_losses) > 0:
            plot_finetune_loss_curves(
                self.train_losses, 
                self.log_steps,
                self.eval_losses if len(self.eval_losses) > 0 else None,
                self.eval_steps if len(self.eval_steps) > 0 else None,
                self.output_dir
            )


def plot_finetune_loss_curves(train_losses, train_steps, eval_losses=None, eval_steps=None, output_dir=None):
    """
    绘制微调训练loss曲线图
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制训练loss
    plt.plot(train_steps, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    
    # 如果有验证loss，也绘制
    if eval_losses is not None and len(eval_losses) > 0:
        plt.plot(eval_steps, eval_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8, marker='o', markersize=4)
    
    plt.title('Qwen Fine-tuning - Training Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    if output_dir:
        loss_plot_path = os.path.join(output_dir, "finetune_loss_curve.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        print(f"微调Loss曲线图已保存到: {loss_plot_path}")
    else:
        plt.savefig("finetune_loss_curve.png", dpi=300, bbox_inches='tight')
        print("微调Loss曲线图已保存到: finetune_loss_curve.png")
    
    plt.close()


def train_model(config):
    """训练模型"""
    # 加载模型和分词器
    try:
        model, tokenizer = load_model_and_tokenizer(config)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 准备数据集（依赖 tokenizer 的 chat template）
    train_dataset, eval_dataset = prepare_dataset(config, tokenizer)
    if train_dataset is None:
        return
    
    # 数据整理器：放在顶层函数外，避免 Windows 多进程 pickling 问题
    data_collator = build_data_collator(tokenizer, config.max_length)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
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
        dataloader_num_workers=0,  # Windows 上避免多进程 pickling 问题
        save_total_limit=3,
        report_to="none",  # 避免触发 tensorboard 依赖
        remove_unused_columns=False,  # 保留 text 列，交给 collator/tokenizer 处理
    )
    
    # 创建loss可视化回调
    loss_callback = LossPlotCallback(config.output_dir)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[loss_callback],  # 添加回调
    )
    
    # 开始训练
    print("开始训练模型...")
    trainer.train()
    
    # 保存模型
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"模型已保存到: {config.output_dir}")
    
    # 如果回调没有自动绘制（备用方案），从训练历史中提取并绘制
    if hasattr(trainer.state, 'log_history') and len(trainer.state.log_history) > 0:
        try:
            train_losses = []
            train_steps = []
            eval_losses = []
            eval_steps = []
            
            for log in trainer.state.log_history:
                if 'loss' in log and 'step' in log:
                    train_losses.append(log['loss'])
                    train_steps.append(log['step'])
                if 'eval_loss' in log and 'step' in log:
                    eval_losses.append(log['eval_loss'])
                    eval_steps.append(log['step'])
            
            if len(train_losses) > 0:
                plot_finetune_loss_curves(
                    train_losses, 
                    train_steps,
                    eval_losses if len(eval_losses) > 0 else None,
                    eval_steps if len(eval_steps) > 0 else None,
                    config.output_dir
                )
        except Exception as e:
            print(f"从训练历史提取loss失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用LoRA微调千问模型")
    parser.add_argument("--model_name", type=str, default="./Qwen-1_8B-Chat", help="基础模型名称或路径")
    parser.add_argument("--output_dir", type=str, default="./qwen_finetuned_model", help="输出目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    
    args = parser.parse_args()
    
    # 创建配置对象
    config = QwenFinetuneConfig()
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