"""
Local Qwen model client.
Gracefully handles missing weights: if model files are absent, returns a clear message.
"""
import os
import glob
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DEFAULT_MODEL_CANDIDATES = [
    "./qwen_finetuned_model",     # 监督微调默认输出目录（优先使用）
    "./qwen_rl_finetuned_model",  # PPO/强化学习微调产物（备用）
    "./Qwen-1_8B-Chat",           # 根目录存放的千问 1.8B 模型
    "./Qwen1.8B-Chat",            # 兼容旧命名
    "./Qwen1.8B",                 # 可能的未区分 Chat 目录
    os.path.expanduser("~/.cache/modelscope/hub/models/qwen/Qwen-1_8B-Chat"),  # ModelScope 下载默认路径
    "./Qwen2-1.5B-Instruct",      # 兼容之前下载的轻量版
    "./Qwen2-7B-Instruct",        # 兼容旧配置
]


class LocalQwenAPI:
    def __init__(self, model_path: Optional[str] = None, base_model_name: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path or self._auto_pick_model_path()
        self.base_model_name = base_model_name or "Qwen/Qwen-1_8B-Chat"
        self.model = None
        self.tokenizer = None
        self.load_error = None
        self._maybe_load()

    def run_smoke_test(self, prompt: str = "请用一句话自我介绍。") -> Dict[str, Any]:
        """
        进行一次简易推理测试，验证模型与分词器是否可用。
        返回 {"status": "ok", "answer": "..."} 或 {"status": "error", "message": "..."}。
        """
        if self.load_error:
            return {"status": "error", "message": self.load_error}
        try:
            # 构造简短对话
            chat_prompt = (
                f"<|im_start|>system\n你是一名友好的助手。<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            inputs = self.tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            resp = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            if "<|im_start|>assistant\n" in resp:
                resp = resp.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
            return {"status": "ok", "answer": resp}
        except Exception as e:
            return {"status": "error", "message": f"推理测试失败: {e}"}

    def _auto_pick_model_path(self) -> Optional[str]:
        for path in DEFAULT_MODEL_CANDIDATES:
            if os.path.isdir(path) and self._has_weights(path):
                return path
        return None

    @staticmethod
    def _has_weights(path: str) -> bool:
        # 检查是否存在至少一个权重分片
        if not os.path.exists(path):
            return False
            
        # 检查 safetensors 格式权重
        safetensors_parts = glob.glob(os.path.join(path, "model*.safetensors"))
        if len(safetensors_parts) > 0:
            return True
            
        # 检查 pytorch 格式权重
        pytorch_parts = glob.glob(os.path.join(path, "pytorch_model*.bin"))
        if len(pytorch_parts) > 0:
            return True
            
        # 检查单个文件权重
        single_pytorch = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(single_pytorch):
            return True
            
        return False

    def _maybe_load(self):
        if not self.model_path:
            self.load_error = (
                f"本地模型未就绪：未找到有效的模型路径。请确保模型权重存在于以下候选路径之一: "
                f"{', '.join(DEFAULT_MODEL_CANDIDATES)}"
            )
            return
            
        if not self._has_weights(self.model_path):
            self.load_error = (
                f"本地模型未就绪：在路径 {self.model_path} 中未找到权重文件。"
                f"请将模型权重放到该路径（需要 model-*.safetensors 或 pytorch_model*.bin 和 tokenizer 文件），"
                f"或运行微调脚本生成 ./qwen_finetuned_model。"
            )
            return
            
        try:
            print(f"正在加载模型分词器，路径: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # 确定基础模型名称
            base_name = self.base_model_name
            # 如果本地目录就是基座权重，则直接加载
            if self._has_weights(self.model_path) and os.path.exists(
                os.path.join(self.model_path, "config.json")
            ):
                base_name = self.model_path

            print(f"正在加载基础模型: {base_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )

            # 如果存在 PEFT 适配权重则尝试加载
            adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                print(f"检测到 LoRA 适配器配置，正在加载: {self.model_path}")
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
            else:
                print("未检测到 LoRA 适配器，使用基础模型")
                self.model = base_model

            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
            self.load_error = None
            print("模型加载成功!")
        except Exception as e:
            self.load_error = f"本地模型加载失败：{str(e)}"
            print(f"模型加载失败: {self.load_error}")

    def get_health_advice(self, symptoms, vision_result=None, oct_result=None):
        if self.load_error:
            return {"status": "error", "message": self.load_error}

        system_msg = "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。"
        context_parts = []
        if vision_result:
            context_parts.append(f"视力检测结果：{vision_result}")
        if oct_result:
            context_parts.append(f"OCT检查结果：{oct_result}")

        user_input = str(symptoms)
        if context_parts:
            user_input = f"{user_input}\n\n" + "；".join(context_parts)

        prompt = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        # 对于千问模型，我们使用正确的模板格式
        if "qwen" in self.model_path.lower() if self.model_path else "qwen" in self.base_model_name.lower():
            prompt = (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\n{user_input}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            # 保持原有的格式，移除反斜杠避免转义错误
            prompt = (
                f"system\n{system_msg}\n"
                f"user\n{user_input}\n"
                "assistant\n"
            )
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            if "<|im_start|>assistant\n" in full_response:
                resp = full_response.split("<|im_start|>assistant\n")[-1]
                resp = resp.split("<|im_end|>")[0].strip()
            elif "assistant\n" in full_response:
                resp = full_response.split("assistant\n")[-1]
                resp = resp.split("\\")[0].strip()
            else:
                resp = full_response[len(prompt):].strip()

            return {"status": "ok", "answer": resp}
        except Exception as e:
            error_msg = f"模型推理失败: {str(e)}"
            return {"status": "error", "message": error_msg}