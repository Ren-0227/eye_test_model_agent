"""
Local Qwen model client.
Gracefully handles missing weights: if model files are absent, returns a clear message.
"""
import os
# 在导入transformers之前，禁用TensorFlow导入以避免DLL加载错误
os.environ["TRANSFORMERS_NO_TF"] = "1"  # 禁用TensorFlow后端
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 抑制TensorFlow日志

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
        
        # 在Windows上设置环境变量，优化文件访问
        if os.name == 'nt':  # Windows
            # 禁用mmap以提高文件访问稳定性
            os.environ.setdefault("HF_HUB_DISABLE_MMAP", "1")
            # 使用单线程加载避免文件句柄冲突
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        
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
        """自动选择模型路径，优先选择微调模型"""
        for path in DEFAULT_MODEL_CANDIDATES:
            # 转换为绝对路径
            abs_path = os.path.abspath(os.path.normpath(path)) if os.path.exists(path) else path
            if os.path.isdir(abs_path):
                # 检查是否是微调模型（有adapter_config.json）
                adapter_config = os.path.join(abs_path, "adapter_config.json")
                if os.path.exists(adapter_config):
                    print(f"[model] 检测到微调模型: {abs_path}")
                    # 微调模型需要基础模型，检查基础模型是否存在
                    # 如果微调模型目录本身有config.json，说明是完整模型
                    if os.path.exists(os.path.join(abs_path, "config.json")):
                        if self._has_weights(abs_path):
                            print(f"[model] 选择微调模型（完整）: {abs_path}")
                            return abs_path
                    # 否则需要基础模型，返回微调模型路径（基础模型会在加载时自动选择）
                    if self._has_weights(abs_path) or os.path.exists(adapter_config):
                        print(f"[model] 选择微调模型（LoRA）: {abs_path}")
                        return abs_path
                # 检查是否有完整权重文件
                elif self._has_weights(abs_path):
                    print(f"[model] 选择基础模型: {abs_path}")
                    return abs_path
        print(f"[model] 未找到可用模型，候选路径: {DEFAULT_MODEL_CANDIDATES}")
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
            
        # 转换为绝对路径，避免Windows相对路径问题
        model_path_abs = os.path.abspath(os.path.normpath(self.model_path))
        
        if not os.path.exists(model_path_abs):
            self.load_error = f"本地模型未就绪：路径不存在: {model_path_abs}"
            return
            
        if not self._has_weights(model_path_abs):
            self.load_error = (
                f"本地模型未就绪：在路径 {model_path_abs} 中未找到权重文件。"
                f"请将模型权重放到该路径（需要 model-*.safetensors 或 pytorch_model*.bin 和 tokenizer 文件），"
                f"或运行微调脚本生成 ./qwen_finetuned_model。"
            )
            return
            
        try:
            print(f"正在加载模型分词器，路径: {model_path_abs}")
            
            # 在Windows上，先检查文件是否可访问
            if os.name == 'nt':
                tokenizer_files = [
                    os.path.join(model_path_abs, "tokenizer.json"),
                    os.path.join(model_path_abs, "tokenizer_config.json"),
                    os.path.join(model_path_abs, "vocab.json"),
                ]
                for tf in tokenizer_files:
                    if os.path.exists(tf):
                        try:
                            # 尝试以只读模式打开文件，检查是否被锁定
                            with open(tf, 'rb') as test_f:
                                test_f.read(1)
                        except (OSError, IOError) as e:
                            if "WinError 6" in str(e) or "句柄无效" in str(e):
                                raise OSError(
                                    f"无法访问分词器文件 {tf}，文件可能被其他进程锁定。\n"
                                    f"请关闭可能使用该文件的程序（如其他Python进程、文件管理器、IDE等），然后重试。\n"
                                    f"原始错误: {e}"
                                ) from e
            
            # 在Windows上，确保路径格式正确，并使用本地文件系统标志
            # 使用绝对路径避免相对路径导致的文件句柄问题
            tokenizer_path = model_path_abs
            
            # 加载tokenizer，在Windows上使用特殊配置
            tokenizer_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True  # 强制使用本地文件
            }
            
            # Windows特殊处理
            if os.name == 'nt':
                tokenizer_kwargs["use_fast"] = False  # 避免快速tokenizer的文件句柄问题
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, 
                **tokenizer_kwargs
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # 确定基础模型名称
            base_name = self.base_model_name
            # 如果本地目录就是基座权重，则直接加载
            if self._has_weights(model_path_abs) and os.path.exists(
                os.path.join(model_path_abs, "config.json")
            ):
                base_name = model_path_abs

            print(f"正在加载基础模型: {base_name}")
            
            # 确保base_name也是绝对路径
            if os.path.isabs(base_name) or os.path.exists(base_name):
                base_name_abs = os.path.abspath(os.path.normpath(base_name))
            else:
                base_name_abs = base_name  # 如果是HuggingFace模型名称，保持原样
            
            # 加载模型，在Windows上使用特殊配置
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            # 设备映射
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = None
            
            # Windows特殊处理：禁用mmap避免文件句柄问题
            if os.name == 'nt' and os.path.exists(base_name_abs):
                model_kwargs["local_files_only"] = True
                # 尝试使用low_cpu_mem_usage减少内存占用和文件锁定
                try:
                    model_kwargs["low_cpu_mem_usage"] = True
                except:
                    pass
            elif os.path.exists(base_name_abs):
                model_kwargs["local_files_only"] = True
            
            # 在Windows上，先检查模型文件是否可访问
            if os.name == 'nt' and os.path.exists(base_name_abs):
                # 测试关键文件访问权限
                test_files = [
                    os.path.join(base_name_abs, "config.json"),
                ]
                # 检查权重文件
                weight_patterns = [
                    os.path.join(base_name_abs, "model-*.safetensors"),
                    os.path.join(base_name_abs, "pytorch_model*.bin"),
                ]
                import glob
                for pattern in weight_patterns:
                    test_files.extend(glob.glob(pattern)[:1])  # 只检查第一个匹配的文件
                
                for test_file in test_files:
                    if os.path.exists(test_file):
                        try:
                            # 尝试以只读模式打开文件，检查是否被锁定
                            with open(test_file, 'rb') as f:
                                f.read(1)  # 尝试读取一个字节
                        except (OSError, IOError) as e:
                            if "WinError 6" in str(e) or "句柄无效" in str(e):
                                raise OSError(
                                    f"无法访问模型文件 {test_file}，文件可能被其他进程锁定。\n"
                                    f"请执行以下操作：\n"
                                    f"1. 关闭所有可能使用该文件的程序（其他Python进程、文件管理器、IDE等）\n"
                                    f"2. 如果使用文件管理器，请关闭包含该文件夹的窗口\n"
                                    f"3. 重启IDE或终端，然后重试\n"
                                    f"4. 如果问题持续，尝试重启系统\n"
                                    f"原始错误: {e}"
                                ) from e
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_name_abs,
                **model_kwargs
            )

            # 如果存在 PEFT 适配权重则尝试加载
            adapter_config_path = os.path.join(model_path_abs, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                print(f"检测到 LoRA 适配器配置，正在加载: {model_path_abs}")
                self.model = PeftModel.from_pretrained(base_model, model_path_abs)
            else:
                print("未检测到 LoRA 适配器，使用基础模型")
                self.model = base_model

            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
            self.load_error = None
            print("模型加载成功!")
        except OSError as e:
            # Windows特定的文件句柄错误
            if "WinError 6" in str(e) or "句柄无效" in str(e):
                error_msg = (
                    f"Windows文件句柄错误：模型路径 {model_path_abs} 可能被其他进程占用，"
                    f"或路径格式不正确。请确保：\n"
                    f"1. 模型文件未被其他程序打开\n"
                    f"2. 路径不包含特殊字符\n"
                    f"3. 有足够的文件访问权限\n"
                    f"原始错误: {str(e)}"
                )
                self.load_error = error_msg
                print(f"模型加载失败: {self.load_error}")
            else:
                self.load_error = f"本地模型加载失败：{str(e)}"
                print(f"模型加载失败: {self.load_error}")
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
