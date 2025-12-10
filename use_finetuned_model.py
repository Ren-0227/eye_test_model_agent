"""
使用本地千问模型的 API 类 (已弃用)
==============================

注意: 此文件已被弃用，请使用 backend.tools.local_qwen_api.LocalQwenAPI 替代。

此文件保留是为了向后兼容，但强烈建议迁移到统一的接口实现:
from backend.tools.local_qwen_api import LocalQwenAPI

该统一接口提供了更好的错误处理、模型路径自动检测等功能。
"""

import warnings
from backend.tools.local_qwen_api import LocalQwenAPI as NewLocalQwenAPI

# 发出弃用警告
warnings.warn(
    "use_finetuned_model.py 已弃用，请使用 backend.tools.local_qwen_api.LocalQwenAPI 替代",
    DeprecationWarning,
    stacklevel=2
)

# 为了向后兼容，保留类名不变
class LocalQwenAPI(NewLocalQwenAPI):
    """使用本地千问模型的 API 类 (已弃用，请使用 backend.tools.local_qwen_api.LocalQwenAPI)"""
    
    def __init__(self, model_path="./qwen_finetuned_model", base_model_name="Qwen/Qwen2-7B-Instruct"):
        """
        初始化本地模型 (已弃用接口)
        
        Args:
            model_path: 微调模型保存路径
            base_model_name: 基础模型名称
        """
        warnings.warn(
            "LocalQwenAPI 已弃用，请使用 backend.tools.local_qwen_api.LocalQwenAPI 替代",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(model_path=model_path, base_model_name=base_model_name)

    def _format_prompt(self, user_input, system_msg=None, vision_result=None, oct_result=None):
        """格式化输入为 Qwen2-Instruct 格式"""
        if system_msg is None:
            system_msg = "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。"
        
        # 构建上下文信息
        context_parts = []
        if vision_result:
            context_parts.append(f"视力检测结果：{vision_result}")
        if oct_result:
            context_parts.append(f"OCT检查结果：{oct_result}")
        
        if context_parts:
            user_input = f"{user_input}\n\n{'；'.join(context_parts)}"
        
        # Qwen2-Instruct 格式
        prompt = f"system\n{system_msg}\n"
        prompt += f"user\n{user_input}\n"
        prompt += "assistant\n"
        
        return prompt
    
    def get_health_advice(self, symptoms, vision_result=None, oct_result=None):
        """
        获取健康建议（兼容原有 API 接口）
        
        Args:
            symptoms: 患者症状描述
            vision_result: 视力检测结果（可选）
            oct_result: OCT检查结果（可选）
        
        Returns:
            诊断建议文本
        """
        try:
            # 格式化输入
            prompt = self._format_prompt(
                str(symptoms),
                vision_result=vision_result,
                oct_result=oct_result
            )
            
            # 编码
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # 生成回复
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
            
            # 解码
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # 提取助手回复部分
            if "assistant\n" in full_response:
                response = full_response.split("assistant\n")[-1]
                response = response.split("")[0].strip()
            else:
                # 如果没有找到标记，返回完整响应（去除输入部分）
                response = full_response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            return f"模型推理失败: {str(e)}"
    
    def chat(self, user_input, system_msg=None, history=None):
        """
        多轮对话接口
        
        Args:
            user_input: 用户输入
            system_msg: 系统提示词
            history: 对话历史（可选）
        
        Returns:
            助手回复
        """
        # 如果有历史记录，构建完整对话
        if history:
            prompt = ""
            if system_msg:
                prompt += f"system\n{system_msg}\n"
            
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                prompt += f"{role}\n{content}\n"
            
            prompt += f"user\n{user_input}"