# api_integration.py
"""
使用本地千问 1.8B（或微调后权重）提供健康咨询接口。
如果模型未下载/未微调，请先运行 finetune 脚本下载或准备模型权重。
"""
from backend.tools.local_qwen_api import LocalQwenAPI


class LocalQwenClient:
    """简单封装，保持与旧 DeepseekAPI 相同的 get_health_advice 接口名。"""

    def __init__(self, model_path=None, base_model_name=None):
        """
        初始化千问客户端
        
        Args:
            model_path: 模型路径，如果为None则自动搜索
            base_model_name: 基础模型名称
        """
        self.api = LocalQwenAPI(model_path=model_path, base_model_name=base_model_name)

    def get_health_advice(self, symptoms, vision_result=None, oct_result=None):
        """
        获取健康建议
        
        Args:
            symptoms: 症状描述
            vision_result: 视力检测结果（可选）
            oct_result: OCT检查结果（可选）
            
        Returns:
            dict: 包含状态和回答的字典
        """
        return self.api.get_health_advice(
            symptoms=symptoms,
            vision_result=vision_result,
            oct_result=oct_result,
        )

    def is_ready(self):
        """
        检查模型是否准备好
        
        Returns:
            bool: 模型是否加载成功
        """
        return self.api.load_error is None

    def get_error(self):
        """
        获取模型加载错误信息
        
        Returns:
            str or None: 错误信息，如果没有错误则返回None
        """
        return self.api.load_error