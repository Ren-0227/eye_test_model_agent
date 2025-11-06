# api_integration.py
import requests

class DeepseekAPI:
    def __init__(self):
        # 设置 API 密钥和端点
        self.api_key = "sk-ckblwoobzunmdgolnyeoeuyiswsytxtoywmepoarropzelgy"
        self.url = "https://api.siliconflow.cn/v1/chat/completions"
        
        # 构造请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_health_advice(self, symptoms):
        """获取健康建议"""
        # 构造请求体
        data = {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "messages": [
                {
                    "role": "user",
                    "content": f"你是一个眼部医疗助手。患者症状：{symptoms}。请给出详细的诊断建议，包括可能的疾病、建议的检查、护理建议和紧急程度。"
                }
            ],
            "stream": False,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5
        }

        try:
            # 发送 POST 请求
            response = requests.post(self.url, headers=self.headers, json=data)
            
            # 处理响应
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    return "API未返回有效内容"
            else:
                return f"请求失败，状态码: {response.status_code}，错误信息: {response.text}"
        except requests.exceptions.RequestException as e:
            return f"网络请求失败: {str(e)}"