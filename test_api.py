import requests

# 1. 设置 API 密钥和端点
API_KEY = "sk-ckblwoobzunmdgolnyeoeuyiswsytxtoywmepoarropzelgy"
url = "https://api.siliconflow.cn/v1/chat/completions"

# 2. 构造请求头
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 3. 构造请求体
data = {
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  # 支持的模型列表见文档
    "messages": [
        {
            "role": "user",
            "content": 
                "你是一个眼部医疗助手,请帮助患者解决问题并输出正确的医疗报告"
        }
    ],
    "stream": False,
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5
}

# 4. 发送 POST 请求
response = requests.post(url, headers=headers, json=data)

# 5. 处理响应
if response.status_code == 200:
    result = response.json()
    print("回答:", result['choices'][0]['message']['content'])
    print("消耗 token 数:", result['usage']['total_tokens'])
else:
    print("请求失败，状态码:", response.status_code)
    print("错误信息:", response.text)