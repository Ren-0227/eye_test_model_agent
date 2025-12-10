import os
from modelscope import snapshot_download

# 可选：如需代理解锁下载，按需取消注释
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 下载到项目根目录下的 Qwen-1_8B-Chat 目录
MODEL_ID = "qwen/Qwen-1_8B-Chat"
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "Qwen-1_8B-Chat"))

model_dir = snapshot_download(
    MODEL_ID,
    revision="v1.0.0",
    cache_dir=os.path.expanduser("~/.cache/modelscope"),  # 缓存仍放用户缓存目录
    local_dir=LOCAL_DIR                # 实际文件保存到项目根目录
)

print(f"模型下载完成，路径：{model_dir}")