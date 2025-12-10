"""
Simple report generator used by orchestrator.
"""
import os
import datetime


def generate_report(user_id: str, title: str, content: str, report_dir: str = "logs/reports") -> str:
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{user_id}_{timestamp}.txt"
    path = os.path.join(report_dir, filename)
    body = f"""=== 眼科健康报告 ===
用户ID: {user_id}
生成时间: {timestamp}
主题: {title}
内容:
{content}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path

