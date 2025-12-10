"""
main.py - CLI agent for multi-turn Q&A (uses latest orchestrator/LLM/OCT).
默认进入多轮问答模式；如需烟囱自测可用 --mode smoke。
"""
import sys
import os
import json
import datetime
import argparse

# 确保能从项目根导入 backend 包
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from backend.tools.local_qwen_api import LocalQwenAPI
from backend.orchestrator import Orchestrator
from backend.tools.image_processing import analyze_image
from backend.tools.vision_test import VisionTester, check_camera_available
from typing import Optional

DEFAULT_IMAGE = os.path.join(BASE_DIR, "archive", "RetinalOCT_Dataset", "test", "AMD", "amd_test_1350.jpg")
DEFAULT_USER = "cli_user"


def smoke_llm(orchestrator: Orchestrator, prompt: str):
    """LLM 单轮测试"""
    print("[main] LLM smoke test...")
    resp = orchestrator.llm.run_smoke_test(prompt)
    print(json.dumps(resp, ensure_ascii=False, indent=2))


def smoke_oct(image_path: str):
    """OCT 分类测试"""
    print("[main] OCT test on:", image_path)
    if not os.path.exists(image_path):
        print(f"[main] image not found: {image_path}")
        return
    try:
        result = analyze_image(image_path)
        print(f"[main] OCT result: {result}")
    except Exception as e:
        print(f"[main] OCT failed: {e}")


def chat_once(orch: Orchestrator, text: str, vision_tester: Optional[VisionTester] = None):
    """调用 orchestrator 的对话流程，自动传入视力检测器"""
    out = orch.process(user_id=DEFAULT_USER, text=text, vision_tester=vision_tester)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def chat_loop(orch: Orchestrator, user_id: str = DEFAULT_USER, vision_tester: Optional[VisionTester] = None):
    """多轮问答循环，输入 q 退出；自动携带视力检测器以便需要时直接调用"""
    print("=== 多轮问答模式，输入 q 退出 ===")
    while True:
        try:
            user_text = input("\n用户: ").strip()
            if user_text.lower() in ("q", "quit", "exit", "bye"):
                print("退出对话。")
                break
            result = orch.process(user_id=user_id, text=user_text, vision_tester=vision_tester)
            if isinstance(result, dict):
                ans = result.get("answer", "")
                action = result.get("action")
                print(f"助手: {ans}")
                if action:
                    print(f"[动作提示] {action}")
                if result.get("data", {}).get("followups"):
                    print(f"[追问建议] {result['data']['followups']}")
                elif result.get("followups"):
                    print(f"[追问建议] {result['followups']}")
                if result.get("data", {}).get("risk_level"):
                    print(f"[风险等级] {result['data']['risk_level']}")
                elif result.get("risk"):
                    print(f"[风险等级] {result['risk']}")
            else:
                print(f"助手: {result}")
        except KeyboardInterrupt:
            print("\n用户中断，退出对话。")
            break
        except Exception as e:
            print(f"对话出错: {e}")
            break


def main():
    parser = argparse.ArgumentParser(description="CLI agent for backend workflow")
    parser.add_argument("--mode", choices=["chat", "smoke"], default="chat", help="chat: 多轮问答; smoke: 自测")
    parser.add_argument("--prompt", default="请用一句话自我介绍。", help="LLM smoke prompt")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="OCT test image path")
    parser.add_argument("--chat", default="我最近看远处有点模糊，左眼0.8右眼0.6", help="chat text (for smoke)")
    args = parser.parse_args()

    print("[main] loading orchestrator...")
    orch = Orchestrator()
    print(f"[main] llm load_error: {orch.llm.load_error}")
    print(f"[main] llm model_path: {getattr(orch.llm, 'model_path', None)}")

    # 尝试初始化视力检测器（若摄像头不可用则继续运行，后续会提示前端/CLI处理）
    vision_tester = None
    try:
        if check_camera_available():
            vision_tester = VisionTester()
            print("[main] vision tester ready.")
        else:
            print("[main] 未检测到可用摄像头，视力检测将返回动作提示。")
    except Exception as e:
        print(f"[main] 视力检测模块初始化失败: {e}")

    if args.mode == "smoke":
        smoke_llm(orch, args.prompt)
        smoke_oct(args.image)
        print("[main] chat test (single turn)...")
        chat_once(orch, args.chat, vision_tester)
        print("[main] done.")
    else:
        chat_loop(orch, DEFAULT_USER, vision_tester)


if __name__ == "__main__":
    main()