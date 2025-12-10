"""
Lightweight resource/availability checks for health endpoint.
"""
import os
from typing import Dict, Optional

from backend.tools.image_processing import load_error as oct_load_error


def check_resources(llm_load_error: Optional[str], model_path: Optional[str] = None) -> Dict[str, any]:
    """
    检查系统资源可用性。
    使用延迟导入避免 vision_test 的依赖问题。
    """
    # 延迟导入 check_camera_available，避免 mediapipe/tensorflow 依赖问题
    camera_available = False
    try:
        from backend.tools.vision_test import check_camera_available
        camera_available = check_camera_available()
    except Exception:
        # 如果导入失败，假设摄像头不可用
        camera_available = False
    
    return {
        "camera_available": camera_available,
        "oct_model_loaded": oct_load_error is None,
        "oct_model_error": oct_load_error,
        "llm_loaded": llm_load_error is None,
        "llm_error": llm_load_error,
        "llm_model_path": model_path,
        "uploads_writable": os.access(".", os.W_OK),
    }

