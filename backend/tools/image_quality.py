"""
Simple image quality checks to avoid bad uploads.
Checks blur (variance of Laplacian) and brightness.
"""
import cv2
import numpy as np
from typing import Dict


def evaluate_image_quality(path: str, blur_threshold: float = 50.0, brightness_low: int = 30, brightness_high: int = 230) -> Dict[str, any]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "reason": "无法读取图片"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur score
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_ok = lap_var >= blur_threshold
    # Brightness
    mean_brightness = float(np.mean(gray))
    bright_ok = brightness_low <= mean_brightness <= brightness_high

    ok = blur_ok and bright_ok
    tips = []
    if not blur_ok:
        tips.append("图片可能过于模糊，请对焦后重新拍摄")
    if not bright_ok:
        tips.append("光照不合适，请增加光照或避免过曝")

    return {
        "ok": ok,
        "blur_score": lap_var,
        "brightness": mean_brightness,
        "tips": tips,
    }

