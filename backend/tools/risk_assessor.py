"""
Risk assessment rules for eye symptoms.
Returns high/medium/low and rationale.
"""
from typing import Dict, Tuple, Optional


HIGH_KW = ["剧烈疼痛", "突然失明", "视网膜脱离", "大量飞蚊", "闪光感", "视野缺损", "化学", "外伤", "烧伤"]
MED_KW = ["视力下降", "持续模糊", "复视", "眼压高", "黄斑病变", "畏光", "红肿", "分泌物", "疼痛", "飞蚊"]


def assess(text: Optional[str], struct: Dict[str, any]) -> Tuple[str, str]:
    if not text and not struct:
        return "low", "无症状描述"
    lowered = text or ""
    # Explicit high-risk flags
    if any(k in lowered for k in HIGH_KW) or "trauma" in struct.get("flags", []):
        return "high", "出现剧烈症状/外伤或视网膜风险信号"
    if any(k in lowered for k in MED_KW):
        return "medium", "存在持续视力问题或中度风险信号"
    if struct and struct.get("present"):
        return "medium", "有症状记录"
    return "low", "未见明显风险关键词"

