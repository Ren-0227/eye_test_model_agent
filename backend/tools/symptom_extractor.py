"""
Rule-based symptom struct extractor for eye health.
Returns structured fields for downstream risk assessment and follow-up Qs.
"""
from typing import Dict, List, Optional


KEYWORDS = {
    "vision_blur": ["模糊", "看不清", "视力下降", "重影", "复视"],
    "pain": ["眼痛", "刺痛", "胀痛", "酸痛"],
    "redness": ["充血", "红肿"],
    "photophobia": ["畏光", "怕光"],
    "floaters": ["飞蚊", "黑影", "飘动的点"],
    "flashes": ["闪光", "闪烁", "闪光感"],
    "itching": ["痒", "瘙痒"],
    "dry": ["干涩", "异物感"],
    "tearing": ["流泪", "流眼泪"],
    "discharge": ["分泌物", "眼屎", "脓"],
    "trauma": ["外伤", "撞", "刮伤", "化学", "烧伤"],
    "diabetes": ["糖尿病"],
    "hypertension": ["高血压"],
}


def extract_structured(text: Optional[str]) -> Dict[str, any]:
    """Very lightweight keyword-based struct extractor."""
    if not text:
        return {"present": [], "history": [], "flags": []}
    present: List[str] = []
    history: List[str] = []
    flags: List[str] = []
    lowered = text
    for k, kws in KEYWORDS.items():
        if any(w in lowered for w in kws):
            present.append(k)
    if "糖尿病" in lowered:
        history.append("diabetes")
    if "高血压" in lowered:
        history.append("hypertension")
    if "外伤" in lowered or "化学" in lowered:
        flags.append("trauma")
    return {"present": present, "history": history, "flags": flags}

