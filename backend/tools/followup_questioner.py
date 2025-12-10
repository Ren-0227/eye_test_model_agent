"""
Generate follow-up questions based on missing critical signals.
"""
from typing import Dict, List


CRITICAL_SLOTS = {
    "vision_blur": "视物模糊是持续的还是间断的？影响远处还是近处？",
    "pain": "是否有眼痛或胀痛？疼痛强度如何？",
    "floaters": "是否突然出现大量飞蚊/黑影？",
    "flashes": "是否有闪光感？",
    "trauma": "近期是否有眼部外伤、化学或异物接触？",
    "redness": "是否明显充血或红肿？伴随分泌物吗？",
    "photophobia": "是否畏光或看强光不适？",
}


def generate_followups(struct: Dict[str, any], max_q: int = 3) -> List[str]:
    present = set(struct.get("present", []))
    flags = set(struct.get("flags", []))
    needed = []

    # Must-ask if absent
    for slot in ["trauma", "flashes", "floaters", "pain"]:
        if slot not in present and slot not in flags:
            needed.append(slot)

    # Additional if space
    for slot in ["vision_blur", "redness", "photophobia"]:
        if slot not in present:
            needed.append(slot)

    questions = []
    for slot in needed:
        if slot in CRITICAL_SLOTS and len(questions) < max_q:
            questions.append(CRITICAL_SLOTS[slot])
    return questions

