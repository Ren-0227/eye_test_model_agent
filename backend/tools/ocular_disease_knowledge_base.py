# ocular_disease_knowledge_base.py
class OcularDiseaseKnowledgeBase:
    def __init__(self):
        self.disease_data = {
            "近视": {
                "symptoms": ["看不清远处物体", "眯眼", "频繁眨眼", "眼睛疲劳"],
                "intro": "近视是眼睛屈光系统的问题，导致远处物体看起来模糊。"
            },
            "远视": {
                "symptoms": ["看不清近处物体", "眼睛疲劳", "头痛", "阅读困难"],
                "intro": "远视是眼睛屈光系统的问题，导致近处物体看起来模糊。"
            },
            "白内障": {
                "symptoms": ["视力逐渐下降", "看东西有雾状感", "对强光敏感", "色彩感知减弱"],
                "intro": "白内障是眼睛晶状体变得混浊，导致视力下降。"
            },
            # 更多病症信息...
        }

    def find_similar_diseases(self, symptoms):
        matched_diseases = {}
        for disease, data in self.disease_data.items():
            common_symptoms = set(data["symptoms"]) & set(symptoms)
            if common_symptoms:
                matched_diseases[disease] = {
                    "symptoms": data["symptoms"],
                    "common_symptoms_with_user": list(common_symptoms),
                    "intro": data["intro"]
                }
        return matched_diseases