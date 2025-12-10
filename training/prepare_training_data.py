# -*- coding: utf-8 -*-
import json
import os
from typing import List, Dict

# ==================== 示例数据生成 ====================
def create_sample_data():
    """创建示例训练数据"""
    
    # 示例对话数据（基于眼科医疗场景）
    sample_conversations = [
        {
            "system": "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。",
            "instruction": "我最近看东西有点模糊，特别是看远处的时候。",
            "output": "根据您描述的视力模糊症状，特别是看远处不清楚，这可能是近视的表现。建议您：\n1. 尽快到眼科医院进行专业的视力检查\n2. 检查包括裸眼视力、矫正视力、眼压和眼底检查\n3. 如果确诊为近视，需要配戴合适的眼镜或隐形眼镜\n4. 注意用眼卫生，避免长时间近距离用眼\n5. 定期复查，监测视力变化情况"
        },
        {
            "system": "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。",
            "instruction": "我的眼睛经常感到干涩、疲劳，有时候还会发红。",
            "output": "您描述的症状（干涩、疲劳、发红）可能是干眼症的表现。建议：\n1. 使用人工泪液眼药水缓解症状\n2. 减少长时间使用电子设备，每20分钟休息20秒，看20英尺外的物体\n3. 保持室内湿度适宜，避免空调直吹\n4. 多眨眼，保持眼部湿润\n5. 如果症状持续或加重，建议到眼科就诊进行泪液分泌测试等检查"
        },
        {
            "system": "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。",
            "instruction": "我最近看东西有重影，而且眼睛有点疼。",
            "output": "重影（复视）伴随眼痛是需要重视的症状。可能的原因包括：\n1. 眼肌问题：如斜视、眼肌麻痹\n2. 屈光不正：散光、近视或远视\n3. 眼部疾病：如角膜炎、虹膜炎\n4. 神经系统问题：如动眼神经麻痹\n\n建议您：\n- 立即到眼科医院就诊，进行详细检查\n- 告知医生症状出现的时间、频率和伴随症状\n- 进行视力检查、眼压检查、眼底检查和眼肌功能检查\n- 如果伴有头痛、恶心等症状，可能需要神经科会诊"
        },
        {
            "system": "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。",
            "instruction": "我的视力检测结果是左眼0.8，右眼0.6，这是什么意思？",
            "output": "视力检测结果显示：\n- 左眼视力：0.8（相当于国际标准的6/7.5）\n- 右眼视力：0.6（相当于国际标准的6/10）\n\n正常视力标准是1.0（6/6），您的双眼视力都低于正常值。\n\n建议：\n1. 进行进一步的屈光检查，确定是否需要配镜\n2. 检查是否有散光、近视或远视\n3. 进行眼底检查，排除器质性病变\n4. 如果视力下降是近期发生的，需要查明原因\n5. 根据检查结果，可能需要配戴眼镜或进行其他治疗"
        },
        {
            "system": "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。",
            "instruction": "OCT检查显示黄斑病变，严重吗？",
            "output": "OCT（光学相干断层扫描）显示黄斑病变需要根据具体情况判断严重程度。\n\n黄斑病变的类型包括：\n1. 年龄相关性黄斑变性（AMD）\n2. 黄斑水肿\n3. 黄斑裂孔\n4. 中心性浆液性脉络膜视网膜病变\n\n建议：\n1. 携带OCT报告到眼科专科医生处详细解读\n2. 根据病变类型和程度制定治疗方案\n3. 可能需要定期复查OCT监测病情变化\n4. 某些类型的黄斑病变需要及时治疗，避免视力进一步下降\n5. 注意保护眼睛，避免强光刺激，补充叶黄素等营养素\n\n请尽快咨询专业眼科医生，根据具体检查结果制定治疗方案。"
        },
        {
            "system": "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。",
            "instruction": "我眼睛里有飞蚊症，很多小黑点在飘。",
            "output": "飞蚊症（玻璃体混浊）是常见的眼部症状，表现为眼前有飘动的小黑点、线条或阴影。\n\n大多数飞蚊症是良性的，但需要排除以下情况：\n1. 玻璃体后脱离：随着年龄增长自然发生\n2. 视网膜裂孔或脱离：需要紧急处理\n3. 玻璃体出血：可能由糖尿病、高血压等引起\n\n建议：\n1. 如果突然出现大量飞蚊或伴有闪光感，需要立即就医\n2. 进行散瞳眼底检查，排除视网膜问题\n3. 如果检查正常，通常不需要特殊治疗\n4. 避免剧烈运动和头部剧烈晃动\n5. 定期复查，监测变化\n\n如果症状突然加重或出现视力下降，请立即就诊。"
        },
        {
            "system": "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。",
            "instruction": "我最近看东西颜色变淡了，而且中心视力下降。",
            "output": "您描述的症状（颜色变淡、中心视力下降）需要高度重视，可能是黄斑病变的表现。\n\n可能的原因：\n1. 年龄相关性黄斑变性（AMD）\n2. 黄斑水肿\n3. 中心性浆液性脉络膜视网膜病变\n4. 其他黄斑疾病\n\n建议：\n1. 立即到眼科医院进行详细检查\n2. 进行OCT检查、眼底荧光血管造影等检查\n3. 早期诊断和治疗对保护视力至关重要\n4. 某些黄斑病变可以通过抗VEGF药物治疗\n5. 注意保护眼睛，避免强光，补充叶黄素、玉米黄质等营养素\n\n请尽快就诊，不要延误治疗时机。"
        },
        {
            "system": "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。",
            "instruction": "我眼睛对光很敏感，一看到强光就流泪。",
            "output": "畏光（光敏感）伴随流泪可能由多种原因引起：\n\n可能的原因：\n1. 角膜炎或结膜炎\n2. 干眼症\n3. 虹膜炎（葡萄膜炎）\n4. 角膜异物或损伤\n5. 偏头痛\n\n建议：\n1. 避免强光刺激，佩戴防紫外线太阳镜\n2. 使用人工泪液缓解症状\n3. 如果伴有眼红、眼痛，需要到眼科检查\n4. 进行裂隙灯检查，排除角膜炎、虹膜炎等\n5. 根据检查结果进行相应治疗\n\n如果症状严重或持续，建议尽快就诊。"
        }
    ]
    
    return sample_conversations


def save_jsonl(data: List[Dict], file_path: str):
    """保存数据为 JSONL 格式"""
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已保存 {len(data)} 条数据到 {file_path}")


def split_train_eval(data: List[Dict], eval_ratio: float = 0.2):
    """划分训练集和验证集"""
    import random
    random.seed(42)
    random.shuffle(data)
    
    split_idx = int(len(data) * (1 - eval_ratio))
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    return train_data, eval_data


def load_from_knowledge_base():
    """从知识库文件加载数据并转换为训练格式"""
    try:
        from ocular_disease_knowledge_base import OcularDiseaseKnowledgeBase
        
        kb = OcularDiseaseKnowledgeBase()
        training_data = []
        
        # 为每种疾病生成训练样本
        for disease, info in kb.disease_data.items():
            symptoms_str = "、".join(info["symptoms"])
            instruction = f"我有以下症状：{symptoms_str}，可能是什么问题？"
            output = f"根据您描述的症状（{symptoms_str}），可能是{disease}。{info['intro']}\n\n建议：\n1. 到眼科医院进行专业检查\n2. 进行视力检查、眼压检查、眼底检查等\n3. 根据检查结果制定治疗方案\n4. 注意用眼卫生，定期复查"
            
            training_data.append({
                "system": "你是一个专业的眼科医疗助手，能够根据患者的症状提供准确的诊断建议。",
                "instruction": instruction,
                "output": output
            })
        
        return training_data
    except Exception as e:
        print(f"从知识库加载数据失败: {e}")
        return []


def main():
    """主函数"""
    print("准备千问模型微调数据...")
    
    # 创建数据目录
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 生成训练数据
    print("1. 生成示例对话数据...")
    sample_data = create_sample_data()
    
    print("2. 从知识库加载数据...")
    kb_data = load_from_knowledge_base()
    
    # 合并数据
    all_data = sample_data + kb_data
    print(f"总共生成 {len(all_data)} 条训练数据")
    
    # 划分训练集和验证集
    print("3. 划分训练集和验证集...")
    train_data, eval_data = split_train_eval(all_data, eval_ratio=0.2)
    
    # 保存数据
    train_path = os.path.join(data_dir, "train_data.jsonl")
    eval_path = os.path.join(data_dir, "eval_data.jsonl")
    
    save_jsonl(train_data, train_path)
    save_jsonl(eval_data, eval_path)
    
    print("\n数据准备完成！")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(eval_data)} 条")
    print(f"\n数据格式说明:")
    print("每行是一个 JSON 对象，包含以下字段:")
    print("  - system: 系统提示词")
    print("  - instruction: 用户输入/问题")
    print("  - output: 助手回复/答案")


if __name__ == "__main__":
    main()

