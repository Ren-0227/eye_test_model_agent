#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成系统架构和流程图
Generate system architecture and flow diagrams
"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_system_architecture():
    """1. 系统整体架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(5, 9.5, '眼部医疗AI系统整体架构图', fontsize=20, fontweight='bold', ha='center')
    
    # 前端层
    frontend_box = FancyBboxPatch((1, 7.5), 8, 1, boxstyle="round,pad=0.1", 
                                   edgecolor='#3498DB', facecolor='#E3F2FD', linewidth=2)
    ax.add_patch(frontend_box)
    ax.text(5, 8, '前端层 (Frontend)', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(2.5, 7.7, '• HTML5界面\n• 用户交互\n• 实时显示', fontsize=10, ha='center', va='top')
    ax.text(5, 7.7, '• 视力检测UI\n• 图片上传\n• 聊天界面', fontsize=10, ha='center', va='top')
    ax.text(7.5, 7.7, '• 数据可视化\n• 历史记录\n• 报告展示', fontsize=10, ha='center', va='top')
    
    # API层
    api_box = FancyBboxPatch((1, 5.5), 8, 1.5, boxstyle="round,pad=0.1", 
                              edgecolor='#2ECC71', facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(api_box)
    ax.text(5, 6.75, 'API服务层 (Flask Backend)', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(2.5, 6.2, '/chat\n对话接口', fontsize=10, ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#2ECC71'))
    ax.text(5, 6.2, '/action/vision-test\n视力检测接口', fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#2ECC71'))
    ax.text(7.5, 6.2, '/api/profile/*\n用户数据接口', fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#2ECC71'))
    
    # 核心业务层
    core_box = FancyBboxPatch((1, 3), 8, 2, boxstyle="round,pad=0.1", 
                              edgecolor='#9B59B6', facecolor='#F3E5F5', linewidth=2)
    ax.add_patch(core_box)
    ax.text(5, 4.7, '核心业务层 (Orchestrator)', fontsize=14, fontweight='bold', ha='center', va='center')
    
    # 工具模块
    tools_y = 4.2
    tool_width = 1.5
    tool_height = 0.6
    tool_spacing = 0.2
    
    tools = [
        ('LLM\n推理', '#E74C3C'),
        ('OCT\n分类', '#E67E22'),
        ('视力\n检测', '#F39C12'),
        ('记忆\n管理', '#16A085'),
        ('风险\n评估', '#3498DB'),
    ]
    
    x_start = 1.5
    for i, (name, color) in enumerate(tools):
        x = x_start + i * (tool_width + tool_spacing)
        tool_box = FancyBboxPatch((x, tools_y), tool_width, tool_height, 
                                  boxstyle="round,pad=0.05", edgecolor=color, 
                                  facecolor='white', linewidth=1.5)
        ax.add_patch(tool_box)
        ax.text(x + tool_width/2, tools_y + tool_height/2, name, fontsize=9, 
                ha='center', va='center', fontweight='bold')
    
    # 数据层
    data_box = FancyBboxPatch((1, 1), 8, 1.5, boxstyle="round,pad=0.1", 
                              edgecolor='#34495E', facecolor='#ECF0F1', linewidth=2)
    ax.add_patch(data_box)
    ax.text(5, 2.2, '数据存储层', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(2.5, 1.5, '• 用户记忆\n• 对话历史', fontsize=10, ha='center', va='center')
    ax.text(5, 1.5, '• 模型文件\n• 训练数据', fontsize=10, ha='center', va='center')
    ax.text(7.5, 1.5, '• 检测结果\n• 报告文件', fontsize=10, ha='center', va='center')
    
    # 连接线
    # 前端到API
    arrow1 = FancyArrowPatch((5, 7.5), (5, 7), arrowstyle='->', lw=2, color='#3498DB')
    ax.add_patch(arrow1)
    
    # API到核心
    arrow2 = FancyArrowPatch((5, 5.5), (5, 5), arrowstyle='->', lw=2, color='#2ECC71')
    ax.add_patch(arrow2)
    
    # 核心到数据
    arrow3 = FancyArrowPatch((5, 3), (5, 2.5), arrowstyle='->', lw=2, color='#9B59B6')
    ax.add_patch(arrow3)
    
    plt.tight_layout()
    plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
    print("系统整体架构图已保存: system_architecture.png")
    plt.close()


def create_data_preprocessing_flow():
    """2. 数据预处理流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9.5, '数据预处理流程图', fontsize=20, fontweight='bold', ha='center')
    
    # OCT图像预处理流程
    ax.text(2.5, 8.5, 'OCT图像数据', fontsize=12, fontweight='bold', ha='center')
    
    steps_oct = [
        ('原始图像\n加载', 2.5, 7.5),
        ('图像增强\n(Resize, Flip, Rotate)', 2.5, 6.5),
        ('归一化\n(ImageNet标准)', 2.5, 5.5),
        ('数据增强\n(ColorJitter)', 2.5, 4.5),
        ('训练集/验证集/测试集\n划分', 2.5, 3.5),
        ('DataLoader\n批处理', 2.5, 2.5),
    ]
    
    for i, (text, x, y) in enumerate(steps_oct):
        box = FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6, boxstyle="round,pad=0.1", 
                             edgecolor='#E74C3C', facecolor='#FFEBEE', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=9, ha='center', va='center')
        if i < len(steps_oct) - 1:
            arrow = FancyArrowPatch((x, y-0.3), (x, y-0.7), arrowstyle='->', lw=1.5, color='#E74C3C')
            ax.add_patch(arrow)
    
    # 文本数据预处理流程
    ax.text(7.5, 8.5, '文本对话数据', fontsize=12, fontweight='bold', ha='center')
    
    steps_text = [
        ('原始JSONL\n数据加载', 7.5, 7.5),
        ('格式转换\n(Instruction格式)', 7.5, 6.5),
        ('Chat Template\n应用', 7.5, 5.5),
        ('Tokenization\n分词', 7.5, 4.5),
        ('Padding & Truncation\n填充截断', 7.5, 3.5),
        ('训练集/验证集\n划分', 7.5, 2.5),
    ]
    
    for i, (text, x, y) in enumerate(steps_text):
        box = FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6, boxstyle="round,pad=0.1", 
                             edgecolor='#3498DB', facecolor='#E3F2FD', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=9, ha='center', va='center')
        if i < len(steps_text) - 1:
            arrow = FancyArrowPatch((x, y-0.3), (x, y-0.7), arrowstyle='->', lw=1.5, color='#3498DB')
            ax.add_patch(arrow)
    
    # 输出
    output_box = FancyBboxPatch((4, 0.5), 2, 1, boxstyle="round,pad=0.1", 
                                edgecolor='#2ECC71', facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 1.2, '预处理完成', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(5, 0.8, '准备训练', fontsize=10, ha='center', va='center')
    
    # 连接线
    arrow1 = FancyArrowPatch((2.5, 2.2), (4.5, 1.2), arrowstyle='->', lw=2, color='#E74C3C')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((7.5, 2.2), (5.5, 1.2), arrowstyle='->', lw=2, color='#3498DB')
    ax.add_patch(arrow2)
    
    plt.tight_layout()
    plt.savefig('data_preprocessing_flow.png', dpi=300, bbox_inches='tight')
    print("数据预处理流程图已保存: data_preprocessing_flow.png")
    plt.close()


def create_model_finetuning_flow():
    """3. 模型微调流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9.5, '模型微调流程图', fontsize=20, fontweight='bold', ha='center')
    
    # 基础模型
    base_box = FancyBboxPatch((3.5, 8), 3, 0.8, boxstyle="round,pad=0.1", 
                             edgecolor='#9B59B6', facecolor='#F3E5F5', linewidth=2)
    ax.add_patch(base_box)
    ax.text(5, 8.4, 'Qwen-1.8B-Chat 基础模型', fontsize=12, fontweight='bold', ha='center', va='center')
    
    # LoRA配置
    lora_box = FancyBboxPatch((1, 6.5), 2, 1, boxstyle="round,pad=0.1", 
                             edgecolor='#E67E22', facecolor='#FFF3E0', linewidth=1.5)
    ax.add_patch(lora_box)
    ax.text(2, 7.2, 'LoRA配置', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(2, 6.9, 'r=16, α=32', fontsize=9, ha='center', va='center')
    ax.text(2, 6.6, 'target: c_attn', fontsize=9, ha='center', va='center')
    
    # 训练数据
    data_box = FancyBboxPatch((7, 6.5), 2, 1, boxstyle="round,pad=0.1", 
                             edgecolor='#3498DB', facecolor='#E3F2FD', linewidth=1.5)
    ax.add_patch(data_box)
    ax.text(8, 7.2, '训练数据', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(8, 6.9, 'JSONL格式', fontsize=9, ha='center', va='center')
    ax.text(8, 6.6, 'Chat Template', fontsize=9, ha='center', va='center')
    
    # 应用LoRA
    apply_box = FancyBboxPatch((3.5, 5.5), 3, 0.8, boxstyle="round,pad=0.1", 
                              edgecolor='#2ECC71', facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(apply_box)
    ax.text(5, 5.9, '应用LoRA适配器', fontsize=12, fontweight='bold', ha='center', va='center')
    
    # 连接线
    arrow1 = FancyArrowPatch((2, 6.5), (4, 5.5), arrowstyle='->', lw=2, color='#E67E22')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((8, 6.5), (6, 5.5), arrowstyle='->', lw=2, color='#3498DB')
    ax.add_patch(arrow2)
    arrow3 = FancyArrowPatch((5, 8), (5, 6.3), arrowstyle='->', lw=2, color='#9B59B6')
    ax.add_patch(arrow3)
    
    # 训练循环
    train_steps = [
        ('前向传播', 2, 4.5),
        ('计算Loss', 5, 4.5),
        ('反向传播', 8, 4.5),
        ('参数更新', 5, 3.5),
    ]
    
    for i, (text, x, y) in enumerate(train_steps):
        box = FancyBboxPatch((x-0.7, y-0.25), 1.4, 0.5, boxstyle="round,pad=0.1", 
                            edgecolor='#E74C3C', facecolor='#FFEBEE', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=9, ha='center', va='center', fontweight='bold')
    
    # 训练循环箭头
    arrows_train = [
        ((2, 4.25), (4.3, 4.25)),
        ((5.7, 4.25), (7.3, 4.25)),
        ((8, 4.25), (5, 3.75)),
        ((5, 3.25), (2, 4.5)),
    ]
    
    for (start, end) in arrows_train:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', lw=1.5, color='#E74C3C', 
                               connectionstyle="arc3,rad=0.2")
        ax.add_patch(arrow)
    
    # 验证和保存
    eval_box = FancyBboxPatch((1, 2), 2.5, 0.8, boxstyle="round,pad=0.1", 
                             edgecolor='#16A085', facecolor='#E0F2F1', linewidth=1.5)
    ax.add_patch(eval_box)
    ax.text(2.25, 2.4, '验证评估', fontsize=11, fontweight='bold', ha='center', va='center')
    
    save_box = FancyBboxPatch((6.5, 2), 2.5, 0.8, boxstyle="round,pad=0.1", 
                             edgecolor='#8E44AD', facecolor='#F4ECF7', linewidth=1.5)
    ax.add_patch(save_box)
    ax.text(7.75, 2.4, '保存模型', fontsize=11, fontweight='bold', ha='center', va='center')
    
    # 连接
    arrow4 = FancyArrowPatch((5, 3.25), (2.25, 2.8), arrowstyle='->', lw=2, color='#16A085')
    ax.add_patch(arrow4)
    arrow5 = FancyArrowPatch((5, 3.25), (7.75, 2.8), arrowstyle='->', lw=2, color='#8E44AD')
    ax.add_patch(arrow5)
    
    # 输出
    output_box = FancyBboxPatch((3.5, 0.5), 3, 1, boxstyle="round,pad=0.1", 
                               edgecolor='#2ECC71', facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 1.2, '微调完成', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(5, 0.8, 'LoRA适配器\n保存', fontsize=10, ha='center', va='center')
    
    arrow6 = FancyArrowPatch((5, 2), (5, 1.5), arrowstyle='->', lw=2, color='#2ECC71')
    ax.add_patch(arrow6)
    
    plt.tight_layout()
    plt.savefig('model_finetuning_flow.png', dpi=300, bbox_inches='tight')
    print("模型微调流程图已保存: model_finetuning_flow.png")
    plt.close()


def create_inference_flow():
    """4. 推理服务执行流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    ax.text(5, 10.5, '推理服务执行流程图', fontsize=20, fontweight='bold', ha='center')
    
    # 用户请求
    user_box = FancyBboxPatch((3.5, 9), 3, 0.8, boxstyle="round,pad=0.1", 
                             edgecolor='#3498DB', facecolor='#E3F2FD', linewidth=2)
    ax.add_patch(user_box)
    ax.text(5, 9.4, '用户请求', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(4, 9.1, '文本', fontsize=9, ha='center', va='center')
    ax.text(6, 9.1, '图片', fontsize=9, ha='center', va='center')
    
    # API接收
    api_box = FancyBboxPatch((3.5, 7.5), 3, 0.8, boxstyle="round,pad=0.1", 
                            edgecolor='#2ECC71', facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(api_box)
    ax.text(5, 7.9, 'Flask API接收', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(5, 7.6, '/chat 端点', fontsize=10, ha='center', va='center')
    
    arrow1 = FancyArrowPatch((5, 9), (5, 8.3), arrowstyle='->', lw=2, color='#3498DB')
    ax.add_patch(arrow1)
    
    # Orchestrator处理
    orch_box = FancyBboxPatch((1, 6), 8, 1, boxstyle="round,pad=0.1", 
                             edgecolor='#9B59B6', facecolor='#F3E5F5', linewidth=2)
    ax.add_patch(orch_box)
    ax.text(5, 6.7, 'Orchestrator 核心处理', fontsize=12, fontweight='bold', ha='center', va='center')
    
    arrow2 = FancyArrowPatch((5, 7.5), (5, 7), arrowstyle='->', lw=2, color='#2ECC71')
    ax.add_patch(arrow2)
    
    # 处理步骤
    steps = [
        ('加载用户\n记忆', 2, 5.5),
        ('提取症状\n信息', 4, 5.5),
        ('风险评估', 6, 5.5),
        ('判断是否需要\n视力检测', 8, 5.5),
    ]
    
    for i, (text, x, y) in enumerate(steps):
        box = FancyBboxPatch((x-0.6, y-0.25), 1.2, 0.5, boxstyle="round,pad=0.05", 
                            edgecolor='#E67E22', facecolor='#FFF3E0', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=8, ha='center', va='center')
    
    # 工具调用
    tools_y = 4.5
    tools = [
        ('OCT\n分类', 2, tools_y, '#E74C3C'),
        ('视力\n检测', 4, tools_y, '#F39C12'),
        ('LLM\n推理', 6, tools_y, '#3498DB'),
        ('生成\n报告', 8, tools_y, '#16A085'),
    ]
    
    for name, x, y, color in tools:
        box = FancyBboxPatch((x-0.6, y-0.25), 1.2, 0.5, boxstyle="round,pad=0.05", 
                            edgecolor=color, facecolor='white', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=9, ha='center', va='center', fontweight='bold')
    
    # 连接工具
    for i, (name, x, y, color) in enumerate(tools):
        arrow = FancyArrowPatch((steps[i][1], 5.25), (x, 4.75), arrowstyle='->', 
                               lw=1.5, color=color, connectionstyle="arc3,rad=0.1")
        ax.add_patch(arrow)
    
    # 结果整合
    merge_box = FancyBboxPatch((3.5, 3.5), 3, 0.8, boxstyle="round,pad=0.1", 
                              edgecolor='#2ECC71', facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(merge_box)
    ax.text(5, 3.9, '结果整合', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(5, 3.6, '结构化响应', fontsize=10, ha='center', va='center')
    
    for name, x, y, color in tools:
        arrow = FancyArrowPatch((x, 4.25), (5, 4.3), arrowstyle='->', lw=1.5, color=color)
        ax.add_patch(arrow)
    
    # 保存记忆
    memory_box = FancyBboxPatch((1, 2.5), 2, 0.6, boxstyle="round,pad=0.1", 
                               edgecolor='#16A085', facecolor='#E0F2F1', linewidth=1.5)
    ax.add_patch(memory_box)
    ax.text(2, 2.8, '保存到\n用户记忆', fontsize=9, ha='center', va='center', fontweight='bold')
    
    arrow_mem = FancyArrowPatch((5, 3.5), (2, 2.8), arrowstyle='->', lw=2, color='#16A085')
    ax.add_patch(arrow_mem)
    
    # 返回响应
    response_box = FancyBboxPatch((6, 2.5), 2, 0.6, boxstyle="round,pad=0.1", 
                                 edgecolor='#8E44AD', facecolor='#F4ECF7', linewidth=1.5)
    ax.add_patch(response_box)
    ax.text(7, 2.8, '返回JSON\n响应', fontsize=9, ha='center', va='center', fontweight='bold')
    
    arrow_resp = FancyArrowPatch((5, 3.5), (7, 2.8), arrowstyle='->', lw=2, color='#8E44AD')
    ax.add_patch(arrow_resp)
    
    # 前端显示
    frontend_box = FancyBboxPatch((3.5, 1), 3, 0.8, boxstyle="round,pad=0.1", 
                                 edgecolor='#3498DB', facecolor='#E3F2FD', linewidth=2)
    ax.add_patch(frontend_box)
    ax.text(5, 1.4, '前端显示结果', fontsize=12, fontweight='bold', ha='center', va='center')
    
    arrow_front1 = FancyArrowPatch((2, 2.5), (4.5, 1.8), arrowstyle='->', lw=1.5, color='#16A085')
    ax.add_patch(arrow_front1)
    arrow_front2 = FancyArrowPatch((7, 2.5), (5.5, 1.8), arrowstyle='->', lw=1.5, color='#8E44AD')
    ax.add_patch(arrow_front2)
    
    plt.tight_layout()
    plt.savefig('inference_flow.png', dpi=300, bbox_inches='tight')
    print("推理服务执行流程图已保存: inference_flow.png")
    plt.close()


if __name__ == "__main__":
    print("正在生成系统架构和流程图...")
    create_system_architecture()
    create_data_preprocessing_flow()
    create_model_finetuning_flow()
    create_inference_flow()
    print("\n所有图表已生成完成！")
    print("生成的文件：")
    print("  1. system_architecture.png - 系统整体架构图")
    print("  2. data_preprocessing_flow.png - 数据预处理流程图")
    print("  3. model_finetuning_flow.png - 模型微调流程图")
    print("  4. inference_flow.png - 推理服务执行流程图")

