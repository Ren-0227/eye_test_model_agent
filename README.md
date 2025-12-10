# 眼部医疗AI智能助手系统

一个基于大语言模型（LLM）和深度学习技术的智能眼科医疗助手系统，集成了自动视力检测、OCT图像分析、风险评估、多轮对话记忆等功能，为患者提供专业的眼部健康咨询和初步诊断建议。

## 📋 目录

- [系统概述](#系统概述)
- [核心特性](#核心特性)
- [Agent设计理念](#agent设计理念)
- [系统架构](#系统架构)
- [完整工作流](#完整工作流)
- [项目结构](#项目结构)
- [核心模块详解](#核心模块详解)
- [安装与配置](#安装与配置)
- [使用方法](#使用方法)
- [模型训练](#模型训练)
- [API文档](#api文档)
- [架构图表](#架构图表)

---

## 🎯 系统概述

本系统是一个**智能Agent系统**，能够：

1. **智能对话**：基于微调的Qwen-1.8B-Chat模型，提供专业的眼科医疗咨询
2. **自动工具调用**：根据用户症状自动触发视力检测、OCT图像分析等工具
3. **记忆管理**：维护用户对话历史、检测结果、诊断记录等上下文信息
4. **风险评估**：自动评估症状风险等级，提供分级建议
5. **多模态输入**：支持文本描述和OCT图像上传

---

## ✨ 核心特性

### 1. 智能Agent工作流
- **自动工具调用**：根据症状关键词、风险等级自动触发视力检测
- **上下文感知**：基于用户历史记录和当前症状进行综合判断
- **多轮对话**：支持连续对话，保持上下文连贯性

### 2. 视力检测工具
- **实时检测**：使用摄像头和MediaPipe进行实时手势识别
- **自动计算**：根据距离和手势方向自动计算视力值
- **结果保存**：检测结果自动保存到用户记忆

### 3. OCT图像分析
- **8类疾病分类**：AMD、CNV、CSR、DME、DR、DRUSEN、MH、NORMAL
- **ResNet50模型**：基于PyTorch的深度学习分类模型
- **质量评估**：自动评估图像质量，提供可靠性建议

### 4. 风险评估系统
- **三级风险**：low、medium、high
- **智能判断**：基于症状关键词和结构化信息
- **追问建议**：自动生成相关追问问题

### 5. 记忆管理
- **用户隔离**：每个用户独立的记忆空间
- **持久化存储**：JSON格式保存，支持跨会话记忆
- **历史追踪**：保存对话历史、检测结果、诊断记录

---

## 🤖 Agent设计理念

本系统采用**Orchestrator模式**的Agent架构：

```
用户请求 → Orchestrator（协调器）→ 工具调用 → LLM推理 → 结果整合 → 响应
```

### Agent核心能力

1. **意图理解**：分析用户输入，提取症状信息
2. **工具选择**：智能判断需要调用哪些工具（视力检测、OCT分析等）
3. **上下文管理**：维护用户记忆，支持多轮对话
4. **结果整合**：将多个工具的结果整合为统一的响应

### 自动工具触发机制

系统会在以下情况自动触发视力检测：

- ✅ 用户明确提到"测视力"、"我要测视力"等关键词
- ✅ 症状描述包含"模糊"、"看不清"、"视力下降"等关键词
- ✅ 风险评估为中/高风险且尚无视力结果
- ✅ 症状描述过短（<10字符）且提到视力相关
- ✅ 风险原因中包含"视力"相关描述

---

## 🏗️ 系统架构

### 架构层次

```
┌─────────────────────────────────────┐
│        前端层 (Frontend)            │
│  • HTML5界面 • 用户交互 • 实时显示  │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      API服务层 (Flask Backend)      │
│  • /chat • /action/vision-test      │
│  • /api/profile/* • /api/diagnosis/* │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│    核心业务层 (Orchestrator)        │
│  ┌──────┐ ┌──────┐ ┌──────┐         │
│  │ LLM  │ │ OCT  │ │ 视力 │         │
│  └──────┘ └──────┘ └──────┘         │
│  ┌──────┐ ┌──────┐                 │
│  │ 记忆 │ │ 风险 │                 │
│  └──────┘ └──────┘                 │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│        数据存储层                    │
│  • 用户记忆 • 模型文件 • 检测结果    │
└─────────────────────────────────────┘
```

### 核心组件

1. **Orchestrator** (`backend/orchestrator.py`)
   - 核心协调器，统一管理所有工具和LLM调用
   - 实现自动工具触发逻辑
   - 整合多模态输入和输出

2. **LLM模块** (`backend/tools/local_qwen_api.py`)
   - 本地Qwen模型加载和推理
   - 支持LoRA微调模型加载
   - 对话模板和生成控制

3. **工具模块** (`backend/tools/`)
   - `vision_test.py`: 视力检测工具
   - `image_processing.py`: OCT图像分类
   - `memory_manager.py`: 用户记忆管理
   - `risk_assessor.py`: 风险评估
   - `symptom_extractor.py`: 症状提取
   - `followup_questioner.py`: 追问生成

---

## 🔄 完整工作流

### CLI模式工作流

```
用户输入症状
    ↓
main.py: chat_loop()
    ↓
orchestrator.process(user_id, text, vision_tester)
    ↓
判断是否需要视力检测 (_should_offer_vision_test)
    ├─ 需要且vision_tester可用 → 自动执行检测 → 保存结果
    └─ 需要但不可用 → 返回action: vision_test
    ↓
提取症状信息 (extract_structured)
    ↓
评估风险等级 (assess)
    ↓
生成追问建议 (generate_followups)
    ↓
调用LLM获取医疗建议 (LocalQwenAPI.generate)
    ↓
保存对话历史到记忆 (update_user_memory)
    ↓
返回结构化响应 {answer, action, data}
```

### Web模式工作流

```
前端发送POST /chat {text, user_id, image?}
    ↓
app.py: chat_endpoint()
    ↓
MedicalAssistant.process_request()
    ↓
orchestrator.process(user_id, text, image_file, vision_tester)
    ↓
（同CLI流程）
    ↓
返回JSON响应
    ↓
前端处理action: vision_test → 调用/action/vision-test
    ↓
显示答案、风险提示、追问建议
```

### 视力检测工作流

```
检测到需要视力检测
    ↓
VisionTester.run_test()
    ↓
打开摄像头，显示E字图
    ↓
MediaPipe检测人脸和手势
    ↓
计算距离和手势方向
    ↓
根据距离和方向计算视力值
    ↓
保存结果到用户记忆
    ↓
返回视力值 {left: 0.8, right: 0.6}
```

### OCT图像分析工作流

```
用户上传OCT图像
    ↓
保存到uploads目录
    ↓
评估图像质量 (evaluate_image_quality)
    ↓
OCT分类模型推理 (analyze_image)
    ↓
返回分类结果和置信度
    ↓
保存结果到用户记忆
    ↓
整合到LLM上下文
```

---

## 📁 项目结构

```
eye_test_model-master/
├── backend/                    # 后端核心代码
│   ├── __init__.py
│   ├── main.py                 # CLI入口
│   ├── app.py                  # Flask Web服务
│   ├── orchestrator.py         # 核心协调器
│   ├── tools/                  # 工具模块
│   │   ├── local_qwen_api.py   # LLM推理接口
│   │   ├── vision_test.py      # 视力检测工具
│   │   ├── image_processing.py # OCT图像分类
│   │   ├── memory_manager.py   # 用户记忆管理
│   │   ├── risk_assessor.py    # 风险评估
│   │   ├── symptom_extractor.py # 症状提取
│   │   ├── followup_questioner.py # 追问生成
│   │   ├── image_quality.py    # 图像质量评估
│   │   ├── reporting.py        # 报告生成
│   │   └── resource_check.py   # 资源检查
│   └── services/               # 服务层（预留）
│
├── training/                   # 模型训练脚本
│   ├── train_oct_pytorch.py    # OCT分类模型训练
│   ├── finetune_qwen.py        # Qwen模型微调
│   ├── process_huatuo_datasets.py # 数据集处理
│   ├── rl_finetune_qwen.py     # 强化学习微调
│   ├── train_reward_model.py   # 奖励模型训练
│   └── generate_architecture_diagrams.py # 架构图生成
│
├── models/                     # 模型文件
│   ├── best_model.pth          # OCT分类模型
│   ├── class_to_idx.json       # 类别映射
│   └── training_history.json   # 训练历史
│
├── Qwen-1_8B-Chat/            # Qwen基础模型
├── qwen_finetuned_model/      # 微调后的模型
│
├── data/                       # 训练数据
│   ├── train_data.jsonl        # 训练集
│   ├── eval_data.jsonl         # 验证集
│   └── reward_data.jsonl       # 奖励数据
│
├── archive/                    # 数据集
│   └── RetinalOCT_Dataset/     # OCT图像数据集
│
├── ui5.html                    # 前端界面
├── user_memory.json            # 用户记忆存储
├── uploads/                    # 上传文件目录
├── logs/                       # 日志目录
│   └── reports/                # 生成的报告
│
├── requirements.txt            # Python依赖
└── README.md                   # 本文档
```

---

## 🔧 核心模块详解

### 1. Orchestrator (`backend/orchestrator.py`)

**功能**：核心协调器，统一管理所有工具和LLM调用

**主要方法**：
- `process(user_id, text, image_file, vision_tester)`: 主入口，处理用户请求
- `_should_offer_vision_test()`: 判断是否需要触发视力检测
- `_needs_vision_test()`: 检查文本中是否包含视力相关关键词
- `tool_status()`: 检查各工具模块的可用状态

**代码结构**：
```python
class Orchestrator:
    def __init__(self):
        self.llm = LocalQwenAPI()  # 初始化LLM
        self.uploads_dir = "uploads"
    
    def process(self, user_id, text, image_file, vision_tester):
        # 1. 加载用户记忆
        # 2. 处理图像（如果有）
        # 3. 判断是否需要视力检测
        # 4. 提取症状、评估风险
        # 5. 调用LLM生成回答
        # 6. 保存记忆
        # 7. 返回结构化响应
```

**保存内容**：
- 用户记忆更新逻辑
- 工具调用协调逻辑
- 响应结构组装

---

### 2. LocalQwenAPI (`backend/tools/local_qwen_api.py`)

**功能**：使用千问开源模型1.8b作为基础模型

**如何下载模型到本地**
-https://modelscope.cn/models/qwen/Qwen-1_8B-Chat/summary  
-连接中有多种方式下载
-或者运行根目录中的download文件

**主要方法**：
- `__init__()`: 自动查找并加载模型（支持LoRA微调模型）
- `generate()`: 生成回答，支持对话历史
- `run_smoke_test()`: 模型可用性测试

**代码结构**：
```python
class LocalQwenAPI:
    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = self._auto_pick_model_path()
        self._maybe_load()  # 延迟加载
    
    def generate(self, prompt, history=None, max_length=512):
        # 构建对话模板
        # Tokenization
        # 模型推理
        # 解码返回
```

**保存内容**：
- 模型自动查找逻辑（支持多个候选路径）
- LoRA适配器加载逻辑
- 对话模板构建（Qwen Chat格式）

---

### 3. VisionTester (`backend/tools/vision_test.py`)

**功能**：使用摄像头和MediaPipe进行实时视力检测

**主要方法**：
- `__init__()`: 初始化摄像头和MediaPipe模型
- `run_test()`: 执行完整的视力检测流程
- `_detect_face_distance()`: 检测人脸距离
- `_detect_hand_gesture()`: 检测手势方向

**代码结构**：
```python
class VisionTester:
    def __init__(self, camera_index=0, target_distance_cm=60):
        self.cap = cv2.VideoCapture(camera_index)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(...)
        self.hands = mp.solutions.hands.Hands(...)
    
    def run_test(self):
        # 1. 显示E字图
        # 2. 检测人脸距离
        # 3. 检测手势方向
        # 4. 计算视力值
        # 5. 返回结果
```

**保存内容**：
- MediaPipe人脸和手势检测逻辑
- 距离计算算法（基于人脸关键点）
- 视力值计算公式
- 实时视频流处理

---

### 4. ImageProcessing (`backend/tools/image_processing.py`)

**功能**：OCT图像的分类分析

**模型训练使用到的数据集连接**
-https://data.mendeley.com/datasets/sncdhf53xc/4

**主要方法**：
- `load_model()`: 加载训练好的ResNet50分类模型
- `analyze_image()`: 对单张图像进行分类
- `OCTClassifier`: ResNet50 + 自定义分类头

**代码结构**：
```python
class OCTClassifier(torch.nn.Module):
    def __init__(self, num_classes=9):
        self.base_model = torch.hub.load('pytorch/vision', 'resnet50')
        self.base_model.fc = torch.nn.Sequential(...)  # 自定义分类头

def analyze_image(image_path):
    # 1. 加载和预处理图像
    # 2. 模型推理
    # 3. 返回类别和置信度
```

**保存内容**：
- ResNet50模型结构定义
- 图像预处理流程（Resize、Normalize）
- 类别映射（8类疾病 + NORMAL）

---

### 5. MemoryManager (`backend/tools/memory_manager.py`)

**功能**：用户记忆的读取和更新

**主要方法**：
- `get_user_memory(user_id)`: 获取指定用户的记忆
- `update_user_memory(user_id, updates)`: 更新用户记忆

**代码结构**：
```python
MEMORY_FILE = "user_memory.json"

def get_user_memory(user_id):
    # 读取JSON文件
    # 返回用户数据字典

def update_user_memory(user_id, updates):
    # 读取现有数据
    # 更新用户数据
    # 写回JSON文件
```

**保存内容**：
- 用户记忆结构：
  ```json
  {
    "user_id": {
      "chat_history": [...],
      "vision_test": {"left": 0.8, "right": 0.6},
      "oct_result": "AMD",
      "last_response": "...",
      "history": [...]
    }
  }
  ```

---

### 6. RiskAssessor (`backend/tools/risk_assessor.py`)

**功能**：基于症状文本评估风险等级

**主要方法**：
- `assess(text, struct)`: 评估风险等级和原因

**保存内容**：
- 风险关键词库（high、medium、low）
- 风险评估逻辑

---

### 7. SymptomExtractor (`backend/tools/symptom_extractor.py`)

**功能**：从用户文本中提取结构化症状信息

**主要方法**：
- `extract_structured(text)`: 提取症状、部位、持续时间等

**保存内容**：
- 症状提取规则和关键词匹配

---

### 8. FollowupQuestioner (`backend/tools/followup_questioner.py`)

**功能**：根据症状生成相关追问问题

**主要方法**：
- `generate_followups(struct)`: 生成追问列表

**保存内容**：
- 追问问题模板库

---

### 9. Main (`backend/main.py`)

**功能**：CLI命令行入口

**主要方法**：
- `main()`: 解析命令行参数
- `chat_loop()`: 多轮对话循环
- `chat_once()`: 单次对话测试

**代码结构**：
```python
def main():
    # 1. 解析参数（--mode, --chat等）
    # 2. 初始化Orchestrator
    # 3. 初始化VisionTester（如果摄像头可用）
    # 4. 进入chat_loop或执行smoke测试

def chat_loop(orch, user_id, vision_tester):
    # 循环读取用户输入
    # 调用orchestrator.process()
    # 显示结果
```

**保存内容**：
- CLI参数解析
- 摄像头可用性检测
- 交互式对话循环

---

### 10. App (`backend/app.py`)

**功能**：Flask Web服务，提供HTTP API

**主要路由**：
- `POST /chat`: 对话接口
- `POST /action/vision-test`: 视力检测接口
- `GET /api/profile/*`: 用户数据接口
- `GET /api/diagnosis/*`: 疾病信息接口

**代码结构**：
```python
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    # 1. 获取user_id（从JSON、cookies或生成新ID）
    # 2. 获取text和image
    # 3. 调用MedicalAssistant.process_request()
    # 4. 返回JSON响应
    # 5. 设置cookie保存user_id

class MedicalAssistant:
    def process_request(self, user_id, text_input, image_file):
        # 1. 懒加载Orchestrator和VisionTester
        # 2. 调用orchestrator.process()
        # 3. 返回结果
```

**保存内容**：
- Flask路由定义
- 用户ID管理（cookie + localStorage）
- 懒加载机制（避免启动时阻塞）
- 错误处理和超时保护

---

## 📦 安装与配置

### 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- 摄像头（用于视力检测）

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd eye_test_model-master
   ```

2. **安装Python依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **安装TensorFlow（用于MediaPipe）**
   ```bash
   pip install tensorflow==2.13.0
   # 或使用CPU版本
   pip install tensorflow-cpu==2.13.0
   ```

4. **下载Qwen模型**
   -https://modelscope.cn/models/qwen/Qwen-1_8B-Chat/summary
   - 模型应放在 `./Qwen-1_8B-Chat/` 目录
   - 或修改 `backend/tools/local_qwen_api.py` 中的模型路径

5. **准备OCT分类模型**
   -数据来源：https://data.mendeley.com/datasets/sncdhf53xc/4
   - 训练模型：`python training/train_oct_pytorch.py`
   - 或使用已有的 `models/best_model.pth`
6. **Lora微调**
   本项目目前未使用大量对话数据进行微调，只是使用自建数据集进行了微调，后面请使用对话数据集微调
   推荐使用数据：
   视觉问答数据集：https://www.selectdataset.com/dataset/18ad58ca06bf3dad37714b9abed39ade
   整个医疗领域的对话数据集：https://github.com/FreedomIntelligence/Huatuo-26M?tab=readme-ov-file
### 配置文件

- **模型路径**：在 `backend/tools/local_qwen_api.py` 中配置
- **记忆文件**：`user_memory.json`
- **上传目录**：`uploads/`

---

## 🚀 使用方法

### CLI模式

**多轮对话模式**：
```bash
python backend/main.py --mode chat
```

**单次测试模式**：
```bash
python backend/main.py --mode smoke --chat "我最近看东西有点模糊"
```

**参数说明**：
- `--mode`: `chat`（多轮对话）或 `smoke`（单次测试）
- `--chat`: 测试文本（smoke模式）
- `--prompt`: LLM测试提示词
- `--image`: OCT测试图像路径

### Web模式

1. **启动Flask服务**：
   ```bash
   python backend/app.py
   ```

2. **访问前端**：
   - 打开浏览器访问 `http://127.0.0.1:5000`
   - 或直接打开 `ui5.html` 文件

3. **使用功能**：
   - **AI医生**：输入症状描述，自动触发工具
   - **视力检测**：点击"开始视力检测"按钮
   - **OCT分析**：上传OCT图像进行分析
   - **疾病诊断**：查看疾病详情和建议

---

## 🎓 模型训练

### 1. OCT分类模型训练

**训练脚本**：`training/train_oct_pytorch.py`

**使用方法**：
```bash
python training/train_oct_pytorch.py \
    --data_dir archive/RetinalOCT_Dataset \
    --output_dir models \
    --epochs 20 \
    --batch_size 16
```

**输出**：
- `models/best_model.pth`: 最佳模型
- `models/class_to_idx.json`: 类别映射
- `models/training_loss_curve.png`: 训练损失曲线

**数据集结构**：
```
RetinalOCT_Dataset/
├── train/
│   ├── AMD/
│   ├── CNV/
│   └── ...
├── val/
└── test/
```

### 2. Qwen模型微调

**数据准备**：
```bash
python training/process_huatuo_datasets.py
```

**LoRA微调**：
```bash
python training/finetune_qwen.py
```

**配置说明**：
- 模型路径：`./Qwen-1_8B-Chat`
- 输出目录：`./qwen_finetuned_model`
- LoRA参数：`r=16, α=32`
- 训练数据：`data/train_data.jsonl`

**输出**：
- `qwen_finetuned_model/adapter_model.safetensors`: LoRA适配器
- `qwen_finetuned_model/finetune_loss_curve.png`: 微调损失曲线

### 3. 强化学习微调

**训练奖励模型**：
```bash
python training/train_reward_model.py
```

**PPO微调**：
```bash
python training/rl_finetune_qwen.py
```

---

## 📡 API文档

### POST /chat

**功能**：对话接口，处理用户文本和图像输入

**请求**：
```json
{
  "text": "我最近看东西有点模糊",
  "user_id": "user_123",
  "image": "<base64或FormData>"
}
```

**响应**：
```json
{
  "status": "ok",
  "answer": "根据您的描述...",
  "action": null,
  "data": {
    "risk_level": "medium",
    "followups": ["模糊持续多久了？", "是看远模糊还是看近模糊？"],
    "risk_reason": "视力下降症状",
    "vision_test": {"left": 0.8, "right": 0.6},
    "oct_result": "AMD",
    "report_path": "logs/reports/user_123_20251210.txt"
  }
}
```

### POST /action/vision-test

**功能**：执行视力检测

**请求**：
```json
{
  "user_id": "user_123"
}
```

**响应**：
```json
{
  "status": "ok",
  "message": "视力检测完成",
  "data": {
    "vision_test": {"left": 0.8, "right": 0.6}
  }
}
```

### GET /api/profile/health-data

**功能**：获取用户健康数据

**响应**：
```json
{
  "status": "ok",
  "current_vision": 0.7,
  "vision_change": 0.1,
  "eye_care_hours": 12,
  "eye_care_change": 3
}
```

### GET /api/diagnosis/disease?type=近视

**功能**：获取疾病详情

**响应**：
```json
{
  "name": "近视",
  "description": "近视是指...",
  "symptoms": ["看远模糊", "眯眼", "视疲劳"],
  "causes": ["遗传因素", "长时间近距离用眼"],
  "treatment": ["佩戴眼镜", "激光手术"],
  "prevention": ["20-20-20法则", "增加户外活动"]
}
```

---

## 📊 架构图表

系统提供了4张架构和流程图，可通过以下命令生成：

```bash
python training/generate_architecture_diagrams.py
```

生成的图表：
1. **system_architecture.png** - 系统整体架构图
2. **data_preprocessing_flow.png** - 数据预处理流程图
3. **model_finetuning_flow.png** - 模型微调流程图
4. **inference_flow.png** - 推理服务执行流程图

详细说明请参考 `training/README_DIAGRAMS.md`

---

## 🔍 技术栈

### 后端
- **Flask**: Web框架
- **PyTorch**: 深度学习框架
- **Transformers**: 大语言模型库
- **PEFT**: 参数高效微调（LoRA）
- **OpenCV**: 计算机视觉
- **MediaPipe**: 手势和人脸检测

### 前端
- **HTML5/CSS3**: 界面
- **JavaScript**: 交互逻辑
- **Fetch API**: HTTP请求

### 模型
- **Qwen-1.8B-Chat**: 基础大语言模型
- **ResNet50**: OCT图像分类
- **LoRA**: 参数高效微调技术

---

## 📝 注意事项

1. **摄像头权限**：视力检测功能需要摄像头权限
2. **模型加载**：首次运行需要加载模型，可能需要较长时间
3. **GPU加速**：建议使用GPU加速LLM推理
4. **TensorFlow版本**：MediaPipe需要TensorFlow 2.13.0
5. **用户记忆**：`user_memory.json` 会在项目根目录自动创建
---

## 📧 联系方式

2402744495@qq.com

---

**最后更新**: 2025-12-10
