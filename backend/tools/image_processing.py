# image_processing.py
import os
import json
import torch
from torchvision import transforms
from PIL import Image

IMG_SIZE = 224

# 默认模型路径
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "best_model.pth")
DEFAULT_CLASS_MAPPING_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "class_to_idx.json")

# 加载类别映射
try:
    with open(DEFAULT_CLASS_MAPPING_PATH, 'r', encoding='utf-8') as f:
        CLASS_TO_IDX = json.load(f)
    # 创建反向映射（索引到类别名称）
    IDX_TO_CLASS = {idx: class_name for class_name, idx in CLASS_TO_IDX.items()}
    # 创建中英文对照字典
    CLASS_LABELS = {class_name: class_name for class_name in CLASS_TO_IDX.keys()}
except Exception as e:
    print(f"加载类别映射失败: {e}")
    CLASS_TO_IDX = {}
    IDX_TO_CLASS = {}
    CLASS_LABELS = {}

# 数据预处理
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class OCTClassifier(torch.nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None  # 初始化模型变量
load_error = None  # 记录模型加载失败原因

def analyze_image(image_path):
    """分析图片并返回概率最大的标签的中文名称"""
    try:
        # 确保模型已加载
        if model is None:
            msg = load_error or "模型未加载"
            raise Exception(msg)

        # 确保图片路径有效
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件未找到: {image_path}")

        print(f"图片路径: {image_path}")  # 调试信息

        img = Image.open(image_path).convert('RGB')
        img_tensor = test_transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # 添加批处理维度

        with torch.no_grad():
            inputs = img_tensor.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 获取概率分布
            _, predicted_class = torch.max(probabilities, 1)

        # 获取预测的类别名称
        if IDX_TO_CLASS:
            # 使用动态加载的类别映射
            predicted_class_name = IDX_TO_CLASS.get(predicted_class.item(), "未知类别")
            return predicted_class_name
        else:
            # 使用默认类别映射
            if predicted_class.item() < len(list(CLASS_LABELS.keys())):
                predicted_class_name = list(CLASS_LABELS.keys())[predicted_class.item()]
                return CLASS_LABELS[predicted_class_name]
            else:
                return "未知类别"
    except FileNotFoundError as e:
        raise FileNotFoundError(f"文件未找到: {e}")
    except PermissionError:
        raise PermissionError("无权限访问文件，请检查文件权限")
    except Exception as e:
        raise Exception(f"图片分析出错: {str(e)}")


def load_model(model_path=DEFAULT_MODEL_PATH, verbose: bool = False):
    """加载模型；默认静默，便于被主流程导入时不刷屏"""
    global model, load_error, CLASS_TO_IDX, IDX_TO_CLASS, CLASS_LABELS
    try:
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载类别映射
        class_mapping_path = DEFAULT_CLASS_MAPPING_PATH
        if not os.path.exists(class_mapping_path):
            # 尝试在模型目录下查找class_to_idx.json
            model_dir = os.path.dirname(model_path)
            class_mapping_path = os.path.join(model_dir, "class_to_idx.json")
            
        if os.path.exists(class_mapping_path):
            with open(class_mapping_path, 'r', encoding='utf-8') as f:
                CLASS_TO_IDX = json.load(f)
            # 创建反向映射（索引到类别名称）
            IDX_TO_CLASS = {idx: class_name for class_name, idx in CLASS_TO_IDX.items()}
            # 创建中英文对照字典
            CLASS_LABELS = {class_name: class_name for class_name in CLASS_TO_IDX.keys()}
        
        # 创建模型实例
        num_classes = len(CLASS_TO_IDX) if CLASS_TO_IDX else 9
        model = OCTClassifier(num_classes=num_classes)
        
        # 加载模型权重，兼容无 base_model 前缀的 state_dict；当前 head 结构已与 best_model.pth 对齐
        checkpoint = torch.load(model_path, map_location=device, weights_only=True) if "weights_only" in torch.load.__code__.co_varnames else torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if all(not k.startswith("base_model.") for k in state_dict.keys()):
            state_dict = {f"base_model.{k}": v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
            
        model = model.to(device)
        model.eval()  # 设置为评估模式
        if verbose:
            print(f"模型加载成功，类别数: {num_classes}")
        load_error = None
    except Exception as e:
        load_error = f"模型加载失败: {str(e)}"
        print(load_error)

# 在模块导入时自动加载模型
load_model(verbose=False)

def reload_model(model_path=DEFAULT_MODEL_PATH):
    """重新加载模型"""
    global model
    if model is not None:
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    load_model(model_path)

if __name__ == "__main__":
    # 测试代码（示例相对路径，需自行放置图片）
    image_path = os.path.join("archive", "RetinalOCT_Dataset", "test", "AMD", "amd_test_1350.jpg")
    try:
        result = analyze_image(image_path)
        print(f"图片分析结果: {result}")
    except FileNotFoundError:
        print("模型文件或图片文件未找到，请检查路径是否正确")
    except Exception as e:
        print(f"图片分析出错: {str(e)}")