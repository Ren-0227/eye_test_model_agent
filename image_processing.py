# image_processing.py
import os
import torch
from torchvision import transforms
from PIL import Image

IMG_SIZE = 224

# 定义中英文对照字典
CLASS_LABELS = {
    "baineizhang": "白内障",
    "binglixingjinshi": "病理性近视",
    "gaoxueya": "高血压",
    "huangbanbingbian": "黄斑病变",
    "jinski": "青光眼",
    "putong": "普通",
    "qingguang": "青光眼",
    "shiwangmotuoluo": "视网膜脱落",
    "tangniaobing": "糖尿病"
}

# 数据预处理
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 模型定义
class OCTClassifier(torch.nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features, 512),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

# 加载模型
model_path = "best_model.pth"  # 模型文件路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None  # 初始化模型变量

def load_model():
    """加载模型"""
    global model
    try:
        model = OCTClassifier(num_classes=len(CLASS_LABELS))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()  # 设置为评估模式
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")

load_model()  # 加载模型

def analyze_image(image_path):
    """分析图片并返回概率最大的标签的中文名称"""
    try:
        # 确保模型已加载
        if model is None:
            raise Exception("模型未加载")

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
        predicted_class_name = list(CLASS_LABELS.keys())[predicted_class.item()]
        # 返回对应的中文名称
        return CLASS_LABELS[predicted_class_name]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"文件未找到: {e}")
    except PermissionError:
        raise PermissionError("无权限访问文件，请检查文件权限")
    except Exception as e:
        raise Exception(f"图片分析出错: {str(e)}")


if __name__ == "__main__":
    # 测试代码
    image_path = r"C:\Users\91144\Desktop\eyes_model\dataset\baineizhang\4670_right.jpg"  # 测试图片路径

    try:
        # 调用图片分析函数
        result = analyze_image(image_path)
        print(f"图片分析结果: {result}")
    except FileNotFoundError:
        print("模型文件或图片文件未找到，请检查路径是否正确")
    except Exception as e:
        print(f"图片分析出错: {str(e)}")