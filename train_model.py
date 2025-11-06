import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.multiprocessing
from tqdm import tqdm  # 新增导入

# --------------------
# 0. 全局配置（新增特殊数据集标识）
# --------------------
SPECIAL_CLASSES = ['jinshi', 'shiwangmotuoluo', 'huangbanbingbian', 'qingguang']
CLASSES = [
    "baineizhang", "binglixingjinshi", "gaoxueya", 
    "huangbanbingbian", "jinski", "putong",
    "qingguang", "shiwangmotuoluo", "tangniaobing"
]
DATA_DIR = r"C:\Users\91144\Desktop\eyes_model\dataset"
IMG_SIZE = 224
BATCH_SIZE = 32

# --------------------
# 1. 智能数据加载模块（核心改进）
# --------------------
def parse_filename(img_name, class_name):
    """支持黄斑病变混合格式的智能解析器[2,5](@ref)"""
    base = os.path.splitext(img_name)[0]
    
    # 特殊病例处理规则（黄斑病变新增两种格式）
    if class_name in SPECIAL_CLASSES:
        # 格式1: Macular Scar1 → patient_id='MS1', eye='both'
        if re.match(r'^Macular Scar\d+$', base, re.IGNORECASE):
            patient_id = 'MS' + re.sub(r'\D', '', base)
            return {'patient_id': patient_id, 'eye': 'both'}
        
        # 格式2: 标准文件名（如4671_left）
        if '_' in base:
            parts = base.rsplit('_', 1)
            if parts[1].lower() in ['left','right']:
                return {'patient_id': parts[0], 'eye': parts[1].lower()}
        
        # 格式3: 原始文件名作为patient_id
        return {'patient_id': base, 'eye': 'both'}
    
    # 标准病例处理
    if '_' in base:
        parts = base.rsplit('_', 1)
        if len(parts) == 2 and parts[1].lower() in ['left','right']:
            return {'patient_id': parts[0], 'eye': parts[1].lower()}
    
    raise ValueError(f"无法解析的文件名: {img_name}")

def build_dataframe():
    """支持混合格式的数据框构建[2,5](@ref)"""
    data = []
    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_path):
            continue
            
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.png','.jpg','.jpeg')):
                continue
                
            try:
                meta = parse_filename(img_name, class_name)
                data.append({
                    **meta,
                    'path': os.path.join(class_path, img_name),
                    'label': class_idx,
                    'data_type': 'special' if class_name in SPECIAL_CLASSES else 'standard'
                })
            except Exception as e:
                print(f"跳过文件 {img_name}: {str(e)}")
                continue
                
    return pd.DataFrame(data)

# --------------------
# 2. 数据预处理模块（新增弹性形变增强）
# --------------------
class ElasticDeform:
    """医学影像专用弹性形变[5](@ref)"""
    def __init__(self, alpha=1000, sigma=30):
        self.alpha = alpha
        self.sigma = sigma
        
    def __call__(self, img):
        # 实现弹性形变逻辑（此处简化为示例）
        return img  # 实际应添加形变代码

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    ElasticDeform(),  # 新增弹性形变
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --------------------
# 3. 数据集类改进（支持动态增强）
# --------------------
class OCTDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.class_weights = self._calc_class_weights()
        
    def _calc_class_weights(self):
        counts = self.df['label'].value_counts().sort_index()
        return 1. / counts
    def __len__(self):
        """返回数据集大小"""
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        
        # 确保标签是标量而不是数组
        label = row['label']
        if isinstance(label, (np.ndarray, pd.Series)):
            label = label.item()  # 转换为标量
            
        # 对特殊数据应用动态增强
        if row['data_type'] == 'special':
            if isinstance(self.transform.transforms[0], ElasticDeform):
                self.transform.transforms[0].alpha = 1500
                
        if self.transform:
            img = self.transform(img)
            
        return img, label  # 确保返回的是标量标签
    
    def get_sampler(self):
        """加权采样解决类别不平衡[9](@ref)"""
        weights = self.df['label'].map(self.class_weights)
        # 修改为将weights转换为numpy数组后再转换为torch张量
        weights_tensor = torch.as_tensor(weights.values, dtype=torch.double)
        return WeightedRandomSampler(weights_tensor, len(weights), replacement=True)

# --------------------
# 4. 模型定义（兼容预训练权重）
# --------------------
class OCTClassifier(torch.nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # 更新加载方式[10](@ref)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features, 512),
            # Simply remove the BatchNorm layer
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

# --------------------
# 5. 训练流程（Windows多进程兼容）
# --------------------
# --------------------
# 6. 训练流程实现
# --------------------
def train_model(model, train_loader, val_loader, num_epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 调试1: 检查数据加载器
    print("\n=== 数据加载调试 ===")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    sample_batch = next(iter(train_loader))
    print(f"输入形状: {sample_batch[0].shape}, 标签形状: {sample_batch[1].shape}")
    
    # 调试2: 检查模型结构
    print("\n=== 模型结构调试 ===")
    print(model)
    test_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    test_output = model(test_input)
    print(f"测试输出形状: {test_output.shape}")
    print(f"预期输出形状: [1, {len(CLASSES)}]")
    
    # 调试3: 确认设备
    print("\n=== 设备调试 ===")
    print(f"使用设备: {device}")
    print(f"模型设备: {next(model.parameters()).device}")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_acc = 0.0
    
    # 等待用户确认
    input("\n调试信息已显示，按Enter键开始训练...")
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # 使用tqdm添加进度条
        train_loop = tqdm(train_loader, desc=f'Train Epoch {epoch+1}', leave=False)
        for inputs, labels in train_loop:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # 更新进度条信息
            train_loop.set_postfix(loss=loss.item())
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        val_loop = tqdm(val_loader, desc=f'Val Epoch {epoch+1}', leave=False)
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
                # 更新进度条信息
                val_loop.set_postfix(loss=loss.item())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'\nBest val Acc: {best_acc:.4f}')
    return model
if __name__ == '__main__':
    # Windows多进程配置[8](@ref)
    torch.multiprocessing.freeze_support()
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    # 数据准备
    df = build_dataframe()
    
    # 按患者划分数据集（保持左右眼一致）[2](@ref)
    patients = df['patient_id'].unique()
    train_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=42)
    train_patients, val_patients = train_test_split(train_patients, test_size=0.1, random_state=42)
    
    # 合并特殊数据集到训练集[4](@ref)
    special_df = df[df['data_type'] == 'special']
    train_df = pd.concat([
        df[(df['patient_id'].isin(train_patients)) & (df['data_type'] == 'standard')],
        special_df
    ])
    
    # 创建数据加载器
    train_dataset = OCTDataset(train_df, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_dataset.get_sampler(),
        num_workers=0 if os.name == 'nt' else 4,  # Windows兼容[8](@ref)
        pin_memory=True
    )
    
    # 创建验证集数据加载器
    val_df = df[df['patient_id'].isin(val_patients)]
    val_dataset = OCTDataset(val_df, transform=train_transform)  # 使用相同的transform
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 4,
        pin_memory=True
    )
    
    # 初始化模型并训练
    model = OCTClassifier(num_classes=len(CLASSES))
    trained_model = train_model(model, train_loader, val_loader, num_epochs=25)
    
    # 保存完整模型（可选）
    torch.save(trained_model, 'full_model.pth')