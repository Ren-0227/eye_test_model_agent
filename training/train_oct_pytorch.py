import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
from tqdm import tqdm
import argparse
import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt

class OCTDataset(Dataset):
    """OCT Image dataset."""
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with filepath and labels
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.df.iloc[idx]["filepath"]
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]["labels"]
        
        # Convert label to index
        if isinstance(label, str):
            # If labels are strings, convert to indices
            unique_labels = sorted(self.df["labels"].unique())
            label = unique_labels.index(label)
        
        if self.transform:
            image = self.transform(image)

        return image, label

def load_data_from_splits(data_dir):
    """
    Load OCT image data from train/val/test directory structure
    """
    splits = ['train', 'val', 'test']
    dataframes = {}
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory {split_dir} does not exist")
            
        filepath = []
        labels = []
        classes = os.listdir(split_dir)
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            images = os.listdir(class_dir)
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                filepath.append(img_path)
                labels.append(class_name)
        
        df = pd.DataFrame({"filepath": filepath, "labels": labels})
        dataframes[split] = df
        print(f"{split} set size: {len(df)}")
        
    return dataframes['train'], dataframes['val'], dataframes['test']


def get_data_transforms():
    """
    Define data transforms for training and validation
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(num_classes, pretrained=True):
    """
    Create CNN model for OCT image classification
    """
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes)
    )
    return model

def plot_loss_curves(train_losses, val_losses, output_dir):
    """
    绘制训练和验证loss曲线图
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('OCT Classification Model - Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    loss_plot_path = os.path.join(output_dir, "training_loss_curve.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"Loss曲线图已保存到: {loss_plot_path}")
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, output_dir):
    """
    Train the OCT image classification model
    """
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Validation]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device).long()
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'loss': loss.item()})
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} '
              f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')
        
        # Save best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            model_path = os.path.join(output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, model_path)
            print(f'Saved best model with accuracy: {best_acc:.4f}')

    # 绘制并保存loss图
    plot_loss_curves(train_losses, val_losses, output_dir)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_acc': best_acc
    }

def main(data_dir, output_dir, epochs=10, batch_size=16, use_split_folders=False):
    """
    Main function to train the OCT image classification model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to absolute paths
    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {data_dir}")
    train_df, val_df, test_df = load_data_from_splits(data_dir)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = OCTDataset(train_df, transform=train_transform)
    valid_dataset = OCTDataset(val_df, transform=val_transform)
    test_dataset = OCTDataset(test_df, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Get class information
    all_labels = pd.concat([train_df["labels"], val_df["labels"], test_df["labels"]])
    classes = sorted(all_labels.unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    num_classes = len(classes)
    
    print(f"Classes: {classes}")
    print(f"Number of classes: {num_classes}")
    
    # Save class mapping
    class_mapping_path = os.path.join(output_dir, "class_to_idx.json")
    with open(class_mapping_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Class mapping saved to: {class_mapping_path}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(num_classes, pretrained=True)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("Starting training...")
    history = train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, 
                         device, epochs, output_dir)
    
    # Final evaluation on test set
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_accuracy = test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_accuracy,
        'class_to_idx': class_to_idx,
        'num_classes': num_classes
    }, final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    history_data = {
        'train_losses': [float(x) for x in history['train_losses']],
        'val_losses': [float(x) for x in history['val_losses']],
        'train_accuracies': [float(x) for x in history['train_accuracies']],
        'val_accuracies': [float(x) for x in history['val_accuracies']],
        'best_val_accuracy': float(history['best_acc']),
        'test_accuracy': float(test_accuracy)
    }
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"Training history saved to: {history_path}")

    print("Training completed!")
    return model, history
    print("="*50)
    print("Training completed successfully!")
    print(f"Best validation accuracy: {history['best_acc']:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Models and metadata saved to: {output_dir}")
    print("="*50)
    
    return model, history
    
    print("="*50)
    print("Training completed successfully!")
    print(f"Best validation accuracy: {history['best_acc']:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Models and metadata saved to: {output_dir}")
    print("="*50)
    
    return model, history

if __name__ == "__main__":
    # Default paths for your specific setup
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "archive", "RetinalOCT_Dataset")
    DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "models")
    
    parser = argparse.ArgumentParser(description="Train OCT image classifier using PyTorch")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"Path to dataset root with train/val/test subfolders (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Where to save the trained model (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.epochs, args.batch_size)