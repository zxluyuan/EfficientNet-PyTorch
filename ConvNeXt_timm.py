"""
基于timm库的ConvNeXt-Tiny快速集成训练脚本
完全对齐Enhanced_train_0.py的配置和指标输出
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from tqdm import tqdm

# 导入timm库
try:
    import timm
    TIMM_AVAILABLE = True
    print("✓ 成功导入timm库")
except ImportError:
    TIMM_AVAILABLE = False
    print("✗ 未找到timm库，请先安装: pip install timm")

def get_data_loaders(data_path, batch_size=16):
    """数据加载器 - 与Enhanced_train_0.py完全对齐"""
    # 单通道图像预处理（适用于RP图像等灰度图）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # 转换为单通道
        transforms.ToTensor(),
        # 单通道标准化
        transforms.Normalize(mean=[0.449], std=[0.227])
    ])
    
    train_dataset = ImageFolder(os.path.join(data_path, 'train'), transform=transform)
    val_dataset = ImageFolder(os.path.join(data_path, 'val'), transform=transform)
    
    return train_dataset, val_dataset
    
    # 完全对齐的DataLoader配置
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0  # 与要求一致
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # 与要求一致
        num_workers=0   # 与要求一致
    )
    
    return train_loader, val_loader, len(train_dataset.classes)

def create_convnext_model(num_classes=6):
    """创建ConvNeXt-Tiny模型 - 从头训练（单通道输入）"""
    if not TIMM_AVAILABLE:
        raise RuntimeError("timm库不可用，请先安装: pip install timm")
    
    # 从头训练ConvNeXt-Tiny（无预训练权重）
    model = timm.create_model(
        'convnext_tiny',
        pretrained=False,        # 不使用预训练权重
        num_classes=num_classes, # 你的类别数
        in_chans=1              # 单通道输入
    )
    
    return model

def train_epoch(model, loader, criterion, optimizer, device, epoch_desc=""):
    """训练一个epoch - 与Enhanced_train_0.py对齐"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with tqdm(total=len(loader), desc=epoch_desc, unit='batch') as pbar:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100.*correct/total
            })
            pbar.update(1)
    
    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    """评估模型 - 与Enhanced_train_0.py对齐"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(loader)
    
    return avg_loss, accuracy, all_preds, all_labels

def save_training_metrics(train_losses, val_losses, train_accs, val_accs, cm, output_dir='/kaggle/working/'):
    """保存训练指标 - 与Enhanced_train_0.py完全对齐"""
    # 保存CSV指标文件（格式完全一致）
    metrics_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Val Loss': val_losses, 
        'Train Accuracy': [acc/100.0 for acc in train_accs],  # 转换为小数
        'Val Accuracy': [acc/100.0 for acc in val_accs]      # 转换为小数
    })
    
    csv_path = os.path.join(output_dir, 'convnext_timm_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"✓ 训练指标已保存到: {csv_path}")
    
    # 保存混淆矩阵（格式完全一致）
    cm_path = os.path.join(output_dir, 'convnext_timm_confusion_matrix.txt')
    np.savetxt(cm_path, cm, fmt='%d')
    print(f"✓ 混淆矩阵已保存到: {cm_path}")

def main():
    """主训练函数 - 完全对齐Enhanced_train_0.py"""
    # 配置参数（与Enhanced_train_0.py完全一致）
    config = {
        'data_path': '/kaggle/input/your-dataset/data',
        'num_classes': 6,
        'epochs': 50,           # 相同训练轮数
        'batch_size': 16,
        'learning_rate': 0.0005, # 相同学习率
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=" * 50)
    print("🚀 基于timm的ConvNeXt-Tiny从头训练启动")
    print("=" * 50)
    print("🎯 训练模式: 单通道输入 | 无预训练权重 | 随机初始化")
    print(f"使用设备: {config['device']}")
    print(f"数据路径: {config['data_path']}")
    print(f"类别数量: {config['num_classes']}")
    print(f"训练轮数: {config['epochs']}")
    print(f"批处理大小: {config['batch_size']}")
    print(f"学习率: {config['learning_rate']}")
    
    # 设置随机种子确保可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 数据加载（完全对齐配置）
    print("\n📂 加载数据集...")
    train_dataset, val_dataset = get_data_loaders(
        config['data_path'], 
        config['batch_size']
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0
    )
    
    print(f"✓ 训练集样本数: {len(train_dataset)}")
    print(f"✓ 验证集样本数: {len(val_dataset)}")
    print(f"✓ 类别数量: {len(train_dataset.classes)}")
    print(f"✓ 训练集类别分布: {dict(zip(train_dataset.classes, np.bincount([y for _, y in train_dataset])))}")
    
    # 模型初始化（使用timm）
    print("\n🧠 初始化ConvNeXt-Tiny模型...")
    model = create_convnext_model(num_classes=len(train_dataset.classes))
    model = model.to(config['device'])
    print(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器（与Enhanced_train_0.py完全对齐）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # 训练循环
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0
    
    print(f"\n🏃 开始训练 ({config['epochs']} epochs)...")
    print("=" * 50)
    
    for epoch in range(config['epochs']):
        # 训练阶段
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config['device'],
            f'Epoch {epoch+1}/{config["epochs"]}'
        )
        
        # 验证阶段
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, config['device']
        )
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # 学习率调度
        scheduler.step()
        
        # 打印结果
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch [{epoch+1:2d}/{config["epochs"]}] - '
              f'LR: {current_lr:.6f} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 添加预警信息
        if epoch >= 2 and train_acc < 25:
            print("⚠️  警告: 训练准确率过低，可能存在以下问题:")
            print("   1. 数据标签可能不正确")
            print("   2. 学习率可能过低")
            print("   3. 数据预处理可能有问题")
            print("   4. 模型初始化可能有问题")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = '/kaggle/working/best_convnext_timm_model.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 保存最佳模型 (acc: {best_val_acc:.2f}%)")
    
    # 生成最终评估结果
    print("\n" + "=" * 50)
    print("📈 训练完成 - 生成最终报告")
    print("=" * 50)
    
    final_val_loss, final_val_acc, final_preds, final_labels = evaluate(
        model, val_loader, criterion, config['device']
    )
    
    # 生成混淆矩阵
    cm = metrics.confusion_matrix(final_labels, final_preds)
    
    # 保存所有结果
    save_training_metrics(
        train_losses, val_losses, train_accuracies, val_accuracies, cm
    )
    
    # 最终统计
    print(f"\n🎯 训练总结:")
    print(f"   最佳验证准确率: {best_val_acc:.2f}%")
    print(f"   最终验证准确率: {final_val_acc:.2f}%")
    print(f"   总训练轮数: {config['epochs']}")
    print(f"   模型保存路径: /kaggle/working/best_convnext_timm_model.pth")
    print(f"   指标文件路径: /kaggle/working/convnext_timm_metrics.csv")
    print(f"   混淆矩阵路径: /kaggle/working/convnext_timm_confusion_matrix.txt")
    print("=" * 50)

if __name__ == '__main__':
    # Kaggle环境检测
    if 'KAGGLE_CONTAINER_NAME' in os.environ or 'IN_KAGGLE' in os.environ:
        print("🔍 检测到Kaggle环境")
        os.environ['IN_KAGGLE'] = '1'
    
    # 运行主函数
    main()