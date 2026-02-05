import os
# 🔧 提前设置环境变量确保CUDA确定性
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # CUDA >= 10.2确定性

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# transforms模块已移除，使用默认数据加载
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import seaborn as sns
import sys
sys.path.insert(0, '/kaggle/input/efficient-1/EfficientNet-PyTorch-MLOps')

# 使用标准交叉熵损失函数
ARC_FACE_AVAILABLE = False
print("✓ 使用标准交叉熵损失函数")

# 正确导入EfficientNet
try:
    # 尝试多种导入方式
    try:
        from efficientnet_pytorch import EfficientNet
        print("✓ 从 efficientnet_pytorch 导入成功")
    except ImportError:
        try:
            from efficientnet_pytorch.model import EfficientNet
            print("✓ 从 efficientnet_pytorch.model 导入成功")
        except ImportError:
            # 最后尝试相对导入
            import efficientnet_pytorch.model
            EfficientNet = efficientnet_pytorch.model.EfficientNet
            print("✓ 通过模块属性访问导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    print("请检查 efficientnet_pytorch 模块是否正确安装")
    raise


def load_efficientnet_with_attention(model_name='efficientnet-b0', num_classes=1000, num_heads=8, reduction=8):
    """
    初始化EfficientNet-B0随机权重，注意力模块使用随机初始化
    
    Args:
        model_name (str): 模型名称
        num_classes (int): 分类数量
        num_heads (int): 注意力头数量
    
    Returns:
        model: 配置好的带注意力的模型
    """
    print("正在初始化EfficientNet-B0随机权重模型...")
    
    # 直接创建带注意力的模型（使用随机初始化）
    print("正在初始化带RP感知注意力的模型...")
    model = EfficientNetWithAttention(model_name=model_name, num_classes=num_classes, num_heads=num_heads, reduction=reduction)
    
    # 验证模型参数状态
    total_params = sum(p.numel() for p in model.parameters())
    attention_params = sum(p.numel() for p in model.attention.parameters())
    backbone_params = total_params - attention_params
    
    print(f"✓ 模型初始化完成:")
    print(f"  - 主干网络参数: {backbone_params:,} (随机初始化)")
    print(f"  - 注意力模块参数: {attention_params:,} (随机初始化)")
    print(f"  - 总参数量: {total_params:,}")
    
    return model


from tqdm import tqdm


class RPAwareAttentionLayer(nn.Module):
    """
    Args:
        in_channels (int): 输入通道数
        num_heads (int): 注意力头数量

    """
    def __init__(self, in_channels):
        super(RPAwareAttentionLayer, self).__init__()
        
        hidden_dim = in_channels // 10
        
        # 使用 1×1 卷积进行通道注意力
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 生成方向敏感的注意力权重
        attention_weights = torch.sigmoid(self.conv2(self.relu(self.conv1(x))))
        return x * attention_weights


class EfficientNetWithAttention(nn.Module):
    def __init__(self, model_name='efficientnet-b0', num_classes=1000, num_heads=8):
        super(EfficientNetWithAttention, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        self.attention = RPAwareAttentionLayer(in_channels=1280)

    def forward(self, x):
        x = self.efficientnet.extract_features(x)
        x = self.attention(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.efficientnet._dropout(x)
        x = self.efficientnet._fc(x)
        return x

    def extract_features(self, x):
        return self.efficientnet.extract_features(x)


# 评估模型函数
# 模型训练
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, num_classes):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    cm = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))
                pbar.update(1)

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # 评估训练集和验证集
        val_loss, val_accuracy, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
        train_accuracy = evaluate_model(model, train_loader, criterion, device)[1]  # 获取训练准确率
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # # 更新混淆矩阵
        # _, preds = torch.max(outputs, 1)
        # for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
        #     cm[true, pred] += 1
        # 更新混淆矩阵（仅在验证集上）
        for true, pred in zip(val_labels, val_preds):
            cm[true, pred] += 1

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, '
              f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        scheduler.step()

    # 保存指标和混淆矩阵
    save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, cm)

    return train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, accuracy, all_preds, all_labels  # 返回损失、准确率、预测和真实标签


# 数据保存
def save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, cm, filename_prefix='metrics'):
    # 保存损失和准确率到 CSV 文件
    metrics_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Val Loss': val_losses,
        'Train Accuracy': train_accuracies,
        'Val Accuracy': val_accuracies
    })
    metrics_df.to_csv(f'/kaggle/working/{filename_prefix}_metrics.csv', index=False)
    print(f'Metrics saved to /kaggle/working/{filename_prefix}_metrics.csv')

    # 保存混淆矩阵到 TXT 文件
    np.savetxt(f'/kaggle/working/{filename_prefix}_confusion_matrix.txt', cm, fmt='%d')
    print(f'Confusion matrix saved to /kaggle/working/{filename_prefix}_confusion_matrix.txt')


# 绘制损失和准确率的函数
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs):
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.xticks(range(0, epochs + 1, 10))  # x轴刻度为0, 10, 20, ...
    plt.yticks(np.arange(0, 1.1, 0.2))  # y轴刻度为0.0, 0.2, 0.4, ..., 1.0
    plt.legend()
    plt.grid(False)
    plt.show()

    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(range(0, epochs + 1, 10))  # x轴刻度为0, 10, 20, ...
    plt.yticks(np.arange(0, 1.1, 0.2))  # y轴刻度为0.0, 0.2, 0.4, ..., 1.0
    plt.legend()
    plt.grid(False)
    plt.show()


# 绘制混淆矩阵的函数
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


# 提取特征向量的函数
def extract_features(model, loader, device):
    model.eval()  # 设置模型为评估模式
    all_features = []
    all_labels = []

    with torch.no_grad():  # 禁用梯度计算
        for images, labels in loader:
            images = images.to(device)

            # 提取中间层输出（特征向量）
            features = model.extract_features(images)
            features = features.view(features.size(0), -1)  # 将特征向量展平为二维

            all_features.extend(features.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_features), np.array(all_labels)


# 保存特征向量到文件
def save_features(features, labels, filename='val_features.npy'):
    np.save(f'/kaggle/working/{filename}', {'features': features, 'labels': labels})
    print(f"Features saved to /kaggle/working/{filename}")


def main(args, num_classes):
    print(f"Using device: {args.device}")
    
    # 🔧 设置确定性训练环境
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    import random
    random.seed(42)
    
    # 设置CUDA确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)  # PyTorch 1.8+
    
    print("✓ 确定性环境已设置 (seed=42)")

    # 直接使用默认变换
    train_dataset = ImageFolder(os.path.join(args.data, 'train'))
    val_dataset = ImageFolder(os.path.join(args.data, 'val'))

    # 使用默认DataLoader配置
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers
    )

    # 使用合并后的可靠加载方式
    model = load_efficientnet_with_attention(
        model_name='efficientnet-b0', 
        num_classes=num_classes, 
        num_heads=args.num_heads,
        reduction=args.reduction
    )
    print("✓ 模型初始化完成，使用随机权重")
    
    model = model.to(args.device)

    # 配置损失函数
    criterion = nn.CrossEntropyLoss()
    print("✓ 使用标准交叉熵损失")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 将num_classes传递给train_model函数
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer, scheduler, args.device, args.epochs, num_classes
    )

    # 绘制训练和验证损失及准确率
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, args.epochs)

    # 提取验证集特征
    val_features, val_labels = extract_features(model, val_loader, args.device)
    save_features(val_features, val_labels, filename='val_features.npy')

    # 评估模型并绘制混淆矩阵
    val_loss, val_accuracy, val_preds, val_labels = evaluate_model(model, val_loader, criterion, args.device)
    cm = metrics.confusion_matrix(val_labels, val_preds)

    # 保存指标和混淆矩阵
    save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, cm)

    # 绘制混淆矩阵
    plot_confusion_matrix(cm, class_names=[str(i) for i in range(len(val_dataset.classes))])


def get_default_args():
    """获取默认训练参数"""
    class Args:
        def __init__(self):
            self.data = '/kaggle/input/your-dataset/data'
            self.epochs = 50
            self.batch_size = 16
            self.lr = 0.0005
            self.workers = 4
            self.image_size = 224
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.num_heads = 8
            self.reduction = 10  # RP感知门控的通道缩减比例
    return Args()

if __name__ == '__main__':
    # Jupyter/Colab兼容性处理
    import sys
    
    # 检查是否在Jupyter环境中
    if 'ipykernel' in sys.modules or 'colab' in sys.modules:
        print("检测到Jupyter/Colab环境，使用默认参数...")
        args = get_default_args()
        num_classes = 6
        main(args, num_classes)
    else:
        # 命令行模式
        parser = argparse.ArgumentParser(description='EfficientNet Classification')
        parser.add_argument('--data', metavar='DIR',
                            default='/kaggle/input/your-dataset/data',
                            help='path to dataset')
        parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
        parser.add_argument('--batch-size', default=16, type=int, help='batch size')
        parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
        parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
        parser.add_argument('--image_size', default=224, type=int, help='image size')
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                            help='device to use for training')
        parser.add_argument('--num_heads', default=8, type=int, help='number of attention heads')
        parser.add_argument('--reduction', default=10, type=int, help='channel reduction ratio for RP-aware gates')

        num_classes = 6  # 这里设置分类的类别

        args = parser.parse_args()
        main(args, num_classes)

