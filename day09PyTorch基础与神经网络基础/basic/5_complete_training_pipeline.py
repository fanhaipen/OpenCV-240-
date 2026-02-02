# 4_complete_training_pipeline.py
print("=== 第4步：完整深度学习训练流程 ===")
print("从数据准备到模型部署的完整项目")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import os

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)

print("🔧 1. 数据准备和预处理")

# 1.1 创建合成数据集（模拟真实数据）
print("\n📊 生成合成数据集...")
X, y = make_classification(
    n_samples=1000,  # 1000个样本
    n_features=20,  # 20个特征
    n_informative=15,  # 15个有用特征
    n_redundant=5,  # 5个冗余特征
    n_classes=3,  # 3个类别
    n_clusters_per_class=1,  # 每个类别1个簇
    random_state=42
)

print(f"数据集形状: X{X.shape}, y{y.shape}")
print(f"类别分布: {np.bincount(y)}")

# 1.2 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1.3 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\n📋 数据集划分:")
print(f"  训练集: {X_train.shape[0]} 样本")
print(f"  验证集: {X_val.shape[0]} 样本")
print(f"  测试集: {X_test.shape[0]} 样本")


# 1.4 创建PyTorch数据集类
class ClassificationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 创建数据集实例
train_dataset = ClassificationDataset(X_train, y_train)
val_dataset = ClassificationDataset(X_val, y_val)
test_dataset = ClassificationDataset(X_test, y_test)

# 1.5 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\n📦 数据加载器:")
print(f"  训练批次: {len(train_loader)}")
print(f"  验证批次: {len(val_loader)}")
print(f"  测试批次: {len(test_loader)}")

print("\n🔧 2. 定义神经网络模型")


class AdvancedClassifier(nn.Module):
    """更复杂的神经网络分类器"""

    def __init__(self, input_size=20, hidden_sizes=[64, 32], num_classes=3, dropout_rate=0.3):
        super(AdvancedClassifier, self).__init__()

        # 构建隐藏层
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # 批归一化，加速训练
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # 防止过拟合
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 创建模型实例
model = AdvancedClassifier(input_size=20, hidden_sizes=[128, 64, 32], num_classes=3)
print(f"\n🧠 模型结构:")
print(model)

# 计算参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 参数统计:")
print(f"  总参数: {total_params:,}")
print(f"  可训练参数: {trainable_params:,}")

print("\n🔧 3. 定义训练组件")

# 3.1 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  使用设备: {device}")

model = model.to(device)

# 3.2 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑防止过拟合
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# 3.3 学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

print(f"\n⚙️ 训练配置:")
print(f"  损失函数: CrossEntropyLoss")
print(f"  优化器: AdamW")
print(f"  初始学习率: 0.001")
print(f"  权重衰减: 1e-4")

print("\n🔧 4. 定义训练和评估函数")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


print("\n🔧 5. 开始训练")

# 记录训练历史
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'learning_rates': []
}

# 早停设置
patience = 10
patience_counter = 0
best_val_acc = 0.0
best_model_state = None

num_epochs = 50
start_time = time.time()

print(f"\n🚀 开始训练 {num_epochs} 个epoch...")
print("-" * 60)

for epoch in range(num_epochs):
    epoch_start = time.time()

    # 训练
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

    # 验证
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    # 调整学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # 记录历史
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['learning_rates'].append(current_lr)

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        # 保存最佳模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc,
        }, 'best_model.pth')
    else:
        patience_counter += 1

    # 打印进度
    epoch_time = time.time() - epoch_start
    print(f'Epoch {epoch + 1:2d}/{num_epochs} | '
          f'训练损失: {train_loss:.4f} | 训练准确率: {train_acc:6.2f}% | '
          f'验证损失: {val_loss:.4f} | 验证准确率: {val_acc:6.2f}% | '
          f'学习率: {current_lr:.6f} | 时间: {epoch_time:.1f}s')

    # 早停检查
    if patience_counter >= patience:
        print(f"\n⏹️  早停触发: 验证准确率 {patience} 个epoch未提升")
        break

# 加载最佳模型
model.load_state_dict(best_model_state)
total_time = time.time() - start_time

print(f"\n✅ 训练完成! 总时间: {total_time:.1f}秒")
print(f"🏆 最佳验证准确率: {best_val_acc:.2f}%")

print("\n🔧 6. 训练结果可视化")

# 创建可视化图表
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 6.1 损失曲线
axes[0, 0].plot(history['train_loss'], label='训练损失', linewidth=2, color='blue')
axes[0, 0].plot(history['val_loss'], label='验证损失', linewidth=2, color='red')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('损失曲线')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 6.2 准确率曲线
axes[0, 1].plot(history['train_acc'], label='训练准确率', linewidth=2, color='blue')
axes[0, 1].plot(history['val_acc'], label='验证准确率', linewidth=2, color='red')
axes[0, 1].axhline(y=best_val_acc, color='green', linestyle='--',
                   label=f'最佳: {best_val_acc:.1f}%')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].set_title('准确率曲线')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 6.3 学习率变化
axes[0, 2].plot(history['learning_rates'], linewidth=2, color='green')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Learning Rate')
axes[0, 2].set_title('学习率变化')
axes[0, 2].grid(True, alpha=0.3)

# 6.4 训练统计
axes[1, 0].axis('off')
stats_text = (
    f"训练统计\n\n"
    f"训练时间: {total_time:.1f}秒\n"
    f"总Epoch数: {len(history['train_loss'])}\n"
    f"最佳验证准确率: {best_val_acc:.2f}%\n"
    f"最终训练准确率: {history['train_acc'][-1]:.2f}%\n"
    f"最终验证准确率: {history['val_acc'][-1]:.2f}%\n"
    f"批次大小: {batch_size}\n"
    f"模型参数: {trainable_params:,}"
)
axes[1, 0].text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 6.5 混淆矩阵（简化版）
axes[1, 1].axis('off')
# 这里可以添加真正的混淆矩阵，为保持简单先留空

# 6.6 特征重要性（简化版）
axes[1, 2].axis('off')
# 这里可以添加特征重要性分析

plt.suptitle('深度学习训练结果分析', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n🔧 7. 模型测试和评估")

# 在测试集上评估最佳模型
test_loss, test_acc = evaluate(model, test_loader, criterion, device)

print(f"\n📊 测试集结果:")
print(f"  测试损失: {test_loss:.4f}")
print(f"  测试准确率: {test_acc:.2f}%")

# 详细分类报告
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = outputs.max(1)

        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

# 计算每个类别的准确率
from sklearn.metrics import classification_report

print(f"\n📈 详细分类报告:")
print(classification_report(all_targets, all_predictions,
                            target_names=['类别 0', '类别 1', '类别 2']))

print("\n🔧 8. 模型保存和部署")

# 8.1 保存完整模型
torch.save(model, 'complete_model.pth')
print("✅ 完整模型已保存为 'complete_model.pth'")

# 8.2 保存模型状态（推荐方式）
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_,
    'input_size': 20,
    'hidden_sizes': [128, 64, 32],
    'num_classes': 3,
    'test_accuracy': test_acc
}, 'model_checkpoint.pth')
print("✅ 模型检查点已保存为 'model_checkpoint.pth'")

# 8.3 保存为TorchScript（生产环境）
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')
print("✅ TorchScript模型已保存为 'model_scripted.pt'")

print("\n🔧 9. 模型加载和推理示例")


# 演示如何加载和使用模型
def load_and_predict(model_path, input_data):
    """加载模型并进行预测"""
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)

    # 创建新模型实例
    loaded_model = AdvancedClassifier(
        input_size=checkpoint['input_size'],
        hidden_sizes=checkpoint['hidden_sizes'],
        num_classes=checkpoint['num_classes']
    )

    # 加载权重
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.to(device)
    loaded_model.eval()

    # 数据预处理（使用保存的scaler参数）
    scaler_mean = checkpoint['scaler_mean']
    scaler_scale = checkpoint['scaler_scale']
    input_scaled = (input_data - scaler_mean) / scaler_scale

    # 预测
    with torch.no_grad():
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
        output = loaded_model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1)

    return prediction.cpu().numpy(), probabilities.cpu().numpy()


# 测试加载的模型
print(f"\n🧪 测试模型加载和预测:")
test_sample = X_test[0:1]  # 取第一个测试样本
prediction, probabilities = load_and_predict('model_checkpoint.pth', test_sample)

print(f"输入样本形状: {test_sample.shape}")
print(f"真实标签: {y_test[0]}")
print(f"预测标签: {prediction[0]}")
print(f"预测概率: {probabilities[0].round(3)}")
print(f"预测是否正确: {'✅' if prediction[0] == y_test[0] else '❌'}")

print("\n🔧 10. 创建预测函数")


def predict_new_sample(model, scaler, sample, class_names=None):
    """对新样本进行预测的便捷函数"""
    if class_names is None:
        class_names = ['类别 0', '类别 1', '类别 2']

    # 数据预处理
    sample_scaled = scaler.transform(sample.reshape(1, -1))
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        output = model(sample_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1)

    pred_class = prediction.item()
    confidence = probabilities[0][pred_class].item()

    print(f"\n🔮 预测结果:")
    print(f"  预测类别: {class_names[pred_class]} (索引: {pred_class})")
    print(f"  置信度: {confidence:.3f}")
    print(f"  所有类别概率:")
    for i, prob in enumerate(probabilities[0]):
        class_name = class_names[i] if class_names else f'类别 {i}'
        print(f"    {class_name}: {prob:.3f}")

    return pred_class, confidence


# 测试新样本预测
print(f"\n🎯 新样本预测演示:")
new_sample = np.random.randn(20)  # 随机生成一个样本
predict_new_sample(model, scaler, new_sample)

print("\n" + "=" * 60)
print("🎉 完整深度学习流程完成！")
print("=" * 60)
print("\n📚 学习总结:")
print("✅ 1. 数据准备和预处理")
print("✅ 2. 神经网络模型设计")
print("✅ 3. 训练循环实现")
print("✅ 4. 模型评估和验证")
print("✅ 5. 模型保存和加载")
print("✅ 6. 新样本预测")
print("\n🚀 下一步: 尝试修改参数，观察对模型性能的影响！")