# day09_first_nn.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

print("🎯 第9天：创建第一个神经网络")
print("=" * 50)

# 1. 生成数据
print("1. 生成数据集...")
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
print(f"数据集形状: X={X.shape}, y={y.shape}")

# 可视化数据
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.title("原始数据分布")
plt.xlabel("特征1")
plt.ylabel("特征2")

# 2. 数据预处理
print("2. 数据预处理...")
# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

print(f"训练集: {X_train_tensor.shape}, 测试集: {X_test_tensor.shape}")

# 3. 定义神经网络
print("3. 定义神经网络模型...")


class SimpleNN(nn.Module):
    """简单的三层神经网络"""

    def __init__(self, input_size=2, hidden_size=10, output_size=1):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x


# 创建模型实例
model = SimpleNN()
print(f"模型结构:\n{model}")

# 4. 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. 训练模型
print("4. 开始训练模型...")
num_epochs = 1000
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    # 训练模式
    model.train()

    # 前向传播
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 计算准确率
    with torch.no_grad():
        # 训练集准确率
        train_predictions = (y_pred > 0.5).float()
        train_accuracy = (train_predictions == y_train_tensor).float().mean()

        # 测试集准确率
        model.eval()
        y_test_pred = model(X_test_tensor)
        test_loss = criterion(y_test_pred, y_test_tensor)
        test_predictions = (y_test_pred > 0.5).float()
        test_accuracy = (test_predictions == y_test_tensor).float().mean()

    # 记录损失和准确率
    train_losses.append(loss.item())
    test_losses.append(test_loss.item())
    train_accuracies.append(train_accuracy.item())
    test_accuracies.append(test_accuracy.item())

    # 每100个epoch打印一次
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"训练损失: {loss.item():.4f}, 训练准确率: {train_accuracy.item():.4f}, "
              f"测试损失: {test_loss.item():.4f}, 测试准确率: {test_accuracy.item():.4f}")

# 6. 可视化训练结果
print("5. 可视化训练结果...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
axes[0].plot(train_losses, label='训练损失', alpha=0.7)
axes[0].plot(test_losses, label='测试损失', alpha=0.7)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('损失曲线')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 准确率曲线
axes[1].plot(train_accuracies, label='训练准确率', alpha=0.7)
axes[1].plot(test_accuracies, label='测试准确率', alpha=0.7)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('准确率曲线')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# 7. 可视化决策边界
print("6. 可视化决策边界...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 生成网格点
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测整个网格
with torch.no_grad():
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    Z = model(grid_tensor)
    Z = (Z > 0.5).float().numpy()
    Z = Z.reshape(xx.shape)

# 决策边界
axes[0].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                edgecolors='k', cmap='viridis', alpha=0.7)
axes[0].set_xlabel('特征1')
axes[0].set_ylabel('特征2')
axes[0].set_title('训练集决策边界')

# 测试集
axes[1].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                edgecolors='k', cmap='viridis', alpha=0.7)
axes[1].set_xlabel('特征1')
axes[1].set_ylabel('特征2')
axes[1].set_title('测试集决策边界')

plt.tight_layout()
plt.show()

# 8. 模型评估
print("7. 最终模型评估...")
model.eval()
with torch.no_grad():
    # 训练集评估
    y_train_pred = model(X_train_tensor)
    train_predictions = (y_train_pred > 0.5).float()
    train_accuracy = (train_predictions == y_train_tensor).float().mean()

    # 测试集评估
    y_test_pred = model(X_test_tensor)
    test_predictions = (y_test_pred > 0.5).float()
    test_accuracy = (test_predictions == y_test_tensor).float().mean()

print(f"最终训练准确率: {train_accuracy.item():.4f}")
print(f"最终测试准确率: {test_accuracy.item():.4f}")

# 9. 保存模型
print("8. 保存模型...")
torch.save(model.state_dict(), 'simple_nn_model.pth')
print("模型已保存为 'simple_nn_model.pth'")

# 10. 加载模型示例
print("9. 加载模型示例...")
new_model = SimpleNN()
new_model.load_state_dict(torch.load('simple_nn_model.pth'))
new_model.eval()

# 测试加载的模型
with torch.no_grad():
    test_pred = new_model(X_test_tensor[:5])  # 预测前5个样本
    print(f"前5个测试样本的预测概率: {test_pred.squeeze().numpy()}")
    print(f"前5个测试样本的实际标签: {y_test[:5]}")

print("\n" + "=" * 50)
print("✅ 第一个神经网络完成！")
print("=" * 50)