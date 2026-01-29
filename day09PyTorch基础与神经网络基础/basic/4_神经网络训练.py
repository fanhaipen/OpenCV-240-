# 3_神经网络训练_改进版.py
print("=== 第3步：理解神经网络训练（改进版）===")

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler  # 新增

# 1. 创建数据（相同）
np.random.seed(42)
n_samples = 20

cats_weight = np.random.uniform(2, 4, n_samples)
cats_height = np.random.uniform(20, 30, n_samples)
cats = np.column_stack([cats_weight, cats_height])
cats_labels = np.zeros(n_samples)

dogs_weight = np.random.uniform(8, 12, n_samples)
dogs_height = np.random.uniform(40, 50, n_samples)
dogs = np.column_stack([dogs_weight, dogs_height])
dogs_labels = np.ones(n_samples)

X = np.vstack([cats, dogs])
y = np.concatenate([cats_labels, dogs_labels])

# 2. 数据标准化（新增关键步骤！）
print("\n2. 数据标准化（重要！）")
print("   - 将体重和身高缩放到相同范围")
print("   - 避免身高影响远大于体重")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"标准化前 - 体重范围: {X[:, 0].min():.1f}到{X[:, 0].max():.1f}")
print(f"标准化前 - 身高范围: {X[:, 1].min():.1f}到{X[:, 1].max():.1f}")
print(f"标准化后 - 体重范围: {X_scaled[:, 0].min():.2f}到{X_scaled[:, 0].max():.2f}")
print(f"标准化后 - 身高范围: {X_scaled[:, 1].min():.2f}到{X_scaled[:, 1].max():.2f}")

# 可视化标准化前后的对比
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(cats[:, 0], cats[:, 1], c='blue', alpha=0.7, label='猫')
plt.scatter(dogs[:, 0], dogs[:, 1], c='red', alpha=0.7, label='狗')
plt.xlabel('体重 (kg)')
plt.ylabel('身高 (cm)')
plt.title('原始数据')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:n_samples, 0], X_scaled[:n_samples, 1], c='blue', alpha=0.7, label='猫')
plt.scatter(X_scaled[n_samples:, 0], X_scaled[n_samples:, 1], c='red', alpha=0.7, label='狗')
plt.xlabel('体重 (标准化)')
plt.ylabel('身高 (标准化)')
plt.title('标准化后数据')
plt.legend()

# 3. 转换为张量
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)  # 使用标准化后的数据
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)


# 4. 定义神经网络（添加批标准化）
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 8)  # 增加到8个神经元
        self.bn1 = nn.BatchNorm1d(8)  # 批标准化
        self.layer2 = nn.Linear(8, 4)  # 新增一层
        self.bn2 = nn.BatchNorm1d(4)  # 批标准化
        self.layer3 = nn.Linear(4, 1)  # 输出层
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)  # 轻微dropout防止过拟合

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.sigmoid(x)
        return x


model = SimpleNN()

# 5. 改进的损失函数和优化器
print("\n3. 定义损失函数和优化器")
print("   - 损失函数: BCELoss（二分类交叉熵）")
print("   - 优化器: Adam（自适应学习率）")
print("   - 学习率: 0.01（更稳定）")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 改用Adam

# 6. 训练循环
print("\n4. 开始训练")
n_epochs = 200
train_losses = []
val_losses = []

# 分割训练集和验证集
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

for epoch in range(n_epochs):
    # 训练
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    train_losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 验证
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())

        # 计算准确率
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y_train_tensor).float().mean()

        val_predictions = (val_outputs > 0.5).float()
        val_accuracy = (val_predictions == y_val_tensor).float().mean()

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch [{epoch + 1:3d}/{n_epochs}], "
              f"训练损失: {loss.item():.4f}, 训练准确率: {accuracy.item():.4f}, "
              f"验证损失: {val_loss.item():.4f}, 验证准确率: {val_accuracy.item():.4f}")

# 7. 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练损失', alpha=0.7)
plt.plot(val_losses, label='验证损失', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失曲线')
plt.legend()
plt.grid(True, alpha=0.3)

# 8. 绘制决策边界
plt.subplot(1, 2, 2)
h = 0.02
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

model.eval()
with torch.no_grad():
    Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
    Z = (Z > 0.5).float().numpy()
    Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_scaled[:n_samples, 0], X_scaled[:n_samples, 1],
            c='blue', alpha=0.7, label='猫')
plt.scatter(X_scaled[n_samples:, 0], X_scaled[n_samples:, 1],
            c='red', alpha=0.7, label='狗')
plt.xlabel('特征1 (标准化)')
plt.ylabel('特征2 (标准化)')
plt.title('决策边界')
plt.legend()

plt.tight_layout()
plt.show()

# 9. 最终评估
print("\n5. 最终模型评估")
model.eval()
with torch.no_grad():
    all_outputs = model(X_tensor)
    predictions = (all_outputs > 0.5).float()
    accuracy = (predictions == y_tensor).float().mean()

    print(f"  最终训练准确率: {accuracy.item():.4f}")

    # 测试新样本
    test_samples = np.array([[3.0, 25.0],  # 猫
                             [10.0, 45.0],  # 狗
                             [5.0, 35.0]])  # 中间值

    # 需要先标准化
    test_scaled = scaler.transform(test_samples)
    test_tensor = torch.tensor(test_scaled, dtype=torch.float32)
    test_pred = model(test_tensor)

    print(f"\n  测试预测:")
    for i, (weight, height) in enumerate(test_samples):
        prob = test_pred[i].item()
        pred = '狗' if prob > 0.5 else '猫'
        print(f"    样本{i + 1}: 体重{weight}kg, 身高{height}cm")
        print(f"      预测概率: {prob:.4f} → 预测: {pred}")
        if i == 0:
            print(f"      期望: 猫 (概率接近0)")
        elif i == 1:
            print(f"      期望: 狗 (概率接近1)")
        else:
            print(f"      期望: 不确定 (中间值)")

print("\n" + "=" * 50)
print("✅ 改进版训练完成！模型表现更好！")
print("=" * 50)