# 0_hello_deeplearning.py
print("=== 第0步：深度学习'Hello World' ===")
print("目标：理解最基本的神经网络，就像1+1=2一样简单")

# 1. 导入必要的库
import torch
import torch.nn as nn
import numpy as np

print("\n🎯 目标：用神经网络学习 y = 2x + 1")
print("   输入x: 1, 2, 3, 4")
print("   输出y: 3, 5, 7, 9 (因为 y = 2x + 1)")
print("   让网络自己发现这个规律！")

# 2. 准备最简单的数据
# 创建训练数据
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0]], dtype=torch.float32)

print(f"\n📊 训练数据:")
for i in range(len(x_train)):
    print(f"  x={x_train[i].item()}, y={y_train[i].item()}")


# 3. 定义最简单的神经网络
# 只有一个神经元！没有隐藏层！
class SimplestNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 一个线性层：y = wx + b
        # 输入1个特征，输出1个值
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# 4. 创建模型
model = SimplestNet()
print("\n🧠 模型结构（超级简单！）:")
print(model)
print(f"可学习参数:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.data}")

# 5. 查看初始预测（还没训练，所以是随机的）
print("\n🔮 训练前的预测（很可能是错的）:")
with torch.no_grad():  # 不计算梯度
    for x in x_train:
        prediction = model(x)
        print(f"  输入 {x.item():.1f} → 预测 {prediction.item():.4f}")

# 6. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # 随机梯度下降

print("\n🎯 损失函数: 均方误差 (MSE)")
print("   衡量预测值与真实值的差距，越小越好")
print("🎯 优化器: SGD (随机梯度下降)")
print("   学习率: 0.01 (每次调整的步伐大小)")

# 7. 开始训练！最简单的训练循环
print("\n🚀 开始训练！")
epochs = 2000
for epoch in range(epochs):  # 重复学习很多次
    # 1. 用当前参数做预测
    outputs = model(x_train)

    # 2. 计算预测有多糟糕
    loss = criterion(outputs, y_train)

    # 3. 分析错误原因
    optimizer.zero_grad()  # 忘记之前的错误
    loss.backward()  # 分析这次错在哪

    # 4. 调整参数
    optimizer.step()  # 修正错误

    # 每10个epoch打印一次
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch + 1:3d}/{epochs}, Loss: {loss.item():.6f}")

# 8. 查看训练后的参数
print("\n✅ 训练完成！")
print(f"\n📈 学到的参数:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.data}")
    if 'weight' in name:
        print(f"    网络学到的 w ≈ {param.data.item():.4f}")
    else:
        print(f"    网络学到的 b ≈ {param.data.item():.4f}")

print("\n🎯 真实的参数应该是: w=2, b=1")

# 9. 测试模型
print("\n🔮 训练后的预测:")
with torch.no_grad():
    for x in x_train:
        prediction = model(x)
        true_y = 2 * x.item() + 1
        error = abs(prediction.item() - true_y)
        print(f"  输入 {x.item():.1f} → 预测 {prediction.item():.4f}, 真实 {true_y}, 误差 {error:.4f}")

# 10. 在新数据上测试
print("\n🧪 在新数据上测试（泛化能力）:")
test_x = torch.tensor([[5.0], [6.0], [100]], dtype=torch.float32)
with torch.no_grad():
    for x in test_x:
        prediction = model(x)
        true_y = 2 * x.item() + 1
        print(f"  输入 {x.item():.1f} → 预测 {prediction.item():.4f}, 真实 {true_y}")

print("\n" + "=" * 50)
print("🎉 恭喜！你完成了第一个深度学习模型！")
print("=" * 50)

# 不同学习率的表现：
# lr=0.1：可能震荡，但收敛快
# lr=0.01：效果最好
# lr=0.001：50个epoch学不完
# lr=0.0001：基本没学