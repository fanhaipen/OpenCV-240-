"""
第9天：PyTorch完整实战（可运行版本）
每一段代码都可以独立运行
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset

print("🔥 PyTorch版本:", torch.__version__)
print("🔥 CUDA可用:", torch.cuda.is_available())


# ============================================================================
# 1. 张量基础操作（完整可运行）
# ============================================================================

def tensor_basics_demo():
    print("\n" + "=" * 60)
    print("1. 张量基础操作")
    print("=" * 60)

    # 创建张量
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.zeros(2, 3)
    z = torch.ones(3, 2)

    print("基本张量创建:")
    print(f"x = {x}")
    print(f"y (2x3零矩阵) = \n{y}")
    print(f"z (3x2一矩阵) = \n{z}")

    # 张量运算
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])

    print("\n张量运算:")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"点积 = {torch.dot(a, b)}")

    # 矩阵乘法
    matrix1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    matrix2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    result = torch.matmul(matrix1, matrix2)
    print(f"矩阵乘法: \n{result}")

    # 形状操作
    tensor_2d = torch.randn(2, 3)
    print(f"\n形状操作:")
    print(f"原始: {tensor_2d.shape}")
    print(f"转置: {tensor_2d.t().shape}")
    print(f"重塑为3x2: {tensor_2d.view(3, 2).shape}")


# 运行第一段
tensor_basics_demo()


# ============================================================================
# 2. 自动求导系统（完整可运行）
# ============================================================================

def autograd_demo():
    print("\n" + "=" * 60)
    print("2. 自动求导系统")
    print("=" * 60)

    # 简单线性函数求导
    x = torch.tensor(2.0, requires_grad=True)
    w = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)

    y = w * x + b  # y = 3 * 2 + 1 = 7
    y.backward()

    print("简单线性函数:")
    print(f"y = {y.item()}")
    print(f"dy/dx = {x.grad}")  # 应该是3
    print(f"dy/dw = {w.grad}")  # 应该是2
    print(f"dy/db = {b.grad}")  # 应该是1

    # 复杂函数求导
    x2 = torch.tensor(2.0, requires_grad=True)
    y2 = x2 ** 2 + 3 * x2 + 1
    y2.backward()

    print(f"\n复杂函数 y = x² + 3x + 1:")
    print(f"当x=2时, y = {y2.item()}")
    print(f"导数 dy/dx = {x2.grad}")  # 2 * 2 + 3 = 7


# 运行第二段
autograd_demo()


# ============================================================================
# 3. 自定义神经网络（完整可运行）
# ============================================================================

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def network_demo():
    print("\n" + "=" * 60)
    print("3. 神经网络模块")
    print("=" * 60)

    # 创建网络
    model = SimpleNet(input_size=10, hidden_size=20, output_size=3)

    print("网络结构:")
    print(model)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数数量: {total_params:,}")

    # 前向传播演示
    batch_size = 4
    x = torch.randn(batch_size, 10)
    output = model(x)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 查看各层参数
    print("\n各层参数形状:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")


# 运行第三段
network_demo()


# ============================================================================
# 4. 数据加载器（完整可运行）
# ============================================================================

def dataloader_demo():
    print("\n" + "=" * 60)
    print("4. 数据加载器")
    print("=" * 60)

    # 创建模拟数据
    num_samples = 100
    input_size = 5
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, 3, (num_samples,))  # 3个类别

    # 创建数据集和数据加载器
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    print(f"数据集大小: {len(dataset)}")
    print(f"批次数量: {len(dataloader)}")
    print(f"批次大小: {dataloader.batch_size}")

    # 查看第一个批次
    for batch_idx, (data, target) in enumerate(dataloader):
        print(f"\n第一个批次:")
        print(f"数据形状: {data.shape}")
        print(f"标签形状: {target.shape}")
        print(f"数据示例: {data[0]}")
        print(f"标签示例: {target[0]}")
        break  # 只看第一个批次


# 运行第四段
dataloader_demo()


# ============================================================================
# 5. 完整训练流程（完整可运行）
# ============================================================================

def complete_training_demo():
    print("\n" + "=" * 60)
    print("5. 完整训练流程")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(42)

    # 生成简单的二分类数据
    num_samples = 200
    X = torch.randn(num_samples, 2)
    # 创建简单的二分类问题（根据点的位置）
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).long()

    # 分割数据集
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 定义简单模型
    class BinaryClassifier(nn.Module):
        def __init__(self):
            super(BinaryClassifier, self).__init__()
            self.fc1 = nn.Linear(2, 10)
            self.fc2 = nn.Linear(10, 2)  # 二分类，输出2个类别
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # 创建模型、损失函数、优化器
    model = BinaryClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练参数
    num_epochs = 100
    train_losses = []
    train_accuracies = []

    print("开始训练...")

    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, target)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # 计算epoch统计
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # 每5个epoch打印一次
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.1f}%')

    # 最终测试
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"\n最终测试准确率: {test_accuracy:.1f}%")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, 'g-', linewidth=2)
    plt.title('训练准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return model, train_losses, train_accuracies


# 运行第五段（训练过程）
model, losses, accuracies = complete_training_demo()


# ============================================================================
# 6. 模型保存与加载（完整可运行）
# ============================================================================

def save_load_demo():
    print("\n" + "=" * 60)
    print("6. 模型保存与加载")
    print("=" * 60)

    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(5, 3)

        def forward(self, x):
            return self.fc(x)

    # 创建模型实例并训练一下（简单演示）
    model = SimpleModel()
    x = torch.randn(10, 5)
    y = torch.randint(0, 3, (10,))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 一次训练步骤（为了有参数可保存）
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    # 保存模型
    torch.save(model.state_dict(), 'model_weights.pth')
    print("模型权重已保存到 'model_weights.pth'")

    # 加载模型
    new_model = SimpleModel()
    new_model.load_state_dict(torch.load('model_weights.pth'))
    print("模型权重已加载")

    # 验证加载的模型
    test_input = torch.randn(1, 5)
    original_output = model(test_input)
    loaded_output = new_model(test_input)

    print(f"原始模型输出: {original_output.detach().numpy()}")
    print(f"加载模型输出: {loaded_output.detach().numpy()}")
    print("输出是否接近:", torch.allclose(original_output, loaded_output, atol=1e-6))


# 运行第六段
save_load_demo()


# ============================================================================
# 7. GPU使用演示（完整可运行）
# ============================================================================

def gpu_demo():
    print("\n" + "=" * 60)
    print("7. GPU使用演示")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建张量
    x = torch.randn(3, 3)
    print(f"原始张量设备: CPU")

    if torch.cuda.is_available():
        # 移动到GPU
        x_gpu = x.to(device)
        print(f"GPU张量设备: {x_gpu.device}")

        # GPU运算
        y_gpu = torch.matmul(x_gpu, x_gpu.t())
        print(f"GPU运算结果形状: {y_gpu.shape}")

        # 移回CPU
        y_cpu = y_gpu.cpu()
        print(f"移回CPU后的设备: {y_cpu.device}")
    else:
        print("CUDA不可用，使用CPU完成演示")
        y_cpu = torch.matmul(x, x.t())
        print(f"CPU运算结果形状: {y_cpu.shape}")


# 运行第七段
gpu_demo()


# ============================================================================
# 8. 线性回归实战（完整可运行）
# ============================================================================

def linear_regression_demo():
    print("\n" + "=" * 60)
    print("8. 线性回归实战")
    print("=" * 60)

    # 生成数据
    torch.manual_seed(42)
    X = torch.linspace(-1, 1, 100).reshape(-1, 1)
    true_w = 2.0
    true_b = 1.0
    y = true_w * X + true_b + 0.1 * torch.randn(X.size())

    # 定义线性回归模型
    class LinearRegression(nn.Module):
        def __init__(self):
            super(LinearRegression, self).__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    model = LinearRegression()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练模型
    losses = []
    for epoch in range(3000):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 50 == 0:
            print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')

    # 获取训练后的参数
    w_pred = model.linear.weight.item()
    b_pred = model.linear.bias.item()

    print(f"\n真实参数: w = {true_w:.3f}, b = {true_b:.3f}")
    print(f"预测参数: w = {w_pred:.3f}, b = {b_pred:.3f}")

    # 绘制结果
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(X.numpy(), y.numpy(), alpha=0.7, label='数据点')
    plt.plot(X.numpy(), model(X).detach().numpy(), 'r-', linewidth=2, label='拟合直线')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('线性回归拟合')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 运行第八段
linear_regression_demo()

print("\n🎉 所有代码段都成功运行完成！")
print("✅ 你已经掌握了PyTorch的核心功能")