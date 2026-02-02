"""
随机梯度下降 (SGD) 详解
核心思想：每次用1个样本计算梯度并更新参数
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. 理论基础
print("🎯 随机梯度下降 (SGD) 理论讲解")
print("="*50)

theory = """
📖 数学原理：
目标：最小化损失函数 L(θ) = Σ[l(θ; x_i, y_i)] / n

批量梯度下降：θ = θ - η * ∇L(θ) = θ - η * Σ[∇l(θ; x_i, y_i)] / n
随机梯度下降：θ = θ - η * ∇l(θ; x_i, y_i)  ← 用单个样本！

🔍 核心区别：梯度估计的样本数量
- Batch GD: 用全部n个样本
- SGD: 用1个样本
- Mini-batch: 用m个样本 (1 < m < n)
"""

print(theory)

# 2. 创建示例数据
np.random.seed(42)
n_samples = 100
x = np.random.randn(n_samples, 1)
true_w, true_b = 3.0, 2.0
y = true_w * x + true_b + np.random.randn(n_samples, 1) * 0.5

print(f"📊 生成 {n_samples} 个样本数据")
print(f"真实参数: w = {true_w:.2f}, b = {true_b:.2f}")

# 3. 实现SGD
def sgd_linear_regression(x, y, lr=0.01, epochs=10):
    n = len(x)
    w, b = 0.0, 0.0
    w_history, b_history, loss_history = [w], [b], []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        epoch_loss = 0

        for i, idx in enumerate(indices):
            xi, yi = x[idx][0], y[idx][0]
            y_pred = w * xi + b
            error = y_pred - yi

            # SGD更新（只用当前样本）
            w_grad = 2 * error * xi
            b_grad = 2 * error
            w -= lr * w_grad
            b -= lr * b_grad

            epoch_loss += error ** 2

        loss_history.append(epoch_loss / n)
        w_history.append(w)
        b_history.append(b)

    return w, b, w_history, b_history, loss_history

# 4. 实现Batch GD对比
def batch_gd_linear_regression(x, y, lr=0.01, epochs=100):
    n = len(x)
    w, b = 0.0, 0.0
    w_history, b_history, loss_history = [w], [b], []

    for epoch in range(epochs):
        y_pred = w * x.flatten() + b
        errors = y_pred - y.flatten()

        # Batch GD更新（用所有样本）
        w_grad = 2 * np.mean(errors * x.flatten())
        b_grad = 2 * np.mean(errors)
        w -= lr * w_grad
        b -= lr * b_grad

        loss_history.append(np.mean(errors ** 2))
        w_history.append(w)
        b_history.append(b)

    return w, b, w_history, b_history, loss_history

# 5. 运行对比
w_sgd, b_sgd, w_hist_sgd, b_hist_sgd, loss_hist_sgd = sgd_linear_regression(x, y, epochs=5)
w_batch, b_batch, w_hist_batch, b_hist_batch, loss_hist_batch = batch_gd_linear_regression(x, y, epochs=20)

print(f"\n📈 训练结果对比:")
print(f"真实参数: w={true_w:.4f}, b={true_b:.4f}")
print(f"SGD结果:  w={w_sgd:.4f}, b={b_sgd:.4f}")
print(f"Batch结果: w={w_batch:.4f}, b={b_batch:.4f}")

# 6. 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 损失曲线对比
axes[0,0].plot(loss_hist_sgd, 'r-', label='SGD', linewidth=5)
axes[0,0].plot(loss_hist_batch, 'b-', label='Batch GD', linewidth=5)
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].set_title('损失函数下降曲线')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 参数w的变化
axes[0,1].plot(w_hist_sgd, 'r-', label='SGD', linewidth=2)
axes[0,1].plot(w_hist_batch, 'b-', label='Batch GD', linewidth=2)
axes[0,1].axhline(y=true_w, color='k', linestyle='--', alpha=0.5, label='真实w')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('w值')
axes[0,1].set_title('参数w的变化')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 参数b的变化
axes[1,0].plot(b_hist_sgd, 'r-', label='SGD', linewidth=2)
axes[1,0].plot(b_hist_batch, 'b-', label='Batch GD', linewidth=2)
axes[1,0].axhline(y=true_b, color='k', linestyle='--', alpha=0.5, label='真实b')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('b值')
axes[1,0].set_title('参数b的变化')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 收敛路径对比
n_points = min(20, len(w_hist_sgd))
axes[1,1].plot(w_hist_sgd[:n_points], b_hist_sgd[:n_points], 'r-o', label='SGD路径', markersize=4)
axes[1,1].plot(w_hist_batch[:n_points], b_hist_batch[:n_points], 'b-s', label='Batch GD路径', markersize=4)
axes[1,1].plot(w_hist_sgd[0], b_hist_sgd[0], 'ko', markersize=8, label='起点')
axes[1,1].plot(true_w, true_b, 'g*', markersize=12, label='目标点')
axes[1,1].set_xlabel('w值')
axes[1,1].set_ylabel('b值')
axes[1,1].set_title('参数空间收敛路径')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. SGD的优缺点总结
print("\n📊 SGD 优缺点总结:")
print("✅ 优点：计算速度快、内存效率高、能跳出局部最优")
print("❌ 缺点：收敛不稳定、最终精度低、需要仔细调参")
print("🎯 适用场景：大规模数据集、在线学习、深度学习训练")

# 8. PyTorch示例
print("\n💻 PyTorch中的SGD使用:")
pytorch_code = """
# 基本使用
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 带动量的SGD（常用）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练循环
for epoch in range(epochs):
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()  # SGD更新
"""
print(pytorch_code)

print("\n🎯 关键总结:")
print("1. SGD: θ = θ - η * ∇l(θ; x_i, y_i) ← 单个样本梯度")
print("2. 实际多用Mini-batch SGD (batch_size=32/64/128)")
print("3. 配合Momentum使用加速收敛")
print("4. 学习率衰减很重要！")