"""
Mini-batch 梯度下降详解
核心思想：每次用一小批数据计算梯度，平衡速度和稳定性
"""

import numpy as np
import matplotlib.pyplot as plt

# 理论基础
print("🎯 Mini-batch GD: 平衡SGD速度和Batch GD稳定性")
print("batch_size选择: 1=SGD, n=Batch GD, 2-256=Mini-batch")

# 创建数据
np.random.seed(42)
n_samples = 100
x = np.random.randn(n_samples, 1)
true_w, true_b = 3.0, 2.0
y = true_w * x + true_b + np.random.randn(n_samples, 1) * 0.5

# Mini-batch GD实现
def train_gd(x, y, batch_size=32, lr=0.01, epochs=10):
    n = len(x)
    w, b = 0.0, 0.0
    loss_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch_indices = indices[i:min(i+batch_size, n)]
            x_batch = x[batch_indices].flatten()
            y_batch = y[batch_indices].flatten()

            # 计算梯度
            errors = w * x_batch + b - y_batch
            w_grad = 2 * np.mean(errors * x_batch)
            b_grad = 2 * np.mean(errors)

            # 更新参数
            w -= lr * w_grad
            b -= lr * b_grad
            loss_history.append(np.mean(errors ** 2))

    return w, b, loss_history

# 测试不同batch_size
batch_sizes = [8, 16, 32, 128]
results = []
loss_data = []

for bs in batch_sizes:
    w, b, loss_hist = train_gd(x, y, bs, epochs=500)
    method = "SGD" if bs == 1 else "Batch GD" if bs == 100 else f"Mini-batch({bs})"
    results.append((method, bs, w, b, loss_hist[-1], len(loss_hist)))
    loss_data.append(loss_hist)

# 结果显示
print("\n📊 结果对比:")
print("方法            | batch_size | 最终w   | 最终b   | 最终损失")
print("-"*60)
for method, bs, w, b, loss, _ in results:
    print(f"{method:15} | {bs:10} | {w:6.3f} | {b:6.3f} | {loss:8.6f}")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

colors = ['red', 'orange', 'green', 'blue']
labels = ['SGD (bs=1)', 'Mini-batch (bs=16)', 'Mini-batch (bs=32)', 'Batch GD (bs=100)']

# 损失曲线对比
ax1 = axes[0,0]
for i, loss_hist in enumerate(loss_data):
    step = max(1, len(loss_hist) // 50)
    x_vals = range(0, len(loss_hist), step)
    y_vals = loss_hist[::step]
    ax1.plot(x_vals, y_vals, color=colors[i], label=labels[i], linewidth=2)
ax1.set_xlabel('更新次数')
ax1.set_ylabel('损失')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 收敛速度对比
ax2 = axes[0,1]
threshold = 0.5
convergence_speed = []
for loss_hist in loss_data:  # 修复这里：从loss改为loss_data
    found = False
    for j, loss_val in enumerate(loss_hist):
        if loss_val < threshold:
            convergence_speed.append(j)
            found = True
            break
    if not found:
        convergence_speed.append(len(loss_hist))

ax2.bar(labels, convergence_speed, color=colors, alpha=0.7)
ax2.set_xlabel('方法')
ax2.set_ylabel('收敛速度')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# 梯度噪声对比
ax3 = axes[1,0]
noise_levels = []
for loss_hist in loss_data:  # 修复这里：从loss改为loss_data
    if len(loss_hist) > 10:
        noise = np.std(loss_hist[-10:]) / np.mean(loss_hist[-10:])
        noise_levels.append(noise)
    else:
        noise_levels.append(0)
ax3.bar(labels, noise_levels, color=colors, alpha=0.7)
ax3.set_xlabel('方法')
ax3.set_ylabel('梯度噪声')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

# 更新频率对比
ax4 = axes[1,1]
update_counts = [len(hist) for hist in loss_data]
ax4.bar(labels, update_counts, color=colors, alpha=0.7)
ax4.set_xlabel('方法')
ax4.set_ylabel('总更新次数')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# 总结
print("\n🎯 核心总结:")
print("1. SGD (bs=1): 更新快但噪声大")
print("2. Batch GD (bs=n): 稳定但收敛慢")
print("3. Mini-batch (bs=32): 最佳平衡点")
print("4. 深度学习常用: 32, 64, 128")

print("\n💻 PyTorch示例:")
print("DataLoader(dataset, batch_size=32)  # Mini-batch GD")
print("DataLoader(dataset, batch_size=1)    # SGD")
print("DataLoader(dataset, batch_size=len(dataset))  # Batch GD")