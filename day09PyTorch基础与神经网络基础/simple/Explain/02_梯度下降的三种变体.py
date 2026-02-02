"""
梯度下降的三种变体对比 - 修复版
1. 批量梯度下降 (Batch Gradient Descent)
2. 随机梯度下降 (Stochastic Gradient Descent)
3. 小批量梯度下降 (Mini-batch Gradient Descent)
"""

import numpy as np
import matplotlib.pyplot as plt

print("=== 梯度下降的三种变体对比 ===")
print()

# 1. 创建模拟数据
np.random.seed(42)
x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = 2 * x_train + 1 + np.random.randn(5) * 0.5  # y = 2x + 1 + 噪声

print("📊 训练数据：")
for i, (x, y) in enumerate(zip(x_train, y_train), 1):
    print(f"  点{i}: x={x:.1f}, 真实y={y:.2f}, 目标y={2*x+1:.1f}")

# 2. 初始化参数
w_init, b_init = 0.0, 0.0
lr = 0.01
n_samples = len(x_train)

print(f"\n🎯 目标：找到 y = w*x + b 中的 w 和 b")
print(f"🤔 初始猜测：w={w_init}, b={b_init}")
print(f"📈 真实值：w≈2.0, b≈1.0")
print()

# 3. 定义损失函数
def compute_loss(w, b, x, y):
    """计算均方误差损失"""
    y_pred = w * x + b
    loss = np.mean((y_pred - y) ** 2)
    return loss

# 4. 批量梯度下降 (Batch Gradient Descent) - 修复：运行完整3个epoch
print("="*60)
print("1. 批量梯度下降 (Batch Gradient Descent)")
print("-"*60)
print("📌 特点：一次用所有数据计算梯度")
print("✅ 优点：梯度准确，方向稳定")
print("❌ 缺点：大数据集时太慢")
print()

def batch_gradient_descent(w, b, x, y, lr=0.01, epochs=3):
    """批量梯度下降 - 每次用全部数据更新"""
    w_history, b_history, loss_history = [w], [b], [compute_loss(w, b, x, y)]

    for epoch in range(epochs):
        print(f"\n📅 Epoch {epoch+1}:")

        # 1. 用当前参数预测所有点
        y_pred = w * x + b
        errors = y_pred - y

        # 2. 计算梯度（用所有数据的平均值）
        w_grad = 2 * np.mean(errors * x)
        b_grad = 2 * np.mean(errors)

        # 3. 更新参数
        w = w - lr * w_grad
        b = b - lr * b_grad

        # 4. 记录
        w_history.append(w)
        b_history.append(b)
        loss_history.append(compute_loss(w, b, x, y))

    return w, b, w_history, b_history, loss_history

# 运行批量梯度下降
w_batch, b_batch, w_hist_batch, b_hist_batch, loss_hist_batch = batch_gradient_descent(
    w_init, b_init, x_train, y_train, lr, epochs=3
)

# 5. 随机梯度下降 (SGD) - 修复：运行完整1个epoch（5个点）
print("\n" + "="*60)
print("2. 随机梯度下降 (Stochastic Gradient Descent)")
print("-"*60)
print("📌 特点：每次用1个数据点计算梯度")
print("✅ 优点：更新快，能跳出局部最优")
print("❌ 缺点：不稳定，有噪声")
print()

def stochastic_gradient_descent(w, b, x, y, lr=0.01, epochs=1):
    """随机梯度下降 - 每次用1个数据点更新"""
    w_history, b_history, loss_history = [w], [b], [compute_loss(w, b, x, y)]
    n = len(x)

    for epoch in range(epochs):
        # 随机打乱数据
        indices = np.random.permutation(n)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        for i, (xi, yi) in enumerate(zip(x_shuffled, y_shuffled), 1):
            # 用这一个点计算梯度
            error = (w * xi + b) - yi
            w_grad = 2 * error * xi
            b_grad = 2 * error

            # 更新参数
            w = w - lr * w_grad
            b = b - lr * b_grad

            # 记录
            w_history.append(w)
            b_history.append(b)
            loss_history.append(compute_loss(w, b, x, y))

    return w, b, w_history, b_history, loss_history

# 运行随机梯度下降
w_sgd, b_sgd, w_hist_sgd, b_hist_sgd, loss_hist_sgd = stochastic_gradient_descent(
    w_init, b_init, x_train, y_train, lr, epochs=1
)

# 6. 小批量梯度下降 (Mini-batch GD) - 修复：运行完整1个epoch
print("\n" + "="*60)
print("3. 小批量梯度下降 (Mini-batch Gradient Descent)")
print("-"*60)
print("📌 特点：一次用一小批数据（batch_size=2）")
print("✅ 优点：平衡速度和稳定性")
print("💡 最常用，深度学习标配")
print()

def minibatch_gradient_descent(w, b, x, y, lr=0.01, batch_size=2, epochs=1):
    """小批量梯度下降 - 每次用batch_size个数据更新"""
    w_history, b_history, loss_history = [w], [b], [compute_loss(w, b, x, y)]
    n = len(x)

    for epoch in range(epochs):
        # 随机打乱数据
        indices = np.random.permutation(n)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        # 分批处理
        for batch_idx in range(0, n, batch_size):
            # 获取一个小批量
            batch_end = min(batch_idx + batch_size, n)
            x_batch = x_shuffled[batch_idx:batch_end]
            y_batch = y_shuffled[batch_idx:batch_end]

            # 用这个batch计算平均梯度
            errors = (w * x_batch + b) - y_batch
            w_grad = 2 * np.mean(errors * x_batch)
            b_grad = 2 * np.mean(errors)

            # 更新参数
            w = w - lr * w_grad
            b = b - lr * b_grad

            # 记录
            w_history.append(w)
            b_history.append(b)
            loss_history.append(compute_loss(w, b, x, y))

    return w, b, w_history, b_history, loss_history

# 运行小批量梯度下降
batch_size = 2
w_mini, b_mini, w_hist_mini, b_hist_mini, loss_hist_mini = minibatch_gradient_descent(
    w_init, b_init, x_train, y_train, lr, batch_size=batch_size, epochs=1
)

# 7. 可视化对比 - 修复：确保所有曲线都显示
print("\n" + "="*60)
print("📈 三种方法的可视化对比")
print("-"*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 设置不同线型和标记，确保区分度
batch_style = {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'markersize': 6}
sgd_style = {'color': 'red', 'marker': 's', 'linestyle': '--', 'linewidth': 2, 'markersize': 6}
mini_style = {'color': 'green', 'marker': '^', 'linestyle': '-.', 'linewidth': 2, 'markersize': 8}

# 1. 损失变化对比
ax1 = axes[0, 0]

# 计算每个方法的更新次数
batch_updates = len(loss_hist_batch) - 1
sgd_updates = len(loss_hist_sgd) - 1
mini_updates = len(loss_hist_mini) - 1

# 批量GD的损失
ax1.plot(range(len(loss_hist_batch)), loss_hist_batch,
         label=f'Batch GD (更新={batch_updates})', **batch_style)

# 随机GD的损失
ax1.plot(range(len(loss_hist_sgd)), loss_hist_sgd,
         label=f'SGD (更新={sgd_updates})', **sgd_style)

# 小批量GD的损失
ax1.plot(range(len(loss_hist_mini)), loss_hist_mini,
         label=f'Mini-batch GD (更新={mini_updates})', **mini_style)

ax1.set_xlabel('更新次数')
ax1.set_ylabel('损失')
ax1.set_title('三种方法的损失变化')
ax1.legend()
ax1.grid(True, alpha=0.3)
# 使用线性坐标而不是对数坐标，让曲线更明显
# ax1.set_yscale('log')

# 2. 参数w的变化
ax2 = axes[0, 1]

# 批量GD
ax2.plot(range(len(w_hist_batch)), w_hist_batch,
         label=f'Batch GD (epoch=3)', **batch_style)
# 随机GD
ax2.plot(range(len(w_hist_sgd)), w_hist_sgd,
         label=f'SGD (5个点)', **sgd_style)
# 小批量GD
ax2.plot(range(len(w_hist_mini)), w_hist_mini,
         label=f'Mini-batch GD (batch_size={batch_size})', **mini_style)

# 添加目标线
ax2.axhline(y=2.0, color='k', linestyle=':', alpha=0.7, label='目标w=2.0')
ax2.set_xlabel('更新次数')
ax2.set_ylabel('w值')
ax2.set_title('参数w的变化')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 参数b的变化
ax3 = axes[0, 2]

# 批量GD
ax3.plot(range(len(b_hist_batch)), b_hist_batch,
         label='Batch GD', **batch_style)
# 随机GD
ax3.plot(range(len(b_hist_sgd)), b_hist_sgd,
         label='SGD', **sgd_style)
# 小批量GD
ax3.plot(range(len(b_hist_mini)), b_hist_mini,
         label='Mini-batch GD', **mini_style)

# 添加目标线
ax3.axhline(y=1.0, color='k', linestyle=':', alpha=0.7, label='目标b=1.0')
ax3.set_xlabel('更新次数')
ax3.set_ylabel('b值')
ax3.set_title('参数b的变化')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 参数空间中的路径
ax4 = axes[1, 0]

# 批量GD路径
ax4.plot(w_hist_batch, b_hist_batch, label='Batch GD',
         **batch_style)
# 随机GD路径
ax4.plot(w_hist_sgd, b_hist_sgd, label='SGD',
         **sgd_style)
# 小批量GD路径
ax4.plot(w_hist_mini, b_hist_mini, label='Mini-batch GD',
         **mini_style)

# 标记起点
ax4.plot(w_hist_batch[0], b_hist_batch[0], 'ko', markersize=10, label='起点')
# 标记目标
ax4.plot(2.0, 1.0, 'y*', markersize=15, label='目标点')

ax4.set_xlabel('w值')
ax4.set_ylabel('b值')
ax4.set_title('参数空间中的更新路径')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. 对比表格
ax5 = axes[1, 1]
ax5.axis('off')

# 计算最终损失
final_loss_batch = compute_loss(w_batch, b_batch, x_train, y_train)
final_loss_sgd = compute_loss(w_sgd, b_sgd, x_train, y_train)
final_loss_mini = compute_loss(w_mini, b_mini, x_train, y_train)

table_data = [
    ['方法', '更新次数', '最终w', '最终b', '最终损失'],
    ['Batch GD', f'{batch_updates}', f'{w_batch:.4f}', f'{b_batch:.4f}', f'{final_loss_batch:.6f}'],
    ['SGD', f'{sgd_updates}', f'{w_sgd:.4f}', f'{b_sgd:.4f}', f'{final_loss_sgd:.6f}'],
    ['Mini-batch', f'{mini_updates}', f'{w_mini:.4f}', f'{b_mini:.4f}', f'{final_loss_mini:.6f}']
]

table = ax5.table(cellText=table_data, loc='center', cellLoc='center',
                  colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)
ax5.set_title(f'三种方法对比 (batch_size={batch_size})')

# 6. 比喻说明
ax6 = axes[1, 2]
ax6.axis('off')

text = """
🎯 三种梯度下降方法对比：

1. 批量梯度下降 (Batch GD)
   👨‍🔬 科学家方法
   - 每次用全部数据
   - 更新次数 = epoch数
   - 稳定但慢

2. 随机梯度下降 (SGD)
   🏃 冒险家方法
   - 每次用1个数据
   - 更新次数 = 数据点数
   - 快速但波动大

3. 小批量梯度下降 (Mini-batch)
   👥 团队方法
   - 每次用一批数据
   - 更新次数 = 批次数
   - 平衡速度与稳定
   - 深度学习最常用

📊 本实验设置：
   - 数据点：5个
   - batch_size：2
   - Batch GD：3次更新
   - SGD：5次更新
   - Mini-batch：3次更新
"""

ax6.text(0.1, 0.5, text, fontsize=10,
         verticalalignment='center',
         transform=ax6.transAxes)
ax6.set_title('方法总结')

plt.tight_layout()
plt.savefig('gradient_descent_variants_fixed.png', dpi=100, bbox_inches='tight')
print("✅ 图表已保存为 gradient_descent_variants_fixed.png")
print()

# 8. 详细对比
print("="*60)
print("📊 详细对比")
print("-"*60)
print(f"数据点数量: {len(x_train)}")
print(f"Batch GD: 每个epoch更新1次，{len(loss_hist_batch)-1}次更新")
print(f"SGD: 每个点更新1次，{len(loss_hist_sgd)-1}次更新")
print(f"Mini-batch (batch_size={batch_size}): 每批更新1次，{len(loss_hist_mini)-1}次更新")
print()

print("🔍 为什么有的曲线不显示？")
print("-"*30)
print("1. 更新次数不同：三种方法的更新频率不同")
print("2. 横坐标不对齐：Batch GD每个epoch一次，SGD每个点一次")
print("3. 损失值差异：不同方法的损失变化幅度不同")
print("4. 绘图设置：可能线型、颜色、标记大小设置不当")
print()

print("✨ 修复措施：")
print("-"*30)
print("1. 统一显示所有方法的完整训练过程")
print("2. 使用不同线型（实线、虚线、点划线）")
print("3. 调整标记大小和颜色对比度")
print("4. 使用线性坐标而非对数坐标")
print()

# 9. 测试不同batch_size
print("="*60)
print("🎮 测试不同batch_size的效果")
print("-"*60)

def test_batch_size(batch_size, epochs=3, lr=0.01):
    """测试不同batch_size的效果"""
    w, b = w_init, b_init
    n = len(x_train)
    loss_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        for i in range(0, n, batch_size):
            batch_end = min(i + batch_size, n)
            x_batch = x_shuffled[i:batch_end]
            y_batch = y_shuffled[i:batch_end]

            # 计算梯度
            errors = (w * x_batch + b) - y_batch
            w_grad = 2 * np.mean(errors * x_batch)
            b_grad = 2 * np.mean(errors)

            # 更新参数
            w -= lr * w_grad
            b -= lr * b_grad

            # 记录损失
            loss_history.append(compute_loss(w, b, x_train, y_train))

    updates = len(loss_history)
    final_loss = loss_history[-1] if loss_history else compute_loss(w, b, x_train, y_train)

    return w, b, updates, final_loss, loss_history

# 测试不同batch_size
batch_sizes = [1, 2, 3, 5]
results = []

for bs in batch_sizes:
    w_final, b_final, updates, final_loss, _ = test_batch_size(bs, epochs=3, lr=0.01)
    results.append((bs, updates, w_final, b_final, final_loss))

print("不同batch_size的对比：")
print("batch_size | 更新次数 | 最终w | 最终b | 最终损失")
print("-" * 50)
for bs, updates, w, b, loss in results:
    if bs == 5:
        method = "Batch GD"
    elif bs == 1:
        method = "SGD"
    else:
        method = f"Mini-batch({bs})"

    print(f"{method:12} | {updates:8d} | {w:6.3f} | {b:6.3f} | {loss:8.6f}")

print("\n" + "="*60)
print("🎯 总结")
print("-"*60)
print("1. Batch GD (batch_size=全部数据)")
print("   - 更新最稳定，但最慢")
print("   - 适合小数据集")
print()
print("2. SGD (batch_size=1)")
print("   - 更新最快，但最不稳定")
print("   - 适合大数据集")
print()
print("3. Mini-batch GD (batch_size=2-256)")
print("   - 平衡速度和稳定性")
print("   - 深度学习最常用")
print("   - batch_size通常为32, 64, 128")
print()
print("💡 关键：在PyTorch中，通过DataLoader的batch_size参数控制！")

plt.show()