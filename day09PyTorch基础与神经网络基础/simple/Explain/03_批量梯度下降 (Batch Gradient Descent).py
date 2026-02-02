import numpy as np
import matplotlib.pyplot as plt

# 1. 创建数据
np.random.seed(42)
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = 2 * x + 1 + np.random.randn(5) * 0.3

# 2. 初始化参数
w, b = 0.0, 0.0
lr, epochs = 0.05, 10

# 3. 损失函数
def loss_fn(w, b, x, y):
    return np.mean((w * x + b - y) ** 2)

# 4. 批量梯度下降训练
w_hist, b_hist, loss_hist = [w], [b], [loss_fn(w, b, x, y)]

for epoch in range(epochs):
    # 计算梯度
    errors = w * x + b - y
    dw = 2 * np.mean(errors * x)
    db = 2 * np.mean(errors)

    # 更新参数
    w -= lr * dw
    b -= lr * db

    # 记录历史
    w_hist.append(w)
    b_hist.append(b)
    loss_hist.append(loss_fn(w, b, x, y))

# 5. 打印结果
print("初始: w=0.0, b=0.0, 损失=", loss_hist[0])
print("最终: w=%.4f, b=%.4f, 损失=%.4f" % (w, b, loss_hist[-1]))
print("目标: w≈2.0, b≈1.0")

# 6. 可视化
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 损失变化
axes[0,0].plot(loss_hist, 'b-o')
axes[0,0].set_title('损失变化')
axes[0,0].set_xlabel('epoch')

# w变化
axes[0,1].plot(w_hist, 'r-s', label='w值')
axes[0,1].axhline(2.0, color='g', linestyle='--', label='目标w=2')
axes[0,1].set_title('参数w变化')
axes[0,1].legend()

# b变化
axes[1,0].plot(b_hist, 'm-^', label='b值')
axes[1,0].axhline(1.0, color='g', linestyle='--', label='目标b=1')
axes[1,0].set_title('参数b变化')
axes[1,0].legend()

# 拟合效果
axes[1,1].scatter(x, y, c='red', label='数据点')
axes[1,1].plot(x, 2*x+1, 'g--', label='真实: y=2x+1')
x_line = np.array([0, 6])
axes[1,1].plot(x_line, w*x_line+b, 'b-', label=f'拟合: y={w:.2f}x+{b:.2f}')
axes[1,1].set_xlim([0, 6])
axes[1,1].legend()
axes[1,1].set_title('拟合效果')

plt.tight_layout()
plt.show()