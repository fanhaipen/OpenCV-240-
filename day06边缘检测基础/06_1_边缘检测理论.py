"""
第6天 - 文件1：边缘检测理论
学习目标：理解边缘检测的基本概念、数学原理和分类
重点：边缘类型、梯度计算、边缘检测步骤
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

print("📘 第6天 - 文件1：边缘检测理论")
print("=" * 50)

# ==================== 1. 什么是边缘？ ====================
print("\n🎯 1. 什么是边缘？")
print("=" * 30)

print("""
边缘 (Edge) 定义：
  图像中像素灰度值发生突变的位置，通常对应：
  - 物体的边界
  - 表面方向的变化
  - 深度的不连续
  - 光照的变化
  - 纹理的变化

为什么检测边缘？
  - 边缘是图像的重要特征
  - 边缘包含了图像的形状信息
  - 边缘可以大大减少数据量
  - 边缘是许多计算机视觉任务的基础
""")

# ==================== 2. 边缘类型 ====================
print("\n📊 2. 边缘类型")
print("=" * 30)


def create_edge_types_demo():
    """创建不同类型的边缘演示"""

    # 创建测试信号
    x = np.linspace(0, 100, 500)

    # 1. 阶梯边缘 (Step Edge)
    step_edge = np.zeros_like(x)
    step_edge[x > 50] = 100

    # 2. 斜坡边缘 (Ramp Edge)
    ramp_edge = np.zeros_like(x)
    ramp_start, ramp_end = 30, 70
    ramp_mask = (x >= ramp_start) & (x <= ramp_end)
    ramp_edge[ramp_mask] = 100 * (x[ramp_mask] - ramp_start) / (ramp_end - ramp_start)
    ramp_edge[x > ramp_end] = 100

    # 3. 屋顶边缘 (Roof Edge)
    roof_edge = np.zeros_like(x)
    roof_center = 50
    roof_width = 20
    roof_mask = (x >= roof_center - roof_width / 2) & (x <= roof_center + roof_width / 2)
    roof_edge[roof_mask] = 100 - 100 * np.abs(x[roof_mask] - roof_center) / (roof_width / 2)

    # 4. 线边缘 (Line Edge)
    line_edge = np.zeros_like(x)
    line_center = 50
    line_width = 4
    line_mask = (x >= line_center - line_width / 2) & (x <= line_center + line_width / 2)
    line_edge[line_mask] = 100

    return x, step_edge, ramp_edge, roof_edge, line_edge


# 创建边缘类型演示
x, step_edge, ramp_edge, roof_edge, line_edge = create_edge_types_demo()

# 显示边缘类型
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. 阶梯边缘
axes[0, 0].plot(x, step_edge, 'b-', linewidth=2)
axes[0, 0].set_title("阶梯边缘 (Step Edge)")
axes[0, 0].set_xlabel("位置")
axes[0, 0].set_ylabel("灰度值")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].text(0.5, 0.9, "像素值突然变化\n对应物体边界",
                transform=axes[0, 0].transAxes, ha='center', fontsize=10)

# 2. 斜坡边缘
axes[0, 1].plot(x, ramp_edge, 'g-', linewidth=2)
axes[0, 1].set_title("斜坡边缘 (Ramp Edge)")
axes[0, 1].set_xlabel("位置")
axes[0, 1].set_ylabel("灰度值")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].text(0.5, 0.9, "像素值逐渐变化\n对应模糊边界",
                transform=axes[0, 1].transAxes, ha='center', fontsize=10)

# 3. 屋顶边缘
axes[1, 0].plot(x, roof_edge, 'r-', linewidth=2)
axes[1, 0].set_title("屋顶边缘 (Roof Edge)")
axes[1, 0].set_xlabel("位置")
axes[1, 0].set_ylabel("灰度值")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].text(0.5, 0.9, "像素值先增后减\n对应细线或山脊",
                transform=axes[1, 0].transAxes, ha='center', fontsize=10)

# 4. 线边缘
axes[1, 1].plot(x, line_edge, 'm-', linewidth=2)
axes[1, 1].set_title("线边缘 (Line Edge)")
axes[1, 1].set_xlabel("位置")
axes[1, 1].set_ylabel("灰度值")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].text(0.5, 0.9, "窄脉冲变化\n对应细线或纹理",
                transform=axes[1, 1].transAxes, ha='center', fontsize=10)

plt.suptitle("四种基本边缘类型", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 3. 边缘的数学表示 ====================
print("\n🧮 3. 边缘的数学表示")
print("=" * 30)


def demonstrate_edge_mathematics():
    """演示边缘的数学表示"""

    print("""
数学表示：
  图像可以看作二维函数 I(x,y)
  边缘出现在灰度变化剧烈的位置

梯度 (Gradient)：
  ∇I = [∂I/∂x, ∂I/∂y]^T
    grad_x[i, j] = I[i, j+1] - I[i, j-1]
    grad_y[i, j] = I[i+1, j] - I[i-1, j]
梯度幅值 (Gradient Magnitude)：
  |∇I| = √((∂I/∂x)² + (∂I/∂y)²)

梯度方向 (Gradient Direction)：
  θ = atan2(∂I/∂y, ∂I/∂x)

边缘检测原理：
  寻找梯度幅值大的位置
  """)

    # 创建一个简单的边缘示例
    edge_example = np.array([
        [10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10],
        [10, 10, 100, 200, 200],
        [10, 10, 200, 200, 200],
        [10, 10, 200, 200, 200]
    ], dtype=np.float32)

    print("边缘示例（5×5像素块）：")
    print(edge_example)
    print()

    # 计算梯度
    # x方向梯度（中心差分）
    grad_x = np.zeros_like(edge_example, dtype=np.float32)
    grad_x[:, 1:-1] = edge_example[:, 2:] - edge_example[:, :-2]

    # y方向梯度
    grad_y = np.zeros_like(edge_example, dtype=np.float32)
    grad_y[1:-1, :] = edge_example[2:, :] - edge_example[:-2, :]

    # 梯度幅值
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 梯度方向（弧度）
    grad_dir = np.arctan2(grad_y, grad_x)

    print("x方向梯度 (∂I/∂x)：")
    print(grad_x.astype(int))
    print()

    print("y方向梯度 (∂I/∂y)：")
    print(grad_y.astype(int))
    print()

    print("梯度幅值 |∇I|：")
    print(grad_mag.astype(int))
    print()

    print("梯度方向 θ（弧度）：")
    print(np.round(grad_dir, 2))
    print()

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 原始图像
    im1 = axes[0, 0].imshow(edge_example, cmap='gray')
    axes[0, 0].set_title("原始图像 I(x,y)")
    axes[0, 0].set_xticks(range(5))
    axes[0, 0].set_yticks(range(5))
    plt.colorbar(im1, ax=axes[0, 0])

    # 标注像素值
    for i in range(5):
        for j in range(5):
            axes[0, 0].text(j, i, f'{edge_example[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if edge_example[i, j] < 100 else 'black')

    # x方向梯度
    im2 = axes[0, 1].imshow(grad_x, cmap='coolwarm', vmin=-200, vmax=200)
    axes[0, 1].set_title("x方向梯度 ∂I/∂x")
    axes[0, 1].set_xticks(range(5))
    axes[0, 1].set_yticks(range(5))
    plt.colorbar(im2, ax=axes[0, 1])

    for i in range(5):
        for j in range(5):
            axes[0, 1].text(j, i, f'{grad_x[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if abs(grad_x[i, j]) < 100 else 'black')

    # y方向梯度
    im3 = axes[0, 2].imshow(grad_y, cmap='coolwarm', vmin=-200, vmax=200)
    axes[0, 2].set_title("y方向梯度 ∂I/∂y")
    axes[0, 2].set_xticks(range(5))
    axes[0, 2].set_yticks(range(5))
    plt.colorbar(im3, ax=axes[0, 2])

    for i in range(5):
        for j in range(5):
            axes[0, 2].text(j, i, f'{grad_y[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if abs(grad_y[i, j]) < 100 else 'black')

    # 梯度幅值
    im4 = axes[1, 0].imshow(grad_mag, cmap='hot')
    axes[1, 0].set_title("梯度幅值 |∇I|")
    axes[1, 0].set_xticks(range(5))
    axes[1, 0].set_yticks(range(5))
    plt.colorbar(im4, ax=axes[1, 0])

    for i in range(5):
        for j in range(5):
            axes[1, 0].text(j, i, f'{grad_mag[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if grad_mag[i, j] < 100 else 'black')

    # 梯度方向
    im5 = axes[1, 1].imshow(grad_dir, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title("梯度方向 θ")
    axes[1, 1].set_xticks(range(5))
    axes[1, 1].set_yticks(range(5))
    plt.colorbar(im5, ax=axes[1, 1])

    for i in range(5):
        for j in range(5):
            axes[1, 1].text(j, i, f'{grad_dir[i, j]:.2f}',
                            ha='center', va='center', fontsize=8,
                            color='white' if abs(grad_dir[i, j]) > 1 else 'black')

    # 向量场表示
    axes[1, 2].quiver(grad_x, grad_y, color='red', scale=100)
    axes[1, 2].set_title("梯度向量场 ∇I")
    axes[1, 2].set_xlim(-0.5, 4.5)
    axes[1, 2].set_ylim(-0.5, 4.5)
    axes[1, 2].invert_yaxis()  # 图像坐标系y轴向下
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle("边缘的数学表示：梯度计算", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return edge_example, grad_x, grad_y, grad_mag, grad_dir


# 演示边缘的数学表示
edge_example, grad_x, grad_y, grad_mag, grad_dir = demonstrate_edge_mathematics()

# ==================== 4. 边缘检测基本步骤 ====================
print("\n🔧 4. 边缘检测基本步骤")
print("=" * 30)


def demonstrate_edge_detection_steps():
    """演示边缘检测的基本步骤"""

    print("""
边缘检测的一般步骤：

1. 噪声抑制
   - 原因：梯度对噪声敏感
   - 方法：高斯滤波、中值滤波等
   - 目标：平滑图像，减少噪声影响

2. 梯度计算
   - 计算图像在x和y方向的梯度
   - 常用算子：Sobel、Prewitt、Roberts
   - 得到梯度幅值和方向

3. 非极大值抑制
   - 原因：梯度幅值大的区域可能很宽
   - 方法：在梯度方向上只保留局部最大值
   - 目标：细化边缘，得到单像素宽边缘

4. 双阈值检测
   - 设置高阈值和低阈值
   - 高阈值以上的点：强边缘
   - 低阈值以下的点：非边缘
   - 中间的点：弱边缘（可能连接）

5. 边缘连接
   - 连接弱边缘到强边缘
   - 方法：滞后阈值、边缘跟踪
   - 目标：得到连续的边缘
  """)

    # 创建示例图片展示各步骤
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 步骤1: 噪声抑制
    # 创建有噪声的边缘
    clean_edge = np.zeros((100, 100), dtype=np.uint8)
    clean_edge[:, 50:] = 200
    noisy_edge = clean_edge.astype(np.float32) + np.random.normal(0, 30, clean_edge.shape)
    noisy_edge = np.clip(noisy_edge, 0, 255).astype(np.uint8)

    # 高斯滤波去噪
    smoothed = cv2.GaussianBlur(noisy_edge, (5, 5), 1.4)

    axes[0, 0].imshow(noisy_edge, cmap='gray')
    axes[0, 0].set_title("步骤1: 有噪声的图像")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(smoothed, cmap='gray')
    axes[0, 1].set_title("高斯滤波后")
    axes[0, 1].axis('off')

    # 步骤2: 梯度计算
    grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    axes[0, 2].imshow(grad_mag, cmap='hot')
    axes[0, 2].set_title("步骤2: 梯度幅值")
    axes[0, 2].axis('off')

    # 步骤3: 非极大值抑制（简化演示）
    # 创建简化的梯度幅值图
    simple_grad = np.array([
        [1, 2, 3, 2, 1],
        [2, 3, 5, 3, 2],
        [3, 5, 8, 5, 3],
        [2, 3, 5, 3, 2],
        [1, 2, 3, 2, 1]
    ], dtype=np.float32)

    # 简化的非极大值抑制
    nms_result = np.zeros_like(simple_grad)
    center = 2
    nms_result[center, center] = simple_grad[center, center]  # 只保留中心最大值

    axes[1, 0].imshow(simple_grad, cmap='hot')
    axes[1, 0].set_title("梯度幅值（粗边缘）")
    axes[1, 0].set_xticks(range(5))
    axes[1, 0].set_yticks(range(5))

    for i in range(5):
        for j in range(5):
            axes[1, 0].text(j, i, f'{simple_grad[i, j]:.0f}',
                            ha='center', va='center', fontsize=8)

    axes[1, 1].imshow(nms_result, cmap='hot')
    axes[1, 1].set_title("步骤3: 非极大值抑制后")
    axes[1, 1].set_xticks(range(5))
    axes[1, 1].set_yticks(range(5))

    for i in range(5):
        for j in range(5):
            axes[1, 1].text(j, i, f'{nms_result[i, j]:.0f}',
                            ha='center', va='center', fontsize=8)

    # 步骤4-5: 双阈值和边缘连接
    # 创建简化的阈值演示
    threshold_demo = np.array([
        [0, 0, 0, 0, 0],
        [0, 30, 50, 30, 0],
        [0, 50, 100, 50, 0],
        [0, 30, 50, 30, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)

    # 双阈值处理
    high_threshold = 80
    low_threshold = 30

    strong_edges = (threshold_demo >= high_threshold).astype(np.float32)
    weak_edges = ((threshold_demo >= low_threshold) & (threshold_demo < high_threshold)).astype(np.float32)

    # 边缘连接（简化）
    connected_edges = strong_edges.copy()
    # 如果弱边缘与强边缘相邻，则保留
    for i in range(1, 4):
        for j in range(1, 4):
            if weak_edges[i, j] > 0:
                # 检查8邻域是否有强边缘
                if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2] > 0):
                    connected_edges[i, j] = 0.5  # 标记为连接的弱边缘

    axes[1, 2].imshow(connected_edges, cmap='hot')
    axes[1, 2].set_title("步骤4-5: 双阈值检测+边缘连接")
    axes[1, 2].set_xticks(range(5))
    axes[1, 2].set_yticks(range(5))

    threshold_info = f"高阈值: {high_threshold}\n低阈值: {low_threshold}\n红色: 强边缘\n黄色: 弱边缘"
    axes[1, 2].text(0.5, -0.2, threshold_info, transform=axes[1, 2].transAxes,
                    ha='center', fontsize=9)

    plt.suptitle("边缘检测的基本步骤", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return noisy_edge, smoothed, grad_mag


# 演示边缘检测步骤
noisy_edge, smoothed, grad_mag = demonstrate_edge_detection_steps()

# ==================== 5. 边缘检测算子分类 ====================
print("\n📈 5. 边缘检测算子分类")
print("=" * 30)


def demonstrate_edge_detector_classification():
    """演示边缘检测算子的分类"""

    print("""
边缘检测算子分类：

1. 一阶微分算子
   - 原理：检测灰度的一阶导数
   - 特点：对阶梯边缘敏感
   - 优点：计算简单
   - 缺点：对噪声敏感
   - 例子：Sobel, Prewitt, Roberts

2. 二阶微分算子
   - 原理：检测灰度的二阶导数
   - 特点：对细线、屋顶边缘敏感
   - 优点：能检测更细的边缘
   - 缺点：对噪声更敏感
   - 例子：Laplacian, LoG

3. 高级边缘检测算子
   - 原理：多步骤算法
   - 特点：结合了去噪、梯度计算、细化等
   - 优点：效果好，鲁棒性强
   - 缺点：计算复杂
   - 例子：Canny, Marr-Hildreth
  """)

    # 创建可视化对比
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    # 创建测试图片
    test_img = np.zeros((100, 100), dtype=np.uint8)
    # 添加阶梯边缘
    test_img[:, 50:] = 200
    # 添加细线
    cv2.line(test_img, (20, 20), (80, 20), 150, 2)
    # 添加高斯噪声
    test_img_noisy = test_img.astype(np.float32) + np.random.normal(0, 20, test_img.shape)
    test_img_noisy = np.clip(test_img_noisy, 0, 255).astype(np.uint8)

    # 1. 一阶微分算子示例
    # Sobel
    sobel_x = cv2.Sobel(test_img_noisy, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(test_img_noisy, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Roberts
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    roberts_gx = cv2.filter2D(test_img_noisy, -1, roberts_x)
    roberts_gy = cv2.filter2D(test_img_noisy, -1, roberts_y)
    roberts_mag = np.sqrt(roberts_gx ** 2 + roberts_gy ** 2)

    # 2. 二阶微分算子示例
    # Laplacian
    laplacian = cv2.Laplacian(test_img_noisy, cv2.CV_64F, ksize=3)

    # LoG (Laplacian of Gaussian)
    blurred = cv2.GaussianBlur(test_img_noisy, (5, 5), 1.4)
    log_result = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    # 3. 高级算子示例
    # Canny
    canny_edges = cv2.Canny(test_img_noisy, 50, 150)

    # 显示结果
    # 第一行：原始图片
    axes[0, 0].imshow(test_img, cmap='gray')
    axes[0, 0].set_title("原始图片")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(test_img_noisy, cmap='gray')
    axes[0, 1].set_title("加噪图片")
    axes[0, 1].axis('off')

    axes[0, 2].axis('off')
    axes[0, 2].text(0.5, 0.5, "边缘检测算子分类",
                    ha='center', va='center', fontsize=12, fontweight='bold')

    # 第二行：一阶微分算子
    axes[1, 0].imshow(np.abs(sobel_mag), cmap='hot')
    axes[1, 0].set_title("Sobel算子\n(一阶微分)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.abs(roberts_mag), cmap='hot')
    axes[1, 1].set_title("Roberts算子\n(一阶微分)")
    axes[1, 1].axis('off')

    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.5,
                    "一阶微分算子特点:\n"
                    "• 检测阶梯边缘\n"
                    "• 对噪声敏感\n"
                    "• 计算简单快速\n"
                    "• 需要设定阈值",
                    fontsize=10, verticalalignment='center')

    # 第三行：二阶微分算子
    axes[2, 0].imshow(np.abs(laplacian), cmap='hot')
    axes[2, 0].set_title("Laplacian算子\n(二阶微分)")
    axes[2, 0].axis('off')

    axes[2, 1].imshow(np.abs(log_result), cmap='hot')
    axes[2, 1].set_title("LoG算子\n(二阶微分)")
    axes[2, 1].axis('off')

    axes[2, 2].axis('off')
    axes[2, 2].text(0.1, 0.5,
                    "二阶微分算子特点:\n"
                    "• 检测细线、屋顶边缘\n"
                    "• 对噪声更敏感\n"
                    "• 产生双边缘\n"
                    "• 零交叉检测",
                    fontsize=10, verticalalignment='center')

    plt.suptitle("边缘检测算子分类与比较", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 显示Canny结果
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(test_img_noisy, cmap='gray')
    ax[0].set_title("原始加噪图片")
    ax[0].axis('off')

    ax[1].imshow(canny_edges, cmap='gray',vmin=0,vmax=255)
    ax[1].set_title("Canny边缘检测")
    ax[1].axis('off')

    plt.suptitle("高级边缘检测算子：Canny", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return test_img_noisy, sobel_mag, laplacian, canny_edges


# 演示算子分类
test_img_noisy, sobel_mag, laplacian, canny_edges = demonstrate_edge_detector_classification()

# ==================== 6. 总结 ====================
print("\n" + "=" * 50)
print("✅ 边缘检测理论总结")
print("=" * 50)

summary = """
📊 边缘检测理论核心知识：

1. 边缘定义
   - 图像中灰度值突变的位置
   - 对应物体边界、纹理变化、深度不连续等
   - 包含图像的重要形状信息

2. 边缘类型
   - 阶梯边缘: 灰度值突然变化
   - 斜坡边缘: 灰度值逐渐变化
   - 屋顶边缘: 灰度值先增后减
   - 线边缘: 窄脉冲变化

3. 数学原理
   - 梯度: ∇I = [∂I/∂x, ∂I/∂y]^T
   - 梯度幅值: |∇I| = √((∂I/∂x)² + (∂I/∂y)²)
   - 梯度方向: θ = atan2(∂I/∂y, ∂I/∂x)
   - 边缘检测: 寻找梯度幅值大的位置

4. 边缘检测步骤
   - 1. 噪声抑制: 使用滤波去除噪声
   - 2. 梯度计算: 计算图像梯度
   - 3. 非极大值抑制: 细化边缘
   - 4. 双阈值检测: 区分强/弱边缘
   - 5. 边缘连接: 得到连续边缘

5. 算子分类
   - 一阶微分算子: Sobel, Prewitt, Roberts
     * 优点: 计算简单
     * 缺点: 对噪声敏感

   - 二阶微分算子: Laplacian, LoG
     * 优点: 检测细边缘
     * 缺点: 对噪声更敏感

   - 高级算子: Canny, Marr-Hildreth
     * 优点: 效果好，鲁棒性强
     * 缺点: 计算复杂

6. 重要概念
   - 信噪比: 边缘信号与噪声的比值
   - 定位精度: 检测到的边缘位置准确度
   - 单边缘响应: 每个真实边缘只检测一次
   - 计算复杂度: 算法的时间和空间需求

7. 应用考虑
   - 噪声水平: 选择抗噪声能力
   - 实时性: 考虑计算速度
   - 精度要求: 选择定位精度
   - 边缘类型: 针对不同边缘选择算子

🎯 关键公式记忆：
   梯度向量: ∇I = [∂I/∂x, ∂I/∂y]^T
   梯度幅值: |∇I| = √((∂I/∂x)² + (∂I/∂y)²)
   梯度方向: θ = atan2(∂I/∂y, ∂I/∂x)
   边缘条件: |∇I| > 阈值
"""

print(summary)
print("\n📁 下一个文件: 06_02_Sobel算子实现.py")
print("  我们将学习最常用的一阶微分算子：Sobel算子！")