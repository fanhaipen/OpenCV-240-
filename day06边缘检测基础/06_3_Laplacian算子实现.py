"""
第6天 - 文件3：Laplacian算子实现
学习目标：掌握Laplacian算子的原理、实现和应用
重点：二阶导数、零交叉检测、LoG算子、边缘细化
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

print("🌀 第6天 - 文件3：Laplacian算子实现")
print("=" * 50)

# ==================== 1. Laplacian算子理论 ====================
print("\n🎯 1. Laplacian算子理论")
print("=" * 30)

print("""
Laplacian算子：

基本概念：
  - 二阶微分算子，用于检测图像的二阶导数
  - 不依赖于边缘方向（各向同性）
  - 对噪声非常敏感
  - 产生双边缘响应

数学原理：
  Laplacian算子计算图像的拉普拉斯算子（二阶导数之和）

数学定义：
  ∇²I = ∂²I/∂x² + ∂²I/∂y²

离散近似：
  常用4邻域卷积核：
    [ 0, -1,  0]
    [-1,  4, -1]
    [ 0, -1,  0]

  8邻域卷积核：
    [-1, -1, -1]
    [-1,  8, -1]
    [-1, -1, -1]

特点：
  - 对噪声极其敏感
  - 产生零交叉点（zero-crossing）
  - 能检测细线和孤立点
  - 边缘定位精度高
  - 各向同性（旋转不变性）

优点：
  - 能检测细边缘
  - 定位精度高
  - 各向同性
  - 能检测灰度变化率的变化

缺点：
  - 对噪声非常敏感
  - 产生双边缘响应
  - 需要零交叉检测
  - 计算复杂度较高

应用场景：
  - 精细边缘检测
  - 斑点检测
  - 图像增强
  - 零交叉检测
  - 与其他算子结合使用
""")

# ==================== 2. Laplacian卷积核详解 ====================
print("\n🔧 2. Laplacian卷积核详解")
print("=" * 30)


def demonstrate_laplacian_kernels():
    """详细讲解Laplacian卷积核"""

    print("Laplacian卷积核的数学原理:")
    print("=" * 40)

    # 定义不同的Laplacian卷积核
    laplacian_4neighbor = np.array([[0, -1, 0],
                                    [-1, 4, -1],
                                    [0, -1, 0]], dtype=np.float32)

    laplacian_8neighbor = np.array([[-1, -1, -1],
                                    [-1, 8, -1],
                                    [-1, -1, -1]], dtype=np.float32)

    laplacian_diagonal = np.array([[-1, 0, -1],
                                   [0, 4, 0],
                                   [-1, 0, -1]], dtype=np.float32)

    print("4邻域Laplacian核:")
    print(laplacian_4neighbor)
    print()

    print("8邻域Laplacian核:")
    print(laplacian_8neighbor)
    print()

    print("对角线Laplacian核:")
    print(laplacian_diagonal)
    print()

    # 解释卷积核的设计原理
    print("卷积核设计原理:")
    print("1. 中心差分: 中心点权重为正，周围点权重为负")
    print("2. 二阶导数: 检测灰度变化的二阶导数")
    print("3. 各向同性: 对各个方向的变化响应相同")
    print("4. 零和性质: 卷积核元素之和为0")
    print()

    # 可视化卷积核
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 4邻域核可视化
    im1 = axes[0, 0].imshow(laplacian_4neighbor, cmap='coolwarm', vmin=-2, vmax=8)
    axes[0, 0].set_title("4邻域Laplacian核")
    axes[0, 0].set_xticks([0, 1, 2])
    axes[0, 0].set_yticks([0, 1, 2])
    plt.colorbar(im1, ax=axes[0, 0])

    for i in range(3):
        for j in range(3):
            axes[0, 0].text(j, i, f'{laplacian_4neighbor[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if abs(laplacian_4neighbor[i, j]) > 2 else 'black',
                            fontsize=12, fontweight='bold')

    # 8邻域核可视化
    im2 = axes[0, 1].imshow(laplacian_8neighbor, cmap='coolwarm', vmin=-2, vmax=8)
    axes[0, 1].set_title("8邻域Laplacian核")
    axes[0, 1].set_xticks([0, 1, 2])
    axes[0, 1].set_yticks([0, 1, 2])
    plt.colorbar(im2, ax=axes[0, 1])

    for i in range(3):
        for j in range(3):
            axes[0, 1].text(j, i, f'{laplacian_8neighbor[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if abs(laplacian_8neighbor[i, j]) > 2 else 'black',
                            fontsize=12, fontweight='bold')

    # 对角线核可视化
    im3 = axes[0, 2].imshow(laplacian_diagonal, cmap='coolwarm', vmin=-2, vmax=8)
    axes[0, 2].set_title("对角线Laplacian核")
    axes[0, 2].set_xticks([0, 1, 2])
    axes[0, 2].set_yticks([0, 1, 2])
    plt.colorbar(im3, ax=axes[0, 2])

    for i in range(3):
        for j in range(3):
            axes[0, 2].text(j, i, f'{laplacian_diagonal[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if abs(laplacian_diagonal[i, j]) > 2 else 'black',
                            fontsize=12, fontweight='bold')

    # 卷积计算演示
    # 创建一个简单的图像区域，模拟边缘
    test_region = np.array([
        [50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50],
        [50, 50, 150, 150, 150],
        [50, 50, 150, 150, 150],
        [50, 50, 150, 150, 150]
    ], dtype=np.float32)

    # 手动卷积计算
    def manual_convolution(image, kernel):
        """手动实现卷积计算"""
        height, width = image.shape
        k_size = kernel.shape[0]
        pad = k_size // 2

        # 边界填充
        padded = np.pad(image, pad, mode='constant')
        result = np.zeros_like(image, dtype=np.float32)

        # 应用卷积
        for i in range(height):
            for j in range(width):
                region = padded[i:i + k_size, j:j + k_size]
                result[i, j] = np.sum(region * kernel)

        return result

    # 计算不同卷积核的结果
    conv_4n = manual_convolution(test_region, laplacian_4neighbor)
    conv_8n = manual_convolution(test_region, laplacian_8neighbor)
    conv_diag = manual_convolution(test_region, laplacian_diagonal)

    # 显示原始图像区域
    im4 = axes[1, 0].imshow(test_region, cmap='gray')
    axes[1, 0].set_title("测试图像区域")
    axes[1, 0].set_xticks(range(5))
    axes[1, 0].set_yticks(range(5))
    plt.colorbar(im4, ax=axes[1, 0])

    for i in range(5):
        for j in range(5):
            axes[1, 0].text(j, i, f'{test_region[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if test_region[i, j] < 100 else 'black')

    # 显示4邻域卷积结果
    im5 = axes[1, 1].imshow(conv_4n, cmap='coolwarm')
    axes[1, 1].set_title("4邻域卷积结果")
    axes[1, 1].set_xticks(range(5))
    axes[1, 1].set_yticks(range(5))
    plt.colorbar(im5, ax=axes[1, 1])

    for i in range(5):
        for j in range(5):
            axes[1, 1].text(j, i, f'{conv_4n[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if abs(conv_4n[i, j]) > 100 else 'black')

    # 显示8邻域卷积结果
    im6 = axes[1, 2].imshow(conv_8n, cmap='coolwarm')
    axes[1, 2].set_title("8邻域卷积结果")
    axes[1, 2].set_xticks(range(5))
    axes[1, 2].set_yticks(range(5))
    plt.colorbar(im6, ax=axes[1, 2])

    for i in range(5):
        for j in range(5):
            axes[1, 2].text(j, i, f'{conv_8n[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if abs(conv_8n[i, j]) > 100 else 'black')

    plt.suptitle("Laplacian卷积核详解与计算演示", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 详细解释卷积计算过程
    print("卷积计算示例 (以中心点[2,2]为例):")
    print("=" * 50)

    center_region = test_region[1:4, 1:4]  # 3x3区域
    print("图像区域 (3x3):")
    print(center_region)
    print()

    print("4邻域Laplacian核:")
    print(laplacian_4neighbor)
    print()

    print("逐元素相乘:")
    element_wise = center_region * laplacian_4neighbor
    print(element_wise)
    print()

    convolution_result = np.sum(element_wise)
    print(f"求和结果: {convolution_result}")
    print(f"这就是该点的Laplacian算子响应值")
    print(f"正值表示局部最小值，负值表示局部最大值")
    print()

    return (laplacian_4neighbor, laplacian_8neighbor, laplacian_diagonal,
            test_region, conv_4n, conv_8n, conv_diag)


# 演示Laplacian卷积核
laplacian_kernels = demonstrate_laplacian_kernels()

# ==================== 3. 一阶导数 vs 二阶导数 ====================
print("\n📊 3. 一阶导数 vs 二阶导数")
print("=" * 30)


def compare_first_second_derivative():
    """比较一阶导数和二阶导数的差异"""

    print("一阶导数 vs 二阶导数:")
    print("=" * 40)

    # 创建测试信号 - 模拟边缘
    x = np.linspace(0, 100, 500)

    # 创建阶梯边缘信号
    edge_signal = np.zeros_like(x)
    edge_signal[x > 50] = 100

    # 添加一些噪声
    noise = np.random.normal(0, 2, x.shape)
    noisy_signal = edge_signal + noise

    # 计算一阶导数（使用中心差分）
    first_derivative = np.zeros_like(noisy_signal)
    first_derivative[1:-1] = (noisy_signal[2:] - noisy_signal[:-2]) / 2

    # 计算二阶导数（使用中心差分）
    second_derivative = np.zeros_like(noisy_signal)
    second_derivative[1:-1] = (noisy_signal[2:] - 2 * noisy_signal[1:-1] + noisy_signal[:-2])

    # 可视化比较
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 原始信号
    axes[0].plot(x, noisy_signal, 'b-', linewidth=2, label='原始信号（含噪声）')
    axes[0].plot(x, edge_signal, 'r--', linewidth=1, alpha=0.7, label='理想边缘')
    axes[0].set_title("原始信号 - 阶梯边缘")
    axes[0].set_xlabel("位置")
    axes[0].set_ylabel("灰度值")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 标记边缘位置
    axes[0].axvline(x=50, color='green', linestyle=':', alpha=0.7, label='真实边缘位置')

    # 一阶导数
    axes[1].plot(x, first_derivative, 'g-', linewidth=2, label='一阶导数')
    axes[1].set_title("一阶导数（梯度）")
    axes[1].set_xlabel("位置")
    axes[1].set_ylabel("导数值")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 标记极值点
    max_deriv_idx = np.argmax(first_derivative)
    axes[1].axvline(x=x[max_deriv_idx], color='red', linestyle=':', alpha=0.7,
                    label=f'最大值位置: x={x[max_deriv_idx]:.1f}')

    # 二阶导数
    axes[2].plot(x, second_derivative, 'r-', linewidth=2, label='二阶导数')
    axes[2].set_title("二阶导数（Laplacian）")
    axes[2].set_xlabel("位置")
    axes[2].set_ylabel("导数值")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # 标记零交叉点
    # 找到零交叉点
    zero_crossings = np.where(np.diff(np.sign(second_derivative)))[0]
    for zc in zero_crossings:
        if 40 < x[zc] < 60:  # 只显示边缘附近的零交叉
            axes[2].axvline(x=x[zc], color='blue', linestyle=':', alpha=0.7,
                            label='零交叉点' if zc == zero_crossings[0] else "")

    plt.tight_layout()
    plt.show()

    # 详细解释差异
    print("一阶导数 vs 二阶导数特性对比:")
    print("=" * 50)

    print("一阶导数（Sobel算子）:")
    print("  - 检测灰度值的变化率（梯度）")
    print("  - 在边缘处达到最大值")
    print("  - 产生单边缘响应")
    print("  - 需要阈值处理")
    print("  - 能提供边缘方向信息")
    print()

    print("二阶导数（Laplacian算子）:")
    print("  - 检测灰度值变化率的变化率")
    print("  - 在边缘处产生零交叉")
    print("  - 产生双边缘响应")
    print("  - 不需要阈值，但需要零交叉检测")
    print("  - 各向同性，不提供方向信息")
    print()

    print("关键差异总结:")
    print("1. 边缘表示: 一阶导数→极值，二阶导数→零交叉")
    print("2. 响应数量: 一阶导数→单响应，二阶导数→双响应")
    print("3. 方向信息: 一阶导数→有方向，二阶导数→无方向")
    print("4. 噪声敏感度: 二阶导数比一阶导数更敏感")
    print("5. 定位精度: 二阶导数定位更精确")
    print()

    return x, noisy_signal, first_derivative, second_derivative, zero_crossings


# 比较一阶和二阶导数
derivative_comparison = compare_first_second_derivative()

# ==================== 4. 零交叉检测 ====================
print("\n🔍 4. 零交叉检测")
print("=" * 30)


def demonstrate_zero_crossing():
    """演示零交叉检测技术"""

    print("零交叉检测原理:")
    print("=" * 40)

    print("""
零交叉检测是Laplacian算子的关键步骤：

原理：
  - Laplacian算子的响应在边缘处通过零点
  - 零交叉点标识了边缘的位置
  - 通过检测符号变化来定位零交叉

检测方法：
  1. 简单零交叉：查找符号变化的点
  2. 阈值零交叉：只有梯度幅值超过阈值的零交叉才被认为是边缘
  3. 多尺度零交叉：在不同尺度下检测零交叉

数学定义：
  零交叉点满足：f(x) * f(x+1) < 0 且 |f(x) - f(x+1)| > 阈值

优点：
  - 不需要手动设置阈值
  - 边缘定位精确
  - 能检测细边缘

缺点：
  - 对噪声敏感
  - 可能产生虚假边缘
  - 计算复杂度较高
    """)

    # 创建测试图像
    test_img = np.zeros((100, 100), dtype=np.float32)

    # 添加各种边缘
    # 垂直边缘
    test_img[:, 40:60] = 100
    test_img[:, 60:] = 200

    # 圆形边缘
    y, x = np.ogrid[0:100, 0:100]
    center = (70, 30)
    radius = 15
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
    test_img[mask] = 150

    # 添加高斯噪声
    noise = np.random.normal(0, 5, test_img.shape)
    noisy_img = np.clip(test_img + noise, 0, 255)

    # 应用Laplacian算子
    laplacian = cv2.Laplacian(noisy_img.astype(np.uint8), cv2.CV_64F, ksize=3)

    # 零交叉检测函数
    def zero_crossing_detection(image, threshold=0):
        """零交叉检测实现"""
        height, width = image.shape
        zc_image = np.zeros_like(image, dtype=np.uint8)

        # 检查每个像素的邻域
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # 检查3x3邻域内的符号变化
                neighbors = [
                    image[i - 1, j - 1], image[i - 1, j], image[i - 1, j + 1],
                    image[i, j - 1], image[i, j + 1],
                    image[i + 1, j - 1], image[i + 1, j], image[i + 1, j + 1]
                ]

                # 检查是否存在零交叉
                positive_count = sum(1 for n in neighbors if n > threshold)
                negative_count = sum(1 for n in neighbors if n < -threshold)

                # 如果同时存在正负值，则认为是零交叉点
                if positive_count > 0 and negative_count > 0:
                    zc_image[i, j] = 255

        return zc_image

    def improved_zero_crossing(image, gradient_threshold=10):
        """改进的零交叉检测（结合梯度信息）"""
        height, width = image.shape
        zc_image = np.zeros_like(image, dtype=np.uint8)

        # 计算梯度幅值（用于阈值判断）
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # 检查梯度是否超过阈值
                if gradient_mag[i, j] < gradient_threshold:
                    continue

                # 检查4邻域的零交叉
                neighbors_4 = [image[i - 1, j], image[i + 1, j], image[i, j - 1], image[i, j + 1]]

                has_positive = any(n > 0 for n in neighbors_4)
                has_negative = any(n < 0 for n in neighbors_4)

                if has_positive and has_negative:
                    zc_image[i, j] = 255

        return zc_image

    # 应用零交叉检测
    zc_simple = zero_crossing_detection(laplacian, threshold=5)
    zc_improved = improved_zero_crossing(laplacian, gradient_threshold=20)

    # 可视化结果
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 第一行：原始图片和Laplacian结果
    axes[0, 0].imshow(noisy_img, cmap='gray')
    axes[0, 0].set_title("原始图片（含噪声）")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(laplacian, cmap='coolwarm')
    axes[0, 1].set_title("Laplacian算子响应")
    axes[0, 1].axis('off')

    # 显示Laplacian响应的符号分布
    sign_map = np.sign(laplacian)
    axes[0, 2].imshow(sign_map, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 2].set_title("Laplacian符号分布\n(红色:正, 蓝色:负)")
    axes[0, 2].axis('off')

    # 第二行：零交叉检测结果
    axes[1, 0].imshow(zc_simple, cmap='gray')
    axes[1, 0].set_title("简单零交叉检测")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(zc_improved, cmap='gray')
    axes[1, 1].set_title("改进零交叉检测\n(结合梯度阈值)")
    axes[1, 1].axis('off')

    # 零交叉检测原理说明
    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.7,
                    "零交叉检测原理:\n\n"
                    "基本条件:\n"
                    "• 在3x3邻域内同时存在\n  正值和负值\n\n"
                    "改进方法:\n"
                    "• 结合梯度幅值阈值\n"
                    "• 减少虚假边缘\n"
                    "• 提高检测质量\n\n"
                    "数学表达:\n"
                    "f(x)*f(x+1) < 0 且\n"
                    "|f(x)-f(x+1)| > 阈值",
                    fontsize=10, verticalalignment='center')

    plt.suptitle("零交叉检测技术", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 零交叉检测的数学演示
    print("零交叉检测数学演示:")
    print("=" * 40)

    # 创建简单的信号演示
    demo_signal = np.array([-2, -1, 0, 1, 2, 1, 0, -1, -2])
    print("示例信号:", demo_signal)

    # 检测零交叉
    zero_cross_points = []
    for i in range(len(demo_signal) - 1):
        if demo_signal[i] * demo_signal[i + 1] < 0:  # 符号变化
            zero_cross_points.append(i)

    print("零交叉点位置:", zero_cross_points)
    print("零交叉点值:", [demo_signal[i] for i in zero_cross_points])
    print()

    return noisy_img, laplacian, zc_simple, zc_improved


# 演示零交叉检测
zc_results = demonstrate_zero_crossing()

# ==================== 5. LoG算子（高斯-拉普拉斯算子）====================
print("\n🌊 5. LoG算子（高斯-拉普拉斯算子）")
print("=" * 30)


def demonstrate_log_operator():
    """演示LoG（Laplacian of Gaussian）算子"""

    print("""
LoG算子（Laplacian of Gaussian）:

基本原理：
  - 先对图像进行高斯滤波去噪
  - 再应用Laplacian算子检测边缘
  - 高斯滤波的尺度参数σ控制平滑程度

数学定义：
  LoG(x, y) = ∇²[G(x, y) * I(x, y)]
  其中 G(x, y) = (1/(2πσ²)) * exp(-(x²+y²)/(2σ²))

离散近似：
  常用5×5 LoG卷积核：
    [ 0,  0, -1,  0,  0]
    [ 0, -1, -2, -1,  0]
    [-1, -2, 16, -2, -1]
    [ 0, -1, -2, -1,  0]
    [ 0,  0, -1,  0,  0]

优点：
  - 结合了高斯平滑和拉普拉斯检测
  - 对噪声鲁棒性更好
  - 能检测多尺度边缘
  - 边缘定位精确

缺点：
  - 计算复杂度较高
  - 需要选择合适的σ值
  - 可能产生虚假边缘

应用场景：
  - 多尺度边缘检测
  - 斑点检测
  - 图像特征提取
  - 医学图像处理
    """)

    # 创建测试图片
    test_img = np.zeros((150, 200), dtype=np.uint8)

    # 添加各种边缘
    cv2.rectangle(test_img, (30, 30), (100, 100), 150, -1)
    cv2.circle(test_img, (150, 80), 30, 200, -1)
    cv2.putText(test_img, "LoG", (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, 180, 2)

    # 添加高斯噪声
    noise = np.random.normal(0, 20, test_img.shape)
    noisy_img = np.clip(test_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 手动实现LoG算子
    def manual_log_operator(image, sigma=1.0):
        """手动实现LoG算子"""
        # 1. 高斯滤波
        size = int(6 * sigma) + 1
        if size % 2 == 0:
            size += 1

        blurred = cv2.GaussianBlur(image, (size, size), sigma)

        # 2. Laplacian算子
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

        return laplacian

    # 使用不同的σ值
    sigmas = [0.5, 1.0, 1.5, 2.0]
    log_results = []

    for sigma in sigmas:
        log_result = manual_log_operator(noisy_img, sigma)
        log_results.append((sigma, log_result))

    # 可视化结果
    fig, axes = plt.subplots(2, 4, figsize=(12, 8))

    # 原始和噪声图片
    axes[0, 0].imshow(test_img, cmap='gray')
    axes[0, 0].set_title("原始图片")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_img, cmap='gray')
    axes[0, 1].set_title("加噪图片")
    axes[0, 1].axis('off')

    # 显示不同σ值的LoG结果
    for idx, (sigma, result) in enumerate(log_results[:3]):
        row = idx // 2
        col = idx % 2 + 2

        axes[row, col].imshow(result, cmap='coolwarm')
        axes[row, col].set_title(f"LoG算子\nσ={sigma}")
        axes[row, col].axis('off')

    # LoG核可视化
    axes[1, 0].axis('off')
    axes[1, 0].text(0.1, 0.7,
                    "LoG卷积核示例 (σ=1.4):\n\n"
                    "[ 0,  0, -1,  0,  0]\n"
                    "[ 0, -1, -2, -1,  0]\n"
                    "[-1, -2, 16, -2, -1]\n"
                    "[ 0, -1, -2, -1,  0]\n"
                    "[ 0,  0, -1,  0,  0]\n\n"
                    "高斯函数:\n"
                    "G(x,y) = exp(-(x²+y²)/(2σ²))\n\n"
                    "Laplacian:\n"
                    "∇²G = (x²+y²-2σ²)/σ⁴ * G",
                    fontsize=9, verticalalignment='center', family='monospace')

    plt.suptitle("LoG算子（高斯-拉普拉斯算子）", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 比较不同σ值的效果
    print("不同σ值对LoG算子的影响:")
    print("=" * 40)

    for sigma, result in log_results:
        # 计算零交叉点数量
        zero_crossings = np.sum(np.abs(np.diff(np.sign(result.flatten()))) > 0) / 2

        # 计算响应强度
        response_strength = np.mean(np.abs(result))

        print(f"σ={sigma}:")
        print(f"  零交叉点数量: {zero_crossings:.0f}")
        print(f"  平均响应强度: {response_strength:.2f}")
        print(f"  边缘粗细: {'细' if sigma < 1 else '中等' if sigma < 1.5 else '粗'}")
        print()

    return noisy_img, log_results


# 演示LoG算子
log_results = demonstrate_log_operator()

# ==================== 6. Laplacian算子的实际应用 ====================
print("\n💼 6. Laplacian算子的实际应用")
print("=" * 30)


def demonstrate_laplacian_applications():
    """演示Laplacian算子的实际应用"""

    print("Laplacian算子的实际应用场景:")
    print("1. 精细边缘检测: 检测细线和细节")
    print("2. 斑点检测: 检测图像中的小点")
    print("3. 图像增强: 锐化图像边缘")
    print("4. 零交叉检测: 精确边缘定位")
    print("5. 多尺度分析: 结合不同尺度检测边缘")
    print()

    # 应用1: 精细边缘检测
    print("应用1: 精细边缘检测")
    print("-" * 20)

    # 创建包含细线的测试图片
    fine_detail_img = np.zeros((150, 200), dtype=np.uint8)

    # 添加细线
    for i in range(0, 150, 10):
        cv2.line(fine_detail_img, (20, i), (180, i), 200, 1)

    # 添加纹理
    for i in range(5):
        for j in range(5):
            x = 30 + j * 30
            y = 30 + i * 20
            cv2.circle(fine_detail_img, (x, y), 2, 150, -1)

    # 添加文字
    cv2.putText(fine_detail_img, "DETAIL", (100, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 180, 1)

    # 应用不同算子
    # Sobel算子
    sobel_x = cv2.Sobel(fine_detail_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(fine_detail_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Laplacian算子
    laplacian = cv2.Laplacian(fine_detail_img, cv2.CV_64F, ksize=3)

    # LoG算子
    blurred = cv2.GaussianBlur(fine_detail_img, (5, 5), 1.0)
    log_result = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    # 可视化比较
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(fine_detail_img, cmap='gray')
    axes[0, 0].set_title("原始图片（细线细节）")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(sobel_mag, cmap='hot')
    axes[0, 1].set_title("Sobel算子")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.abs(laplacian), cmap='hot')
    axes[0, 2].set_title("Laplacian算子")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(np.abs(log_result), cmap='hot')
    axes[1, 0].set_title("LoG算子 (σ=1.0)")
    axes[1, 0].axis('off')

    # 应用2: 斑点检测
    print("应用2: 斑点检测")
    print("-" * 20)

    # 创建包含斑点的图片
    spot_img = np.zeros((100, 150), dtype=np.uint8)

    # 添加不同大小的斑点
    cv2.circle(spot_img, (30, 30), 3, 200, -1)
    cv2.circle(spot_img, (70, 30), 5, 200, -1)
    cv2.circle(spot_img, (110, 30), 8, 200, -1)

    # 添加高斯噪声
    spot_noisy = spot_img.astype(np.float32) + np.random.normal(0, 10, spot_img.shape)
    spot_noisy = np.clip(spot_noisy, 0, 255).astype(np.uint8)

    # 应用Laplacian进行斑点检测
    spot_laplacian = cv2.Laplacian(spot_noisy, cv2.CV_64F, ksize=3)

    # 斑点响应：负的局部极值
    spot_response = -spot_laplacian  # 斑点对应负响应

    axes[1, 1].imshow(spot_img, cmap='gray')
    axes[1, 1].set_title("斑点图片")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(spot_response, cmap='hot')
    axes[1, 2].set_title("斑点检测响应\n(负Laplacian)")
    axes[1, 2].axis('off')

    plt.suptitle("Laplacian算子的实际应用", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 应用3: 图像锐化
    print("应用3: 图像锐化")
    print("-" * 20)

    # 使用Laplacian进行图像锐化
    def laplacian_sharpening(image, alpha=0.3):
        """使用Laplacian进行图像锐化"""
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)

        # 锐化：原始图像减去Laplacian（因为中心为正）
        sharpened = image.astype(np.float64) - alpha * laplacian

        # 裁剪到有效范围
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        return sharpened

    # 测试锐化效果
    test_sharp_img = fine_detail_img.copy()
    sharpened = laplacian_sharpening(test_sharp_img, alpha=0.2)

    # 显示锐化效果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(test_sharp_img, cmap='gray')
    axes[0].set_title("原始图片")
    axes[0].axis('off')

    axes[1].imshow(sharpened, cmap='gray')
    axes[1].set_title("Laplacian锐化后")
    axes[1].axis('off')

    # 计算锐化增强效果
    edge_enhancement = np.mean(np.abs(sharpened.astype(np.float32) - test_sharp_img.astype(np.float32)))
    axes[2].bar(['边缘增强度'], [edge_enhancement], color='skyblue')
    axes[2].set_title(f"边缘增强效果\n平均变化: {edge_enhancement:.2f}")
    axes[2].set_ylabel("平均像素变化")
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle("Laplacian图像锐化应用", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return fine_detail_img, sobel_mag, laplacian, log_result, spot_img, spot_response, sharpened


# 演示实际应用
application_results = demonstrate_laplacian_applications()

# ==================== 7. Laplacian算子与其他算子对比 ====================
print("\n🔍 7. Laplacian算子与其他算子对比")
print("=" * 30)


def compare_laplacian_with_others():
    """比较Laplacian算子与其他边缘检测算子"""

    print("Laplacian vs 其他边缘检测算子:")
    print("=" * 40)

    # 创建测试图片
    test_img = np.zeros((150, 200), dtype=np.uint8)

    # 添加不同类型边缘
    # 阶梯边缘
    test_img[30:80, 50:100] = 100
    test_img[30:80, 100:150] = 200

    # 细线
    cv2.line(test_img, (20, 100), (180, 100), 150, 1)
    cv2.line(test_img, (20, 110), (180, 110), 150, 1)

    # 圆形
    cv2.circle(test_img, (160, 50), 20, 180, -1)

    # 添加高斯噪声
    noise = np.random.normal(0, 15, test_img.shape)
    noisy_img = np.clip(test_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 应用不同算子
    operators = []

    # 1. Sobel算子
    sobel_x = cv2.Sobel(noisy_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(noisy_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    operators.append(("Sobel", sobel_mag))

    # 2. Laplacian算子
    laplacian = cv2.Laplacian(noisy_img, cv2.CV_64F, ksize=3)
    operators.append(("Laplacian", np.abs(laplacian)))

    # 3. LoG算子
    blurred = cv2.GaussianBlur(noisy_img, (5, 5), 1.0)
    log_result = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    operators.append(("LoG (σ=1.0)", np.abs(log_result)))

    # 4. Canny算子（作为参考）
    canny_edges = cv2.Canny(noisy_img, 50, 150)
    operators.append(("Canny", canny_edges.astype(np.float64)))

    # 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 原始和噪声图片
    axes[0, 0].imshow(test_img, cmap='gray')
    axes[0, 0].set_title("原始图片")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_img, cmap='gray')
    axes[0, 1].set_title("加噪图片")
    axes[0, 1].axis('off')

    # 显示不同算子的结果
    positions = [(0, 2), (1, 0), (1, 1), (1, 2)]
    for idx, ((name, result), (row, col)) in enumerate(zip(operators, positions)):
        axes[row, col].imshow(result, cmap='hot')
        axes[row, col].set_title(f"{name}算子")
        axes[row, col].axis('off')

    # 算子特性说明
    axes[0, 2].axis('off')
    axes[0, 2].text(0.1, 0.5,
                    "算子特性对比:\n\n"
                    "Sobel算子:\n"
                    "  - 一阶导数\n"
                    "  - 对噪声中等敏感\n"
                    "  - 有方向性\n\n"
                    "Laplacian算子:\n"
                    "  - 二阶导数\n"
                    "  - 对噪声敏感\n"
                    "  - 各向同性",
                    fontsize=9, verticalalignment='center')

    # 性能对比
    times = []
    names = []

    for name, _ in operators:
        if name == "Canny":
            continue  # Canny计算复杂度不同，单独处理

        start_time = time.time()
        for _ in range(100):  # 重复100次
            if name == "Sobel":
                cv2.Sobel(noisy_img, cv2.CV_64F, 1, 0, ksize=3)
                cv2.Sobel(noisy_img, cv2.CV_64F, 0, 1, ksize=3)
            elif name == "Laplacian":
                cv2.Laplacian(noisy_img, cv2.CV_64F, ksize=3)
            elif name == "LoG (σ=1.0)":
                blurred = cv2.GaussianBlur(noisy_img, (5, 5), 1.0)
                cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        end_time = time.time()

        avg_time = (end_time - start_time) / 100
        times.append(avg_time)
        names.append(name)

    # 添加Canny时间
    start_time = time.time()
    for _ in range(100):
        cv2.Canny(noisy_img, 50, 150)
    end_time = time.time()
    times.append((end_time - start_time) / 100)
    names.append("Canny")

    plt.suptitle("Laplacian算子与其他算子对比", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 绘制性能对比图
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['blue', 'green', 'orange', 'red']
    bars = ax.bar(names, times, color=colors)
    ax.set_title("计算时间对比")
    ax.set_ylabel("时间 (秒)")
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱状图上显示数值
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.0001,
                f'{time_val:.6f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # 算子特性总结
    print("算子特性总结:")
    print("=" * 40)
    print("Sobel算子:")
    print("  - 优点: 计算快，有一定抗噪能力")
    print("  - 缺点: 边缘较粗，对细线检测差")
    print("  - 适用: 实时处理，一般边缘检测")
    print()

    print("Laplacian算子:")
    print("  - 优点: 定位精确，能检测细线")
    print("  - 缺点: 对噪声敏感，产生双边缘")
    print("  - 适用: 精细边缘检测，无噪声环境")
    print()

    print("LoG算子:")
    print("  - 优点: 抗噪性好，多尺度检测")
    print("  - 缺点: 计算复杂，需要调参")
    print("  - 适用: 多尺度边缘检测，医学图像")
    print()

    print("Canny算子:")
    print("  - 优点: 抗噪性好，单边缘响应")
    print("  - 缺点: 计算复杂，需要调参")
    print("  - 适用: 高质量边缘检测")
    print()

    return operators, times


# 比较Laplacian与其他算子
comparison_results = compare_laplacian_with_others()

# ==================== 8. 练习与挑战 ====================
print("\n💪 8. 练习与挑战")
print("=" * 30)

print("""
练习题：

1. 基础练习：
   a) 实现手动Laplacian算子，支持4邻域和8邻域
   b) 实现零交叉检测算法
   c) 实现LoG算子，支持不同σ值

2. 进阶练习：
   a) 实现自适应Laplacian阈值
   b) 实现多尺度LoG边缘检测
   c) 实现Laplacian金字塔

3. 思考题：
   a) 为什么Laplacian算子对噪声敏感？
   b) 零交叉检测的原理是什么？
   c) LoG算子相比普通Laplacian有什么优势？
   d) 在什么情况下应该使用Laplacian算子？
""")

# 练习框架代码
print("\n💻 练习框架代码：")

print("""
# 练习1a: 手动Laplacian算子
def manual_laplacian(image, neighbor_type=4):
    if neighbor_type == 4:
        kernel = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]], dtype=np.float32)
    elif neighbor_type == 8:
        kernel = np.array([[-1, -1, -1],
                          [-1, 8, -1],
                          [-1, -1, -1]], dtype=np.float32)
    else:
        raise ValueError("neighbor_type must be 4 or 8")

    # 使用filter2D计算卷积
    result = cv2.filter2D(image.astype(np.float32), -1, kernel)
    return result

# 练习1b: 零交叉检测
def zero_crossing_detection_advanced(image, threshold=0.1):
    height, width = image.shape
    zc_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(1, height-1):
        for j in range(1, width-1):
            # 检查4邻域的符号变化
            neighbors = [image[i-1, j], image[i+1, j], 
                        image[i, j-1], image[i, j+1]]

            # 检查是否存在跨越零点的变化
            max_pos = max(n for n in neighbors if n > 0)
            min_neg = min(n for n in neighbors if n < 0)

            if max_pos > 0 and min_neg < 0 and (max_pos - min_neg) > threshold:
                zc_image[i, j] = 255

    return zc_image

# 练习1c: LoG算子
def log_operator(image, sigma=1.0, ksize=None):
    if ksize is None:
        ksize = int(6*sigma) + 1
        if ksize % 2 == 0:
            ksize += 1

    # 高斯滤波
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    # Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    return laplacian

# 练习2a: 自适应Laplacian阈值
def adaptive_laplacian_threshold(image, neighbor_type=4):
    # 计算Laplacian
    laplacian = manual_laplacian(image, neighbor_type)

    # 自适应阈值
    mean_val = np.mean(np.abs(laplacian))
    std_val = np.std(laplacian)
    threshold = mean_val + 2 * std_val

    # 二值化
    binary = (np.abs(laplacian) > threshold).astype(np.uint8) * 255

    return binary, threshold

# 练习2b: 多尺度LoG
def multi_scale_log(image, sigmas=[0.5, 1.0, 1.5, 2.0]):
    results = []

    for sigma in sigmas:
        log_result = log_operator(image, sigma)
        results.append((sigma, log_result))

    # 合并多尺度结果
    combined = np.zeros_like(image, dtype=np.float32)
    for sigma, result in results:
        combined += result / len(sigmas)

    return combined, results
""")

# ==================== 9. 总结 ====================
print("\n" + "=" * 50)
print("✅ Laplacian算子总结")
print("=" * 50)

summary = """
📊 Laplacian算子核心知识：

1. 数学原理
   - 二阶导数算子: ∇²I = ∂²I/∂x² + ∂²I/∂y²
   - 离散卷积核: 4邻域: [[0,-1,0],[-1,4,-1],[0,-1,0]]
                 8邻域: [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
   - 零和性质: 卷积核元素之和为0

2. 实现方法
   - OpenCV: cv2.Laplacian(src, ddepth, ksize, scale, delta)
   - 手动实现: 卷积计算，支持不同邻域
   - LoG算子: 先高斯滤波，再Laplacian

3. 关键概念
   - 零交叉检测: 检测符号变化的点
   - 二阶导数: 检测灰度变化率的变化
   - 各向同性: 对各个方向响应相同
   - 双边缘响应: 每个边缘产生两个响应

4. 性能特点
   - 时间复杂度: O(N²k²)，N为图像尺寸，k为核大小
   - 空间复杂度: O(N²)
   - 噪声敏感度: 高，对噪声非常敏感
   - 定位精度: 高，边缘定位精确

5. 优点
   - 边缘定位精度高
   - 能检测细线和细节
   - 各向同性，不依赖方向
   - 不需要阈值处理（使用零交叉）

6. 缺点
   - 对噪声非常敏感
   - 产生双边缘响应
   - 需要零交叉检测
   - 计算复杂度较高

7. 实际应用
   - 精细边缘检测: 检测细线、纹理
   - 斑点检测: 检测小点、孤立点
   - 图像锐化: 增强图像边缘
   - 零交叉检测: 精确边缘定位
   - 多尺度分析: 结合不同尺度

8. 最佳实践
   - 预处理: 必须先进行高斯滤波
   - 参数选择: σ值影响检测尺度
   - 后处理: 零交叉检测和连接
   - 结合使用: 与一阶算子结合使用

🎯 核心代码记忆：
   # OpenCV Laplacian基本用法
   laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)

   # LoG算子实现
   blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
   log_result = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

   # 零交叉检测
   def zero_crossing(image, threshold=0):
       height, width = image.shape
       zc = np.zeros((height, width), dtype=np.uint8)
       for i in range(1, height-1):
           for j in range(1, width-1):
               if (image[i,j] > 0 and image[i+1,j] < 0) or 
                  (image[i,j] < 0 and image[i+1,j] > 0):
                   zc[i,j] = 255
       return zc
"""

print(summary)
print("\n📁 下一个文件: 06_04_Canny边缘检测.py")
print("  我们将学习最经典的边缘检测算法：Canny算子！")