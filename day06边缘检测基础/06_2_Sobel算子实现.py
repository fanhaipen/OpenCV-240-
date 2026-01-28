"""
第6天 - 文件2：Sobel算子实现
学习目标：掌握Sobel算子的原理、实现和应用
重点：Sobel卷积核、梯度计算、边缘方向、实际应用
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

print("🔍 第6天 - 文件2：Sobel算子实现")
print("=" * 50)

# ==================== 1. Sobel算子理论 ====================
print("\n🎯 1. Sobel算子理论")
print("=" * 30)

print("""
Sobel算子：

基本概念：
  - 一阶微分算子，用于计算图像梯度
  - 结合了高斯平滑和微分操作
  - 对噪声有一定的抑制能力

数学原理：
  Sobel算子通过卷积计算图像的近似梯度

卷积核：
  x方向核（检测垂直边缘）：
    [-1, 0, 1]
    [-2, 0, 2]
    [-1, 0, 1]

  y方向核（检测水平边缘）：
    [-1, -2, -1]
    [ 0,  0,  0]
    [ 1,  2,  1]

梯度计算：
  Gx = I * Sobel_x  (x方向梯度)
  Gy = I * Sobel_y  (y方向梯度)
  梯度幅值: |G| = √(Gx² + Gy²)
  梯度方向: θ = atan2(Gy, Gx)

特点：
  - 计算简单快速
  - 对噪声有一定鲁棒性
  - 能检测边缘方向和强度
  - 边缘定位精度较好

优点：
  - 实现简单
  - 计算效率高
  - 能提供边缘方向信息
  - 对噪声有一定抑制作用

缺点：
  - 对噪声仍较敏感
  - 边缘可能较粗
  - 对复杂纹理效果一般

应用场景：
  - 实时边缘检测
  - 图像特征提取
  - 计算机视觉预处理
  - 方向估计
""")

# ==================== 2. Sobel卷积核详解 ====================
print("\n🔧 2. Sobel卷积核详解")
print("=" * 30)


def demonstrate_sobel_kernels():
    """详细讲解Sobel卷积核"""

    print("Sobel卷积核的数学原理:")
    print("=" * 40)

    # 定义Sobel卷积核
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)

    print("x方向卷积核 (检测垂直边缘):")
    print(sobel_x)
    print()

    print("y方向卷积核 (检测水平边缘):")
    print(sobel_y)
    print()

    # 解释卷积核的设计原理
    print("卷积核设计原理:")
    print("1. 中心差分: 核中心为0，计算相邻像素的差异")
    print("2. 权重分配: 中心行权重更大，增强中心像素的重要性")
    print("3. 平滑效果: 垂直方向加权平均，抑制噪声")
    print("4. 方向性: x核检测垂直边缘，y核检测水平边缘")
    print()

    # 可视化卷积核
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # x方向核可视化
    im1 = axes[0, 0].imshow(sobel_x, cmap='coolwarm', vmin=-2, vmax=2)
    axes[0, 0].set_title("Sobel X方向核\n(检测垂直边缘)")
    axes[0, 0].set_xticks([0, 1, 2])
    axes[0, 0].set_yticks([0, 1, 2])
    plt.colorbar(im1, ax=axes[0, 0])

    # 在图中显示数值
    for i in range(3):
        for j in range(3):
            axes[0, 0].text(j, i, f'{sobel_x[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if abs(sobel_x[i, j]) > 1 else 'black',
                            fontsize=12, fontweight='bold')

    # y方向核可视化
    im2 = axes[0, 1].imshow(sobel_y, cmap='coolwarm', vmin=-2, vmax=2)
    axes[0, 1].set_title("Sobel Y方向核\n(检测水平边缘)")
    axes[0, 1].set_xticks([0, 1, 2])
    axes[0, 1].set_yticks([0, 1, 2])
    plt.colorbar(im2, ax=axes[0, 1])

    for i in range(3):
        for j in range(3):
            axes[0, 1].text(j, i, f'{sobel_y[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if abs(sobel_y[i, j]) > 1 else 'black',
                            fontsize=12, fontweight='bold')

    # 卷积计算演示
    # 创建一个简单的图像区域
    test_region = np.array([
        [10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10],
        [10, 10, 100, 200, 200],
        [10, 10, 200, 200, 200],
        [10, 10, 200, 200, 200]
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

    # 计算卷积结果
    conv_x = manual_convolution(test_region, sobel_x)
    conv_y = manual_convolution(test_region, sobel_y)

    # 显示原始图像区域
    im3 = axes[0, 2].imshow(test_region, cmap='gray')
    axes[0, 2].set_title("测试图像区域")
    axes[0, 2].set_xticks(range(5))
    axes[0, 2].set_yticks(range(5))
    plt.colorbar(im3, ax=axes[0, 2])

    for i in range(5):
        for j in range(5):
            axes[0, 2].text(j, i, f'{test_region[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if test_region[i, j] < 100 else 'black')

    # 显示x方向卷积结果
    im4 = axes[1, 0].imshow(conv_x, cmap='coolwarm')
    axes[1, 0].set_title("X方向卷积结果")
    axes[1, 0].set_xticks(range(5))
    axes[1, 0].set_yticks(range(5))
    plt.colorbar(im4, ax=axes[1, 0])

    for i in range(5):
        for j in range(5):
            axes[1, 0].text(j, i, f'{conv_x[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if abs(conv_x[i, j]) < 100 else 'black')

    # 显示y方向卷积结果
    im5 = axes[1, 1].imshow(conv_y, cmap='coolwarm')
    axes[1, 1].set_title("Y方向卷积结果")
    axes[1, 1].set_xticks(range(5))
    axes[1, 1].set_yticks(range(5))
    plt.colorbar(im5, ax=axes[1, 1])

    for i in range(5):
        for j in range(5):
            axes[1, 1].text(j, i, f'{conv_y[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if abs(conv_y[i, j]) < 100 else 'black')

    # 梯度幅值计算
    gradient_magnitude = np.sqrt(conv_x ** 2 + conv_y ** 2)
    im6 = axes[1, 2].imshow(gradient_magnitude, cmap='hot')
    axes[1, 2].set_title("梯度幅值 |G| = √(Gx² + Gy²)")
    axes[1, 2].set_xticks(range(5))
    axes[1, 2].set_yticks(range(5))
    plt.colorbar(im6, ax=axes[1, 2])

    for i in range(5):
        for j in range(5):
            axes[1, 2].text(j, i, f'{gradient_magnitude[i, j]:.0f}',
                            ha='center', va='center',
                            color='white' if gradient_magnitude[i, j] < 100 else 'black')

    plt.suptitle("Sobel卷积核详解与计算演示", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 详细解释卷积计算过程
    print("卷积计算示例 (以中心点[2,2]为例):")
    print("=" * 50)

    center_region = test_region[1:4, 1:4]  # 3x3区域
    print("图像区域 (3x3):")
    print(center_region)
    print()

    print("Sobel X核:")
    print(sobel_x)
    print()

    print("逐元素相乘:")
    element_wise = center_region * sobel_x
    print(element_wise)
    print()

    convolution_result = np.sum(element_wise)
    print(f"求和结果: {convolution_result}")
    print(f"这就是该点的x方向梯度值")
    print()

    return sobel_x, sobel_y, test_region, conv_x, conv_y, gradient_magnitude


# 演示Sobel卷积核
sobel_x, sobel_y, test_region, conv_x, conv_y, grad_mag = demonstrate_sobel_kernels()

# ==================== 3. 手动实现Sobel算子 ====================
print("\n🔧 3. 手动实现Sobel算子")
print("=" * 30)


def manual_sobel_implementation():
    """手动实现完整的Sobel算子"""

    print("手动实现Sobel算子步骤:")
    print("1. 边界处理")
    print("2. 分别计算x和y方向梯度")
    print("3. 计算梯度幅值和方向")
    print("4. 可选：梯度幅值归一化")
    print()

    def manual_sobel(image, ksize=3, normalize=True):
        """
        手动实现Sobel算子

        参数:
            image: 输入图像
            ksize: 卷积核大小（必须为奇数）
            normalize: 是否归一化梯度幅值

        返回:
            grad_x: x方向梯度
            grad_y: y方向梯度
            magnitude: 梯度幅值
            direction: 梯度方向（弧度）
        """

        if ksize != 3:
            raise ValueError("手动实现目前只支持3x3卷积核")

        height, width = image.shape
        pad = ksize // 2

        # 定义Sobel卷积核
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=np.float32)

        # 边界填充（反射填充）
        padded = np.pad(image.astype(np.float32), pad, mode='reflect')

        # 初始化输出
        grad_x = np.zeros_like(image, dtype=np.float32)
        grad_y = np.zeros_like(image, dtype=np.float32)

        # 应用卷积
        for i in range(pad, height + pad):
            for j in range(pad, width + pad):
                # 提取3x3区域
                region = padded[i - pad:i + pad + 1, j - pad:j + pad + 1]

                # 计算梯度
                grad_x[i - pad, j - pad] = np.sum(region * sobel_x)
                grad_y[i - pad, j - pad] = np.sum(region * sobel_y)

        # 计算梯度幅值
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 计算梯度方向
        direction = np.arctan2(grad_y, grad_x)

        # 可选：归一化到0-255
        if normalize:
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        return grad_x, grad_y, magnitude, direction

    def fast_manual_sobel(image, normalize=True):
        """
        快速手动实现（使用向量化操作）
        """
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=np.float32)

        # 使用OpenCV的filter2D加速计算
        grad_x = cv2.filter2D(image.astype(np.float32), -1, sobel_x)
        grad_y = cv2.filter2D(image.astype(np.float32), -1, sobel_y)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        direction = np.arctan2(grad_y, grad_x)

        if normalize:
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        return grad_x, grad_y, magnitude, direction

    # 创建测试图片
    def create_test_image():
        """创建用于Sobel测试的图片"""
        img = np.zeros((200, 300), dtype=np.uint8)

        # 添加各种边缘
        # 垂直边缘
        img[30:80, 100:150] = 100
        img[30:80, 150:200] = 200

        # 水平边缘
        img[100:120, 50:250] = 150

        # 斜边缘
        for i in range(50):
            x = 50 + i
            y = 150 + i
            if x < 300 and y < 200:
                img[y, x] = 180
                if y + 1 < 200:
                    img[y + 1, x] = 180

        # 圆形边缘
        cv2.circle(img, (250, 80), 20, 120, -1)

        # 文字边缘
        cv2.putText(img, "SOBEL", (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, 200, 2)

        return img

    # 创建测试图片
    test_img = create_test_image()

    print("测试手动实现Sobel算子...")
    print(f"图片尺寸: {test_img.shape[1]}x{test_img.shape[0]}")

    # 测试手动实现
    start_time = time.time()
    grad_x_manual, grad_y_manual, mag_manual, dir_manual = manual_sobel(test_img)
    manual_time = time.time() - start_time

    start_time = time.time()
    grad_x_fast, grad_y_fast, mag_fast, dir_fast = fast_manual_sobel(test_img)
    fast_time = time.time() - start_time

    # 使用OpenCV的Sobel函数
    start_time = time.time()
    sobel_x_cv = cv2.Sobel(test_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_cv = cv2.Sobel(test_img, cv2.CV_64F, 0, 1, ksize=3)
    mag_cv = np.sqrt(sobel_x_cv ** 2 + sobel_y_cv ** 2)
    mag_cv = cv2.normalize(mag_cv, None, 0, 255, cv2.NORM_MINMAX)
    cv_time = time.time() - start_time

    print(f"计算时间对比:")
    print(f"  基本手动实现: {manual_time:.4f}秒")
    print(f"  快速手动实现: {fast_time:.4f}秒")
    print(f"  OpenCV实现: {cv_time:.4f}秒")
    print()

    # 比较结果差异
    diff_x = np.max(np.abs(sobel_x_cv - grad_x_fast))
    diff_y = np.max(np.abs(sobel_y_cv - grad_y_fast))
    diff_mag = np.max(np.abs(mag_cv - mag_fast))

    print(f"结果差异 (与OpenCV对比):")
    print(f"  X梯度最大差异: {diff_x:.6f}")
    print(f"  Y梯度最大差异: {diff_y:.6f}")
    print(f"  幅值最大差异: {diff_mag:.6f}")
    print()

    # 可视化结果
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    # 第一行：原始图片和梯度分量
    axes[0, 0].imshow(test_img, cmap='gray')
    axes[0, 0].set_title("原始图片")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.abs(grad_x_fast), cmap='hot')
    axes[0, 1].set_title("X方向梯度 (手动实现)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.abs(grad_y_fast), cmap='hot')
    axes[0, 2].set_title("Y方向梯度 (手动实现)")
    axes[0, 2].axis('off')

    axes[0, 3].imshow(mag_fast, cmap='hot')
    axes[0, 3].set_title("梯度幅值 (手动实现)")
    axes[0, 3].axis('off')

    # 第二行：OpenCV实现和方向可视化
    axes[1, 0].imshow(mag_cv, cmap='hot')
    axes[1, 0].set_title("梯度幅值 (OpenCV)")
    axes[1, 0].axis('off')

    # 梯度方向可视化（使用HSV色彩空间）
    hsv_direction = np.zeros((test_img.shape[0], test_img.shape[1], 3))
    hsv_direction[:, :, 0] = (dir_fast + np.pi) / (2 * np.pi) * 180  # 色调：方向
    hsv_direction[:, :, 1] = 1.0  # 饱和度：最大
    hsv_direction[:, :, 2] = mag_fast / 255.0  # 明度：梯度幅值

    rgb_direction = cv2.cvtColor((hsv_direction * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    axes[1, 1].imshow(rgb_direction)
    axes[1, 1].set_title("梯度方向 (颜色表示)")
    axes[1, 1].axis('off')

    # 方向图例
    legend = np.zeros((100, 300, 3), dtype=np.uint8)
    for i in range(300):
        hue = i / 300 * 180  # 0-180度
        legend[:, i, 0] = hue
        legend[:, i, 1] = 255
        legend[:, i, 2] = 255

    legend_rgb = cv2.cvtColor(legend, cv2.COLOR_HSV2RGB)
    axes[1, 2].imshow(legend_rgb)
    axes[1, 2].set_title("方向图例\n0°: 红, 90°: 青, 180°: 红")
    axes[1, 2].axis('off')
    axes[1, 2].text(150, 50, "梯度方向颜色编码", ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold')

    # 性能对比
    times = [manual_time, fast_time, cv_time]
    labels = ['基本手动', '快速手动', 'OpenCV']
    colors = ['lightblue', 'lightgreen', 'lightcoral']

    axes[1, 3].bar(labels, times, color=colors)
    axes[1, 3].set_title("计算时间对比")
    axes[1, 3].set_ylabel("时间 (秒)")
    axes[1, 3].grid(True, alpha=0.3, axis='y')

    # 在柱状图上显示数值
    for i, v in enumerate(times):
        axes[1, 3].text(i, v + 0.001, f'{v:.4f}s',
                        ha='center', va='bottom', fontweight='bold')

    plt.suptitle("手动实现Sobel算子", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return (test_img, grad_x_fast, grad_y_fast, mag_fast, dir_fast,
            sobel_x_cv, sobel_y_cv, mag_cv, manual_time, fast_time, cv_time)


# 手动实现Sobel算子
manual_results = manual_sobel_implementation()

# ==================== 4. OpenCV Sobel函数详解 ====================
print("\n🔧 4. OpenCV Sobel函数详解")
print("=" * 30)


def demonstrate_opencv_sobel():
    """详细演示OpenCV的Sobel函数"""

    print("OpenCV Sobel函数:")
    print("cv2.Sobel(src, ddepth, dx, dy, ksize, scale, delta, borderType)")
    print()
    print("参数说明:")
    print("  src: 输入图像")
    print("  ddepth: 输出图像深度")
    print("    - cv2.CV_8U: 8位无符号整数 (0-255)")
    print("    - cv2.CV_16S: 16位有符号整数")
    print("    - cv2.CV_32F: 32位浮点数")
    print("    - cv2.CV_64F: 64位浮点数")
    print("  dx: x方向导数阶数")
    print("  dy: y方向导数阶数")
    print("  ksize: Sobel核大小 (1, 3, 5, 7)")
    print("  scale: 缩放因子")
    print("  delta: 添加到结果的增量")
    print("  borderType: 边界填充类型")
    print()

    # 创建测试图片
    test_img = np.zeros((150, 200), dtype=np.uint8)
    test_img[50:100, 50:150] = 255
    cv2.circle(test_img, (100, 75), 20, 150, -1)
    cv2.putText(test_img, "TEST", (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 200, 2)

    # 测试不同参数
    test_cases = [
        # (dx, dy, ksize, ddepth, description)
        (1, 0, 3, cv2.CV_64F, "X方向梯度, 3x3核, 64F"),
        (0, 1, 3, cv2.CV_64F, "Y方向梯度, 3x3核, 64F"),
        (1, 0, 5, cv2.CV_64F, "X方向梯度, 5x5核, 64F"),
        (0, 1, 5, cv2.CV_64F, "Y方向梯度, 5x5核, 64F"),
        (1, 0, 3, cv2.CV_8U, "X方向梯度, 3x3核, 8U"),
        (0, 1, 3, cv2.CV_8U, "Y方向梯度, 3x3核, 8U"),
        (2, 0, 3, cv2.CV_64F, "X方向二阶导数, 3x3核"),
        (0, 2, 3, cv2.CV_64F, "Y方向二阶导数, 3x3核"),
    ]

    results = []

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    # 显示原始图片
    axes[0, 0].imshow(test_img, cmap='gray')
    axes[0, 0].set_title("原始图片")
    axes[0, 0].axis('off')

    for idx, (dx, dy, ksize, ddepth, description) in enumerate(test_cases):
        row = (idx + 1) // 3
        col = (idx + 1) % 3

        if row < 3 and col < 3:
            # 应用Sobel
            sobel_result = cv2.Sobel(test_img, ddepth, dx, dy, ksize=ksize)

            # 处理不同的深度类型
            if ddepth in [cv2.CV_8U, cv2.CV_16S]:
                # 取绝对值
                sobel_result = cv2.convertScaleAbs(sobel_result)

            # 显示结果
            axes[row, col].imshow(np.abs(sobel_result), cmap='hot')
            axes[row, col].set_title(description)
            axes[row, col].axis('off')

            # 保存结果用于分析
            results.append((description, sobel_result))

    # 添加参数说明
    axes[2, 2].axis('off')
    axes[2, 2].text(0.1, 0.5,
                    "参数影响总结:\n\n"
                    "ksize (核大小):\n"
                    "  - 3x3: 标准Sobel核\n"
                    "  - 5x5: 更大的平滑效果\n\n"
                    "ddepth (深度):\n"
                    "  - CV_8U: 0-255, 可能截断负值\n"
                    "  - CV_64F: 浮点数, 保留正负\n\n"
                    "dx/dy (导数阶数):\n"
                    "  - 1: 一阶导数\n"
                    "  - 2: 二阶导数",
                    fontsize=10, verticalalignment='center')

    plt.suptitle("OpenCV Sobel函数不同参数效果", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 详细分析不同参数的效果
    print("参数影响分析:")
    print("=" * 40)

    for description, result in results:
        # 计算统计信息
        min_val = np.min(result)
        max_val = np.max(result)
        mean_val = np.mean(np.abs(result))

        print(f"{description}:")
        print(f"  最小值: {min_val:.2f}, 最大值: {max_val:.2f}, 平均绝对值: {mean_val:.2f}")

        if "8U" in description and min_val < 0:
            print("  注意: 8U类型会截断负值，使用convertScaleAbs处理")
        print()

    return test_img, results


# 演示OpenCV Sobel函数
test_img, sobel_results = demonstrate_opencv_sobel()

# ==================== 5. Sobel算子在实际中的应用 ====================
print("\n💼 5. Sobel算子在实际中的应用")
print("=" * 30)


def demonstrate_sobel_applications():
    """演示Sobel算子的实际应用"""

    print("Sobel算子的实际应用场景:")
    print("1. 边缘检测: 检测图像中的物体边界")
    print("2. 方向估计: 估计边缘的方向")
    print("3. 特征提取: 提取图像的特征")
    print("4. 图像增强: 增强图像的边缘信息")
    print("5. 计算机视觉预处理: 为其他算法准备数据")
    print()

    # 应用1: 边缘检测
    print("应用1: 边缘检测")
    print("-" * 20)

    # 使用真实图片
    # 创建一个模拟真实场景的图片
    real_world_img = np.zeros((200, 300), dtype=np.uint8)

    # 添加各种物体
    # 矩形物体
    cv2.rectangle(real_world_img, (30, 30), (120, 100), 180, -1)
    # 圆形物体
    cv2.circle(real_world_img, (200, 80), 40, 150, -1)
    # 三角形
    pts = np.array([[250, 150], [280, 100], [310, 150]], np.int32)
    cv2.fillPoly(real_world_img, [pts], 120)
    # 文字
    cv2.putText(real_world_img, "OpenCV", (100, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 200, 2)

    # 添加噪声
    noise = np.random.normal(0, 20, real_world_img.shape)
    noisy_img = np.clip(real_world_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 应用Sobel
    sobel_x = cv2.Sobel(noisy_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(noisy_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX)

    # 阈值处理得到二值边缘
    _, binary_edges = cv2.threshold(sobel_mag_norm, 50, 255, cv2.THRESH_BINARY)

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(real_world_img, cmap='gray')
    axes[0, 0].set_title("原始图片")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_img, cmap='gray')
    axes[0, 1].set_title("加噪图片")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(sobel_mag_norm, cmap='hot')
    axes[0, 2].set_title("Sobel梯度幅值")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(binary_edges, cmap='gray')
    axes[1, 0].set_title("二值化边缘")
    axes[1, 0].axis('off')

    # 应用2: 方向估计
    print("应用2: 方向估计")
    print("-" * 20)

    # 计算梯度方向
    gradient_dir = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

    # 将方向量化为8个方向
    dir_bins = 8
    dir_quantized = ((gradient_dir + 180) / 360 * dir_bins).astype(int) % dir_bins

    # 创建方向直方图
    dir_hist, _ = np.histogram(dir_quantized, bins=dir_bins, range=(0, dir_bins))

    axes[1, 1].bar(range(dir_bins), dir_hist)
    axes[1, 1].set_title("梯度方向直方图")
    axes[1, 1].set_xlabel("方向 (45度间隔)")
    axes[1, 1].set_ylabel("像素数量")
    axes[1, 1].grid(True, alpha=0.3)

    # 应用3: 边缘增强
    print("应用3: 边缘增强")
    print("-" * 20)

    # 将边缘加到原始图片
    enhanced_img = cv2.addWeighted(real_world_img, 0.7,
                                   binary_edges.astype(np.uint8), 0.3, 0)

    axes[1, 2].imshow(enhanced_img, cmap='gray')
    axes[1, 2].set_title("边缘增强结果")
    axes[1, 2].axis('off')

    plt.suptitle("Sobel算子在实际中的应用", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 性能分析
    print("性能分析:")
    print("-" * 20)

    # 测试不同尺寸图片的计算时间
    sizes = [(100, 100), (200, 200), (400, 400), (800, 800)]
    times = []

    for h, w in sizes:
        test_img = np.random.randint(0, 256, (h, w), dtype=np.uint8)

        start_time = time.time()
        for _ in range(10):  # 重复10次取平均
            sobel_x = cv2.Sobel(test_img, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(test_img, cv2.CV_64F, 0, 1, ksize=3)
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        times.append(avg_time)

        print(f"尺寸 {w}x{h}: 平均时间 {avg_time:.6f}秒")

    # 绘制性能曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    pixel_counts = [h * w for h, w in sizes]
    ax.plot(pixel_counts, times, 'bo-', linewidth=2, markersize=8)
    ax.set_title("Sobel算子计算时间 vs 图片尺寸")
    ax.set_xlabel("像素数量")
    ax.set_ylabel("计算时间 (秒)")
    ax.grid(True, alpha=0.3)

    # 添加趋势线
    z = np.polyfit(pixel_counts, times, 1)
    p = np.poly1d(z)
    ax.plot(pixel_counts, p(pixel_counts), 'r--', alpha=0.5, label=f'线性趋势: y={z[0]:.2e}x+{z[1]:.2e}')
    ax.legend()

    plt.tight_layout()
    plt.show()

    return real_world_img, noisy_img, sobel_mag_norm, binary_edges, dir_hist, enhanced_img


# 演示实际应用
app_results = demonstrate_sobel_applications()

# ==================== 6. Sobel算子与其他算子对比 ====================
# ==================== 6. Sobel算子与其他算子对比 ====================
print("\n🔍 6. Sobel算子与其他算子对比")
print("=" * 30)


def compare_sobel_with_others():
    """比较Sobel算子与其他边缘检测算子"""

    print("Sobel vs 其他边缘检测算子:")
    print("=" * 40)

    # 创建测试图片
    test_img = np.zeros((150, 200), dtype=np.uint8)
    test_img[50:100, 50:150] = 255
    cv2.circle(test_img, (100, 75), 20, 150, -1)

    # 添加噪声
    noise = np.random.normal(0, 15, test_img.shape)
    noisy_img = np.clip(test_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 应用不同算子
    operators = []

    # 1. Sobel
    sobel_x = cv2.Sobel(noisy_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(noisy_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    operators.append(("Sobel", sobel_mag))

    # 2. Prewitt
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]], dtype=np.float32)

    prewitt_gx = cv2.filter2D(noisy_img.astype(np.float32), -1, prewitt_x)
    prewitt_gy = cv2.filter2D(noisy_img.astype(np.float32), -1, prewitt_y)
    prewitt_mag = np.sqrt(prewitt_gx ** 2 + prewitt_gy ** 2)
    operators.append(("Prewitt", prewitt_mag))

    # 3. Roberts
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    roberts_gx = cv2.filter2D(noisy_img.astype(np.float32), -1, roberts_x)
    roberts_gy = cv2.filter2D(noisy_img.astype(np.float32), -1, roberts_y)
    roberts_mag = np.sqrt(roberts_gx ** 2 + roberts_gy ** 2)
    operators.append(("Roberts", roberts_mag))

    # 4. Scharr (改进的Sobel)
    scharr_x = cv2.Scharr(noisy_img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(noisy_img, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(scharr_x ** 2 + scharr_y ** 2)
    operators.append(("Scharr", scharr_mag))

    # 可视化对比
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))  # 改为3x3

    # 原始和噪声图片
    axes[0, 0].imshow(test_img, cmap='gray')
    axes[0, 0].set_title("原始图片")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_img, cmap='gray')
    axes[0, 1].set_title("加噪图片")
    axes[0, 1].axis('off')

    # 显示不同算子的结果
    # 将4个算子放在特定的位置
    positions = [(0, 2), (1, 0), (1, 1), (1, 2)]  # 定义位置

    for idx, ((name, result), (row, col)) in enumerate(zip(operators, positions)):
        axes[row, col].imshow(result, cmap='hot')
        axes[row, col].set_title(f"{name}算子")
        axes[row, col].axis('off')

    # 性能对比
    times = []
    names = []

    for name, _ in operators:
        start_time = time.time()
        for _ in range(100):  # 重复100次
            if name == "Sobel":
                cv2.Sobel(noisy_img, cv2.CV_64F, 1, 0, ksize=3)
                cv2.Sobel(noisy_img, cv2.CV_64F, 0, 1, ksize=3)
            elif name == "Scharr":
                cv2.Scharr(noisy_img, cv2.CV_64F, 1, 0)
                cv2.Scharr(noisy_img, cv2.CV_64F, 0, 1)
        end_time = time.time()

        avg_time = (end_time - start_time) / 100
        times.append(avg_time)
        names.append(name)

    # 将性能对比图放在(2,0)位置
    axes[2, 0].bar(names, times, color=['blue', 'green', 'orange', 'red'])
    axes[2, 0].set_title("计算时间对比")
    axes[2, 0].set_ylabel("时间 (秒)")
    axes[2, 0].grid(True, alpha=0.3, axis='y')

    # 在柱状图上显示数值
    for i, v in enumerate(times):
        axes[2, 0].text(i, v + 0.0001, f'{v:.6f}s',
                        ha='center', va='bottom', fontweight='bold')

    # 算子特性对比说明
    axes[2, 1].axis('off')
    axes[2, 1].text(0.1, 0.5,
                    "算子特性对比:\n\n"
                    "Sobel算子:\n"
                    "  - 优点: 计算效率高\n"
                    "  - 缺点: 对噪声较敏感\n\n"
                    "Prewitt算子:\n"
                    "  - 优点: 计算简单\n"
                    "  - 缺点: 抗噪能力弱\n\n"
                    "Roberts算子:\n"
                    "  - 优点: 计算量最小\n"
                    "  - 缺点: 对噪声非常敏感",
                    fontsize=9, verticalalignment='center')

    axes[2, 2].axis('off')
    axes[2, 2].text(0.1, 0.5,
                    "Scharr算子:\n"
                    "  - 优点: 旋转对称性更好\n"
                    "  - 缺点: 计算量稍大\n\n"
                    "选择建议:\n"
                    "• 实时处理: Sobel\n"
                    "• 无噪声: Roberts\n"
                    "• 高精度: Scharr\n"
                    "• 简单应用: Prewitt",
                    fontsize=9, verticalalignment='center')

    plt.suptitle("Sobel算子与其他算子对比", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 算子特性对比文本
    print("算子特性对比:")
    print("=" * 40)
    print("Sobel算子:")
    print("  - 优点: 计算效率高，有一定抗噪能力")
    print("  - 缺点: 对噪声仍敏感，边缘较粗")
    print("  - 适用: 实时处理，一般精度要求")
    print()

    print("Prewitt算子:")
    print("  - 优点: 计算简单，实现容易")
    print("  - 缺点: 抗噪能力弱于Sobel")
    print("  - 适用: 简单应用，无噪声环境")
    print()

    print("Roberts算子:")
    print("  - 优点: 计算量最小，定位精度高")
    print("  - 缺点: 对噪声非常敏感")
    print("  - 适用: 无噪声图片，实时性要求极高")
    print()

    print("Scharr算子:")
    print("  - 优点: 旋转对称性更好，精度更高")
    print("  - 缺点: 计算量稍大")
    print("  - 适用: 高精度要求，不介意计算成本")
    print()

    return operators, times


# 比较Sobel与其他算子
comparison_results = compare_sobel_with_others()

# ==================== 7. 练习与挑战 ====================
print("\n💪 7. 练习与挑战")
print("=" * 30)

print("""
练习题：

1. 基础练习：
   a) 实现手动Sobel算子，支持不同核大小
   b) 实现梯度方向直方图统计
   c) 实现基于Sobel的边缘增强

2. 进阶练习：
   a) 实现自适应Sobel阈值
   b) 实现多尺度Sobel边缘检测
   c) 实现Sobel算子的GPU加速版本

3. 思考题：
   a) Sobel算子的卷积核为什么这样设计？
   b) 如何选择Sobel算子的阈值？
   c) Sobel算子的优缺点是什么？
   d) 在什么情况下应该使用Sobel算子？
""")

# 练习框架代码
print("\n💻 练习框架代码：")

print("""
# 练习1a: 支持不同核大小的Sobel算子
def adaptive_sobel(image, ksize=3):
    if ksize not in [1, 3, 5, 7]:
        raise ValueError("ksize必须是1, 3, 5, 7中的一个")

    if ksize == 1:
        # 1x1核，实际上就是原始图片
        grad_x = image.copy()
        grad_y = image.copy()
    else:
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)

    return magnitude, direction

# 练习1b: 梯度方向直方图
def gradient_orientation_histogram(gradient_dir, num_bins=8):
    # 将方向从[-π, π]转换到[0, 2π]
    dir_positive = gradient_dir + np.pi

    # 量化为num_bins个方向
    bin_size = 2 * np.pi / num_bins
    quantized = (dir_positive / bin_size).astype(int) % num_bins

    # 计算直方图
    hist, _ = np.histogram(quantized, bins=num_bins, range=(0, num_bins))

    return hist

# 练习1c: 边缘增强
def edge_enhancement_sobel(image, alpha=0.3):
    # 计算Sobel梯度幅值
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # 归一化
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # 边缘增强
    enhanced = cv2.addWeighted(image, 1-alpha, magnitude_norm.astype(np.uint8), alpha, 0)

    return enhanced

# 练习2a: 自适应阈值
def adaptive_sobel_threshold(image, ksize=3, method='mean'):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    if method == 'mean':
        threshold = np.mean(magnitude)
    elif method == 'median':
        threshold = np.median(magnitude)
    elif method == 'otsu':
        # 使用Otsu方法
        magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(magnitude_norm.astype(np.uint8), 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    else:
        threshold = np.mean(magnitude) + np.std(magnitude)

    binary = (magnitude > threshold).astype(np.uint8) * 255
    return binary
""")

# ==================== 8. 总结 ====================
print("\n" + "=" * 50)
print("✅ Sobel算子总结")
print("=" * 50)

summary = """
📊 Sobel算子核心知识：

1. 数学原理
   - 卷积核: Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
             Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]
   - 梯度计算: G = √(Gx² + Gy²)
   - 方向计算: θ = atan2(Gy, Gx)

2. 实现方法
   - OpenCV: cv2.Sobel(src, ddepth, dx, dy, ksize)
   - 手动实现: 卷积计算，边界处理
   - 快速实现: 使用filter2D加速

3. 参数选择
   - ksize: 1,3,5,7 (常用3)
   - ddepth: CV_8U, CV_16S, CV_32F, CV_64F
   - dx/dy: 导数阶数 (1为一阶，2为二阶)
   - scale: 缩放因子
   - delta: 偏移量

4. 性能特点
   - 时间复杂度: O(N²k²), N为图像尺寸，k为核大小
   - 空间复杂度: O(N²)
   - 计算效率: 高，适合实时处理
   - 内存需求: 低

5. 优点
   - 计算简单快速
   - 有一定抗噪能力
   - 能提供方向信息
   - 边缘定位较好
   - 实现简单

6. 缺点
   - 对噪声仍敏感
   - 边缘较粗
   - 对复杂纹理效果一般
   - 需要手动设置阈值

7. 实际应用
   - 实时边缘检测
   - 图像特征提取
   - 计算机视觉预处理
   - 方向估计
   - 边缘增强

8. 最佳实践
   - 预处理: 先高斯滤波去噪
   - 阈值选择: 使用自适应阈值
   - 核大小: 噪声大时用5x5，一般用3x3
   - 深度: 需要负值时用CV_64F
   - 后处理: 非极大值抑制细化边缘

🎯 核心代码记忆：
   # OpenCV Sobel基本用法
   grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
   grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
   magnitude = np.sqrt(grad_x**2 + grad_y**2)
   direction = np.arctan2(grad_y, grad_x)

   # 手动Sobel卷积核
   sobel_x = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
   sobel_y = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]])
"""

print(summary)
print("\n📁 下一个文件: 06_03_Laplacian算子实现.py")
print("  我们将学习二阶微分算子：Laplacian算子！")