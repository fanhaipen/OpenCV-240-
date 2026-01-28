"""
第5天 - 文件3：高斯滤波实现（修复版）
修复了子图索引错误
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time

print("🌈 第5天 - 文件3：高斯滤波实现（修复版）")
print("=" * 50)

# ==================== 1. 高斯滤波理论 ====================
print("\n🎯 1. 高斯滤波理论")
print("=" * 30)

print("""
高斯滤波 (Gaussian Filter)：

数学原理：
  使用高斯函数作为权重函数，对图像进行加权平均

一维高斯函数：
  G(x) = (1/√(2πσ²)) × exp(-x²/(2σ²))

二维高斯函数：
  G(x,y) = (1/(2πσ²)) × exp(-(x²+y²)/(2σ²))

卷积核：
  权重由高斯函数计算，中心权重最大，向四周递减

特点：
  1. 线性滤波
  2. 可分离性：可分解为水平+垂直滤波
  3. 旋转对称性
  4. 单峰性：权重从中心向四周单调递减
  5. 傅里叶变换后仍是高斯函数

优势（相比均值滤波）：
  - 更好的边缘保持能力
  - 更自然的模糊效果
  - 可调节的平滑程度
  - 频域特性更好
""")

# ==================== 2. 高斯函数可视化 ====================
print("\n📊 2. 高斯函数可视化")
print("=" * 30)


def visualize_gaussian_function():
    """可视化高斯函数"""

    # 创建坐标网格
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    # 不同标准差的高斯函数
    sigmas = [0.5, 1.0, 1.5, 2.0]

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    for i, sigma in enumerate(sigmas):
        # 计算二维高斯函数
        Z = (1 / (2 * math.pi * sigma ** 2)) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

        # 2D等高线图
        ax = axes[0, i]
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
        ax.set_title(f'2D高斯函数 σ={sigma}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(contour, ax=ax)

        # 1D截面
        ax = axes[1, i]
        z_1d = (1 / (math.sqrt(2 * math.pi) * sigma)) * np.exp(-x ** 2 / (2 * sigma ** 2))
        ax.plot(x, z_1d, 'r-', linewidth=2)
        ax.set_title(f'1D高斯函数 σ={sigma}')
        ax.set_xlabel('x')
        ax.set_ylabel('G(x)')
        ax.grid(True, alpha=0.3)

        # 显示函数值
        print(f"σ={sigma}: G(0)={z_1d[50]:.4f}, 半高宽: {sigma * 2.355:.2f}")

    plt.suptitle("高斯函数可视化（不同标准差）", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return x, z_1d


# 可视化高斯函数
x_coords, gaussian_1d = visualize_gaussian_function()

# ==================== 3. 高斯核生成 ====================
print("\n🔧 3. 高斯核生成")
print("=" * 30)


def generate_gaussian_kernel(size, sigma):
    """
    生成高斯核

    参数:
        size: 核大小（奇数）
        sigma: 标准差

    返回:
        高斯核矩阵
    """
    if size % 2 == 0:
        raise ValueError("核大小必须是奇数")

    # 创建坐标网格
    k = size // 2
    x = np.arange(-k, k + 1)
    y = np.arange(-k, k + 1)
    X, Y = np.meshgrid(x, y)

    # 计算高斯函数值
    kernel = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

    # 归一化，使和为1
    kernel = kernel / np.sum(kernel)

    return kernel


def demonstrate_gaussian_kernels():
    """演示不同参数的高斯核"""

    # 测试不同参数组合
    param_combinations = [
        (3, 0.5), (3, 1.0),
        (5, 0.8), (5, 1.5),
        (7, 1.0), (7, 2.0),
        (9, 1.5), (9, 3.0)
    ]

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    print("高斯核参数分析:")
    print("-" * 40)

    for idx, (size, sigma) in enumerate(param_combinations):
        kernel = generate_gaussian_kernel(size, sigma)

        row = idx // 4
        col = idx % 4

        # 显示核矩阵
        im = axes[row, col].imshow(kernel, cmap='hot')
        axes[row, col].set_title(f'Size: {size}×{size}, σ={sigma}')
        axes[row, col].set_xticks(range(size))
        axes[row, col].set_yticks(range(size))

        # 在图中显示数值
        for i in range(size):
            for j in range(size):
                axes[row, col].text(j, i, f'{kernel[i, j]:.3f}',
                                    ha='center', va='center',
                                    color='white' if kernel[i, j] > np.max(kernel) / 2 else 'black',
                                    fontsize=8)

        # 打印核信息
        print(f"核 {size}×{size}, σ={sigma}: ")
        print(f"  中心权重: {kernel[size // 2, size // 2]:.4f}")
        print(f"  总和: {np.sum(kernel):.6f}")
        print(f"  有效半径: {sigma * 3:.1f}像素")

    plt.suptitle("不同参数的高斯核", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return param_combinations


# 演示高斯核
kernel_params = demonstrate_gaussian_kernels()

# ==================== 4. 手动实现高斯滤波 ====================
print("\n🔧 4. 手动实现高斯滤波")
print("=" * 30)


def manual_gaussian_filter(image, sigma=1.0, kernel_size=None):
    """
    手动实现高斯滤波

    参数:
        image: 输入图片
        sigma: 标准差
        kernel_size: 核大小（自动计算如果为None）

    返回:
        滤波后的图片
    """
    # 自动确定核大小（3σ原则）
    if kernel_size is None:
        kernel_size = int(6 * sigma + 1)
        # 确保为奇数
        if kernel_size % 2 == 0:
            kernel_size += 1

    if kernel_size % 2 == 0:
        raise ValueError("核大小必须是奇数")

    # 生成高斯核
    kernel = generate_gaussian_kernel(kernel_size, sigma)

    height, width = image.shape
    pad = kernel_size // 2

    # 边界填充（反射填充）
    padded = np.pad(image, pad, mode='reflect')

    # 创建输出图片
    filtered = np.zeros_like(image, dtype=np.float32)

    # 应用卷积
    for i in range(pad, height + pad):
        for j in range(pad, width + pad):
            # 提取局部区域
            region = padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
            # 加权平均
            filtered[i - pad, j - pad] = np.sum(region * kernel)

    return np.clip(filtered, 0, 255).astype(np.uint8)


# 创建测试图片
def create_test_image_for_gaussian():
    """创建用于高斯滤波测试的图片"""
    height, width = 200, 300
    img = np.zeros((height, width), dtype=np.uint8)

    # 梯度背景
    for i in range(height):
        img[i, :] = int(50 + 150 * i / height)

    # 添加测试图案
    # 锐利边缘
    cv2.rectangle(img, (30, 30), (120, 80), 200, -1)
    cv2.rectangle(img, (180, 30), (270, 80), 50, -1)

    # 精细细节
    for i in range(5):
        y = 100 + i * 15
        cv2.line(img, (50, y), (250, y), 150, 1)

    # 点图案
    for i in range(3):
        for j in range(5):
            x = 60 + j * 40
            y = 150 + i * 20
            cv2.circle(img, (x, y), 3, 255, -1)

    cv2.putText(img, "GAUSSIAN FILTER", (70, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

    return img


# 添加高斯噪声的函数
def add_gaussian_noise(image, mean=0, std=25):
    """添加高斯噪声"""
    noise = np.random.normal(mean, std, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# 创建测试图片并添加噪声
test_img = create_test_image_for_gaussian()
noisy_img_gaussian = add_gaussian_noise(test_img, std=25)

print("测试高斯滤波...")
print(f"图片尺寸: {test_img.shape[1]}x{test_img.shape[0]}")
print(f"噪声水平: σ=25")

# 测试不同σ值的高斯滤波
sigmas = [0.5, 1.0, 1.5, 2.0, 3.0]
results_manual = []
computation_times = []

for sigma in sigmas:
    print(f"\n测试 σ={sigma}:")

    start_time = time.time()
    filtered = manual_gaussian_filter(noisy_img_gaussian, sigma)
    end_time = time.time()

    results_manual.append((sigma, filtered))
    computation_times.append(end_time - start_time)

    # 计算噪声减少效果
    original_noise = np.std(noisy_img_gaussian.astype(np.float32) - test_img.astype(np.float32))
    current_noise = np.std(filtered.astype(np.float32) - test_img.astype(np.float32))
    reduction = 100 * (original_noise - current_noise) / original_noise

    print(f"  计算时间: {computation_times[-1]:.4f}秒")
    print(f"  噪声减少: {reduction:.1f}%")
    print(f"  自动核大小: {int(6 * sigma + 1)}×{int(6 * sigma + 1)}")

# 显示手动实现结果 - 修复版
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# 原始和噪声图片
axes[0, 0].imshow(test_img, cmap='gray')
axes[0, 0].set_title("原始图片")
axes[0, 0].axis('off')

axes[0, 1].imshow(noisy_img_gaussian, cmap='gray')
axes[0, 1].set_title("高斯噪声图片")
axes[0, 1].axis('off')

# 显示不同σ的滤波结果 - 修复索引
positions = [(0, 2), (1, 0), (1, 1), (1, 2)]  # 定义4个滤波结果的位置
for i, ((sigma, img), (row, col)) in enumerate(zip(results_manual[:4], positions)):
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(f"手动高斯滤波 σ={sigma}")
    axes[row, col].axis('off')

# 删除多余的子图（如果有）
if len(results_manual) < 4:
    fig.delaxes(axes[1, 2])

plt.suptitle("手动高斯滤波实现（不同σ值）", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 5. OpenCV高斯滤波实现 ====================
print("\n🔧 5. OpenCV高斯滤波实现")
print("=" * 30)


def demonstrate_opencv_gaussian():
    """演示OpenCV高斯滤波"""

    print("OpenCV高斯滤波函数: cv2.GaussianBlur()")
    print("参数: src, ksize, sigmaX, sigmaY=0, borderType=BORDER_DEFAULT")
    print()

    # 测试不同参数
    test_cases = [
        ((5, 5), 1.0, "小核细平滑"),
        ((9, 9), 1.5, "中核中平滑"),
        ((15, 15), 2.0, "大核强平滑"),
        ((0, 0), 1.5, "自动核大小"),
        ((9, 9), 0.5, "小σ锐利"),
        ((9, 9), 3.0, "大σ模糊")
    ]

    results_opencv = []
    opencv_times = []

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for idx, (ksize, sigma, description) in enumerate(test_cases):
        start_time = time.time()

        if ksize == (0, 0):
            # 自动计算核大小
            filtered = cv2.GaussianBlur(noisy_img_gaussian, ksize, sigmaX=sigma)
            actual_ksize = int(6 * sigma + 1)
            if actual_ksize % 2 == 0:
                actual_ksize += 1
            ksize_display = f"auto({actual_ksize})"
        else:
            filtered = cv2.GaussianBlur(noisy_img_gaussian, ksize, sigmaX=sigma)
            ksize_display = f"{ksize[0]}×{ksize[1]}"

        end_time = time.time()

        results_opencv.append((ksize_display, sigma, description, filtered))
        opencv_times.append(end_time - start_time)

        # 计算效果指标
        original_noise = np.std(noisy_img_gaussian.astype(np.float32) - test_img.astype(np.float32))
        current_noise = np.std(filtered.astype(np.float32) - test_img.astype(np.float32))
        reduction = 100 * (original_noise - current_noise) / original_noise

        print(f"测试 {description}:")
        print(f"  核大小: {ksize_display}, σ={sigma}")
        print(f"  计算时间: {opencv_times[-1]:.4f}秒")
        print(f"  噪声减少: {reduction:.1f}%")

        # 显示结果
        row = idx // 3
        col = idx % 3
        axes[row, col].imshow(filtered, cmap='gray')
        axes[row, col].set_title(f"{description}\n核{ksize_display}, σ={sigma}")
        axes[row, col].axis('off')

    plt.suptitle("OpenCV高斯滤波不同参数效果", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return results_opencv, opencv_times


# 演示OpenCV实现
opencv_results, opencv_times = demonstrate_opencv_gaussian()

# ==================== 6. 高斯滤波 vs 均值滤波对比 ====================
print("\n🔍 6. 高斯滤波 vs 均值滤波对比")
print("=" * 30)


def compare_gaussian_vs_mean():
    """对比高斯滤波和均值滤波"""

    print("高斯滤波 vs 均值滤波对比分析:")
    print("=" * 50)

    # 测试条件
    kernel_size = 7
    sigma = 1.0  # 对应的高斯σ

    # 1. 均值滤波
    start_time = time.time()
    mean_filtered = cv2.blur(noisy_img_gaussian, (kernel_size, kernel_size))
    mean_time = time.time() - start_time

    # 2. 高斯滤波
    start_time = time.time()
    gaussian_filtered = cv2.GaussianBlur(noisy_img_gaussian, (kernel_size, kernel_size), sigma)
    gaussian_time = time.time() - start_time

    # 3. 计算性能指标
    original_noise = np.std(noisy_img_gaussian.astype(np.float32) - test_img.astype(np.float32))
    mean_noise = np.std(mean_filtered.astype(np.float32) - test_img.astype(np.float32))
    gaussian_noise = np.std(gaussian_filtered.astype(np.float32) - test_img.astype(np.float32))

    mean_reduction = 100 * (original_noise - mean_noise) / original_noise
    gaussian_reduction = 100 * (original_noise - gaussian_noise) / original_noise

    # 边缘保持度评估（简化：使用梯度幅值）
    gradient_original = np.mean(np.abs(cv2.Sobel(test_img, cv2.CV_64F, 1, 1)))
    gradient_mean = np.mean(np.abs(cv2.Sobel(mean_filtered, cv2.CV_64F, 1, 1)))
    gradient_gaussian = np.mean(np.abs(cv2.Sobel(gaussian_filtered, cv2.CV_64F, 1, 1)))

    edge_preservation_mean = gradient_mean / gradient_original
    edge_preservation_gaussian = gradient_gaussian / gradient_original

    print(f"核大小: {kernel_size}×{kernel_size}")
    print(f"高斯σ: {sigma}")
    print()
    print("性能对比:")
    print(f"  均值滤波 - 时间: {mean_time:.4f}s, 噪声减少: {mean_reduction:.1f}%")
    print(f"  高斯滤波 - 时间: {gaussian_time:.4f}s, 噪声减少: {gaussian_reduction:.1f}%")
    print()
    print("边缘保持度 (越高越好):")
    print(f"  均值滤波: {edge_preservation_mean:.3f}")
    print(f"  高斯滤波: {edge_preservation_gaussian:.3f}")
    print(f"  高斯优势: {edge_preservation_gaussian / edge_preservation_mean:.1f}倍")

    # 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 第一行
    axes[0, 0].imshow(test_img, cmap='gray')
    axes[0, 0].set_title("原始图片")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_img_gaussian, cmap='gray')
    axes[0, 1].set_title("噪声图片")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(mean_filtered, cmap='gray')
    axes[0, 2].set_title(f"均值滤波 {kernel_size}×{kernel_size}")
    axes[0, 2].axis('off')

    # 第二行
    axes[1, 0].imshow(gaussian_filtered, cmap='gray')
    axes[1, 0].set_title(f"高斯滤波 {kernel_size}×{kernel_size}, σ={sigma}")
    axes[1, 0].axis('off')

    # 局部放大对比
    mean_local = mean_filtered[80:120, 100:140]
    gaussian_local = gaussian_filtered[80:120, 100:140]

    axes[1, 1].imshow(mean_local, cmap='gray')
    axes[1, 1].set_title("均值滤波局部")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(gaussian_local, cmap='gray')
    axes[1, 2].set_title("高斯滤波局部")
    axes[1, 2].axis('off')

    plt.suptitle("高斯滤波 vs 均值滤波对比", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return (mean_filtered, gaussian_filtered, mean_time, gaussian_time,
            mean_reduction, gaussian_reduction, edge_preservation_mean, edge_preservation_gaussian)


# 对比高斯滤波和均值滤波
comparison_results = compare_gaussian_vs_mean()

# ==================== 7. 高斯滤波的可分离性 ====================
print("\n⚡ 7. 高斯滤波的可分离性")
print("=" * 30)


def demonstrate_separability():
    """演示高斯滤波的可分离性"""

    print("高斯滤波的可分离性:")
    print("二维卷积 = 水平卷积 × 垂直卷积")
    print("这使计算复杂度从O(N²M²)降到O(2NM²)")
    print()

    # 创建测试图片
    test_pattern = np.zeros((100, 100), dtype=np.uint8)
    test_pattern[20:80, 20:80] = 255

    sigma = 2.0
    kernel_size = 9

    # 1. 标准二维卷积
    start_time = time.time()
    standard_result = cv2.GaussianBlur(test_pattern, (kernel_size, kernel_size), sigma)
    standard_time = time.time() - start_time

    # 2. 可分离卷积
    start_time = time.time()

    # 生成1D高斯核
    k = kernel_size // 2
    x = np.arange(-k, k + 1)
    kernel_1d = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / np.sum(kernel_1d)

    # 水平滤波
    horizontal = cv2.filter2D(test_pattern, -1, kernel_1d.reshape(1, -1))
    # 垂直滤波
    separable_result = cv2.filter2D(horizontal, -1, kernel_1d.reshape(-1, 1))
    separable_time = time.time() - start_time

    # 3. 验证结果一致性
    diff = np.max(np.abs(standard_result.astype(np.float32) - separable_result.astype(np.float32)))

    print(f"σ={sigma}, 核大小: {kernel_size}×{kernel_size}")
    print(f"标准卷积时间: {standard_time:.6f}秒")
    print(f"可分离卷积时间: {separable_time:.6f}秒")
    print(f"加速比: {standard_time / separable_time:.2f}倍")
    print(f"结果差异: {diff:.6f} (应该接近0)")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 第一行
    axes[0, 0].imshow(test_pattern, cmap='gray')
    axes[0, 0].set_title("原始图片")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(standard_result, cmap='gray')
    axes[0, 1].set_title("标准二维卷积")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(separable_result, cmap='gray')
    axes[0, 2].set_title("可分离卷积")
    axes[0, 2].axis('off')

    # 第二行：显示1D核和计算过程
    axes[1, 0].plot(x, kernel_1d, 'ro-', linewidth=2, markersize=4)
    axes[1, 0].set_title("1D高斯核")
    axes[1, 0].set_xlabel('位置')
    axes[1, 0].set_ylabel('权重')
    axes[1, 0].grid(True, alpha=0.3)

    # 显示中间结果（水平滤波）
    axes[1, 1].imshow(horizontal, cmap='gray')
    axes[1, 1].set_title("水平滤波结果")
    axes[1, 1].axis('off')

    # 显示性能对比
    axes[1, 2].bar(['标准卷积', '可分离卷积'], [standard_time, separable_time],
                   color=['blue', 'orange'])
    axes[1, 2].set_title("计算时间对比")
    axes[1, 2].set_ylabel('时间 (秒)')
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.suptitle("高斯滤波的可分离性演示", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return standard_result, separable_result, standard_time, separable_time


# 演示可分离性
standard_result, separable_result, standard_time, separable_time = demonstrate_separability()

# ==================== 8. 实际应用案例 ====================
print("\n💼 8. 实际应用案例")
print("=" * 30)


def demonstrate_real_world_applications():
    """演示高斯滤波在实际中的应用"""

    print("高斯滤波的实际应用场景:")
    print("1. 图像预处理: 为特征提取减少噪声")
    print("2. 人像美化: 皮肤平滑处理")
    print("3. 图像金字塔构建: 多尺度分析")
    print("4. 边缘检测预处理: 减少噪声干扰")
    print("5. 艺术效果: 创建柔和模糊")
    print()

    # 模拟不同应用场景
    applications = [
        ("图像预处理", "preprocess", (5, 5), 1.0),
        ("皮肤平滑", "skin_smoothing", (9, 9), 1.5),
        ("艺术模糊", "artistic_blur", (15, 15), 2.5),
        ("边缘检测预处理", "edge_preprocess", (3, 3), 0.8)
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for idx, (app_name, app_type, ksize, sigma) in enumerate(applications):
        row = idx // 2
        col = (idx % 2) * 2

        if app_type == "preprocess":
            # 图像预处理
            original = noisy_img_gaussian.copy()
            processed = cv2.GaussianBlur(original, ksize, sigma)

            axes[row, col].imshow(original, cmap='gray')
            axes[row, col].set_title(f"{app_name}\n原始")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(processed, cmap='gray')
            axes[row, col + 1].set_title(f"{app_name}\n高斯滤波后")
            axes[row, col + 1].axis('off')

        elif app_type == "skin_smoothing":
            # 模拟皮肤平滑
            skin_img = np.random.randint(150, 200, (100, 100), dtype=np.uint8)
            # 添加一些纹理
            for i in range(0, 100, 5):
                cv2.line(skin_img, (0, i), (100, i), 170, 1)

            # 添加模拟毛孔
            for _ in range(20):
                x, y = np.random.randint(0, 100, 2)
                cv2.circle(skin_img, (x, y), 1, 180, -1)

            smoothed = cv2.GaussianBlur(skin_img, ksize, sigma)

            axes[row, col].imshow(skin_img, cmap='gray',vmin=0,vmax=255)
            axes[row, col].set_title(f"{app_name}\n原始皮肤")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(smoothed, cmap='gray',vmin=0,vmax=255)
            axes[row, col + 1].set_title(f"{app_name}\n平滑后")
            axes[row, col + 1].axis('off')

        elif app_type == "artistic_blur":
            # 艺术模糊
            original = test_img.copy()
            blurred = cv2.GaussianBlur(original, ksize, sigma)

            axes[row, col].imshow(original, cmap='gray',vmin=0,vmax=255)
            axes[row, col].set_title(f"{app_name}\n原始")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(blurred, cmap='gray',vmin=0,vmax=255)
            axes[row, col + 1].set_title(f"{app_name}\n艺术模糊")
            axes[row, col + 1].axis('off')

        elif app_type == "edge_preprocess":
            # 边缘检测预处理
            original = noisy_img_gaussian.copy()
            preprocessed = cv2.GaussianBlur(original, ksize, sigma)

            # 边缘检测对比
            edges_original = cv2.Canny(original, 50, 150)
            edges_processed = cv2.Canny(preprocessed, 50, 150)

            axes[row, col].imshow(edges_original, cmap='gray',vmin=0,vmax=255)
            axes[row, col].set_title(f"{app_name}\n直接边缘检测")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(edges_processed, cmap='gray',vmin=0,vmax=255)
            axes[row, col + 1].set_title(f"{app_name}\n滤波后边缘检测")
            axes[row, col + 1].axis('off')

    plt.suptitle("高斯滤波在实际场景中的应用", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    # 应用建议
    print("\n高斯滤波参数选择指南:")
    print("-" * 40)
    print("1. 预处理: σ=0.8-1.5, 小核 (3×3, 5×5)")
    print("2. 去噪: σ=1.0-2.0, 中核 (5×5, 7×7)")
    print("3. 模糊效果: σ=2.0-4.0, 大核 (9×9, 15×15)")
    print("4. 实时处理: 使用可分离实现")
    print("5. 注意: σ与核大小匹配 (核大小 ≈ 6σ+1)")


# 演示实际应用
demonstrate_real_world_applications()

# ==================== 9. 练习与挑战 ====================
print("\n💪 9. 练习与挑战")
print("=" * 30)

print("""
练习题：

1. 基础练习：
   a) 实现函数，生成不同σ值的高斯核
   b) 对比不同σ值对滤波效果的影响
   c) 实现可分离高斯滤波

2. 进阶练习：
   a) 实现自适应高斯滤波，根据局部纹理调整σ
   b) 比较高斯滤波与双边滤波的效果
   c) 实现多尺度高斯滤波（高斯金字塔）

3. 思考题：
   a) 为什么高斯滤波比均值滤波更好地保持边缘？
   b) 如何选择最优的σ值？
   c) 高斯滤波的可分离性有什么实际意义？
   d) 高斯滤波在频域中有什么特性？
""")

# 练习框架代码
print("\n💻 练习框架代码：")

print("""
# 练习1a: 生成不同σ值的高斯核
def generate_gaussian_kernels(sigmas=[0.5, 1.0, 1.5, 2.0], size=7):
    kernels = {}
    for sigma in sigmas:
        kernel = generate_gaussian_kernel(size, sigma)
        kernels[sigma] = kernel
    return kernels

# 练习1c: 可分离高斯滤波实现
def separable_gaussian_blur(image, sigma=1.0):
    # 生成1D高斯核
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    kernel_1d = cv2.getGaussianKernel(ksize, sigma)

    # 水平滤波
    horizontal = cv2.filter2D(image, -1, kernel_1d.T)
    # 垂直滤波
    result = cv2.filter2D(horizontal, -1, kernel_1d)

    return result

# 练习2a: 自适应高斯滤波框架
def adaptive_gaussian_filter(image, base_sigma=1.0, max_sigma=3.0):
    height, width = image.shape
    filtered = np.zeros_like(image, dtype=np.float32)

    for i in range(height):
        for j in range(width):
            # 计算局部纹理复杂度
            local_region = image[max(0, i-2):min(height, i+3), 
                                 max(0, j-2):min(width, j+3)]
            local_variance = np.var(local_region)

            # 根据局部方差调整σ
            if local_variance > 1000:  # 高纹理区域
                sigma = base_sigma * 0.5
            elif local_variance < 100:  # 平滑区域
                sigma = base_sigma * 2.0
            else:  # 中等纹理
                sigma = base_sigma

            sigma = min(max(sigma, 0.5), max_sigma)

            # 应用局部高斯滤波
            ksize = int(6 * sigma + 1)
            if ksize % 2 == 0:
                ksize += 1

            pad = ksize // 2
            region = image[max(0, i-pad):min(height, i+pad+1), 
                          max(0, j-pad):min(width, j+pad+1)]

            if region.size > 0:
                kernel = generate_gaussian_kernel(region.shape[0], sigma)
                filtered[i, j] = np.sum(region * kernel)

    return filtered.astype(np.uint8)
""")

# ==================== 10. 总结 ====================
print("\n" + "=" * 50)
print("✅ 高斯滤波总结")
print("=" * 50)

summary = """
📊 高斯滤波核心知识：

1. 数学原理
   - 高斯函数: G(x) = (1/√(2πσ²)) × exp(-x²/(2σ²))
   - 权重: 中心大，四周小，按高斯分布递减
   - 归一化: 核元素和为1

2. 参数选择
   - σ (标准差): 控制平滑程度
   - 核大小: 通常为6σ+1（奇数）
   - σ小: 细节保持好，去噪弱
   - σ大: 平滑效果好，会模糊

3. 实现方法
   - OpenCV: cv2.GaussianBlur()
   - 手动实现: 生成高斯核，卷积计算
   - 可分离实现: 水平+垂直滤波，速度快

4. 性能特点
   - 时间复杂度: O(N²M²) 原始，O(2NM²) 可分离
   - 空间复杂度: O(M²) 存储核
   - 线性滤波: 满足叠加性和齐次性
   - 旋转对称: 各向同性

5. 与均值滤波对比
   - 边缘保持: 高斯 >> 均值
   - 平滑自然度: 高斯 > 均值
   - 计算复杂度: 高斯 ≈ 均值（可分离时）
   - 应用范围: 高斯更广泛

6. 实际应用
   - 图像预处理: 去除噪声
   - 特征提取: 减少干扰
   - 艺术效果: 创建模糊
   - 多尺度分析: 构建金字塔

7. 最佳实践
   - 预处理: σ=0.8-1.5, 小核
   - 一般去噪: σ=1.0-2.0, 中核
   - 强平滑: σ=2.0-4.0, 大核
   - 实时处理: 使用可分离实现

🎯 核心代码记忆：
   # OpenCV实现
   blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

   # 手动生成高斯核
   def gaussian_kernel(size, sigma):
       k = size // 2
       x = np.arange(-k, k+1)
       X, Y = np.meshgrid(x, x)
       kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
       return kernel / np.sum(kernel)
"""

print(summary)
print("\n📁 下一个文件: 05_04_中值滤波实现.py")
print("  我们将学习非线性滤波 - 中值滤波！")