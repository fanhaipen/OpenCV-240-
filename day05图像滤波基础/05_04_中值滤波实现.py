"""
第5天 - 文件4：中值滤波实现
学习目标：掌握中值滤波的原理、实现和应用
重点：非线性滤波、排序统计、椒盐噪声去除
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


print("🔢 第5天 - 文件4：中值滤波实现")
print("=" * 50)

# ==================== 1. 中值滤波理论 ====================
print("\n🎯 1. 中值滤波理论")
print("=" * 30)

print("""
中值滤波 (Median Filter)：

数学原理：
  用邻域内像素的中值（排序后的中间值）替换中心像素值

计算步骤：
  1. 提取邻域内所有像素值
  2. 将这些像素值排序
  3. 取排序后的中间值作为输出

公式：
  I'(x,y) = median{ I(x+i, y+j) | i,j ∈ [-k,k] }

其中：
  k: 滤波核半径
  median: 中值运算符

特点：
  1. 非线性滤波
  2. 基于排序统计
  3. 有效去除椒盐噪声
  4. 完全保持边缘不模糊
  5. 计算相对较慢

优势（相比线性滤波）：
  - 完全去除孤立噪声点
  - 边缘保持能力极佳
  - 不产生新的灰度值
  - 适合处理脉冲噪声

局限性：
  - 计算复杂度高
  - 对高斯噪声效果一般
  - 可能丢失细节
  - 窗口大小需为奇数
""")

# ==================== 2. 中值计算原理演示 ====================
print("\n📊 2. 中值计算原理演示")
print("=" * 30)


def demonstrate_median_calculation():
    """演示中值计算原理"""

    # 创建一个3×3的示例像素块
    pixels_3x3 = np.array([
        [10, 20, 30],
        [40, 250, 60],  # 中心像素250是噪声点
        [70, 80, 90]
    ])

    # 创建一个5×5的示例像素块
    pixels_5x5 = np.array([
        [10, 20, 30, 40, 50],
        [60, 70, 80, 90, 100],  # 中心像素0是噪声点
        [110, 120, 0, 140, 150],
        [160, 170, 180, 190, 200],
        [210, 220, 230, 240, 250]
    ])

    print("3×3像素块示例:")
    print(pixels_3x3)
    print(f"原始中心像素值: {pixels_3x3[1, 1]}")

    # 计算3×3中值
    flat_3x3 = pixels_3x3.flatten()
    sorted_3x3 = np.sort(flat_3x3)
    median_3x3 = sorted_3x3[len(sorted_3x3) // 2]

    print(f"展开后的像素值: {flat_3x3}")
    print(f"排序后的像素值: {sorted_3x3}")
    print(f"中值: {median_3x3}")
    print(f"中值索引: {len(sorted_3x3) // 2}")

    print("\n" + "-" * 50)
    print("5×5像素块示例:")
    print(pixels_5x5)
    print(f"原始中心像素值: {pixels_5x5[2, 2]}")

    # 计算5×5中值
    flat_5x5 = pixels_5x5.flatten()
    sorted_5x5 = np.sort(flat_5x5)
    median_5x5 = sorted_5x5[len(sorted_5x5) // 2]

    print(f"展开后的像素值: {flat_5x5}")
    print(f"中值: {median_5x5}")
    print(f"中值索引: {len(sorted_5x5) // 2}")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 3×3示例
    axes[0, 0].imshow(pixels_3x3, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("3×3像素块")
    axes[0, 0].set_xticks(range(3))
    axes[0, 0].set_yticks(range(3))
    axes[0, 0].grid(True, color='red', linewidth=0.5)

    for i in range(3):
        for j in range(3):
            color = 'white' if pixels_3x3[i, j] > 127 else 'black'
            axes[0, 0].text(j, i, str(pixels_3x3[i, j]),
                            ha='center', va='center', color=color)

    # 3×3排序可视化
    axes[0, 1].plot(sorted_3x3, 'bo-', linewidth=2, markersize=6)
    axes[0, 1].set_title("排序后的像素值")
    axes[0, 1].set_xlabel("索引")
    axes[0, 1].set_ylabel("像素值")
    axes[0, 1].grid(True, alpha=0.3)

    # 标记中值
    median_idx = len(sorted_3x3) // 2
    axes[0, 1].plot(median_idx, median_3x3, 'ro', markersize=10, label=f'中值={median_3x3}')
    axes[0, 1].legend()

    # 5×5示例
    axes[0, 2].imshow(pixels_5x5, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title("5×5像素块")
    axes[0, 2].set_xticks(range(5))
    axes[0, 2].set_yticks(range(5))
    axes[0, 2].grid(True, color='red', linewidth=0.5)

    for i in range(5):
        for j in range(5):
            color = 'white' if pixels_5x5[i, j] > 127 else 'black'
            if i == 2 and j == 2:  # 中心像素特殊标记
                axes[0, 2].text(j, i, str(pixels_5x5[i, j]),
                                ha='center', va='center', color='red', fontweight='bold')
            else:
                axes[0, 2].text(j, i, str(pixels_5x5[i, j]),
                                ha='center', va='center', color=color)

    # 5×5排序可视化
    axes[1, 0].plot(sorted_5x5, 'go-', linewidth=2, markersize=4)
    axes[1, 0].set_title("排序后的像素值")
    axes[1, 0].set_xlabel("索引")
    axes[1, 0].set_ylabel("像素值")
    axes[1, 0].grid(True, alpha=0.3)

    # 标记中值
    median_idx = len(sorted_5x5) // 2
    axes[1, 0].plot(median_idx, median_5x5, 'ro', markersize=10, label=f'中值={median_5x5}')
    axes[1, 0].legend()

    # 中值滤波原理说明
    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.5,
                    "中值滤波原理总结:\n\n"
                    "1. 提取邻域像素\n"
                    "2. 排序所有像素值\n"
                    "3. 取中间值作为输出\n\n"
                    "特性:\n"
                    "• 非线性操作\n"
                    "• 完全去除孤立噪声点\n"
                    "• 保持边缘清晰\n"
                    "• 计算复杂度: O(n log n)",
                    fontsize=10, verticalalignment='center')

    # 示例对比
    original_center_3x3 = pixels_3x3[1, 1]
    original_center_5x5 = pixels_5x5[2, 2]

    axes[1, 2].bar(['3×3原始', '3×3中值', '5×5原始', '5×5中值'],
                   [original_center_3x3, median_3x3, original_center_5x5, median_5x5],
                   color=['blue', 'green', 'blue', 'green'])
    axes[1, 2].set_title("中值滤波效果对比")
    axes[1, 2].set_ylabel("像素值")
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.suptitle("中值计算原理演示", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return pixels_3x3, median_3x3, pixels_5x5, median_5x5


# 演示中值计算原理
pixels_3x3, median_3x3, pixels_5x5, median_5x5 = demonstrate_median_calculation()

# ==================== 3. 手动实现中值滤波 ====================
print("\n🔧 3. 手动实现中值滤波")
print("=" * 30)


def manual_median_filter(image, kernel_size=3):
    """
    手动实现中值滤波

    参数:
        image: 输入图片
        kernel_size: 滤波核大小（奇数）

    返回:
        滤波后的图片
    """
    if kernel_size % 2 == 0:
        raise ValueError("核大小必须是奇数")

    height, width = image.shape
    pad = kernel_size // 2

    # 边界填充（反射填充）
    padded = np.pad(image, pad, mode='reflect')

    # 创建输出图片
    filtered = np.zeros_like(image, dtype=np.uint8)

    # 应用中值滤波
    for i in range(pad, height + pad):
        for j in range(pad, width + pad):
            # 提取局部区域
            region = padded[i - pad:i + pad + 1, j - pad:j + pad + 1]

            # 计算中值
            median_val = np.median(region)
            filtered[i - pad, j - pad] = median_val

    return filtered


def manual_median_filter_optimized(image, kernel_size=3):
    """
    优化版手动中值滤波（使用快速选择算法思想）
    注意：这只是示意，实际仍使用numpy的median
    """
    if kernel_size % 2 == 0:
        raise ValueError("核大小必须是奇数")

    height, width = image.shape
    pad = kernel_size // 2

    # 边界填充
    padded = np.pad(image, pad, mode='reflect')

    # 创建输出图片
    filtered = np.zeros_like(image, dtype=np.uint8)

    # 预计算一些值
    kernel_area = kernel_size * kernel_size
    mid_index = kernel_area // 2  # 中值索引

    # 应用中值滤波
    for i in range(pad, height + pad):
        for j in range(pad, width + pad):
            # 提取局部区域并展平
            region = padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
            flat_region = region.flatten()

            # 使用numpy的partition进行部分排序（类似快速选择）
            # 这比完整排序更快
            sorted_partial = np.partition(flat_region, mid_index)
            median_val = sorted_partial[mid_index]

            filtered[i - pad, j - pad] = median_val

    return filtered


# 创建测试图片
def create_test_image_for_median():
    """创建用于中值滤波测试的图片"""
    height, width = 200, 300
    img = np.zeros((height, width), dtype=np.uint8)

    # 梯度背景
    for i in range(height):
        img[i, :] = int(50 + 150 * i / height)

    # 添加锐利边缘
    cv2.rectangle(img, (30, 30), (120, 80), 200, -1)
    cv2.rectangle(img, (180, 30), (270, 80), 50, -1)

    # 添加细线
    for i in range(5):
        y = 100 + i * 15
        cv2.line(img, (50, y), (250, y), 150, 1)

    # 添加小点
    for i in range(3):
        for j in range(5):
            x = 60 + j * 40
            y = 150 + i * 20
            cv2.circle(img, (x, y), 3, 255, -1)

    cv2.putText(img, "MEDIAN FILTER", (70, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

    return img


# 添加椒盐噪声的函数
def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """添加椒盐噪声"""
    noisy = image.copy()
    total_pixels = image.size

    # 盐噪声（白点）
    num_salt = int(total_pixels * salt_prob)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[coords[0], coords[1]] = 255

    # 椒噪声（黑点）
    num_pepper = int(total_pixels * pepper_prob)
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy


# 创建测试图片并添加噪声
test_img_median = create_test_image_for_median()
salt_pepper_img = add_salt_pepper_noise(test_img_median, 0.03, 0.03)

print("测试中值滤波...")
print(f"图片尺寸: {test_img_median.shape[1]}x{test_img_median.shape[0]}")
print(f"噪声类型: 椒盐噪声 (盐: 3%, 椒: 3%)")

# 测试不同核大小的中值滤波
kernel_sizes = [3, 5, 7, 9]
results_manual_median = []
computation_times = []

for ksize in kernel_sizes:
    print(f"\n测试核大小: {ksize}×{ksize}")

    start_time = time.time()
    filtered = manual_median_filter(salt_pepper_img, ksize)
    end_time = time.time()

    results_manual_median.append((ksize, filtered))
    computation_times.append(end_time - start_time)

    # 计算噪声去除效果
    # 统计剩余噪声点（接近0或255的像素）
    remaining_noise = np.sum((filtered == 0) | (filtered == 255)) - np.sum(
        (test_img_median == 0) | (test_img_median == 255))
    original_noise = np.sum((salt_pepper_img == 0) | (salt_pepper_img == 255)) - np.sum(
        (test_img_median == 0) | (test_img_median == 255))

    if original_noise > 0:
        noise_reduction = 100 * (original_noise - remaining_noise) / original_noise
    else:
        noise_reduction = 100

    print(f"  计算时间: {computation_times[-1]:.4f}秒")
    print(f"  原始噪声点: {original_noise}")
    print(f"  剩余噪声点: {remaining_noise}")
    print(f"  噪声去除率: {noise_reduction:.1f}%")

# 显示手动实现结果
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# 原始和噪声图片
axes[0, 0].imshow(test_img_median, cmap='gray')
axes[0, 0].set_title("原始图片")
axes[0, 0].axis('off')

axes[0, 1].imshow(salt_pepper_img, cmap='gray')
axes[0, 1].set_title("椒盐噪声图片 (6%)")
axes[0, 1].axis('off')

# 显示不同核大小的滤波结果
positions = [(0, 2), (1, 0), (1, 1), (1, 2)]
for idx, ((ksize, img), (row, col)) in enumerate(zip(results_manual_median, positions)):
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(f"手动中值滤波 {ksize}×{ksize}")
    axes[row, col].axis('off')

plt.suptitle("手动中值滤波实现（不同核大小）", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 4. OpenCV中值滤波实现 ====================
print("\n🔧 4. OpenCV中值滤波实现")
print("=" * 30)


def demonstrate_opencv_median():
    """演示OpenCV中值滤波"""

    print("OpenCV中值滤波函数: cv2.medianBlur()")
    print("参数: src, ksize (必须是大于1的奇数)")
    print()

    # 测试不同核大小
    kernel_sizes = [3, 5, 7, 9, 15, 21]

    results_opencv = []
    opencv_times = []

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for idx, ksize in enumerate(kernel_sizes):
        start_time = time.time()
        filtered = cv2.medianBlur(salt_pepper_img, ksize)
        end_time = time.time()

        results_opencv.append((ksize, filtered))
        opencv_times.append(end_time - start_time)

        # 计算性能指标
        original_noise = np.sum((salt_pepper_img == 0) | (salt_pepper_img == 255)) - np.sum(
            (test_img_median == 0) | (test_img_median == 255))
        remaining_noise = np.sum((filtered == 0) | (filtered == 255)) - np.sum(
            (test_img_median == 0) | (test_img_median == 255))

        if original_noise > 0:
            noise_reduction = 100 * (original_noise - remaining_noise) / original_noise
        else:
            noise_reduction = 100

        print(f"核大小 {ksize}×{ksize}:")
        print(f"  计算时间: {opencv_times[-1]:.4f}秒")
        print(f"  噪声去除率: {noise_reduction:.1f}%")

        # 显示结果（只显示前6个）
        if idx < 6:
            row = idx // 3
            col = idx % 3
            axes[row, col].imshow(filtered, cmap='gray')
            axes[row, col].set_title(f"OpenCV中值滤波\n{ksize}×{ksize}")
            axes[row, col].axis('off')

    plt.suptitle("OpenCV中值滤波不同核大小效果", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return results_opencv, opencv_times


# 演示OpenCV实现
opencv_results, opencv_times = demonstrate_opencv_median()

# ==================== 5. 中值滤波 vs 线性滤波对比 ====================
print("\n🔍 5. 中值滤波 vs 线性滤波对比")
print("=" * 30)


def compare_median_vs_linear():
    """对比中值滤波和线性滤波"""

    print("中值滤波 vs 线性滤波对比分析:")
    print("=" * 50)

    # 测试条件
    kernel_size = 5

    # 1. 中值滤波
    start_time = time.time()
    median_filtered = cv2.medianBlur(salt_pepper_img, kernel_size)
    median_time = time.time() - start_time

    # 2. 均值滤波
    start_time = time.time()
    mean_filtered = cv2.blur(salt_pepper_img, (kernel_size, kernel_size))
    mean_time = time.time() - start_time

    # 3. 高斯滤波
    start_time = time.time()
    gaussian_filtered = cv2.GaussianBlur(salt_pepper_img, (kernel_size, kernel_size), 1.0)
    gaussian_time = time.time() - start_time

    # 计算噪声去除效果
    original_noise = np.sum((salt_pepper_img == 0) | (salt_pepper_img == 255)) - np.sum(
        (test_img_median == 0) | (test_img_median == 255))

    median_noise = np.sum((median_filtered == 0) | (median_filtered == 255)) - np.sum(
        (test_img_median == 0) | (test_img_median == 255))
    mean_noise = np.sum((mean_filtered == 0) | (mean_filtered == 255)) - np.sum(
        (test_img_median == 0) | (test_img_median == 255))
    gaussian_noise = np.sum((gaussian_filtered == 0) | (gaussian_filtered == 255)) - np.sum(
        (test_img_median == 0) | (test_img_median == 255))

    median_reduction = 100 * (original_noise - median_noise) / original_noise if original_noise > 0 else 100
    mean_reduction = 100 * (original_noise - mean_noise) / original_noise if original_noise > 0 else 100
    gaussian_reduction = 100 * (original_noise - gaussian_noise) / original_noise if original_noise > 0 else 100

    # 修复：边缘保持度评估 - 使用正确的方法
    def calculate_edge_preservation(original, filtered):
        """
        计算边缘保持度
        使用Canny边缘检测，比较边缘像素的保持情况
        """
        # 1. 检测原始图片的边缘
        edges_original = cv2.Canny(original, 50, 150)

        # 2. 检测滤波后图片的边缘
        edges_filtered = cv2.Canny(filtered, 50, 150)

        # 3. 计算重叠的边缘像素
        overlap = np.sum((edges_original > 0) & (edges_filtered > 0))
        total_original_edges = np.sum(edges_original > 0)

        # 避免除以0
        if total_original_edges == 0:
            return 0

        # 4. 计算保持率
        preservation_rate = overlap / total_original_edges

        return preservation_rate

    # 计算边缘保持度
    edge_preservation_median = calculate_edge_preservation(test_img_median, median_filtered)
    edge_preservation_mean = calculate_edge_preservation(test_img_median, mean_filtered)
    edge_preservation_gaussian = calculate_edge_preservation(test_img_median, gaussian_filtered)

    print(f"核大小: {kernel_size}×{kernel_size}")
    print()
    print("性能对比:")
    print(f"  中值滤波 - 时间: {median_time:.4f}s, 噪声去除: {median_reduction:.1f}%")
    print(f"  均值滤波 - 时间: {mean_time:.4f}s, 噪声去除: {mean_reduction:.1f}%")
    print(f"  高斯滤波 - 时间: {gaussian_time:.4f}s, 噪声去除: {gaussian_reduction:.1f}%")
    print()
    print("边缘保持度 (越高越好，范围0-1):")
    print(f"  中值滤波: {edge_preservation_median:.3f}")
    print(f"  均值滤波: {edge_preservation_mean:.3f}")
    print(f"  高斯滤波: {edge_preservation_gaussian:.3f}")

    # 可视化对比
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    # 第一行
    axes[0, 0].imshow(test_img_median, cmap='gray')
    axes[0, 0].set_title("原始图片")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(salt_pepper_img, cmap='gray')
    axes[0, 1].set_title("椒盐噪声图片")
    axes[0, 1].axis('off')

    # 显示局部放大（噪声区域）
    noise_region = salt_pepper_img[80:120, 100:140]
    axes[0, 2].imshow(noise_region, cmap='gray')
    axes[0, 2].set_title("噪声局部放大")
    axes[0, 2].axis('off')

    # 第二行：不同滤波结果
    axes[1, 0].imshow(median_filtered, cmap='gray')
    axes[1, 0].set_title(f"中值滤波 {kernel_size}×{kernel_size}")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mean_filtered, cmap='gray')
    axes[1, 1].set_title(f"均值滤波 {kernel_size}×{kernel_size}")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(gaussian_filtered, cmap='gray')
    axes[1, 2].set_title(f"高斯滤波 {kernel_size}×{kernel_size}")
    axes[1, 2].axis('off')

    # 第三行：边缘检测对比
    edges_original = cv2.Canny(test_img_median, 50, 150)
    edges_median = cv2.Canny(median_filtered, 50, 150)
    edges_mean = cv2.Canny(mean_filtered, 50, 150)

    axes[2, 0].imshow(edges_original, cmap='gray')
    axes[2, 0].set_title("原始边缘")
    axes[2, 0].axis('off')

    axes[2, 1].imshow(edges_median, cmap='gray')
    axes[2, 1].set_title("中值滤波边缘")
    axes[2, 1].axis('off')

    axes[2, 2].imshow(edges_mean, cmap='gray')
    axes[2, 2].set_title("均值滤波边缘")
    axes[2, 2].axis('off')

    plt.suptitle("中值滤波 vs 线性滤波对比（椒盐噪声）", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 性能对比可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 计算时间对比
    times = [median_time, mean_time, gaussian_time]
    axes[0].bar(['中值滤波', '均值滤波', '高斯滤波'], times,
                color=['blue', 'green', 'orange'])
    axes[0].set_title("计算时间对比")
    axes[0].set_ylabel("时间 (秒)")
    axes[0].grid(True, alpha=0.3, axis='y')

    # 噪声去除率对比
    reductions = [median_reduction, mean_reduction, gaussian_reduction]
    axes[1].bar(['中值滤波', '均值滤波', '高斯滤波'], reductions,
                color=['blue', 'green', 'orange'])
    axes[1].set_title("噪声去除率对比")
    axes[1].set_ylabel("去除率 (%)")
    axes[1].set_ylim([0, 100])
    axes[1].grid(True, alpha=0.3, axis='y')

    # 边缘保持度对比
    edge_preservations = [edge_preservation_median, edge_preservation_mean, edge_preservation_gaussian]
    axes[2].bar(['中值滤波', '均值滤波', '高斯滤波'], edge_preservations,
                color=['blue', 'green', 'orange'])
    axes[2].set_title("边缘保持度对比")
    axes[2].set_ylabel("保持度")
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle("滤波器性能综合对比", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return (median_filtered, mean_filtered, gaussian_filtered,
            median_time, mean_time, gaussian_time,
            median_reduction, mean_reduction, gaussian_reduction,
            edge_preservation_median, edge_preservation_mean, edge_preservation_gaussian)

# 对比中值滤波和线性滤波
compare_median_vs_linear()

# ==================== 6. 中值滤波的特性分析 ====================
print("\n🔬 6. 中值滤波的特性分析")
print("=" * 30)


def analyze_median_filter_properties():
    """分析中值滤波的特性"""

    print("中值滤波的特性分析:")
    print("=" * 40)

    # 创建测试图案
    height, width = 100, 100

    # 1. 边缘保持测试
    edge_img = np.zeros((height, width), dtype=np.uint8)
    edge_img[:, width // 2:] = 255  # 锐利边缘

    # 添加噪声
    noisy_edge = add_salt_pepper_noise(edge_img, 0.05, 0.05)

    # 应用中值滤波
    filtered_edge = cv2.medianBlur(noisy_edge, 5)

    # 2. 角落保持测试
    corner_img = np.zeros((height, width), dtype=np.uint8)
    # 创建一个角落
    for i in range(height):
        for j in range(width):
            if i < 60 and j < 60:
                corner_img[i, j] = 200
            elif i >= 60 and j >= 60:
                corner_img[i, j] = 100

    noisy_corner = add_salt_pepper_noise(corner_img, 0.03, 0.03)
    filtered_corner = cv2.medianBlur(noisy_corner, 5)

    # 3. 细节保持测试
    detail_img = np.zeros((height, width), dtype=np.uint8)
    # 创建细线图案
    for i in range(0, height, 10):
        cv2.line(detail_img, (0, i), (width, i), 150, 1)

    noisy_detail = add_salt_pepper_noise(detail_img, 0.02, 0.02)
    filtered_detail = cv2.medianBlur(noisy_detail, 5)

    # 可视化特性分析
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    # 第一行：边缘保持
    axes[0, 0].imshow(edge_img, cmap='gray')
    axes[0, 0].set_title("原始边缘")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_edge, cmap='gray')
    axes[0, 1].set_title("加噪边缘")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(filtered_edge, cmap='gray')
    axes[0, 2].set_title("中值滤波后边缘")
    axes[0, 2].axis('off')

    # 第二行：角落保持
    axes[1, 0].imshow(corner_img, cmap='gray')
    axes[1, 0].set_title("原始角落")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(noisy_corner, cmap='gray')
    axes[1, 1].set_title("加噪角落")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(filtered_corner, cmap='gray')
    axes[1, 2].set_title("中值滤波后角落")
    axes[1, 2].axis('off')

    # 第三行：细节保持
    axes[2, 0].imshow(detail_img, cmap='gray')
    axes[2, 0].set_title("原始细节")
    axes[2, 0].axis('off')

    axes[2, 1].imshow(noisy_detail, cmap='gray')
    axes[2, 1].set_title("加噪细节")
    axes[2, 1].axis('off')

    axes[2, 2].imshow(filtered_detail, cmap='gray')
    axes[2, 2].set_title("中值滤波后细节")
    axes[2, 2].axis('off')

    plt.suptitle("中值滤波特性分析", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 分析结论
    print("\n中值滤波特性总结:")
    print("-" * 30)
    print("1. 边缘保持: 极佳，边缘保持清晰")
    print("2. 角落保持: 良好，角落形状基本不变")
    print("3. 细节保持: 中等，细线可能被部分破坏")
    print("4. 噪声去除: 对椒盐噪声效果极佳")
    print("5. 计算速度: 相对较慢（需要排序）")

    return edge_img, filtered_edge, corner_img, filtered_corner, detail_img, filtered_detail


# 分析中值滤波特性
edge_img, filtered_edge, corner_img, filtered_corner, detail_img, filtered_detail = analyze_median_filter_properties()

# ==================== 7. 实际应用案例 ====================
print("\n💼 7. 实际应用案例")
print("=" * 30)


def demonstrate_real_world_applications():
    """演示中值滤波在实际中的应用"""

    print("中值滤波的实际应用场景:")
    print("1. 文档扫描: 去除墨迹斑点")
    print("2. 医学影像: 去除X光片噪声")
    print("3. 天文图像: 去除宇宙射线噪声")
    print("4. 监控视频: 去除雪花噪声")
    print("5. 老照片修复: 去除划痕和污点")
    print()

    # 模拟不同应用场景
    applications = [
        ("文档扫描", "document", 3),
        ("医学影像", "medical", 5),
        ("监控视频", "surveillance", 3),
        ("老照片修复", "old_photo", 7)
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for idx, (app_name, app_type, ksize) in enumerate(applications):
        row = idx // 2
        col = (idx % 2) * 2

        if app_type == "document":
            # 文档扫描
            # 创建模拟文档
            doc = np.ones((100, 150), dtype=np.uint8) * 200
            # 添加文字
            cv2.putText(doc, "DOCUMENT", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 50, 2)
            cv2.putText(doc, "Sample text for", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, 50, 1)
            cv2.putText(doc, "document scanning.", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, 50, 1)

            # 添加墨迹斑点
            noisy_doc = add_salt_pepper_noise(doc, 0.03, 0.02)
            filtered_doc = cv2.medianBlur(noisy_doc, ksize)

            axes[row, col].imshow(noisy_doc, cmap='gray')
            axes[row, col].set_title(f"{app_name}\n原始(有墨迹)")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(filtered_doc, cmap='gray')
            axes[row, col + 1].set_title(f"{app_name}\n中值滤波后")
            axes[row, col + 1].axis('off')

        elif app_type == "medical":
            # 医学影像
            # 创建模拟X光片
            medical = np.random.randint(100, 200, (100, 150), dtype=np.uint8)
            # 添加骨骼结构
            cv2.rectangle(medical, (40, 20), (110, 80), 240, 10)
            cv2.circle(medical, (75, 50), 15, 230, 5)

            # 添加传感器噪声
            noisy_medical = add_salt_pepper_noise(medical, 0.02, 0.01)
            filtered_medical = cv2.medianBlur(noisy_medical, ksize)

            axes[row, col].imshow(noisy_medical, cmap='gray')
            axes[row, col].set_title(f"{app_name}\n原始(有噪声)")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(filtered_medical, cmap='gray')
            axes[row, col + 1].set_title(f"{app_name}\n中值滤波后")
            axes[row, col + 1].axis('off')

        elif app_type == "surveillance":
            # 监控视频
            # 创建模拟监控画面
            surveillance = np.random.randint(30, 100, (100, 150), dtype=np.uint8)
            # 添加移动物体
            cv2.rectangle(surveillance, (60, 40), (90, 70), 150, -1)

            # 添加雪花噪声
            noisy_surveillance = add_salt_pepper_noise(surveillance, 0.05, 0.05)
            filtered_surveillance = cv2.medianBlur(noisy_surveillance, ksize)

            axes[row, col].imshow(noisy_surveillance, cmap='gray')
            axes[row, col].set_title(f"{app_name}\n原始(雪花噪声)")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(filtered_surveillance, cmap='gray')
            axes[row, col + 1].set_title(f"{app_name}\n中值滤波后")
            axes[row, col + 1].axis('off')

        elif app_type == "old_photo":
            # 老照片修复
            # 创建模拟老照片
            photo = np.random.randint(100, 180, (100, 150), dtype=np.uint8)
            # 添加人脸
            cv2.circle(photo, (75, 40), 20, 200, -1)  # 头部
            cv2.circle(photo, (65, 35), 3, 50, -1)  # 左眼
            cv2.circle(photo, (85, 35), 3, 50, -1)  # 右眼
            cv2.ellipse(photo, (75, 50), (15, 8), 0, 0, 180, 50, 3)  # 嘴巴

            # 添加划痕和污点
            noisy_photo = add_salt_pepper_noise(photo, 0.04, 0.03)
            # 添加一些线状划痕
            cv2.line(noisy_photo, (20, 20), (130, 20), 0, 2)
            cv2.line(noisy_photo, (10, 80), (140, 80), 255, 1)

            filtered_photo = cv2.medianBlur(noisy_photo, ksize)

            axes[row, col].imshow(noisy_photo, cmap='gray')
            axes[row, col].set_title(f"{app_name}\n原始(有划痕污点)")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(filtered_photo, cmap='gray')
            axes[row, col + 1].set_title(f"{app_name}\n中值滤波后")
            axes[row, col + 1].axis('off')

    plt.suptitle("中值滤波在实际场景中的应用", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    # 应用建议
    print("\n中值滤波参数选择指南:")
    print("-" * 40)
    print("1. 轻度噪声: 核大小 3×3")
    print("2. 中度噪声: 核大小 5×5")
    print("3. 重度噪声: 核大小 7×7 或更大")
    print("4. 细线保护: 使用较小核 (避免破坏细节)")
    print("5. 实时处理: 注意计算时间 (中值滤波较慢)")


# 演示实际应用
demonstrate_real_world_applications()

# ==================== 8. 练习与挑战 ====================
print("\n💪 8. 练习与挑战")
print("=" * 30)

print("""
练习题：

1. 基础练习：
   a) 实现手动中值滤波，对比不同核大小效果
   b) 实现加权中值滤波（中心像素权重更高）
   c) 实现自适应中值滤波，根据局部噪声调整窗口大小

2. 进阶练习：
   a) 实现彩色图片的中值滤波（分别处理每个通道）
   b) 实现快速中值滤波算法（使用直方图或增量更新）
   c) 实现多级中值滤波（多层中值滤波组合）

3. 思考题：
   a) 为什么中值滤波能完全去除椒盐噪声？
   b) 中值滤波在什么情况下会破坏图像细节？
   c) 如何优化中值滤波的计算速度？
   d) 中值滤波与排序统计有什么联系？
""")

# 练习框架代码
print("\n💻 练习框架代码：")

print("""
# 练习1a: 手动中值滤波实现
def manual_median_filter_color(image, kernel_size=3):
    # 处理彩色图片
    b, g, r = cv2.split(image)

    b_filtered = manual_median_filter(b, kernel_size)
    g_filtered = manual_median_filter(g, kernel_size)
    r_filtered = manual_median_filter(r, kernel_size)

    filtered = cv2.merge([b_filtered, g_filtered, r_filtered])
    return filtered

# 练习1b: 加权中值滤波框架
def weighted_median_filter(image, kernel_size=3, center_weight=3):
    # 中心像素权重更高
    height, width = image.shape
    pad = kernel_size // 2

    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image, dtype=np.uint8)

    for i in range(pad, height + pad):
        for j in range(pad, width + pad):
            region = padded[i-pad:i+pad+1, j-pad:j+pad+1]
            flat_region = region.flatten()

            # 复制中心像素多次，增加权重
            center_value = region[pad, pad]
            weighted_values = np.concatenate([
                flat_region, 
                np.full(center_weight-1, center_value)  # 中心像素重复
            ])

            median_val = np.median(weighted_values)
            filtered[i-pad, j-pad] = median_val

    return filtered

# 练习1c: 自适应中值滤波框架
def adaptive_median_filter(image, max_window=7):
    height, width = image.shape
    filtered = np.zeros_like(image, dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            window_size = 3
            while window_size <= max_window:
                pad = window_size // 2

                # 提取局部区域
                i_start = max(0, i - pad)
                i_end = min(height, i + pad + 1)
                j_start = max(0, j - pad)
                j_end = min(width, j + pad + 1)

                region = image[i_start:i_end, j_start:j_end]
                flat_region = region.flatten()

                median_val = np.median(flat_region)
                min_val = np.min(flat_region)
                max_val = np.max(flat_region)

                # 检查中值是否为噪声
                if min_val < median_val < max_val:
                    # 检查当前像素是否为噪声
                    if min_val < image[i, j] < max_val:
                        filtered[i, j] = image[i, j]  # 不是噪声，保持原值
                    else:
                        filtered[i, j] = median_val  # 是噪声，用中值替换
                    break
                else:
                    window_size += 2  # 增大窗口
            else:
                # 达到最大窗口仍未找到合适中值
                filtered[i, j] = median_val

    return filtered
""")

# ==================== 9. 总结 ====================
print("\n" + "=" * 50)
print("✅ 中值滤波总结")
print("=" * 50)

summary = """
📊 中值滤波核心知识：

1. 数学原理
   - 操作: 取邻域像素的中值
   - 公式: I'(x,y) = median{ I(x+i, y+j) }
   - 计算: 排序 → 取中间值
   - 非线性: 不满足叠加性和齐次性

2. 实现方法
   - OpenCV: cv2.medianBlur(src, ksize)
   - 手动实现: 提取区域 → 排序 → 取中值
   - 核大小: 必须为奇数 (3, 5, 7, ...)

3. 性能特点
   - 时间复杂度: O(N²M² log M²) 原始
   - 可优化: 使用快速选择算法
   - 内存: 需要存储窗口内所有像素
   - 稳定性: 对脉冲噪声鲁棒

4. 优势
   - 完全去除椒盐噪声
   - 极佳的边缘保持能力
   - 不产生新的灰度值
   - 对孤立噪声点敏感

5. 局限性
   - 计算复杂度高
   - 对高斯噪声效果一般
   - 大窗口会模糊细节
   - 可能破坏细线和角落

6. 与线性滤波对比
   - 噪声去除: 中值 > 线性 (对椒盐噪声)
   - 边缘保持: 中值 >> 线性
   - 计算速度: 线性 > 中值
   - 适用噪声: 中值适合脉冲噪声，线性适合高斯噪声

7. 实际应用
   - 文档扫描去噪
   - 医学影像处理
   - 老照片修复
   - 监控视频去雪花
   - 天文图像处理

8. 最佳实践
   - 轻度噪声: 3×3窗口
   - 中度噪声: 5×5窗口
   - 重度噪声: 7×7窗口
   - 细节保护: 使用自适应窗口
   - 实时处理: 考虑计算成本

🎯 核心代码记忆：
   # OpenCV实现
   filtered = cv2.medianBlur(image, ksize)

   # 手动实现
   def median_filter_manual(image, ksize=3):
       height, width = image.shape
       pad = ksize // 2
       filtered = np.zeros_like(image)

       for i in range(pad, height-pad):
           for j in range(pad, width-pad):
               region = image[i-pad:i+pad+1, j-pad:j+pad+1]
               filtered[i, j] = np.median(region)

       return filtered
"""

print(summary)
print("\n📁 下一个文件: 05_05_双边滤波实现.py")
print("  我们将学习边缘保持滤波 - 双边滤波！")