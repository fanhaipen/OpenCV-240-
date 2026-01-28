"""
第5天 - 文件5：双边滤波实现
学习目标：掌握双边滤波的原理、实现和应用
重点：空间域权重、值域权重、边缘保持、参数调优
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math

print("🌈 第5天 - 文件5：双边滤波实现")
print("=" * 50)

# ==================== 1. 双边滤波理论 ====================
print("\n🎯 1. 双边滤波理论")
print("=" * 30)

print("""
双边滤波 (Bilateral Filter)：

数学原理：
  结合空间域权重和值域权重，实现边缘保持的平滑滤波

核心思想：
  1. 空间域权重 (Spatial Domain Weight)
     - 基于像素位置距离
     - 类似高斯滤波，距离越近权重越大
     - 公式: G_s(||p-q||) = exp(-||p-q||² / (2σ_s²))

  2. 值域权重 (Range Domain Weight) 
     - 基于像素灰度值相似度
     - 灰度值越接近权重越大
     - 公式: G_r(|I_p - I_q|) = exp(-|I_p - I_q|² / (2σ_r²))

  3. 组合权重
     - 总权重: W = G_s × G_r
     - 滤波结果: I'(p) = (Σ_q G_s(||p-q||) × G_r(|I_p - I_q|) × I_q) / (Σ_q W)

特点：
  - 非线性滤波
  - 边缘保持能力强
  - 计算复杂度高
  - 参数敏感（σ_s, σ_r）

优势：
  - 同时实现平滑和边缘保持
  - 对纹理和细节保护好
  - 适合处理具有丰富纹理的图像

局限性：
  - 计算速度慢
  - 参数选择复杂
  - 对强噪声效果有限

应用场景：
  - 人像美颜（皮肤平滑）
  - 纹理图像去噪
  - 高动态范围图像处理
  - 艺术效果处理
""")

# ==================== 2. 权重函数可视化 ====================
print("\n📊 2. 权重函数可视化")
print("=" * 30)


def visualize_bilateral_weights():
    """可视化双边滤波的权重函数"""

    # 空间域权重（距离权重）
    distances = np.linspace(0, 10, 100)
    spatial_sigmas = [1.0, 2.0, 3.0]

    # 值域权重（灰度相似度权重）
    intensity_diffs = np.linspace(0, 100, 100)
    range_sigmas = [10.0, 30.0, 50.0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 空间域权重
    for sigma_s in spatial_sigmas:
        spatial_weights = np.exp(-distances ** 2 / (2 * sigma_s ** 2))
        axes[0, 0].plot(distances, spatial_weights,
                        label=f'σ_s={sigma_s}', linewidth=2)

    axes[0, 0].set_title('空间域权重 (距离权重)')
    axes[0, 0].set_xlabel('像素距离')
    axes[0, 0].set_ylabel('权重')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 值域权重
    for sigma_r in range_sigmas:
        range_weights = np.exp(-intensity_diffs ** 2 / (2 * sigma_r ** 2))
        axes[0, 1].plot(intensity_diffs, range_weights,
                        label=f'σ_r={sigma_r}', linewidth=2)

    axes[0, 1].set_title('值域权重 (灰度相似度权重)')
    axes[0, 1].set_xlabel('灰度值差异')
    axes[0, 1].set_ylabel('权重')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 组合权重示例
    sigma_s = 2.0
    sigma_r = 30.0

    # 创建网格
    D, I = np.meshgrid(distances, intensity_diffs)

    # 计算组合权重
    spatial_part = np.exp(-D ** 2 / (2 * sigma_s ** 2))
    range_part = np.exp(-I ** 2 / (2 * sigma_r ** 2))
    combined_weights = spatial_part * range_part

    # 3D可视化
    from mpl_toolkits.mplot3d import Axes3D

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    surface = ax.plot_surface(D, I, combined_weights, cmap='viridis', alpha=0.8)
    ax.set_title('组合权重 (空间域×值域)')
    ax.set_xlabel('像素距离')
    ax.set_ylabel('灰度差异')
    ax.set_zlabel('组合权重')
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

    # 权重应用示例
    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.5,
                    "双边滤波权重应用:\n\n"
                    "情况1: 相近像素 + 相似灰度\n"
                    "  - 高空间权重 ✓\n"
                    "  - 高值域权重 ✓\n"
                    "  - 总权重: 高 ✓\n\n"
                    "情况2: 相近像素 + 不同灰度 (边缘)\n"
                    "  - 高空间权重 ✓\n"
                    "  - 低值域权重 ✗\n"
                    "  - 总权重: 低 ✗\n\n"
                    "情况3: 远距离像素 + 相似灰度\n"
                    "  - 低空间权重 ✗\n"
                    "  - 高值域权重 ✓\n"
                    "  - 总权重: 低 ✗",
                    fontsize=10, verticalalignment='center')

    plt.suptitle("双边滤波权重函数可视化", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return distances, spatial_sigmas, intensity_diffs, range_sigmas


# 可视化权重函数
distances, spatial_sigmas, intensity_diffs, range_sigmas = visualize_bilateral_weights()

# ==================== 3. 创建测试图片 ====================
print("\n🎨 3. 创建测试图片")
print("=" * 30)


def create_test_image_for_bilateral():
    """创建用于双边滤波测试的图片"""
    height, width = 200, 300
    img = np.zeros((height, width), dtype=np.uint8)

    # 创建丰富的纹理和边缘
    # 梯度背景
    for i in range(height):
        img[i, :] = int(50 + 150 * i / height)

    # 添加锐利边缘
    cv2.rectangle(img, (30, 30), (120, 80), 200, -1)  # 亮矩形
    cv2.rectangle(img, (180, 30), (270, 80), 50, -1)  # 暗矩形

    # 添加纹理区域
    for i in range(10, 90, 20):
        for j in range(150, 250, 15):
            cv2.circle(img, (j, i), 3, 150, -1)

    # 添加细线（测试边缘保持）
    for i in range(5):
        y = 100 + i * 15
        cv2.line(img, (50, y), (250, y), 100, 2)

    # 添加文字
    cv2.putText(img, "BILATERAL FILTER", (70, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    cv2.putText(img, "Edge Preserving", (90, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 150, 1)

    return img


# 添加高斯噪声
def add_gaussian_noise(image, mean=0, std=25):
    """添加高斯噪声"""
    noise = np.random.normal(mean, std, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# 创建测试图片
test_img_bilateral = create_test_image_for_bilateral()
noisy_img_bilateral = add_gaussian_noise(test_img_bilateral, std=20)

print("测试图片创建完成")
print(f"图片尺寸: {test_img_bilateral.shape[1]}x{test_img_bilateral.shape[0]}")
print(f"噪声类型: 高斯噪声 (σ=20)")

# 显示测试图片
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(test_img_bilateral, cmap='gray')
axes[0].set_title("原始测试图片")
axes[0].axis('off')

axes[1].imshow(noisy_img_bilateral, cmap='gray')
axes[1].set_title("高斯噪声图片")
axes[1].axis('off')
plt.tight_layout()
plt.show()

# ==================== 4. OpenCV双边滤波实现 ====================
print("\n🔧 4. OpenCV双边滤波实现")
print("=" * 30)


def demonstrate_opencv_bilateral():
    """演示OpenCV双边滤波"""

    print("OpenCV双边滤波函数: cv2.bilateralFilter()")
    print("参数: src, d, sigmaColor, sigmaSpace[, borderType]")
    print("  d: 滤波直径（邻域直径）")
    print("  sigmaColor: 值域标准差，控制灰度相似度权重")
    print("  sigmaSpace: 空间域标准差，控制空间距离权重")
    print()

    # 测试不同参数
    test_cases = [
        (5, 25, 25, "小核细平滑"),
        (9, 50, 50, "标准参数"),
        (9, 10, 75, "强边缘保持"),
        (9, 100, 25, "强平滑弱边缘"),
        (15, 75, 75, "大核强平滑"),
        (9, 150, 150, "强平滑")
    ]

    results_opencv = []
    opencv_times = []

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for idx, (d, sigma_color, sigma_space, description) in enumerate(test_cases):
        start_time = time.time()
        filtered = cv2.bilateralFilter(noisy_img_bilateral, d, sigma_color, sigma_space)
        end_time = time.time()

        results_opencv.append((d, sigma_color, sigma_space, description, filtered))
        opencv_times.append(end_time - start_time)

        # 计算性能指标
        original_noise = np.std(noisy_img_bilateral.astype(np.float32) - test_img_bilateral.astype(np.float32))
        current_noise = np.std(filtered.astype(np.float32) - test_img_bilateral.astype(np.float32))

        if original_noise > 0:
            noise_reduction = 100 * (original_noise - current_noise) / original_noise
        else:
            noise_reduction = 100

        print(f"测试 {description}:")
        print(f"  直径: {d}, σ_color: {sigma_color}, σ_space: {sigma_space}")
        print(f"  计算时间: {opencv_times[-1]:.4f}秒")
        print(f"  噪声减少: {noise_reduction:.1f}%")

        # 显示结果
        row = idx // 3
        col = idx % 3
        if idx < 6:
            axes[row, col].imshow(filtered, cmap='gray')
            axes[row, col].set_title(f"{description}\nd={d},σ_c={sigma_color},σ_s={sigma_space}")
            axes[row, col].axis('off')

    plt.suptitle("OpenCV双边滤波不同参数效果", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return results_opencv, opencv_times


# 演示OpenCV实现
opencv_results, opencv_times = demonstrate_opencv_bilateral()

# ==================== 5. 双边滤波参数影响分析 ====================
print("\n🔍 5. 双边滤波参数影响分析")
print("=" * 30)


def analyze_parameter_effects():
    """分析双边滤波参数的影响"""

    print("双边滤波参数影响分析:")
    print("=" * 50)

    # 测试sigma_color的影响
    sigma_colors = [10, 25, 50, 75, 100]
    sigma_space_fixed = 50
    d_fixed = 9

    results_sigma_color = []

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for idx, sigma_color in enumerate(sigma_colors):
        filtered = cv2.bilateralFilter(noisy_img_bilateral, d_fixed, sigma_color, sigma_space_fixed)
        results_sigma_color.append((sigma_color, filtered))

        # 计算边缘保持度
        edges_original = cv2.Canny(test_img_bilateral, 50, 150)
        edges_filtered = cv2.Canny(filtered, 50, 150)
        edge_overlap = np.sum((edges_original > 0) & (edges_filtered > 0))
        total_original_edges = np.sum(edges_original > 0)
        edge_preservation = edge_overlap / total_original_edges if total_original_edges > 0 else 0

        # 计算噪声减少
        original_noise = np.std(noisy_img_bilateral.astype(np.float32) - test_img_bilateral.astype(np.float32))
        current_noise = np.std(filtered.astype(np.float32) - test_img_bilateral.astype(np.float32))
        noise_reduction = 100 * (original_noise - current_noise) / original_noise if original_noise > 0 else 100

        print(f"σ_color={sigma_color}:")
        print(f"  噪声减少: {noise_reduction:.1f}%")
        print(f"  边缘保持: {edge_preservation:.3f}")

        # 显示结果
        if idx < 5:
            row = idx // 3
            col = idx % 3
            axes[row, col].imshow(filtered, cmap='gray')
            axes[row, col].set_title(
                f"σ_color={sigma_color}\n噪声减少:{noise_reduction:.1f}%\n边缘保持:{edge_preservation:.3f}")
            axes[row, col].axis('off')

    # 参数影响总结
    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.5,
                    "σ_color (值域标准差) 影响:\n\n"
                    "小σ_color (10-25):\n"
                    "  - 强边缘保持\n"
                    "  - 弱噪声去除\n"
                    "  - 适合纹理丰富图像\n\n"
                    "中σ_color (50):\n"
                    "  - 平衡效果\n"
                    "  - 一般应用\n\n"
                    "大σ_color (75-100):\n"
                    "  - 强噪声去除\n"
                    "  - 边缘可能模糊\n"
                    "  - 适合平滑区域",
                    fontsize=10, verticalalignment='center')

    plt.suptitle("σ_color参数影响分析 (σ_space=50, d=9)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 测试sigma_space的影响
    sigma_spaces = [10, 25, 50, 75, 100]
    sigma_color_fixed = 50
    d_fixed = 9

    results_sigma_space = []

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for idx, sigma_space in enumerate(sigma_spaces):
        filtered = cv2.bilateralFilter(noisy_img_bilateral, d_fixed, sigma_color_fixed, sigma_space)
        results_sigma_space.append((sigma_space, filtered))

        # 计算性能指标
        edges_original = cv2.Canny(test_img_bilateral, 50, 150)
        edges_filtered = cv2.Canny(filtered, 50, 150)
        edge_overlap = np.sum((edges_original > 0) & (edges_filtered > 0))
        total_original_edges = np.sum(edges_original > 0)
        edge_preservation = edge_overlap / total_original_edges if total_original_edges > 0 else 0

        original_noise = np.std(noisy_img_bilateral.astype(np.float32) - test_img_bilateral.astype(np.float32))
        current_noise = np.std(filtered.astype(np.float32) - test_img_bilateral.astype(np.float32))
        noise_reduction = 100 * (original_noise - current_noise) / original_noise if original_noise > 0 else 100

        print(f"σ_space={sigma_space}:")
        print(f"  噪声减少: {noise_reduction:.1f}%")
        print(f"  边缘保持: {edge_preservation:.3f}")

        # 显示结果
        if idx < 5:
            row = idx // 3
            col = idx % 3
            axes[row, col].imshow(filtered, cmap='gray')
            axes[row, col].set_title(
                f"σ_space={sigma_space}\n噪声减少:{noise_reduction:.1f}%\n边缘保持:{edge_preservation:.3f}")
            axes[row, col].axis('off')

    # 参数影响总结
    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.5,
                    "σ_space (空间域标准差) 影响:\n\n"
                    "小σ_space (10-25):\n"
                    "  - 局部平滑\n"
                    "  - 计算快\n"
                    "  - 适合细节保护\n\n"
                    "中σ_space (50):\n"
                    "  - 平衡效果\n"
                    "  - 一般应用\n\n"
                    "大σ_space (75-100):\n"
                    "  - 全局平滑\n"
                    "  - 计算慢\n"
                    "  - 适合大区域平滑",
                    fontsize=10, verticalalignment='center')

    plt.suptitle("σ_space参数影响分析 (σ_color=50, d=9)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return results_sigma_color, results_sigma_space


# 分析参数影响
sigma_color_results, sigma_space_results = analyze_parameter_effects()

# ==================== 6. 双边滤波 vs 其他滤波对比 ====================
print("\n🔍 6. 双边滤波 vs 其他滤波对比")
print("=" * 30)


def compare_bilateral_vs_others():
    """对比双边滤波与其他滤波"""

    print("双边滤波 vs 其他滤波对比分析:")
    print("=" * 50)

    # 测试条件
    kernel_size = 9
    sigma_color = 50
    sigma_space = 50

    # 1. 双边滤波
    start_time = time.time()
    bilateral_filtered = cv2.bilateralFilter(noisy_img_bilateral, kernel_size, sigma_color, sigma_space)
    bilateral_time = time.time() - start_time

    # 2. 高斯滤波
    start_time = time.time()
    gaussian_filtered = cv2.GaussianBlur(noisy_img_bilateral, (kernel_size, kernel_size), 1.5)
    gaussian_time = time.time() - start_time

    # 3. 均值滤波
    start_time = time.time()
    mean_filtered = cv2.blur(noisy_img_bilateral, (kernel_size, kernel_size))
    mean_time = time.time() - start_time

    # 4. 中值滤波
    start_time = time.time()
    median_filtered = cv2.medianBlur(noisy_img_bilateral, kernel_size)
    median_time = time.time() - start_time

    # 计算性能指标
    original_noise = np.std(noisy_img_bilateral.astype(np.float32) - test_img_bilateral.astype(np.float32))

    bilateral_noise = np.std(bilateral_filtered.astype(np.float32) - test_img_bilateral.astype(np.float32))
    gaussian_noise = np.std(gaussian_filtered.astype(np.float32) - test_img_bilateral.astype(np.float32))
    mean_noise = np.std(mean_filtered.astype(np.float32) - test_img_bilateral.astype(np.float32))
    median_noise = np.std(median_filtered.astype(np.float32) - test_img_bilateral.astype(np.float32))

    bilateral_reduction = 100 * (original_noise - bilateral_noise) / original_noise if original_noise > 0 else 100
    gaussian_reduction = 100 * (original_noise - gaussian_noise) / original_noise if original_noise > 0 else 100
    mean_reduction = 100 * (original_noise - mean_noise) / original_noise if original_noise > 0 else 100
    median_reduction = 100 * (original_noise - median_noise) / original_noise if original_noise > 0 else 100

    # 计算边缘保持度
    def calculate_edge_preservation(original, filtered):
        edges_original = cv2.Canny(original, 50, 150)
        edges_filtered = cv2.Canny(filtered, 50, 150)
        edge_overlap = np.sum((edges_original > 0) & (edges_filtered > 0))
        total_original_edges = np.sum(edges_original > 0)
        return edge_overlap / total_original_edges if total_original_edges > 0 else 0

    bilateral_edge = calculate_edge_preservation(test_img_bilateral, bilateral_filtered)
    gaussian_edge = calculate_edge_preservation(test_img_bilateral, gaussian_filtered)
    mean_edge = calculate_edge_preservation(test_img_bilateral, mean_filtered)
    median_edge = calculate_edge_preservation(test_img_bilateral, median_filtered)

    print(f"核大小: {kernel_size}×{kernel_size}")
    print(f"双边滤波参数: σ_color={sigma_color}, σ_space={sigma_space}")
    print()
    print("性能对比:")
    print(f"  双边滤波 - 时间: {bilateral_time:.4f}s, 噪声减少: {bilateral_reduction:.1f}%")
    print(f"  高斯滤波 - 时间: {gaussian_time:.4f}s, 噪声减少: {gaussian_reduction:.1f}%")
    print(f"  均值滤波 - 时间: {mean_time:.4f}s, 噪声减少: {mean_reduction:.1f}%")
    print(f"  中值滤波 - 时间: {median_time:.4f}s, 噪声减少: {median_reduction:.1f}%")
    print()
    print("边缘保持度 (越高越好):")
    print(f"  双边滤波: {bilateral_edge:.3f}")
    print(f"  高斯滤波: {gaussian_edge:.3f}")
    print(f"  均值滤波: {mean_edge:.3f}")
    print(f"  中值滤波: {median_edge:.3f}")

    # 可视化对比
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))

    # 第一行：原始和噪声
    axes[0, 0].imshow(test_img_bilateral, cmap='gray')
    axes[0, 0].set_title("原始图片")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_img_bilateral, cmap='gray')
    axes[0, 1].set_title("噪声图片")
    axes[0, 1].axis('off')

    # 第二行：滤波结果
    images_row1 = [bilateral_filtered, gaussian_filtered, mean_filtered, median_filtered]
    titles_row1 = ["双边滤波", "高斯滤波", "均值滤波", "中值滤波"]

    for i in range(4):
        axes[1, i].imshow(images_row1[i], cmap='gray')
        axes[1, i].set_title(titles_row1[i])
        axes[1, i].axis('off')

    # 第三行：边缘检测对比
    edges_original = cv2.Canny(test_img_bilateral, 50, 150)
    edges_bilateral = cv2.Canny(bilateral_filtered, 50, 150)
    edges_gaussian = cv2.Canny(gaussian_filtered, 50, 150)
    edges_mean = cv2.Canny(mean_filtered, 50, 150)
    edges_median = cv2.Canny(median_filtered, 50, 150)

    edges_images = [edges_original, edges_bilateral, edges_gaussian, edges_mean]
    edges_titles = ["原始边缘", "双边滤波边缘", "高斯滤波边缘", "均值滤波边缘"]

    for i in range(4):
        axes[2, i].imshow(edges_images[i], cmap='gray')
        axes[2, i].set_title(edges_titles[i])
        axes[2, i].axis('off')

    plt.suptitle("双边滤波 vs 其他滤波对比（高斯噪声）", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 性能对比可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 计算时间对比
    times = [bilateral_time, gaussian_time, mean_time, median_time]
    axes[0].bar(['双边滤波', '高斯滤波', '均值滤波', '中值滤波'], times,
                color=['blue', 'orange', 'green', 'red'])
    axes[0].set_title("计算时间对比")
    axes[0].set_ylabel("时间 (秒)")
    axes[0].grid(True, alpha=0.3, axis='y')

    # 噪声去除率对比
    reductions = [bilateral_reduction, gaussian_reduction, mean_reduction, median_reduction]
    axes[1].bar(['双边滤波', '高斯滤波', '均值滤波', '中值滤波'], reductions,
                color=['blue', 'orange', 'green', 'red'])
    axes[1].set_title("噪声去除率对比")
    axes[1].set_ylabel("去除率 (%)")
    axes[1].set_ylim([0, 100])
    axes[1].grid(True, alpha=0.3, axis='y')

    # 边缘保持度对比
    edge_preservations = [bilateral_edge, gaussian_edge, mean_edge, median_edge]
    axes[2].bar(['双边滤波', '高斯滤波', '均值滤波', '中值滤波'], edge_preservations,
                color=['blue', 'orange', 'green', 'red'])
    axes[2].set_title("边缘保持度对比")
    axes[2].set_ylabel("保持度")
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle("滤波器性能综合对比", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return (bilateral_filtered, gaussian_filtered, mean_filtered, median_filtered,
            bilateral_time, gaussian_time, mean_time, median_time,
            bilateral_reduction, gaussian_reduction, mean_reduction, median_reduction,
            bilateral_edge, gaussian_edge, mean_edge, median_edge)


# 对比双边滤波与其他滤波
comparison_results = compare_bilateral_vs_others()

# ==================== 7. 实际应用案例 ====================
print("\n💼 7. 实际应用案例")
print("=" * 30)


def demonstrate_real_world_applications():
    """演示双边滤波在实际中的应用"""

    print("双边滤波的实际应用场景:")
    print("1. 人像美颜: 皮肤平滑处理")
    print("2. 纹理保护: 去噪同时保护细节")
    print("3. 医学影像: 增强诊断特征")
    print("4. 艺术处理: 创建油画效果")
    print("5. 高动态范围: 色调映射")
    print()

    # 模拟不同应用场景
    applications = [
        ("人像美颜", "portrait", 9, 25, 25),
        ("纹理保护", "texture", 9, 10, 50),
        ("艺术效果", "artistic", 15, 50, 50),
        ("医学影像", "medical", 9, 30, 30)
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for idx, (app_name, app_type, d, sigma_color, sigma_space) in enumerate(applications):
        row = idx // 2
        col = (idx % 2) * 2

        if app_type == "portrait":
            # 人像美颜
            # 创建模拟皮肤
            skin = np.ones((100, 100), dtype=np.uint8) * 180
            # 添加皮肤纹理
            for i in range(0, 100, 5):
                cv2.line(skin, (0, i), (100, i), 170, 1)

            # 添加模拟瑕疵
            for _ in range(20):
                x, y = np.random.randint(0, 100, 2)
                cv2.circle(skin, (x, y), 2, 200, -1)  # 斑点
            for _ in range(5):
                x, y = np.random.randint(0, 100, 2)
                cv2.circle(skin, (x, y), 1, 150, -1)  # 毛孔

            # 添加眼睛、嘴巴
            cv2.circle(skin, (40, 40), 8, 50, -1)  # 左眼
            cv2.circle(skin, (60, 40), 8, 50, -1)  # 右眼
            cv2.ellipse(skin, (50, 60), (20, 10), 0, 0, 180, 50, 2)  # 嘴巴

            smoothed = cv2.bilateralFilter(skin, d, sigma_color, sigma_space)

            axes[row, col].imshow(skin, cmap='gray')
            axes[row, col].set_title(f"{app_name}\n原始皮肤")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(smoothed, cmap='gray')
            axes[row, col + 1].set_title(f"{app_name}\n双边滤波后")
            axes[row, col + 1].axis('off')

        elif app_type == "texture":
            # 纹理保护
            # 创建纹理图片
            texture = np.zeros((100, 150), dtype=np.uint8)
            # 添加网格纹理
            for i in range(0, 100, 10):
                cv2.line(texture, (0, i), (150, i), 200, 2)
            for j in range(0, 150, 10):
                cv2.line(texture, (j, 0), (j, 100), 200, 2)

            # 添加噪声
            noisy_texture = add_gaussian_noise(texture, std=20)
            filtered_texture = cv2.bilateralFilter(noisy_texture, d, sigma_color, sigma_space)

            axes[row, col].imshow(noisy_texture, cmap='gray')
            axes[row, col].set_title(f"{app_name}\n噪声纹理")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(filtered_texture, cmap='gray')
            axes[row, col + 1].set_title(f"{app_name}\n双边滤波后")
            axes[row, col + 1].axis('off')

        elif app_type == "artistic":
            # 艺术效果
            original = test_img_bilateral[50:150, 50:200]
            artistic = cv2.bilateralFilter(original, d, sigma_color, sigma_space)

            axes[row, col].imshow(original, cmap='gray')
            axes[row, col].set_title(f"{app_name}\n原始")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(artistic, cmap='gray')
            axes[row, col + 1].set_title(f"{app_name}\n油画效果")
            axes[row, col + 1].axis('off')

        elif app_type == "medical":
            # 医学影像
            # 创建模拟X光片
            medical = np.random.randint(120, 200, (100, 150), dtype=np.uint8)
            # 添加骨骼结构
            cv2.rectangle(medical, (40, 20), (110, 80), 240, 8)
            cv2.circle(medical, (75, 50), 10, 230, 5)

            # 添加噪声
            noisy_medical = add_gaussian_noise(medical, std=15)
            filtered_medical = cv2.bilateralFilter(noisy_medical, d, sigma_color, sigma_space)

            axes[row, col].imshow(noisy_medical, cmap='gray')
            axes[row, col].set_title(f"{app_name}\n噪声影像")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(filtered_medical, cmap='gray')
            axes[row, col + 1].set_title(f"{app_name}\n双边滤波后")
            axes[row, col + 1].axis('off')

    plt.suptitle("双边滤波在实际场景中的应用", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    # 应用建议
    print("\n双边滤波参数选择指南:")
    print("-" * 40)
    print("1. 人像美颜: d=5-9, σ_color=20-30, σ_space=20-30")
    print("2. 纹理保护: d=5-9, σ_color=10-20, σ_space=30-50")
    print("3. 艺术效果: d=9-15, σ_color=30-50, σ_space=30-50")
    print("4. 医学影像: d=5-9, σ_color=20-40, σ_space=20-40")
    print("5. 实时处理: 使用小d值 (d≤7)")
    print("\n一般原则:")
    print("  - σ_color控制平滑程度: 小值保护细节，大值平滑更强")
    print("  - σ_space控制影响范围: 小值局部平滑，大值全局平滑")


# 演示实际应用
demonstrate_real_world_applications()

# ==================== 8. 练习与挑战 ====================
print("\n💪 8. 练习与挑战")
print("=" * 30)

print("""
练习题：

1. 基础练习：
   a) 使用OpenCV的双边滤波处理自己的照片
   b) 对比不同σ_color和σ_space参数的效果
   c) 实现彩色图片的双边滤波（分别处理每个通道）

2. 进阶练习：
   a) 实现自适应双边滤波，根据局部特征调整参数
   b) 实现快速双边滤波算法（使用近似方法加速）
   c) 实现多尺度双边滤波（结合图像金字塔）

3. 思考题：
   a) 为什么双边滤波能同时实现平滑和边缘保持？
   b) 双边滤波的计算复杂度为什么高？如何优化？
   c) 在什么情况下双边滤波效果最好？
   d) 如何选择最优的σ_color和σ_space参数？
""")

# 练习框架代码
print("\n💻 练习框架代码：")

print("""
# 练习1a: 使用OpenCV处理彩色图片
def bilateral_filter_color(image, d=9, sigma_color=50, sigma_space=50):
    # 分离通道
    b, g, r = cv2.split(image)

    # 对每个通道分别应用双边滤波
    b_filtered = cv2.bilateralFilter(b, d, sigma_color, sigma_space)
    g_filtered = cv2.bilateralFilter(g, d, sigma_color, sigma_space)
    r_filtered = cv2.bilateralFilter(r, d, sigma_color, sigma_space)

    # 合并通道
    filtered = cv2.merge([b_filtered, g_filtered, r_filtered])
    return filtered

# 练习1b: 参数调优函数
def tune_bilateral_parameters(image, d_values, sigma_color_values, sigma_space_values):
    best_params = None
    best_score = -np.inf

    for d in d_values:
        for sigma_color in sigma_color_values:
            for sigma_space in sigma_space_values:
                # 应用双边滤波
                filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

                # 计算评估分数（可以根据需要定义）
                # 例如：噪声减少 + 边缘保持
                score = evaluate_filter_quality(image, filtered)

                if score > best_score:
                    best_score = score
                    best_params = (d, sigma_color, sigma_space)

    return best_params, best_score

# 练习2a: 自适应双边滤波框架
def adaptive_bilateral_filter(image, base_d=5, base_sigma_color=30, base_sigma_space=30):
    height, width = image.shape
    filtered = np.zeros_like(image, dtype=np.float32)

    for i in range(height):
        for j in range(width):
            # 计算局部特征
            local_region = image[max(0, i-2):min(height, i+3), 
                                max(0, j-2):min(width, j+3)]
            local_variance = np.var(local_region)

            # 根据局部特征调整参数
            if local_variance > 500:  # 高纹理/边缘区域
                sigma_color = base_sigma_color * 0.5
                sigma_space = base_sigma_space * 0.8
            else:  # 平滑区域
                sigma_color = base_sigma_color * 1.5
                sigma_space = base_sigma_space * 1.2

            # 应用局部双边滤波
            pad = base_d // 2
            region = image[max(0, i-pad):min(height, i+pad+1), 
                          max(0, j-pad):min(width, j+pad+1)]

            if region.size > 0:
                # 简化：使用OpenCV的bilateralFilter处理局部区域
                # 注意：这只是一个框架，实际实现会更复杂
                filtered_region = cv2.bilateralFilter(region, base_d, sigma_color, sigma_space)
                # 取中心像素
                filtered[i, j] = filtered_region[region.shape[0]//2, region.shape[1]//2]

    return filtered.astype(np.uint8)
""")

# ==================== 9. 总结 ====================
print("\n" + "=" * 50)
print("✅ 双边滤波总结")
print("=" * 50)

summary = """
📊 双边滤波核心知识：

1. 数学原理
   - 空间域权重: G_s(||p-q||) = exp(-||p-q||²/(2σ_s²))
   - 值域权重: G_r(|I_p-I_q|) = exp(-|I_p-I_q|²/(2σ_r²))
   - 组合权重: W = G_s × G_r
   - 归一化加权平均

2. 参数意义
   - d: 滤波直径，影响计算区域大小
   - σ_color: 值域标准差，控制灰度相似度权重
     * 小值: 强边缘保持，弱平滑
     * 大值: 弱边缘保持，强平滑
   - σ_space: 空间域标准差，控制空间距离权重
     * 小值: 局部平滑，计算快
     * 大值: 全局平滑，计算慢

3. 实现方法
   - OpenCV: cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
   - 手动实现: 计算双重权重，加权平均
   - 计算复杂度: O(N²d²)，d为直径

4. 性能特点
   - 非线性滤波
   - 边缘保持能力极强
   - 计算复杂度高
   - 对高斯噪声效果较好
   - 对椒盐噪声效果一般

5. 与其他滤波对比
   - vs 高斯滤波: 双边滤波边缘保持更好
   - vs 均值滤波: 双边滤波细节保护更好
   - vs 中值滤波: 双边滤波对高斯噪声更有效
   - 计算速度: 均值 < 高斯 < 中值 < 双边

6. 实际应用
   - 人像美颜: 皮肤平滑，保护五官
   - 纹理图像: 去噪同时保护纹理
   - 医学影像: 增强特征，减少伪影
   - 艺术处理: 创建油画、水彩效果

7. 最佳实践
   - 人像处理: d=5-9, σ_color=20-30, σ_space=20-30
   - 纹理保护: d=5-9, σ_color=10-20, σ_space=30-50
   - 艺术效果: d=9-15, σ_color=30-50, σ_space=30-50
   - 实时处理: 使用小d值 (d≤7)
   - 参数调优: 从标准参数开始，根据效果微调

🎯 核心代码记忆：
   # OpenCV实现
   filtered = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

   # 标准参数设置
   d = 9          # 滤波直径
   sigma_color = 50  # 值域标准差
   sigma_space = 50  # 空间域标准差

   # 应用
   result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
"""

print(summary)
print("\n📁 第5天学习完成！")
print("  我们掌握了4种重要的图像滤波器：")
print("  1. 均值滤波 - 简单快速")
print("  2. 高斯滤波 - 平滑自然")
print("  3. 中值滤波 - 去椒盐噪声")
print("  4. 双边滤波 - 边缘保持")
print("\n🎉 明天我们将开始第6天的学习：边缘检测基础！")