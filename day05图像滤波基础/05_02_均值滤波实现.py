"""
第5天 - 文件2：均值滤波实现
学习目标：掌握均值滤波的原理、实现和优化
重点：手动实现、OpenCV实现、边界处理、性能对比
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

print("📊 第5天 - 文件2：均值滤波实现")
print("=" * 50)

# ==================== 1. 均值滤波理论回顾 ====================
print("\n🎯 1. 均值滤波理论回顾")
print("=" * 30)

print("""
均值滤波 (Mean/Average Filter)：

数学原理：
  用邻域内像素的平均值替换中心像素值

公式：
  I'(x,y) = (1/(M×N)) × Σ_{i=-a}^{a} Σ_{j=-b}^{b} I(x+i, y+j)

其中：
  M×N: 滤波核大小（通常为奇数）
  a = (M-1)/2, b = (N-1)/2

卷积核（3×3示例）：
  [1/9, 1/9, 1/9]
  [1/9, 1/9, 1/9]
  [1/9, 1/9, 1/9]

特点：
  1. 线性滤波
  2. 简单快速
  3. 有效去除高斯噪声
  4. 会使图像模糊，边缘不清晰
  5. 对椒盐噪声效果一般
""")

# ==================== 2. 创建测试图片 ====================
print("\n🎨 2. 创建测试图片")
print("=" * 30)


def create_test_image_with_details():
    """创建包含细节的测试图片"""
    height, width = 200, 300
    img = np.zeros((height, width), dtype=np.uint8)

    # 添加梯度背景
    for i in range(height):
        img[i, :] = int(100 + 100 * i / height)

    # 添加测试图案
    # 1. 边缘（锐利变化）
    cv2.line(img, (0, 50), (width, 50), 200, 2)
    cv2.line(img, (0, 150), (width, 150), 50, 2)

    # 2. 文字（高频细节）
    cv2.putText(img, "MEAN FILTER", (80, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    # 3. 小点（孤立噪声）
    cv2.circle(img, (50, 180), 2, 255, -1)
    cv2.circle(img, (100, 180), 2, 0, -1)

    # 4. 网格（周期性图案）
    for i in range(0, width, 20):
        cv2.line(img, (i, 0), (i, height), 150, 1)
    for j in range(0, height, 20):
        cv2.line(img, (0, j), (width, j), 150, 1)

    return img


# 创建测试图片
clean_img = create_test_image_with_details()

print(f"测试图片创建完成")
print(f"尺寸: {clean_img.shape[1]}x{clean_img.shape[0]}")
print(f"数据类型: {clean_img.dtype}")
print(f"值范围: [{clean_img.min()}, {clean_img.max()}]")

# 显示原始图片
plt.figure(figsize=(8, 6))
plt.imshow(clean_img, cmap='gray')
plt.title("原始测试图片（含边缘、文字、网格等细节）")
plt.colorbar(label='灰度值')
plt.axis('off')
plt.tight_layout()
plt.show()

# ==================== 3. 添加噪声用于测试 ====================
print("\n🎨 3. 添加噪声用于测试")
print("=" * 30)


def add_gaussian_noise(image, mean=0, std=25):
    """添加高斯噪声"""
    noise = np.random.normal(mean, std, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
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


# 创建有噪声的图片
gaussian_noisy = add_gaussian_noise(clean_img, std=30)
salt_pepper_noisy = add_salt_pepper_noise(clean_img, 0.02, 0.02)

# 显示噪声图片
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

images = [clean_img, gaussian_noisy, salt_pepper_noisy]
titles = ["原始图片", "高斯噪声 (std=30)", "椒盐噪声 (2%)"]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

    # 计算噪声水平
    if title != "原始图片":
        noise_level = np.std(img.astype(np.float32) - clean_img.astype(np.float32))
        ax.text(0.5, -0.1, f'噪声水平: {noise_level:.1f}',
                transform=ax.transAxes, ha='center', fontsize=9)

plt.suptitle("不同噪声类型对比", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

# ==================== 4. 手动实现均值滤波 ====================
print("\n🔧 4. 手动实现均值滤波")
print("=" * 30)


def manual_mean_filter(image, kernel_size=3, border_type='zero'):
    """
    手动实现均值滤波

    参数:
        image: 输入图片
        kernel_size: 滤波核大小（奇数）
        border_type: 边界处理类型 ('zero', 'replicate', 'reflect')

    返回:
        滤波后的图片
    """
    if kernel_size % 2 == 0:
        raise ValueError("滤波核大小必须是奇数")

    height, width = image.shape
    pad = kernel_size // 2

    # 边界填充
    if border_type == 'zero':
        padded = np.pad(image, pad, mode='constant', constant_values=0)
    elif border_type == 'replicate':
        padded = np.pad(image, pad, mode='edge')
    elif border_type == 'reflect':
        padded = np.pad(image, pad, mode='reflect')
    else:
        raise ValueError(f"不支持的边界类型: {border_type}")

    # 创建输出图片
    filtered = np.zeros_like(image, dtype=np.float32)

    # 计算均值
    for i in range(pad, height + pad):
        for j in range(pad, width + pad):
            # 提取局部区域
            region = padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
            # 计算平均值
            filtered[i - pad, j - pad] = np.mean(region)

    return filtered.astype(np.uint8)


def manual_mean_filter_optimized(image, kernel_size=3):
    """优化版手动均值滤波（使用积分图加速）"""
    height, width = image.shape
    pad = kernel_size // 2

    # 转换为浮点数
    img_float = image.astype(np.float32)

    # 创建输出图片
    filtered = np.zeros_like(img_float)

    # 计算积分图
    integral = np.cumsum(np.cumsum(img_float, axis=0), axis=1)

    # 填充积分图边界
    integral = np.pad(integral, ((1, 0), (1, 0)), mode='constant', constant_values=0)

    # 使用积分图快速计算区域和
    for i in range(height):
        for j in range(width):
            # 计算区域边界
            i1 = max(0, i - pad)
            j1 = max(0, j - pad)
            i2 = min(height - 1, i + pad)
            j2 = min(width - 1, j + pad)

            # 计算区域面积
            area = (i2 - i1 + 1) * (j2 - j1 + 1)

            # 使用积分图计算区域和
            # 注意：积分图索引偏移了1
            sum_val = (integral[i2 + 1, j2 + 1] - integral[i1, j2 + 1] -
                       integral[i2 + 1, j1] + integral[i1, j1])

            # 计算平均值
            filtered[i, j] = sum_val / area

    return filtered.astype(np.uint8)


# 测试不同核大小的均值滤波
print("测试手动均值滤波（不同核大小）:")

kernel_sizes = [3, 5, 7, 9]
results_manual = []
results_time = []

for ksize in kernel_sizes:
    print(f"\n滤波核大小: {ksize}×{ksize}")

    start_time = time.time()
    filtered = manual_mean_filter(gaussian_noisy, ksize, 'replicate')
    end_time = time.time()

    results_manual.append(filtered)
    results_time.append(end_time - start_time)

    print(f"  计算时间: {results_time[-1]:.4f}秒")
    print(f"  噪声减少: {np.std(gaussian_noisy.astype(np.float32) - clean_img.astype(np.float32)):.1f} → "
          f"{np.std(filtered.astype(np.float32) - clean_img.astype(np.float32)):.1f}")

# 显示手动滤波结果
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# 第一行：原始和噪声图片
axes[0, 0].imshow(clean_img, cmap='gray')
axes[0, 0].set_title("原始图片")
axes[0, 0].axis('off')

axes[0, 1].imshow(gaussian_noisy, cmap='gray')
axes[0, 1].set_title("高斯噪声图片")
axes[0, 1].axis('off')

axes[0, 2].imshow(salt_pepper_noisy, cmap='gray')
axes[0, 2].set_title("椒盐噪声图片")
axes[0, 2].axis('off')

# 第二行：不同核大小滤波结果
for idx, (ksize, img) in enumerate(zip(kernel_sizes[:3], results_manual[:3])):
    axes[1, idx].imshow(img, cmap='gray')
    axes[1, idx].set_title(f"手动均值滤波 {ksize}×{ksize}")
    axes[1, idx].axis('off')

# 显示性能信息
axes[1, 2].axis('off')
axes[1, 2].text(0.1, 0.5,
                "手动均值滤波性能:\n\n"
                f"3×3: {results_time[0]:.4f}秒\n"
                f"5×5: {results_time[1]:.4f}秒\n"
                f"7×7: {results_time[2]:.4f}秒\n"
                f"9×9: {results_time[3]:.4f}秒\n\n"
                "注意: 随着核大小增加,\n"
                "计算时间平方增长",
                fontsize=10, verticalalignment='center')

plt.suptitle("手动均值滤波效果（高斯噪声）", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 5. OpenCV实现均值滤波 ====================
print("\n🔧 5. OpenCV实现均值滤波")
print("=" * 30)


def demonstrate_opencv_mean_filter():
    """演示OpenCV均值滤波"""

    print("OpenCV提供两种均值滤波函数:")
    print("1. cv2.blur(): 标准均值滤波")
    print("2. cv2.boxFilter(): 可控制归一化的方框滤波")
    print()

    # 测试不同函数
    kernel_size = (5, 5)
    kernel_area = kernel_size[0] * kernel_size[1]  # 5×5=25

    # 1. 使用cv2.blur
    start_time = time.time()
    blur_result = cv2.blur(gaussian_noisy, kernel_size)
    blur_time = time.time() - start_time

    # 2. 使用cv2.boxFilter（默认归一化）
    start_time = time.time()
    box_result = cv2.boxFilter(gaussian_noisy, -1, kernel_size, normalize=True)
    box_time = time.time() - start_time

    # 3. 使用cv2.boxFilter（不归一化）- 修正版本
    # 注意：必须使用更大的数据类型或浮点类型，否则会溢出
    start_time = time.time()

    # 方法1：使用浮点类型（推荐，避免溢出）
    gaussian_float = gaussian_noisy.astype(np.float32)
    box_no_norm_float = cv2.boxFilter(gaussian_float, cv2.CV_32F, kernel_size, normalize=False)
    box_no_norm = np.clip(box_no_norm_float / kernel_area, 0, 255).astype(np.uint8)

    # 方法2：使用更大的整数类型
    # box_no_norm_uint16 = cv2.boxFilter(
    #     gaussian_noisy.astype(np.uint16),
    #     cv2.CV_16U,
    #     kernel_size,
    #     normalize=False
    # )
    # box_no_norm = np.clip(box_no_norm_uint16 / kernel_area, 0, 255).astype(np.uint8)

    box_no_norm_time = time.time() - start_time

    print(f"滤波核大小: {kernel_size[0]}×{kernel_size[1]} (面积={kernel_area})")
    print(f"cv2.blur 计算时间: {blur_time:.6f}秒")
    print(f"cv2.boxFilter(归一化) 计算时间: {box_time:.6f}秒")
    print(f"cv2.boxFilter(不归一化) 计算时间: {box_no_norm_time:.6f}秒")
    print()

    # 显示像素值范围
    print("各方法结果像素范围:")
    print(f"  cv2.blur: [{blur_result.min()}, {blur_result.max()}]")
    print(f"  cv2.boxFilter(归一化): [{box_result.min()}, {box_result.max()}]")
    print(f"  cv2.boxFilter(不归一化+手动归一化): [{box_no_norm.min()}, {box_no_norm.max()}]")
    print()

    # 比较结果差异
    # 将结果转换为浮点型以避免整数溢出
    blur_float = blur_result.astype(np.float32)
    box_float = box_result.astype(np.float32)
    box_no_norm_float_result = box_no_norm.astype(np.float32)

    # 计算绝对差异
    diff_blur_box = np.sum(np.abs(blur_float - box_float))
    diff_blur_manual = np.sum(np.abs(blur_float - box_no_norm_float_result))

    print("各方法间差异统计:")
    print(f"  cv2.blur 与 cv2.boxFilter(归一化) 差异总和: {diff_blur_box:.2f}")
    print(f"  cv2.blur 与 cv2.boxFilter(不归一化) 差异总和: {diff_blur_manual:.2f}")

    # 计算平均像素差异
    num_pixels = blur_result.shape[0] * blur_result.shape[1]
    avg_diff_blur_box = diff_blur_box / num_pixels
    avg_diff_blur_manual = diff_blur_manual / num_pixels

    print(f"  cv2.blur 与 cv2.boxFilter(归一化) 平均像素差异: {avg_diff_blur_box:.6f}")
    print(f"  cv2.blur 与 cv2.boxFilter(不归一化) 平均像素差异: {avg_diff_blur_manual:.6f}")
    print()

    # 检查是否完全相同
    if np.array_equal(blur_result, box_result):
        print("✅ cv2.blur 和 cv2.boxFilter(归一化) 结果完全相同")
    else:
        print("⚠️  cv2.blur 和 cv2.boxFilter(归一化) 结果有微小差异")

    if np.array_equal(blur_result, box_no_norm):
        print("✅ cv2.blur 和 cv2.boxFilter(不归一化+手动归一化) 结果完全相同")
    else:
        print("⚠️  cv2.blur 和 cv2.boxFilter(不归一化+手动归一化) 结果有微小差异")

    # 显示结果
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 第一行
    axes[0, 0].imshow(gaussian_noisy, cmap='gray')
    axes[0, 0].set_title("高斯噪声图片")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(blur_result, cmap='gray')
    axes[0, 1].set_title(f"cv2.blur\n{kernel_size[0]}×{kernel_size[1]}")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(box_result, cmap='gray')
    axes[0, 2].set_title(f"cv2.boxFilter(归一化)\n{kernel_size[0]}×{kernel_size[1]}")
    axes[0, 2].axis('off')

    # 第二行
    axes[1, 0].imshow(box_no_norm, cmap='gray')
    axes[1, 0].set_title(f"cv2.boxFilter(不归一化)\n{kernel_size[0]}×{kernel_size[1]}")
    axes[1, 0].axis('off')

    # 显示差异
    diff_img = np.abs(blur_result.astype(np.float32) - box_result.astype(np.float32))
    axes[1, 1].imshow(diff_img, cmap='hot')
    axes[1, 1].set_title("差异图 (blur vs boxFilter)")
    axes[1, 1].axis('off')

    # 显示性能比较
    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.5,
                    "OpenCV均值滤波对比:\n\n"
                    "cv2.blur():\n"
                    "  - 标准均值滤波\n"
                    "  - 自动归一化\n"
                    "  - 使用方便\n\n"
                    "cv2.boxFilter():\n"
                    "  - 可控制归一化\n"
                    "  - 更灵活\n"
                    "  - 可用于非归一化滤波\n\n"
                    "性能差异很小，\n"
                    "通常使用cv2.blur()即可",
                    fontsize=10, verticalalignment='center')

    plt.suptitle("OpenCV均值滤波实现对比", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return blur_result, box_result, box_no_norm


# 演示OpenCV实现
blur_result, box_result, box_no_norm = demonstrate_opencv_mean_filter()

# ==================== 6. 不同边界处理对比 ====================
print("\n🔍 6. 不同边界处理对比")
print("=" * 30)


def demonstrate_border_handling_mean():
    """演示均值滤波的不同边界处理"""

    # 创建一个小测试图片
    test_img = np.array([
        [10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50]
    ], dtype=np.uint8)

    kernel_size = (3, 3)

    # 不同边界处理 - 注意：cv2.blur不支持BORDER_WRAP
    border_types = [
        (cv2.BORDER_CONSTANT, "常数填充 (0)"),
        (cv2.BORDER_REPLICATE, "复制填充"),
        (cv2.BORDER_REFLECT, "反射填充"),
        (cv2.BORDER_REFLECT_101, "反射填充101"),
        # (cv2.BORDER_WRAP, "循环填充"),  # 移除，因为cv2.blur不支持
    ]

    results = []

    print("不同边界处理方法对比 (5×5图片, 3×3均值滤波):")
    print("-" * 50)
    print("注意: cv2.blur()不支持BORDER_WRAP边界类型")
    print()

    for border_type, border_name in border_types:
        # 使用filter2D来演示，它可以支持更多边界类型
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        filtered = cv2.filter2D(test_img, -1, kernel, borderType=border_type)

        results.append((border_name, filtered))

        print(f"\n{border_name}:")
        print(filtered)

    # 可视化结果
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 原始图片
    axes[0, 0].imshow(test_img, cmap='gray')
    axes[0, 0].set_title("原始图片")
    axes[0, 0].grid(True, which='both', color='red', linestyle='-', linewidth=0.5)
    axes[0, 0].set_xticks(range(5))
    axes[0, 0].set_yticks(range(5))

    for i, (border_name, filtered) in enumerate(results):
        row = (i + 1) // 3
        col = (i + 1) % 3
        axes[row, col].imshow(filtered, cmap='gray')
        axes[row, col].set_title(border_name, fontsize=10)
        axes[row, col].grid(True, which='both', color='red', linestyle='-', linewidth=0.5)
        axes[row, col].set_xticks(range(5))
        axes[row, col].set_yticks(range(5))

        # 在图中显示数值
        for y in range(5):
            for x in range(5):
                axes[row, col].text(x, y, f'{filtered[y, x]:.0f}',
                                    ha='center', va='center',
                                    color='white' if filtered[y, x] < 25 else 'black',
                                    fontsize=8)

    # 显示边界处理说明
    axes[1, 0].axis('off')
    axes[1, 0].text(0.1, 0.5,
                    "边界处理说明:\n\n"
                    "BORDER_CONSTANT: 用0填充边界\n"
                    "BORDER_REPLICATE: 复制边缘像素\n"
                    "BORDER_REFLECT: 镜像反射边界\n"
                    "BORDER_REFLECT_101: 改进的镜像反射\n"
                    "BORDER_WRAP: cv2.blur不支持",
                    fontsize=9, verticalalignment='center')

    plt.suptitle("均值滤波的不同边界处理方法", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return test_img, results

# 演示边界处理
test_img_border, border_results = demonstrate_border_handling_mean()

# ==================== 7. 均值滤波性能分析 ====================
print("\n📈 7. 均值滤波性能分析")
print("=" * 30)


def analyze_mean_filter_performance():
    """分析均值滤波性能"""

    print("均值滤波性能分析:")
    print("=" * 40)

    # 测试不同核大小
    kernel_sizes = [3, 5, 7, 9, 11, 15, 21, 31]

    manual_times = []
    opencv_times = []
    noise_reductions = []

    for ksize in kernel_sizes:
        kernel = (ksize, ksize)

        # 1. 手动实现时间
        start_time = time.time()
        manual_result = manual_mean_filter(gaussian_noisy, ksize, 'replicate')
        manual_time = time.time() - start_time

        # 2. OpenCV实现时间
        start_time = time.time()
        opencv_result = cv2.blur(gaussian_noisy, kernel)
        opencv_time = time.time() - start_time

        # 3. 噪声减少效果
        original_noise = np.std(gaussian_noisy.astype(np.float32) - clean_img.astype(np.float32))
        manual_noise = np.std(manual_result.astype(np.float32) - clean_img.astype(np.float32))
        reduction = 100 * (original_noise - manual_noise) / original_noise

        manual_times.append(manual_time)
        opencv_times.append(opencv_time)
        noise_reductions.append(reduction)

        print(f"核大小 {ksize:2d}×{ksize:<2d}: "
              f"手动 {manual_time:.4f}s, OpenCV {opencv_time:.4f}s, "
              f"噪声减少 {reduction:.1f}%")

    # 性能可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. 计算时间对比
    axes[0, 0].plot(kernel_sizes, manual_times, 'b-o', label='手动实现', linewidth=2)
    axes[0, 0].plot(kernel_sizes, opencv_times, 'r-s', label='OpenCV实现', linewidth=2)
    axes[0, 0].set_xlabel('滤波核大小')
    axes[0, 0].set_ylabel('计算时间 (秒)')
    axes[0, 0].set_title('计算时间对比')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. 加速比
    speedup = [m / o if o > 0 else 0 for m, o in zip(manual_times, opencv_times)]
    axes[0, 1].bar(range(len(kernel_sizes)), speedup, color='green', alpha=0.7)
    axes[0, 1].set_xlabel('滤波核大小索引')
    axes[0, 1].set_ylabel('加速比 (手动/OpenCV)')
    axes[0, 1].set_title('OpenCV加速效果')
    axes[0, 1].set_xticks(range(len(kernel_sizes)))
    axes[0, 1].set_xticklabels([f'{k}×{k}' for k in kernel_sizes], rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. 噪声减少效果
    axes[1, 0].plot(kernel_sizes, noise_reductions, 'g-^', linewidth=2)
    axes[1, 0].set_xlabel('滤波核大小')
    axes[1, 0].set_ylabel('噪声减少百分比 (%)')
    axes[1, 0].set_title('噪声减少效果')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 时间与效果权衡
    axes[1, 1].scatter(opencv_times, noise_reductions, s=100, c='purple', alpha=0.6)
    for i, ksize in enumerate(kernel_sizes):
        axes[1, 1].annotate(f'{ksize}×{ksize}',
                            (opencv_times[i], noise_reductions[i]),
                            xytext=(5, 5), textcoords='offset points')
    axes[1, 1].set_xlabel('计算时间 (秒)')
    axes[1, 1].set_ylabel('噪声减少百分比 (%)')
    axes[1, 1].set_title('时间-效果权衡分析')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("均值滤波性能综合分析", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 分析结论
    print("\n" + "=" * 40)
    print("性能分析结论:")
    print("-" * 40)
    print(f"1. OpenCV比手动实现快 {np.mean(speedup):.1f}倍")
    print(f"2. 最佳核大小: 5×5 到 9×9 (权衡时间与效果)")
    print(f"3. 核大小 > 15 时，时间增长明显，效果提升有限")
    print(f"4. 噪声减少在 7×7 核时达到 {max(noise_reductions):.1f}%")

    return kernel_sizes, manual_times, opencv_times, noise_reductions


# 性能分析
kernel_sizes, manual_times, opencv_times, noise_reductions = analyze_mean_filter_performance()

# ==================== 8. 均值滤波的局限性 ====================
print("\n⚠️ 8. 均值滤波的局限性")
print("=" * 30)


def demonstrate_mean_filter_limitations():
    """演示均值滤波的局限性"""

    print("均值滤波的主要局限性:")
    print("1. 使图像模糊，损失边缘信息")
    print("2. 对椒盐噪声效果不佳")
    print("3. 大核会导致严重模糊")
    print("4. 对脉冲噪声敏感")
    print()

    # 测试不同场景下的局限性
    # 1. 边缘保持测试
    print("测试1: 边缘保持能力")
    edge_img = np.zeros((100, 100), dtype=np.uint8)
    edge_img[:, 50:] = 255  # 创建锐利边缘

    edge_blur = cv2.blur(edge_img, (15, 15))

    # 2. 椒盐噪声测试
    print("测试2: 椒盐噪声处理")
    salt_pepper_test = clean_img.copy()
    salt_pepper_test = add_salt_pepper_noise(salt_pepper_test, 0.05, 0.05)
    salt_pepper_blur = cv2.blur(salt_pepper_test, (5, 5))

    # 3. 细节损失测试
    print("测试3: 细节损失")
    detail_img = clean_img.copy()
    detail_blur = cv2.blur(detail_img, (9, 9))

    # 显示局限性
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    # 第一行：边缘测试
    axes[0, 0].imshow(edge_img, cmap='gray')
    axes[0, 0].set_title("原始边缘")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(edge_blur, cmap='gray')
    axes[0, 1].set_title("均值滤波后 (15×15)")
    axes[0, 1].axis('off')

    # 边缘剖面
    edge_profile_original = edge_img[50, :]
    edge_profile_blur = edge_blur[50, :]
    axes[0, 2].plot(edge_profile_original, 'b-', label='原始', linewidth=2)
    axes[0, 2].plot(edge_profile_blur, 'r-', label='滤波后', linewidth=2)
    axes[0, 2].set_title("边缘剖面")
    axes[0, 2].set_xlabel('X位置')
    axes[0, 2].set_ylabel('灰度值')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 第二行：椒盐噪声测试
    axes[1, 0].imshow(salt_pepper_test, cmap='gray')
    axes[1, 0].set_title("椒盐噪声 (5%)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(salt_pepper_blur, cmap='gray')
    axes[1, 1].set_title("均值滤波后 (5×5)")
    axes[1, 1].axis('off')

    # 显示局部放大
    zoom_region = salt_pepper_test[80:120, 120:160]
    zoom_blur = salt_pepper_blur[80:120, 120:160]

    axes[1, 2].imshow(np.hstack([zoom_region, zoom_blur]), cmap='gray')
    axes[1, 2].set_title("局部放大对比")
    axes[1, 2].axis('off')
    axes[1, 2].axvline(x=40, color='red', linestyle='--', linewidth=2)

    # 第三行：细节损失测试
    axes[2, 0].imshow(detail_img, cmap='gray')
    axes[2, 0].set_title("原始细节")
    axes[2, 0].axis('off')

    axes[2, 1].imshow(detail_blur, cmap='gray')
    axes[2, 1].set_title("均值滤波后 (9×9)")
    axes[2, 1].axis('off')

    # 显示局限性总结
    axes[2, 2].axis('off')
    axes[2, 2].text(0.1, 0.5,
                    "均值滤波局限性总结:\n\n"
                    "1. 边缘模糊:\n"
                    "   锐利边缘变模糊\n"
                    "   边缘定位不准确\n\n"
                    "2. 椒盐噪声:\n"
                    "   只能扩散，不能去除\n"
                    "   黑白点变成灰色斑点\n\n"
                    "3. 细节损失:\n"
                    "   小细节被平滑掉\n"
                    "   纹理信息丢失\n\n"
                    "4. 应用建议:\n"
                    "   适用于高斯噪声\n"
                    "   不适用于需要保持\n"
                    "   边缘和细节的场景",
                    fontsize=9, verticalalignment='center')

    plt.suptitle("均值滤波的局限性分析", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return edge_img, edge_blur, salt_pepper_test, salt_pepper_blur


# 演示局限性
edge_img, edge_blur, salt_test, salt_blur = demonstrate_mean_filter_limitations()

# ==================== 9. 实际应用案例 ====================
print("\n💼 9. 实际应用案例")
print("=" * 30)


def demonstrate_real_world_applications():
    """演示均值滤波在实际中的应用"""

    print("均值滤波的实际应用场景:")
    print("1. 图像预处理: 为后续处理减少噪声")
    print("2. 简单去噪: 快速去除轻微噪声")
    print("3. 图像模糊: 创建艺术效果")
    print("4. 降采样预处理: 减少锯齿效应")
    print()

    # 模拟不同应用场景
    applications = [
        ("图像预处理", "preprocess"),
        ("简单去噪", "denoise"),
        ("艺术模糊", "artistic"),
        ("降采样预处理", "downsample")
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for idx, (app_name, app_type) in enumerate(applications):
        row = idx // 2
        col = (idx % 2) * 2

        if app_type == "preprocess":
            # 图像预处理：边缘检测前的去噪
            original = gaussian_noisy.copy()
            processed = cv2.blur(original, (3, 3))
            # 边缘检测对比
            edges_original = cv2.Canny(original, 50, 150)
            edges_processed = cv2.Canny(processed, 50, 150)

            axes[row, col].imshow(original, cmap='gray')
            axes[row, col].set_title(f"{app_name}\n原始")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(processed, cmap='gray')
            axes[row, col + 1].set_title(f"{app_name}\n均值滤波后")
            axes[row, col + 1].axis('off')

        elif app_type == "denoise":
            # 简单去噪
            original = gaussian_noisy.copy()
            processed = cv2.blur(original, (5, 5))

            axes[row, col].imshow(original, cmap='gray')
            axes[row, col].set_title(f"{app_name}\n噪声图片")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(processed, cmap='gray')
            axes[row, col + 1].set_title(f"{app_name}\n去噪后")
            axes[row, col + 1].axis('off')

        elif app_type == "artistic":
            # 艺术模糊
            original = clean_img.copy()
            processed = cv2.blur(original, (15, 15))

            axes[row, col].imshow(original, cmap='gray')
            axes[row, col].set_title(f"{app_name}\n原始")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(processed, cmap='gray')
            axes[row, col + 1].set_title(f"{app_name}\n艺术模糊")
            axes[row, col + 1].axis('off')

        elif app_type == "downsample":
            # 降采样预处理
            original = clean_img.copy()
            # 先模糊再降采样
            blurred = cv2.blur(original, (3, 3))
            downsampled = cv2.resize(blurred, (0, 0), fx=0.5, fy=0.5)
            # 直接降采样（不模糊）
            direct_down = cv2.resize(original, (0, 0), fx=0.5, fy=0.5)

            axes[row, col].imshow(direct_down, cmap='gray')
            axes[row, col].set_title(f"{app_name}\n直接降采样")
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(downsampled, cmap='gray')
            axes[row, col + 1].set_title(f"{app_name}\n先模糊后降采样")
            axes[row, col + 1].axis('off')

    plt.suptitle("均值滤波在实际场景中的应用", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    # 应用建议
    print("\n应用建议:")
    print("-" * 30)
    print("1. 预处理: 使用小核 (3×3 或 5×5)")
    print("2. 简单去噪: 根据噪声水平选择核大小")
    print("3. 艺术效果: 使用大核创造模糊效果")
    print("4. 实时处理: 均值滤波计算快，适合实时应用")
    print("5. 注意: 避免过度模糊，损失重要信息")


# 演示实际应用
demonstrate_real_world_applications()

# ==================== 10. 练习与挑战 ====================
print("\n💪 10. 练习与挑战")
print("=" * 30)

print("""
练习题：

1. 基础练习：
   a) 实现一个函数，可以对彩色图片进行均值滤波
   b) 比较不同边界处理对滤波结果的影响
   c) 实现可分离均值滤波（先水平后垂直）

2. 进阶练习：
   a) 实现自适应均值滤波，根据局部噪声水平调整核大小
   b) 实现加权均值滤波（中心权重更高）
   c) 比较均值滤波与后续将学的高斯滤波的区别

3. 思考题：
   a) 为什么均值滤波会使图像变模糊？
   b) 如何选择最佳的滤波核大小？
   c) 在什么情况下应该使用均值滤波？
   d) 均值滤波的时间复杂度是多少？
""")

# 练习框架代码
print("\n💻 练习框架代码：")

print("""
# 练习1a: 彩色图片均值滤波
def mean_filter_color(image, kernel_size=3):
    # 分离通道
    b, g, r = cv2.split(image)

    # 对每个通道分别滤波
    b_filtered = cv2.blur(b, (kernel_size, kernel_size))
    g_filtered = cv2.blur(g, (kernel_size, kernel_size))
    r_filtered = cv2.blur(r, (kernel_size, kernel_size))

    # 合并通道
    filtered = cv2.merge([b_filtered, g_filtered, r_filtered])
    return filtered

# 练习1c: 可分离均值滤波
def separable_mean_filter(image, kernel_size=3):
    # 可分离滤波：先水平后垂直
    # 创建1D核
    kernel_1d = np.ones(kernel_size, dtype=np.float32) / kernel_size

    # 水平滤波
    horizontal = cv2.filter2D(image, -1, kernel_1d.reshape(1, -1))

    # 垂直滤波
    filtered = cv2.filter2D(horizontal, -1, kernel_1d.reshape(-1, 1))
    return filtered

# 练习2a: 自适应均值滤波
def adaptive_mean_filter(image, min_size=3, max_size=11, noise_threshold=20):
    # 根据局部噪声水平自适应选择核大小
    height, width = image.shape
    filtered = np.zeros_like(image, dtype=np.float32)

    for i in range(height):
        for j in range(width):
            # 计算局部噪声水平
            local_region = image[max(0, i-1):min(height, i+2), 
                                 max(0, j-1):min(width, j+2)]
            local_std = np.std(local_region)

            # 根据噪声水平选择核大小
            if local_std > noise_threshold * 2:
                ksize = max_size
            elif local_std > noise_threshold:
                ksize = (min_size + max_size) // 2
            else:
                ksize = min_size

            # 确保ksize为奇数
            ksize = ksize if ksize % 2 == 1 else ksize + 1

            # 应用均值滤波
            pad = ksize // 2
            region = image[max(0, i-pad):min(height, i+pad+1), 
                          max(0, j-pad):min(width, j+pad+1)]
            filtered[i, j] = np.mean(region)

    return filtered.astype(np.uint8)
""")

# ==================== 11. 总结 ====================
print("\n" + "=" * 50)
print("✅ 均值滤波总结")
print("=" * 50)

summary = """
📊 均值滤波核心知识：

1. 数学原理
   - 公式：I'(x,y) = (1/(M×N)) × ΣΣ I(x+i, y+j)
   - 卷积核：所有元素为1/(M×N)
   - 线性操作：满足叠加性和齐次性

2. 实现方法
   - 手动实现：双重循环计算局部平均
   - OpenCV实现：cv2.blur() 或 cv2.boxFilter()
   - 优化方法：积分图加速、可分离滤波

3. 参数选择
   - 核大小：通常3×3, 5×5, 7×7（奇数）
   - 边界处理：补零、复制、反射、循环
   - 应用场景：根据噪声水平和细节要求选择

4. 性能特点
   - 时间复杂度：O(N²×M²) 原始，可优化到O(N²)
   - 空间复杂度：O(1) 额外空间
   - 优点：简单、快速、线性
   - 缺点：模糊边缘、对椒盐噪声效果差

5. 实际应用
   - 图像预处理
   - 简单去噪
   - 艺术模糊效果
   - 降采样预处理

6. 最佳实践
   - 小核用于预处理 (3×3)
   - 中核用于一般去噪 (5×5, 7×7)
   - 大核用于艺术效果 (>9×9)
   - 避免过度模糊重要细节

🎯 核心代码记忆：
   # OpenCV实现
   blurred = cv2.blur(image, (ksize, ksize))

   # 手动实现
   def mean_filter_manual(image, ksize=3):
       height, width = image.shape
       pad = ksize // 2
       filtered = np.zeros_like(image)
       for i in range(pad, height-pad):
           for j in range(pad, width-pad):
               region = image[i-pad:i+pad+1, j-pad:j+pad+1]
               filtered[i, j] = np.mean(region)
       return filtered
"""

print(summary)
print("\n📁 下一个文件: 05_03_高斯滤波实现.py")
print("  我们将学习更优秀的高斯滤波！")