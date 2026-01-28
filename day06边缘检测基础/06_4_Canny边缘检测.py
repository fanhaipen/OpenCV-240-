"""
第6天 - 文件4：Canny边缘检测
学习目标：掌握Canny边缘检测算法的原理、实现和应用
重点：高斯滤波、梯度计算、非极大值抑制、双阈值检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

print("🔍 第6天 - 文件4：Canny边缘检测")
print("=" * 50)

# ==================== 1. Canny算法理论 ====================
print("\n🎯 1. Canny边缘检测算法理论")
print("=" * 30)

print("""
Canny边缘检测算法：

基本概念：
  - 由John Canny于1986年提出
  - 目前最经典的边缘检测算法之一
  - 多阶段算法，结合多种技术
  - 目标是实现最优的边缘检测

算法步骤：
  1. 高斯滤波：降低噪声影响
  2. 计算梯度：使用Sobel算子计算梯度幅值和方向
  3. 非极大值抑制：细化边缘，只保留局部最大值
  4. 双阈值检测：使用高低阈值连接边缘
  5. 边缘连接：通过滞后阈值连接边缘

Canny算法的三个评价标准：
  1. 低错误率：尽可能少地检测非边缘点
  2. 高定位精度：检测到的边缘点应该与实际边缘点尽可能接近
  3. 单边缘响应：对单一边缘只产生单一边缘响应

数学原理：
  - 高斯滤波: G(x,y) = (1/(2πσ²)) * exp(-(x²+y²)/(2σ²))
  - 梯度计算: 使用Sobel算子
  - 非极大值抑制: 比较梯度方向上的相邻像素
  - 双阈值: 高阈值T_high, 低阈值T_low

参数说明：
  - sigma: 高斯滤波的标准差，控制平滑程度
  - low_threshold: 低阈值，用于弱边缘检测
  - high_threshold: 高阈值，用于强边缘检测

优点：
  - 抗噪声能力强
  - 边缘定位精确
  - 单边缘响应
  - 参数可调，适应不同场景

缺点：
  - 计算复杂度较高
  - 需要手动调整参数
  - 对纹理复杂图像可能产生过多边缘

应用场景：
  - 高质量边缘检测
  - 计算机视觉预处理
  - 图像分割
  - 目标检测
  - 特征提取
""")

# ==================== 2. Canny算法步骤详解 ====================
print("\n🔧 2. Canny算法步骤详解")
print("=" * 30)


def demonstrate_canny_steps():
    """详细演示Canny算法的每个步骤"""

    print("Canny算法详细步骤:")
    print("=" * 40)

    # 创建测试图片
    test_img = np.zeros((150, 200), dtype=np.uint8)

    # 添加各种边缘
    # 矩形
    cv2.rectangle(test_img, (30, 30), (100, 100), 150, -1)
    # 圆形
    cv2.circle(test_img, (150, 80), 30, 200, -1)
    # 斜线
    cv2.line(test_img, (20, 120), (180, 140), 180, 2)
    # 文字
    cv2.putText(test_img, "CANNY", (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 220, 1)

    # 添加高斯噪声
    noise = np.random.normal(0, 20, test_img.shape)
    noisy_img = np.clip(test_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    print("步骤1: 高斯滤波")
    print("-" * 20)

    # 1. 高斯滤波
    sigma = 1.4
    ksize = int(6 * sigma) + 1
    if ksize % 2 == 0:
        ksize += 1

    blurred = cv2.GaussianBlur(noisy_img, (ksize, ksize), sigma)
    print(f"高斯滤波参数: sigma={sigma}, 核大小={ksize}x{ksize}")
    print(f"目标: 减少噪声，同时保留边缘信息")
    print()

    print("步骤2: 计算梯度")
    print("-" * 20)

    # 2. 计算梯度（使用Sobel算子）
    grad_x = cv2.Sobel(blurred.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值和方向
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_direction = np.arctan2(grad_y, grad_x) * 180 / np.pi  # 转换为角度

    # 将方向归一化到0-180度
    gradient_direction = np.mod(gradient_direction, 180)

    print(f"梯度计算完成")
    print(f"梯度幅值范围: {np.min(gradient_magnitude):.2f} - {np.max(gradient_magnitude):.2f}")
    print(f"梯度方向范围: 0° - 180°")
    print()

    print("步骤3: 非极大值抑制")
    print("-" * 20)

    # 3. 非极大值抑制
    def non_maximum_suppression(magnitude, direction):
        """非极大值抑制实现"""
        height, width = magnitude.shape
        suppressed = np.zeros_like(magnitude)

        # 将方向量化为4个方向: 0°, 45°, 90°, 135°
        quantized_direction = np.zeros_like(direction, dtype=np.int32)
        quantized_direction[(0 <= direction) & (direction < 22.5)] = 0  # 0°
        quantized_direction[(157.5 <= direction) & (direction <= 180)] = 0  # 0°
        quantized_direction[(22.5 <= direction) & (direction < 67.5)] = 45  # 45°
        quantized_direction[(67.5 <= direction) & (direction < 112.5)] = 90  # 90°
        quantized_direction[(112.5 <= direction) & (direction < 157.5)] = 135  # 135°

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                dir_val = quantized_direction[i, j]
                mag_val = magnitude[i, j]

                # 根据方向比较相邻像素
                if dir_val == 0:  # 水平方向
                    neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
                elif dir_val == 45:  # 45°方向
                    neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]
                elif dir_val == 90:  # 垂直方向
                    neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
                elif dir_val == 135:  # 135°方向
                    neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]

                # 如果是局部最大值，则保留
                if mag_val >= max(neighbors):
                    suppressed[i, j] = mag_val

        return suppressed

    nms_result = non_maximum_suppression(gradient_magnitude, gradient_direction)
    print("非极大值抑制完成")
    print(f"抑制后非零像素比例: {np.sum(nms_result > 0) / nms_result.size * 100:.2f}%")
    print()

    print("步骤4: 双阈值检测")
    print("-" * 20)

    # 4. 双阈值检测
    def double_threshold(image, low_ratio=0.1, high_ratio=0.3):
        """双阈值检测实现"""
        # 计算高低阈值
        high_threshold = np.max(image) * high_ratio
        low_threshold = high_threshold * low_ratio

        # 创建结果图像
        result = np.zeros_like(image, dtype=np.uint8)

        # 强边缘
        strong_edges = (image >= high_threshold)
        # 弱边缘
        weak_edges = (image >= low_threshold) & (image < high_threshold)

        result[strong_edges] = 255  # 强边缘
        result[weak_edges] = 127  # 弱边缘

        return result, high_threshold, low_threshold

    # 应用双阈值
    high_ratio = 0.2
    low_ratio = 0.1
    threshold_result, high_thresh, low_thresh = double_threshold(nms_result, low_ratio, high_ratio)

    print(f"双阈值参数: 高阈值={high_thresh:.2f}, 低阈值={low_thresh:.2f}")
    print(f"强边缘像素数: {np.sum(threshold_result == 255)}")
    print(f"弱边缘像素数: {np.sum(threshold_result == 127)}")
    print()

    print("步骤5: 边缘连接")
    print("-" * 20)

    # 5. 边缘连接（滞后阈值）
    def edge_tracking_by_hysteresis(threshold_image):
        """边缘连接实现"""
        height, width = threshold_image.shape
        result = np.zeros((height, width), dtype=np.uint8)

        # 标记强边缘
        strong_edges = (threshold_image == 255)
        result[strong_edges] = 255

        # 8邻域连接弱边缘
        visited = np.zeros_like(threshold_image, dtype=bool)

        def connect_weak_edges(i, j):
            """递归连接弱边缘"""
            if i < 0 or i >= height or j < 0 or j >= width:
                return
            if visited[i, j]:
                return

            visited[i, j] = True

            # 如果是弱边缘且与强边缘相连，则标记为强边缘
            if threshold_image[i, j] == 127:
                # 检查8邻域是否有强边缘
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < height and 0 <= nj < width and
                                result[ni, nj] == 255):
                            result[i, j] = 255
                            # 继续检查相邻的弱边缘
                            for ddi in [-1, 0, 1]:
                                for ddj in [-1, 0, 1]:
                                    connect_weak_edges(i + ddi, j + ddj)
                            break

        # 遍历所有像素，连接弱边缘
        for i in range(height):
            for j in range(width):
                if threshold_image[i, j] == 127 and not visited[i, j]:
                    connect_weak_edges(i, j)

        return result

    final_edges = edge_tracking_by_hysteresis(threshold_result)
    print("边缘连接完成")
    print(f"最终边缘像素数: {np.sum(final_edges == 255)}")
    print()

    # 使用OpenCV的Canny函数作为对比
    opencv_canny = cv2.Canny(noisy_img, low_thresh, high_thresh)

    # 可视化所有步骤
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 第一行：原始到梯度计算
    axes[0, 0].imshow(noisy_img, cmap='gray')
    axes[0, 0].set_title("1. 原始图片（含噪声）")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title("2. 高斯滤波后")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(gradient_magnitude, cmap='hot')
    axes[0, 2].set_title("3. 梯度幅值")
    axes[0, 2].axis('off')

    # 梯度方向可视化
    dir_vis = (gradient_direction / 180.0 * 255).astype(np.uint8)
    axes[0, 3].imshow(dir_vis, cmap='hsv')
    axes[0, 3].set_title("4. 梯度方向\n(HSV色彩空间)")
    axes[0, 3].axis('off')

    # 第二行：非极大值抑制到最终结果
    axes[1, 0].imshow(nms_result, cmap='hot')
    axes[1, 0].set_title("5. 非极大值抑制")
    axes[1, 0].axis('off')

    # 双阈值结果（用不同颜色显示强边缘和弱边缘）
    threshold_vis = np.zeros((threshold_result.shape[0], threshold_result.shape[1], 3), dtype=np.uint8)
    threshold_vis[threshold_result == 255] = [255, 0, 0]  # 强边缘：红色
    threshold_vis[threshold_result == 127] = [0, 0, 255]  # 弱边缘：蓝色
    axes[1, 1].imshow(threshold_vis)
    axes[1, 1].set_title("6. 双阈值检测\n(红:强边缘, 蓝:弱边缘)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(final_edges, cmap='gray')
    axes[1, 2].set_title("7. 边缘连接后")
    axes[1, 2].axis('off')

    axes[1, 3].imshow(opencv_canny, cmap='gray')
    axes[1, 3].set_title("8. OpenCV Canny")
    axes[1, 3].axis('off')

    plt.suptitle("Canny边缘检测算法步骤详解", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 统计信息
    print("Canny算法统计信息:")
    print("=" * 40)
    print(f"原始图片尺寸: {noisy_img.shape[1]}x{noisy_img.shape[0]}")
    print(f"高斯滤波核大小: {ksize}x{ksize}")
    print(f"梯度计算: Sobel 3x3")
    print(f"非极大值抑制: 保留局部最大值")
    print(f"双阈值: 高={high_thresh:.2f}, 低={low_thresh:.2f}")
    print(f"强边缘像素: {np.sum(threshold_result == 255)}")
    print(f"弱边缘像素: {np.sum(threshold_result == 127)}")
    print(f"最终边缘像素: {np.sum(final_edges == 255)}")
    print(f"OpenCV Canny边缘像素: {np.sum(opencv_canny == 255)}")
    print()

    return (noisy_img, blurred, gradient_magnitude, gradient_direction,
            nms_result, threshold_result, final_edges, opencv_canny,
            high_thresh, low_thresh)


# 演示Canny算法步骤
canny_steps = demonstrate_canny_steps()

# ==================== 3. 非极大值抑制详解 ====================
print("\n🎯 3. 非极大值抑制详解")
print("=" * 30)


def demonstrate_non_maximum_suppression():
    """详细演示非极大值抑制原理"""

    print("非极大值抑制原理:")
    print("=" * 40)

    print("""
非极大值抑制目的：
  - 细化边缘，使边缘宽度为1个像素
  - 只保留梯度方向上的局部最大值
  - 消除非极大值的边缘响应

实现步骤：
  1. 将梯度方向量化为4个方向: 0°, 45°, 90°, 135°
  2. 对于每个像素，检查其梯度方向上的两个相邻像素
  3. 如果当前像素的梯度值不是局部最大值，则抑制它
  4. 只保留局部最大值的像素

方向量化：
  - 0°: 水平方向，比较左右像素
  - 45°: 对角线方向，比较右上和左下像素
  - 90°: 垂直方向，比较上下像素
  - 135°: 对角线方向，比较左上和右下像素

数学表达：
  对于像素(i,j)，如果满足：
  magnitude(i,j) >= magnitude(相邻像素1) 且
  magnitude(i,j) >= magnitude(相邻像素2)
  则保留该像素，否则抑制
    """)

    # 创建简单的测试区域演示NMS
    demo_region = np.array([
        [10, 20, 30, 25, 15],
        [15, 80, 90, 85, 20],  # 中心行有局部最大值
        [20, 85, 100, 95, 25],  # 中心点100是局部最大值
        [18, 75, 88, 80, 22],
        [12, 25, 30, 28, 18]
    ], dtype=np.float32)

    # 假设梯度方向为90°（垂直方向）
    demo_direction = np.full_like(demo_region, 90)  # 所有方向都是90°

    def demo_nms(magnitude, direction):
        """演示用的简化NMS"""
        height, width = magnitude.shape
        suppressed = np.zeros_like(magnitude)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                dir_val = direction[i, j]
                mag_val = magnitude[i, j]

                # 简单演示：只考虑垂直方向
                if dir_val == 90:  # 垂直方向
                    up_neighbor = magnitude[i - 1, j]
                    down_neighbor = magnitude[i + 1, j]

                    if mag_val >= up_neighbor and mag_val >= down_neighbor:
                        suppressed[i, j] = mag_val
                else:
                    suppressed[i, j] = mag_val  # 其他方向暂时不处理

        return suppressed

    nms_demo = demo_nms(demo_region, demo_direction)

    # 可视化NMS过程
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 原始梯度幅值
    im1 = axes[0].imshow(demo_region, cmap='hot')
    axes[0].set_title("原始梯度幅值")
    axes[0].set_xticks(range(5))
    axes[0].set_yticks(range(5))
    plt.colorbar(im1, ax=axes[0])

    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, f'{demo_region[i, j]:.0f}',
                         ha='center', va='center',
                         color='white' if demo_region[i, j] > 50 else 'black',
                         fontweight='bold')

    # 梯度方向（假设都是90°）
    dir_vis = np.full_like(demo_region, 90)
    im2 = axes[1].imshow(dir_vis, cmap='hsv', vmin=0, vmax=180)
    axes[1].set_title("梯度方向\n(全部90°)")
    axes[1].set_xticks(range(5))
    axes[1].set_yticks(range(5))
    plt.colorbar(im2, ax=axes[1])

    for i in range(5):
        for j in range(5):
            axes[1].text(j, i, '90°',
                         ha='center', va='center',
                         color='white', fontweight='bold')

    # NMS结果
    im3 = axes[2].imshow(nms_demo, cmap='hot')
    axes[2].set_title("非极大值抑制后")
    axes[2].set_xticks(range(5))
    axes[2].set_yticks(range(5))
    plt.colorbar(im3, ax=axes[2])

    for i in range(5):
        for j in range(5):
            if nms_demo[i, j] > 0:
                axes[2].text(j, i, f'{nms_demo[i, j]:.0f}',
                             ha='center', va='center',
                             color='white' if nms_demo[i, j] > 50 else 'black',
                             fontweight='bold')
            else:
                axes[2].text(j, i, '0',
                             ha='center', va='center',
                             color='black', fontweight='bold')

    plt.suptitle("非极大值抑制原理演示", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    # 详细解释中心点[2,2]的NMS过程
    print("非极大值抑制计算示例 (中心点[2,2]):")
    print("=" * 50)

    center_value = demo_region[2, 2]  # 值100
    up_value = demo_region[1, 2]  # 值90
    down_value = demo_region[3, 2]  # 值88

    print(f"中心点[2,2]梯度值: {center_value}")
    print(f"上方像素[1,2]梯度值: {up_value}")
    print(f"下方像素[3,2]梯度值: {down_value}")
    print()

    print("比较过程:")
    print(f"中心值 >= 上方值: {center_value >= up_value}")
    print(f"中心值 >= 下方值: {center_value >= down_value}")
    print(f"两个条件都满足，因此保留中心点")
    print(f"非极大值抑制后值: {nms_demo[2, 2]}")
    print()

    # 演示不同方向的NMS
    print("不同方向的相邻像素比较:")
    print("=" * 40)

    directions = [0, 45, 90, 135]
    direction_names = ["0° (水平)", "45° (对角线)", "90° (垂直)", "135° (对角线)"]
    neighbor_positions = [
        ["左像素", "右像素"],
        ["右上像素", "左下像素"],
        ["上像素", "下像素"],
        ["左上像素", "右下像素"]
    ]

    for dir_val, dir_name, neighbors in zip(directions, direction_names, neighbor_positions):
        print(f"{dir_name}: 比较{neighbors[0]}和{neighbors[1]}")

    print()

    return demo_region, nms_demo


# 演示非极大值抑制
nms_demo = demonstrate_non_maximum_suppression()

# ==================== 4. 双阈值检测详解 ====================
print("\n🎯 4. 双阈值检测详解")
print("=" * 30)


def demonstrate_double_threshold():
    """详细演示双阈值检测原理"""

    print("双阈值检测原理:")
    print("=" * 40)

    print("""
双阈值检测目的：
  - 区分强边缘和弱边缘
  - 减少虚假边缘检测
  - 通过滞后阈值连接边缘

阈值选择原则：
  - 高阈值: 只保留确信度高的强边缘
  - 低阈值: 包含可能的弱边缘
  - 通常比例: 高阈值:低阈值 = 2:1 或 3:1

边缘分类：
  - 强边缘: 梯度值 >= 高阈值，确定是边缘
  - 弱边缘: 低阈值 <= 梯度值 < 高阈值，可能是边缘
  - 非边缘: 梯度值 < 低阈值，不是边缘

边缘连接规则：
  - 强边缘直接保留
  - 弱边缘只有在与强边缘相连时才保留
  - 孤立的弱边缘被抑制

优势：
  - 减少噪声引起的虚假边缘
  - 能够连接断裂的边缘
  - 提高边缘检测的连续性
    """)

    # 创建梯度幅值测试数据
    np.random.seed(42)  # 固定随机种子以便重现
    test_gradient = np.random.rand(8, 10) * 255

    # 手动设置一些明显的边缘
    test_gradient[2:6, 3:7] = 180  # 强边缘区域
    test_gradient[4, 5] = 220  # 最强点

    # 设置一些弱边缘
    test_gradient[1, 2] = 80
    test_gradient[6, 8] = 70

    # 应用双阈值
    high_threshold = 150
    low_threshold = 75

    def apply_double_threshold(gradient, high_thresh, low_thresh):
        """应用双阈值"""
        result = np.zeros_like(gradient, dtype=np.uint8)

        # 强边缘
        strong_edges = gradient >= high_thresh
        result[strong_edges] = 255

        # 弱边缘
        weak_edges = (gradient >= low_thresh) & (gradient < high_thresh)
        result[weak_edges] = 127

        return result, strong_edges, weak_edges

    threshold_result, strong_mask, weak_mask = apply_double_threshold(test_gradient, high_threshold, low_threshold)

    # 可视化双阈值检测
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 原始梯度幅值
    im1 = axes[0].imshow(test_gradient, cmap='hot')
    axes[0].set_title("梯度幅值分布")
    axes[0].set_xticks(range(10))
    axes[0].set_yticks(range(8))
    plt.colorbar(im1, ax=axes[0])

    for i in range(8):
        for j in range(10):
            color = 'white' if test_gradient[i, j] > 127 else 'black'
            axes[0].text(j, i, f'{test_gradient[i, j]:.0f}',
                         ha='center', va='center', color=color, fontweight='bold')

    # 双阈值结果
    axes[1].imshow(threshold_result, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("双阈值检测结果\n(白:强边缘, 灰:弱边缘, 黑:非边缘)")
    axes[1].set_xticks(range(10))
    axes[1].set_yticks(range(8))

    for i in range(8):
        for j in range(10):
            if threshold_result[i, j] == 255:
                axes[1].text(j, i, '强', ha='center', va='center',
                             color='red', fontweight='bold', fontsize=10)
            elif threshold_result[i, j] == 127:
                axes[1].text(j, i, '弱', ha='center', va='center',
                             color='blue', fontweight='bold', fontsize=10)

    # 边缘连接演示
    def demonstrate_edge_connection(threshold_image):
        """演示边缘连接"""
        height, width = threshold_image.shape
        result = np.zeros_like(threshold_image, dtype=np.uint8)

        # 复制强边缘
        result[threshold_image == 255] = 255

        # 简单边缘连接：如果弱边缘与强边缘相邻，则保留
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if threshold_image[i, j] == 127:
                    # 检查8邻域是否有强边缘
                    if np.any(threshold_image[i - 1:i + 2, j - 1:j + 2] == 255):
                        result[i, j] = 255

        return result

    connected_edges = demonstrate_edge_connection(threshold_result)

    axes[2].imshow(connected_edges, cmap='gray')
    axes[2].set_title("边缘连接后")
    axes[2].set_xticks(range(10))
    axes[2].set_yticks(range(8))

    for i in range(8):
        for j in range(10):
            if connected_edges[i, j] == 255:
                axes[2].text(j, i, '边', ha='center', va='center',
                             color='red', fontweight='bold', fontsize=10)

    plt.suptitle("双阈值检测与边缘连接原理演示", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    # 统计信息
    print("双阈值检测统计信息:")
    print("=" * 40)
    print(f"高阈值: {high_threshold}")
    print(f"低阈值: {low_threshold}")
    print(f"强边缘像素数: {np.sum(strong_mask)}")
    print(f"弱边缘像素数: {np.sum(weak_mask)}")
    print(f"边缘连接后像素数: {np.sum(connected_edges == 255)}")
    print()

    return test_gradient, threshold_result, connected_edges


# 演示双阈值检测
threshold_demo = demonstrate_double_threshold()

# ==================== 5. OpenCV Canny函数详解 ====================
print("\n🔧 5. OpenCV Canny函数详解")
print("=" * 30)


def demonstrate_opencv_canny():
    """详细演示OpenCV的Canny函数"""

    print("OpenCV Canny函数:")
    print("cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]])")
    print()
    print("参数说明:")
    print("  image: 输入图像 (8位灰度图)")
    print("  threshold1: 低阈值")
    print("  threshold2: 高阈值")
    print("  edges: 输出边缘图像 (可选)")
    print("  apertureSize: Sobel算子的孔径大小 (默认3)")
    print("  L2gradient: 是否使用L2范数计算梯度 (默认False, 使用L1)")
    print()

    # 创建测试图片
    test_img = np.zeros((150, 200), dtype=np.uint8)
    cv2.rectangle(test_img, (30, 30), (100, 100), 150, -1)
    cv2.circle(test_img, (150, 80), 30, 200, -1)

    # 添加噪声
    noise = np.random.normal(0, 20, test_img.shape)
    noisy_img = np.clip(test_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 测试不同参数
    test_cases = [
        # (low_thresh, high_thresh, apertureSize, L2gradient, description)
        (50, 150, 3, False, "默认参数 (50,150,3,L1)"),
        (30, 90, 3, False, "低阈值 (30,90)"),
        (100, 200, 3, False, "高阈值 (100,200)"),
        (50, 150, 5, False, "大孔径 (apertureSize=5)"),
        (50, 150, 3, True, "L2梯度计算"),
        (50, 150, 5, True, "大孔径+L2梯度"),
    ]

    results = []

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for idx, (low_thresh, high_thresh, aperture, l2grad, description) in enumerate(test_cases):
        row = idx // 3
        col = idx % 3

        # 应用Canny
        edges = cv2.Canny(noisy_img, low_thresh, high_thresh,
                          apertureSize=aperture, L2gradient=l2grad)
        results.append((description, edges))

        # 显示结果
        axes[row, col].imshow(edges, cmap='gray')
        axes[row, col].set_title(description)
        axes[row, col].axis('off')

        # 统计边缘像素
        edge_count = np.sum(edges == 255)
        axes[row, col].text(0.5, -0.1, f"边缘像素: {edge_count}",
                            transform=axes[row, col].transAxes,
                            ha='center', fontsize=9)

    plt.suptitle("OpenCV Canny函数不同参数效果", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 参数影响分析
    print("参数影响分析:")
    print("=" * 40)

    for description, edges in results:
        edge_count = np.sum(edges == 255)
        print(f"{description}:")
        print(f"  边缘像素数: {edge_count}")
        print(f"  边缘密度: {edge_count / edges.size * 100:.2f}%")
        print()

    return noisy_img, results


# 演示OpenCV Canny函数
opencv_canny_results = demonstrate_opencv_canny()

# ==================== 6. Canny算子参数调优 ====================
print("\n🔧 6. Canny算子参数调优")
print("=" * 30)


def demonstrate_canny_parameter_tuning():
    """演示Canny算子的参数调优"""

    print("Canny参数调优指南:")
    print("=" * 40)

    # 创建不同特征的测试图片
    def create_test_images():
        """创建不同特征的测试图片"""
        images = []

        # 1. 简单几何图形
        simple_img = np.zeros((150, 200), dtype=np.uint8)
        cv2.rectangle(simple_img, (30, 30), (100, 100), 150, -1)
        cv2.circle(simple_img, (150, 80), 30, 200, -1)
        images.append(("简单图形", simple_img))

        # 2. 复杂纹理
        texture_img = np.zeros((150, 200), dtype=np.uint8)
        for i in range(0, 150, 10):
            for j in range(0, 200, 10):
                cv2.rectangle(texture_img, (j, i), (j + 5, i + 5), 150, -1)
        cv2.circle(texture_img, (100, 75), 40, 200, -1)
        images.append(("复杂纹理", texture_img))

        # 3. 低对比度
        low_contrast = np.zeros((150, 200), dtype=np.uint8)
        cv2.rectangle(low_contrast, (30, 30), (100, 100), 80, -1)
        cv2.rectangle(low_contrast, (100, 30), (170, 100), 100, -1)
        images.append(("低对比度", low_contrast))

        return images

    test_images = create_test_images()

    # 定义不同的参数组合
    param_sets = [
        # (low_thresh, high_thresh, description)
        (20, 60, "低阈值 (敏感)"),
        (50, 150, "中等阈值 (平衡)"),
        (100, 200, "高阈值 (保守)"),
        (30, 200, "宽阈值范围"),
        (80, 120, "窄阈值范围"),
    ]

    # 对每个图片测试不同参数
    for img_name, test_img in test_images:
        print(f"\n测试图片: {img_name}")
        print("-" * 30)

        # 添加噪声
        noise = np.random.normal(0, 15, test_img.shape)
        noisy_img = np.clip(test_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        fig, axes = plt.subplots(1, len(param_sets) + 1, figsize=(15, 3))

        # 显示原始图片
        axes[0].imshow(noisy_img, cmap='gray')
        axes[0].set_title(f"{img_name}\n原始图片")
        axes[0].axis('off')

        # 测试不同参数
        for idx, (low_thresh, high_thresh, description) in enumerate(param_sets):
            edges = cv2.Canny(noisy_img, low_thresh, high_thresh)

            axes[idx + 1].imshow(edges, cmap='gray')
            axes[idx + 1].set_title(f"{description}\n({low_thresh},{high_thresh})")
            axes[idx + 1].axis('off')

            # 统计边缘像素
            edge_count = np.sum(edges == 255)
            edge_density = edge_count / edges.size * 100

            # 在子图下方显示统计信息
            axes[idx + 1].text(0.5, -0.15, f"{edge_count}像素\n{edge_density:.1f}%",
                               transform=axes[idx + 1].transAxes,
                               ha='center', fontsize=8)

            print(f"  参数{description}: 边缘像素={edge_count}, 密度={edge_density:.1f}%")

        plt.suptitle(f"Canny参数调优 - {img_name}", fontsize=16, y=1.1)
        plt.tight_layout()
        plt.show()

    # 自适应阈值方法
    print("\n自适应阈值方法:")
    print("=" * 40)

    def adaptive_canny_thresholds(image, sigma=0.33):
        """计算自适应阈值"""
        # 计算中值
        median = np.median(image)

        # 基于中值计算阈值
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))

        return lower, upper

    # 测试自适应阈值
    for img_name, test_img in test_images:
        noise = np.random.normal(0, 15, test_img.shape)
        noisy_img = np.clip(test_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 计算自适应阈值
        lower, upper = adaptive_canny_thresholds(noisy_img, sigma=0.33)

        # 应用Canny
        edges_adaptive = cv2.Canny(noisy_img, lower, upper)

        # 与固定阈值比较
        edges_fixed = cv2.Canny(noisy_img, 50, 150)

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        axes[0].imshow(noisy_img, cmap='gray')
        axes[0].set_title(f"{img_name}\n原始图片")
        axes[0].axis('off')

        axes[1].imshow(edges_fixed, cmap='gray')
        axes[1].set_title(f"固定阈值 (50,150)")
        axes[1].axis('off')

        axes[2].imshow(edges_adaptive, cmap='gray')
        axes[2].set_title(f"自适应阈值\n({lower},{upper})")
        axes[2].axis('off')

        # 统计信息
        fixed_count = np.sum(edges_fixed == 255)
        adaptive_count = np.sum(edges_adaptive == 255)

        axes[1].text(0.5, -0.15, f"{fixed_count}像素",
                     transform=axes[1].transAxes, ha='center', fontsize=9)
        axes[2].text(0.5, -0.15, f"{adaptive_count}像素",
                     transform=axes[2].transAxes, ha='center', fontsize=9)

        plt.suptitle(f"自适应阈值 vs 固定阈值 - {img_name}", fontsize=16, y=1.1)
        plt.tight_layout()
        plt.show()

        print(f"{img_name}: 固定阈值边缘数={fixed_count}, 自适应阈值边缘数={adaptive_count}")

    return test_images


# 演示参数调优
parameter_tuning_results = demonstrate_canny_parameter_tuning()

# ==================== 7. Canny算子的实际应用 ====================
print("\n💼 7. Canny算子的实际应用")
print("=" * 30)


def demonstrate_canny_applications():
    """演示Canny算子的实际应用"""

    print("Canny算子的实际应用场景:")
    print("1. 物体检测: 检测图像中的物体轮廓")
    print("2. 图像分割: 分割图像中的不同区域")
    print("3. 特征提取: 提取图像的特征点")
    print("4. 医学影像: 检测医学图像中的结构")
    print("5. 工业检测: 检测产品缺陷")
    print()

    # 应用1: 物体轮廓检测
    print("应用1: 物体轮廓检测")
    print("-" * 20)

    # 创建包含多个物体的测试图片
    object_img = np.zeros((200, 300), dtype=np.uint8)

    # 添加多个物体
    cv2.rectangle(object_img, (30, 30), (100, 100), 180, -1)  # 矩形
    cv2.circle(object_img, (200, 80), 40, 200, -1)  # 圆形
    cv2.ellipse(object_img, (120, 150), (60, 30), 0, 0, 360, 160, -1)  # 椭圆
    cv2.putText(object_img, "OBJECTS", (180, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 220, 2)

    # 添加噪声
    noise = np.random.normal(0, 15, object_img.shape)
    noisy_objects = np.clip(object_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 应用Canny
    edges = cv2.Canny(noisy_objects, 50, 150)

    # 轮廓查找
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    contour_img = cv2.cvtColor(noisy_objects, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

    # 应用2: 图像分割
    print("应用2: 图像分割")
    print("-" * 20)

    # 创建分割演示
    segmentation_img = np.zeros((150, 200), dtype=np.uint8)
    cv2.rectangle(segmentation_img, (20, 20), (90, 90), 100, -1)
    cv2.rectangle(segmentation_img, (110, 20), (180, 90), 200, -1)
    cv2.line(segmentation_img, (100, 0), (100, 150), 150, 3)  # 分割线

    # 应用Canny
    seg_edges = cv2.Canny(segmentation_img, 30, 100)

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 物体检测
    axes[0, 0].imshow(noisy_objects, cmap='gray')
    axes[0, 0].set_title("原始物体图片")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title("Canny边缘检测")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(contour_img)
    axes[0, 2].set_title("轮廓提取结果")
    axes[0, 2].axis('off')

    # 统计轮廓信息
    contour_info = f"检测到轮廓数: {len(contours)}\n"
    for i, cnt in enumerate(contours[:3]):  # 只显示前3个
        area = cv2.contourArea(cnt)
        contour_info += f"轮廓{i + 1}面积: {area:.1f}\n"
    axes[0, 2].text(0.5, -0.1, contour_info, transform=axes[0, 2].transAxes,
                    ha='center', fontsize=9)

    # 图像分割
    axes[1, 0].imshow(segmentation_img, cmap='gray')
    axes[1, 0].set_title("原始分割图片")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(seg_edges, cmap='gray')
    axes[1, 1].set_title("Canny边缘")
    axes[1, 1].axis('off')

    # 应用3: 边缘密度分析
    print("应用3: 边缘密度分析")
    print("-" * 20)

    # 计算边缘密度图
    def calculate_edge_density(edges, window_size=15):
        """计算边缘密度"""
        height, width = edges.shape
        density = np.zeros_like(edges, dtype=np.float32)

        pad = window_size // 2

        for i in range(pad, height - pad):
            for j in range(pad, width - pad):
                window = edges[i - pad:i + pad + 1, j - pad:j + pad + 1]
                density[i, j] = np.sum(window == 255) / (window_size ** 2)

        return density

    edge_density = calculate_edge_density(seg_edges, window_size=15)

    im = axes[1, 2].imshow(edge_density, cmap='hot')
    axes[1, 2].set_title("边缘密度图")
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle("Canny算子的实际应用", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 应用4: 不同噪声水平的鲁棒性测试
    print("应用4: 不同噪声水平的鲁棒性测试")
    print("-" * 20)

    noise_levels = [0, 10, 20, 30, 40, 50]
    edge_counts = []

    for noise_std in noise_levels:
        # 创建测试图片
        test_img = np.zeros((100, 150), dtype=np.uint8)
        cv2.rectangle(test_img, (30, 30), (120, 80), 200, -1)

        # 添加噪声
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, test_img.shape)
            noisy = np.clip(test_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        else:
            noisy = test_img

        # 应用Canny
        edges = cv2.Canny(noisy, 50, 150)
        edge_count = np.sum(edges == 255)
        edge_counts.append(edge_count)

        print(f"噪声标准差={noise_std}: 边缘像素数={edge_count}")

    # 绘制鲁棒性曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise_levels, edge_counts, 'bo-', linewidth=2, markersize=8)
    ax.set_title("Canny算子对不同噪声水平的鲁棒性")
    ax.set_xlabel("噪声标准差")
    ax.set_ylabel("边缘像素数")
    ax.grid(True, alpha=0.3)

    # 标记噪声水平
    for i, (x, y) in enumerate(zip(noise_levels, edge_counts)):
        ax.text(x, y + 50, f'{y}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    return object_img, edges, contour_img, segmentation_img, seg_edges, edge_density, edge_counts


# 演示实际应用
application_results = demonstrate_canny_applications()

# ==================== 8. Canny算子与其他算子对比 ====================
print("\n🔍 8. Canny算子与其他算子对比")
print("=" * 30)


def compare_canny_with_others():
    """比较Canny算子与其他边缘检测算子"""

    print("Canny vs 其他边缘检测算子:")
    print("=" * 40)

    # 创建测试图片
    test_img = np.zeros((150, 200), dtype=np.uint8)

    # 添加各种边缘
    cv2.rectangle(test_img, (30, 30), (100, 100), 150, -1)
    cv2.circle(test_img, (150, 80), 30, 200, -1)
    cv2.line(test_img, (20, 120), (180, 140), 180, 1)  # 细线

    # 添加噪声
    noise = np.random.normal(0, 20, test_img.shape)
    noisy_img = np.clip(test_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 应用不同算子
    operators = []

    # 1. Sobel算子
    sobel_x = cv2.Sobel(noisy_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(noisy_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_binary = (sobel_mag > 50).astype(np.uint8) * 255
    operators.append(("Sobel", sobel_binary))

    # 2. Laplacian算子
    laplacian = cv2.Laplacian(noisy_img, cv2.CV_64F, ksize=3)
    laplacian_binary = (np.abs(laplacian) > 30).astype(np.uint8) * 255
    operators.append(("Laplacian", laplacian_binary))

    # 3. LoG算子
    blurred = cv2.GaussianBlur(noisy_img, (5, 5), 1.0)
    log_result = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    log_binary = (np.abs(log_result) > 20).astype(np.uint8) * 255
    operators.append(("LoG", log_binary))

    # 4. Canny算子
    canny_edges = cv2.Canny(noisy_img, 50, 150)
    operators.append(("Canny", canny_edges))

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
        axes[row, col].imshow(result, cmap='gray')
        axes[row, col].set_title(f"{name}算子")
        axes[row, col].axis('off')

        # 统计边缘像素
        edge_count = np.sum(result == 255)
        edge_density = edge_count / result.size * 100
        axes[row, col].text(0.5, -0.1, f"{edge_count}像素\n{edge_density:.1f}%",
                            transform=axes[row, col].transAxes,
                            ha='center', fontsize=9)

    # 算子特性说明
    axes[0, 2].axis('off')
    axes[0, 2].text(0.1, 0.6,
                    "算子特性对比:\n\n"
                    "Sobel算子:\n"
                    "  - 一阶导数\n"
                    "  - 计算简单快速\n"
                    "  - 边缘较粗\n\n"
                    "Laplacian算子:\n"
                    "  - 二阶导数\n"
                    "  - 对噪声敏感\n"
                    "  - 产生双边缘",
                    fontsize=9, verticalalignment='center')

    plt.suptitle("Canny算子与其他算子对比", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 性能对比
    times = []
    names = []

    for name, _ in operators:
        start_time = time.time()
        for _ in range(100):  # 重复100次
            if name == "Sobel":
                cv2.Sobel(noisy_img, cv2.CV_64F, 1, 0, ksize=3)
                cv2.Sobel(noisy_img, cv2.CV_64F, 0, 1, ksize=3)
            elif name == "Laplacian":
                cv2.Laplacian(noisy_img, cv2.CV_64F, ksize=3)
            elif name == "LoG":
                blurred = cv2.GaussianBlur(noisy_img, (5, 5), 1.0)
                cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
            elif name == "Canny":
                cv2.Canny(noisy_img, 50, 150)
        end_time = time.time()

        avg_time = (end_time - start_time) / 100
        times.append(avg_time)
        names.append(name)

    # 绘制性能对比图
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['blue', 'green', 'orange', 'red']
    bars = ax.bar(names, times, color=colors)
    ax.set_title("边缘检测算子计算时间对比")
    ax.set_ylabel("平均计算时间 (秒)")
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
    print("  - 优点: 计算快，实现简单")
    print("  - 缺点: 边缘粗，抗噪能力一般")
    print("  - 适用: 实时处理，简单应用")
    print()

    print("Laplacian算子:")
    print("  - 优点: 定位精确，能检测细线")
    print("  - 缺点: 对噪声敏感，产生双边缘")
    print("  - 适用: 无噪声环境，精细边缘检测")
    print()

    print("LoG算子:")
    print("  - 优点: 抗噪性好，多尺度检测")
    print("  - 缺点: 计算复杂，需要调参")
    print("  - 适用: 多尺度边缘检测")
    print()

    print("Canny算子:")
    print("  - 优点: 抗噪性强，单边缘响应，定位精确")
    print("  - 缺点: 计算复杂，需要调参")
    print("  - 适用: 高质量边缘检测，复杂场景")
    print()

    return operators, times


# 比较Canny与其他算子
comparison_results = compare_canny_with_others()

# ==================== 9. 练习与挑战 ====================
print("\n💪 9. 练习与挑战")
print("=" * 30)

print("""
练习题：

1. 基础练习：
   a) 实现手动Canny算法的完整步骤
   b) 实现自适应Canny阈值
   c) 实现多尺度Canny边缘检测

2. 进阶练习：
   a) 实现彩色图像的Canny边缘检测
   b) 实现Canny算子的GPU加速版本
   c) 实现Canny算子的实时视频处理

3. 思考题：
   a) 为什么Canny算法需要非极大值抑制？
   b) 双阈值检测相比单阈值有什么优势？
   c) 如何为不同图片自动选择合适的Canny参数？
   d) Canny算子在什么情况下效果最好？
""")

# 练习框架代码
print("\n💻 练习框架代码：")

print("""
# 练习1a: 完整手动Canny实现
def manual_canny(image, sigma=1.4, low_ratio=0.1, high_ratio=0.3):
    # 1. 高斯滤波
    ksize = int(6*sigma) + 1
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    # 2. 计算梯度
    grad_x = cv2.Sobel(blurred.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
    direction = np.mod(direction, 180)

    # 3. 非极大值抑制
    nms_result = non_maximum_suppression(magnitude, direction)

    # 4. 双阈值检测
    threshold_result, high_thresh, low_thresh = double_threshold(nms_result, low_ratio, high_ratio)

    # 5. 边缘连接
    final_edges = edge_tracking_by_hysteresis(threshold_result)

    return final_edges, magnitude, direction, nms_result, threshold_result

# 练习1b: 自适应Canny阈值
def adaptive_canny(image, sigma=1.4, sigma_ratio=0.33):
    # 计算自适应阈值
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma_ratio) * median))
    upper = int(min(255, (1.0 + sigma_ratio) * median))

    # 应用Canny
    edges = cv2.Canny(image, lower, upper)

    return edges, lower, upper

# 练习1c: 多尺度Canny
def multi_scale_canny(image, sigmas=[0.5, 1.0, 1.5, 2.0]):
    all_edges = []

    for sigma in sigmas:
        # 计算该尺度的Canny
        ksize = int(6*sigma) + 1
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

        # 自适应阈值
        median = np.median(blurred)
        lower = int(max(0, (1.0 - 0.33) * median))
        upper = int(min(255, (1.0 + 0.33) * median))

        edges = cv2.Canny(blurred, lower, upper)
        all_edges.append(edges)

    # 合并多尺度结果
    combined = np.zeros_like(image, dtype=np.uint8)
    for edges in all_edges:
        combined = cv2.bitwise_or(combined, edges)

    return combined, all_edges

# 练习2a: 彩色图像Canny
def color_canny(image, sigma=1.4, low_ratio=0.1, high_ratio=0.3):
    # 分离通道
    b, g, r = cv2.split(image)

    # 对每个通道应用Canny
    edges_b = cv2.Canny(b, 50, 150)
    edges_g = cv2.Canny(g, 50, 150)
    edges_r = cv2.Canny(r, 50, 150)

    # 合并通道边缘
    combined = cv2.bitwise_or(edges_b, edges_g)
    combined = cv2.bitwise_or(combined, edges_r)

    return combined

# 练习2c: 实时视频Canny处理
def realtime_canny_video(camera_index=0, low_thresh=50, high_thresh=150):
    cap = cv2.VideoCapture(camera_index)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 应用Canny
        edges = cv2.Canny(gray, low_thresh, high_thresh)

        # 显示结果
        cv2.imshow('Original', frame)
        cv2.imshow('Canny Edges', edges)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
""")

# ==================== 10. 总结 ====================
print("\n" + "=" * 50)
print("✅ Canny边缘检测总结")
print("=" * 50)

summary = """
📊 Canny边缘检测核心知识：

1. 算法步骤
   - 1. 高斯滤波: 使用高斯滤波去除噪声
   - 2. 计算梯度: 使用Sobel算子计算梯度幅值和方向
   - 3. 非极大值抑制: 只保留梯度方向上的局部最大值
   - 4. 双阈值检测: 使用高阈值和低阈值区分边缘
   - 5. 边缘连接: 通过滞后阈值连接边缘

2. 关键概念
   - 高斯滤波: 控制平滑程度，减少噪声
   - 梯度计算: 检测边缘强度和方向
   - 非极大值抑制: 细化边缘，得到单像素边缘
   - 双阈值: 区分强边缘和弱边缘
   - 边缘连接: 连接断裂的边缘

3. 参数选择
   - 高斯sigma: 控制平滑程度，通常1.0-2.0
   - 高阈值: 控制强边缘检测，通常为最大梯度的20-30%
   - 低阈值: 通常为高阈值的40-50%
   - 孔径大小: Sobel核大小，通常为3

4. 性能特点
   - 时间复杂度: O(N²k² + N²)，N为图像尺寸，k为高斯核大小
   - 空间复杂度: O(N²)
   - 计算效率: 中等，不适合实时高分辨率处理
   - 内存需求: 需要存储梯度幅值和方向

5. 优点
   - 抗噪声能力强
   - 边缘定位精确
   - 单边缘响应
   - 边缘连续性较好
   - 参数可调，适应不同场景

6. 缺点
   - 计算复杂度较高
   - 需要手动调整参数
   - 对纹理复杂图像可能产生过多边缘
   - 对弱边缘可能检测不完整

7. 实际应用
   - 高质量边缘检测
   - 计算机视觉预处理
   - 图像分割
   - 目标检测
   - 特征提取
   - 医学影像分析
   - 工业检测

8. 最佳实践
   - 预处理: 确保输入图像质量
   - 参数调优: 根据具体场景调整参数
   - 后处理: 根据需要连接或细化边缘
   - 多尺度: 对复杂图像使用多尺度分析
   - 自适应: 使用自适应阈值提高鲁棒性

🎯 核心代码记忆：
   # OpenCV Canny基本用法
   edges = cv2.Canny(image, low_threshold, high_threshold)

   # 手动Canny实现框架
   def canny_manual(image, sigma=1.4, low_ratio=0.1, high_ratio=0.3):
       blurred = cv2.GaussianBlur(image, sigma=sigma)
       grad_x, grad_y = cv2.Sobel(blurred, cv2.CV_64F, 1, 0), cv2.Sobel(blurred, cv2.CV_64F, 0, 1)
       magnitude = np.sqrt(grad_x**2 + grad_y**2)
       direction = np.arctan2(grad_y, grad_x)
       nms_result = non_maximum_suppression(magnitude, direction)
       threshold_result = double_threshold(nms_result, low_ratio, high_ratio)
       final_edges = edge_tracking(threshold_result)
       return final_edges

   # 自适应阈值
   median = np.median(image)
   lower = int(max(0, (1.0 - 0.33) * median))
   upper = int(min(255, (1.0 + 0.33) * median))
"""

print(summary)
print("\n📁 第6天学习完成！")
print("  我们已经掌握了4种重要的边缘检测算法：")
print("  1. Sobel算子 - 一阶微分，计算快速")
print("  2. Laplacian算子 - 二阶微分，定位精确")
print("  3. LoG算子 - 高斯-拉普拉斯，抗噪性好")
print("  4. Canny算子 - 多阶段算法，效果最优")
print("\n🎉 明天我们将进入第7天的学习：图像形态学操作！")