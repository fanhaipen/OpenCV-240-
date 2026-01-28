"""
第7天 - 形态学基础操作完整教程
学习目标：掌握腐蚀、膨胀、开运算、闭运算
重点：基本原理、实际应用、参数调优
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

print("🔍 第7天 - 形态学基础操作完整教程")
print("=" * 60)

# ==================== 1. 腐蚀操作（Erosion） ====================
print("\n💎 1. 腐蚀操作（Erosion）")
print("=" * 50)


def demonstrate_erosion():
    """详细演示腐蚀操作"""

    print("腐蚀操作（Erosion）:")
    print("-" * 40)

    print("""
腐蚀操作原理：
  - 用结构元素扫描图像的每一个像素
  - 如果结构元素完全包含在目标区域内，则保留中心像素
  - 否则，删除该像素（设置为背景）

数学表达：
  A ⊖ B = {z | B_z ⊆ A}
  其中A是图像，B是结构元素，B_z是B平移z后的集合

效果：
  - 消除边界点，使边界向内部收缩
  - 消除小且无意义的物体
  - 断开细小的连接
  - 平滑较大物体的边界

应用场景：
  - 去除小噪声点
  - 分离相连的物体
  - 细化物体
  - 消除毛刺
""")

    # 创建测试图像
    img = np.zeros((150, 200), dtype=np.uint8)

    # 添加各种形状
    cv2.rectangle(img, (20, 20), (60, 60), 255, -1)  # 正方形
    cv2.circle(img, (100, 40), 20, 255, -1)  # 圆形
    cv2.rectangle(img, (140, 20), (180, 60), 255, -1)  # 长方形

    # 添加小噪声点
    noise_points = [(10, 10), (15, 15), (190, 5), (5, 140), (195, 145)]
    for x, y in noise_points:
        img[y, x] = 255

    # 添加细连接
    cv2.line(img, (30, 80), (170, 80), 255, 2)

    # 应用不同大小的腐蚀
    kernel_sizes = [3, 5, 7]
    erosion_results = []

    for size in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)) #对于cv2.MORPH_RECT（矩形结构元素），得到的是全1的矩阵：
        eroded = cv2.erode(img, kernel, iterations=1)
        erosion_results.append((f"{size}x{size}", eroded, kernel))

    # 可视化结果
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    # 原始图像
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("原始图像")
    axes[0, 0].axis('off')

    # 显示结构元素
    for i, (name, _, kernel) in enumerate(erosion_results):
        axes[0, i + 1].imshow(kernel * 255)
        axes[0, i + 1].set_title(f"结构元素\n{name}")
        axes[0, i + 1].set_xticks([])
        axes[0, i + 1].set_yticks([])

    # 腐蚀结果
    for i, (name, result, _) in enumerate(erosion_results):
        axes[1, i].imshow(result, cmap='gray')
        axes[1, i].set_title(f"腐蚀结果\n{name}")
       # axes[1, i].axis('off')

        # 统计信息
        original_pixels = np.sum(img == 255)
        eroded_pixels = np.sum(result == 255)
        reduction = (original_pixels - eroded_pixels) / original_pixels * 100
        axes[1, i].set_xlabel(f"减少: {reduction:.1f}%")

    # 迭代效果对比
    kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    iterations = [1, 2, 3, 5]
    iteration_results = []

    for iters in iterations:
        eroded = img.copy()
        for _ in range(iters):
            eroded = cv2.erode(eroded, kernel_3x3)
        iteration_results.append((f"Iteration {iters}", eroded))

    axes[1, 3].axis('off')
    info_text = "Erosion Iteration Analysis:\n\n"
    for name, result in iteration_results:
        pixels = np.sum(result == 255)
        info_text += f"{name}: {pixels} pixels\n"

    axes[1, 3].text(0.1, 0.5, info_text, fontsize=10,
                    verticalalignment='center', fontfamily='monospace')

    plt.suptitle("腐蚀操作效果演示", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 实际应用示例：去噪
    print("\n实际应用：噪声去除")
    print("-" * 40)

    # 创建有噪声的图像
    noisy_img = np.zeros((100, 200), dtype=np.uint8)
    cv2.rectangle(noisy_img, (50, 30), (150, 70), 255, -1)

    # 添加椒盐噪声
    salt_pepper = np.random.random(noisy_img.shape) < 0.1
    noisy_img[salt_pepper] = 255

    # 应用腐蚀去噪
    denoise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    denoised = cv2.erode(noisy_img, denoise_kernel)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(noisy_img, cmap='gray')
    axes[0].set_title("有噪声图像")
    axes[0].axis('off')

    axes[1].imshow(denoise_kernel * 255, cmap='gray')
    axes[1].set_title("去噪核")
    axes[1].axis('off')

    axes[2].imshow(denoised, cmap='gray')
    axes[2].set_title("腐蚀去噪后")
    axes[2].axis('off')

    plt.suptitle("腐蚀操作在噪声去除中的应用", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return img, erosion_results, iteration_results


# 演示腐蚀操作
erosion_results = demonstrate_erosion()

# ==================== 2. 膨胀操作（Dilation） ====================
print("\n💎 2. 膨胀操作（Dilation）")
print("=" * 50)


def demonstrate_dilation():
    """详细演示膨胀操作"""

    print("膨胀操作（Dilation）:")
    print("-" * 40)

    print("""
膨胀操作原理：
  - 用结构元素扫描图像的每一个像素
  - 如果结构元素与目标区域有交集，则设置中心像素为前景
  - 否则，保持为背景

数学表达：
  A ⊕ B = {z | (B̂)_z ∩ A ≠ ∅}
  其中A是图像，B是结构元素，B̂是B的反射，(B̂)_z是反射平移z

效果：
  - 扩大边界点，使边界向外部扩展
  - 填充物体中的空洞
  - 连接相邻的物体
  - 平滑物体边界

应用场景：
  - 连接断裂的部分
  - 填充空洞
  - 扩大物体尺寸
  - 边界平滑
""")

    # 创建测试图像（有断裂和空洞）
    img = np.zeros((150, 200), dtype=np.uint8)

    # 添加有断裂的线条
    cv2.line(img, (20, 30), (50, 30), 255, 3)  # 线段1
    cv2.line(img, (70, 30), (100, 30), 255, 3)  # 线段2（断开）
    cv2.line(img, (120, 30), (150, 30), 255, 3)  # 线段3

    # 添加有空洞的形状
    cv2.rectangle(img, (30, 70), (80, 120), 255, -1)  # 实心矩形
    cv2.rectangle(img, (100, 70), (150, 120), 255, 2)  # 空心矩形（有空洞）

    # 添加小物体
    img[130:135, 20:25] = 255  # 小方块1
    img[130:135, 40:45] = 255  # 小方块2

    # 应用不同大小的膨胀
    kernel_sizes = [3, 5, 7]
    dilation_results = []

    for size in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        dilated = cv2.dilate(img, kernel, iterations=1)
        dilation_results.append((f"{size}x{size}", dilated, kernel))

    # 可视化结果
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    # 原始图像
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("原始图像\n(有断裂和空洞)")
    axes[0, 0].axis('off')

    # 显示结构元素
    for i, (name, _, kernel) in enumerate(dilation_results):
        axes[0, i + 1].imshow(kernel * 255, cmap='gray')
        axes[0, i + 1].set_title(f"结构元素\n{name}")
        axes[0, i + 1].set_xticks([])
        axes[0, i + 1].set_yticks([])

    # 膨胀结果
    for i, (name, result, _) in enumerate(dilation_results):
        axes[1, i].imshow(result, cmap='gray')
        axes[1, i].set_title(f"膨胀结果\n{name}")
        axes[1, i].axis('off')

        # 统计信息
        original_pixels = np.sum(img == 255)
        dilated_pixels = np.sum(result == 255)
        increase = (dilated_pixels - original_pixels) / original_pixels * 100
        axes[1, i].set_xlabel(f"增加: {increase:.1f}%")

    # 连接断裂的专项演示
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    connected = cv2.dilate(img, kernel_horizontal, iterations=2)

    axes[1, 3].imshow(connected, cmap='gray')
    axes[1, 3].set_title("水平连接\n15x3核")
    axes[1, 3].axis('off')

    plt.suptitle("膨胀操作效果演示", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 实际应用示例：字符连接
    print("\n实际应用：字符修复")
    print("-" * 40)

    # 创建断裂的字符
    broken_text = np.zeros((80, 200), dtype=np.uint8)
    cv2.putText(broken_text, "HELLO", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    # 模拟字符断裂（擦除部分像素）
    broken_text[25:35, 50:150] = 0

    # 应用膨胀修复
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    repaired = cv2.dilate(broken_text, repair_kernel, iterations=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(broken_text, cmap='gray')
    axes[0].set_title("断裂字符")
    axes[0].axis('off')

    axes[1].imshow(repair_kernel * 255, cmap='gray')
    axes[1].set_title("修复核")
    axes[1].axis('off')

    axes[2].imshow(repaired, cmap='gray')
    axes[2].set_title("膨胀修复后")
    axes[2].axis('off')

    plt.suptitle("膨胀操作在字符修复中的应用", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return img, dilation_results, broken_text, repaired


# 演示膨胀操作
dilation_results = demonstrate_dilation()

# ==================== 3. 开运算（Opening） ====================
print("\n💎 3. 开运算（Opening）")
print("=" * 50)


def demonstrate_opening():
    """详细演示开运算"""

    print("开运算（Opening）:")
    print("-" * 40)

    print("""
开运算原理：
  - 先腐蚀后膨胀
  - 公式：opening = dilate(erode(image))

数学表达：
  A ∘ B = (A ⊖ B) ⊕ B

效果：
  - 消除小物体
  - 平滑大物体边界
  - 断开细连接
  - 在纤细点处分离物体

应用场景：
  - 去除小噪声点
  - 分离相连的物体
  - 消除毛刺
  - 背景提取
""")

    # 创建测试图像
    img = np.zeros((150, 200), dtype=np.uint8)

    # 添加大物体
    cv2.rectangle(img, (30, 30), (100, 100), 255, -1)

    # 添加小噪声
    for i in range(20):
        x = np.random.randint(120, 190)
        y = np.random.randint(10, 40)
        cv2.circle(img, (x, y), 2, 255, -1)

    # 添加细连接
    cv2.line(img, (50, 110), (80, 110), 255, 1)

    # 应用开运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 分步演示
    eroded = cv2.erode(img, kernel)
    opened = cv2.dilate(eroded, kernel)

    # 直接开运算
    opening_direct = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    images = [
        ("原始图像\n(有小噪声和细连接)", img, 'gray'),
        ("腐蚀后\n(去除小物体)", eroded, 'gray'),
        ("膨胀后\n(恢复大小)", opened, 'gray'),
        ("直接开运算", opening_direct, 'gray'),
    ]

    for i, (title, image, cmap) in enumerate(images):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(image, cmap=cmap)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

        # 统计信息
        white_pixels = np.sum(image == 255)
        if i > 0:
            change = (white_pixels - np.sum(img == 255)) / np.sum(img == 255) * 100
            axes[row, col].set_xlabel(f"像素: {white_pixels} ({change:+.1f}%)")
        else:
            axes[row, col].set_xlabel(f"像素: {white_pixels}")

    # 不同核大小的开运算比较
    kernel_sizes = [3, 5, 7, 9]
    opening_comparison = []

    for size in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        opening_comparison.append((f"{size}x{size}", opened))

    axes[1, 2].axis('off')
    comparison_text = "不同核大小效果:\n\n"
    for name, result in opening_comparison:
        pixels = np.sum(result == 255)
        comparison_text += f"{name}: {pixels}像素\n"

    axes[1, 2].text(0.1, 0.5, comparison_text, fontsize=10,
                    verticalalignment='center', fontfamily='monospace')

    plt.suptitle("开运算效果演示", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 实际应用：指纹图像处理
    print("\n实际应用：指纹图像增强")
    print("-" * 40)

    # 创建模拟指纹图像
    fingerprint = np.zeros((120, 120), dtype=np.uint8)

    # 添加指纹纹路
    angles = [0, 30, 60, 90, 120, 150]
    for angle in angles:
        center = (60, 60)
        length = 40
        end_x = int(center[0] + length * np.cos(np.radians(angle)))
        end_y = int(center[1] + length * np.sin(np.radians(angle)))
        cv2.line(fingerprint, center, (end_x, end_y), 255, 2)

    # 添加噪声
    noise = np.random.random(fingerprint.shape) < 0.1
    noisy_fingerprint = fingerprint.copy()
    noisy_fingerprint[noise] = 255

    # 应用开运算去噪
    fingerprint_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_fingerprint = cv2.morphologyEx(noisy_fingerprint, cv2.MORPH_OPEN, fingerprint_kernel)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    images = [
        ("原始指纹", fingerprint, 'gray'),
        ("加噪指纹", noisy_fingerprint, 'gray'),
        ("开运算去噪", cleaned_fingerprint, 'gray'),
        ("结构元素", fingerprint_kernel * 255, 'gray')
    ]

    for i, (title, image, cmap) in enumerate(images):
        axes[i].imshow(image, cmap=cmap)
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.suptitle("开运算在指纹图像处理中的应用", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return img, opening_comparison, noisy_fingerprint, cleaned_fingerprint


# 演示开运算
opening_results = demonstrate_opening()

# ==================== 4. 闭运算（Closing） ====================
print("\n💎 4. 闭运算（Closing）")
print("=" * 50)


def demonstrate_closing():
    """详细演示闭运算"""

    print("闭运算（Closing）:")
    print("-" * 40)

    print("""
闭运算原理：
  - 先膨胀后腐蚀
  - 公式：closing = erode(dilate(image))

数学表达：
  A • B = (A ⊕ B) ⊖ B

效果：
  - 填充小空洞
  - 连接邻近物体
  - 平滑边界
  - 消除小暗区域

应用场景：
  - 填充物体中的空洞
  - 连接断裂部分
  - 平滑轮廓
  - 前景提取
""")

    # 创建测试图像
    img = np.zeros((150, 200), dtype=np.uint8)

    # 添加有空洞的物体
    cv2.rectangle(img, (30, 30), (100, 100), 255, -1)
    cv2.rectangle(img, (40, 40), (90, 90), 0, -1)  # 内部空洞

    # 添加断裂
    cv2.line(img, (120, 30), (140, 30), 255, 3)
    cv2.line(img, (150, 30), (170, 30), 255, 3)  # 断开

    # 添加小暗区域
    small_dark = [(130, 70), (140, 80), (150, 75)]
    for x, y in small_dark:
        cv2.circle(img, (x, y), 3, 0, -1)

    # 应用闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 分步演示
    dilated = cv2.dilate(img, kernel)
    closed = cv2.erode(dilated, kernel)

    # 直接闭运算
    closing_direct = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    images = [
        ("原始图像\n(有空洞和断裂)", img, 'gray'),
        ("膨胀后\n(填充空洞)", dilated, 'gray'),
        ("腐蚀后\n(恢复形状)", closed, 'gray'),
        ("直接闭运算", closing_direct, 'gray'),
    ]

    for i, (title, image, cmap) in enumerate(images):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(image, cmap=cmap)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

        # 统计信息
        white_pixels = np.sum(image == 255)
        if i > 0:
            change = (white_pixels - np.sum(img == 255)) / np.sum(img == 255) * 100
            axes[row, col].set_xlabel(f"像素: {white_pixels} ({change:+.1f}%)")
        else:
            axes[row, col].set_xlabel(f"像素: {white_pixels}")

    # 不同核大小的闭运算比较
    kernel_sizes = [3, 5, 7, 9]
    closing_comparison = []

    for size in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        closing_comparison.append((f"{size}x{size}", closed))

    axes[1, 2].axis('off')
    comparison_text = "不同核大小效果:\n\n"
    for name, result in closing_comparison:
        pixels = np.sum(result == 255)
        comparison_text += f"{name}: {pixels}像素\n"

    axes[1, 2].text(0.1, 0.5, comparison_text, fontsize=10,
                    verticalalignment='center', fontfamily='monospace')

    plt.suptitle("闭运算效果演示", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 实际应用：医学图像处理
    print("\n实际应用：血管连接")
    print("-" * 40)

    # 创建模拟血管图像
    vessels = np.zeros((120, 120), dtype=np.uint8)

    # 添加血管网络（有断裂）
    cv2.line(vessels, (20, 20), (100, 20), 255, 2)
    cv2.line(vessels, (20, 60), (100, 60), 255, 2)
    cv2.line(vessels, (20, 100), (100, 100), 255, 2)

    # 垂直线条（有断裂）
    cv2.line(vessels, (20, 20), (20, 100), 255, 2)
    cv2.line(vessels, (60, 20), (60, 100), 255, 2)
    cv2.line(vessels, (100, 20), (100, 100), 255, 2)

    # 添加空洞
    vessels[30:40, 30:40] = 0

    # 应用闭运算连接
    vessel_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    connected_vessels = cv2.morphologyEx(vessels, cv2.MORPH_CLOSE, vessel_kernel)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    images = [
        ("断裂血管", vessels, 'gray'),
        ("闭运算连接", connected_vessels, 'gray'),
        ("结构元素", vessel_kernel * 255, 'gray')
    ]

    for i, (title, image, cmap) in enumerate(images[:3]):
        axes[i].imshow(image, cmap=cmap)
        axes[i].set_title(title)
        axes[i].axis('off')

    # 统计信息
    axes[3].axis('off')
    stats_text = f"血管连接统计:\n\n"
    stats_text += f"原始像素: {np.sum(vessels == 255)}\n"
    stats_text += f"连接后像素: {np.sum(connected_vessels == 255)}\n"
    stats_text += f"增加: {np.sum(connected_vessels == 255) - np.sum(vessels == 255)}像素"
    axes[3].text(0.1, 0.5, stats_text, fontsize=10,
                 verticalalignment='center', fontfamily='monospace')

    plt.suptitle("闭运算在血管连接中的应用", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return img, closing_comparison, vessels, connected_vessels


# 演示闭运算
closing_results = demonstrate_closing()

# ==================== 5. 综合比较 ====================
print("\n📊 5. 四种操作的对比分析")
print("=" * 50)


def compare_all_operations():
    """对比四种形态学操作"""

    print("四种形态学操作对比:")
    print("-" * 40)

    # 创建测试图像
    img = np.zeros((200, 300), dtype=np.uint8)

    # 添加各种特征
    cv2.rectangle(img, (20, 20), (80, 80), 255, -1)  # 矩形
    cv2.circle(img, (150, 50), 30, 255, -1)  # 圆形
    cv2.rectangle(img, (200, 20), (250, 80), 255, 2)  # 空心矩形

    # 添加噪声
    for _ in range(10):
        x = np.random.randint(10, 290)
        y = np.random.randint(120, 190)
        cv2.circle(img, (x, y), 2, 255, -1)

    # 添加细线
    cv2.line(img, (20, 100), (280, 100), 255, 1)

    # 应用不同操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    eroded = cv2.erode(img, kernel)
    dilated = cv2.dilate(img, kernel)
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    images = [
        ("原始图像", img, 'gray'),
        ("腐蚀", eroded, 'gray'),
        ("膨胀", dilated, 'gray'),
        ("开运算", opened, 'gray'),
        ("闭运算", closed, 'gray')
    ]

    for i, (title, image, cmap) in enumerate(images):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(image, cmap=cmap)
        axes[row, col].set_title(title, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')

        # 显示像素数量
        white_pixels = np.sum(image == 255)
        axes[row, col].set_xlabel(f"白色像素: {white_pixels}")

    # 操作效果对比表
    operations = ['原始', '腐蚀', '膨胀', '开运算', '闭运算']
    pixel_counts = [np.sum(img == 255),
                    np.sum(eroded == 255),
                    np.sum(dilated == 255),
                    np.sum(opened == 255),
                    np.sum(closed == 255)]

    changes = [0,
               (pixel_counts[1] - pixel_counts[0]) / pixel_counts[0] * 100,
               (pixel_counts[2] - pixel_counts[0]) / pixel_counts[0] * 100,
               (pixel_counts[3] - pixel_counts[0]) / pixel_counts[0] * 100,
               (pixel_counts[4] - pixel_counts[0]) / pixel_counts[0] * 100]

    axes[1, 2].axis('off')
    comparison_table = "形态学操作效果对比:\n\n"
    comparison_table += f"{'操作':<10} {'像素数':<10} {'变化':<10}\n"
    comparison_table += "-" * 30 + "\n"

    for op, count, change in zip(operations, pixel_counts, changes):
        comparison_table += f"{op:<10} {count:<10} {change:+.1f}%\n"

    axes[1, 2].text(0.1, 0.5, comparison_table, fontsize=10,
                    verticalalignment='center', fontfamily='monospace',
                    fontweight='bold')

    plt.suptitle("四种形态学操作对比", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

    # 实际应用：综合处理流程
    print("\n实际应用：综合处理流程")
    print("-" * 40)

    # 创建复杂图像
    complex_img = np.zeros((150, 200), dtype=np.uint8)
    cv2.putText(complex_img, "MORPHOLOGY", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.rectangle(complex_img, (20, 60), (180, 100), 255, 1)

    # 添加噪声
    noise_mask = np.random.random(complex_img.shape) < 0.1
    complex_img[noise_mask] = 255

    # 添加空洞
    complex_img[80:85, 80:120] = 0

    # 处理流程
    process_steps = []
    process_names = []

    # 1. 腐蚀去噪
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    step1 = cv2.erode(complex_img, kernel_erode)
    process_steps.append(step1)
    process_names.append("腐蚀去噪")

    # 2. 膨胀连接
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    step2 = cv2.dilate(step1, kernel_dilate)
    process_steps.append(step2)
    process_names.append("膨胀连接")

    # 3. 开运算平滑
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    step3 = cv2.morphologyEx(step2, cv2.MORPH_OPEN, kernel_open)
    process_steps.append(step3)
    process_names.append("开运算平滑")

    # 4. 闭运算填充
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    step4 = cv2.morphologyEx(step3, cv2.MORPH_CLOSE, kernel_close)
    process_steps.append(step4)
    process_names.append("闭运算填充")

    # 可视化处理流程
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    all_images = [complex_img] + process_steps
    all_names = ["原始图像"] + process_names

    for i, (image, name) in enumerate(zip(all_images, all_names)):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(name, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
        axes[row, col].set_xlabel(f"白色像素: {np.sum(image == 255)}")

    plt.suptitle("形态学处理流程示例", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

    return img, eroded, dilated, opened, closed, complex_img, process_steps


# 运行综合比较
comparison_results = compare_all_operations()

# ==================== 6. 总结与应用建议 ====================
print("\n📋 6. 总结与应用建议")
print("=" * 50)

print("""
形态学操作总结：

1. 腐蚀（Erosion）:
   - 作用: 缩小物体，去除小物体
   - 应用: 去噪、分离物体、细化
   - 参数: 结构元素大小、形状、迭代次数
   - 注意: 可能会丢失重要信息

2. 膨胀（Dilation）:
   - 作用: 扩大物体，填充空洞
   - 应用: 连接物体、填充空洞、扩大特征
   - 参数: 结构元素大小、形状、迭代次数
   - 注意: 可能会连接不应连接的部分

3. 开运算（Opening）:
   - 作用: 先腐蚀后膨胀，消除小物体
   - 应用: 去除小噪声、分离接触物体
   - 参数: 结构元素大小和形状
   - 注意: 适合去除比结构元素小的亮点

4. 闭运算（Closing）:
   - 作用: 先膨胀后腐蚀，填充小空洞
   - 应用: 填充空洞、连接断裂
   - 参数: 结构元素大小和形状
   - 注意: 适合填充比结构元素小的暗点

选择建议:

1. 去噪: 开运算
2. 填充空洞: 闭运算
3. 分离物体: 腐蚀或开运算
4. 连接断裂: 膨胀或闭运算
5. 边缘检测: 形态学梯度
6. 提取骨架: 形态细化
7. 大小分析: 颗粒分析
8. 纹理提取: 顶帽/黑帽变换

结构元素选择:

1. 矩形核: 通用，计算快
2. 椭圆核: 各向同性处理
3. 十字核: 对角线连接
4. 自定义核: 特定形状处理

参数调优:

1. 核大小: 决定影响范围
2. 核形状: 决定影响方向
3. 迭代次数: 决定强度
4. 组合使用: 开+闭运算组合

实际应用技巧:

1. 从小核开始，逐渐增大
2. 先尝试简单的矩形核
3. 注意核的大小和形状对结果的影响
4. 组合使用时注意顺序
5. 考虑使用形态学梯度提取边界
6. 顶帽变换用于提取亮细节
7. 黑帽变换用于提取暗细节
""")

# 创建应用示例
print("\n💡 快速参考示例代码:")
print("-" * 40)

quick_reference_code = """
# 1. 基本操作
import cv2
import numpy as np

# 读取图像
img = cv2.imread('image.jpg', 0)

# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 腐蚀
eroded = cv2.erode(img, kernel)

# 膨胀
dilated = cv2.dilate(img, kernel)

# 开运算
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# 闭运算
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# 形态学梯度
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# 顶帽变换
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# 黑帽变换
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
"""

print(quick_reference_code)

print("""
常见问题与解决方案:

1. 问题: 操作效果太强/太弱
   解决方案: 调整核大小或迭代次数

2. 问题: 丢失重要特征
   解决方案: 使用更小的核或不同的形状

3. 问题: 计算时间太长
   解决方案: 减小图像尺寸或使用矩形核

4. 问题: 边缘处理不当
   解决方案: 使用borderType参数控制边界填充

5. 问题: 结果不符合预期
   解决方案: 分步调试，查看中间结果
""")

print("\n✅ 形态学基础操作学习完成！")
print("📚 下一节：形态学梯度、顶帽/黑帽变换、骨架提取")
print("=" * 60)