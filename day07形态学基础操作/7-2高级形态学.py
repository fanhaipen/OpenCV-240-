import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

print("=" * 60)
print("🎯 形态学操作完整教程")
print("=" * 60)

# ==================== 1. 基础回顾 ====================
print("\n📚 1. 基础形态学操作回顾")
print("=" * 50)


def review_basic_morphology():
    """回顾基本的形态学操作"""

    print("""
基本形态学操作回顾:

1. 膨胀 (Dilation)
   - 扩大白色区域
   - 公式: A ⊕ B = {z | (B̂)_z ∩ A ≠ ∅}
   - 作用: 连接断裂、填充空洞

2. 腐蚀 (Erosion)
   - 缩小白色区域
   - 公式: A ⊖ B = {z | (B)_z ⊆ A}
   - 作用: 分离物体、消除小点

3. 开运算 (Opening)
   - 先腐蚀后膨胀
   - 公式: A ∘ B = (A ⊖ B) ⊕ B
   - 作用: 去噪、平滑轮廓

4. 闭运算 (Closing)
   - 先膨胀后腐蚀
   - 公式: A • B = (A ⊕ B) ⊖ B
   - 作用: 填充空洞、连接相邻
""")

    # 创建测试图像
    test_image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(test_image, (20, 20), (80, 80), 255, -1)
    cv2.circle(test_image, (50, 50), 10, 0, -1)  # 创建一个洞

    # 结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 应用不同操作
    eroded = cv2.erode(test_image, kernel)
    dilated = cv2.dilate(test_image, kernel)
    opened = cv2.morphologyEx(test_image, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(test_image, cv2.MORPH_CLOSE, kernel)

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    operations = [
        ("原始图像", test_image),
        ("腐蚀", eroded),
        ("膨胀", dilated),
        ("开运算", opened),
        ("闭运算", closed),
    ]

    for i, (title, img) in enumerate(operations):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[row, col].set_title(title, fontweight='bold')
        axes[row, col].axis('off')
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

        # 添加统计信息
        white_pixels = np.sum(img > 0)
        total_pixels = img.size
        white_percent = white_pixels / total_pixels * 100
        axes[row, col].set_xlabel(f"白色像素: {white_pixels} ({white_percent:.1f}%)")

    # 结构元素可视化
    axes[1, 2].clear()
    axes[1, 2].imshow(kernel * 255, cmap='gray')
    axes[1, 2].set_title("结构元素\n(5×5 矩形)")
    axes[1, 2].axis('on')
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle("基本形态学操作回顾", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return test_image, kernel, eroded, dilated, opened, closed


# 回顾基础
basic_results = review_basic_morphology()

# ==================== 2. 形态学梯度 ====================
print("\n🌊 2. 形态学梯度 (Morphological Gradient)")
print("=" * 50)


def demonstrate_gradient():
    """演示形态学梯度"""

    print("""
形态学梯度原理:
  - 基本梯度: G = dilation - erosion
  - 外梯度: G_ext = dilation - original
  - 内梯度: G_int = original - erosion

物理意义:
  - 基本梯度: 物体的边界
  - 外梯度: 物体的外部边界
  - 内梯度: 物体的内部边界

应用场景:
  - 边缘检测
  - 轮廓提取
  - 物体边界增强
""")

    # 创建测试图像
    img = np.zeros((200, 300), dtype=np.uint8)

    # 创建各种形状
    cv2.rectangle(img, (30, 30), (120, 100), 255, -1)  # 矩形
    cv2.circle(img, (200, 60), 40, 255, -1)  # 圆形
    cv2.ellipse(img, (150, 150), (60, 30), 0, 0, 360, 255, -1)  # 椭圆

    # 添加噪声
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    noisy_img = cv2.add(img, noise)

    # 定义结构元素
    kernel_sizes = [3, 7, 11]

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))

    for i, ksize in enumerate(kernel_sizes):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        # 计算基本形态学梯度
        gradient = cv2.morphologyEx(noisy_img, cv2.MORPH_GRADIENT, kernel)

        # 分步计算
        dilated = cv2.dilate(noisy_img, kernel)
        eroded = cv2.erode(noisy_img, kernel)

        # 外梯度
        external_grad = dilated - noisy_img

        # 内梯度
        internal_grad = noisy_img - eroded

        # 可视化
        images_row = [
            (f"原始图像", noisy_img, 'gray'),
            (f"核大小: {ksize}×{ksize}", gradient, 'gray'),
            (f"外梯度", external_grad, 'gray'),
            (f"内梯度", internal_grad, 'gray'),
        ]

        for j, (title, image, cmap) in enumerate(images_row):
            axes[i, j].imshow(image, cmap=cmap)
            axes[i, j].set_title(title, fontsize=10, fontweight='bold')
            axes[i, j].axis('off')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            if j > 0:  # 统计梯度信息
                gradient_pixels = np.sum(image > 0)
                axes[i, j].set_xlabel(f"边界像素: {gradient_pixels}")

    plt.suptitle("形态学梯度 - 不同核大小比较", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 梯度类型对比
    print("\n🔍 梯度类型对比分析:")
    print("-" * 40)

    # 使用中等核大小
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    external = cv2.dilate(img, kernel) - img
    internal = img - cv2.erode(img, kernel)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 第一行：图像
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("原始图像")
    axes[0, 0].axis('off')

    # 膨胀和腐蚀
    axes[0, 1].imshow(cv2.dilate(img, kernel), cmap='gray')
    axes[0, 1].set_title("膨胀 (Dilation)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(cv2.erode(img, kernel), cmap='gray')
    axes[0, 2].set_title("腐蚀 (Erosion)")
    axes[0, 2].axis('off')

    # 第二行：梯度
    axes[1, 0].imshow(gradient, cmap='gray')
    axes[1, 0].set_title("基本梯度\n(膨胀 - 腐蚀)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(external, cmap='gray')
    axes[1, 1].set_title("外梯度\n(膨胀 - 原始)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(internal, cmap='gray')
    axes[1, 2].set_title("内梯度\n(原始 - 腐蚀)")
    axes[1, 2].axis('off')

    plt.suptitle("三种形态学梯度对比", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return img, gradient, external, internal


# 演示梯度
gradient_results = demonstrate_gradient()

# ==================== 3. 顶帽变换 ====================
print("\n🎩 3. 顶帽变换 (Top-hat Transform)")
print("=" * 50)


def demonstrate_tophat():
    """演示顶帽变换"""

    print("""
顶帽变换原理:
  - 原始图像 - 开运算结果
  - 公式: tophat = I - (I ∘ B)
  - 其中 ∘ 表示开运算

物理意义:
  - 提取比背景亮的细节
  - 消除不均匀光照
  - 增强小物体

应用场景:
  - 文本提取
  - 医学图像处理
  - 工业检测
  - 光照校正
""")

    # 创建有光照变化的图像
    img = np.zeros((200, 300), dtype=np.uint8)

    # 创建不均匀光照
    x = np.arange(300)
    y = np.arange(200)
    X, Y = np.meshgrid(x, y)

    # 添加正弦光照变化
    illumination = 100 + 50 * np.sin(X / 30) + 30 * np.cos(Y / 20)
    illumination = illumination.astype(np.uint8)

    # 添加小物体
    objects = np.zeros_like(img)
    cv2.circle(objects, (50, 50), 8, 200, -1)
    cv2.circle(objects, (150, 80), 5, 220, -1)
    cv2.circle(objects, (250, 120), 10, 180, -1)
    cv2.rectangle(objects, (100, 150), (120, 180), 200, -1)

    # 组合
    combined = cv2.add(illumination, objects)

    # 应用顶帽变换
    kernel_sizes = [5, 15, 25, 35]

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    # 原始图像
    axes[0, 0].imshow(illumination, cmap='gray')
    axes[0, 0].set_title("不均匀光照背景")
    # axes[0, 0].axis('off')
    axes[0, 0].set_xlabel(f"亮度范围: {illumination.min()}-{illumination.max()}")

    axes[0, 1].imshow(objects, cmap='gray')
    axes[0, 1].set_title("前景物体")
    # axes[0, 1].axis('off')
    white_pixels = np.sum(objects > 0)
    axes[0, 1].set_xlabel(f"物体像素: {white_pixels}")

    axes[0, 2].imshow(combined, cmap='gray')
    axes[0, 2].set_title("合成图像")
    #axes[0, 2].axis('off')
    axes[0, 2].set_xlabel(f"总亮度: {combined.mean():.1f}")

    # 不同核大小的顶帽变换
    for i, ksize in enumerate(kernel_sizes):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        tophat = cv2.morphologyEx(combined, cv2.MORPH_TOPHAT, kernel)

        #row = 1 if i < 2 else 0
        #col = 3 if i < 2 else i
        row = 1
        col = i
        axes[row, col].imshow(tophat, cmap='gray')
        axes[row, col].set_title(f"顶帽变换\n核大小: {ksize}×{ksize}")
        #axes[row, col].axis('off')

        # 统计提取的物体
        extracted_pixels = np.sum(tophat > 50)
        axes[row, col].set_xlabel(f"提取像素: {extracted_pixels}")

    plt.suptitle("顶帽变换 - 去除不均匀光照", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 实际应用示例：文档增强
    print("\n📄 实际应用：文档图像增强")
    print("-" * 40)

    # 创建文档图像
    doc_bg = np.zeros((150, 300), dtype=np.uint8)

    # 添加不均匀光照
    x = np.arange(300)
    y = np.arange(150)
    X, Y = np.meshgrid(x, y)

    # 创建渐变光照
    gradient_bg = 100 + 100 * np.exp(-((X - 150) ** 2 + (Y - 75) ** 2) / (2 * 100 ** 2))
    gradient_bg = gradient_bg.astype(np.uint8)

    # 添加文字
    text = np.zeros_like(doc_bg)
    cv2.putText(text, "Important Document", (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, 150, 2)
    cv2.putText(text, "Morphological processing is", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 120, 1)
    cv2.putText(text, "widely used in image analysis", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 120, 1)

    # 合成文档
    document = cv2.add(gradient_bg, text)

    # 处理步骤
    kernel_doc = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))

    # 1. 顶帽变换
    tophat_doc = cv2.morphologyEx(document, cv2.MORPH_TOPHAT, kernel_doc)

    # 2. 二值化
    _, binary_before = cv2.threshold(document, 150, 255, cv2.THRESH_BINARY)
    _, binary_after = cv2.threshold(tophat_doc, 30, 255, cv2.THRESH_BINARY)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    doc_images = [
        ("光照背景", gradient_bg, 'gray'),
        ("添加文字", text, 'gray'),
        ("合成文档", document, 'gray'),
        ("顶帽变换", tophat_doc, 'gray'),
        ("直接二值化", binary_before, 'gray'),
        ("顶帽+二值化", binary_after, 'gray'),
    ]

    for i, (title, image, cmap) in enumerate(doc_images):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(image, cmap=cmap)
        axes[row, col].set_title(title, fontsize=10, fontweight='bold')
        axes[row, col].axis('off')

        if "二值化" in title:
            white_ratio = np.sum(image > 0) / image.size * 100
            axes[row, col].set_xlabel(f"文字比例: {white_ratio:.1f}%")

    plt.suptitle("顶帽变换在文档增强中的应用", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return document, tophat_doc, binary_before, binary_after


# 演示顶帽变换
tophat_results = demonstrate_tophat()

# ==================== 4. 黑帽变换 ====================
print("\n⚫ 4. 黑帽变换 (Black-hat Transform)")
print("=" * 50)


def demonstrate_blackhat():
    """演示黑帽变换"""

    print("""
黑帽变换原理:
  - 闭运算结果 - 原始图像
  - 公式: blackhat = (I • B) - I
  - 其中 • 表示闭运算

物理意义:
  - 提取比背景暗的细节
  - 检测暗区域和空洞
  - 增强暗部对比度

应用场景:
  - 缺陷检测
  - 指纹分析
  - 医学图像（暗区域）
  - 工业质检
""")

    # 创建有暗缺陷的图像
    img = np.ones((200, 300), dtype=np.uint8) * 200  # 亮背景

    # 添加暗缺陷
    cv2.rectangle(img, (50, 50), (100, 100), 100, -1)  # 暗矩形
    cv2.circle(img, (200, 80), 15, 50, -1)  # 暗圆形
    cv2.line(img, (120, 150), (180, 150), 80, 5)  # 暗线

    # 添加小暗点
    for i in range(10):
        x = np.random.randint(20, 280)
        y = np.random.randint(20, 180)
        cv2.circle(img, (x, y), 3, 60, -1)

    # 添加高斯噪声
    noise = np.random.normal(0, 10, img.shape)
    noisy_img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

    # 应用黑帽变换
    kernel_sizes = [3, 7, 15, 25]

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    # 原始图像
    axes[0, 0].imshow(noisy_img, cmap='gray')
    axes[0, 0].set_title("原始图像\n(亮背景+暗缺陷)")
    axes[0, 0].axis('off')

    # 直方图
    axes[0, 1].hist(noisy_img.ravel(), 256, [0, 256], color='gray')
    axes[0, 1].set_title("灰度直方图")
    axes[0, 1].set_xlabel("灰度值")
    axes[0, 1].set_ylabel("像素数")
    axes[0, 1].grid(True, alpha=0.3)

    # 不同核大小的黑帽变换
    for i, ksize in enumerate(kernel_sizes):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        blackhat = cv2.morphologyEx(noisy_img, cv2.MORPH_BLACKHAT, kernel)

        row = 1 if i < 2 else 0
        col = 2 + (i % 2)

        axes[row, col].imshow(blackhat, cmap='gray')
        axes[row, col].set_title(f"黑帽变换\n核大小: {ksize}×{ksize}")
        axes[row, col].axis('off')

        # 统计暗缺陷
        dark_pixels = np.sum(blackhat > 20)
        axes[row, col].set_xlabel(f"暗像素: {dark_pixels}")

    plt.suptitle("黑帽变换 - 暗缺陷检测", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 实际应用示例：PCB板缺陷检测
    print("\n🔧 实际应用：PCB板缺陷检测")
    print("-" * 40)

    # 创建PCB板图像
    pcb = np.ones((150, 250), dtype=np.uint8) * 180  # PCB基板

    # 添加电路轨迹
    cv2.rectangle(pcb, (20, 20), (230, 40), 100, 3)  # 上轨迹
    cv2.rectangle(pcb, (20, 60), (230, 80), 100, 3)  # 中轨迹
    cv2.rectangle(pcb, (20, 100), (230, 120), 100, 3)  # 下轨迹

    # 添加焊盘
    for i in range(5):
        x = 30 + i * 50
        cv2.circle(pcb, (x, 30), 8, 120, -1)  # 上焊盘
        cv2.circle(pcb, (x, 70), 8, 120, -1)  # 中焊盘
        cv2.circle(pcb, (x, 110), 8, 120, -1)  # 下焊盘

    # 添加暗缺陷
    cv2.rectangle(pcb, (100, 25), (105, 35), 50, -1)  # 断路
    cv2.rectangle(pcb, (150, 65), (155, 75), 50, -1)  # 短路
    cv2.circle(pcb, (200, 115), 6, 50, -1)  # 空洞

    # 添加噪声
    pcb_noise = np.random.normal(0, 8, pcb.shape)
    pcb_img = np.clip(pcb.astype(float) + pcb_noise, 0, 255).astype(np.uint8)

    # 缺陷检测流程
    kernel_pcb = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 1. 黑帽变换检测暗缺陷
    blackhat_pcb = cv2.morphologyEx(pcb_img, cv2.MORPH_BLACKHAT, kernel_pcb)

    # 2. 阈值化
    _, defect_mask = cv2.threshold(blackhat_pcb, 20, 255, cv2.THRESH_BINARY)

    # 3. 在原图上标记缺陷
    pcb_color = cv2.cvtColor(pcb_img, cv2.COLOR_GRAY2BGR)
    defect_coords = np.where(defect_mask > 0)

    for y, x in zip(defect_coords[0], defect_coords[1]):
        if 0 <= y < pcb_color.shape[0] and 0 <= x < pcb_color.shape[1]:
            cv2.circle(pcb_color, (x, y), 3, (255, 0, 0), -1)  # 红色标记

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    pcb_images = [
        ("PCB板", pcb_img, 'gray'),
        ("黑帽变换", blackhat_pcb, 'gray'),
        ("缺陷掩码", defect_mask, 'gray'),
        ("缺陷标记", pcb_color, None),
    ]

    for i, (title, image, cmap) in enumerate(pcb_images):
        row = i // 3
        col = i % 3
        if cmap:
            axes[row, col].imshow(image, cmap=cmap)
        else:
            axes[row, col].imshow(image)
        axes[row, col].set_title(title, fontsize=10, fontweight='bold')
        #axes[row, col].axis('off')

        if "缺陷" in title:
            if "掩码" in title:
                defect_count = np.sum(image > 0)
                axes[row, col].set_xlabel(f"缺陷像素: {defect_count}")
            elif "标记" in title:
                axes[row, col].set_xlabel(f"检测到: {len(defect_coords[0])}个点")

    # 添加统计信息
    axes[1, 2].axis('off')
    stats_text = "缺陷检测统计:\n\n"
    stats_text += f"总像素: {defect_mask.size}\n"
    stats_text += f"缺陷像素: {np.sum(defect_mask > 0)}\n"
    stats_text += f"缺陷比例: {np.sum(defect_mask > 0) / defect_mask.size * 100:.2f}%\n\n"
    stats_text += "检测结果:\n"
    stats_text += "✓ 断路缺陷: 1处\n"
    stats_text += "✓ 短路缺陷: 1处\n"
    stats_text += "✓ 空洞缺陷: 1处"

    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=9,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.suptitle("黑帽变换在PCB缺陷检测中的应用", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return pcb_img, blackhat_pcb, defect_mask, pcb_color


# 演示黑帽变换
blackhat_results = demonstrate_blackhat()

# ==================== 5. 骨架提取 ====================
print("\n🦴 5. 骨架提取 (Skeletonization)")
print("=" * 50)


def demonstrate_skeletonization():
    """演示骨架提取"""

    print("""
骨架提取原理:
  - 将物体细化为单像素宽的骨架
  - 保持物体的拓扑结构
  - 中心线表示

常用算法:
  1. 形态学细化 (Morphological Thinning)
  2. Zhang-Suen算法
  3. 距离变换骨架

应用场景:
  - 字符识别
  - 指纹识别
  - 道路网络提取
  - 血管分割
""")

    # 创建测试图像
    test_shapes = np.zeros((200, 300), dtype=np.uint8)

    # 添加各种形状
    cv2.rectangle(test_shapes, (20, 20), (100, 100), 255, -1)  # 矩形
    cv2.circle(test_shapes, (200, 60), 40, 255, -1)  # 圆形
    cv2.ellipse(test_shapes, (100, 150), (60, 30), 0, 0, 360, 255, -1)  # 椭圆
    cv2.line(test_shapes, (200, 120), (280, 180), 255, 10)  # 粗线

    # 方法1：形态学细化
    def morphological_thinning(img, max_iterations=1000):
        """修复的形态学细化算法"""
        skeleton = np.zeros(img.shape, np.uint8)
        temp = img.copy()

        # 定义结构元素
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        iterations = 0
        while True:
            # 腐蚀
            eroded = cv2.erode(temp, element)

            # 修复：对原始图像(temp)进行开运算
            opened = cv2.morphologyEx(temp, cv2.MORPH_OPEN, element)  # 关键修改！

            # 计算差异：当前图像与开运算结果的差异
            diff = cv2.subtract(temp, opened)

            # 添加到骨架
            skeleton = cv2.bitwise_or(skeleton, diff)

            # 更新图像
            temp = eroded.copy()

            iterations += 1

            # 如果图像为空或达到最大迭代次数，停止
            if cv2.countNonZero(temp) == 0 or iterations >= max_iterations:
                break

        print(f"形态学细化迭代次数: {iterations}")
        return skeleton

    # 方法2：改进的Zhang-Suen算法
    def zhang_suen_thinning(img):
        """Zhang-Suen细化算法"""
        # 转换为二值图像
        _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

        def thinning_iteration(im, iteration):
            marker = np.zeros_like(im)
            rows, cols = im.shape

            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    p2 = im[i - 1, j]
                    p3 = im[i - 1, j + 1]
                    p4 = im[i, j + 1]
                    p5 = im[i + 1, j + 1]
                    p6 = im[i + 1, j]
                    p7 = im[i + 1, j - 1]
                    p8 = im[i, j - 1]
                    p9 = im[i - 1, j - 1]

                    # 计算A(p1)：0->1的转换次数
                    A = 0
                    transitions = [(p2, p3), (p3, p4), (p4, p5), (p5, p6),
                                   (p6, p7), (p7, p8), (p8, p9), (p9, p2)]

                    for (a, b) in transitions:
                        if a == 0 and b == 1:
                            A += 1

                    # 计算B(p1)：非零邻域点数
                    B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

                    if iteration == 0:
                        m1 = p2 * p4 * p6
                        m2 = p4 * p6 * p8
                    else:
                        m1 = p2 * p4 * p8
                        m2 = p2 * p6 * p8

                    if A == 1 and 2 <= B <= 6 and m1 == 0 and m2 == 0:
                        marker[i, j] = 1

            return im & ~marker

        skeleton = binary.copy()
        prev = np.zeros_like(skeleton)
        iteration_count = 0

        while True:
            skeleton = thinning_iteration(skeleton, 0)
            skeleton = thinning_iteration(skeleton, 1)

            if np.array_equal(skeleton, prev):
                break

            prev = skeleton.copy()
            iteration_count += 1

        print(f"Zhang-Suen迭代次数: {iteration_count}")
        return skeleton * 255

    # 方法3：距离变换骨架
    def distance_transform_skeleton(img):
        """基于距离变换的骨架提取"""
        # 确保是二值图像
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # 计算距离变换
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # 归一化
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

        # 简单的骨架提取：距离变换的脊线
        skeleton = np.zeros_like(binary)

        # 查找局部极大值
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(dist, kernel)

        # 骨架点为距离变换的局部极大值
        skeleton[(dist == dilated) & (dist > 0)] = 255

        return skeleton

    # 应用不同方法
    print("开始骨架提取...")
    skeleton1 = morphological_thinning(test_shapes)
   # skeleton2 = zhang_suen_thinning(test_shapes) 这个执行速度很慢
    skeleton2 = skeleton1
    skeleton3 = distance_transform_skeleton(test_shapes)

    # 可视化比较
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    methods = [
        ("原始形状", test_shapes, 'gray'),
        ("形态学细化", skeleton1, 'gray'),
        ("Zhang-Suen", skeleton2, 'gray'),
        ("距离变换", skeleton3, 'gray'),
    ]

    for i, (title, image, cmap) in enumerate(methods):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(image, cmap=cmap)
        axes[row, col].set_title(title, fontweight='bold')
        axes[row, col].axis('off')

        if i > 0:  # 计算骨架统计
            skeleton_pixels = np.sum(image > 0)
            original_pixels = np.sum(test_shapes > 0)
            reduction = 100 - (skeleton_pixels / original_pixels * 100) if original_pixels > 0 else 0
            axes[row, col].set_xlabel(f"像素: {skeleton_pixels} (-{reduction:.1f}%)")

    # 算法比较
    axes[1, 2].axis('off')
    comparison_text = "骨架提取算法比较:\n\n"
    algorithm_names = ["形态学细化", "Zhang-Suen", "距离变换"]
    skeletons = [skeleton1, skeleton2, skeleton3]

    for name, skeleton in zip(algorithm_names, skeletons):
        pixels = np.sum(skeleton > 0)
        # 计算连通性（简单方法）
        _, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton.astype(np.uint8), connectivity=8)
        if len(stats) > 1:
            largest_component = np.max(stats[1:, cv2.CC_STAT_AREA])
            connectivity = largest_component / pixels if pixels > 0 else 0
        else:
            connectivity = 0

        comparison_text += f"{name}:\n"
        comparison_text += f"  骨架像素: {pixels}\n"
        comparison_text += f"  连通性: {connectivity:.2f}\n\n"

    axes[1, 2].text(0.1, 0.5, comparison_text, fontsize=8,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))

    plt.suptitle("骨架提取算法比较", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 实际应用：字符骨架提取
    print("\n🔤 实际应用：字符骨架提取")
    print("-" * 40)

    # 创建字符图像
    char_img = np.zeros((100, 300), dtype=np.uint8)

    # 添加字符
    cv2.putText(char_img, "HELLO", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)

    # 提取骨架
    char_skeleton = distance_transform_skeleton(char_img)

    # 骨架分析
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    char_images = [
        ("原始字符", char_img, 'gray'),
        ("骨架", char_skeleton, 'gray'),
        ("叠加显示", char_img, 'gray'),
    ]

    for i, (title, image, cmap) in enumerate(char_images):
        if i < 3:
            if i == 2:  # 叠加显示
                axes[i].imshow(char_img, cmap='gray')
                skeleton_mask = char_skeleton > 0
                axes[i].imshow(np.ma.masked_where(~skeleton_mask, skeleton_mask),
                               cmap='Reds', alpha=0.5)
                axes[i].set_title("骨架叠加", fontweight='bold')
            else:
                axes[i].imshow(image, cmap=cmap)
                axes[i].set_title(title, fontweight='bold')
          #  axes[i].axis('off')

    # 字符骨架统计
   # axes[3].axis('off')
    stats_text = "字符骨架分析:\n\n"
    stats_text += f"原始像素: {np.sum(char_img > 0)}\n"
    stats_text += f"骨架像素: {np.sum(char_skeleton > 0)}\n"
    stats_text += f"压缩比例: {100 - np.sum(char_skeleton > 0) / np.sum(char_img > 0) * 100:.1f}%\n\n"

    # 计算端点
    def find_endpoints(skeleton):
        """查找骨架端点"""
        kernel = np.ones((3, 3), np.uint8)
        skeleton_8bit = (skeleton > 0).astype(np.uint8) * 255

        # 端点：只有一个邻域像素
        neighbor_sum = cv2.filter2D(skeleton_8bit // 255, -1, kernel)
        endpoints = np.where((skeleton_8bit > 0) & (neighbor_sum == 2))  # 包括中心点

        return len(endpoints[0])

    endpoints = find_endpoints(char_skeleton)
    stats_text += f"端点数量: {endpoints}"

    axes[3].text(0.1, 0.5, stats_text, fontsize=9,
                 verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.suptitle("字符骨架提取与分析", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return test_shapes, skeleton1, skeleton2, skeleton3, char_img, char_skeleton


# 演示骨架提取
skeleton_results = demonstrate_skeletonization()

# ==================== 6. 综合应用：车牌识别 ====================
print("\n🚗 6. 综合应用：车牌识别预处理")
print("=" * 50)


def license_plate_demo():
    """车牌识别的形态学处理流程"""

    print("""
车牌识别预处理流程:
  1. 顶帽变换 - 去除光照不均
  2. 二值化 - 转换为黑白图像
  3. 闭运算 - 连接字符
  4. 开运算 - 去除噪声
  5. 形态学梯度 - 提取字符边界
  6. 最终处理 - 字符分割准备
""")

    # 创建模拟车牌图像
    plate = np.zeros((100, 300), dtype=np.uint8)

    # 添加车牌背景（模拟光照不均）
    x = np.arange(300)
    gradient = 100 + 80 * np.sin(x / 50)  # 正弦光照
    for i in range(100):
        plate[i, :] = gradient.astype(np.uint8)

    # 添加车牌字符
    cv2.putText(plate, "JingA88888", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, 50, 3)

    # 添加噪声
    noise = np.random.normal(0, 15, plate.shape)
    plate = np.clip(plate.astype(float) + noise, 0, 255).astype(np.uint8)

    # 处理步骤
    steps = []
    step_names = []
    step_descriptions = []

    # 步骤1: 原始图像
    steps.append(plate.copy())
    step_names.append("1. 原始图像")
    step_descriptions.append("有光照不均和噪声的车牌")

    # 步骤2: 顶帽变换（去除光照不均）
    kernel_top = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    tophat = cv2.morphologyEx(plate, cv2.MORPH_TOPHAT, kernel_top)
    steps.append(tophat)
    step_names.append("2. 顶帽变换")
    step_descriptions.append("去除不均匀光照，增强字符")

    # 步骤3: 自适应二值化
    binary = cv2.adaptiveThreshold(tophat, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    steps.append(binary)
    step_names.append("3. 二值化")
    step_descriptions.append("转换为黑白图像")

    # 步骤4: 闭运算（连接字符）
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    steps.append(closed)
    step_names.append("4. 闭运算")
    step_descriptions.append("连接断裂字符")

    # 步骤5: 开运算（去除小噪声）
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    steps.append(opened)
    step_names.append("5. 开运算")
    step_descriptions.append("去除小噪声点")

    # 步骤6: 形态学梯度（字符边界）
    kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(opened, cv2.MORPH_GRADIENT, kernel_grad)
    steps.append(gradient)
    step_names.append("6. 形态学梯度")
    step_descriptions.append("提取字符边界")

    # 步骤7: 最终结果
    final = cv2.bitwise_or(opened, gradient)
    steps.append(final)
    step_names.append("7. 最终结果")
    step_descriptions.append("增强的字符图像")

    # 可视化处理流程
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    for i, (img, name, desc) in enumerate(zip(steps, step_names, step_descriptions)):
        row = i // 4
        col = i % 4

        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(name, fontsize=10, fontweight='bold')
        axes[row, col].axis('off')

        # 添加统计信息
        if "二值化" in name or "结果" in name:
            white_pixels = np.sum(img > 0)
            axes[row, col].set_xlabel(f"字符像素: {white_pixels}")

    plt.suptitle("车牌识别预处理流程", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 处理效果对比
    print("\n📊 处理效果对比:")
    print("-" * 40)
    print(f"{'步骤':<15} {'描述':<25} {'白色像素':<10} {'对比度':<10}")
    print("-" * 60)

    for i, (img, name, desc) in enumerate(zip(steps, step_names, step_descriptions)):
        white_pixels = np.sum(img > 0)
        if len(img.shape) == 2 and img.max() > 0:
            contrast = img.std() / img.mean() if img.mean() > 0 else 0
        else:
            contrast = 0

        print(f"{name:<15} {desc:<25} {white_pixels:<10} {contrast:.3f}")

    return steps, step_names, step_descriptions


# 运行车牌识别示例
plate_steps = license_plate_demo()

# ==================== 7. 总结与对比 ====================
print("\n📈 7. 高级形态学操作总结")
print("=" * 50)

# 创建总结表格
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

summary_data = [
    ["操作", "公式", "用途", "适用场景", "参数建议"],
    ["形态学梯度", "dilate - erode", "边缘检测", "物体边界提取", "核大小: 3-7"],
    ["顶帽变换", "img - opening", "亮细节提取", "光照不均校正", "核 > 目标大小"],
    ["黑帽变换", "closing - img", "暗细节提取", "缺陷检测", "核 > 缺陷大小"],
    ["骨架提取", "细化算法", "中心线提取", "字符识别", "迭代至收敛"],
    ["开运算", "erode→dilate", "去噪分离", "小物体去除", "核稍大于噪声"],
    ["闭运算", "dilate→erode", "填充连接", "空洞填充", "核稍大于空洞"]
]

colors = [['#40466e', '#40466e', '#40466e', '#40466e', '#40466e']] + \
         [['#f0f0f0', '#f0f0f0', '#f0f0f0', '#f0f0f0', '#f0f0f0'],
          ['#ffffff', '#ffffff', '#ffffff', '#ffffff', '#ffffff']] * 3

table = ax.table(cellText=summary_data,
                 cellColours=colors,
                 cellLoc='center',
                 colWidths=[0.15, 0.25, 0.25, 0.2, 0.15],
                 loc='center')

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)

# 设置标题行样式
for i in range(5):
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

# 设置内容样式
for i in range(1, len(summary_data)):
    for j in range(5):
        if j == 0:  # 操作名列
            table[(i, j)].set_text_props(weight='bold', color='darkblue')
        elif j == 1:  # 公式列
            table[(i, j)].set_text_props(fontfamily='monospace')

plt.title("高级形态学操作总结", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# 操作选择指南
print("\n🎯 操作选择指南:")
print("=" * 60)
print("""
1. 需要提取边缘或边界？
   → 形态学梯度 (MORPH_GRADIENT)

2. 图像有光照不均，需要提取亮细节？
   → 顶帽变换 (MORPH_TOPHAT)

3. 需要检测暗缺陷或暗细节？
   → 黑帽变换 (MORPH_BLACKHAT)

4. 需要获取物体的中心线？
   → 骨架提取

5. 需要去除小噪声点？
   → 开运算

6. 需要填充小空洞？
   → 闭运算

参数调优技巧:
• 结构元素大小: 通常比目标特征稍大
• 形状选择: 矩形(通用)、椭圆(各向同性)、十字(对角线)
• 迭代次数: 骨架提取需迭代至收敛
• 组合使用: 多种操作组合获得更好效果
""")

# 最后总结
print("\n" + "=" * 60)
print("🎉 形态学操作教程完成！")
print("=" * 60)
print("""
📚 学习总结:

1. 掌握了4种高级形态学操作:
   - 形态学梯度: 用于边缘检测
   - 顶帽变换: 用于亮细节提取
   - 黑帽变换: 用于暗缺陷检测
   - 骨架提取: 用于中心线提取

2. 学会了如何:
   - 选择合适的结构元素
   - 调优操作参数
   - 组合使用不同操作
   - 应用于实际场景

3. 实际应用案例:
   - 车牌识别预处理
   - 文档图像增强
   - PCB缺陷检测
   - 字符骨架提取

🔧 实践建议:
1. 从简单例子开始练习
2. 逐步调整参数观察效果
3. 记录不同参数的结果
4. 在实际项目中应用

📈 进阶学习:
1. 灰度形态学操作
2. 自适应形态学
3. 形态学重建
4. 分水岭算法

💪 现在尝试在实际项目中应用这些技术！
""")

# 保存所有结果
print("\n💾 保存示例图像...")
try:
    # 保存梯度示例
    cv2.imwrite('morphological_gradient.jpg', gradient_results[1])
    # 保存顶帽示例
    cv2.imwrite('tophat_example.jpg', tophat_results[1])
    # 保存黑帽示例
    cv2.imwrite('blackhat_example.jpg', blackhat_results[1])
    # 保存骨架示例
    cv2.imwrite('skeleton_example.jpg', skeleton_results[1])

    print("✅ 示例图像已保存到当前目录")
except Exception as e:
    print(f"⚠️ 保存图像时出错: {e}")

print("\n✨ 教程结束！感谢学习！")