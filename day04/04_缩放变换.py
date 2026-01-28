"""
文件4：缩放变换实现
学习目标：掌握图片缩放变换的原理和实现
重点：等比例缩放、非等比例缩放、插值算法、质量保持
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("📏 第4天 - 文件4：缩放变换实现")
print("=" * 50)

# ==================== 1. 缩放变换理论 ====================
print("\n🎯 1. 缩放变换理论")
print("=" * 30)

print("""
缩放变换 (Scaling)：

数学定义：
   x' = sx · x
   y' = sy · y

矩阵表示（齐次坐标）：
   [x']   [sx 0  0] [x]
   [y'] = [0  sy 0] [y]
   [1 ]   [0  0  1] [1]

OpenCV使用2×3矩阵：
   M = [sx 0 0]
       [0 sy 0]

特殊情况：
1. 等比例缩放：sx = sy
2. 非等比例缩放：sx ≠ sy
3. 反射：sx或sy为负数

几何意义：
   - 改变图片大小
   - 保持形状但改变尺寸
   - 可等比例或不等比例
""")

# ==================== 2. 创建测试图片 ====================
print("\n🎨 2. 创建测试图片")
print("=" * 30)


def create_test_image_with_details():
    """创建带细节的测试图片"""
    # 创建300x200的图片
    height, width = 200, 300
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 设置渐变背景
    for x in range(width):
        # 从左到右的渐变
        r = int(150 + 100 * x / width)
        g = int(100 + 100 * x / width)
        b = int(50 + 150 * x / width)
        img[:, x] = [b, g, r]  # BGR格式

    # 添加细节图案
    # 1. 细线条网格
    for i in range(0, width, 10):
        cv2.line(img, (i, 0), (i, height), (80, 80, 80), 1)
    for j in range(0, height, 10):
        cv2.line(img, (0, j), (width, j), (80, 80, 80), 1)

    # 2. 圆形图案
    for i in range(3):
        for j in range(4):
            center_x = 50 + j * 80
            center_y = 40 + i * 60
            radius = 20
            color = (0, 0, 255) if (i + j) % 2 == 0 else (0, 255, 0)
            cv2.circle(img, (center_x, center_y), radius, color, 2)
            # 在圆内添加小圆
            cv2.circle(img, (center_x, center_y), 5, (255, 255, 255), -1)

    # 3. 文字
    cv2.putText(img, f"Original: {width}x{height}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "Detail Test Image", (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 4. 对角线
    cv2.line(img, (0, 0), (width, height), (255, 255, 0), 2)
    cv2.line(img, (width, 0), (0, height), (255, 255, 0), 2)

    return img


# 创建测试图片
test_img = create_test_image_with_details()
img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

print(f"测试图片创建完成")
print(f"图片尺寸: {test_img.shape[1]}x{test_img.shape[0]}")
print(f"图片包含: 渐变背景、网格、圆形、文字、对角线等细节")

# 显示原始图片
plt.figure(figsize=(8, 5))
plt.imshow(img_rgb)
plt.title("原始测试图片（带多种细节）")
plt.axis('off')
plt.tight_layout()
plt.show()

# ==================== 3. 缩放变换实现 ====================
print("\n📏 3. 缩放变换实现")
print("=" * 30)


def scale_image(image, scale_x, scale_y=None, interpolation=cv2.INTER_LINEAR):
    """
    缩放图片

    参数:
        image: 输入图片
        scale_x: x方向缩放比例
        scale_y: y方向缩放比例，如果为None则与scale_x相同
        interpolation: 插值方法

    返回:
        缩放后的图片
    """
    height, width = image.shape[:2]

    if scale_y is None:
        scale_y = scale_x

    # 计算新尺寸
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)

    print(f"缩放参数:")
    print(f"  原始尺寸: {width}x{height}")
    print(f"  缩放比例: sx={scale_x:.2f}, sy={scale_y:.2f}")
    print(f"  新尺寸: {new_width}x{new_height}")
    print(f"  插值方法: {interpolation}")

    # 应用缩放变换
    scaled = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    return scaled, (new_width, new_height)


# 测试不同的缩放参数
print("\n测试不同的缩放参数:")

# 案例1：等比例放大1.5倍
print("\n案例1: 等比例放大1.5倍")
scaled1, size1 = scale_image(test_img, 1.5)

# 案例2：等比例缩小0.5倍
print("\n案例2: 等比例缩小0.5倍")
scaled2, size2 = scale_image(test_img, 0.5)

# 案例3：非等比例缩放（宽放大，高缩小）
print("\n案例3: 非等比例缩放 (1.8x, 0.6y)")
scaled3, size3 = scale_image(test_img, 1.8, 0.6)

# 案例4：只改变宽度
print("\n案例4: 只改变宽度 (2.0x, 1.0y)")
scaled4, size4 = scale_image(test_img, 2.0, 1.0)

# 案例5：只改变高度
print("\n案例5: 只改变高度 (1.0x, 0.7y)")
scaled5, size5 = scale_image(test_img, 1.0, 0.7)

# ==================== 4. 显示缩放结果 ====================
print("\n🖼️ 4. 显示缩放结果")
print("=" * 30)

# 创建对比图
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# 原始图片
axes[0, 0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"原始图片\n{test_img.shape[1]}x{test_img.shape[0]}")
axes[0, 0].axis('off')

# 案例1：等比例放大1.5倍
axes[0, 1].imshow(cv2.cvtColor(scaled1, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f"等比例放大1.5倍\n{size1[0]}x{size1[1]}")
axes[0, 1].axis('off')

# 案例2：等比例缩小0.5倍
axes[0, 2].imshow(cv2.cvtColor(scaled2, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f"等比例缩小0.5倍\n{size2[0]}x{size2[1]}")
axes[0, 2].axis('off')

# 案例3：非等比例缩放
axes[1, 0].imshow(cv2.cvtColor(scaled3, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f"非等比例缩放\n1.8x, 0.6y\n{size3[0]}x{size3[1]}")
axes[1, 0].axis('off')

# 案例4：只改变宽度
axes[1, 1].imshow(cv2.cvtColor(scaled4, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f"只改变宽度\n2.0x, 1.0y\n{size4[0]}x{size4[1]}")
axes[1, 1].axis('off')

# 案例5：只改变高度
axes[1, 2].imshow(cv2.cvtColor(scaled5, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title(f"只改变高度\n1.0x, 0.7y\n{size5[0]}x{size5[1]}")
axes[1, 2].axis('off')

# 显示缩放原理
axes[2, 0].text(0.1, 0.5,
                "缩放变换总结：\n\n"
                "等比例缩放：\n"
                "  sx = sy\n"
                "  保持宽高比\n\n"
                "非等比例缩放：\n"
                "  sx ≠ sy\n"
                "  改变宽高比\n\n"
                "OpenCV函数：\n"
                "cv2.resize(image,\n"
                "          (new_w, new_h),\n"
                "          interpolation)",
                fontsize=10, verticalalignment='center')
axes[2, 0].set_title("缩放变换原理")
axes[2, 0].axis('off')

# 显示缩放矩阵
axes[2, 1].text(0.1, 0.5,
                "缩放矩阵：\n\n"
                "齐次坐标形式：\n"
                "[sx 0  0]\n"
                "[0  sy 0]\n"
                "[0  0  1]\n\n"
                "OpenCV格式（2×3）：\n"
                "[sx 0 0]\n"
                "[0 sy 0]\n\n"
                "缩放因子：\n"
                "sx, sy > 1: 放大\n"
                "0 < sx, sy < 1: 缩小\n"
                "sx, sy < 0: 反射+缩放",
                fontsize=10, verticalalignment='center')
axes[2, 1].set_title("缩放矩阵")
axes[2, 1].axis('off')

# 显示尺寸变化
original_area = test_img.shape[0] * test_img.shape[1]
scaled_areas = [size[0] * size[1] for size in [size1, size2, size3, size4, size5]]
scaled_ratios = [area / original_area for area in scaled_areas]

axes[2, 2].bar(['放大1.5x', '缩小0.5x', '非等比', '只改宽', '只改高'],
               scaled_ratios, color=['red', 'blue', 'green', 'purple', 'orange'])
axes[2, 2].set_title("面积变化比例")
axes[2, 2].set_ylabel("面积比例（相对原始）")
axes[2, 2].grid(True, alpha=0.3)
axes[2, 2].tick_params(axis='x', rotation=45)

plt.suptitle("缩放变换效果演示", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 5. 插值算法比较 ====================
print("\n🔍 5. 插值算法比较")
print("=" * 30)

print("""
缩放时的插值算法：

1. INTER_NEAREST: 最近邻插值
   - 速度最快，质量最低
   - 有明显的锯齿
   - 适合像素艺术

2. INTER_LINEAR: 双线性插值（默认）
   - 速度和质量平衡
   - 适用于大多数情况
   - 轻微的模糊

3. INTER_CUBIC: 双三次插值
   - 质量更好，速度较慢
   - 更平滑的边缘
   - 适合放大图片

4. INTER_AREA: 区域插值
   - 缩小图片时效果最好
   - 避免莫尔纹
   - 放大时类似INTER_NEAREST

5. INTER_LANCZOS4: Lanczos插值
   - 最高质量，最慢速度
   - 适合高质量放大
""")

# 创建一个小图片用于测试插值
small_img = np.zeros((20, 20, 3), dtype=np.uint8)
# 创建图案
for i in range(20):
    for j in range(20):
        if (i + j) % 4 == 0:
            small_img[i, j] = [0, 0, 255]  # 红色
        if (i * j) % 7 == 0:
            small_img[i, j] = [0, 255, 0]  # 绿色
        if (i - j) % 5 == 0:
            small_img[i, j] = [255, 0, 0]  # 蓝色

# 定义不同的插值方法
interpolation_methods = [
    (cv2.INTER_NEAREST, "最近邻插值"),
    (cv2.INTER_LINEAR, "双线性插值"),
    (cv2.INTER_CUBIC, "双三次插值"),
    (cv2.INTER_AREA, "区域插值"),
    (cv2.INTER_LANCZOS4, "Lanczos插值")
]

# 放大10倍观察插值效果
scale_factor = 10
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 原始小图片
axes[0, 0].imshow(cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"原始图片\n20x20")
axes[0, 0].axis('off')

# 放大后的图片
for idx, (method, title) in enumerate(interpolation_methods, 1):
    row, col = idx // 3, idx % 3
    resized = cv2.resize(small_img, (20 * scale_factor, 20 * scale_factor), interpolation=method)
    axes[row, col].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(f"{title}\n放大{scale_factor}倍")
    axes[row, col].axis('off')

plt.suptitle("不同插值算法的效果对比（放大10倍）", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# 测试缩小时的插值效果
print("\n测试缩小时的插值效果（从200x200缩小到50x50）:")

# 创建200x200的测试图片
medium_img = np.zeros((200, 200, 3), dtype=np.uint8)
# 添加细节
for i in range(0, 200, 10):
    cv2.line(medium_img, (i, 0), (i, 200), (150, 150, 150), 1)
    cv2.line(medium_img, (0, i), (200, i), (150, 150, 150), 1)
# 添加一些文字
cv2.putText(medium_img, "OpenCV", (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(medium_img, "Python", (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 原始图片
axes[0, 0].imshow(cv2.cvtColor(medium_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"原始图片\n200x200")
axes[0, 0].axis('off')

# 缩小后的图片
for idx, (method, title) in enumerate(interpolation_methods, 1):
    row, col = idx // 3, idx % 3
    resized = cv2.resize(medium_img, (50, 50), interpolation=method)
    # 放大显示以便观察
    resized_big = cv2.resize(resized, (200, 200), interpolation=cv2.INTER_NEAREST)
    axes[row, col].imshow(cv2.cvtColor(resized_big, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(f"{title}\n缩小到50x50")
    axes[row, col].axis('off')

plt.suptitle("不同插值算法的效果对比（缩小4倍）", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 6. 缩放变换的数学验证 ====================
print("\n🧮 6. 缩放变换的数学验证")
print("=" * 30)


def verify_scaling():
    """验证缩放变换的数学正确性"""

    # 定义测试点
    test_points = np.array([
        [1, 0],  # 右侧点
        [0, 2],  # 下方点
        [3, 4],  # 斜向点
        [-1, 1]  # 左上点
    ], dtype=np.float32)

    # 缩放参数
    sx, sy = 2.0, 0.5

    print(f"缩放参数: sx={sx}, sy={sy}")
    print(f"验证点缩放变换:")
    print("-" * 40)

    for i, point in enumerate(test_points):
        x, y = point

        # 手动计算
        x_manual = x * sx
        y_manual = y * sy

        # 矩阵计算
        M = np.array([[sx, 0, 0], [0, sy, 0]])
        point_homo = np.append(point, 1)  # 齐次坐标
        point_transformed = np.dot(M, point_homo)
        x_matrix = point_transformed[0]
        y_matrix = point_transformed[1]

        # 验证结果
        match = abs(x_manual - x_matrix) < 1e-10 and abs(y_manual - y_matrix) < 1e-10

        print(f"点 {i}: ({x}, {y})")
        print(f"  手动计算: ({x_manual}, {y_manual})")
        print(f"  矩阵计算: ({x_matrix:.1f}, {y_matrix:.1f})")
        print(f"  结果一致: {'✓' if match else '✗'}")
        print()


verify_scaling()

# ==================== 7. 实际应用案例 ====================
print("\n💼 7. 实际应用案例")
print("=" * 30)

print("""
缩放变换的实际应用：

1. 缩略图生成：快速显示图片预览
2. 响应式设计：适配不同屏幕尺寸
3. 图片预处理：统一输入尺寸用于机器学习
4. 打印优化：调整图片到打印尺寸
5. 内存优化：减少大图片的内存占用
""")


# 演示缩略图生成
def demonstrate_thumbnail_generation():
    """演示缩略图生成"""

    # 模拟不同尺寸的图片
    image_sizes = [(800, 600), (1200, 800), (600, 900), (1000, 1000)]
    thumbnails = []

    print("生成缩略图（统一缩放到200x150）:")
    print("-" * 40)

    for i, (width, height) in enumerate(image_sizes, 1):
        # 创建模拟图片
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(img, f"Image {i}: {width}x{height}",
                    (width // 4, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    min(width, height) / 400, (255, 255, 255), 2)

        # 计算缩放比例，保持宽高比
        target_width, target_height = 200, 150
        scale = min(target_width / width, target_height / height)

        # 生成缩略图
        thumb = cv2.resize(img, (0, 0), fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)

        # 如果缩略图尺寸小于目标尺寸，填充
        if thumb.shape[1] < target_width or thumb.shape[0] < target_height:
            padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            y_offset = (target_height - thumb.shape[0]) // 2
            x_offset = (target_width - thumb.shape[1]) // 2
            padded[y_offset:y_offset + thumb.shape[0],
            x_offset:x_offset + thumb.shape[1]] = thumb
            thumb = padded

        thumbnails.append(thumb)

        print(f"图片{i} ({width}x{height}) → 缩略图 ({thumb.shape[1]}x{thumb.shape[0]})")

    # 显示结果
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for i, (size, thumb) in enumerate(zip(image_sizes, thumbnails)):
        row, col = i // 2, i % 2
        axes[row, col * 2].text(0.5, 0.5, f"原图: {size[0]}x{size[1]}",
                                ha='center', va='center', fontsize=12)
        axes[row, col * 2].set_title(f"图片{i + 1}")
        axes[row, col * 2].axis('off')

        axes[row, col * 2 + 1].imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
        axes[row, col * 2 + 1].set_title(f"缩略图: {thumb.shape[1]}x{thumb.shape[0]}")
        axes[row, col * 2 + 1].axis('off')

    plt.suptitle("缩略图生成演示", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return thumbnails


# 演示缩略图生成
thumbnails = demonstrate_thumbnail_generation()


# 演示图片预处理（统一尺寸）
def demonstrate_image_preprocessing():
    """演示图片预处理（统一尺寸）"""

    # 模拟不同尺寸的训练图片
    train_images = []
    sizes = [(28, 28), (32, 32), (64, 64), (128, 128)]

    for i, (h, w) in enumerate(sizes):
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        cv2.putText(img, f"{w}x{h}", (w // 4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        train_images.append(img)

    # 统一缩放到64x64
    target_size = (64, 64)
    preprocessed_images = []

    print(f"\n图片预处理：统一缩放到{target_size[0]}x{target_size[1]}")
    print("-" * 40)

    for i, img in enumerate(train_images):
        h, w = img.shape[:2]
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        preprocessed_images.append(resized)
        print(f"图片{i + 1}: {w}x{h} → {target_size[0]}x{target_size[1]}")

    # 显示结果
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for i, (img, resized) in enumerate(zip(train_images, preprocessed_images)):
        row, col = i // 2, i % 2
        axes[row, col * 2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, col * 2].set_title(f"原图: {img.shape[1]}x{img.shape[0]}")
        axes[row, col * 2].axis('off')

        axes[row, col * 2 + 1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        axes[row, col * 2 + 1].set_title(f"处理后: {resized.shape[1]}x{resized.shape[0]}")
        axes[row, col * 2 + 1].axis('off')

    plt.suptitle("图片预处理：统一尺寸（用于机器学习）", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return train_images, preprocessed_images


# 演示图片预处理
train_imgs, preprocessed_imgs = demonstrate_image_preprocessing()

# ==================== 8. 缩放变换的逆变换 ====================
print("\n🔄 8. 缩放变换的逆变换")
print("=" * 30)

print("""
缩放变换的逆变换：

如果缩放矩阵是 S = [sx 0 0]
                   [0 sy 0]

那么逆矩阵是 S⁻¹ = [1/sx 0    0]
                  [0    1/sy 0]

注意：当sx或sy为0时，逆变换不存在
""")


def demonstrate_inverse_scaling():
    """演示逆缩放变换"""

    # 创建简单图片
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 70), (0, 0, 255), -1)  # 红色方块
    cv2.putText(img, "Test", (35, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 缩放参数
    scale_factor = 0.5

    # 正向缩放（缩小）
    scaled_down = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor,
                             interpolation=cv2.INTER_LINEAR)

    # 逆向缩放（放大回原尺寸）
    # 注意：由于信息丢失，不能完全恢复
    scaled_up = cv2.resize(scaled_down, (100, 100),
                           interpolation=cv2.INTER_LINEAR)

    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"原始图片\n100x100")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(scaled_down, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"缩小到{scale_factor}倍\n{scaled_down.shape[1]}x{scaled_down.shape[0]}")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(scaled_up, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"放大回100x100\n(有信息损失)")
    axes[2].axis('off')

    plt.suptitle("缩放变换的逆变换（有损变换）", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    # 比较原始和恢复的图片
    original_center = img[40:60, 40:60].mean()
    restored_center = scaled_up[40:60, 40:60].mean()

    print(f"原始图片中心区域平均值: {original_center:.1f}")
    print(f"恢复图片中心区域平均值: {restored_center:.1f}")
    print(f"差异: {abs(original_center - restored_center):.1f}")
    print("注意：缩放是有损变换，信息无法完全恢复")

    return img, scaled_down, scaled_up


# 演示逆变换
img_orig, img_down, img_up = demonstrate_inverse_scaling()

# ==================== 9. 练习与挑战 ====================
print("\n💪 9. 练习与挑战")
print("=" * 30)

print("""
练习题：

1. 基础练习：
   a) 将图片等比例放大2倍
   b) 将图片等比例缩小到原图的1/4
   c) 将图片宽度放大1.5倍，高度不变

2. 进阶练习：
   a) 实现批量生成缩略图的功能
   b) 创建函数，保持宽高比将图片缩放到指定大小
   c) 比较不同插值算法在放大和缩小时的性能差异

3. 思考题：
   a) 为什么缩放是有损变换？
   b) 如何选择最适合的插值算法？
   c) 在什么情况下应该使用非等比例缩放？
""")

# 练习框架代码
print("\n💻 练习框架代码：")

print("""
# 练习1a: 等比例放大2倍
def exercise_1a(image):
    height, width = image.shape[:2]
    scaled = cv2.resize(image, (width*2, height*2), 
                       interpolation=cv2.INTER_CUBIC)
    return scaled

# 练习2b: 保持宽高比缩放
def resize_keep_aspect_ratio(image, target_size):
    # target_size: (target_width, target_height)
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # 计算缩放比例
    scale = min(target_w/w, target_h/h)

    # 计算新尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 缩放图片
    resized = cv2.resize(image, (new_w, new_h), 
                        interpolation=cv2.INTER_LINEAR)

    # 如果需要，填充到目标尺寸
    if new_w < target_w or new_h < target_h:
        new_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        new_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return new_image

    return resized

# 练习3a: 缩放的有损性
def demonstrate_lossy_scaling():
    # 创建包含精细细节的图片
    # 多次缩放后观察细节损失
    pass
""")

# ==================== 10. 总结 ====================
print("\n" + "=" * 50)
print("✅ 缩放变换总结")
print("=" * 50)

summary = """
📊 缩放变换核心知识：

1. 数学原理
   - 公式：x' = sx·x, y' = sy·y
   - 矩阵：S = [sx 0 0; 0 sy 0]
   - 逆变换：S⁻¹ = [1/sx 0 0; 0 1/sy 0]

2. OpenCV实现
   - 函数：cv2.resize(image, (new_w, new_h), interpolation)
   - 相对缩放：cv2.resize(image, (0,0), fx=scale_x, fy=scale_y)
   - 插值算法：INTER_NEAREST, INTER_LINEAR, INTER_CUBIC等

3. 关键函数
   def scale_image(image, scale_x, scale_y=None):
       if scale_y is None: scale_y = scale_x
       new_w = int(w * scale_x)
       new_h = int(h * scale_y)
       return cv2.resize(image, (new_w, new_h), interpolation)

4. 应用场景
   - 缩略图生成
   - 图片预处理
   - 响应式设计
   - 内存优化

5. 注意事项
   - 缩放是有损操作
   - 选择合适的插值算法
   - 保持宽高比防止变形
   - 大比例缩小可能导致信息丢失

6. 插值算法选择指南
   - 放大图片：INTER_CUBIC 或 INTER_LANCZOS4
   - 缩小图片：INTER_AREA
   - 快速处理：INTER_LINEAR
   - 像素艺术：INTER_NEAREST

🎯 核心代码记忆：
   resized = cv2.resize(img, (new_w, new_h), interpolation=...)

   或相对缩放：
   resized = cv2.resize(img, (0,0), fx=scale_x, fy=scale_y)
"""

print(summary)
print("\n📁 下一个文件: 05_镜像变换.py")
print("  我们将学习图片的镜像变换！")