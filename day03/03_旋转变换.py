"""
文件3：旋转变换实现
学习目标：掌握图片旋转变换的原理和实现
重点：旋转矩阵、旋转中心、角度计算、边界调整
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

print("🔄 第3天 - 文件3：旋转变换实现")
print("=" * 50)

# ==================== 1. 旋转变换理论 ====================
print("\n🎯 1. 旋转变换理论")
print("=" * 30)

print("""
旋转变换 (Rotation)：

数学定义（绕原点旋转角度θ）：
   x' = x·cosθ - y·sinθ
   y' = x·sinθ + y·cosθ

矩阵表示（齐次坐标）：
   [x']   [cosθ -sinθ 0] [x]
   [y'] = [sinθ  cosθ 0] [y]
   [1 ]   [0     0    1] [1]

OpenCV使用2×3矩阵：
   M = [α β (1-α)·center_x - β·center_y]
       [-β α β·center_x + (1-α)·center_y]

其中：
   α = scale·cosθ
   β = scale·sinθ

几何意义：
   - 绕指定点旋转指定角度
   - 可以同时进行缩放
   - 保持形状，改变方向
""")

# ==================== 2. 创建测试图片 ====================
print("\n🎨 2. 创建测试图片")
print("=" * 30)


def create_test_image_with_direction():
    """创建带方向标记的测试图片"""
    # 创建300x200的图片
    height, width = 200, 300
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 设置背景色
    img[:, :] = [40, 40, 100]  # 深蓝色背景

    # 添加网格线
    grid_size = 20
    for i in range(0, width, grid_size):
        cv2.line(img, (i, 0), (i, height), (80, 80, 80), 1)
    for j in range(0, height, grid_size):
        cv2.line(img, (0, j), (width, j), (80, 80, 80), 1)

    # 添加坐标轴
    center_x, center_y = width // 2, height // 2
    cv2.line(img, (0, center_y), (width, center_y), (150, 150, 150), 2)  # x轴
    cv2.line(img, (center_x, 0), (center_x, height), (150, 150, 150), 2)  # y轴

    # 添加原点标记
    cv2.circle(img, (center_x, center_y), 5, (255, 255, 255), -1)
    cv2.putText(img, "O", (center_x + 5, center_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 添加方向箭头（指向右上角）
    arrow_length = 60
    cv2.arrowedLine(img, (center_x, center_y),
                    (center_x + arrow_length, center_y - arrow_length),
                    (0, 255, 255), 3, tipLength=0.2)

    # 添加文字标记
    cv2.putText(img, "N", (center_x, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "S", (center_x, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "W", (10, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "E", (width - 15, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 添加测试形状
    # 红色三角形（指向右侧）
    triangle_pts = np.array([[center_x + 50, center_y],
                             [center_x + 80, center_y - 20],
                             [center_x + 80, center_y + 20]], np.int32)
    cv2.fillPoly(img, [triangle_pts], (0, 0, 255))

    # 绿色矩形
    cv2.rectangle(img, (center_x - 60, center_y - 30),
                  (center_x - 20, center_y + 30), (0, 255, 0), -1)

    # 添加角度标记
    cv2.putText(img, f"Size: {width}x{height}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "Original Image", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img, (center_x, center_y)


# 创建测试图片
test_img, center = create_test_image_with_direction()
img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

print(f"测试图片创建完成")
print(f"图片尺寸: {test_img.shape[1]}x{test_img.shape[0]}")
print(f"旋转中心: {center}")

# 显示原始图片
plt.figure(figsize=(8, 5))
plt.imshow(img_rgb)
plt.title("原始测试图片（带方向标记）")
plt.axis('off')
plt.tight_layout()
plt.show()

# ==================== 3. 旋转变换实现 ====================
print("\n🔄 3. 旋转变换实现")
print("=" * 30)


def rotate_image_manual(image, angle_degrees, center=None, scale=1.0):
    """
    手动实现图片旋转（理解原理用）

    参数:
        image: 输入图片
        angle_degrees: 旋转角度（正数逆时针，负数顺时针）
        center: 旋转中心，如果为None则使用图片中心
        scale: 缩放比例

    返回:
        旋转后的图片
    """
    height, width = image.shape[:2]

    if center is None:
        center = (width // 2, height // 2)

    # 将角度转换为弧度
    angle_rad = math.radians(angle_degrees)
    cos_angle = math.cos(angle_rad) * scale
    sin_angle = math.sin(angle_rad) * scale

    # 计算旋转矩阵
    # 公式：M = [α β (1-α)·center_x - β·center_y]
    #         [-β α β·center_x + (1-α)·center_y]
    center_x, center_y = center
    alpha = cos_angle
    beta = sin_angle

    M = np.float32([
        [alpha, beta, (1 - alpha) * center_x - beta * center_y],
        [-beta, alpha, beta * center_x + (1 - alpha) * center_y]
    ])

    print(f"手动计算旋转矩阵（角度={angle_degrees}°，缩放={scale}）:")
    print(f"  α = cos({angle_degrees}°) * {scale} = {alpha:.3f}")
    print(f"  β = sin({angle_degrees}°) * {scale} = {beta:.3f}")
    print(f"  旋转中心: ({center_x}, {center_y})")
    print(f"  变换矩阵:")
    print(f"  M = [[{alpha:.3f}, {beta:.3f}, {(1 - alpha) * center_x - beta * center_y:.1f}],")
    print(f"       [{-beta:.3f}, {alpha:.3f}, {beta * center_x + (1 - alpha) * center_y:.1f}]]")

    # 应用旋转变换
    rotated = cv2.warpAffine(image, M, (width, height))

    return rotated, M


def rotate_image_opencv(image, angle_degrees, center=None, scale=1.0):
    """
    使用OpenCV内置函数旋转图片

    参数:
        image: 输入图片
        angle_degrees: 旋转角度（正数逆时针，负数顺时针）
        center: 旋转中心，如果为None则使用图片中心
        scale: 缩放比例

    返回:
        旋转后的图片
    """
    height, width = image.shape[:2]

    if center is None:
        center = (width // 2, height // 2)

    # 使用OpenCV内置函数获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle_degrees, scale)

    print(f"OpenCV旋转矩阵（角度={angle_degrees}°，缩放={scale}）:")
    print(f"  旋转中心: {center}")
    print(f"  变换矩阵:")
    print(f"  M = [[{M[0, 0]:.3f}, {M[0, 1]:.3f}, {M[0, 2]:.1f}],")
    print(f"       [{M[1, 0]:.3f}, {M[1, 1]:.3f}, {M[1, 2]:.1f}]]")

    # 应用旋转变换
    rotated = cv2.warpAffine(image, M, (width, height))

    return rotated, M


# 测试不同的旋转参数
print("\n测试不同的旋转参数:")

# 案例1：旋转45度（逆时针）
print("\n案例1: 旋转45度（逆时针）")
rotated1, M1 = rotate_image_opencv(test_img, 45, center)

# 案例2：旋转-30度（顺时针）
print("\n案例2: 旋转-30度（顺时针）")
rotated2, M2 = rotate_image_opencv(test_img, -30, center)

# 案例3：旋转90度
print("\n案例3: 旋转90度")
rotated3, M3 = rotate_image_opencv(test_img, 90, center)

# 案例4：旋转180度
print("\n案例4: 旋转180度")
rotated4, M4 = rotate_image_opencv(test_img, 180, center)

# 案例5：旋转45度并缩放0.8倍
print("\n案例5: 旋转45度并缩放0.8倍")
rotated5, M5 = rotate_image_opencv(test_img, 45, center, 0.8)

# 案例6：旋转45度并缩放1.2倍
print("\n案例6: 旋转45度并缩放1.2倍")
rotated6, M6 = rotate_image_opencv(test_img, 45, center, 1.2)

# ==================== 4. 显示旋转结果 ====================
print("\n🖼️ 4. 显示旋转结果")
print("=" * 30)

# 创建对比图
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# 原始图片
axes[0, 0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("原始图片")
axes[0, 0].axis('off')

# 案例1：旋转45度
axes[0, 1].imshow(cv2.cvtColor(rotated1, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f"旋转45°\n(逆时针)")
axes[0, 1].axis('off')

# 案例2：旋转-30度
axes[0, 2].imshow(cv2.cvtColor(rotated2, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f"旋转-30°\n(顺时针)")
axes[0, 2].axis('off')

# 案例3：旋转90度
axes[1, 0].imshow(cv2.cvtColor(rotated3, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f"旋转90°")
axes[1, 0].axis('off')

# 案例4：旋转180度
axes[1, 1].imshow(cv2.cvtColor(rotated4, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f"旋转180°")
axes[1, 1].axis('off')

# 案例5：旋转45度缩放0.8倍
axes[1, 2].imshow(cv2.cvtColor(rotated5, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title(f"旋转45°+缩放0.8")
axes[1, 2].axis('off')

# 案例6：旋转45度缩放1.2倍
axes[2, 0].imshow(cv2.cvtColor(rotated6, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title(f"旋转45°+缩放1.2")
axes[2, 0].axis('off')

# 显示旋转矩阵
axes[2, 1].text(0.1, 0.5,
                "旋转变换总结：\n\n"
                "旋转矩阵：\n"
                "M = cv2.getRotationMatrix2D(\n"
                "    center, angle, scale)\n\n"
                "参数说明：\n"
                "angle > 0: 逆时针旋转\n"
                "angle < 0: 顺时针旋转\n"
                "scale = 1: 保持大小\n"
                "scale < 1: 缩小\n"
                "scale > 1: 放大",
                fontsize=10, verticalalignment='center')
axes[2, 1].set_title("旋转变换原理")
axes[2, 1].axis('off')

# 显示三角函数值
angles = [0, 30, 45, 60, 90, 180, 270, 360]
angle_info = "常用角度三角函数值：\n\n"
angle_info += "角度  sin     cos\n"
angle_info += "-" * 25 + "\n"
for angle in angles:
    rad = math.radians(angle)
    angle_info += f"{angle:3d}° {math.sin(rad):.3f}  {math.cos(rad):.3f}\n"

axes[2, 2].text(0.1, 0.5, angle_info,
                fontsize=9, verticalalignment='center', fontfamily='monospace')
axes[2, 2].set_title("三角函数参考")
axes[2, 2].axis('off')

plt.suptitle("旋转变换效果演示", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 5. 旋转边界问题与解决 ====================
print("\n🔍 5. 旋转边界问题与解决")
print("=" * 30)

print("""
旋转边界问题：

当图片旋转时，角点会超出原始边界
解决方案：
1. 保持原始画布大小 → 部分内容被裁剪
2. 调整画布大小 → 完整显示旋转后的图片
""")


def rotate_image_with_boundary_adjustment(image, angle_degrees, center=None, scale=1.0):
    """
    旋转图片并调整画布大小以完整显示

    参数:
        image: 输入图片
        angle_degrees: 旋转角度
        center: 旋转中心
        scale: 缩放比例

    返回:
        旋转后的图片（完整显示）
    """
    height, width = image.shape[:2]

    if center is None:
        center = (width // 2, height // 2)

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle_degrees, scale)

    # 计算旋转后的边界框
    cos_angle = abs(math.cos(math.radians(angle_degrees)))
    sin_angle = abs(math.sin(math.radians(angle_degrees)))

    # 新宽度和高度
    new_width = int((height * sin_angle) + (width * cos_angle))
    new_height = int((height * cos_angle) + (width * sin_angle))

    # 调整旋转矩阵的平移部分，使中心对齐
    M[0, 2] += (new_width / 2) - center[0]
    M[1, 2] += (new_height / 2) - center[1]

    print(f"边界调整：")
    print(f"  原始尺寸: {width}x{height}")
    print(f"  新尺寸: {new_width}x{new_height}")
    print(f"  增加: {new_width - width}x{new_height - height}")

    # 应用旋转变换，使用新的画布大小
    rotated = cv2.warpAffine(image, M, (new_width, new_height))

    return rotated, M, (new_width, new_height)


# 演示边界调整
print("\n演示旋转边界调整：")

# 创建一个小测试图片
small_img = np.zeros((100, 100, 3), dtype=np.uint8)
small_img[25:75, 25:75] = [0, 0, 255]  # 红色方块
small_center = (50, 50)

# 旋转45度，不调整边界
rotated_no_adjust, _ = rotate_image_opencv(small_img, 45, small_center)

# 旋转45度，调整边界
rotated_with_adjust, _, new_size = rotate_image_with_boundary_adjustment(small_img, 45, small_center)

# 显示对比
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB))
axes[0].set_title("原始图片\n100x100")
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(rotated_no_adjust, cv2.COLOR_BGR2RGB))
axes[1].set_title("旋转45°\n不调整边界\n(部分被裁剪)")
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(rotated_with_adjust, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"旋转45°\n调整边界\n{new_size[0]}x{new_size[1]}")
axes[2].axis('off')

plt.suptitle("旋转边界调整对比", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

# ==================== 6. 旋转的数学验证 ====================
print("\n🧮 6. 旋转的数学验证")
print("=" * 30)


def verify_rotation():
    """验证旋转变换的数学正确性"""

    # 定义测试点
    test_points = np.array([
        [1, 0],  # 右侧点
        [0, 1],  # 下方点
        [-1, 0],  # 左侧点
        [0, -1]  # 上方点
    ], dtype=np.float32)

    # 旋转角度
    angle_degrees = 30
    angle_rad = math.radians(angle_degrees)

    print(f"旋转角度: {angle_degrees}°")
    print(f"验证点绕原点旋转:")
    print("-" * 40)

    for i, point in enumerate(test_points):
        x, y = point

        # 手动计算旋转
        x_manual = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        y_manual = x * math.sin(angle_rad) + y * math.cos(angle_rad)

        # 使用矩阵计算
        # 创建旋转矩阵
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        point_matrix = np.dot(R, point)
        x_matrix = point_matrix[0]
        y_matrix = point_matrix[1]

        # 验证结果
        match = abs(x_manual - x_matrix) < 1e-10 and abs(y_manual - y_matrix) < 1e-10

        direction = ["右", "下", "左", "上"][i]
        print(f"点 {direction}: ({x}, {y})")
        print(f"  手动计算: ({x_manual:.3f}, {y_manual:.3f})")
        print(f"  矩阵计算: ({x_matrix:.3f}, {y_matrix:.3f})")
        print(f"  结果一致: {'✓' if match else '✗'}")
        print()


verify_rotation()

# 验证绕任意点旋转
print("\n验证绕任意点旋转:")
print("-" * 40)


def rotate_point_around_center(point, center, angle_degrees):
    """计算点绕中心旋转后的位置"""
    x, y = point
    cx, cy = center
    angle_rad = math.radians(angle_degrees)

    # 将点平移到原点
    x_translated = x - cx
    y_translated = y - cy

    # 绕原点旋转
    x_rotated = x_translated * math.cos(angle_rad) - y_translated * math.sin(angle_rad)
    y_rotated = x_translated * math.sin(angle_rad) + y_translated * math.cos(angle_rad)

    # 平移回原位置
    x_final = x_rotated + cx
    y_final = y_rotated + cy

    return (x_final, y_final)


# 测试
point = (10, 5)
center = (0, 0)
angle = 90
result = rotate_point_around_center(point, center, angle)
print(f"点{point}绕中心{center}旋转{angle}°: {result}")

point = (10, 5)
center = (2, 2)
angle = 90
result = rotate_point_around_center(point, center, angle)
print(f"点{point}绕中心{center}旋转{angle}°: ({result[0]:.1f}, {result[1]:.1f})")

# ==================== 7. 实际应用案例 ====================
print("\n💼 7. 实际应用案例")
print("=" * 30)

print("""
旋转变换的实际应用：

1. 图片校正：校正倾斜的文档、照片
2. 数据增强：为机器学习生成多角度训练数据
3. 图片浏览：实现图片旋转查看功能
4. 游戏开发：角色、物体的旋转
5. 计算机视觉：特征点方向归一化
""")


# 演示图片校正应用
def demonstrate_image_correction():
    """演示图片校正应用"""

    # 创建一个"倾斜"的文档图片
    height, width = 200, 300
    doc_img = np.zeros((height, width, 3), dtype=np.uint8)

    # 设置白色背景
    doc_img[:, :] = [255, 255, 255]

    # 添加一些文字行（模拟文档）
    for i in range(5):
        y_pos = 40 + i * 30
        cv2.putText(doc_img, f"Document Line {i + 1}", (30, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # 添加一个边框
    cv2.rectangle(doc_img, (20, 20), (width - 20, height - 20), (0, 0, 0), 2)

    # 倾斜图片（旋转-5度）
    center = (width // 2, height // 2)
    M_tilt = cv2.getRotationMatrix2D(center, -5, 1.0)
    tilted_doc = cv2.warpAffine(doc_img, M_tilt, (width, height),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(200, 200, 200))

    # 校正图片（旋转+5度）
    M_correct = cv2.getRotationMatrix2D(center, 5, 1.0)
    corrected_doc = cv2.warpAffine(tilted_doc, M_correct, (width, height))

    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(cv2.cvtColor(doc_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("原始文档")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(tilted_doc, cv2.COLOR_BGR2RGB))
    axes[1].set_title("倾斜文档（-5°）")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(corrected_doc, cv2.COLOR_BGR2RGB))
    axes[2].set_title("校正后文档")
    axes[2].axis('off')

    plt.suptitle("图片校正应用", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return doc_img, tilted_doc, corrected_doc


# 演示图片校正
doc_orig, doc_tilted, doc_corrected = demonstrate_image_correction()

# 演示数据增强
print("\n演示数据增强（为机器学习生成多角度样本）:")


def demonstrate_data_augmentation():
    """演示数据增强：生成多角度样本"""

    # 创建一个简单的"目标"图片
    target_img = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.circle(target_img, (40, 40), 30, (0, 0, 255), -1)  # 红色圆形
    cv2.arrowedLine(target_img, (40, 40), (70, 40), (255, 255, 255), 2, tipLength=0.2)

    # 生成多个旋转角度
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    augmented_images = []

    for angle in angles:
        M = cv2.getRotationMatrix2D((40, 40), angle, 1.0)
        rotated = cv2.warpAffine(target_img, M, (80, 80))
        augmented_images.append(rotated)

    # 显示增强结果
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))

    for idx, (angle, img) in enumerate(zip(angles, augmented_images)):
        row, col = idx // 4, idx % 4
        axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(f"{angle}°")
        axes[row, col].axis('off')

    plt.suptitle("数据增强：多角度样本生成", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return target_img, augmented_images


# 演示数据增强
target_img, augmented_imgs = demonstrate_data_augmentation()

# ==================== 8. 旋转变换的逆变换 ====================
print("\n🔄 8. 旋转变换的逆变换")
print("=" * 30)

print("""
旋转变换的逆变换：

如果旋转矩阵是 R(θ) = [cosθ -sinθ]
                      [sinθ  cosθ]

那么逆矩阵是 R⁻¹ = R(-θ) = [cosθ  sinθ]
                          [-sinθ cosθ]

即：旋转-θ角度可以回到原始位置
""")


def demonstrate_inverse_rotation():
    """演示逆旋转变换"""

    # 创建简单图片
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (80, 80), (0, 0, 255), -1)  # 红色方块
    cv2.arrowedLine(img, (60, 60), (90, 60), (255, 255, 255), 2, tipLength=0.2)

    center = (60, 60)
    angle = 45

    # 正向旋转
    M_forward = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_forward = cv2.warpAffine(img, M_forward, (120, 120))

    # 逆向旋转（返回原始位置）
    M_inverse = cv2.getRotationMatrix2D(center, -angle, 1.0)
    img_inverse = cv2.warpAffine(img_forward, M_inverse, (120, 120))

    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("原始图片")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img_forward, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"正向旋转{angle}°")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(img_inverse, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"逆向旋转{-angle}°\n(返回原始位置)")
    axes[2].axis('off')

    plt.suptitle("旋转变换的逆变换", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    # 验证是否返回原始位置
    # 比较原始图片和逆变换后图片的中心区域
    original_center = img[50:70, 50:70].mean()
    inverse_center = img_inverse[50:70, 50:70].mean()

    if abs(original_center - inverse_center) < 1:
        print("✓ 验证通过：逆向旋转成功返回原始位置")
    else:
        print("✗ 验证失败：逆向旋转未返回原始位置")

    return img, img_forward, img_inverse


# 演示逆变换
img_orig, img_fwd, img_inv = demonstrate_inverse_rotation()

# ==================== 9. 练习与挑战 ====================
print("\n💪 9. 练习与挑战")
print("=" * 30)

print("""
练习题：

1. 基础练习：
   a) 将图片逆时针旋转30度
   b) 将图片顺时针旋转45度
   c) 将图片旋转90度并缩小到80%

2. 进阶练习：
   a) 实现函数，自动检测并校正倾斜的文档图片
   b) 创建动画，让图片连续旋转
   c) 实现批量处理，将文件夹中所有图片旋转到指定角度

3. 思考题：
   a) 为什么旋转180度的图片看起来是倒置的？
   b) 旋转45度和旋转405度有区别吗？
   c) 如何判断一张图片是否被旋转过？
""")

# 练习框架代码
print("\n💻 练习框架代码：")

print("""
# 练习1a: 逆时针旋转30度
def exercise_1a(image):
    height, width = image.shape[:2]
    center = (width//2, height//2)
    M = cv2.getRotationMatrix2D(center, 30, 1.0)
    rotated = cv2.warpAffine(image, M, (width, height))
    return rotated

# 练习2a: 自动检测并校正倾斜文档
def auto_correct_skew(image):
    # 1. 转换为灰度图
    # 2. 检测边缘
    # 3. 检测直线
    # 4. 计算平均角度
    # 5. 旋转校正
    pass

# 练习3b: 旋转45度和405度的区别
def compare_angles():
    # 旋转45度
    M1 = cv2.getRotationMatrix2D(center, 45, 1.0)
    # 旋转405度 (45 + 360)
    M2 = cv2.getRotationMatrix2D(center, 405, 1.0)
    # 比较两个矩阵
    pass
""")

# ==================== 10. 总结 ====================
print("\n" + "=" * 50)
print("✅ 旋转变换总结")
print("=" * 50)

summary = """
📊 旋转变换核心知识：

1. 数学原理
   - 公式：x' = x·cosθ - y·sinθ, y' = x·sinθ + y·cosθ
   - 矩阵：R = [cosθ -sinθ; sinθ cosθ]
   - 齐次坐标：扩展为3×3矩阵

2. OpenCV实现
   - 函数：cv2.getRotationMatrix2D(center, angle, scale)
   - 应用：cv2.warpAffine(image, M, size)
   - 边界处理：自动调整画布大小

3. 关键函数
   def rotate_image(image, angle, center=None, scale=1.0):
       if center is None:
           center = (w//2, h//2)
       M = cv2.getRotationMatrix2D(center, angle, scale)
       return cv2.warpAffine(image, M, (w, h))

4. 应用场景
   - 图片校正
   - 数据增强
   - 游戏开发
   - 计算机视觉

5. 注意事项
   - angle>0逆时针，angle<0顺时针
   - 旋转中心影响结果
   - 边界裁剪问题
   - 旋转+缩放组合

6. 重要概念
   - 三角函数：sin, cos
   - 逆变换：旋转-θ角度
   - 周期特性：旋转360°回到原位置
   - 边界调整：完整显示旋转后图片

🎯 核心代码记忆：
   M = cv2.getRotationMatrix2D(center, angle, scale)
   result = cv2.warpAffine(img, M, (w, h))
"""

print(summary)
print("\n📁 下一个文件: 04_缩放变换.py")
print("  我们将学习图片的缩放变换！")