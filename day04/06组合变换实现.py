"""
组合变换实现
学习目标：掌握多个几何变换的组合应用
重点：变换矩阵组合、变换顺序、仿射变换
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

print("🔀 组合变换实现")
print("=" * 50)

# ==================== 1. 组合变换理论 ====================
print("\n🎯 1. 组合变换理论")
print("=" * 30)

print("""
组合变换 (Combined Transformation)：

数学原理：
多个变换可以通过矩阵乘法组合成一个变换矩阵。

变换顺序：
变换顺序非常重要！矩阵乘法不满足交换律。
M_combined = M3 × M2 × M1
应用顺序：先应用M1，然后M2，最后M3

常见组合：
1. 先平移后旋转 ≠ 先旋转后平移
2. 先缩放后平移 ≠ 先平移后缩放
3. 任意多个变换的组合

OpenCV实现：
可以通过矩阵乘法组合变换矩阵，然后使用warpAffine一次应用。
""")

# ==================== 2. 创建测试图片 ====================
print("\n🎨 2. 创建测试图片")
print("=" * 30)


def create_test_image_for_combined():
    """创建用于组合变换的测试图片"""
    # 创建300x200的图片
    height, width = 200, 300
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 设置背景色
    img[:, :] = [40, 40, 100]  # 深蓝色背景

    # 添加网格
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

    # 添加测试图案
    # 箭头指向右侧
    cv2.arrowedLine(img, (center_x, center_y),
                    (center_x + 60, center_y), (0, 255, 255), 3, tipLength=0.2)

    # 添加数字标记
    cv2.putText(img, "1", (center_x + 30, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "2", (center_x + 60, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 添加图片信息
    cv2.putText(img, f"Original: {width}x{height}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "Combined Transform Test", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img, (center_x, center_y)


# 创建测试图片
test_img, center = create_test_image_for_combined()
img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

print(f"测试图片创建完成")
print(f"图片尺寸: {test_img.shape[1]}x{test_img.shape[0]}")
print(f"中心点: {center}")

# 显示原始图片
plt.figure(figsize=(8, 5))
plt.imshow(img_rgb)
plt.title("原始测试图片（用于组合变换）")
plt.axis('off')
plt.tight_layout()
plt.show()

# ==================== 3. 变换顺序的重要性演示 ====================
print("\n🔄 3. 变换顺序的重要性演示")
print("=" * 30)


def demonstrate_transform_order():
    """演示变换顺序的重要性"""

    height, width = test_img.shape[:2]
    center_x, center_y = center

    # 定义变换参数
    tx, ty = 80, 0  # 平移参数
    angle = 30  # 旋转角度

    print("演示两种变换顺序：")
    print(f"  平移参数: tx={tx}, ty={ty}")
    print(f"  旋转角度: {angle}度")
    print()

    # 情况1：先平移后旋转
    print("情况1: 先平移后旋转")

    # 创建平移矩阵
    M_translate = np.float32([[1, 0, tx],
                              [0, 1, ty]])

    # 创建旋转矩阵（绕图片中心旋转）
    M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 组合变换：先平移后旋转
    # 注意：OpenCV的warpAffine使用M × point，所以变换顺序是M_rotate × M_translate
    # 但我们需要先应用平移，后应用旋转，所以组合矩阵是M_rotate × M_translate
    M_combined1 = np.dot(M_rotate,
                         np.vstack([M_translate, [0, 0, 1]]))[:2, :]

    result1 = cv2.warpAffine(test_img, M_combined1, (width, height))

    # 情况2：先旋转后平移
    print("情况2: 先旋转后平移")

    # 组合变换：先旋转后平移
    M_combined2 = np.dot(
        np.vstack([M_translate, [0, 0, 1]]),
        np.vstack([M_rotate, [0, 0, 1]])
    )[:2, :]

    result2 = cv2.warpAffine(test_img, M_combined2, (width, height))

    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("原始图片")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
    axes[1].set_title("先平移后旋转")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(result2, cv2.COLOR_BGR2RGB))
    axes[2].set_title("先旋转后平移")
    axes[2].axis('off')

    plt.suptitle("变换顺序的重要性演示", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    # 显示变换矩阵
    print("\n变换矩阵对比:")
    print("先平移后旋转的矩阵:")
    print(M_combined1)
    print("\n先旋转后平移的矩阵:")
    print(M_combined2)
    print("\n两个矩阵是否相同?", np.array_equal(M_combined1, M_combined2))

    return result1, result2, M_combined1, M_combined2


# 演示变换顺序
result_order1, result_order2, M1, M2 = demonstrate_transform_order()

# ==================== 4. 多变换组合实现 ====================
print("\n🎯 4. 多变换组合实现")
print("=" * 30)


def create_combined_transform(translations=None, rotations=None, scales=None,
                              center=None, image_size=None):
    """
    创建组合变换矩阵

    参数:
        translations: 平移列表，每个元素为(tx, ty)
        rotations: 旋转列表，每个元素为(角度, 缩放)
        scales: 缩放列表，每个元素为(scale_x, scale_y)
        center: 旋转中心
        image_size: 图片尺寸(width, height)，用于计算默认中心

    返回:
        组合变换矩阵
    """
    if translations is None:
        translations = []
    if rotations is None:
        rotations = []
    if scales is None:
        scales = []

    if image_size is not None and center is None:
        center = (image_size[0] // 2, image_size[1] // 2)

    # 从单位矩阵开始
    M_combined = np.eye(3)

    # 应用缩放变换
    for scale_x, scale_y in scales:
        M_scale = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1]
        ])
        M_combined = np.dot(M_scale, M_combined)

    # 应用旋转变换
    for angle, scale in rotations:
        if center is None:
            raise ValueError("旋转需要指定中心点")

        # 将角度转换为弧度
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad) * scale
        sin_a = math.sin(angle_rad) * scale

        # 绕指定点旋转的矩阵
        cx, cy = center
        M_rotate = np.array([
            [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy],
            [sin_a, cos_a, -sin_a * cx + (1 - cos_a) * cy],
            [0, 0, 1]
        ])
        M_combined = np.dot(M_rotate, M_combined)

    # 应用平移变换
    for tx, ty in translations:
        M_translate = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])
        M_combined = np.dot(M_translate, M_combined)

    # 返回2×3矩阵（OpenCV格式）
    return M_combined[:2, :]


# 测试不同的组合变换
print("\n测试不同的组合变换:")

height, width = test_img.shape[:2]

# 案例1：平移 + 旋转
print("\n案例1: 平移(50,0) + 旋转30度")
M_case1 = create_combined_transform(
    translations=[(50, 0)],
    rotations=[(30, 1.0)],
    center=center,
    image_size=(width, height)
)
result_case1 = cv2.warpAffine(test_img, M_case1, (width, height))

# 案例2：旋转 + 平移
print("\n案例2: 旋转30度 + 平移(50,0)")
M_case2 = create_combined_transform(
    rotations=[(30, 1.0)],
    translations=[(50, 0)],
    center=center,
    image_size=(width, height)
)
result_case2 = cv2.warpAffine(test_img, M_case2, (width, height))

# 案例3：缩放 + 旋转 + 平移
print("\n案例3: 缩放0.8倍 + 旋转45度 + 平移(30,20)")
M_case3 = create_combined_transform(
    scales=[(0.8, 0.8)],
    rotations=[(45, 1.0)],
    translations=[(30, 20)],
    center=center,
    image_size=(width, height)
)
result_case3 = cv2.warpAffine(test_img, M_case3, (width, height))

# 案例4：多个变换组合
print("\n案例4: 复杂组合 (缩放0.7 + 旋转-15 + 平移(40,-20) + 旋转20)")
M_case4 = create_combined_transform(
    scales=[(0.7, 0.7)],
    rotations=[(-15, 1.0), (20, 1.0)],
    translations=[(40, -20)],
    center=center,
    image_size=(width, height)
)
result_case4 = cv2.warpAffine(test_img, M_case4, (width, height))

# ==================== 5. 显示组合变换结果 ====================
print("\n🖼️ 5. 显示组合变换结果")
print("=" * 30)

# 创建对比图
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# 原始图片
axes[0, 0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"原始图片")
axes[0, 0].axis('off')

# 案例1：平移 + 旋转
axes[0, 1].imshow(cv2.cvtColor(result_case1, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f"案例1: 平移+旋转")
axes[0, 1].axis('off')

# 案例2：旋转 + 平移
axes[0, 2].imshow(cv2.cvtColor(result_case2, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f"案例2: 旋转+平移")
axes[0, 2].axis('off')

# 案例3：缩放 + 旋转 + 平移
axes[1, 0].imshow(cv2.cvtColor(result_case3, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f"案例3: 缩放+旋转+平移")
axes[1, 0].axis('off')

# 案例4：复杂组合
axes[1, 1].imshow(cv2.cvtColor(result_case4, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f"案例4: 复杂组合")
axes[1, 1].axis('off')

# 显示组合变换原理
axes[1, 2].text(0.1, 0.5,
                "组合变换原理：\n\n"
                "矩阵乘法组合：\n"
                "M = M3 × M2 × M1\n\n"
                "应用顺序：\n"
                "先应用M1，然后M2，\n"
                "最后M3\n\n"
                "重要：\n"
                "矩阵乘法不满足交换律\n"
                "变换顺序影响结果",
                fontsize=10, verticalalignment='center')
axes[1, 2].set_title("组合变换原理")
axes[1, 2].axis('off')

# 显示变换矩阵示例
axes[2, 0].text(0.1, 0.5,
                "变换矩阵示例：\n\n"
                "平移矩阵：\n"
                "[1 0 tx]\n"
                "[0 1 ty]\n\n"
                "旋转矩阵：\n"
                "[cosθ -sinθ cx(1-cosθ)+cy·sinθ]\n"
                "[sinθ cosθ -cx·sinθ+cy(1-cosθ)]\n\n"
                "缩放矩阵：\n"
                "[sx 0 0]\n"
                "[0 sy 0]",
                fontsize=9, verticalalignment='center')
axes[2, 0].set_title("基本变换矩阵")
axes[2, 0].axis('off')

# 显示组合矩阵
axes[2, 1].text(0.1, 0.5,
                "组合矩阵计算：\n\n"
                "使用齐次坐标：\n"
                "点P = [x, y, 1]ᵀ\n\n"
                "变换应用：\n"
                "P' = M × P\n\n"
                "OpenCV格式：\n"
                "使用2×3矩阵，\n"
                "省略最后一行[0,0,1]",
                fontsize=10, verticalalignment='center')
axes[2, 1].set_title("矩阵计算")
axes[2, 1].axis('off')

# 显示仿射变换
axes[2, 2].text(0.1, 0.5,
                "仿射变换：\n\n"
                "一般形式：\n"
                "x' = a·x + b·y + tx\n"
                "y' = c·x + d·y + ty\n\n"
                "矩阵形式：\n"
                "[a b tx]\n"
                "[c d ty]\n\n"
                "包含：平移、旋转、\n"
                "缩放、剪切等线性变换",
                fontsize=10, verticalalignment='center')
axes[2, 2].set_title("仿射变换")
axes[2, 2].axis('off')

plt.suptitle("组合变换效果演示", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 6. 仿射变换自定义实现 ====================
print("\n🎯 6. 仿射变换自定义实现")
print("=" * 30)


def demonstrate_affine_transform():
    """演示仿射变换自定义实现"""

    height, width = test_img.shape[:2]

    print("仿射变换的一般形式：")
    print("  x' = a·x + b·y + tx")
    print("  y' = c·x + d·y + ty")
    print()

    # 定义仿射变换参数
    # 这里创建一个包含旋转、缩放、剪切的变换
    angle = 30  # 旋转角度
    scale = 0.8  # 缩放比例
    shear = 0.2  # 剪切参数

    # 将角度转换为弧度
    angle_rad = math.radians(angle)

    # 计算仿射变换参数
    a = scale * math.cos(angle_rad) + shear * math.sin(angle_rad)
    b = scale * (-math.sin(angle_rad)) + shear * math.cos(angle_rad)
    c = scale * math.sin(angle_rad)
    d = scale * math.cos(angle_rad)
    tx = 50
    ty = 30

    print("仿射变换参数：")
    print(f"  a = {a:.3f}  (缩放+旋转+剪切)")
    print(f"  b = {b:.3f}  (旋转+剪切)")
    print(f"  c = {c:.3f}  (旋转)")
    print(f"  d = {d:.3f}  (缩放+旋转)")
    print(f"  tx = {tx}")
    print(f"  ty = {ty}")

    # 创建仿射变换矩阵
    M_affine = np.float32([[a, b, tx],
                           [c, d, ty]])

    # 应用仿射变换
    result_affine = cv2.warpAffine(test_img, M_affine, (width, height))

    # 显示结果
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("原始图片")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(result_affine, cv2.COLOR_BGR2RGB))
    axes[1].set_title("自定义仿射变换")
    axes[1].axis('off')

    plt.suptitle("仿射变换自定义实现", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return result_affine, M_affine


# 演示仿射变换
result_affine, M_affine = demonstrate_affine_transform()

# ==================== 7. 实际应用案例 ====================
print("\n💼 7. 实际应用案例")
print("=" * 30)

print("""
组合变换的实际应用：

1. 图片校正：校正倾斜、透视变形的图片
2. 增强现实：将虚拟物体放置在真实场景中
3. 图像配准：将多张图片对齐到同一坐标系
4. 计算机视觉：特征点匹配和图像对齐
5. 数据增强：为机器学习生成复杂的变换样本
""")


# 演示图片校正应用
def demonstrate_image_correction():
    """演示图片校正应用"""

    # 创建一个"倾斜"的文档图片
    height, width = 200, 300
    doc_img = np.zeros((height, width, 3), dtype=np.uint8)

    # 设置白色背景
    doc_img[:, :] = [255, 255, 255]

    # 添加文档内容
    cv2.putText(doc_img, "Document Title", (80, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.line(doc_img, (50, 70), (250, 70), (0, 0, 0), 1)

    for i in range(5):
        y_pos = 100 + i * 25
        cv2.putText(doc_img, f"Line {i + 1}: Sample text for document.",
                    (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 应用倾斜变换（模拟扫描倾斜）
    # 使用仿射变换创建倾斜效果
    pts1 = np.float32([[50, 50], [250, 50], [50, 150]])
    pts2 = np.float32([[60, 40], [260, 60], [40, 160]])  # 轻微倾斜

    M_skew = cv2.getAffineTransform(pts1, pts2)
    skewed_doc = cv2.warpAffine(doc_img, M_skew, (width, height))

    # 校正图片（通过逆变换）
    M_correct = cv2.getAffineTransform(pts2, pts1)
    corrected_doc = cv2.warpAffine(skewed_doc, M_correct, (width, height))

    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(cv2.cvtColor(doc_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("原始文档")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(skewed_doc, cv2.COLOR_BGR2RGB))
    axes[1].set_title("倾斜文档（扫描结果）")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(corrected_doc, cv2.COLOR_BGR2RGB))
    axes[2].set_title("校正后文档")
    axes[2].axis('off')

    plt.suptitle("图片校正应用：文档倾斜校正", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return doc_img, skewed_doc, corrected_doc, M_skew, M_correct


# 演示图片校正
doc_orig, doc_skewed, doc_corrected, M_skew, M_correct = demonstrate_image_correction()

# 演示数据增强
print("\n演示数据增强：为机器学习生成复杂变换样本")


def demonstrate_complex_augmentation():
    """演示复杂数据增强"""

    # 创建简单的目标图片
    target_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(target_img, (50, 50), 30, (0, 0, 255), -1)  # 红色圆形
    cv2.arrowedLine(target_img, (50, 50), (80, 50), (255, 255, 255), 2, tipLength=0.2)

    # 生成多个复杂变换的样本
    augmented_samples = []

    # 定义多个变换组合
    transforms = [
        ("平移+旋转", [(30, 20)], [(45, 1.0)], None, (50, 50)),
        ("旋转+缩放", None, [(30, 1.0)], [(0.8, 0.8)], (50, 50)),
        ("复杂组合", [(20, -10)], [(-15, 1.0), (10, 1.0)], [(1.2, 0.9)], (50, 50)),
        ("仿射变换", None, None, None, None)  # 特殊处理
    ]

    for name, translations, rotations, scales, center in transforms:
        if name == "仿射变换":
            # 自定义仿射变换
            M = np.float32([[0.9, 0.2, 20],
                            [-0.1, 1.1, 15]])
        else:
            M = create_combined_transform(
                translations=translations,
                rotations=rotations,
                scales=scales,
                center=center,
                image_size=(100, 100)
            )

        transformed = cv2.warpAffine(target_img, M, (100, 100))
        augmented_samples.append((name, transformed))

    # 显示结果
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    # 原始图片
    axes[0, 0].imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("原始样本")
    axes[0, 0].axis('off')

    axes[0, 1].text(0.5, 0.5, "数据增强：\n生成多个变换样本\n用于训练机器学习模型",
                    ha='center', va='center', fontsize=10)
    axes[0, 1].set_title("增强目的")
    axes[0, 1].axis('off')

    # 显示增强样本
    for i, (name, img) in enumerate(augmented_samples):
        row, col = (i + 2) // 4, (i + 2) % 4
        axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(name)
        axes[row, col].axis('off')

    plt.suptitle("数据增强：复杂变换样本生成", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return target_img, augmented_samples


# 演示数据增强
target_img, augmented_samples = demonstrate_complex_augmentation()

# ==================== 8. 练习与挑战 ====================
print("\n💪 8. 练习与挑战")
print("=" * 30)

print("""
练习题：

1. 基础练习：
   a) 实现先平移(30,20)后旋转45度的组合变换
   b) 实现先缩放0.7倍后平移(50,0)的组合变换
   c) 实现旋转、平移、缩放的任意组合

2. 进阶练习：
   a) 实现函数，根据三个点的对应关系计算仿射变换矩阵
   b) 实现图片的透视变换（需要4个点）
   c) 实现批量处理，对视频帧应用稳定的组合变换

3. 思考题：
   a) 为什么变换顺序会影响最终结果？
   b) 如何计算组合变换的逆变换？
   c) 在什么情况下应该使用仿射变换而不是单个变换？
""")

# 练习框架代码
print("\n💻 练习框架代码：")

print("""
# 练习1a: 先平移后旋转
def exercise_1a(image, tx=30, ty=20, angle=45):
    height, width = image.shape[:2]
    center = (width//2, height//2)

    # 创建平移矩阵
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])

    # 创建旋转矩阵
    M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 组合：先平移后旋转
    M_combined = np.dot(M_rotate, np.vstack([M_translate, [0, 0, 1]]))[:2, :]

    result = cv2.warpAffine(image, M_combined, (width, height))
    return result

# 练习2a: 根据三个点计算仿射变换矩阵
def get_affine_transform_from_points(src_points, dst_points):
    # src_points: 源图片上的三个点
    # dst_points: 目标位置上的三个点
    # 返回: 仿射变换矩阵

    if len(src_points) != 3 or len(dst_points) != 3:
        raise ValueError("需要三个点")

    src_pts = np.float32(src_points)
    dst_pts = np.float32(dst_points)

    M = cv2.getAffineTransform(src_pts, dst_pts)
    return M

# 练习3b: 计算组合变换的逆变换
def get_inverse_transform(M):
    # M是2×3变换矩阵
    # 转换为3×3齐次坐标矩阵
    M_homo = np.vstack([M, [0, 0, 1]])

    # 计算逆矩阵
    M_inv_homo = np.linalg.inv(M_homo)

    # 返回2×3矩阵
    return M_inv_homo[:2, :]
""")

# ==================== 9. 总结 ====================
print("\n" + "=" * 50)
print("✅ 组合变换总结")
print("=" * 50)

summary = """
📊 组合变换核心知识：

1. 数学原理
   - 通过矩阵乘法组合变换：M = M3 × M2 × M1
   - 应用顺序：从右到左（先M1，后M2，最后M3）
   - 矩阵乘法不满足交换律：A×B ≠ B×A

2. 仿射变换
   - 一般形式：x' = a·x + b·y + tx, y' = c·x + d·y + ty
   - 矩阵形式：[a b tx; c d ty]
   - 包含：平移、旋转、缩放、剪切

3. OpenCV实现
   - 组合矩阵：np.dot(M2, M1)  # 先M1后M2
   - 应用变换：cv2.warpAffine(img, M_combined, size)
   - 点对应：cv2.getAffineTransform(src_pts, dst_pts)

4. 关键函数
   def combine_transforms(translations, rotations, scales, center):
       # 从单位矩阵开始
       M = np.eye(3)
       # 按顺序应用变换
       # 返回M[:2, :] (2×3矩阵)

5. 应用场景
   - 图片校正
   - 图像配准
   - 增强现实
   - 数据增强
   - 计算机视觉

6. 注意事项
   - 变换顺序至关重要
   - 使用齐次坐标方便组合
   - 多次插值会累积误差
   - 组合变换可能改变图片边界

🎯 核心代码记忆：
   # 组合两个变换矩阵
   M_combined = np.dot(M2, np.vstack([M1, [0, 0, 1]]))[:2, :]

   # 应用组合变换
   result = cv2.warpAffine(img, M_combined, (w, h))
"""

print(summary)
print("\n📁 下一个文件: 07_综合项目_图片编辑器.py")
print("  我们将综合运用所有变换知识，构建一个完整的图片编辑器！")
