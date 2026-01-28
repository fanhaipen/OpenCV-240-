"""
文件2：平移变换实现
学习目标：掌握图片平移变换的原理和实现
重点：平移矩阵、warpAffine函数、边界处理
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("🚀 第3天 - 文件2：平移变换实现")
print("=" * 50)

# ==================== 1. 平移变换理论 ====================
print("\n🎯 1. 平移变换理论")
print("=" * 30)

print("""
平移变换 (Translation)：

数学定义：
   x' = x + tx
   y' = y + ty

矩阵表示（齐次坐标）：
   [x']   [1 0 tx] [x]
   [y'] = [0 1 ty] [y]
   [1 ]   [0 0 1 ] [1]

OpenCV使用2×3矩阵：
   M = [1 0 tx]
       [0 1 ty]

几何意义：
   - 所有点沿相同方向移动相同距离
   - 保持形状、大小、方向不变
   - 是最简单的等距变换
""")

# ==================== 2. 创建测试图片 ====================
print("\n🎨 2. 创建测试图片")
print("=" * 30)


def create_test_image_with_marker():
    """创建带标记的测试图片"""
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
    cv2.line(img, (0, height // 2), (width, height // 2), (150, 150, 150), 2)  # x轴
    cv2.line(img, (width // 2, 0), (width // 2, height), (150, 150, 150), 2)  # y轴

    # 添加原点标记
    origin = (width // 2, height // 2)
    cv2.circle(img, origin, 5, (255, 255, 255), -1)
    cv2.putText(img, "O(0,0)", (origin[0] + 5, origin[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 添加测试形状
    # 红色三角形
    triangle_pts = np.array([[origin[0] - 60, origin[1] - 30],
                             [origin[0] - 90, origin[1] + 30],
                             [origin[0] - 30, origin[1] + 30]], np.int32)
    cv2.fillPoly(img, [triangle_pts], (0, 0, 255))

    # 绿色矩形
    cv2.rectangle(img, (origin[0] + 20, origin[1] - 40),
                  (origin[0] + 80, origin[1] + 20), (0, 255, 0), -1)

    # 蓝色圆形
    cv2.circle(img, (origin[0] - 60, origin[1] + 80), 25, (255, 0, 0), -1)

    # 添加坐标标注
    cv2.putText(img, f"Size: {width}x{height}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "Original Image", (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img, origin


# 创建测试图片
test_img, origin = create_test_image_with_marker()
img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

print(f"测试图片创建完成")
print(f"图片尺寸: {test_img.shape[1]}x{test_img.shape[0]}")
print(f"原点坐标: {origin}")

# 显示原始图片
plt.figure(figsize=(8, 5))
plt.imshow(img_rgb)
plt.title("原始测试图片")
plt.axis('off')
plt.tight_layout()
plt.show()

# ==================== 3. 平移变换实现 ====================
print("\n🔄 3. 平移变换实现")
print("=" * 30)


def translate_image(image, tx, ty, border_mode=cv2.BORDER_CONSTANT, border_value=(0, 0, 0)):
    """
    平移图片

    参数:
        image: 输入图片
        tx: x方向平移量（正数向右，负数向左）
        ty: y方向平移量（正数向下，负数向上）
        border_mode: 边界处理模式
        border_value: 边界填充颜色（当border_mode为BORDER_CONSTANT时使用）

    返回:
        平移后的图片
    """
    height, width = image.shape[:2]

    # 创建平移矩阵
    # 注意：OpenCV使用2×3矩阵，省略齐次坐标的最后一行
    M = np.float32([[1, 0, tx],  # 向右平移tx像素
                    [0, 1, ty]])  # 向下平移ty像素

    print(f"平移矩阵:")
    print(f"  M = [[1, 0, {tx}],")
    print(f"       [0, 1, {ty}]]")
    print(f"  平移量: tx={tx}, ty={ty}")

    # 应用平移变换
    if border_mode == cv2.BORDER_CONSTANT:
        translated = cv2.warpAffine(image, M, (width, height),
                                    borderMode=border_mode, borderValue=border_value)
    else:
        translated = cv2.warpAffine(image, M, (width, height),
                                    borderMode=border_mode)

    return translated, M


# 测试不同的平移参数
print("\n测试不同的平移参数:")

# 案例1：向右平移50像素，向下平移30像素
print("\n案例1: 向右50像素，向下30像素")
translated1, M1 = translate_image(test_img, 50, 30)

# 案例2：向左平移40像素，向上平移20像素
print("\n案例2: 向左40像素，向上20像素")
translated2, M2 = translate_image(test_img, -40, -20)

# 案例3：只向右平移，不上下平移
print("\n案例3: 只向右平移80像素")
translated3, M3 = translate_image(test_img, 80, 0)

# 案例4：大距离平移（部分移出画面）
print("\n案例4: 大距离平移(100, 80)，部分移出画面")
translated4, M4 = translate_image(test_img, 100, 80)

# ==================== 4. 显示平移结果 ====================
print("\n🖼️ 4. 显示平移结果")
print("=" * 30)

# 创建对比图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原始图片
axes[0, 0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("原始图片")
axes[0, 0].axis('off')

# 案例1：向右下平移
axes[0, 1].imshow(cv2.cvtColor(translated1, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f"向右平移50，向下平移30\ntx=50, ty=30")
axes[0, 1].axis('off')

# 案例2：向左上平移
axes[0, 2].imshow(cv2.cvtColor(translated2, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f"向左平移40，向上平移20\ntx=-40, ty=-20")
axes[0, 2].axis('off')

# 案例3：只向右平移
axes[1, 0].imshow(cv2.cvtColor(translated3, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f"只向右平移80\ntx=80, ty=0")
axes[1, 0].axis('off')

# 案例4：大距离平移
axes[1, 1].imshow(cv2.cvtColor(translated4, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f"大距离平移\ntx=100, ty=80")
axes[1, 1].axis('off')

# 显示变换矩阵
axes[1, 2].text(0.1, 0.5,
                "平移变换总结：\n\n"
                "平移矩阵：\n"
                "M = [1 0 tx]\n"
                "    [0 1 ty]\n\n"
                "参数说明：\n"
                "tx > 0: 向右平移\n"
                "tx < 0: 向左平移\n"
                "ty > 0: 向下平移\n"
                "ty < 0: 向上平移\n\n"
                "OpenCV函数：\n"
                "cv2.warpAffine()",
                fontsize=11, verticalalignment='center')
axes[1, 2].set_title("平移变换原理")
axes[1, 2].axis('off')

plt.suptitle("平移变换效果演示", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 5. 边界处理演示 ====================
print("\n🔍 5. 边界处理演示")
print("=" * 30)

print("""
当图片平移后超出边界时，OpenCV提供多种边界处理方式：

1. BORDER_CONSTANT: 用指定颜色填充边界（默认黑色）
2. BORDER_REPLICATE: 复制边缘像素
3. BORDER_REFLECT: 镜像反射边界
4. BORDER_WRAP: 重复图片
5. BORDER_REFLECT_101: 镜像反射，边界像素不重复
""")

# 创建小测试图片
small_img = np.zeros((100, 100, 3), dtype=np.uint8)
small_img[30:70, 30:70] = [0, 0, 255]  # 红色方块

# 定义不同的边界处理方式
border_modes = [
    (cv2.BORDER_CONSTANT, "常量填充", (0, 255, 0)),  # 绿色填充
    (cv2.BORDER_REPLICATE, "复制边缘", None),
    (cv2.BORDER_REFLECT, "镜像反射", None),
    (cv2.BORDER_WRAP, "重复图片", None)
]

# 应用大距离平移，观察不同边界处理
M_big = np.float32([[1, 0, 60],  # 向右平移60
                    [0, 1, 40]])  # 向下平移40

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for idx, (border_mode, title, border_value) in enumerate(border_modes):
    row, col = idx // 2, idx % 2

    if border_mode == cv2.BORDER_CONSTANT and border_value:
        translated = cv2.warpAffine(small_img, M_big, (100, 100),
                                    borderMode=border_mode, borderValue=border_value)
    else:
        translated = cv2.warpAffine(small_img, M_big, (100, 100),
                                    borderMode=border_mode)

    axes[row, col].imshow(cv2.cvtColor(translated, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(f"{title}\n(tx=60, ty=40)")
    axes[row, col].axis('off')

plt.suptitle("不同边界处理方式的效果", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 6. 平移变换的数学验证 ====================
print("\n🧮 6. 平移变换的数学验证")
print("=" * 30)


def verify_translation():
    """验证平移变换的数学正确性"""

    # 定义测试点
    test_points = np.array([
        [0, 0],  # 原点
        [50, 0],  # 右边
        [0, 30],  # 下边
        [-20, -10]  # 左上
    ], dtype=np.float32)

    # 转换为齐次坐标
    points_homogeneous = np.column_stack([test_points, np.ones(len(test_points))])

    # 平移参数
    tx, ty = 25, 15

    # 创建平移矩阵
    M = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

    print(f"平移参数: tx={tx}, ty={ty}")
    print("\n测试点坐标验证:")
    print("-" * 40)

    for i, point in enumerate(test_points):
        # 原始坐标
        x, y = point

        # 手动计算
        x_manual = x + tx
        y_manual = y + ty

        # 矩阵计算
        point_homo = points_homogeneous[i]
        point_transformed = np.dot(M, point_homo)
        x_matrix = point_transformed[0]
        y_matrix = point_transformed[1]

        # 验证结果
        match = abs(x_manual - x_matrix) < 1e-10 and abs(y_manual - y_matrix) < 1e-10

        print(f"点 {i}: ({x}, {y})")
        print(f"  手动计算: ({x_manual}, {y_manual})")
        print(f"  矩阵计算: ({x_matrix:.2f}, {y_matrix:.2f})")
        print(f"  结果一致: {'✓' if match else '✗'}")
        print()


verify_translation()

# ==================== 7. 实际应用案例 ====================
print("\n💼 7. 实际应用案例")
print("=" * 30)

print("""
平移变换的实际应用：

1. 图片对齐：将多张图片对齐到同一位置
2. 数据增强：为机器学习生成训练数据
3. 图片合成：将多个元素放置到正确位置
4. 全景拼接：对齐多张图片以创建全景图
5. 相机校正：校正相机位置偏移
""")


# 演示图片对齐应用
def demonstrate_image_alignment():
    """演示图片对齐应用"""

    # 创建两张有偏移的"相同"图片
    img1 = np.zeros((150, 200, 3), dtype=np.uint8)
    img1[50:100, 50:150] = [0, 0, 255]  # 红色方块

    img2 = np.zeros((150, 200, 3), dtype=np.uint8)
    img2[70:120, 80:180] = [0, 0, 255]  # 红色方块，有偏移

    # 计算偏移量
    offset_x = 50 - 80  # img1的x起始 - img2的x起始
    offset_y = 50 - 70  # img1的y起始 - img2的y起始

    print(f"检测到偏移: x方向{offset_x}像素, y方向{offset_y}像素")
    print(f"对齐img2到img1的位置...")

    # 对齐img2到img1
    M_align = np.float32([[1, 0, -offset_x],  # 注意符号
                          [0, 1, -offset_y]])
    img2_aligned = cv2.warpAffine(img2, M_align, (200, 150))

    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title("参考图片 (img1)")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title("待对齐图片 (img2)")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2RGB))
    axes[2].set_title("对齐后的图片")
    axes[2].axis('off')

    plt.suptitle("图片对齐应用", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return img1, img2, img2_aligned


# 演示图片对齐
img1, img2, img2_aligned = demonstrate_image_alignment()

# ==================== 8. 平移变换的逆变换 ====================
print("\n🔄 8. 平移变换的逆变换")
print("=" * 30)

print("""
平移变换的逆变换：

如果平移矩阵是 M = [1 0 tx]
                  [0 1 ty]

那么逆矩阵是 M⁻¹ = [1 0 -tx]
                  [0 1 -ty]

应用逆变换可以将图片移回原始位置。
""")


def demonstrate_inverse_translation():
    """演示逆平移变换"""

    # 创建简单图片
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    img[40:80, 40:80] = [0, 0, 255]  # 红色方块

    # 定义平移参数
    tx, ty = 30, 20

    # 正向平移
    M_forward = np.float32([[1, 0, tx], [0, 1, ty]])
    img_forward = cv2.warpAffine(img, M_forward, (120, 120))

    # 逆向平移（返回原始位置）
    M_inverse = np.float32([[1, 0, -tx], [0, 1, -ty]])
    img_inverse = cv2.warpAffine(img_forward, M_inverse, (120, 120))

    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("原始图片")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img_forward, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"正向平移\ntx={tx}, ty={ty}")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(img_inverse, cv2.COLOR_BGR2RGB))
    axes[2].set_title("逆向平移\n(返回原始位置)")
    axes[2].axis('off')

    plt.suptitle("平移变换的逆变换", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    # 验证是否返回原始位置
    original_center = img[40:80, 40:80].mean()
    inverse_center = img_inverse[40:80, 40:80].mean()

    if abs(original_center - inverse_center) < 1:
        print("✓ 验证通过：逆向平移成功返回原始位置")
    else:
        print("✗ 验证失败：逆向平移未返回原始位置")

    return img, img_forward, img_inverse


# 演示逆变换
img_orig, img_fwd, img_inv = demonstrate_inverse_translation()

# ==================== 9. 练习与挑战 ====================
print("\n💪 9. 练习与挑战")
print("=" * 30)

print("""
练习题：

1. 基础练习：
   a) 将图片向右平移100像素，向下平移50像素
   b) 将图片向左平移30像素，向上平移20像素
   c) 创建先向右平移60像素，再向左平移60像素的变换，观察结果

2. 进阶练习：
   a) 实现函数，根据鼠标点击位置自动对齐图片
   b) 创建动画，让图片在屏幕上平滑移动
   c) 实现批量处理，将文件夹中所有图片对齐到同一位置

3. 思考题：
   a) 平移变换会改变图片的内容吗？为什么？
   b) 当平移量超过图片大小时会发生什么？
   c) 如何判断两张图片是否只有平移差异？
""")

# 练习框架代码
print("\n💻 练习框架代码：")

print("""
# 练习1a: 向右平移100像素，向下平移50像素
def exercise_1a():
    # 你的代码
    pass

# 练习2a: 鼠标点击对齐
def mouse_callback(event, x, y, flags, param):
    # 处理鼠标事件
    pass

# 练习3c: 判断图片是否只有平移差异
def has_only_translation(img1, img2, threshold=5):
    # 比较两张图片
    # 如果只有平移差异，返回True和偏移量
    # 否则返回False
    pass
""")

# ==================== 10. 总结 ====================
print("\n" + "=" * 50)
print("✅ 平移变换总结")
print("=" * 50)

summary = """
📊 平移变换核心知识：

1. 数学原理
   - 公式：x' = x + tx, y' = y + ty
   - 矩阵：[1 0 tx; 0 1 ty]
   - 逆变换：[1 0 -tx; 0 1 -ty]

2. OpenCV实现
   - 函数：cv2.warpAffine()
   - 参数：图片、变换矩阵、输出尺寸
   - 边界处理：BORDER_CONSTANT等

3. 关键函数
   def translate_image(image, tx, ty):
       M = np.float32([[1, 0, tx], [0, 1, ty]])
       return cv2.warpAffine(image, M, (width, height))

4. 应用场景
   - 图片对齐
   - 数据增强
   - 全景拼接
   - 元素定位

5. 注意事项
   - tx>0向右，tx<0向左
   - ty>0向下，ty<0向上
   - 边界处理影响结果
   - 平移是等距变换，保持形状

🎯 核心代码记忆：
   M = [[1, 0, tx],
        [0, 1, ty]]
   result = cv2.warpAffine(img, M, (w, h))
"""

print(summary)
print("\n📁 下一个文件: 03_旋转变换.py")
print("  我们将学习图片的旋转变换！")