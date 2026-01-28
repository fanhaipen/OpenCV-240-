"""
文件1：几何变换理论回顾
学习目标：巩固几何变换的数学基础
重点：变换矩阵、齐次坐标、变换顺序
"""

import numpy as np
import matplotlib.pyplot as plt

print("📐 第3天 - 文件1：几何变换理论回顾")
print("=" * 50)

# ==================== 1. 坐标系基础 ====================
print("\n🎯 1. 坐标系基础")
print("=" * 30)

print("""
图像坐标系 vs 数学坐标系：

1. 数学坐标系：
   - 原点在左下角
   - x轴向右，y轴向上
   - 点表示为 (x, y)

2. 图像坐标系：
   - 原点在左上角
   - x轴向右，y轴向下
   - 像素访问：img[y, x]  # 先行后列！

注意：OpenCV和Matplotlib都使用图像坐标系
""")

# 演示坐标系
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 数学坐标系
axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
axes[0].set_xlim(-5, 5)
axes[0].set_ylim(-5, 5)
axes[0].grid(True, alpha=0.3)
axes[0].set_title("数学坐标系")
axes[0].set_xlabel("x轴")
axes[0].set_ylabel("y轴")
axes[0].set_aspect('equal')

# 添加坐标轴箭头
axes[0].arrow(0, 0, 4, 0, head_width=0.2, head_length=0.3, fc='r', ec='r')
axes[0].arrow(0, 0, 0, 4, head_width=0.2, head_length=0.3, fc='r', ec='r')

# 图像坐标系
axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
axes[1].set_xlim(-5, 5)
axes[1].set_ylim(5, -5)  # 注意：y轴反向
axes[1].grid(True, alpha=0.3)
axes[1].set_title("图像坐标系")
axes[1].set_xlabel("x轴（列）")
axes[1].set_ylabel("y轴（行）")
axes[1].set_aspect('equal')

# 添加坐标轴箭头
axes[1].arrow(0, 0, 4, 0, head_width=0.2, head_length=0.3, fc='r', ec='r')
axes[1].arrow(0, 0, 0, 4, head_width=0.2, head_length=0.3, fc='r', ec='r')

plt.tight_layout()
plt.show()

# ==================== 2. 向量和矩阵基础 ====================
print("\n🎯 2. 向量和矩阵基础")
print("=" * 30)

print("""
向量：有大小和方向的量
   v = [x, y]  # 二维向量

矩阵：数字的矩形阵列
   M = [a b]
       [c d]

矩阵乘法：行×列
""")

# 演示矩阵乘法
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
v = np.array([2, 3])

print("示例矩阵A:")
print(A)
print("\n示例矩阵B:")
print(B)
print("\n向量v:")
print(v)

# 矩阵乘法
C = np.dot(A, B)
print("\n矩阵乘法 A × B:")
print(C)

# 矩阵与向量乘法
v_transformed = np.dot(A, v)
print("\n矩阵与向量乘法 A × v:")
print(v_transformed)

# ==================== 3. 齐次坐标 ====================
print("\n🎯 3. 齐次坐标")
print("=" * 30)

print("""
为什么要用齐次坐标？

问题：平移无法用2×2矩阵表示
   x' = x + tx
   y' = y + ty

解决方案：增加一维
   点P = (x, y, 1)

平移矩阵：
   [x']   [1 0 tx] [x]
   [y'] = [0 1 ty] [y]
   [1 ]   [0 0 1 ] [1]

优势：
1. 统一表示所有变换
2. 方便组合多个变换
3. 方便处理无穷远点
""")

# 演示齐次坐标
point = np.array([3, 4, 1])  # 齐次坐标
translation_matrix = np.array([
    [1, 0, 5],  # 向右平移5
    [0, 1, 2],  # 向下平移2
    [0, 0, 1]
])

translated_point = np.dot(translation_matrix, point)
print(f"原始点: ({point[0]}, {point[1]})")
print(f"平移矩阵: tx=5, ty=2")
print(f"变换后点: ({translated_point[0]}, {translated_point[1]})")

# ==================== 4. 基本变换矩阵 ====================
print("\n🎯 4. 基本变换矩阵")
print("=" * 30)


def create_rotation_matrix(angle_degrees):
    """创建旋转矩阵（齐次坐标）"""
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    return np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])


def create_scaling_matrix(sx, sy):
    """创建缩放矩阵（齐次坐标）"""
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])


def create_translation_matrix(tx, ty):
    """创建平移矩阵（齐次坐标）"""
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])


# 演示各种变换矩阵
angle = 30
R = create_rotation_matrix(angle)
S = create_scaling_matrix(1.5, 0.8)
T = create_translation_matrix(5, 3)

print(f"旋转矩阵（{angle}度）:")
print(R[:2, :])  # 只显示前两行（OpenCV格式）

print(f"\n缩放矩阵（sx=1.5, sy=0.8）:")
print(S[:2, :])

print(f"\n平移矩阵（tx=5, ty=3）:")
print(T[:2, :])

# ==================== 5. 变换组合 ====================
print("\n🎯 5. 变换组合")
print("=" * 30)

print("""
重要：矩阵乘法不满足交换律
A × B ≠ B × A

变换顺序重要：
先旋转后平移 ≠ 先平移后旋转
""")

# 演示变换顺序的重要性
point = np.array([1, 0, 1])  # 点(1,0)

# 先旋转45度，后平移(2,0)
M1 = np.dot(create_translation_matrix(2, 0), create_rotation_matrix(45))
result1 = np.dot(M1, point)

# 先平移(2,0)，后旋转45度
M2 = np.dot(create_rotation_matrix(45), create_translation_matrix(2, 0))
result2 = np.dot(M2, point)

print(f"点P: ({point[0]}, {point[1]})")
print(f"\n先旋转45度，后平移(2,0): ({result1[0]:.2f}, {result1[1]:.2f})")
print(f"先平移(2,0)，后旋转45度: ({result2[0]:.2f}, {result2[1]:.2f})")

# 可视化变换顺序
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# 创建测试点
points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float32)
points_homogeneous = np.column_stack([points, np.ones(len(points))])

# 原始点
axes[0].plot(points[:, 0], points[:, 1], 'b-o', linewidth=2)
axes[0].fill(points[:, 0], points[:, 1], 'b', alpha=0.3)
axes[0].set_xlim(-1, 4)
axes[0].set_ylim(-1, 4)
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)
axes[0].set_title("原始图形")

# 先旋转后平移
transformed1 = []
for p in points_homogeneous:
    p_trans = np.dot(M1, p)
    transformed1.append(p_trans[:2])
transformed1 = np.array(transformed1)

axes[1].plot(transformed1[:, 0], transformed1[:, 1], 'r-o', linewidth=2)
axes[1].fill(transformed1[:, 0], transformed1[:, 1], 'r', alpha=0.3)
axes[1].set_xlim(-1, 4)
axes[1].set_ylim(-1, 4)
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)
axes[1].set_title("先旋转后平移")

# 先平移后旋转
transformed2 = []
for p in points_homogeneous:
    p_trans = np.dot(M2, p)
    transformed2.append(p_trans[:2])
transformed2 = np.array(transformed2)

axes[2].plot(transformed2[:, 0], transformed2[:, 1], 'g-o', linewidth=2)
axes[2].fill(transformed2[:, 0], transformed2[:, 1], 'g', alpha=0.3)
axes[2].set_xlim(-1, 4)
axes[2].set_ylim(-1, 4)
axes[2].set_aspect('equal')
axes[2].grid(True, alpha=0.3)
axes[2].set_title("先平移后旋转")

plt.tight_layout()
plt.show()

# ==================== 6. 仿射变换矩阵 ====================
print("\n🎯 6. 仿射变换矩阵")
print("=" * 30)

print("""
仿射变换 = 线性变换 + 平移变换

一般形式：
x' = a·x + b·y + tx
y' = c·x + d·y + ty

矩阵形式（齐次坐标）：
[x']   [a b tx] [x]
[y'] = [c d ty] [y]
[1 ]   [0 0 1 ] [1]

OpenCV使用2×3矩阵，省略最后一行：
M = [a b tx]
    [c d ty]
""")


def create_affine_matrix(a, b, c, d, tx, ty):
    """创建仿射变换矩阵"""
    return np.array([
        [a, b, tx],
        [c, d, ty],
        [0, 0, 1]
    ])


# 示例：包含旋转、缩放、剪切的仿射变换
theta = np.radians(30)  # 30度
scale = 1.5
shear = 0.2

M_affine = create_affine_matrix(
    a=scale * np.cos(theta),  # 旋转+缩放
    b=scale * (-np.sin(theta)) + shear,  # 旋转+剪切
    c=scale * np.sin(theta),  # 旋转+缩放
    d=scale * np.cos(theta),  # 旋转+缩放
    tx=10,  # 平移
    ty=5
)

print("仿射变换矩阵示例：")
print("包含：旋转30度 + 缩放1.5倍 + 轻微剪切 + 平移(10,5)")
print("\n变换矩阵（3×3齐次坐标）:")
print(M_affine)
print("\nOpenCV格式（2×3矩阵）:")
print(M_affine[:2, :])

# ==================== 7. 实际应用中的注意事项 ====================
print("\n🎯 7. 实际应用中的注意事项")
print("=" * 30)

print("""
实际图像处理中的考虑：

1. 离散化问题
   - 理论：连续变换
   - 实际：离散像素
   - 解决：插值算法

2. 边界处理
   - 变换后可能超出边界
   - 解决：填充策略

3. 性能考虑
   - 矩阵运算优化
   - 批量处理

4. 数值精度
   - 浮点数误差
   - 整数坐标转换
""")

# 演示离散化问题
point_continuous = np.array([1.7, 2.3])
print(f"\n连续坐标: ({point_continuous[0]}, {point_continuous[1]})")
print(f"最近邻取整: ({int(round(point_continuous[0]))}, {int(round(point_continuous[1]))})")

# ==================== 8. 总结与练习 ====================
print("\n" + "=" * 50)
print("✅ 理论回顾总结")
print("=" * 50)

summary = """
📊 核心概念总结：

1. 坐标系
   - 数学坐标系 vs 图像坐标系
   - 注意：img[y, x] 先行后列

2. 齐次坐标
   - 点表示为 (x, y, 1)
   - 统一所有变换表示
   - 方便组合变换

3. 变换矩阵
   - 平移：[1 0 tx; 0 1 ty; 0 0 1]
   - 旋转：[cosθ -sinθ 0; sinθ cosθ 0; 0 0 1]
   - 缩放：[sx 0 0; 0 sy 0; 0 0 1]

4. 变换顺序
   - 矩阵乘法不满足交换律
   - 从右向左应用变换
   - 先旋转后平移 ≠ 先平移后旋转

5. 仿射变换
   - 线性变换 + 平移
   - 保持直线和平行性
   - OpenCV使用2×3矩阵
"""

print(summary)

# 练习题
print("\n🔍 练习题：")
print("1. 点P(2,3)先旋转90度，再平移(4,5)，计算新坐标")
print("2. 创建先平移(3,2)后缩放2倍的变换矩阵")
print("3. 解释为什么变换顺序很重要")
print("4. 编写函数计算点绕任意点旋转的结果")

# 练习答案框架
print("\n💡 练习参考答案框架：")

print("""
# 1. 点P(2,3)先旋转90度，再平移(4,5)
import numpy as np

def rotate_point(point, angle_degrees):
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return np.dot(R, point)

def translate_point(point, tx, ty):
    return point + np.array([tx, ty])

# 你的代码在这里
""")

print("\n📁 下一个文件: 02_平移变换.py")
print("  我们将动手实现图片的平移变换！")