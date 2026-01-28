"""
镜像变换实现
学习目标：掌握图片镜像变换的原理和实现
重点：水平镜像、垂直镜像、对角线镜像、实际应用
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("🪞 镜像变换实现")
print("=" * 50)

# ==================== 1. 镜像变换理论 ====================
print("\n🎯 1. 镜像变换理论")
print("=" * 30)

print("""
镜像变换 (Mirror/Flipping)：

数学定义：
1. 水平镜像：x' = -x, y' = y
2. 垂直镜像：x' = x, y' = -y
3. 对角线镜像：x' = -x, y' = -y

矩阵表示（齐次坐标）：
1. 水平镜像：
   [x']   [-1 0 width-1] [x]
   [y'] = [0  1 0      ] [y]
   [1 ]   [0  0 1      ] [1]

2. 垂直镜像：
   [x']   [1 0 0      ] [x]
   [y'] = [0 -1 height-1] [y]
   [1 ]   [0 0 1      ] [1]

3. 对角线镜像：
   [x']   [-1 0 width-1 ] [x]
   [y'] = [0  -1 height-1] [y]
   [1 ]   [0  0 1       ] [1]

OpenCV使用flipCode参数：
   flipCode = 0: 垂直镜像
   flipCode = 1: 水平镜像
   flipCode = -1: 同时水平和垂直镜像

几何意义：
   - 类似照镜子的效果
   - 保持形状和大小，改变方向
   - 可用于数据增强、图片校正
""")

# ==================== 2. 创建测试图片 ====================
print("\n🎨 2. 创建测试图片")
print("=" * 30)


def create_asymmetric_test_image():
    """创建非对称测试图片"""
    # 创建300x200的图片
    height, width = 200, 300
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 设置渐变背景
    for x in range(width):
        r = int(100 + 100 * x / width)
        g = int(50 + 150 * x / width)
        b = int(150 + 50 * x / width)
        img[:, x] = [b, g, r]  # BGR格式

    # 添加非对称图案
    # 1. 左侧三角形
    left_triangle = np.array([[50, 50], [50, 150], [150, 100]], np.int32)
    cv2.fillPoly(img, [left_triangle], (0, 0, 255))  # 红色

    # 2. 右侧矩形
    cv2.rectangle(img, (200, 50), (280, 150), (0, 255, 0), -1)  # 绿色

    # 3. 左上角圆形
    cv2.circle(img, (80, 60), 25, (255, 0, 0), -1)  # 蓝色

    # 4. 添加方向文字
    cv2.putText(img, "L", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "R", (260, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "TOP", (140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(img, "BOTTOM", (130, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # 5. 添加坐标轴
    cv2.line(img, (width // 2, 0), (width // 2, height), (200, 200, 200), 1)  # 垂直中线
    cv2.line(img, (0, height // 2), (width, height // 2), (200, 200, 200), 1)  # 水平中线

    # 6. 添加图片信息
    cv2.putText(img, f"Original: {width}x{height}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "Asymmetric Test", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img


# 创建测试图片
test_img = create_asymmetric_test_image()
img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

print(f"测试图片创建完成")
print(f"图片尺寸: {test_img.shape[1]}x{test_img.shape[0]}")
print(f"图片特点: 非对称设计，包含左(L)右(R)标记")

# 显示原始图片
plt.figure(figsize=(8, 5))
plt.imshow(img_rgb)
plt.title("原始测试图片（非对称设计）")
plt.axis('off')
plt.tight_layout()
plt.show()

# ==================== 3. 镜像变换实现 ====================
print("\n🔄 3. 镜像变换实现")
print("=" * 30)


def mirror_image_cv2(image, flip_code):
    """
    使用OpenCV内置函数进行镜像变换

    参数:
        image: 输入图片
        flip_code: 翻转代码
            0: 垂直镜像
            1: 水平镜像
            -1: 同时水平和垂直镜像

    返回:
        镜像后的图片
    """
    if flip_code not in [0, 1, -1]:
        raise ValueError("flip_code must be 0, 1, or -1")

    flip_name = {
        0: "垂直镜像",
        1: "水平镜像",
        -1: "同时水平和垂直镜像"
    }

    print(f"应用{flip_name[flip_code]}: flip_code={flip_code}")

    # 应用镜像变换
    mirrored = cv2.flip(image, flip_code)

    return mirrored


def mirror_image_manual(image, flip_type='horizontal'):
    """
    手动实现镜像变换（理解原理用）

    参数:
        image: 输入图片
        flip_type: 镜像类型
            'horizontal': 水平镜像
            'vertical': 垂直镜像
            'both': 同时水平和垂直镜像

    返回:
        镜像后的图片
    """
    height, width = image.shape[:2]

    if flip_type == 'horizontal':
        # 水平镜像：左右翻转
        print("手动实现水平镜像")
        mirrored = np.zeros_like(image)
        for y in range(height):
            for x in range(width):
                mirrored[y, x] = image[y, width - 1 - x]

    elif flip_type == 'vertical':
        # 垂直镜像：上下翻转
        print("手动实现垂直镜像")
        mirrored = np.zeros_like(image)
        for y in range(height):
            for x in range(width):
                mirrored[y, x] = image[height - 1 - y, x]

    elif flip_type == 'both':
        # 同时水平和垂直镜像
        print("手动实现同时水平和垂直镜像")
        mirrored = np.zeros_like(image)
        for y in range(height):
            for x in range(width):
                mirrored[y, x] = image[height - 1 - y, width - 1 - x]

    else:
        raise ValueError("flip_type must be 'horizontal', 'vertical', or 'both'")

    return mirrored


# 测试不同的镜像变换
print("\n测试不同的镜像变换:")

# 案例1：水平镜像（左右翻转）
print("\n案例1: 水平镜像（左右翻转）")
mirrored_h1 = mirror_image_cv2(test_img, 1)
mirrored_h2 = mirror_image_manual(test_img, 'horizontal')

# 案例2：垂直镜像（上下翻转）
print("\n案例2: 垂直镜像（上下翻转）")
mirrored_v1 = mirror_image_cv2(test_img, 0)
mirrored_v2 = mirror_image_manual(test_img, 'vertical')

# 案例3：同时水平和垂直镜像
print("\n案例3: 同时水平和垂直镜像")
mirrored_b1 = mirror_image_cv2(test_img, -1)
mirrored_b2 = mirror_image_manual(test_img, 'both')

# ==================== 4. 显示镜像结果 ====================
print("\n🖼️ 4. 显示镜像结果")
print("=" * 30)

# 创建对比图
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# 第一行：原始图片和OpenCV实现
axes[0, 0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"原始图片\n{test_img.shape[1]}x{test_img.shape[0]}")
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(mirrored_h1, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f"OpenCV水平镜像\nflip_code=1")
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(mirrored_v1, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f"OpenCV垂直镜像\nflip_code=0")
axes[0, 2].axis('off')

# 第二行：手动实现
axes[1, 0].imshow(cv2.cvtColor(mirrored_h2, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f"手动水平镜像")
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(mirrored_v2, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f"手动垂直镜像")
axes[1, 1].axis('off')

axes[1, 2].imshow(cv2.cvtColor(mirrored_b2, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title(f"手动同时镜像")
axes[1, 2].axis('off')

# 第三行：OpenCV同时镜像和原理说明
axes[2, 0].imshow(cv2.cvtColor(mirrored_b1, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title(f"OpenCV同时镜像\nflip_code=-1")
axes[2, 0].axis('off')

# 显示镜像原理
axes[2, 1].text(0.1, 0.5,
                "镜像变换总结：\n\n"
                "OpenCV函数：\n"
                "cv2.flip(img, flipCode)\n\n"
                "参数说明：\n"
                "flipCode = 0: 垂直镜像\n"
                "flipCode = 1: 水平镜像\n"
                "flipCode = -1: 同时镜像\n\n"
                "数学原理：\n"
                "水平: x' = width-1-x\n"
                "垂直: y' = height-1-y",
                fontsize=10, verticalalignment='center')
axes[2, 1].set_title("镜像变换原理")
axes[2, 1].axis('off')

# 显示矩阵形式
axes[2, 2].text(0.1, 0.5,
                "镜像变换矩阵：\n\n"
                "水平镜像矩阵：\n"
                "[-1 0 width-1]\n"
                "[0  1 0      ]\n"
                "[0  0 1      ]\n\n"
                "垂直镜像矩阵：\n"
                "[1 0 0      ]\n"
                "[0 -1 height-1]\n"
                "[0 0 1      ]\n\n"
                "同时镜像矩阵：\n"
                "[-1 0 width-1 ]\n"
                "[0  -1 height-1]\n"
                "[0  0 1       ]",
                fontsize=9, verticalalignment='center')
axes[2, 2].set_title("镜像变换矩阵")
axes[2, 2].axis('off')

plt.suptitle("镜像变换效果演示", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 5. 镜像变换的数学验证 ====================
print("\n🧮 5. 镜像变换的数学验证")
print("=" * 30)


def verify_mirror_transformation():
    """验证镜像变换的数学正确性"""

    # 图片尺寸
    width, height = 300, 200

    # 定义测试点
    test_points = np.array([
        [0, 0],  # 左上角
        [width - 1, 0],  # 右上角
        [0, height - 1],  # 左下角
        [width - 1, height - 1],  # 右下角
        [width // 2, height // 2]  # 中心点
    ], dtype=np.float32)

    print(f"图片尺寸: {width}x{height}")
    print(f"验证镜像变换:")
    print("-" * 40)

    for i, point in enumerate(test_points):
        x, y = point

        # 水平镜像计算
        x_horizontal = width - 1 - x
        y_horizontal = y

        # 垂直镜像计算
        x_vertical = x
        y_vertical = height - 1 - y

        # 同时镜像计算
        x_both = width - 1 - x
        y_both = height - 1 - y

        print(f"点 {i}: ({int(x)}, {int(y)})")
        print(f"  水平镜像: ({int(x_horizontal)}, {int(y_horizontal)})")
        print(f"  垂直镜像: ({int(x_vertical)}, {int(y_vertical)})")
        print(f"  同时镜像: ({int(x_both)}, {int(y_both)})")
        print()


verify_mirror_transformation()

# ==================== 6. 实际应用案例 ====================
print("\n💼 6. 实际应用案例")
print("=" * 30)

print("""
镜像变换的实际应用：

1. 数据增强：为机器学习生成镜像样本
2. 图片校正：校正扫描文档的方向
3. 游戏开发：角色左右转身效果
4. 图片浏览：提供镜像查看功能
5. 医学影像：生成对称视图辅助诊断
""")


# 演示数据增强应用
def demonstrate_data_augmentation():
    """演示数据增强：生成镜像样本"""

    # 创建一个简单的"目标"图片
    target_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # 绘制一个非对称箭头
    arrow_points = np.array([[50, 20], [80, 50], [60, 50], [60, 80], [40, 80], [40, 50], [20, 50]], np.int32)
    cv2.fillPoly(target_img, [arrow_points], (0, 0, 255))  # 红色箭头
    cv2.putText(target_img, "F", (45, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 生成镜像样本
    mirrored_h = cv2.flip(target_img, 1)  # 水平镜像
    mirrored_v = cv2.flip(target_img, 0)  # 垂直镜像
    mirrored_b = cv2.flip(target_img, -1)  # 同时镜像

    # 显示结果
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    images = [target_img, mirrored_h, mirrored_v, mirrored_b]
    titles = ["原始图片", "水平镜像", "垂直镜像", "同时镜像"]

    for i, (img, title) in enumerate(zip(images, titles)):
        row, col = i // 2, i % 2
        axes[row, col * 2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, col * 2].set_title(title)
        axes[row, col * 2].axis('off')

        # 添加样本标注
        axes[row, col * 2 + 1].text(0.5, 0.5, f"增强样本 {i + 1}\n用于训练模型",
                                    ha='center', va='center', fontsize=12)
        axes[row, col * 2 + 1].set_title("训练样本")
        axes[row, col * 2 + 1].axis('off')

    plt.suptitle("数据增强：生成镜像样本用于机器学习", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return target_img, mirrored_h, mirrored_v, mirrored_b


# 演示数据增强
target, mirrored_h, mirrored_v, mirrored_b = demonstrate_data_augmentation()


# 演示游戏角色转身效果
def demonstrate_game_character():
    """演示游戏角色转身效果"""

    # 创建角色朝右的图片
    char_right = np.zeros((100, 100, 3), dtype=np.uint8)

    # 绘制朝右的角色（简单表示）
    cv2.circle(char_right, (60, 50), 20, (0, 0, 255), -1)  # 红色头部
    cv2.rectangle(char_right, (50, 70), (70, 90), (0, 255, 0), -1)  # 绿色身体
    cv2.putText(char_right, ">", (40, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 朝右标记

    # 通过水平镜像得到朝左的角色
    char_left = cv2.flip(char_right, 1)

    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(cv2.cvtColor(char_right, cv2.COLOR_BGR2RGB))
    axes[0].set_title("角色朝右")
    axes[0].axis('off')

    axes[1].text(0.5, 0.5, "按下左键\n角色转身",
                 ha='center', va='center', fontsize=14)
    axes[1].set_title("游戏事件")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(char_left, cv2.COLOR_BGR2RGB))
    axes[2].set_title("角色朝左（镜像）")
    axes[2].axis('off')

    plt.suptitle("游戏开发：角色转身效果实现", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return char_right, char_left


# 演示游戏角色
char_right, char_left = demonstrate_game_character()

# ==================== 7. 镜像变换的逆变换 ====================
print("\n🔄 7. 镜像变换的逆变换")
print("=" * 30)

print("""
镜像变换的逆变换：

镜像变换是自身的逆变换！
应用两次相同的镜像变换会回到原始图片。

数学上：
M · M = I  （单位矩阵）

所以：
水平镜像的逆变换 = 水平镜像
垂直镜像的逆变换 = 垂直镜像
同时镜像的逆变换 = 同时镜像
""")


def demonstrate_inverse_mirror():
    """演示镜像变换的逆变换"""

    # 创建简单图片
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (90, 90), (0, 0, 255), -1)  # 红色方块
    cv2.putText(img, "ABC", (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 水平镜像
    mirrored_h = cv2.flip(img, 1)

    # 再次水平镜像（逆变换）
    restored_h = cv2.flip(mirrored_h, 1)

    # 验证是否恢复
    is_restored = np.array_equal(img, restored_h)

    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("原始图片")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(mirrored_h, cv2.COLOR_BGR2RGB))
    axes[1].set_title("水平镜像")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(restored_h, cv2.COLOR_BGR2RGB))
    axes[2].set_title("再次水平镜像\n(恢复原始)")
    axes[2].axis('off')

    plt.suptitle("镜像变换的逆变换：M·M = I", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    print(f"验证结果: 图片{'成功' if is_restored else '未能'}恢复")

    return img, mirrored_h, restored_h, is_restored


# 演示逆变换
img_orig, img_mirrored, img_restored, restored = demonstrate_inverse_mirror()

# ==================== 8. 练习与挑战 ====================
print("\n💪 8. 练习与挑战")
print("=" * 30)

print("""
练习题：

1. 基础练习：
   a) 对图片进行水平镜像
   b) 对图片进行垂直镜像
   c) 对图片同时进行水平和垂直镜像

2. 进阶练习：
   a) 实现批量处理，将文件夹中所有图片生成镜像版本
   b) 创建函数，检测图片是否是对称的
   c) 实现图片的任意角度镜像（沿任意直线镜像）

3. 思考题：
   a) 为什么镜像变换是自身的逆变换？
   b) 如何判断两张图片是否是镜像关系？
   c) 在什么情况下应该使用镜像变换？
""")

# 练习框架代码
print("\n💻 练习框架代码：")

print("""
# 练习1a: 水平镜像
def exercise_1a(image):
    mirrored = cv2.flip(image, 1)
    return mirrored

# 练习2b: 检测图片对称性
def check_symmetry(image, axis='vertical'):
    # axis: 'vertical'检测垂直对称，'horizontal'检测水平对称
    if axis == 'vertical':
        half_width = image.shape[1] // 2
        left_half = image[:, :half_width]
        right_half = image[:, half_width:]
        right_half_mirrored = cv2.flip(right_half, 1)
        # 比较左右两半
        diff = np.sum(np.abs(left_half - right_half_mirrored))
        return diff < threshold
    # 类似处理水平对称
    pass

# 练习3b: 判断两张图片是否是镜像关系
def are_mirror_images(img1, img2):
    # 检查img2是否是img1的水平镜像
    img1_mirrored = cv2.flip(img1, 1)
    diff = np.sum(np.abs(img1_mirrored - img2))
    return diff < threshold
""")

# ==================== 9. 总结 ====================
print("\n" + "=" * 50)
print("✅ 镜像变换总结")
print("=" * 50)

summary = """
📊 镜像变换核心知识：

1. 数学原理
   - 水平镜像：x' = width-1-x, y' = y
   - 垂直镜像：x' = x, y' = height-1-y
   - 同时镜像：x' = width-1-x, y' = height-1-y

2. OpenCV实现
   - 函数：cv2.flip(image, flipCode)
   - flipCode=0: 垂直镜像
   - flipCode=1: 水平镜像
   - flipCode=-1: 同时水平和垂直镜像

3. 关键特性
   - 镜像变换是自身的逆变换
   - 保持图片大小不变
   - 改变图片方向
   - 可用于数据增强

4. 应用场景
   - 数据增强：生成训练样本
   - 游戏开发：角色转身
   - 图片校正：方向调整
   - 医学影像：对称分析

5. 注意事项
   - 镜像会改变文字方向（文字会反向）
   - 非对称图案镜像后可能改变意义
   - 某些场景下镜像不适用（如文字识别）

🎯 核心代码记忆：
   mirrored = cv2.flip(img, flipCode)

   其中：
   flipCode = 0  # 垂直镜像
   flipCode = 1  # 水平镜像
   flipCode = -1 # 同时镜像
"""

print(summary)
print("\n📁 下一个文件: 06_组合变换.py")
print("  我们将学习多个变换的组合应用！")

# 测试代码
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🧪 测试代码运行")
    print("=" * 50)

    # 创建简单测试图片
    test_img_small = np.zeros((50, 50, 3), dtype=np.uint8)
    test_img_small[10:40, 10:40] = [0, 0, 255]  # 红色方块

    # 测试水平镜像
    mirrored_h_test = cv2.flip(test_img_small, 1)
    print("水平镜像测试完成")

    # 测试垂直镜像
    mirrored_v_test = cv2.flip(test_img_small, 0)
    print("垂直镜像测试完成")

    # 测试同时镜像
    mirrored_b_test = cv2.flip(test_img_small, -1)
    print("同时镜像测试完成")

    # 验证逆变换
    restored_test = cv2.flip(mirrored_h_test, 1)
    is_correct = np.array_equal(test_img_small, restored_test)
    print(f"逆变换验证: {'通过' if is_correct else '失败'}")

    print("所有测试完成！")
