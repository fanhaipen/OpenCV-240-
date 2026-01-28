"""
模块2：颜色空间基础
学习目标：理解RGB、灰度、HSV颜色空间
核心概念：颜色模型、通道分离、颜色转换
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("🌈 模块2：颜色空间基础")
print("=" * 50)

# ==================== 理论讲解 ====================
"""
理论部分：不同的颜色"描述语言"

1. RGB颜色模型
   - 加法混色：红+绿+蓝=白色
   - 设备相关（显示器、相机）
   - 三通道相互关联

2. 灰度图像
   - 只有亮度信息，没有颜色
   - 计算：Gray = 0.299R + 0.587G + 0.114B
   - 减少计算量，适合纹理分析

3. HSV颜色模型
   - 更符合人类认知
   - H（色相）：0-180°，颜色种类
   - S（饱和度）：0-255，颜色鲜艳程度
   - V（明度）：0-255，亮度
"""

print("🎓 核心概念讲解")
print("=" * 30)
print("""
颜色空间对比：

| 颜色空间 | 适合解决的问题 | 在AI中的应用 |
|---------|--------------|-------------|
| RGB     | 显示、存储     | 最常用      |
| 灰度    | 纹理分析、边缘检测 | 减少计算量  |
| HSV     | 颜色识别、分割  | 颜色稳定性好 |
| Lab     | 颜色差异计算   | 图像质量评估 |

重要公式：
  灰度 = 0.299×R + 0.587×G + 0.114×B
  （人眼对不同颜色的敏感度权重）
""")

# ==================== 实践1：创建彩色测试图像 ====================
print("\n🔬 实践1：创建彩色测试图像")
print("-" * 30)


def create_color_test_image():
    """创建彩色测试图像"""
    img = np.zeros((200, 300, 3), dtype=np.uint8)

    # 创建颜色条
    colors = [
        ([0, 0, 255], "Red"),  # 红
        ([0, 255, 0], "Green"),  # 绿
        ([255, 0, 0], "Blue"),  # 蓝
        ([0, 255, 255], "Yellow"),  # 黄
        ([255, 0, 255], "Purple"),  # 紫
        ([255, 255, 0], "Cyan")  # 青
    ]

    bar_width = 300 // len(colors)

    for i, (color, name) in enumerate(colors):
        x_start = i * bar_width
        x_end = (i + 1) * bar_width
        img[:, x_start:x_end] = color

        # 添加文字
        cv2.putText(img, name, (x_start + 10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img


# 创建彩色图像
color_img = create_color_test_image()
color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

print("创建彩色测试图像完成")
print(f"图像尺寸: {color_img.shape[1]}x{color_img.shape[0]}")
print(f"颜色模式: BGR (OpenCV默认)")

# ==================== 实践2：RGB转灰度 ====================
print("\n🔬 实践2：RGB转灰度图像")
print("-" * 30)

# 转换为灰度
gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

print("灰度转换公式: Gray = 0.299×R + 0.587×G + 0.114×B")
print(f"原始形状: {color_img.shape} (高度, 宽度, 3通道)")
print(f"灰度形状: {gray_img.shape} (高度, 宽度, 1通道)")
print(f"数据量减少: {color_img.size / gray_img.size:.1f}倍")

# ==================== 实践3：RGB转HSV ====================
print("\n🔬 实践3：RGB转HSV颜色空间")
print("-" * 30)

# 转换为HSV
hsv_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

# 分离HSV通道
h_channel = hsv_img[:, :, 0]  # 色相 (0-179)
s_channel = hsv_img[:, :, 1]  # 饱和度 (0-255)
v_channel = hsv_img[:, :, 2]  # 明度 (0-255)


print("HSV颜色空间:")
print(f"  H通道(色相): {h_channel.min()} - {h_channel.max()} (0-179°)")
print(f"  S通道(饱和度): {s_channel.min()} - {s_channel.max()} (0-255)")
print(f"  V通道(明度): {v_channel.min()} - {v_channel.max()} (0-255)")
# 在分离通道后添加：
print("V通道最小值:", v_channel.min())  # 应该是 255
print("V通道最大值:", v_channel.max())  # 应该是 255

print(f"v_channel 类型: {v_channel.dtype}")
# ==================== 实践4：RGB通道分离 ====================
print("\n🔬 实践4：RGB通道分离")
print("-" * 30)

# 分离BGR通道
b_channel = color_img[:, :, 0]  # 蓝色通道
g_channel = color_img[:, :, 1]  # 绿色通道
r_channel = color_img[:, :, 2]  # 红色通道

print("RGB通道统计:")
print(f"  B通道均值: {b_channel.mean():.1f}, 标准差: {b_channel.std():.1f}")
print(f"  G通道均值: {g_channel.mean():.1f}, 标准差: {g_channel.std():.1f}")
print(f"  R通道均值: {r_channel.mean():.1f}, 标准差: {r_channel.std():.1f}")

# ==================== 显示所有结果 ====================
print("\n📊 显示所有颜色空间结果")
print("-" * 30)

plt.figure(figsize=(15, 10))

# 1. 原始RGB
plt.subplot(3, 4, 1)
plt.imshow(color_rgb)
plt.title("1. 原始RGB")
plt.axis('off')

# 2. 灰度
plt.subplot(3, 4, 2)
plt.imshow(gray_img, cmap='gray')
plt.title("2. 灰度")
plt.axis('off')

# 3. HSV
plt.subplot(3, 4, 3)
plt.imshow(hsv_img)
plt.title("3. HSV颜色空间")
plt.axis('off')

# 4-6. HSV通道
plt.subplot(3, 4, 4)
plt.imshow(h_channel, cmap='hsv')
plt.title("4. H通道 (色相)")
plt.axis('off')

plt.subplot(3, 4, 5)
plt.imshow(s_channel, cmap='gray', vmin=0, vmax=255)
plt.title("5. S通道 (饱和度)")
plt.axis('off')

# 当你不设置 vmin 和 vmax 时，Matplotlib 会默认执行以下操作： 
# 寻找最小值 (\(min\))：在你的数据中找到最小值。寻找最大值 (\(max\))：在你的数据中找到最大值。
# 映射颜色：将 \(min\) 映射为黑色（cmap='gray' 的起点），将 \(max\) 映射为白色（终点）。 为什么全是 255 反而显示为黑色？ 
# 如果你的 v_channel 中所有像素值都是 255： 你的 \(min\) 是 255。你的 \(max\) 也是 255。在这种 \(min==max\) 的极端情况下，Matplotlib 的内部归一化逻辑（Normalization）会失效。
plt.subplot(3, 4, 6)
plt.imshow(v_channel, cmap='gray', vmin=0, vmax=255)
# plt.imshow(v_channel, cmap='gray')
plt.title("6. V通道 (明度)")
plt.axis('off')

# 7-9. RGB通道
plt.subplot(3, 4, 7)
plt.imshow(b_channel, cmap='Blues')
plt.title("7. B通道 (蓝色)")
plt.axis('off')

plt.subplot(3, 4, 8)
plt.imshow(g_channel, cmap='Greens')
plt.title("8. G通道 (绿色)")
plt.axis('off')

plt.subplot(3, 4, 9)
plt.imshow(r_channel, cmap='Reds')
plt.title("9. R通道 (红色)")
plt.axis('off')

# 10. 颜色空间对比
plt.subplot(3, 4, 10)
color_spaces = ["RGB", "灰度", "HSV"]
heights = [color_img.size, gray_img.size, hsv_img.size]
colors = ['red', 'gray', 'orange']
plt.bar(color_spaces, heights, color=colors)
plt.title("10. 数据量对比")
plt.ylabel("字节数")
plt.grid(True, alpha=0.3)

# 11. 颜色空间应用
plt.subplot(3, 4, 11)
plt.text(0.1, 0.5,
         "颜色空间应用：\n\n"
         "RGB: 显示、存储\n"
         "灰度: 人脸检测\n"
         "    文字识别\n"
         "HSV: 颜色跟踪\n"
         "    图像分割",
         fontsize=10)
plt.title("11. 应用场景")
plt.axis('off')

# 12. 转换公式
plt.subplot(3, 4, 12)
plt.text(0.1, 0.5,
         "转换公式：\n\n"
         "RGB→灰度：\n"
         "Gray=0.299R+0.587G+0.114B\n\n"
         "RGB→HSV：\n"
         "V=max(R,G,B)\n"
         "S=(V-min)/V\n"
         "H=60°×(差值)/(V-min)",
         fontsize=8)
plt.title("12. 转换公式")
plt.axis('off')

plt.suptitle("颜色空间转换演示", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 今日总结 ====================
print("\n" + "=" * 50)
print("✅ 模块2学习总结")
print("=" * 50)

summary = """
📊 今日核心概念：

1. RGB颜色模型
   - 加法混色，设备相关
   - 三通道：红、绿、蓝
   - 适合显示和存储

2. 灰度图像
   - 只有亮度，没有颜色
   - 减少计算量，适合纹理分析
   - 公式: 0.299R + 0.587G + 0.114B

3. HSV颜色模型
   - 人类感知模型
   - H: 色相 (颜色种类)
   - S: 饱和度 (颜色鲜艳度)
   - V: 明度 (亮度)

🎯 核心函数：
  - RGB转灰度: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  - RGB转HSV: cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  - 通道分离: b, g, r = cv2.split(img)
  - 通道合并: img = cv2.merge([b, g, r])
"""

print(summary)
print("\n📁 下一个文件: 03_基本几何变换.py")
print("  我们将学习平移、旋转、缩放、镜像等几何变换！")