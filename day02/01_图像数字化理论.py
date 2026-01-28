"""
模块1：图像数字化理论
学习目标：理解像素、分辨率、量化
核心概念：采样、量化、颜色深度
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

print("📘 模块1：图像数字化理论")
print("=" * 50)

# ==================== 理论讲解 ====================
"""
理论部分：图像如何从现实世界变成数字？

1. 采样 (Sampling)
   - 在空间上离散化
   - 决定分辨率：单位长度的像素数量
   - 分辨率 = 宽度 × 高度

2. 量化 (Quantization)
   - 在亮度上离散化
   - 决定颜色深度：每个像素用多少位表示
   - 8位/通道 = 256级亮度 (0-255)

现实世界 → 采样 → 量化 → 数字图像
连续信号 → 离散像素 → 数字值 → 数字矩阵
"""

print("🎓 核心概念讲解")
print("=" * 30)
print("""
1. 像素 (Pixel)
   - 图像的最小单位
   - 每个像素有位置(x,y)和颜色值(R,G,B)

2. 分辨率
   - 图像包含的像素数量
   - 格式：宽度 × 高度 (如1920×1080)
   - 分辨率越高，细节越多

3. 颜色深度
   - 每个颜色通道的位数
   - 8位/通道 = 256级 (0-255)
   - RGB三个通道 = 24位 = 1677万种颜色
""")

# ==================== 实践1：理解像素 ====================
print("\n🔬 实践1：理解像素")
print("-" * 30)

# 创建一个5x5的微小图像
tiny_image = np.zeros((5, 5, 3), dtype=np.uint8)

# 设置一些像素
tiny_image[0, 0] = [255, 0, 0]  # 红色
tiny_image[0, 4] = [0, 255, 0]  # 绿色
tiny_image[4, 0] = [0, 0, 255]  # 蓝色
tiny_image[4, 4] = [255, 255, 0]  # 黄色
tiny_image[2, 2] = [255, 255, 255]  # 白色

print("创建一个5x5的微型图像：")
print(f"形状: {tiny_image.shape} (高度, 宽度, 通道)")
print(f"数据类型: {tiny_image.dtype}")
print(f"总像素: {tiny_image.shape[0] * tiny_image.shape[1]}")

# 显示这个小图像
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(tiny_image)
plt.title("5x5像素图像")
plt.axis('off')

# 添加像素坐标
for i in range(5):
    for j in range(5):
        color = 'white' if np.mean(tiny_image[i, j]) < 128 else 'black'
        plt.text(j, i, f'({i},{j})', ha='center', va='center',
                 color=color, fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor='gray', alpha=0.3))

# ==================== 实践2：分辨率演示 ====================
print("\n🔬 实践2：分辨率对图像的影响")
print("-" * 30)

# 创建高分辨率原图
high_res = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.circle(high_res, (50, 50), 40, (0, 0, 255), -1)  # 红色圆形
cv2.putText(high_res, "High", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# 降低分辨率
low_res = cv2.resize(high_res, (20, 20), interpolation=cv2.INTER_LINEAR)
# 放大回原尺寸以便对比
low_res_big = cv2.resize(low_res, (100, 100), interpolation=cv2.INTER_NEAREST)

plt.subplot(1, 2, 2)
plt.imshow(np.hstack([cv2.cvtColor(high_res, cv2.COLOR_BGR2RGB),
                      cv2.cvtColor(low_res_big, cv2.COLOR_BGR2RGB)]))
plt.title("高分辨率(100x100) vs 低分辨率(20x20)")
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"高分辨率: 100x100 = 10,000像素")
print(f"低分辨率: 20x20 = 400像素")
print(f"像素数量比: 25:1")

# ==================== 实践3：量化演示 ====================
print("\n🔬 实践3：量化位数对图像的影响")
print("-" * 30)

# 创建渐变图像
gradient = np.zeros((50, 256, 3), dtype=np.uint8)
for x in range(256):
    gradient[:, x] = [x, x, x]  # 灰度渐变

plt.figure(figsize=(12, 8))

# 不同量化级别
bit_depths = [8, 4, 2, 1]

for i, bits in enumerate(bit_depths, 1):
    # 量化处理
    levels = 2 ** bits
    quantized = (gradient // (256 // levels)) * (256 // levels)

    plt.subplot(2, 2, i)
    plt.imshow(quantized, cmap='gray')
    plt.title(f'{bits}位量化 = {levels}个灰度级')
    plt.axis('off')

    # 在图像上添加量化步长
    step = 256 // levels
    for level in range(levels):
        x_pos = level * step + step // 2
        plt.text(x_pos, 25, str(level * step), ha='center', va='center',
                 color='red' if level * step < 128 else 'white', fontsize=8)

    print(f"  {bits}位: {levels}个级别, 步长={step}")

plt.suptitle('量化位数对图像的影响', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 今日总结 ====================
print("\n" + "=" * 50)
print("✅ 模块1学习总结")
print("=" * 50)

summary = """
📊 今日核心概念：

1. 采样 (Sampling)
   - 空间离散化，决定分辨率
   - 分辨率 = 宽度 × 高度
   - 高分辨率 = 更多细节

2. 量化 (Quantization)  
   - 亮度离散化，决定颜色深度
   - 8位/通道 = 256级亮度
   - 24位RGB = 1677万种颜色

3. 像素操作
   - 图像[y, x] 访问像素
   - 索引从0开始
   - 注意：先行(y)后列(x)

🎯 核心代码：
  - 创建图像: np.zeros((h, w, 3), dtype=np.uint8)
  - 访问像素: image[y, x] = [b, g, r]
  - 调整分辨率: cv2.resize()
  - 量化处理: image // step * step
"""

print(summary)
print("\n📁 下一个文件: 02_颜色空间基础.py")
print("  我们将学习RGB、HSV、灰度等颜色空间！")