"""
深入分析uint8类型的显示问题
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

print("🔍 深入分析uint8类型的显示问题")
print("=" * 50)

# 创建一个测试用例
test_img = np.ones((100, 100), dtype=np.uint8) * 255
print("测试数据:")
print(f"  数据类型: {test_img.dtype}")
print(f"  形状: {test_img.shape}")
print(f"  最小值: {test_img.min()}")
print(f"  最大值: {test_img.max()}")
print(f"  所有值都是255: {(test_img == 255).all()}")

# 显示对比
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# 显示1：无vmin/vmax
im1 = axes[0].imshow(test_img, cmap='gray')
axes[0].set_title("1. 无vmin/vmax\n应该显示白色")
axes[0].axis('off')
cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
cbar1.set_label('值')

# 显示2：有vmin/vmax
im2 = axes[1].imshow(test_img, cmap='gray', vmin=0, vmax=255)
axes[1].set_title("2. 有vmin=0, vmax=255\n应该显示白色")
axes[1].axis('off')
cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
cbar2.set_label('值')

# 显示3：检查实际值
axes[2].imshow(test_img, cmap='gray')
axes[2].set_title("3. 检查像素值")
axes[2].axis('off')

# 在图片上显示一些像素值
for i in range(0, 100, 20):
    for j in range(0, 100, 20):
        color = 'white' if test_img[i, j] < 128 else 'black'
        axes[2].text(j, i, str(test_img[i, j]),
                    ha='center', va='center',
                    color=color, fontsize=6)

# 显示4：创建真正的全白图片对比
white_img = np.full((100, 100), 255, dtype=np.uint8)
im4 = axes[3].imshow(white_img, cmap='gray')
axes[3].set_title("4. 真正的全白图片\n对比用")
axes[3].axis('off')
plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# 现在让我们创建一个模拟你情况的问题
print("\n" + "=" * 50)
print("🔬 模拟可能出现的问题")
print("=" * 50)

# 创建一个有问题的v_channel
# 可能的情况：v_channel 实际上是全0，但打印显示255
print("模拟情况1: 数据是0，但打印显示255？")
problem_data = np.zeros((10, 10), dtype=np.uint8)
print(f"  实际值: 全0")
print(f"  但如果你错误地打印了其他变量，可能会显示255")

# 让我们创建一个有隐藏问题的数据
print("\n模拟情况2: 数据有NaN或inf？")
problem_data2 = np.full((10, 10), 255, dtype=np.uint8)
# 在某个位置放入一个特殊值
problem_data2[5, 5] = 0
print(f"  数据: 大部分255，但有一个0")
print(f"  min={problem_data2.min()}, max={problem_data2.max()}")
print(f"  显示时，由于有0，整个图片可能变暗")

# 显示这个有问题的数据
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

im1 = axes[0].imshow(problem_data2, cmap='gray')
axes[0].set_title("有0和255的混合")
axes[0].axis('off')

# 放大显示中心区域
center_region = problem_data2[3:8, 3:8]
im2 = axes[1].imshow(center_region, cmap='gray')
axes[1].set_title("中心区域放大")
axes[1].axis('off')
# 添加数值
for i in range(5):
    for j in range(5):
        axes[1].text(j, i, str(center_region[i, j]),
                    ha='center', va='center',
                    color='red', fontsize=10)

# 直方图
axes[2].hist(problem_data2.ravel(), bins=[0, 1, 254, 255, 256],
            color='blue', alpha=0.7, edgecolor='black')
axes[2].set_title("值分布直方图")
axes[2].set_xlabel("像素值")
axes[2].set_ylabel("频数")
axes[2].set_xticks([0, 1, 254, 255])
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()