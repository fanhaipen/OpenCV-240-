"""
第5天 - 文件1：图像滤波理论基础
学习目标：理解图像滤波的基本概念、原理和分类
重点：卷积操作、滤波器类型、噪声模型
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

print("🎓 第5天 - 文件1：图像滤波理论基础")
print("=" * 50)

# ==================== 1. 什么是图像滤波 ====================
print("\n🎯 1. 什么是图像滤波")
print("=" * 30)

print("""
图像滤波 (Image Filtering)：

定义：对图像进行局部或全局处理，以增强某些特征或抑制某些特征

为什么需要滤波？
1. 去除噪声：图片拍摄、传输过程中的随机干扰
2. 增强特征：突出边缘、纹理等特征
3. 图像复原：修复受损的图像
4. 图像分析：为后续处理做准备

滤波的本质：
  输入图片 → 滤波器 → 输出图片
  I(x,y)   →  F    →  O(x,y)
""")

# ==================== 2. 图像噪声模型 ====================
print("\n🎯 2. 图像噪声模型")
print("=" * 30)

print("""
常见图像噪声类型：

1. 高斯噪声 (Gaussian Noise)
   - 最常见，呈正态分布
   - 原因：电子电路热噪声
   - 特点：每个像素都受影响，幅度随机

2. 椒盐噪声 (Salt-and-Pepper Noise)
   - 随机出现的黑白点
   - 原因：传输错误、传感器故障
   - 特点：部分像素被极大或极小值替换

3. 均匀噪声 (Uniform Noise)
   - 在一定范围内均匀分布
   - 较少见

4. 泊松噪声 (Poisson Noise)
   - 光子计数噪声
   - 在低光照条件下明显
""")


def create_noisy_images():
    """创建带不同噪声的测试图片"""
    # 创建干净测试图片
    height, width = 200, 300
    clean_img = np.zeros((height, width), dtype=np.uint8)

    # 添加一些图案
    cv2.rectangle(clean_img, (50, 50), (150, 150), 200, -1)  # 灰色矩形
    cv2.circle(clean_img, (225, 100), 40, 150, -1)  # 浅灰圆形
    cv2.line(clean_img, (20, 180), (280, 180), 100, 3)  # 水平线

    # 添加文字
    cv2.putText(clean_img, "Clean Image", (80, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

    # 1. 添加高斯噪声
    gaussian_noise = np.zeros((height, width), dtype=np.uint8)
    cv2.randn(gaussian_noise, 0, 30)  # 均值0，标准差30
    gaussian_img = cv2.add(clean_img, gaussian_noise)

    # 2. 添加椒盐噪声
    salt_pepper_img = clean_img.copy()
    num_salt = int(0.01 * height * width)  # 1%的盐噪声
    num_pepper = int(0.01 * height * width)  # 1%的椒噪声

    # 添加盐噪声（白色点）
    coords = [np.random.randint(0, i - 1, num_salt) for i in clean_img.shape]
    salt_pepper_img[coords[0], coords[1]] = 255

    # 添加椒噪声（黑色点）
    coords = [np.random.randint(0, i - 1, num_pepper) for i in clean_img.shape]
    salt_pepper_img[coords[0], coords[1]] = 0

    # 3. 添加均匀噪声
    uniform_noise = np.random.randint(-30, 30, (height, width), dtype=np.int16)
    uniform_img = np.clip(clean_img.astype(np.int16) + uniform_noise, 0, 255).astype(np.uint8)

    return clean_img, gaussian_img, salt_pepper_img, uniform_img


# 创建噪声图片
clean, gaussian_noisy, salt_pepper_noisy, uniform_noisy = create_noisy_images()

# 显示噪声图片
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

images = [clean, gaussian_noisy, salt_pepper_noisy, uniform_noisy]
titles = ["干净图片", "高斯噪声", "椒盐噪声", "均匀噪声"]

for idx, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis('off')

    # 显示直方图
    if idx > 0:  # 为噪声图片添加直方图
        ax_hist = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
        ax_hist.hist(img.ravel(), bins=50, range=(0, 255), color='blue', alpha=0.7)
        ax_hist.set_xlabel('灰度值')
        ax_hist.set_ylabel('频数')
        ax_hist.set_title('直方图')

plt.suptitle("不同噪声类型对比", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 3. 卷积操作原理 ====================
print("\n🎯 3. 卷积操作原理")
print("=" * 30)

print("""
卷积 (Convolution)：

数学定义：
  I'[i,j] = Σ_{u=-k}^{k} Σ_{v=-k}^{k} K[u,v]·I[i+u, j+v]

其中：
  I: 输入图片
  I': 输出图片
  K: 卷积核（滤波器核）
  k: 卷积核半径

卷积核特性：
1. 大小：通常为奇数（3×3, 5×5, 7×7）
2. 权重：决定滤波器行为
3. 归一化：通常权重和为1（保持亮度）

边界处理：
1. 补零 (Zero Padding)：边界外补0
2. 复制 (Replicate)：复制边界像素
3. 反射 (Reflect)：反射边界像素
4. 循环 (Wrap)：循环使用
""")


def demonstrate_convolution():
    """演示卷积操作"""

    # 创建一个小测试图片
    test_image = np.array([
        [10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50],
        [10, 20, 30, 40, 50]
    ], dtype=np.float32)

    # 定义卷积核
    kernel_3x3 = np.array([
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9]
    ], dtype=np.float32)  # 均值滤波核

    kernel_edge = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)  # 边缘检测核

    print("原始图片 (5×5):")
    print(test_image)

    # 手动计算3×3均值滤波
    print("\n卷积核 (均值滤波 3×3):")
    print(kernel_3x3)

    # 计算输出
    output = np.zeros((3, 3), dtype=np.float32)
    for i in range(0, 3):  # 边界不处理
        for j in range(0, 3):
            # 提取3×3区域
            region = test_image[i:i + 3, j:j + 3]
            # 卷积计算
            output[i, j ] = np.sum(region * kernel_3x3)

    print("\n卷积结果 (3×3):")
    print(output)

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title("原始图片")
    axes[0].grid(True, which='both', color='red', linestyle='-', linewidth=0.5)
    axes[0].set_xticks(range(5))
    axes[0].set_yticks(range(5))

    # 显示卷积核
    axes[1].imshow(kernel_3x3, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title("卷积核 (均值滤波)")
    axes[1].grid(True, which='both', color='white', linestyle='-', linewidth=0.5)
    axes[1].set_xticks(range(3))
    axes[1].set_yticks(range(3))

    # 在核上显示数值
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f'{kernel_3x3[i, j]:.2f}',
                         ha='center', va='center', color='white' if abs(kernel_3x3[i, j]) < 0.5 else 'black')

    axes[2].imshow(output, cmap='gray')
    axes[2].set_title("卷积结果")
    axes[2].grid(True, which='both', color='red', linestyle='-', linewidth=0.5)
    axes[2].set_xticks(range(3))
    axes[2].set_yticks(range(3))

    plt.suptitle("卷积操作演示", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return test_image, kernel_3x3, output


# 演示卷积
test_img, kernel, conv_result = demonstrate_convolution()

# ==================== 4. 滤波器分类 ====================
print("\n🎯 4. 滤波器分类")
print("=" * 30)

print("""
按操作域分类：
1. 空间域滤波 (Spatial Domain)
   - 直接在像素上操作
   - 使用卷积核
   - 如：均值滤波、高斯滤波

2. 频域滤波 (Frequency Domain)
   - 转换到频域处理
   - 使用傅里叶变换
   - 如：低通滤波、高通滤波

按线性性质分类：
1. 线性滤波 (Linear Filtering)
   - 满足叠加性和齐次性
   - 可用卷积表示
   - 如：均值滤波、高斯滤波

2. 非线性滤波 (Nonlinear Filtering)
   - 不满足线性性质
   - 如：中值滤波、双边滤波

按功能分类：
1. 平滑滤波 (Smoothing/Blurring)
   - 去除噪声，模糊细节
   - 如：均值滤波、高斯滤波

2. 锐化滤波 (Sharpening)
   - 增强边缘和细节
   - 如：拉普拉斯滤波、Sobel滤波

3. 边缘检测 (Edge Detection)
   - 提取边缘信息
   - 如：Canny、Sobel

4. 形态学滤波 (Morphological)
   - 基于形状的处理
   - 如：腐蚀、膨胀
""")


# 演示不同滤波器效果
def demonstrate_filter_types():
    """演示不同类型滤波器的效果"""

    # 创建测试图片（带噪声的简单图案）
    height, width = 150, 200
    test_img = np.zeros((height, width), dtype=np.uint8)

    # 添加一些几何形状
    cv2.rectangle(test_img, (30, 30), (80, 80), 200, -1)
    cv2.circle(test_img, (150, 60), 25, 150, -1)
    cv2.line(test_img, (20, 120), (180, 120), 100, 3)

    # 添加高斯噪声
    noise = np.zeros((height, width), dtype=np.uint8)
    cv2.randn(noise, 0, 25)
    noisy_img = cv2.add(test_img, noise)

    # 应用不同滤波器
    # 1. 均值滤波（线性平滑）
    mean_filtered = cv2.blur(noisy_img, (5, 5))

    # 2. 高斯滤波（线性平滑）
    gaussian_filtered = cv2.GaussianBlur(noisy_img, (5, 5), 1.0)

    # 3. 中值滤波（非线性平滑）
    median_filtered = cv2.medianBlur(noisy_img, 5)

    # 4. 锐化滤波（线性锐化）
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]], dtype=np.float32)
    sharpened = cv2.filter2D(noisy_img, -1, kernel_sharpen)

    # 显示结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    images = [test_img, noisy_img, mean_filtered,
              gaussian_filtered, median_filtered, sharpened]
    titles = ["原始图片", "加噪图片", "均值滤波\n(线性平滑)",
              "高斯滤波\n(线性平滑)", "中值滤波\n(非线性平滑)", "锐化滤波\n(线性锐化)"]

    for idx, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

        # 在加噪图片上显示噪声统计
        if idx == 1:
            noise_level = np.std(noisy_img.astype(np.float32) - test_img.astype(np.float32))
            ax.text(0.5, -0.1, f'噪声标准差: {noise_level:.1f}',
                    transform=ax.transAxes, ha='center', fontsize=9)

    plt.suptitle("不同滤波器效果对比", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return test_img, noisy_img, mean_filtered, gaussian_filtered, median_filtered, sharpened


# 演示滤波器类型
clean_img, noisy_img, mean_filt, gauss_filt, median_filt, sharp_filt = demonstrate_filter_types()

# ==================== 5. 边界处理策略 ====================
# ==================== 5. 边界处理策略 ====================
print("\n🎯 5. 边界处理策略")
print("=" * 30)

print("""
卷积边界处理：

当卷积核在图片边界时，部分核会超出图片范围
常见处理方法：

1. 补零填充 (Zero Padding)
   - 边界外补0
   - 公式：P'[i,j] = 0 (当i,j超出边界)
   - 优点：简单
   - 缺点：边界变暗

2. 复制填充 (Replicate)
   - 复制最近的边界像素
   - 公式：P'[i,j] = P[clamp(i), clamp(j)]
   - 优点：保持边界亮度
   - 缺点：可能产生边缘效应

3. 反射填充 (Reflect)
   - 反射边界像素
   - 公式：P'[i,j] = P[reflect(i), reflect(j)]
   - 优点：边界连续
   - 缺点：计算复杂

4. 循环填充 (Wrap)
   - 循环使用图片
   - 公式：P'[i,j] = P[i%H, j%W]
   - 优点：保持周期性
   - 缺点：不适用于非周期图片
   
   
   填充就是在卷积前给图片加个"边框"：
   为什么加：防止变小，利用边界
   加多少：通常加(卷积核-1)/2
   怎么加：补零、复制、镜像、循环(注意反射填充和复制一样当填充一层的时候)
    怎么选：深度学习补零，图像处理镜像
    
""")


def demonstrate_border_handling():
    """演示不同边界处理方法"""

    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    # 创建更容易显示差异的测试图片
    test_img = np.array([
        [0, 0, 0, 0, 0],
        [0, 100, 200, 100, 0],
        [0, 200, 255, 200, 0],
        [0, 100, 200, 100, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)

    # 使用有明显方向性的卷积核
    kernel = np.array([
        [1, 2, 4],
        [0, 0, 0],
        [-1, -2, -4]
    ], dtype=np.float32)

    print("原始图片 (5×5):")
    print(test_img)

    print("\n卷积核 (3×3 非对称):")
    print(kernel)

    # 使用不同边界处理
    border_types = [
        (cv2.BORDER_CONSTANT, "补零填充"),
        (cv2.BORDER_REPLICATE, "复制填充"),
        (cv2.BORDER_REFLECT, "反射填充"),
    ]

    # 创建画布，注意现在填充2像素，所以填充后图片大小是9×9
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 存储结果用于比较
    results = {}

    for idx, (border_type, title) in enumerate(border_types):
        ax_top = axes[0, idx]  # 第一行，第idx列
        ax_bottom = axes[1, idx]  # 第二行，第idx列

        # 创建填充后的图片 - 填充2像素
        if border_type == cv2.BORDER_CONSTANT:
            bordered_img = cv2.copyMakeBorder(test_img, 2, 2, 2, 2,
                                              border_type, value=0)
        else:
            bordered_img = cv2.copyMakeBorder(test_img, 2, 2, 2, 2,
                                              border_type)

        # 显示填充后的图片
        ax_top.imshow(bordered_img, cmap='viridis')
        ax_top.set_title(f"{title}\n(填充后 9×9)", fontsize=10)
        ax_top.grid(True, which='both', color='white', linestyle='-', linewidth=0.5)
        ax_top.set_xticks(range(9))
        ax_top.set_yticks(range(9))

        # 在图中显示数值
        for i in range(9):
            for j in range(9):
                pixel_value = bordered_img[i, j]
                text_color = 'white' if pixel_value > 128 else 'black'
                ax_top.text(j, i, f'{pixel_value:.0f}',
                            ha='center', va='center',
                            color=text_color, fontsize=6)

        # 应用滤波
        filtered = cv2.filter2D(test_img, -1, kernel, borderType=border_type)

        # 存储结果
        results[title] = filtered

        # 显示滤波结果
        im = ax_bottom.imshow(filtered, cmap='viridis')
        ax_bottom.set_title(f"{title}\n(滤波后 5×5)", fontsize=10)
        ax_bottom.grid(True, which='both', color='white', linestyle='-', linewidth=0.5)
        ax_bottom.set_xticks(range(5))
        ax_bottom.set_yticks(range(5))

        # 在图中显示数值
        for i in range(5):
            for j in range(5):
                pixel_value = filtered[i, j]
                # 根据值的大小选择文字颜色
                text_color = 'white' if abs(pixel_value) > 300 else 'black'
                ax_bottom.text(j, i, f'{pixel_value:.0f}',
                               ha='center', va='center',
                               color=text_color, fontsize=8)

        print(f"\n{title} 结果:")
        print(filtered)

    plt.suptitle("不同边界处理方法对比 - 填充2像素", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 比较复制填充和反射填充的差异
    print("\n" + "=" * 60)
    print("复制填充 vs 反射填充 差异分析:")
    print("=" * 60)

    replicate_result = results["复制填充"]
    reflect_result = results["反射填充"]

    # 计算绝对差异
    diff = np.abs(replicate_result - reflect_result)

    print(f"\n绝对差异矩阵 (复制填充 - 反射填充):")
    print(diff)

    print(f"\n最大差异: {diff.max():.2f}")
    print(f"平均差异: {diff.mean():.2f}")
    print(f"总差异: {diff.sum():.2f}")

    # 找出差异最大的位置
    if diff.max() > 0:
        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"\n差异最大的位置: {max_diff_idx}, 值: {diff[max_diff_idx]:.2f}")
        print(f"  复制填充该位置值: {replicate_result[max_diff_idx]:.2f}")
        print(f"  反射填充该位置值: {reflect_result[max_diff_idx]:.2f}")
    else:
        print("\n两种填充方式结果完全相同")

    # 可视化差异
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    im1 = ax1.imshow(replicate_result, cmap='viridis')
    ax1.set_title("复制填充结果")
    ax1.grid(True, which='both', color='white', linestyle='-', linewidth=0.5)
    ax1.set_xticks(range(5))
    ax1.set_yticks(range(5))
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(reflect_result, cmap='viridis')
    ax2.set_title("反射填充结果")
    ax2.grid(True, which='both', color='white', linestyle='-', linewidth=0.5)
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    plt.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(diff, cmap='hot')
    ax3.set_title("两者绝对差异")
    ax3.grid(True, which='both', color='white', linestyle='-', linewidth=0.5)
    ax3.set_xticks(range(5))
    ax3.set_yticks(range(5))
    plt.colorbar(im3, ax=ax3)

    plt.suptitle("复制填充 vs 反射填充 结果对比", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    return test_img, kernel

# 演示边界处理
test_img_border, kernel_border = demonstrate_border_handling()

# ==================== 6. 滤波器性能指标 ====================
print("\n🎯 6. 滤波器性能指标")
print("=" * 30)

print("""
评价滤波器性能的指标：

1. 噪声抑制能力
   - 滤波后噪声的减少程度
   - 可用信噪比(SNR)衡量
   - SNR = 信号功率 / 噪声功率

2. 细节保留能力
   - 滤波后重要特征的保持程度
   - 如边缘、纹理的保持

3. 计算复杂度
   - 滤波器的计算时间
   - 与卷积核大小、类型相关

4. 内存使用
   - 滤波器需要的内存空间

5. 适用场景
   - 不同噪声类型适用不同滤波器
   - 实时性要求

常见滤波器对比：
| 滤波器   | 噪声抑制 | 细节保留 | 计算复杂度 | 适用噪声     |
|----------|----------|----------|------------|--------------|
| 均值滤波 | 中等     | 差       | 低         | 高斯噪声     |
| 高斯滤波 | 好       | 中等     | 中等       | 高斯噪声     |
| 中值滤波 | 很好     | 好       | 中等       | 椒盐噪声     |
| 双边滤波 | 好       | 很好     | 高         | 多种噪声     |
""")


# 演示性能比较
def demonstrate_performance_comparison():
    """演示不同滤波器的性能比较"""

    # 创建测试图片
    height, width = 200, 300
    original = np.zeros((height, width), dtype=np.uint8)

    # 添加一些细节
    cv2.rectangle(original, (50, 50), (150, 150), 200, -1)
    cv2.circle(original, (225, 100), 40, 150, -1)
    cv2.line(original, (20, 180), (280, 180), 100, 3)

    # 添加混合噪声（高斯+椒盐）
    noisy = original.copy()

    # 添加高斯噪声
    gaussian_noise = np.zeros((height, width), dtype=np.uint8)
    cv2.randn(gaussian_noise, 0, 25)
    noisy = cv2.add(noisy, gaussian_noise)

    # 添加椒盐噪声
    num_salt = int(0.005 * height * width)  # 0.5%盐噪声
    num_pepper = int(0.005 * height * width)  # 0.5%椒噪声

    coords = [np.random.randint(0, i - 1, num_salt) for i in original.shape]
    noisy[coords[0], coords[1]] = 255

    coords = [np.random.randint(0, i - 1, num_pepper) for i in original.shape]
    noisy[coords[0], coords[1]] = 0

    # 应用不同滤波器
    import time

    filters = [
        ("均值滤波 (5×5)", lambda img: cv2.blur(img, (5, 5))),
        ("高斯滤波 (5×5)", lambda img: cv2.GaussianBlur(img, (5, 5), 1.0)),
        ("中值滤波 (5×5)", lambda img: cv2.medianBlur(img, 5)),
        ("双边滤波", lambda img: cv2.bilateralFilter(img, 9, 75, 75))
    ]

    results = []
    computation_times = []

    for name, filter_func in filters:
        start_time = time.time()
        filtered = filter_func(noisy)
        end_time = time.time()

        results.append((name, filtered))
        computation_times.append((name, (end_time - start_time) * 1000))  # 转换为毫秒

    # 计算性能指标
    print("\n性能比较:")
    print("-" * 60)
    print(f"{'滤波器':<20} {'计算时间(ms)':<15} {'SNR提升(dB)':<15} {'边缘保持':<10}")
    print("-" * 60)

    for (name, filtered), (name_time, comp_time) in zip(results, computation_times):
        # 计算SNR提升
        noise_before = np.std(noisy.astype(np.float32) - original.astype(np.float32))
        noise_after = np.std(filtered.astype(np.float32) - original.astype(np.float32))

        if noise_after > 0:
            snr_improvement = 20 * np.log10(noise_before / noise_after)
        else:
            snr_improvement = float('inf')

        # 计算边缘保持（简化：使用Sobel边缘检测）
        sobel_original = cv2.Sobel(original, cv2.CV_64F, 1, 1)
        sobel_filtered = cv2.Sobel(filtered, cv2.CV_64F, 1, 1)

        edge_preservation = np.sum(np.abs(sobel_filtered)) / np.sum(np.abs(sobel_original))

        print(f"{name:<20} {comp_time:<15.2f} {snr_improvement:<15.2f} {edge_preservation:<10.3f}")

    # 显示结果
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))

    display_images = [original, noisy] + [img for _, img in results]
    display_titles = ["原始图片", "加噪图片"] + [name for name, _ in results]

    for idx, (ax, img, title) in enumerate(zip(axes.flat, display_images, display_titles)):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

        # 在第一个图片上显示信息
        if idx == 0:
            ax.text(0.5, -0.1, "参考标准", transform=ax.transAxes, ha='center', fontsize=9)
        elif idx == 1:
            noise_level = np.std(noisy.astype(np.float32) - original.astype(np.float32))
            ax.text(0.5, -0.1, f'噪声水平: {noise_level:.1f}',
                    transform=ax.transAxes, ha='center', fontsize=9)

    plt.suptitle("不同滤波器性能比较", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return original, noisy, results, computation_times


# 演示性能比较
original_perf, noisy_perf, filter_results, comp_times = demonstrate_performance_comparison()

# ==================== 7. 实际应用场景 ====================
print("\n🎯 7. 实际应用场景")
print("=" * 30)

print("""
图像滤波的实际应用：

1. 数码摄影
   - 降噪：去除高ISO产生的噪声
   - 锐化：增强图片细节
   - 美颜：皮肤平滑处理

2. 医学影像
   - MRI/CT图像去噪
   - 增强诊断特征
   - 去除扫描伪影

3. 视频监控
   - 实时视频降噪
   - 运动检测预处理
   - 低光照增强

4. 遥感图像
   - 卫星图像去噪
   - 特征提取预处理
   - 多光谱图像融合

5. 计算机视觉
   - 特征检测预处理
   - 图像配准
   - 目标识别增强

6. 手机应用
   - 实时滤镜
   - 人像模式
   - 夜景模式
""")


# 演示实际应用
def demonstrate_real_world_applications():
    """演示实际应用场景"""

    # 模拟不同场景
    scenarios = [
        ("📸 数码摄影 - 人像美颜", "portrait"),
        ("🏥 医学影像 - X光增强", "medical"),
        ("🎥 视频监控 - 低光照", "surveillance"),
        ("🛰️ 遥感图像 - 卫星图", "satellite")
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for idx, (title, scenario_type) in enumerate(scenarios):
        # 创建模拟图片
        height, width = 150, 200

        if scenario_type == "portrait":
            # 模拟人像（皮肤+特征）
            img = np.ones((height, width), dtype=np.uint8) * 180
            # 添加皮肤纹理噪声
            texture_noise = np.random.randint(-10, 10, (height, width), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + texture_noise, 0, 255).astype(np.uint8)
            # 添加眼睛、嘴巴
            cv2.circle(img, (80, 50), 10, 50, -1)  # 左眼
            cv2.circle(img, (120, 50), 10, 50, -1)  # 右眼
            cv2.ellipse(img, (100, 90), (30, 15), 0, 0, 180, 50, 3)  # 嘴巴

        elif scenario_type == "medical":
            # 模拟X光影像
            img = np.random.randint(100, 200, (height, width), dtype=np.uint8)
            # 添加骨骼结构
            cv2.rectangle(img, (60, 30), (140, 120), 250, 15)  # 主要骨骼
            cv2.circle(img, (100, 100), 20, 240, 8)  # 关节

        elif scenario_type == "surveillance":
            # 模拟监控视频帧
            img = np.random.randint(20, 60, (height, width), dtype=np.uint8)  # 低光照背景
            # 添加运动物体
            cv2.rectangle(img, (80, 60), (120, 100), 150, -1)  # 移动物体
            # 添加运动模糊
            kernel_motion = np.eye(5) / 5
            img = cv2.filter2D(img, -1, kernel_motion)

        elif scenario_type == "satellite":
            # 模拟卫星图像
            img = np.zeros((height, width), dtype=np.uint8)
            # 添加地形特征
            cv2.rectangle(img, (30, 30), (170, 170), 100, -1)  # 地面
            cv2.circle(img, (50, 50), 15, 200, -1)  # 建筑
            cv2.line(img, (100, 30), (100, 170), 150, 5)  # 道路
            # 添加传感器噪声
            noise = np.random.randint(-20, 20, (height, width), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 添加噪声
        noisy_img = img.copy()
        gaussian_noise = np.zeros((height, width), dtype=np.uint8)
        cv2.randn(gaussian_noise, 0, 20)
        noisy_img = cv2.add(noisy_img, gaussian_noise)

        # 应用合适的滤波器
        if scenario_type == "portrait":
            # 人像美颜：双边滤波
            filtered = cv2.bilateralFilter(noisy_img, 9, 50, 50)
        elif scenario_type == "medical":
            # 医学影像：中值滤波+锐化
            denoised = cv2.medianBlur(noisy_img, 3)
            kernel_sharpen = np.array([[0, -1, 0],
                                       [-1, 5, -1],
                                       [0, -1, 0]], dtype=np.float32)
            filtered = cv2.filter2D(denoised, -1, kernel_sharpen)
        elif scenario_type == "surveillance":
            # 监控视频：高斯滤波
            filtered = cv2.GaussianBlur(noisy_img, (3, 3), 1.0)
        else:  # satellite
            # 卫星图像：均值滤波
            filtered = cv2.blur(noisy_img, (3, 3))

        # 显示原始、加噪、滤波结果
        row = idx // 2
        col = (idx % 2) * 2

        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f"{title}\n原始", fontsize=9)
        axes[row, col].axis('off')

        axes[row, col + 1].imshow(filtered, cmap='gray')
        axes[row, col + 1].set_title(f"{title}\n滤波后", fontsize=9)
        axes[row, col + 1].axis('off')

    plt.suptitle("图像滤波在实际场景中的应用", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    return True


# 演示实际应用
demonstrate_real_world_applications()

# ==================== 8. 练习与挑战 ====================
print("\n💪 8. 练习与挑战")
print("=" * 30)

print("""
练习题：

1. 基础练习：
   a) 手动实现3×3均值滤波
   b) 比较不同大小卷积核的效果
   c) 测试不同边界处理方法

2. 进阶练习：
   a) 实现自适应滤波器，根据局部噪声调整参数
   b) 比较不同滤波器对不同噪声类型的效果
   c) 实现实时视频滤波

3. 思考题：
   a) 为什么高斯滤波比均值滤波更好地保留边缘？
   b) 中值滤波为什么能有效去除椒盐噪声？
   c) 双边滤波如何同时实现平滑和边缘保持？
""")

# 练习框架代码
print("\n💻 练习框架代码：")

print("""
# 练习1a: 手动实现3×3均值滤波
def manual_mean_filter(image, kernel_size=3):
    height, width = image.shape[:2]
    filtered = np.zeros_like(image, dtype=np.float32)

    pad = kernel_size // 2

    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            # 提取局部区域
            region = image[i-pad:i+pad+1, j-pad:j+pad+1]
            # 计算均值
            filtered[i, j] = np.mean(region)

    return filtered.astype(image.dtype)

# 练习2a: 自适应滤波器框架
def adaptive_filter(image, noise_std=20):
    # 根据局部统计调整滤波参数
    height, width = image.shape[:2]
    filtered = np.zeros_like(image)

    for i in range(1, height-1):
        for j in range(1, width-1):
            # 计算局部统计
            region = image[i-1:i+2, j-1:j+2]
            local_std = np.std(region)

            # 根据局部噪声调整滤波强度
            if local_std > noise_std * 1.5:
                # 高噪声区域：强滤波
                filtered[i, j] = np.median(region)
            else:
                # 低噪声区域：弱滤波
                filtered[i, j] = np.mean(region)

    return filtered

# 练习3a: 高斯滤波边缘保持分析
def analyze_gaussian_edge_preservation():
    # 创建测试图片（带边缘）
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, 50:] = 255  # 锐利边缘

    # 应用不同滤波器
    mean_filtered = cv2.blur(img, (5, 5))
    gaussian_filtered = cv2.GaussianBlur(img, (5, 5), 1.0)

    # 分析边缘保持
    # 高斯滤波权重中心大，边缘小，更好地保留边缘
""")

# ==================== 9. 总结 ====================
print("\n" + "=" * 50)
print("✅ 图像滤波理论总结")
print("=" * 50)

summary = """
📊 图像滤波核心知识：

1. 基本概念
   - 滤波目的：去噪、增强、特征提取
   - 噪声类型：高斯、椒盐、均匀、泊松
   - 卷积操作：局部邻域加权平均

2. 滤波器分类
   - 按操作域：空间域、频域
   - 按线性性质：线性、非线性
   - 按功能：平滑、锐化、边缘检测

3. 边界处理
   - 补零填充：简单，但边界变暗
   - 复制填充：保持边界亮度
   - 反射填充：边界连续
   - 循环填充：保持周期性

4. 性能指标
   - 噪声抑制能力
   - 细节保留能力
   - 计算复杂度
   - 内存使用

5. 实际应用
   - 数码摄影：降噪、美颜
   - 医学影像：增强诊断
   - 视频监控：实时处理
   - 遥感图像：特征提取

6. 核心公式
   - 卷积：I'[i,j] = ΣΣ K[u,v]·I[i+u, j+v]
   - 高斯函数：G(x,y) = (1/(2πσ²))·exp(-(x²+y²)/(2σ²))
   - 信噪比：SNR = 10·log₁₀(信号功率/噪声功率)

🎯 学习路线：
  1. 理解卷积操作和边界处理
  2. 掌握不同噪声类型和特点
  3. 学会选择合适滤波器
  4. 理解滤波器性能权衡
  5. 应用实际场景解决问题
"""

print(summary)
print("\n📁 下一个文件: 05_02_均值滤波实现.py")
print("  我们将动手实现均值滤波！")