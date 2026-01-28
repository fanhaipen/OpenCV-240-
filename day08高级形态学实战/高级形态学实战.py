"""
第8天 - 高级形态学实战
完整教程：形态学重构、分水岭算法、实战案例
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("🎯 第8天 - 高级形态学实战")
print("=" * 60)

# ==================== 1. 形态学重构 ====================
print("\n🔁 1. 形态学重构 (Morphological Reconstruction)")
print("=" * 50)


def morphological_reconstruction_demo():
    """形态学重构完整演示"""

    print("""
形态学重构原理：

标记-掩码重构过程：
1. 标记图像 (Marker) - 重构的起点
2. 掩码图像 (Mask) - 重构的边界约束  
3. 迭代膨胀直到稳定

数学表达：
R(f) = lim_{n→∞} δ^{(n)}_g(f)
其中 δ_g 是在掩码 g 约束下的膨胀

应用场景：
- 图像修复和填充
- 连通分量提取
- 去除小物体但保留大物体
- 孔洞填充
""")

    # 1.1 创建测试图像
    image = np.zeros((200, 300), dtype=np.uint8)

    # 添加各种形状
    cv2.rectangle(image, (30, 30), (80, 80), 200, -1)  # 矩形1
    cv2.rectangle(image, (50, 50), (100, 100), 255, -1)  # 矩形2
    cv2.circle(image, (150, 60), 30, 180, -1)  # 圆形1
    cv2.circle(image, (180, 90), 20, 220, -1)  # 圆形2
    cv2.ellipse(image, (250, 150), (50, 30), 0, 0, 360, 200, -1)  # 椭圆

    # 添加噪声
    np.random.seed(42)
    for _ in range(20):
        x, y = np.random.randint(0, 290), np.random.randint(0, 190)
        cv2.circle(image, (x, y), 3, 255, -1)

    # 添加孔洞
    image[40:45, 40:45] = 0
    image[160:165, 170:175] = 0

    # 1.2 实现形态学重构函数
    def morphological_reconstruction(marker, mask, kernel_size=3, max_iter=1000):
        """形态学重构实现"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        prev = np.zeros_like(marker)
        recon = marker.copy()

        iterations = 0
        for i in range(max_iter):
            dilated = cv2.dilate(recon, kernel)
            recon = np.minimum(dilated, mask)

            if np.array_equal(recon, prev):
                iterations = i + 1
                break
            prev = recon.copy()
        else:
            iterations = max_iter

        return recon, iterations

    # 1.3 创建不同的标记
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 标记1：腐蚀结果
    marker1 = cv2.erode(image, kernel)

    # 标记2：阈值处理
    _, marker2 = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

    # 标记3：距离变换
    dist = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    local_max = (dist == cv2.dilate(dist, np.ones((3, 3)))).astype(np.uint8) * 255
    marker3 = cv2.bitwise_and(image, image, mask=local_max)

    # 1.4 执行重构
    recon1, iter1 = morphological_reconstruction(marker1, image)
    recon2, iter2 = morphological_reconstruction(marker2, image)
    recon3, iter3 = morphological_reconstruction(marker3, image)

    # 1.5 可视化结果
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))

    images = [
        ("原始图像", image, 'gray'),
        ("标记1: 腐蚀", marker1, 'gray'),
        ("重构结果1", recon1, 'gray'),
        ("差异1", np.abs(image.astype(int) - recon1.astype(int)), 'hot'),
        ("标记2: 阈值", marker2, 'gray'),
        ("重构结果2", recon2, 'gray'),
        ("差异2", np.abs(image.astype(int) - recon2.astype(int)), 'hot'),
        ("标记3: 局部极大值", marker3, 'gray'),
        ("重构结果3", recon3, 'gray'),
        ("差异3", np.abs(image.astype(int) - recon3.astype(int)), 'hot')
    ]

    for i, (title, img, cmap) in enumerate(images):
        row = i // 4
        col = i % 4
        im = axes[row, col].imshow(img, cmap=cmap)
        axes[row, col].set_title(title, fontsize=10, fontweight='bold')
        axes[row, col].axis('off')

        if 'hot' in cmap:
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)

    # 统计信息
    axes[2, 2].axis('off')
    axes[2, 3].axis('off')

    stats_text = "重构效果统计:\n\n"
    stats_text += f"标记1(腐蚀):\n"
    stats_text += f"  迭代次数: {iter1}\n"
    stats_text += f"  恢复率: {np.sum(recon1 > 0) / np.sum(image > 0) * 100:.1f}%\n\n"

    stats_text += f"标记2(阈值):\n"
    stats_text += f"  迭代次数: {iter2}\n"
    stats_text += f"  恢复率: {np.sum(recon2 > 0) / np.sum(image > 0) * 100:.1f}%\n\n"

    stats_text += f"标记3(局部极大值):\n"
    stats_text += f"  迭代次数: {iter3}\n"
    stats_text += f"  恢复率: {np.sum(recon3 > 0) / np.sum(image > 0) * 100:.1f}%"

    axes[2, 2].text(0.1, 0.5, stats_text, fontsize=9,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))

    axes[2, 3].text(0.1, 0.5, "重构应用建议:\n\n"
                              "1. 腐蚀标记: 适合填充孔洞\n"
                              "2. 阈值标记: 适合提取大物体\n"
                              "3. 局部极大值: 适合分离粘连物体",
                    fontsize=9, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.suptitle("形态学重构 - 不同标记方法比较", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 1.6 实际应用：图像修复
    print("\n🔧 实际应用：图像修复")
    print("-" * 40)

    # 创建损坏图像
    text_img = np.ones((150, 300), dtype=np.uint8) * 200
    cv2.putText(text_img, "MORPHOLOGY", (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, 50, 3)
    cv2.putText(text_img, "RECONSTRUCTION", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, 50, 3)

    # 模拟损坏
    damaged = text_img.copy()
    height, width = damaged.shape
    for _ in range(20):  # 添加20个损坏点
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        size = np.random.randint(3, 8)
        cv2.circle(damaged, (x, y), size, 200, -1)  # 用背景色覆盖

    # 修复过程
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 标记：腐蚀后的图像
    repair_marker = cv2.erode(damaged, repair_kernel)

    # 重构修复
    repaired, _ = morphological_reconstruction(repair_marker,
                                               text_img)

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    repair_images = [
        ("原始图像", text_img, 'gray'),
        ("损坏图像", damaged, 'gray'),
        ("修复标记", repair_marker, 'gray'),
        ("重构修复", repaired, 'gray')
    ]

    for i, (title, img, cmap) in enumerate(repair_images):
        axes[i].imshow(img, cmap=cmap,vmin=0,vmax=255)
        axes[i].set_title(title, fontweight='bold')
        axes[i].axis('off')

        if i > 0:
            similarity = np.sum(img == text_img) / text_img.size * 100
            axes[i].set_xlabel(f"相似度: {similarity:.1f}%")

    plt.suptitle("形态学重构在图像修复中的应用", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return image, [recon1, recon2, recon3], damaged, repaired


# 运行形态学重构
print("执行形态学重构演示...")
recon_results = morphological_reconstruction_demo()
print("✅ 形态学重构演示完成！")

# ==================== 2. 分水岭算法 ====================
print("\n💧 2. 分水岭算法 (Watershed Algorithm)")
print("=" * 50)


def watershed_demo():
    """分水岭算法完整演示"""

    print("""
分水岭算法原理：

将图像视为地形表面：
1. 亮度值 = 海拔高度
2. 局部极小值 = 注水点
3. 分水岭 = 区域边界

算法步骤：
1. 计算距离变换
2. 找到局部极小值（标记）
3. 应用分水岭分割
4. 提取边界

应用场景：
- 图像分割
- 物体分离  
- 接触物体分割
- 医学图像分析
""")

    # 2.1 创建测试图像
    image = np.zeros((300, 400), dtype=np.uint8)

    # 创建接触的圆形
    centers = [(100, 100), (150, 100), (200, 100), (125, 150), (175, 150)]
    radii = [40, 35, 30, 25, 20]

    for center, radius in zip(centers, radii):
        cv2.circle(image, center, radius, 255, -1)

    # 添加噪声
    np.random.seed(42)
    noise = np.random.normal(0, 10, image.shape)
    noisy_image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

    # 2.2 分水岭分割流程
    # 步骤1: 预处理
    blurred = cv2.GaussianBlur(noisy_image, (5, 5), 0)

    # 步骤2: 二值化
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # 步骤3: 距离变换
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
    dist_normalized = np.uint8(dist_normalized)

    # 步骤4: 前景标记
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # 步骤5: 背景标记
    sure_bg = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # 步骤6: 未知区域
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 步骤7: 标记图像
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 步骤8: 应用分水岭
    color_image = cv2.cvtColor(noisy_image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_image, markers)

    # 2.3 可视化流程
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    watershed_steps = [
        ("原始图像", noisy_image, 'gray'),
        ("二值化", binary, 'gray'),
        ("距离变换", dist_normalized, 'hot'),
        ("确信前景", sure_fg, 'gray'),
        ("确信背景", sure_bg, 'gray'),
        ("未知区域", unknown, 'gray'),
        ("分水岭标记", markers, 'jet'),
    ]

    for i, (title, img, cmap) in enumerate(watershed_steps):
        row = i // 4
        col = i % 4
        im = axes[row, col].imshow(img, cmap=cmap)
        axes[row, col].set_title(title, fontweight='bold')
        axes[row, col].axis('off')

        if cmap in ['hot', 'jet']:
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)

    # 结果显示
    result = color_image.copy()
    result[markers == -1] = [0, 0, 255]  # 分水岭边界标为红色

    axes[1, 3].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title("分水岭结果\n(红色=边界)")
    axes[1, 3].axis('off')

    plt.suptitle("分水岭算法分割流程", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 2.4 统计信息
    num_regions = np.max(markers) - 1
    boundary_pixels = np.sum(markers == -1)
    print(f"分割出的区域数量: {num_regions}")
    print(f"分水岭边界像素: {boundary_pixels}")

    # 2.5 实际应用：细胞计数
    print("\n🔬 实际应用：细胞计数")
    print("-" * 40)

    # 创建模拟细胞图像
    cells = np.zeros((200, 200), dtype=np.uint8)

    # 添加细胞
    np.random.seed(42)
    for _ in range(15):
        x = np.random.randint(30, 170)
        y = np.random.randint(30, 170)
        r = np.random.randint(8, 15)
        intensity = np.random.randint(180, 220)
        cv2.circle(cells, (x, y), r, intensity, -1)

    # 添加一些接触的细胞
    cv2.circle(cells, (100, 50), 12, 200, -1)
    cv2.circle(cells, (120, 50), 10, 210, -1)

    cv2.circle(cells, (50, 100), 10, 190, -1)
    cv2.circle(cells, (65, 100), 9, 200, -1)

    # 添加噪声
    cells = cv2.GaussianBlur(cells, (3, 3), 0)
    noise = np.random.normal(0, 8, cells.shape)
    cells = np.clip(cells.astype(float) + noise, 0, 255).astype(np.uint8)

    # 细胞计数流程
    # 预处理
    _, cell_binary = cv2.threshold(cells, 150, 255, cv2.THRESH_BINARY)

    # 去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cell_cleaned = cv2.morphologyEx(cell_binary, cv2.MORPH_OPEN, kernel)

    # 距离变换
    cell_dist = cv2.distanceTransform(cell_cleaned, cv2.DIST_L2, 5)

    # 前景标记
    _, cell_sure_fg = cv2.threshold(cell_dist, 0.3 * cell_dist.max(), 255, 0)
    cell_sure_fg = np.uint8(cell_sure_fg)

    # 分水岭
    cell_color = cv2.cvtColor(cells, cv2.COLOR_GRAY2BGR)
    _, cell_markers = cv2.connectedComponents(cell_sure_fg)
    cell_markers = cell_markers + 1
    cell_markers[cell_cleaned == 0] = 0

    cell_result = cv2.watershed(cell_color, cell_markers)

    # 标记结果
    cell_color[cell_result == -1] = [0, 0, 255]  # 边界红色
    for i in range(2, np.max(cell_result) + 1):
        mask = cell_result == i
        cell_color[mask] = [np.random.randint(0, 255),
                            np.random.randint(0, 255),
                            np.random.randint(0, 255)]

    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    cell_images = [
        ("原始细胞", cells, 'gray'),
        ("二值化", cell_binary, 'gray'),
        ("去噪后", cell_cleaned, 'gray'),
        ("距离变换", cv2.normalize(cell_dist, None, 0, 255, cv2.NORM_MINMAX), 'hot'),
        ("前景标记", cell_sure_fg, 'gray'),
        ("分水岭标记", cell_result, 'jet'),
        ("分割结果", cv2.cvtColor(cell_color, cv2.COLOR_BGR2RGB), None),
    ]

    for i, (title, img, cmap) in enumerate(cell_images):
        row = i // 4
        col = i % 4
        if cmap:
            axes[row, col].imshow(img, cmap=cmap)
        else:
            axes[row, col].imshow(img)
        axes[row, col].set_title(title, fontweight='bold')
        axes[row, col].axis('off')

        if "分割结果" in title:
            cell_count = np.max(cell_result) - 1
            axes[row, col].set_xlabel(f"细胞计数: {cell_count}")

    # 添加统计信息
    axes[1, 3].axis('off')
    stats_text = "细胞计数结果:\n\n"
    stats_text += f"检测到细胞数: {np.max(cell_result) - 1}\n\n"
    stats_text += "分水岭优点:\n"
    stats_text += "• 可分离接触细胞\n"
    stats_text += "• 自动确定细胞数量\n"
    stats_text += "• 无需预设细胞大小\n\n"
    stats_text += "注意事项:\n"
    stats_text += "• 需要合适的预处理\n"
    stats_text += "• 对噪声敏感\n"
    stats_text += "• 可能需要后处理"

    axes[1, 3].text(0.1, 0.5, stats_text, fontsize=9,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.suptitle("分水岭算法在细胞计数中的应用", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return noisy_image, markers, cells, cell_result


# 运行分水岭算法
print("执行分水岭算法演示...")
watershed_results = watershed_demo()
print("✅ 分水岭算法演示完成！")

# ==================== 3. 综合实战案例 ====================
print("\n🎯 3. 综合实战案例")
print("=" * 50)


def practical_applications():
    """综合实战案例演示"""

    print("""
实战案例包括：
1. 工业视觉 - 零件缺陷检测
2. 医学影像 - 血管分割  
3. 文档处理 - 表格线提取
4. 遥感图像 - 道路提取
""")

    # 3.1 工业零件检测
    print("\n🏭 案例1：工业零件缺陷检测")
    print("-" * 40)

    # 创建零件图像
    parts = np.ones((200, 300), dtype=np.uint8) * 180

    # 添加零件
    cv2.rectangle(parts, (30, 30), (80, 80), 100, 2)  # 外框
    cv2.rectangle(parts, (100, 30), (150, 80), 100, 2)
    cv2.rectangle(parts, (170, 30), (220, 80), 100, 2)

    # 添加内孔
    cv2.circle(parts, (55, 55), 10, 255, -1)
    cv2.circle(parts, (125, 55), 10, 255, -1)
    cv2.circle(parts, (195, 55), 10, 255, -1)

    # 添加缺陷
    cv2.circle(parts, (55, 55), 3, 0, -1)  # 孔洞
    cv2.line(parts, (130, 40), (130, 70), 0, 2)  # 裂纹

    # 检测缺陷
    # 顶帽变换增强缺陷
    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(parts, cv2.MORPH_TOPHAT, kernel)

    # 阈值化
    _, defects = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)

    # 在原图上标记
    parts_color = cv2.cvtColor(parts, cv2.COLOR_GRAY2BGR)
    defect_coords = np.where(defects > 0)
    for y, x in zip(defect_coords[0], defect_coords[1]):
        cv2.circle(parts_color, (x, y), 2, (0, 0, 255), -1)

    # 3.2 血管分割
    print("\n❤️ 案例2：血管分割")
    print("-" * 40)

    # 创建血管图像
    vessels = np.zeros((200, 200), dtype=np.uint8)

    # 主血管
    cv2.line(vessels, (20, 100), (180, 100), 180, 5)
    cv2.line(vessels, (100, 20), (100, 180), 180, 5)

    # 分支血管
    angles = [30, 45, 60, 120, 135, 150, 210, 225, 240, 300, 315, 330]
    for angle in angles:
        rad = np.radians(angle)
        length = 40
        end_x = int(100 + length * np.cos(rad))
        end_y = int(100 + length * np.sin(rad))
        cv2.line(vessels, (100, 100), (end_x, end_y), 150, 2)

    # 添加噪声
    vessels = cv2.GaussianBlur(vessels, (3, 3), 0)
    noise = np.random.normal(0, 8, vessels.shape)
    vessels = np.clip(vessels.astype(float) + noise, 0, 255).astype(np.uint8)

    # 血管增强
    vessel_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vessel_enhanced = cv2.morphologyEx(vessels, cv2.MORPH_BLACKHAT, vessel_kernel)

    # 阈值分割
    _, vessel_binary = cv2.threshold(vessel_enhanced, 20, 255, cv2.THRESH_BINARY)

    # 3.3 文档表格提取
    print("\n📄 案例3：表格线提取")
    print("-" * 40)

    # 创建文档
    document = np.ones((250, 300), dtype=np.uint8) * 200

    # 表格标题
    cv2.putText(document, "Sales Report 2024", (80, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 50, 2)

    # 表格
    rows, cols = 6, 4
    cell_height, cell_width = 30, 70

    # 画表格线
    for i in range(rows + 1):
        y = 60 + i * cell_height
        cv2.line(document, (20, y), (20 + cols * cell_width, y), 0, 1)

    for j in range(cols + 1):
        x = 20 + j * cell_width
        cv2.line(document, (x, 60), (x, 60 + rows * cell_height), 0, 1)

    # 表格内容
    headers = ["ID", "Product", "Qty", "Price"]
    data = [
        ["001", "Laptop", "5", "$5000"],
        ["002", "Mouse", "20", "$200"],
        ["003", "Keyboard", "15", "$450"],
        ["004", "Monitor", "8", "$2400"],
        ["005", "Total", "", "$8050"]
    ]

    # 添加文本
    for i, header in enumerate(headers):
        x = 20 + i * cell_width + 10
        cv2.putText(document, header, (x, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

    for row_idx, row_data in enumerate(data):
        y = 60 + (row_idx + 1) * cell_height + 20
        for col_idx, cell_text in enumerate(row_data):
            x = 20 + col_idx * cell_width + 10
            cv2.putText(document, cell_text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)

    # 表格线检测
    # 水平线
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(255 - document, cv2.MORPH_OPEN, horizontal_kernel)

    # 垂直线
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(255 - document, cv2.MORPH_OPEN, vertical_kernel)

    # 合并
    table_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # 3.4 可视化所有案例
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))

    # 工业检测
    industrial_imgs = [
        ("零件图像", parts, 'gray'),
        ("顶帽增强", tophat, 'gray'),
        ("缺陷检测", defects, 'gray'),
        ("缺陷标记", cv2.cvtColor(parts_color, cv2.COLOR_BGR2RGB), None)
    ]

    for i, (title, img, cmap) in enumerate(industrial_imgs):
        if cmap:
            axes[0, i].imshow(img, cmap=cmap)
        else:
            axes[0, i].imshow(img)
        axes[0, i].set_title(title, fontweight='bold')
        axes[0, i].axis('off')

        if "缺陷检测" in title:
            defect_count = np.sum(img > 0)
            axes[0, i].set_xlabel(f"缺陷像素: {defect_count}")

    # 血管分割
    vessel_imgs = [
        ("血管图像", vessels, 'gray'),
        ("黑帽增强", vessel_enhanced, 'gray'),
        ("二值分割", vessel_binary, 'gray'),
        ("分割结果", cv2.cvtColor(cv2.applyColorMap(vessel_binary, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB), None)
    ]

    for i, (title, img, cmap) in enumerate(vessel_imgs):
        if cmap:
            axes[1, i].imshow(img, cmap=cmap)
        else:
            axes[1, i].imshow(img)
        axes[1, i].set_title(title, fontweight='bold')
        axes[1, i].axis('off')

        if "二值分割" in title:
            vessel_pixels = np.sum(img > 0)
            axes[1, i].set_xlabel(f"血管像素: {vessel_pixels}")

    # 表格提取
    table_imgs = [
        ("文档图像", document, 'gray'),
        ("水平线", horizontal_lines, 'gray'),
        ("垂直线", vertical_lines, 'gray'),
        ("表格线", table_lines, 'gray')
    ]

    for i, (title, img, cmap) in enumerate(table_imgs):
        axes[2, i].imshow(img, cmap=cmap)
        axes[2, i].set_title(title, fontweight='bold')
        axes[2, i].axis('off')

        if "表格线" in title:
            line_pixels = np.sum(img > 0)
            axes[2, i].set_xlabel(f"线像素: {line_pixels}")

    plt.suptitle("形态学实战应用案例", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 3.5 总结统计
    print("\n📊 案例总结:")
    print("-" * 40)
    print(f"1. 工业检测: 检测到 {np.sum(defects > 0)} 个缺陷像素")
    print(f"2. 血管分割: 提取到 {np.sum(vessel_binary > 0)} 个血管像素")
    print(f"3. 表格提取: 检测到 {np.sum(table_lines > 0)} 个表格线像素")

    return parts, defects, vessels, vessel_binary, document, table_lines


# 运行实战案例
print("执行实战案例演示...")
practical_results = practical_applications()
print("✅ 实战案例演示完成！")

# ==================== 4. 总结与对比 ====================
print("\n📈 4. 高级形态学操作总结")
print("=" * 50)

# 创建总结表格
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

summary_data = [
    ["操作", "OpenCV函数", "主要用途", "关键参数", "应用场景"],
    ["形态学重构", "无内置，需自定义", "图像修复/填充", "标记图像, 核大小", "图像修复, 孔洞填充"],
    ["分水岭", "cv2.watershed()", "图像分割", "标记, 距离变换", "细胞计数, 物体分割"],
    ["顶帽变换", "cv2.morphologyEx(MORPH_TOPHAT)", "亮细节提取", "核大小 > 目标", "缺陷检测, 光照校正"],
    ["黑帽变换", "cv2.morphologyEx(MORPH_BLACKHAT)", "暗细节提取", "核大小 > 目标", "血管分割, 暗缺陷"],
    ["形态学梯度", "cv2.morphologyEx(MORPH_GRADIENT)", "边缘提取", "核大小: 3-7", "边界检测, 轮廓提取"],
    ["骨架提取", "细化算法", "中心线提取", "迭代至收敛", "字符识别, 路径提取"]
]

table = ax.table(cellText=summary_data,
                 cellLoc='center',
                 colWidths=[0.15, 0.2, 0.25, 0.2, 0.2],
                 loc='center',
                 colColours=['#40466e'] * 5,
                 cellColours=[['#40466e'] * 5] +
                             [['#f0f0f0'] * 5, ['#ffffff'] * 5] * 3)

# 设置样式
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)

# 标题行
for i in range(5):
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

plt.title("高级形态学操作总结表", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# 操作选择指南
print("\n🎯 操作选择指南:")
print("=" * 60)
print("""
1. 图像修复/孔洞填充？
   → 形态学重构 (Morphological Reconstruction)

2. 分离接触的物体？
   → 分水岭算法 (Watershed)

3. 提取亮细节/缺陷？
   → 顶帽变换 (Top-hat)

4. 提取暗细节/血管？
   → 黑帽变换 (Black-hat)

5. 需要物体边界？
   → 形态学梯度 (Gradient)

6. 需要中心线？
   → 骨架提取 (Skeletonization)

7. 去噪但保留形状？
   → 开运算 (Opening)

8. 填充小孔但保留形状？
   → 闭运算 (Closing)
""")

# 参数调优技巧
print("\n⚙️ 参数调优技巧:")
print("=" * 60)
print("""
1. 结构元素形状:
   - 矩形: 通用形状
   - 椭圆: 各向同性操作
   - 十字: 对角线敏感

2. 核大小选择:
   - 比目标特征稍大
   - 通常为奇数 (3,5,7...)
   - 从3x3开始尝试

3. 迭代次数:
   - 腐蚀/膨胀: 1-3次
   - 骨架提取: 迭代至收敛
   - 重构: 迭代至收敛

4. 组合策略:
   - 先开运算去噪，再梯度提取边缘
   - 先顶帽增强，再阈值分割
   - 距离变换 + 分水岭
""")

# 保存结果
print("\n💾 保存示例图像...")
try:
    # 保存重构示例
    cv2.imwrite('morph_reconstruction.jpg', recon_results[0])

    # 保存分水岭示例
    cv2.imwrite('watershed_result.jpg', watershed_results[0])

    # 保存实战案例
    cv2.imwrite('industrial_defects.jpg', practical_results[1])
    cv2.imwrite('vessel_segmentation.jpg', practical_results[3])
    cv2.imwrite('table_extraction.jpg', practical_results[5])

    print("✅ 所有图像已保存到当前目录:")
    print("   - morph_reconstruction.jpg (形态学重构)")
    print("   - watershed_result.jpg (分水岭分割)")
    print("   - industrial_defects.jpg (工业缺陷检测)")
    print("   - vessel_segmentation.jpg (血管分割)")
    print("   - table_extraction.jpg (表格线提取)")
except Exception as e:
    print(f"⚠️ 保存图像时出错: {e}")

# 学习进度总结
print("\n" + "=" * 60)
print("🎉 第8天学习完成！")
print("=" * 60)
print("""
📚 今日学习内容总结:

1. 形态学重构
   - 标记-掩码重构原理
   - 图像修复和填充
   - 自定义迭代实现

2. 分水岭算法
   - 距离变换计算
   - 标记生成策略
   - 物体分割应用

3. 实战案例
   - 工业缺陷检测
   - 医学血管分割
   - 文档表格提取

🔧 掌握的核心技能:
✓ 形态学重构的实现和应用
✓ 分水岭分割流程
✓ 多领域形态学应用
✓ 参数调优和效果评估

📈 下一步学习建议:
1. 尝试在真实图像上应用这些技术
2. 调整参数观察效果变化
3. 组合多种形态学操作
4. 实现自定义形态学算法

💪 现在你已经掌握了OpenCV形态学的全部核心内容！
接下来的学习中，我们将进入图像分割和特征提取的高级主题。
""")

print("\n✨ 第8天教程结束！恭喜完成高级形态学学习！ ✨")
print("=" * 60)