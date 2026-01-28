"""
边缘检测实际应用 - 精简完整版
学习目标：掌握边缘检测在多个领域的实际应用
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("🎯 边缘检测实际应用")
print("=" * 50)

# ==================== 1. 图像分割应用 ====================
print("\n1️⃣ 图像分割应用")
print("=" * 30)


def image_segmentation_demo():
    """基于边缘的图像分割"""
    # 创建测试图像
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(img, (50, 30), (150, 100), 150, -1)
    cv2.circle(img, (220, 80), 40, 200, -1)

    # 边缘检测
    edges = cv2.Canny(img, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建分割结果
    segmented = np.zeros((200, 300, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 100:
            cv2.drawContours(segmented, [cnt], -1, colors[i % 3], -1)

    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("原始图像")
    axes[0].axis('off')

    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title("边缘检测")
    axes[1].axis('off')

    axes[2].imshow(segmented)
    axes[2].set_title("分割结果")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"检测到 {len(contours)} 个区域")
    return img, edges, segmented


# 执行图像分割
seg_result = image_segmentation_demo()

# ==================== 2. 目标检测应用 ====================
print("\n2️⃣ 目标检测应用")
print("=" * 30)


def object_detection_demo():
    """基于边缘的目标检测"""
    # 创建包含多个物体的场景
    scene = np.zeros((300, 400, 3), dtype=np.uint8)

    # 添加不同物体
    cv2.rectangle(scene, (50, 50), (150, 150), (255, 0, 0), -1)  # 蓝色矩形
    cv2.circle(scene, (280, 100), 40, (0, 255, 0), -1)  # 绿色圆形
    cv2.rectangle(scene, (180, 200), (300, 280), (0, 0, 255), -1)  # 红色矩形

    # 转换为灰度并检测边缘
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 检测轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原始图像上绘制检测结果
    result = scene.copy()
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(result, f'Obj{i + 1}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    axes[0].set_title("原始场景")
    axes[0].axis('off')

    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title("边缘图")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[2].set_title("目标检测结果")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"检测到 {len(contours)} 个目标")
    return scene, edges, result


# 执行目标检测
detection_result = object_detection_demo()

# ==================== 3. 工业检测应用 ====================
print("\n3️⃣ 工业检测应用")
print("=" * 30)


def industrial_inspection_demo():
    """工业零件缺陷检测"""
    # 创建正常零件
    part = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(part, (100, 100), 60, 200, -1)
    cv2.circle(part, (100, 100), 20, 0, -1)  # 中心孔

    # 创建有缺陷的零件
    defective = part.copy()
    cv2.line(defective, (60, 60), (140, 140), 100, 3)  # 裂纹
    cv2.circle(defective, (150, 60), 8, 100, -1)  # 凹坑

    # 检测缺陷
    edges_normal = cv2.Canny(part, 30, 100)
    edges_defect = cv2.Canny(defective, 30, 100)

    # 分析轮廓差异
    contours_defect, _ = cv2.findContours(edges_defect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_normal, _ = cv2.findContours(edges_normal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 标记缺陷
    result = cv2.cvtColor(defective, cv2.COLOR_GRAY2BGR)
    for cnt in contours_defect:
        area = cv2.contourArea(cnt)
        if area < 500 and area > 10:  # 小轮廓可能是缺陷
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 显示结果
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(part, cmap='gray')
    axes[0, 0].set_title("正常零件")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(defective, cmap='gray')
    axes[0, 1].set_title("有缺陷零件")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(edges_defect, cmap='gray')
    axes[1, 0].set_title("缺陷边缘")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(result)
    axes[1, 1].set_title("缺陷检测")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    defect_count = len(contours_defect) - len(contours_normal)
    print(f"检测到 {max(0, defect_count)} 个缺陷")
    return part, defective, result


# 执行工业检测
industrial_result = industrial_inspection_demo()

# ==================== 4. 医学影像应用 ====================
print("\n4️⃣ 医学影像应用")
print("=" * 30)


def medical_imaging_demo():
    """医学细胞分析"""
    # 创建细胞图像
    cells = np.zeros((250, 300), dtype=np.uint8)

    # 添加细胞
    cell_positions = [(80, 80), (180, 100), (100, 160), (220, 180)]
    radii = [25, 30, 20, 35]

    for (x, y), r in zip(cell_positions, radii):
        cv2.circle(cells, (x, y), r, 200, -1)
        cv2.circle(cells, (x, y), r // 3, 150, -1)  # 细胞核

    # 添加一个异常细胞（形状不规则）
    irregular = np.array([[40, 200], [60, 190], [80, 210], [70, 230], [50, 220]], np.int32)
    cv2.fillPoly(cells, [irregular], 180)

    # 边缘检测
    edges = cv2.Canny(cells, 30, 100)

    # 细胞分析
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = cv2.cvtColor(cells, cv2.COLOR_GRAY2BGR)
    analysis_results = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 10:  # 过滤噪声
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            # 判断细胞状态
            if circularity > 0.8:
                status = "Yes"
                color = (0, 255, 0)  # 绿色
            else:
                status = "No"
                color = (0, 0, 255)  # 红色

            # 标记细胞
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result, status, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            analysis_results.append((i, area, circularity, status))

    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(cells, cmap='gray')
    axes[0].set_title("细胞图像")
    axes[0].axis('off')

    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title("细胞边缘")
    axes[1].axis('off')

    axes[2].imshow(result)
    axes[2].set_title("细胞分析")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    # 打印分析结果
    print("\n细胞分析结果:")
    print("-" * 20)
    for i, area, circularity, status in analysis_results:
        print(f"细胞{i + 1}: 面积={area:.1f}, 圆度={circularity:.3f}, 状态={status}")

    return cells, edges, result


# 执行医学影像分析
medical_result = medical_imaging_demo()

# ==================== 5. 自动驾驶应用 ====================
print("\n5️⃣ 自动驾驶应用")
print("=" * 30)


def autonomous_driving_demo():
    """车道线检测"""
    # 创建道路场景
    road = np.zeros((300, 500, 3), dtype=np.uint8)

    # 道路
    cv2.rectangle(road, (0, 100), (500, 300), (100, 100, 100), -1)

    # 车道线
    cv2.line(road, (100, 100), (100, 300), (255, 255, 255), 5)  # 左车道线
    cv2.line(road, (400, 100), (400, 300), (255, 255, 255), 5)  # 右车道线

    # 中央虚线
    for y in range(120, 300, 40):
        cv2.line(road, (250, y), (250, y + 20), (255, 255, 255), 3)

    # 障碍物
    cv2.rectangle(road, (200, 200), (280, 250), (0, 0, 255), -1)

    # 车道线检测
    gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=30, maxLineGap=20)

    # 绘制检测到的车道线
    lane_detection = road.copy()
    left_lanes = []
    right_lanes = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 计算斜率，过滤水平线
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > 0.3:  # 有效的车道线斜率
                    if slope < 0:  # 左车道线
                        left_lanes.append(line[0])
                        color = (0, 255, 255)  # 黄色
                    else:  # 右车道线
                        right_lanes.append(line[0])
                        color = (255, 0, 255)  # 紫色

                    cv2.line(lane_detection, (x1, y1), (x2, y2), color, 2)

    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(cv2.cvtColor(road, cv2.COLOR_BGR2RGB))
    axes[0].set_title("道路场景")
    axes[0].axis('off')

    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title("边缘检测")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(lane_detection, cv2.COLOR_BGR2RGB))
    axes[2].set_title("车道线检测")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"检测到左车道线: {len(left_lanes)} 条")
    print(f"检测到右车道线: {len(right_lanes)} 条")

    return road, edges, lane_detection


# 执行自动驾驶演示
autonomous_result = autonomous_driving_demo()

# ==================== 6. 应用总结 ====================
print("\n📊 应用总结")
print("=" * 30)

print("""
边缘检测在实际应用中的总结：

1. 图像分割
   - 用途: 将图像划分为有意义的区域
   - 方法: 边缘检测 → 轮廓查找 → 区域填充
   - 优势: 基于边界的分割更准确

2. 目标检测
   - 用途: 识别和定位图像中的物体
   - 方法: 边缘检测 → 轮廓分析 → 边界框标记
   - 优势: 对形状变化鲁棒性强

3. 工业检测
   - 用途: 产品质量控制，缺陷检测
   - 方法: 比较正常与缺陷样本的边缘差异
   - 优势: 能够检测微小缺陷

4. 医学影像
   - 用途: 细胞分析，病变检测
   - 方法: 边缘特征提取 → 形状分析 → 分类判断
   - 优势: 提供定量分析指标

5. 自动驾驶
   - 用途: 车道线检测，障碍物识别
   - 方法: 边缘检测 → 直线检测 → 路径规划
   - 优势: 实时性好，计算效率高

通用优势:
- 对光照变化不敏感
- 保留重要的结构信息
- 计算相对高效
- 适用于实时应用

注意事项:
- 需要合适的阈值选择
- 对噪声敏感，需要预处理
- 复杂纹理可能产生过多边缘
""")

# ==================== 7. 完整代码示例 ====================
print("\n💻 完整代码示例")
print("=" * 30)


# 展示一个完整的应用示例
def complete_edge_detection_pipeline(image_path=None):
    """完整的边缘检测应用管道"""

    if image_path:
        # 从文件加载图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # 创建示例图像
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (200, 200), (255, 0, 0), -1)
        cv2.circle(image, (300, 150), 60, (0, 255, 0), -1)

    # 转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原始图像上绘制结果
    result = image.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # 过滤小轮廓
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # 显示完整流程
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    titles = ['原始图像', '灰度图', '高斯模糊', '边缘检测', '轮廓查找', '最终结果']
    images = [image, gray, blurred, edges, edges, result]

    for i, (ax, title, img) in enumerate(zip(axes.flat, titles, images)):
        if i == 4:  # 轮廓查找
            contour_img = np.zeros_like(gray)
            cv2.drawContours(contour_img, contours, -1, 255, 1)
            ax.imshow(contour_img, cmap='gray')
        elif len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)

        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"处理完成！检测到 {len(contours)} 个轮廓")


# 运行完整示例
complete_edge_detection_pipeline()

print("\n🎉 边缘检测应用演示完成！")
print("=" * 50)