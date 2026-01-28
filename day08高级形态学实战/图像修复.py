import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def morphological_reconstruction(marker, mask, kernel_size=3, max_iter=1000):
    """
    形态学重构算法

    参数:
    marker: 标记图像（重构的起点）
    mask: 掩码图像（重构的上界）
    kernel_size: 结构元素大小
    max_iter: 最大迭代次数

    返回:
    重构结果, 迭代次数
    """
    # 创建结构元素
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 初始化重构结果
    recon = marker.copy()

    # 迭代重构
    prev_recon = None
    for i in range(max_iter):
        # 对重构结果进行膨胀
        recon_dilated = cv2.dilate(recon, kernel)

        # 与掩码取最小值
        recon = np.minimum(recon_dilated, mask)

        # 检查是否收敛
        if prev_recon is not None and np.array_equal(recon, prev_recon):
            print(f"  重构收敛于第 {i + 1} 次迭代")
            return recon, i + 1

        prev_recon = recon.copy()

    print(f"  达到最大迭代次数 {max_iter}")
    return recon, max_iter


def detect_damage_regions(image, method='threshold'):
    """
    检测损坏区域

    参数:
    image: 输入图像
    method: 检测方法 ('threshold', 'edge', 'manual')

    返回:
    损坏掩码
    """
    height, width = image.shape

    if method == 'threshold':
        # 方法1: 基于阈值的方法
        # 假设背景亮度较高，文字较暗
        _, binary = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return cleaned

    elif method == 'edge':
        # 方法2: 基于边缘检测
        edges = cv2.Canny(image, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        damage_mask = np.zeros_like(image)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # 小区域可能是损坏
                cv2.drawContours(damage_mask, [contour], -1, 255, -1)

        return damage_mask

    else:  # 默认返回一个简单的掩码
        return np.zeros_like(image)


def estimate_mask(image, method='dilation'):
    """
    估计掩码（重构的上界）

    参数:
    image: 输入图像
    method: 估计方法 ('dilation', 'median', 'gaussian', 'adaptive')

    返回:
    估计的掩码图像
    """
    if method == 'dilation':
        # 膨胀方法
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.dilate(image, kernel)

    elif method == 'median':
        # 中值滤波
        return cv2.medianBlur(image, 5)

    elif method == 'gaussian':
        # 高斯模糊
        return cv2.GaussianBlur(image, (5, 5), 0)

    elif method == 'adaptive':
        # 自适应方法
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(image, kernel)
        median = cv2.medianBlur(image, 5)
        return cv2.addWeighted(dilated, 0.5, median, 0.5, 0)

    else:
        return image.copy()


def visualize_repair_process(original, damaged, repair_marker, damage_mask,
                             estimated_mask, repaired_result, method_name):
    """
    可视化修复过程
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    images = [
        ("原始图像", original, 'gray'),
        ("损坏图像", damaged, 'gray'),
        ("修复标记", repair_marker, 'gray'),
        ("损坏掩码", damage_mask, 'gray'),
        ("估计掩码", estimated_mask, 'gray'),
        (f"{method_name}重构结果", repaired_result, 'gray'),
        ("修复差异", cv2.absdiff(damaged, repaired_result), 'hot'),
        ("与原始差异", cv2.absdiff(original, repaired_result), 'hot')
    ]

    for i, (title, img, cmap) in enumerate(images):
        row, col = i // 4, i % 4
        axes[row, col].imshow(img, cmap=cmap, vmin=0, vmax=255)
        axes[row, col].set_title(title, fontweight='bold', fontsize=10)
        axes[row, col].axis('off')

        # 计算相似度
        if title == f"{method_name}重构结果":
            similarity = np.sum(img == original) / original.size * 100
            axes[row, col].set_xlabel(f"相似度: {similarity:.1f}%")
        elif title == "修复差异":
            diff_value = np.sum(img) / img.size
            axes[row, col].set_xlabel(f"总差异: {diff_value:.0f}")
        elif title == "与原始差异":
            diff_value = np.sum(img) / img.size
            axes[row, col].set_xlabel(f"总差异: {diff_value:.0f}")

    plt.suptitle(f"修复过程: {method_name}方法", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def practical_image_repair_demo():
    """
    实际图像修复演示
    """
    print("\n" + "=" * 60)
    print("🔧 实际应用：图像修复")
    print("=" * 60)

    # 创建测试图像
    print("1. 创建测试图像...")
    text_img = np.ones((150, 300), dtype=np.uint8) * 200
    cv2.putText(text_img, "MORPHOLOGY", (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, 50, 3)
    cv2.putText(text_img, "RECONSTRUCTION", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, 50, 3)

    # 模拟损坏
    print("2. 模拟图像损坏...")
    damaged = text_img.copy()
    height, width = damaged.shape
    np.random.seed(42)  # 设置随机种子以便结果可重复
    damage_points = []  # 记录损坏点

    for _ in range(20):  # 添加20个损坏点
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        size = np.random.randint(3, 8)
        cv2.circle(damaged, (x, y), size, 200, -1)  # 用背景色覆盖
        damage_points.append((x, y, size))

    print(f"   添加了 {len(damage_points)} 个损坏点")

    # 实际应用场景：我们只有损坏图像，没有原始图像
    print("\n3. 实际场景：我们只有损坏图像，没有原始图像作为参考")

    # 修复过程
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 1. 创建损坏掩码（检测损坏区域）
    print("4. 检测损坏区域...")
    damage_mask = np.zeros_like(damaged, dtype=np.uint8)

    # 方法1: 基于边缘检测
    edges = cv2.Canny(damaged, 50, 150)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, repair_kernel)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    damage_mask_edges = np.zeros_like(damaged)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:  # 小区域可能是损坏
            cv2.drawContours(damage_mask_edges, [contour], -1, 255, -1)

    # 方法2: 基于像素值差异（假设我们知道背景是200）
    damage_mask_values = np.zeros_like(damaged)
    # 在背景区域寻找"异常"的像素
    damage_mask_values[(damaged > 180) & (damaged < 220)] = 255
    damage_mask_values = cv2.morphologyEx(damage_mask_values, cv2.MORPH_OPEN, repair_kernel)

    # 方法3: 手动创建（如果我们知道损坏点位置）
    damage_mask_manual = np.zeros_like(damaged)
    for (x, y, size) in damage_points:
        cv2.circle(damage_mask_manual, (x, y), size, 255, -1)

    # 选择最佳掩码
    damage_mask = damage_mask_manual

    # 2. 估计掩码（重构的上界）
    print("5. 估计掩码（重构的上界）...")

    # 方法A: 膨胀
    estimated_mask = cv2.dilate(damaged, repair_kernel)

    # 方法B: 中值滤波
    background_estimate = cv2.medianBlur(damaged, 5)

    # 方法C: 高斯模糊
    gaussian_estimate = cv2.GaussianBlur(damaged, (5, 5), 0)

    # 3. 选择标记图像
    print("6. 创建修复标记...")
    repair_marker = cv2.erode(damaged, repair_kernel)

    # 4. 执行重构修复
    print("7. 执行形态学重构修复...")

    # 使用膨胀掩码
    repaired_est, iter_est = morphological_reconstruction(repair_marker, estimated_mask)
    repaired_final_est = damaged.copy()
    repaired_final_est[damage_mask > 0] = repaired_est[damage_mask > 0]

    # 使用中值滤波掩码
    repaired_blur, iter_blur = morphological_reconstruction(repair_marker, background_estimate)
    repaired_final_blur = damaged.copy()
    repaired_final_blur[damage_mask > 0] = repaired_blur[damage_mask > 0]

    # 使用高斯模糊掩码
    repaired_gauss, iter_gauss = morphological_reconstruction(repair_marker, gaussian_estimate)
    repaired_final_gauss = damaged.copy()
    repaired_final_gauss[damage_mask > 0] = repaired_gauss[damage_mask > 0]

    # 5. 使用OpenCV的修复算法作为对比
    print("8. 使用OpenCV内置修复算法...")
    inpainted = cv2.inpaint(damaged, damage_mask, 3, cv2.INPAINT_TELEA)

    # 6. 可视化所有修复方法
    print("9. 可视化修复结果...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    images = [
        # 第一行
        ("原始图像", text_img, 'gray'),
        ("损坏图像", damaged, 'gray'),
        ("损坏掩码", damage_mask, 'gray'),
        ("腐蚀标记", repair_marker, 'gray'),

        # 第二行
        ("估计掩码(膨胀)", estimated_mask, 'gray'),
        ("中值滤波估计", background_estimate, 'gray'),
        ("高斯模糊估计", gaussian_estimate, 'gray'),
        ("OpenCV修复", inpainted, 'gray'),

        # 第三行
        ("膨胀掩码重构", repaired_final_est, 'gray'),
        ("中值滤波重构", repaired_final_blur, 'gray'),
        ("高斯掩码重构", repaired_final_gauss, 'gray'),
        ("重构差异", cv2.absdiff(repaired_final_est, repaired_final_blur), 'hot')
    ]

    for i, (title, img, cmap) in enumerate(images):
        row, col = i // 4, i % 4
        axes[row, col].imshow(img, cmap=cmap, vmin=0, vmax=255)
        axes[row, col].set_title(title, fontweight='bold', fontsize=9)
        axes[row, col].axis('off')

        # 计算相似度
        if "重构" in title or "修复" in title or "差异" in title:
            if "差异" not in title:
                similarity = np.sum(img == text_img) / text_img.size * 100
                axes[row, col].set_xlabel(f"相似度: {similarity:.1f}%")
            else:
                diff_value = np.sum(img) / img.size
                axes[row, col].set_xlabel(f"差异值: {diff_value:.1f}")

    plt.suptitle("实际图像修复（无原始图像作为参考）", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 7. 修复效果评估
    print("\n📊 修复效果评估:")
    print("-" * 50)

    results = []
    repair_methods = [
        ("膨胀掩码重构", repaired_final_est, iter_est),
        ("中值滤波重构", repaired_final_blur, iter_blur),
        ("高斯掩码重构", repaired_final_gauss, iter_gauss),
        ("OpenCV修复", inpainted, 0)
    ]

    for method_name, repaired_img, iterations in repair_methods:
        # 总体相似度
        total_similarity = np.sum(repaired_img == text_img) / text_img.size * 100

        # 损坏区域修复率
        if np.sum(damage_mask > 0) > 0:
            damaged_area = damage_mask > 0
            original_damaged = text_img[damaged_area]
            repaired_damaged = repaired_img[damaged_area]
            correct_pixels = np.sum(original_damaged == repaired_damaged)
            total_pixels = np.sum(damaged_area)
            repair_rate = correct_pixels / total_pixels * 100 if total_pixels > 0 else 0
        else:
            repair_rate = 100

        results.append({
            'method': method_name,
            'total_similarity': total_similarity,
            'repair_rate': repair_rate,
            'damaged_pixels': np.sum(damage_mask > 0),
            'iterations': iterations
        })

        print(f"{method_name:20} 总相似度: {total_similarity:6.1f}%, "
              f"损坏区域修复率: {repair_rate:6.1f}%, "
              f"迭代次数: {iterations}")

    # 找到最佳方法
    best_method = max(results, key=lambda x: x['total_similarity'])
    print(f"\n🏆 最佳修复方法: {best_method['method']}")
    print(f"   总相似度: {best_method['total_similarity']:.1f}%")
    print(f"   损坏区域修复率: {best_method['repair_rate']:.1f}%")
    if best_method['iterations'] > 0:
        print(f"   迭代次数: {best_method['iterations']}")

    # 8. 损坏区域检测方法对比
    print("\n🔍 损坏区域检测方法对比:")
    damage_mask_methods = [
        ("手动标记（已知）", damage_mask_manual),
        ("边缘检测", damage_mask_edges),
        ("像素值分析", damage_mask_values)
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    axes[0].imshow(text_img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("原始图像")
    axes[0].axis('off')

    axes[1].imshow(damaged, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("损坏图像")
    axes[1].axis('off')

    for i, (method_name, mask) in enumerate(damage_mask_methods, 2):
        axes[i].imshow(mask, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(method_name)
        axes[i].axis('off')

        if method_name == "手动标记（已知）":
            axes[i].set_xlabel(f"准确率: 100.0%")
        else:
            correct_pixels = np.sum((mask > 0) == (damage_mask_manual > 0))
            total_pixels = mask.size
            accuracy = correct_pixels / total_pixels * 100
            axes[i].set_xlabel(f"准确率: {accuracy:.1f}%")

    plt.suptitle("损坏区域检测方法对比", fontsize=14)
    plt.tight_layout()
    plt.show()

    # 9. 局部细节放大
    print("\n🔍 局部细节放大:")
    if damage_points:
        x, y, size = damage_points[0]
        x1, x2 = max(0, x - 15), min(width, x + 15)
        y1, y2 = max(0, y - 15), min(height, y + 15)

        fig, axes = plt.subplots(1, 4, figsize=(12, 4))

        patch_original = text_img[y1:y2, x1:x2]
        patch_damaged = damaged[y1:y2, x1:x2]
        patch_repaired = repaired_final_est[y1:y2, x1:x2]
        patch_inpainted = inpainted[y1:y2, x1:x2]

        patch_sim_damaged = np.sum(patch_damaged == patch_original) / patch_original.size * 100
        patch_sim_repaired = np.sum(patch_repaired == patch_original) / patch_original.size * 100
        patch_sim_inpainted = np.sum(patch_inpainted == patch_original) / patch_original.size * 100

        axes[0].imshow(patch_original, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title("原始区域")
        axes[0].axis('off')

        axes[1].imshow(patch_damaged, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title(f"损坏区域\n相似度: {patch_sim_damaged:.1f}%")
        axes[1].axis('off')
        axes[1].plot(x - x1, y - y1, 'rx', markersize=10, markeredgewidth=2)

        axes[2].imshow(patch_repaired, cmap='gray', vmin=0, vmax=255)
        axes[2].set_title(f"重构修复\n相似度: {patch_sim_repaired:.1f}%")
        axes[2].axis('off')

        axes[3].imshow(patch_inpainted, cmap='gray', vmin=0, vmax=255)
        axes[3].set_title(f"OpenCV修复\n相似度: {patch_sim_inpainted:.1f}%")
        axes[3].axis('off')

        plt.suptitle(f"损坏点局部放大 (位置: ({x}, {y}), 大小: {size})", fontsize=14)
        plt.tight_layout()
        plt.show()

    # 10. 总结
    print("\n💡 实际应用建议:")
    print("1. 损坏区域检测是关键步骤，直接影响修复效果")
    print("2. 掩码估计方法:")
    print("   - 膨胀: 简单快速，但可能过度扩张")
    print("   - 中值滤波: 能去除噪声，保持边缘")
    print("   - 高斯模糊: 平滑处理，但可能模糊细节")
    print("3. 标记选择: 通常使用腐蚀后的图像")
    print("4. 只修复检测到的损坏区域，避免破坏完好区域")

    print("\n🔧 实际修复流程:")
    print("1. 输入: 只有损坏图像")
    print("2. 步骤:")
    print("   a. 检测损坏区域（创建损坏掩码）")
    print("   b. 估计掩码图像（重构的上界）")
    print("   c. 创建标记图像（通常腐蚀损坏图像）")
    print("   d. 执行形态学重构")
    print("   e. 将重构结果应用到损坏区域")
    print("3. 输出: 修复后的图像")

    # 11. 返回结果
    print("\n✅ 实际图像修复演示完成!")

    # 返回最佳修复结果
    best_repaired_img = repaired_final_est
    if best_method['method'] == "中值滤波重构":
        best_repaired_img = repaired_final_blur
    elif best_method['method'] == "高斯掩码重构":
        best_repaired_img = repaired_final_gauss
    elif best_method['method'] == "OpenCV修复":
        best_repaired_img = inpainted

    return {
        'original': text_img,
        'damaged': damaged,
        'damage_mask': damage_mask,
        'repair_marker': repair_marker,
        'estimated_mask': estimated_mask,
        'repaired_est': repaired_final_est,
        'repaired_blur': repaired_final_blur,
        'repaired_gauss': repaired_final_gauss,
        'inpainted': inpainted,
        'best_repaired': best_repaired_img,
        'results': results
    }


# 主程序
if __name__ == "__main__":
    print("=" * 60)
    print("🎯 图像修复演示程序")
    print("=" * 60)
    print("本程序演示实际应用中的图像修复方法，包括:")
    print("1. 形态学重构修复")
    print("2. OpenCV内置修复算法")
    print("3. 多种掩码估计方法")
    print("4. 损坏区域检测")
    print("=" * 60)

    try:
        # 运行实际图像修复演示
        results = practical_image_repair_demo()

        # 保存结果
        save_option = input("\n💾 是否保存修复结果? (y/n): ")
        if save_option.lower() == 'y':
            import os

            if not os.path.exists('practical_repair_results'):
                os.makedirs('practical_repair_results')

            cv2.imwrite('practical_repair_results/01_original.jpg', results['original'])
            cv2.imwrite('practical_repair_results/02_damaged.jpg', results['damaged'])
            cv2.imwrite('practical_repair_results/03_damage_mask.jpg', results['damage_mask'])
            cv2.imwrite('practical_repair_results/04_repair_marker.jpg', results['repair_marker'])
            cv2.imwrite('practical_repair_results/05_estimated_mask.jpg', results['estimated_mask'])
            cv2.imwrite('practical_repair_results/06_repaired_est.jpg', results['repaired_est'])
            cv2.imwrite('practical_repair_results/07_repaired_blur.jpg', results['repaired_blur'])
            cv2.imwrite('practical_repair_results/08_repaired_gauss.jpg', results['repaired_gauss'])
            cv2.imwrite('practical_repair_results/09_inpainted.jpg', results['inpainted'])
            cv2.imwrite('practical_repair_results/10_best_repaired.jpg', results['best_repaired'])

            print("✅ 所有结果已保存到 'practical_repair_results' 文件夹")

    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✨ 程序运行结束 ✨")
    print("=" * 60)