import cv2
import numpy as np
from matplotlib import pyplot as plt


class LicensePlateRecognizer:
    def __init__(self):
        self.plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    def load_image(self, path):
        """加载图像并转换为RGB格式"""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法加载图像: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def preprocess_image(self, img):
        """图像预处理：灰度化、高斯模糊、边缘检测[2](@ref)"""
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 高斯模糊减少噪声[2,6](@ref)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Sobel边缘检测[2](@ref)
        sobelx = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_8U, 0, 1, ksize=3)
        sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

        # 二值化[2](@ref)
        _, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return gray, blurred, sobel, binary

    def locate_license_plate(self, img, binary):
        """在二值图像中查找车牌区域[2,4](@ref)"""
        # 形态学操作：闭操作连接车牌区域[2](@ref)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓[2](@ref)
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 对轮廓按面积排序（降序）[2](@ref)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # 存储候选车牌区域
        plate_candidates = []

        for contour in contours:
            # 计算轮廓的周长
            perimeter = cv2.arcLength(contour, True)
            # 多边形逼近[2](@ref)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # 如果是四边形（车牌通常是矩形）[2](@ref)
            if len(approx) == 4:
                # 计算轮廓的边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)

                # 根据宽高比筛选（典型车牌宽高比约为3:1到5:1）[2,6](@ref)
                if 2.5 < aspect_ratio < 5.5:
                    plate_candidates.append((x, y, w, h))

        return plate_candidates

    def color_based_detection(self, img):
        """基于颜色的车牌检测方法[3,8](@ref)"""
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # 蓝色车牌范围（中国普通车牌）
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 黄色车牌范围（中国大巴、货车等）
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 合并掩码
        color_mask = cv2.bitwise_or(blue_mask, yellow_mask)

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        plate_candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = w * h

            # 根据宽高比和面积筛选[3](@ref)
            if (2.5 < aspect_ratio < 5.5) and (area > 2000):
                plate_candidates.append((x, y, w, h))

        return plate_candidates

    def enhance_plate_image(self, plate_img):
        """车牌图像增强[7](@ref)"""
        # 转换为灰度图
        gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)

        # 直方图均衡化增强对比度[6](@ref)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_plate)

        # 二值化[2](@ref)
        _, binary_plate = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary_plate

    def segment_characters(self, plate_img):
        """字符分割[4,7](@ref)"""
        # 图像增强
        binary_plate = self.enhance_plate_image(plate_img)

        # 形态学操作连接字符笔画[4](@ref)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary_plate, kernel, iterations=1)

        # 查找轮廓[4](@ref)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        characters = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = w * h

            # 根据字符的宽高比和面积筛选[4,7](@ref)
            if (0.2 < aspect_ratio < 1.2) and (area > 100) and (h > plate_img.shape[0] * 0.3):
                characters.append((x, y, w, h))

        # 按x坐标排序（从左到右）[7](@ref)
        characters = sorted(characters, key=lambda char: char[0])

        return characters, binary_plate

    def recognize_characters(self, plate_img, characters):
        """字符识别[2,6](@ref)"""
        binary_plate = self.enhance_plate_image(plate_img)
        recognized_text = ""

        # 配置Tesseract[2](@ref)
        config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'

        for i, (x, y, w, h) in enumerate(characters):
            # 提取单个字符
            char_img = binary_plate[y:y + h, x:x + w]

            # 调整字符图像大小以提高识别准确率
            char_img = cv2.resize(char_img, (20, 40))

            # 使用Tesseract识别[2](@ref)
            try:
                text = pytesseract.image_to_string(char_img, config=config)
                recognized_text += text.strip()
            except:
                recognized_text += "?"

        return recognized_text

    def recognize_plate_text(self, plate_img):
        """直接识别整个车牌文本[2](@ref)"""
        binary_plate = self.enhance_plate_image(plate_img)

        # 配置Tesseract[2](@ref)
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'

        try:
            text = pytesseract.image_to_string(binary_plate, config=config)
            # 清理识别结果
            clean_text = ''.join(e for e in text if e.isalnum())
            return clean_text
        except:
            return "识别失败"

    def visualize_process(self, img, gray, blurred, sobel, binary, plate_img, plate_text, characters=[]):
        """可视化处理过程和结果[2](@ref)"""
        plt.figure(figsize=(20, 12))

        # 原始图像
        plt.subplot(2, 4, 1)
        plt.imshow(img)
        plt.title('原始图像')
        plt.axis('off')

        # 灰度图像
        plt.subplot(2, 4, 2)
        plt.imshow(gray, cmap='gray')
        plt.title('灰度图像')
        plt.axis('off')

        # 高斯模糊
        plt.subplot(2, 4, 3)
        plt.imshow(blurred, cmap='gray')
        plt.title('高斯模糊')
        plt.axis('off')

        # Sobel边缘检测
        plt.subplot(2, 4, 4)
        plt.imshow(sobel, cmap='gray')
        plt.title('Sobel边缘检测')
        plt.axis('off')

        # 二值图像
        plt.subplot(2, 4, 5)
        plt.imshow(binary, cmap='gray')
        plt.title('二值图像')
        plt.axis('off')

        # 车牌区域
        plt.subplot(2, 4, 6)
        plate_display = plate_img.copy()
        for (x, y, w, h) in characters:
            cv2.rectangle(plate_display, (x, y), (x + w, y + h), (255, 0, 0), 2)
        plt.imshow(plate_display)
        plt.title(f'车牌区域: {plate_text}')
        plt.axis('off')

        # 增强后的车牌
        plt.subplot(2, 4, 7)
        enhanced_plate = self.enhance_plate_image(plate_img)
        plt.imshow(enhanced_plate, cmap='gray')
        plt.title('增强车牌')
        plt.axis('off')

        # 字符分割结果
        plt.subplot(2, 4, 8)
        if characters:
            char_display = plate_img.copy()
            for i, (x, y, w, h) in enumerate(characters):
                cv2.rectangle(char_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(char_display, str(i + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            plt.imshow(char_display)
            plt.title('字符分割')
        else:
            plt.imshow(plate_img)
            plt.title('车牌图像')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def main(self, image_path, use_color_detection=True):
        """主函数：执行车牌识别流程[2](@ref)"""
        # 1. 加载图像
        img = self.load_image(image_path)

        # 2. 预处理图像
        gray, blurred, sobel, binary = self.preprocess_image(img)

        # 3. 查找车牌区域（使用两种方法）
        if use_color_detection:
            plate_candidates = self.color_based_detection(img)
        else:
            plate_candidates = self.locate_license_plate(img, binary)

        if not plate_candidates:
            print("未检测到车牌区域")
            return None

        # 获取第一个候选车牌区域
        x, y, w, h = plate_candidates[0]

        # 4. 提取车牌图像
        plate_img = img[y:y + h, x:x + w]

        # 5. 字符分割
        characters, binary_plate = self.segment_characters(plate_img)

        # 6. 字符识别
        if characters:
            plate_text = self.recognize_characters(plate_img, characters)
        else:
            plate_text = self.recognize_plate_text(plate_img)

        # 7. 可视化处理过程和结果
        self.visualize_process(img, gray, blurred, sobel, binary, plate_img, plate_text, characters)

        # 在原始图像上绘制车牌区域和识别结果
        result_img = img.copy()
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(result_img, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        plt.figure(figsize=(12, 8))
        plt.imshow(result_img)
        plt.title(f'最终识别结果: {plate_text}')
        plt.axis('off')
        plt.show()

        return plate_text


# 使用示例
if __name__ == "__main__":
    # 创建识别器实例
    recognizer = LicensePlateRecognizer()

    # 执行车牌识别
    image_path = 'car_plate.jpg'  # 请替换为你的图像路径

    try:
        result = recognizer.main(image_path)
        if result:
            print(f"✅ 识别到的车牌号码: {result}")
        else:
            print("❌ 未能识别车牌")
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")