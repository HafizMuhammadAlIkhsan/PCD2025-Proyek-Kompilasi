import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

class ImageProcessor:
    def process_image(self, image_data):
        try:
            np_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            b, g, r = cv2.split(img)
            return {"R": r.tolist(), "G": g.tolist(), "B": b.tolist()}
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")

    def process_arithmetic_operation(self, image_data, operation, value=None):
        try:
            np_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            if operation == "add":
                result_img = cv2.add(img, np.full(img.shape, value, dtype=np.uint8))
            elif operation == "subtract":
                result_img = cv2.subtract(img, np.full(img.shape, value, dtype=np.uint8))
            elif operation == "max":
                result_img = np.maximum(img, np.full(img.shape, value, dtype=np.uint8))
            elif operation == "min":
                result_img = np.minimum(img, np.full(img.shape, value, dtype=np.uint8))
            elif operation == "inverse":
                result_img = cv2.bitwise_not(img)
            else:
                raise ValueError("Invalid operation")
            return self._convert_to_pil(result_img)
        except Exception as e:
            raise ValueError(f"Arithmetic operation failed: {str(e)}")

    def process_logic_operation(self, image_data1, image_data2, operation):
        try:
            np_array1 = np.frombuffer(image_data1, np.uint8)
            img1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)
            if img1 is None:
                raise ValueError("Failed to decode first image")
            if operation == "not":
                result_img = cv2.bitwise_not(img1)
            elif operation in ["and", "xor"]:
                if image_data2 is None:
                    raise ValueError("Second image required for AND/XOR operations")
                np_array2 = np.frombuffer(image_data2, np.uint8)
                img2 = cv2.imdecode(np_array2, cv2.IMREAD_COLOR)
                if img2 is None:
                    raise ValueError("Failed to decode second image")
                if operation == "and":
                    result_img = cv2.bitwise_and(img1, img2)
                elif operation == "xor":
                    result_img = cv2.bitwise_xor(img1, img2)
            else:
                raise ValueError("Invalid operation")
            return self._convert_to_pil(result_img)
        except Exception as e:
            raise ValueError(f"Logic operation failed: {str(e)}")

    def process_grayscale_conversion(self, image_data):
        try:
            np_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return self._convert_to_pil(gray_img)
        except Exception as e:
            raise ValueError(f"Grayscale conversion failed: {str(e)}")

    def process_histogram_generation(self, image_data):
        try:
            np_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grayscale_hist = self._generate_grayscale_histogram(gray_img)
            color_hist = self._generate_color_histogram(img)
            return grayscale_hist, color_hist
        except Exception as e:
            raise ValueError(f"Histogram generation failed: {str(e)}")

    def process_histogram_equalization(self, image_data):
        try:
            np_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Failed to decode image")
            equalized_img = cv2.equalizeHist(img)
            return self._convert_to_pil(equalized_img)
        except Exception as e:
            raise ValueError(f"Histogram equalization failed: {str(e)}")

    def process_histogram_specification(self, image_data, ref_image_data):
        try:
            np_array = np.frombuffer(image_data, np.uint8)
            ref_np_array = np.frombuffer(ref_image_data, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
            ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_GRAYSCALE)
            if img is None or ref_img is None:
                raise ValueError("Failed to decode image(s)")
            specified_img = cv2.equalizeHist(ref_img)  # Simplified as per original code
            return self._convert_to_pil(specified_img)
        except Exception as e:
            raise ValueError(f"Histogram specification failed: {str(e)}")

    def process_image_statistics(self, image_data):
        try:
            np_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Failed to decode image")
            mean_intensity = np.mean(img)
            std_deviation = np.std(img)
            return mean_intensity, std_deviation
        except Exception as e:
            raise ValueError(f"Image statistics failed: {str(e)}")

    def _convert_to_pil(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        pil_img = Image.fromarray(img_rgb)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    def _generate_grayscale_histogram(self, gray_img):
        plt.figure()
        plt.hist(gray_img.ravel(), 256, [0, 256])
        buf = BytesIO()
        plt.savefig(buf, format="PNG")
        plt.close()
        return buf.getvalue()

    def _generate_color_histogram(self, img):
        plt.figure()
        for i, color in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
        buf = BytesIO()
        plt.savefig(buf, format="PNG")
        plt.close()
        return buf.getvalue()