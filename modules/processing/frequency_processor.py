import numpy as np
import cv2
from io import BytesIO
from PIL import Image

class FrequencyProcessor:
    def process_frequency_operation(self, image_data, operation, param=None):
        try:
            np_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            
            if operation == "convolution":
                processed_img = self._apply_convolution(img, param)
            elif operation == "padding":
                processed_img = self._apply_zero_padding(img, int(param))
            elif operation == "filter":
                processed_img = self._apply_filter(img, param)
            elif operation == "fourier":
                processed_img = self._apply_fourier_transform(img)
            elif operation == "noise_reduction":
                processed_img = self._reduce_periodic_noise(img)
            else:
                raise ValueError("Invalid operation")
            
            return self._convert_to_pil(processed_img)
        except Exception as e:
            raise ValueError(f"Frequency operation failed: {str(e)}")

    def _apply_convolution(self, image, kernel_type="average"):
        kernels = {
            "average": np.ones((3, 3), np.float32) / 9,
            "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            "edge": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        }
        kernel = kernels.get(kernel_type, kernels["average"])
        return cv2.filter2D(image, -1, kernel)

    def _apply_zero_padding(self, image, padding_size=10):
        return cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    def _apply_filter(self, image, filter_type="low"):
        if filter_type == "low":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif filter_type == "high":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return cv2.filter2D(image, -1, kernel)
        elif filter_type == "band":
            low_pass = cv2.GaussianBlur(image, (9, 9), 0)
            high_pass = image - low_pass
            return low_pass + high_pass
        else:
            raise ValueError("Invalid filter type")

    def _apply_fourier_transform(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        return mag_norm.astype(np.uint8)

    def _reduce_periodic_noise(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        mask[crow-r:crow+r, ccol-r:ccol+r] = 0
        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        return img_back_norm.astype(np.uint8)

    def _convert_to_pil(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        pil_img = Image.fromarray(img_rgb)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()