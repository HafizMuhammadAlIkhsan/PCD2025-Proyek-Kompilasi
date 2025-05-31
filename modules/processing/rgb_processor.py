import numpy as np
import cv2

class RGBProcessor:
    def process_rgb_split(self, image_data):
        try:
            np_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            b, g, r = cv2.split(img)
            return {"R": r.tolist(), "G": g.tolist(), "B": b.tolist()}
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")