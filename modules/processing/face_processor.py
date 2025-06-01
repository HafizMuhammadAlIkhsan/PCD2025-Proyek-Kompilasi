import streamlit as st
import numpy as np
import cv2
import os
from io import BytesIO
from PIL import Image
import time

class FaceProcessor:
    def __init__(self):
        self.base_data_dir = os.path.join(os.getcwd(), "data")
        self.dataset_dir = os.path.join(self.base_data_dir, "dataset")
        self.processed_dir = os.path.join(self.base_data_dir, "processed_dataset")
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def capture_faces(self, person_name):
        person_folder = os.path.join(self.dataset_dir, person_name)
        if os.path.exists(person_folder):
            raise ValueError(f"Folder untuk '{person_name}' sudah ada di dataset. Silakan gunakan nama lain atau hapus folder yang sudah ada.")
        os.makedirs(person_folder, exist_ok=True)

        cap = cv2.VideoCapture(0)
        time.sleep(2)
        if not cap.isOpened():
            raise ValueError("Tidak dapat membuka webcam.")

        num_images = 0
        max_images = 4
        images = []
        placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            while num_images < max_images:
                ret, frame = cap.read()
                if not ret:
                    raise ValueError("Tidak bisa membaca frame.")
                
                faces = self.face_cascade.detectMultiScale(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        face = frame[y:y+h, x:x+w]
                        img_name = os.path.join(person_folder, f"img_{num_images}.jpg")
                        cv2.imwrite(img_name, face)
                        num_images += 1
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        placeholder.image(frame_rgb, channels="RGB", caption=f"Gambar {num_images}/{max_images}")
                        progress_bar.progress(num_images / max_images)
                        status_text.text(f"Menyimpan gambar {num_images} dari {max_images}...")
                        images.append(img_name)
                        break
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    placeholder.image(frame_rgb, channels="RGB", caption="Tidak ada wajah terdeteksi.")
                
                time.sleep(0.1)
            
            return {"count": num_images, "images": images}
        finally:
            cap.release()
            placeholder.empty()
            progress_bar.empty()
            status_text.empty()

    def process_dataset_images(self, person_name, process_option):
        input_folder = os.path.join(self.dataset_dir, person_name)
        if not os.path.exists(input_folder):
            raise ValueError(f"No dataset found for '{person_name}'")
        
        output_folder = os.path.join(self.processed_dir, person_name)
        os.makedirs(output_folder, exist_ok=True)
        
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        processed_images = []
        
        for idx, file_name in enumerate(image_files):
            img_path = os.path.join(input_folder, file_name)
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to read image {file_name}")
            
            if process_option == "Tambah Noise Salt and Pepper":
                processed = self._add_salt_pepper_noise(image)
                out_name = f"noise_{file_name}"
            elif process_option == "Hilangkan Noise":
                noisy = self._add_salt_pepper_noise(image)
                processed = self._remove_noise(noisy)
                out_name = f"denoise_{file_name}"
            elif process_option == "Tajamkan Gambar":
                processed = self._sharpen_image(image)
                out_name = f"sharpen_{file_name}"
            else:
                raise ValueError("Invalid processing option")
            
            out_path = os.path.join(output_folder, out_name)
            cv2.imwrite(out_path, processed)
            processed_images.append(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        
        return processed_images

    def get_dataset_names(self):
        return [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]

    def _add_salt_pepper_noise(self, image, salt_prob=0.01, pepper_prob=0.01):
        noisy = np.copy(image)
        num_salt = np.ceil(salt_prob * image.size)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 255
        num_pepper = np.ceil(pepper_prob * image.size)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 0
        return noisy

    def _remove_noise(self, image):
        return cv2.medianBlur(image, 5)

    def _sharpen_image(self, image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)