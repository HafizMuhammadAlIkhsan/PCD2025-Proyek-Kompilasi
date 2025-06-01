import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class ShapeProcessor:
    def generate_freeman_chain_code(self, contour):
        chain_code = []
        if len(contour) < 2:
            return chain_code

        directions = {
            (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
            (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7
        }

        for i in range(len(contour)):
            p1 = contour[i][0]
            p2 = contour[(i + 1) % len(contour)][0]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            norm_dx = np.sign(dx)
            norm_dy = np.sign(dy)
            code = directions.get((norm_dx, norm_dy))
            if code is not None:
                chain_code.append(code)
        return chain_code

    def process_freeman_chain_code(self, uploaded_file, threshold_value=127):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return None, "Gagal memuat gambar. Pastikan file valid."

        _, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        chain_code_str = "Tidak ada kontur ditemukan."
        img_contour_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        axs[0, 0].imshow(img, cmap='gray')
        axs[0, 0].set_title('Citra Asli (Grayscale)')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(binary_img, cmap='gray')
        axs[0, 1].set_title('Citra Biner (Hasil Threshold)')
        axs[0, 1].axis('off')

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(img_contour_display, [largest_contour], -1, (0, 255, 0), 1)
            chain_code_result = self.generate_freeman_chain_code(largest_contour)

            max_line_len = 70
            wrapped_code = ""
            current_line_len = 0
            for i, code_str in enumerate(map(str, chain_code_result)):
                item = code_str + (", " if i < len(chain_code_result) - 1 else "")
                if current_line_len + len(item) > max_line_len:
                    wrapped_code += "\n"
                    current_line_len = 0
                wrapped_code += item
                current_line_len += len(item)

            chain_code_str = (
                f"Jumlah Kontur Total: {len(contours)}\n"
                f"Kode Rantai Kontur Terbesar (Panjang {len(chain_code_result)}):\n"
                f"{wrapped_code}"
            )

        img_rgb_display = cv2.cvtColor(img_contour_display, cv2.COLOR_BGR2RGB)
        axs[1, 0].imshow(img_rgb_display)
        axs[1, 0].set_title('Kontur Terbesar Terdeteksi')
        axs[1, 0].axis('off')

        axs[1, 1].axis('off')
        axs[1, 1].text(0.05, 0.95, chain_code_str, ha='left', va='top', fontsize=9, wrap=True)
        axs[1, 1].set_title('Hasil Kode Rantai')

        plt.tight_layout(pad=1.5)
        plt.suptitle("Analisis Kode Rantai", fontsize=16)
        plt.subplots_adjust(top=0.92)

        return fig, chain_code_str

    def process_canny_edge(self, uploaded_file, low_threshold=50, high_threshold=150):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return None, "Gagal memuat gambar. Pastikan file valid."

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Citra Asli')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(blurred, cmap='gray')
        plt.title('Grayscale + Gaussian Blur')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(edges, cmap='gray')
        plt.title(f'Tepi Canny (Th={low_threshold},{high_threshold})')
        plt.axis('off')

        plt.tight_layout()
        return fig, None

    def create_binary_text_image(self, height=150, width=400):
        binary_img = np.zeros((height, width), dtype=np.uint8)
        text_lines = ["Baris Teks Satu", "Ini Baris Dua", "Testing 123"]
        start_y = 40
        line_height = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = 255
        thickness = 2

        y = start_y
        for line in text_lines:
            x = 20
            cv2.putText(binary_img, line, (x, y), font, font_scale, font_color, thickness)
            y += line_height
        
        return binary_img

    def process_integral_projection_default(self, uploaded_file=None):
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None, "Gagal memuat gambar. Pastikan file valid."
            _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        else:
            binary_img = self.create_binary_text_image()

        binary_norm = binary_img / 255.0
        horizontal_projection = np.sum(binary_norm, axis=0)
        vertical_projection = np.sum(binary_norm, axis=1)

        fig = plt.figure(figsize=(10, 7))
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)

        ax_img = fig.add_subplot(gs[1, 0])
        ax_img.imshow(binary_img, cmap='gray', aspect='auto')
        ax_img.set_title('Citra Biner')
        ax_img.set_xlabel('Indeks Kolom')
        ax_img.set_ylabel('Indeks Baris (0 di atas)')

        ax_hproj = fig.add_subplot(gs[0, 0], sharex=ax_img)
        ax_hproj.plot(np.arange(binary_img.shape[1]), horizontal_projection, color='blue')
        ax_hproj.set_title('Proyeksi Horizontal (Profil Vertikal)')
        ax_hproj.set_ylabel('Jumlah Piksel Putih')
        plt.setp(ax_hproj.get_xticklabels(), visible=False)
        ax_hproj.grid(axis='y', linestyle='--', alpha=0.6)

        ax_vproj = fig.add_subplot(gs[1, 1], sharey=ax_img)
        ax_vproj.plot(vertical_projection, np.arange(binary_img.shape[0]), color='red')
        ax_vproj.set_title('Proyeksi Vertikal')
        ax_vproj.set_xlabel('Jumlah Piksel Putih')
        ax_vproj.invert_yaxis()
        plt.setp(ax_vproj.get_yticklabels(), visible=False)
        ax_vproj.grid(axis='x', linestyle='--', alpha=0.6)

        plt.suptitle("Visualisasi Proyeksi Integral pada Citra Teks", fontsize=14)
        plt.tight_layout()

        return fig, None

    def process_integral_projection_otsu(self, uploaded_file):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, "Gagal memuat gambar. Pastikan file valid."

        _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_norm = binary_img / 255.0

        horizontal_projection = np.sum(binary_norm, axis=0)
        vertical_projection = np.sum(binary_norm, axis=1)

        height, width = binary_norm.shape
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)

        ax_img = fig.add_subplot(gs[1, 0])
        ax_img.imshow(binary_norm, cmap='gray')
        ax_img.set_title('Citra Biner (Objek=1)')
        ax_img.set_xlabel('Indeks Kolom')
        ax_img.set_ylabel('Indeks Baris')

        ax_hproj = fig.add_subplot(gs[0, 0], sharex=ax_img)
        ax_hproj.plot(np.arange(width), horizontal_projection, color='blue')
        ax_hproj.set_title('Proyeksi Horizontal (Profil Vertikal)')
        ax_hproj.set_ylabel('Jumlah Piksel')
        plt.setp(ax_hproj.get_xticklabels(), visible=False)
        ax_hproj.grid(axis='y', linestyle='--', alpha=0.6)

        ax_vproj = fig.add_subplot(gs[1, 1], sharey=ax_img)
        ax_vproj.plot(vertical_projection, np.arange(height), color='red')
        ax_vproj.set_title('Proyeksi Vertikal')
        ax_vproj.set_xlabel('Jumlah Piksel')
        ax_vproj.invert_yaxis()
        plt.setp(ax_vproj.get_yticklabels(), visible=False)
        ax_vproj.grid(axis='x', linestyle='--', alpha=0.6)

        plt.suptitle("Analisis Proyeksi Integral", fontsize=14)
        plt.tight_layout()

        return fig, None