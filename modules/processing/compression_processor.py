import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import subprocess

class CompressionProcessor:
    def __init__(self):
        self.output_ssim_dir = os.path.join("output", "ssim")
        self.output_compressed_dir = os.path.join("output", "compressed")
        os.makedirs(self.output_ssim_dir, exist_ok=True)
        os.makedirs(self.output_compressed_dir, exist_ok=True)

    def compress_jpeg(self, uploaded_file):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_original = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        if img_original is None:
            raise ValueError("Failed to load image.")

        is_color = len(img_original.shape) == 3
        img_original_cv = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) if is_color else img_original
        original_size_bytes = len(file_bytes)

        jpeg_qualities = [95, 75, 50, 25, 10]
        results = []
        min_dim = min(img_original_cv.shape[:2])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        win_size = max(3, win_size)

        for quality in jpeg_qualities:
            jpeg_path = os.path.join(self.output_ssim_dir, f"image_jpeg_q{quality}.jpg")
            cv2.imwrite(jpeg_path, img_original, [cv2.IMWRITE_JPEG_QUALITY, quality])
            compressed_size_bytes = os.path.getsize(jpeg_path)

            img_compressed = cv2.imread(jpeg_path, cv2.IMREAD_UNCHANGED)
            img_compressed_cv = cv2.cvtColor(img_compressed, cv2.COLOR_BGR2RGB) if is_color else img_compressed

            if img_original_cv.shape != img_compressed_cv.shape:
                continue

            psnr_value = cv2.PSNR(img_original_cv, img_compressed_cv)
            try:
                ssim_value = ssim(
                    img_original_cv, img_compressed_cv,
                    channel_axis=2 if is_color else None,
                    win_size=win_size,
                    data_range=img_original_cv.max() - img_original_cv.min()
                )
            except:
                ssim_value = None

            results.append({
                'Method': f'JPEG_Q{quality}',
                'Quality': quality,
                'File Size (KB)': compressed_size_bytes / 1024,
                'Compression Ratio': original_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else float('inf'),
                'PSNR (dB)': round(psnr_value, 2),
                'SSIM': round(ssim_value, 4) if ssim_value is not None else 'N/A'
            })

        # Image comparison plot (Q95, Q10)
        img_q95 = cv2.imread(os.path.join(self.output_ssim_dir, 'image_jpeg_q95.jpg'), cv2.IMREAD_UNCHANGED)
        img_q10 = cv2.imread(os.path.join(self.output_ssim_dir, 'image_jpeg_q10.jpg'), cv2.IMREAD_UNCHANGED)
        cmap_val = None if is_color else 'gray'
        
        fig_comparison, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_original_cv, cmap=cmap_val)
        axes[0].set_title(f'Original ({original_size_bytes / 1024:.2f} KB)')
        axes[0].axis('off')

        if img_q95 is not None:
            q95_size = os.path.getsize(os.path.join(self.output_ssim_dir, 'image_jpeg_q95.jpg'))
            img_q95_vis = cv2.cvtColor(img_q95, cv2.COLOR_BGR2RGB) if is_color else img_q95
            axes[1].imshow(img_q95_vis, cmap=cmap_val)
            axes[1].set_title(f'JPEG Q95 ({q95_size / 1024:.2f} KB)')
            axes[1].axis('off')
        else:
            axes[1].set_title('JPEG Q95 (Error)')
            axes[1].axis('off')

        if img_q10 is not None:
            q10_size = os.path.getsize(os.path.join(self.output_ssim_dir, 'image_jpeg_q10.jpg'))
            img_q10_vis = cv2.cvtColor(img_q10, cv2.COLOR_BGR2RGB) if is_color else img_q10
            axes[2].imshow(img_q10_vis, cmap=cmap_val)
            axes[2].set_title(f'JPEG Q10 ({q10_size / 1024:.2f} KB)')
            axes[2].axis('off')
        else:
            axes[2].set_title('JPEG Q10 (Error)')
            axes[2].axis('off')

        plt.tight_layout()

        # Quality vs. File Size plot
        qualities = [r['Quality'] for r in results]
        file_sizes = [r['File Size (KB)'] for r in results]
        fig_sizes = plt.figure(figsize=(8, 6))
        plt.plot(qualities, file_sizes, marker='o', color='b', label='File Size (KB)')
        plt.xlabel('JPEG Quality')
        plt.ylabel('File Size (KB)')
        plt.title('JPEG Quality vs. File Size')
        plt.gca().invert_xaxis()
        plt.grid(True)
        plt.legend()

        # Quality vs. PSNR and SSIM plot
        psnr_values = [r['PSNR (dB)'] for r in results]
        ssim_values = [r['SSIM'] if r['SSIM'] != 'N/A' else 0 for r in results]
        fig_metrics, ax1 = plt.subplots(figsize=(8, 6))
        ax1.set_xlabel('JPEG Quality')
        ax1.set_ylabel('PSNR (dB)', color='tab:red')
        ax1.plot(qualities, psnr_values, marker='o', color='tab:red', label='PSNR')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.invert_xaxis()

        ax2 = ax1.twinx()
        ax2.set_ylabel('SSIM', color='tab:blue')
        ax2.plot(qualities, ssim_values, marker='s', color='tab:blue', label='SSIM')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        plt.title('JPEG Quality vs. PSNR and SSIM')
        plt.grid(True)

        results_df = pd.DataFrame(results)
        return fig_comparison, fig_metrics, fig_sizes, results_df

    def compress_png(self, uploaded_file):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_original = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        if img_original is None:
            raise ValueError("Failed to load image.")

        is_color = len(img_original.shape) == 3
        img_original_cv = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) if is_color else img_original
        original_size_bytes = len(file_bytes)

        png_compression_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        results = []
        min_dim = min(img_original_cv.shape[:2])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        win_size = max(3, win_size)

        for level in png_compression_levels:
            png_path = os.path.join(self.output_compressed_dir, f"image_png_level{level}.png")
            cv2.imwrite(png_path, img_original, [cv2.IMWRITE_PNG_COMPRESSION, level])
            png_size_bytes = os.path.getsize(png_path)

            try:
                subprocess.run(['optipng', '-o7', png_path], check=True, capture_output=True)
                png_size_opt = os.path.getsize(png_path)
            except:
                png_size_opt = png_size_bytes

            img_compressed = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
            img_compressed_cv = cv2.cvtColor(img_compressed, cv2.COLOR_BGR2RGB) if is_color else img_compressed

            psnr_value = cv2.PSNR(img_original_cv, img_compressed_cv)
            mse = np.mean((img_original_cv.astype(float) - img_compressed_cv.astype(float)) ** 2)
            psnr_manual = float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

            try:
                ssim_value = ssim(
                    img_original_cv, img_compressed_cv,
                    channel_axis=2 if is_color else None,
                    win_size=win_size,
                    data_range=img_original_cv.max() - img_original_cv.min()
                )
            except:
                ssim_value = None

            is_identical = np.array_equal(img_original_cv, img_compressed_cv)

            results.append({
                'Method': f'PNG_Level_{level}',
                'Quality': 'Lossless',
                'File Size (KB)': round(png_size_bytes / 1024, 2),
                'Optimized Size (KB)': round(png_size_opt / 1024, 2),
                'Compression Ratio': round(original_size_bytes / png_size_bytes, 2) if png_size_bytes > 0 else 'Inf',
                'PSNR (dB)': 'Inf' if psnr_value == float('inf') else round(psnr_value, 2),
                'SSIM': round(ssim_value, 4) if ssim_value is not None else 'N/A',
                'Identical': is_identical
            })

        # Image comparison plot (Level 9)
        png_path_9 = os.path.join(self.output_compressed_dir, 'image_png_level9.png')
        img_png9 = cv2.imread(png_path_9, cv2.IMREAD_UNCHANGED)
        cmap_val = None if is_color else 'gray'

        fig_comparison, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_original_cv, cmap=cmap_val)
        axes[0].set_title(f'Original ({original_size_bytes / 1024:.2f} KB)')
        axes[0].axis('off')

        if img_png9 is not None:
            png_size_9 = os.path.getsize(png_path_9)
            img_png9_vis = cv2.cvtColor(img_png9, cv2.COLOR_BGR2RGB) if is_color else img_png9
            axes[1].imshow(img_png9_vis, cmap=cmap_val)
            axes[1].set_title(f'PNG Level 9 ({png_size_9 / 1024:.2f} KB)')
            axes[1].axis('off')
        else:
            axes[1].set_title('PNG Level 9 (Error)')
            axes[1].axis('off')

        plt.tight_layout()

        # File size comparison plot
        labels = ['Original'] + [r['Method'] for r in results]
        sizes = [original_size_bytes / 1024] + [r['File Size (KB)'] for r in results]
        fig_sizes = plt.figure(figsize=(12, 6))
        bars = plt.bar(labels, sizes, color='skyblue')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f} KB', ha='center', va='bottom')
        plt.xticks(rotation=90)
        plt.ylabel('File Size (KB)')
        plt.title('Original vs PNG Compression Levels')
        plt.tight_layout()

        results_df = pd.DataFrame(results)
        return fig_comparison, fig_sizes, results_df