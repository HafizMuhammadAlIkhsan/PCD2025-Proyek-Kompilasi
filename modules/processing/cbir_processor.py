import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage import util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier

class CbirProcessor:
    def __init__(self):
        self.dataset_path = os.path.join("data", "cbir_dataset")
        os.makedirs(self.dataset_path, exist_ok=True)

    def _extract_color_histogram(self, image, bins=(8, 8, 8)):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def _extract_lbp_histogram(self, image, num_points=24, radius=8):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, num_points, radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), 
                               range=(0, num_points + 2), density=True)
        return hist

    def _extract_combined_features(self, image):
        color = self._extract_color_histogram(image)
        texture = self._extract_lbp_histogram(image)
        return np.hstack([color, texture])

    def _load_dataset(self):
        dataset = []
        labels = []
        filenames = []
        for filename in os.listdir(self.dataset_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(self.dataset_path, filename)
                image = cv2.imread(path)
                if image is not None:
                    dataset.append(image)
                    label = os.path.splitext(filename)[0]
                    labels.append(label)
                    filenames.append(path)
        return dataset, labels, filenames

    def process_cbir(self, file_bytes , mode):
        file_bytes = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        query_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        db_images, db_labels, db_paths = self._load_dataset()
        if len(db_images) < 1:
            raise ValueError("No images found in data/cbir_dataset/.")

        if mode == "Color":
            query_feature = self._extract_color_histogram(query_img)
            db_features = [self._extract_color_histogram(img) for img in db_images]
        elif mode == "Texture":
            query_feature = self._extract_lbp_histogram(query_img)
            db_features = [self._extract_lbp_histogram(img) for img in db_images]
        else:
            query_feature = self._extract_combined_features(query_img)
            db_features = [self._extract_combined_features(img) for img in db_images]

        similarities = cosine_similarity([query_feature], db_features)[0]
        sorted_idx = np.argsort(similarities)[::-1]
        similar_images = [
            (cv2.cvtColor(db_images[idx], cv2.COLOR_BGR2RGB), db_labels[idx], similarities[idx])
            for idx in sorted_idx[:6]
        ]

        max_k = min(len(db_features), 10)
        return query_img_rgb, similar_images, None, max_k

    def predict_cbir_knn(self, file_bytes , mode, k):
        file_bytes = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        query_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        db_images, db_labels, _ = self._load_dataset()
        if len(db_images) < 1:
            raise ValueError("No images found in data/cbir_dataset/.")

        if mode == "Color":
            query_feature = self._extract_color_histogram(query_img)
            db_features = [self._extract_color_histogram(img) for img in db_images]
        elif mode == "Texture":
            query_feature = self._extract_lbp_histogram(query_img)
            db_features = [self._extract_lbp_histogram(img) for img in db_images]
        else:
            query_feature = self._extract_combined_features(query_img)
            db_features = [self._extract_combined_features(img) for img in db_images]

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(db_features, db_labels)
        return knn.predict([query_feature])[0]

    def _rgb_to_yiq(self, rgb):
        rgb_norm = rgb.astype(np.float32) / 255.0
        transform_matrix = np.array([
            [0.299, 0.587, 0.114],
            [0.596, -0.274, -0.322],
            [0.211, -0.523, 0.312]
        ])
        height, width, _ = rgb_norm.shape
        rgb_reshaped = rgb_norm.reshape(height * width, 3)
        yiq_reshaped = np.dot(rgb_reshaped, transform_matrix.T)
        yiq = yiq_reshaped.reshape(height, width, 3)
        return yiq

    def _display_multiple(self, images, titles, cmaps=None, figsize=(12, 8)):
        n = len(images)
        rows = int(np.ceil(n / 3))
        cols = min(n, 3)
        fig = plt.figure(figsize=figsize)
        for i in range(n):
            plt.subplot(rows, cols, i+1)
            cmap = cmaps[i] if isinstance(cmaps, list) and i < len(cmaps) else cmaps if cmaps else None
            plt.imshow(images[i], cmap=cmap)
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        return fig

    def process_color_space_analysis(self, uploaded_file):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        R, G, B = cv2.split(image_rgb)
        fig_rgb = self._display_multiple([R, G, B], ['Red Channel', 'Green Channel', 'Blue Channel'], cmaps='gray')

        image_xyz = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2XYZ)
        X, Y, Z = cv2.split(image_xyz)
        fig_xyz = self._display_multiple([X, Y, Z], ['X Component', 'Y Component (Luminance)', 'Z Component'], cmaps='gray')

        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
        L, a, b = cv2.split(image_lab)
        fig_lab = self._display_multiple([L, a, b], ['L Component (Luminance)', 'a Component (Green-Red)', 'b Component (Blue-Yellow)'], cmaps='gray')

        image_ycbcr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(image_ycbcr)
        fig_ycbcr = self._display_multiple([Y, Cb, Cr], ['Y Component (Luminance)', 'Cb Component (Blue Chrominance)', 'Cr Component (Red Chrominance)'], cmaps='gray')

        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(image_hsv)
        fig_hsv = self._display_multiple([H, S, V], ['Hue Component', 'Saturation Component', 'Value Component'], cmaps='gray')

        image_yiq = self._rgb_to_yiq(image_rgb)
        Y_yiq = np.clip(image_yiq[:, :, 0], 0, 1)
        I = np.clip(image_yiq[:, :, 1], 0, 1)
        Q = np.clip(image_yiq[:, :, 2], 0, 1)
        fig_yiq = self._display_multiple([Y_yiq, I, Q], ['Y Component (Luminance)', 'I Component (In-phase)', 'Q Component (Quadrature)'], cmaps='gray')

        luminance_components = {
            'Y from YCbCr': Y,
            'L from Lab': L,
            'Y from YIQ': Y_yiq * 255,
            'V from HSV': V
        }
        fig_lum = plt.figure(figsize=(12, 8))
        i = 1
        for name, component in luminance_components.items():
            plt.subplot(2, 2, i)
            plt.imshow(component, cmap='gray')
            plt.title(name)
            plt.axis('off')
            i += 1
        plt.tight_layout()

        return (image_rgb, fig_rgb, image_xyz, fig_xyz, image_lab, fig_lab,
                image_ycbcr, fig_ycbcr, image_hsv, fig_hsv, image_yiq, fig_yiq, fig_lum)

    def _compute_texture_statistics(self, image, window_size=15):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.float32(image)
        feature_maps = {}
        mean = cv2.boxFilter(image, -1, (window_size, window_size))
        feature_maps['mean'] = mean
        mean_sqr = cv2.boxFilter(image*image, -1, (window_size, window_size))
        variance = mean_sqr - mean*mean
        variance = np.maximum(variance, 0)
        feature_maps['variance'] = variance
        std_dev = np.sqrt(variance)
        feature_maps['std_dev'] = std_dev
        norm_maps = {key: cv2.normalize(value, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) 
                     for key, value in feature_maps.items()}
        return norm_maps

    def _compute_lbp(self, image, radius=3, n_points=24, method='uniform'):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(image, n_points, radius, method)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return lbp, hist

    def _compute_gabor_filters(self, image, frequencies, orientations):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.float32(image)
        filtered_imgs = []
        titles = []
        magnitude = np.zeros_like(image)
        for frequency in frequencies:
            for theta in orientations:
                kernel_size = int(2 * np.ceil(frequency) + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel = cv2.getGaborKernel(
                    (kernel_size, kernel_size), sigma=frequency/3, theta=theta, 
                    lambd=frequency, gamma=0.5, psi=0
                )
                filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
                magnitude += filtered * filtered
                filtered_norm = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                filtered_imgs.append(filtered_norm)
                titles.append(f'Gabor: f={frequency:.1f}, θ={theta:.1f}')
        magnitude = np.sqrt(magnitude)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        filtered_imgs.append(magnitude)
        titles.append('Gabor Magnitude')
        return filtered_imgs, titles, magnitude

    def _compute_law_texture_energy(self, image):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.float32(image)
        L5 = np.array([1, 4, 6, 4, 1])
        E5 = np.array([-1, -2, 0, 2, 1])
        S5 = np.array([-1, 0, 2, 0, -1])
        R5 = np.array([1, -4, 6, -4, 1])
        W5 = np.array([-1, 2, 0, -2, 1])
        filters_1d = {'L5': L5, 'E5': E5, 'S5': S5, 'R5': R5, 'W5': W5}
        filters_2d = {}
        texture_maps = {}
        for name_i, filter_i in filters_1d.items():
            for name_j, filter_j in filters_1d.items():
                filter_name = f"{name_i}{name_j}"
                filter_2d = np.outer(filter_i, filter_j)
                filters_2d[filter_name] = filter_2d
                filtered = cv2.filter2D(image, -1, filter_2d)
                energy = cv2.boxFilter(np.abs(filtered), -1, (15, 15), normalize=True)
                energy_norm = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                texture_maps[filter_name] = energy_norm
        selected_maps = ['L5E5', 'E5S5', 'S5S5', 'R5R5', 'L5S5', 'E5E5']
        selected_images = [texture_maps[name] for name in selected_maps]
        selected_titles = [f'Law: {name}' for name in selected_maps]
        return texture_maps, selected_images, selected_titles

    def process_texture_analysis(self, uploaded_file):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        fig_orig = self._display_multiple([image_rgb, image_gray], 
                                         ['Original RGB Image', 'Grayscale Image'], 
                                         cmaps=[None, 'gray'])

        stat_maps = self._compute_texture_statistics(image_gray)
        fig_stat = self._display_multiple([stat_maps['mean'], stat_maps['variance'], stat_maps['std_dev']],
                                         ['Local Mean', 'Local Variance', 'Local Std Dev'],
                                         cmaps='jet')

        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(image_gray, distances=distances, angles=angles, 
                                           levels=256, symmetric=True, normed=True)
        glcm_data = {
            'Property': ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'],
            'Value': [graycoprops(glcm, prop)[0, 0] for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]
        }
        glcm_df = pd.DataFrame(glcm_data)
        fig_glcm = plt.figure(figsize=(8, 7))
        plt.imshow(glcm[:, :, 0, 0], cmap='viridis')
        plt.colorbar(label='Frequency')
        plt.title('GLCM Matrix (distance=1, angle=0)')
        plt.tight_layout()

        lbp_image, lbp_hist = self._compute_lbp(image_gray)
        fig_lbp = plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.imshow(lbp_image, cmap='jet')
        plt.title('LBP Texture Map')
        plt.axis('off')
        plt.subplot(122)
        plt.bar(range(len(lbp_hist)), lbp_hist)
        plt.title('LBP Histogram')
        plt.xlabel('LBP Value')
        plt.ylabel('Frequency')
        plt.tight_layout()

        frequencies = [5, 10, 15]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        gabor_images, gabor_titles, _ = self._compute_gabor_filters(image_gray, frequencies, orientations)
        fig_gabor = self._display_multiple(gabor_images[:6] + [gabor_images[-1]], 
                                          gabor_titles[:6] + ['Gabor Magnitude'], 
                                          cmaps='jet')

        _, selected_law_images, selected_law_titles = self._compute_law_texture_energy(image_gray)
        fig_law = self._display_multiple(selected_law_images, selected_law_titles, cmaps='jet')

        return fig_orig, fig_stat, fig_glcm, glcm_df, fig_lbp, fig_gabor, fig_law