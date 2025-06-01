import streamlit as st
from modules.ui.image_ui import ImageUI
from modules.processing.rgb_processor import RGBProcessor
from modules.processing.advanced_image_processor import AdvancedImageProcessor
from modules.processing.frequency_processor import FrequencyProcessor
from modules.processing.face_processor import FaceProcessor
from modules.processing.shape_processor import ShapeProcessor
from modules.processing.compression_processor import CompressionProcessor
from modules.processing.cbir_processor import CbirProcessor

class StreamlitApp:
    def __init__(self):
        self.features = {
            "Task 1: Image Upload & RGB Split": self.run_task1_rgb_split,
            "Task 2: Arithmetic Operations": self.run_task2_arithmetic_operations,
            "Task 2: Logic Operations": self.run_task2_logic_operations,
            "Task 2: Grayscale Conversion": self.run_task2_grayscale_conversion,
            "Task 2: Histogram Generation": self.run_task2_histogram_generation,
            "Task 2: Histogram Equalization": self.run_task2_histogram_equalization,
            "Task 2: Histogram Specification": self.run_task2_histogram_specification,
            "Task 2: Image Statistics": self.run_task2_image_statistics,
            "Task 3: Convolution & Frequency Operations": self.run_task3_frequency_operations,
            "Task 4: Add Faces to Dataset": self.run_task4_add_faces,
            "Task 4: Process Dataset Images": self.run_task4_process_dataset,
            "Task 5: Freeman Chain Code": self.run_task5_freeman_chain_code,
            "Task 5: Canny Edge Detection": self.run_task5_canny_edge_detection,
            "Task 5: Integral Projection (Default)": self.run_task5_integral_projection_default,
            "Task 5: Integral Projection (Otsu)": self.run_task5_integral_projection_otsu,
            "Task 6: JPEG Compression": self.run_task6_jpeg_compression,
            "Task 6: PNG Compression": self.run_task6_png_compression,
            "Task 7: CBIR": self.run_task7_cbir,
            "Task 7: Color Space Analysis": self.run_task7_color_space_analysis,
            "Task 7: Texture Analysis": self.run_task7_texture_analysis
        }
        self.image_ui = ImageUI()
        self.rgb_processor = RGBProcessor()
        self.advanced_processor = AdvancedImageProcessor()
        self.frequency_processor = FrequencyProcessor()
        self.face_processor = FaceProcessor()
        self.shape_processor = ShapeProcessor()
        self.compression_processor = CompressionProcessor()
        self.cbir_processor = CbirProcessor()

    def run_task1_rgb_split(self):
        self.image_ui.display_task1_rgb_split(self.rgb_processor)

    def run_task2_arithmetic_operations(self):
        self.image_ui.display_task2_arithmetic_operations(self.advanced_processor)

    def run_task2_logic_operations(self):
        self.image_ui.display_task2_logic_operations(self.advanced_processor)

    def run_task2_grayscale_conversion(self):
        self.image_ui.display_task2_grayscale_conversion(self.advanced_processor)

    def run_task2_histogram_generation(self):
        self.image_ui.display_task2_histogram_generation(self.advanced_processor)

    def run_task2_histogram_equalization(self):
        self.image_ui.display_task2_histogram_equalization(self.advanced_processor)

    def run_task2_histogram_specification(self):
        self.image_ui.display_task2_histogram_specification(self.advanced_processor)

    def run_task2_image_statistics(self):
        self.image_ui.display_task2_image_statistics(self.advanced_processor)

    def run_task3_frequency_operations(self):
        self.image_ui.display_task3_frequency_operations(self.frequency_processor)

    def run_task4_add_faces(self):
        self.image_ui.display_task4_add_faces(self.face_processor)

    def run_task4_process_dataset(self):
        self.image_ui.display_task4_process_dataset(self.face_processor)

    def run_task5_freeman_chain_code(self):
        self.image_ui.display_task5_freeman_chain_code(self.shape_processor)

    def run_task5_canny_edge_detection(self):
        self.image_ui.display_task5_canny_edge_detection(self.shape_processor)

    def run_task5_integral_projection_default(self):
        self.image_ui.display_task5_integral_projection_default(self.shape_processor)

    def run_task5_integral_projection_otsu(self):
        self.image_ui.display_task5_integral_projection_otsu(self.shape_processor)

    def run_task6_jpeg_compression(self):
        self.image_ui.display_task6_jpeg_compression(self.compression_processor)

    def run_task6_png_compression(self):
        self.image_ui.display_task6_png_compression(self.compression_processor)

    def run_task7_cbir(self):
        self.image_ui.display_task7_cbir(self.cbir_processor)

    def run_task7_color_space_analysis(self):
        self.image_ui.display_task7_color_space_analysis(self.cbir_processor)

    def run_task7_texture_analysis(self):
        self.image_ui.display_task7_texture_analysis(self.cbir_processor)

    def run(self):
        st.sidebar.title("Navigation")
        feature = st.sidebar.selectbox("Select Task", list(self.features.keys()))
        self.features[feature]()

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()