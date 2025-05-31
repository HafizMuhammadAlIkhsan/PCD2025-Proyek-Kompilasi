import streamlit as st
from modules.ui.image_ui import ImageUI
from modules.processing.rgb_processor import RGBProcessor
from modules.processing.advanced_image_processor import AdvancedImageProcessor
from modules.processing.frequency_processor import FrequencyProcessor

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
            "Task 3: Convolution & Frequency Operations": self.run_task3_frequency_operations
        }
        self.image_ui = ImageUI()
        self.rgb_processor = RGBProcessor()
        self.advanced_processor = AdvancedImageProcessor()
        self.frequency_processor = FrequencyProcessor()

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

    def run(self):
        st.sidebar.title("Navigation")
        feature = st.sidebar.selectbox("Select Task", list(self.features.keys()))
        self.features[feature]()

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()