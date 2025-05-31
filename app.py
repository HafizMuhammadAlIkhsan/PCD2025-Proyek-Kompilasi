import streamlit as st
from modules.ui.image_ui import ImageUI
from modules.processing.image_processor import ImageProcessor

class StreamlitApp:
    def __init__(self):
        self.features = {
            "Image Upload & RGB Split": self.run_image_rgb_split,
            "Arithmetic Operations": self.run_arithmetic_operations,
            "Logic Operations": self.run_logic_operations,
            "Grayscale Conversion": self.run_grayscale_conversion,
            "Histogram Generation": self.run_histogram_generation,
            "Histogram Equalization": self.run_histogram_equalization,
            "Histogram Specification": self.run_histogram_specification,
            "Image Statistics": self.run_image_statistics
        }
        self.image_ui = ImageUI()
        self.image_processor = ImageProcessor()

    def run_image_rgb_split(self):
        self.image_ui.display_image_upload(self.image_processor)

    def run_arithmetic_operations(self):
        self.image_ui.display_arithmetic_operations(self.image_processor)

    def run_logic_operations(self):
        self.image_ui.display_logic_operations(self.image_processor)

    def run_grayscale_conversion(self):
        self.image_ui.display_grayscale_conversion(self.image_processor)

    def run_histogram_generation(self):
        self.image_ui.display_histogram_generation(self.image_processor)

    def run_histogram_equalization(self):
        self.image_ui.display_histogram_equalization(self.image_processor)

    def run_histogram_specification(self):
        self.image_ui.display_histogram_specification(self.image_processor)

    def run_image_statistics(self):
        self.image_ui.display_image_statistics(self.image_processor)

    def run(self):
        st.sidebar.title("Navigation")
        feature = st.sidebar.selectbox("Select Feature", list(self.features.keys()))
        self.features[feature]()

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()