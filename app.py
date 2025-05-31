import streamlit as st
from modules.ui.image_ui import ImageUI
from modules.processing.image_processor import ImageProcessor

class StreamlitApp:
    def __init__(self):
        self.features = {
            "Image Upload & RGB Split": self.run_image_feature
        }
        self.image_ui = ImageUI()
        self.image_processor = ImageProcessor()

    def run_image_feature(self):
        self.image_ui.display_image_upload(self.image_processor)

    def run(self):
        st.sidebar.title("Navigation")
        feature = st.sidebar.selectbox("Select Feature", list(self.features.keys()))
        self.features[feature]()

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()