import streamlit as st
import numpy as np

class ImageUI:
    def display_image_upload(self, image_processor):
        st.header("Image Upload & RGB Split")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image_data = uploaded_file.read()
                rgb_array = image_processor.process_image(image_data)
                
                st.image(image_data, caption="Uploaded Image", use_container_width=True)
                st.write("RGB Channel Arrays:")
                st.json(rgb_array)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")