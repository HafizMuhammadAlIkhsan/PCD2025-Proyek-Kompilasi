import streamlit as st
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

class ImageUI:
    def display_image_upload(self, image_processor):
        st.header("RGB Split")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="rgb_upload")
        if uploaded_file is not None:
            try:
                image_data = uploaded_file.read()
                rgb_array = image_processor.process_image(image_data)
                st.image(image_data, caption="Uploaded Image", use_container_width=True)
                st.write("RGB Channel Arrays:")
                st.json(rgb_array)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_arithmetic_operations(self, image_processor):
        st.header("Arithmetic Operations")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="arithmetic_upload")
        operation = st.selectbox("Select Operation", ["add", "subtract", "max", "min", "inverse"])
        value = st.slider("Value (for add/subtract/max/min)", 0, 255, 50) if operation != "inverse" else None
        if uploaded_file is not None:
            try:
                image_data = uploaded_file.read()
                result_img = image_processor.process_arithmetic_operation(image_data, operation, value)
                st.image(image_data, caption="Original Image", use_container_width=True)
                st.image(result_img, caption="Modified Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_logic_operations(self, image_processor):
        st.header("Logic Operations")
        uploaded_file1 = st.file_uploader("Upload first image", type=["jpg", "jpeg", "png"], key="logic_upload1")
        operation = st.selectbox("Select Operation", ["not", "and", "xor"])
        uploaded_file2 = st.file_uploader("Upload second image (required for AND/XOR)", type=["jpg", "jpeg", "png"], key="logic_upload2") if operation in ["and", "xor"] else None

        if uploaded_file1 is not None and (operation == "not" or (operation in ["and", "xor"] and uploaded_file2 is not None)):
            try:
                image_data1 = uploaded_file1.read()
                image_data2 = uploaded_file2.read() if uploaded_file2 else None
                np_img1 = cv2.imdecode(np.frombuffer(image_data1, np.uint8), cv2.IMREAD_COLOR)
                np_img2 = cv2.imdecode(np.frombuffer(image_data2, np.uint8), cv2.IMREAD_COLOR) if image_data2 else None
                if operation in ["and", "xor"]:
                    if np_img1.shape != np_img2.shape or np_img1.dtype != np_img2.dtype:
                        st.error("Error: Both images must have the same size and type for logic operations like AND/XOR.")
                        return
                result_img = image_processor.process_logic_operation(image_data1, image_data2, operation)
                st.image(image_data1, caption="First Image", use_container_width=True)
                if image_data2:
                    st.image(image_data2, caption="Second Image", use_container_width=True)
                st.image(result_img, caption="Result Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_grayscale_conversion(self, image_processor):
        st.header("Grayscale Conversion")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="grayscale_upload")
        if uploaded_file is not None:
            try:
                image_data = uploaded_file.read()
                gray_img = image_processor.process_grayscale_conversion(image_data)
                st.image(image_data, caption="Original Image", use_container_width=True)
                st.image(gray_img, caption="Grayscale Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_histogram_generation(self, image_processor):
        st.header("Histogram Generation")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="histogram_upload")
        if uploaded_file is not None:
            try:
                image_data = uploaded_file.read()
                grayscale_hist, color_hist = image_processor.process_histogram_generation(image_data)
                st.image(image_data, caption="Uploaded Image", use_container_width=True)
                st.image(grayscale_hist, caption="Grayscale Histogram", use_container_width=True)
                st.image(color_hist, caption="Color Histogram", use_container_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_histogram_equalization(self, image_processor):
        st.header("Histogram Equalization")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="equalize_upload")
        if uploaded_file is not None:
            try:
                image_data = uploaded_file.read()
                equalized_img = image_processor.process_histogram_equalization(image_data)
                st.image(image_data, caption="Original Image", use_container_width=True)
                st.image(equalized_img, caption="Equalized Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_histogram_specification(self, image_processor):
        st.header("Histogram Specification")
        uploaded_file = st.file_uploader("Upload source image", type=["jpg", "jpeg", "png"], key="specify_upload")
        ref_file = st.file_uploader("Upload reference image", type=["jpg", "jpeg", "png"], key="specify_ref_upload")
        if uploaded_file is not None and ref_file is not None:
            try:
                image_data = uploaded_file.read()
                ref_image_data = ref_file.read()
                specified_img = image_processor.process_histogram_specification(image_data, ref_image_data)
                st.image(image_data, caption="Source Image", use_container_width=True)
                st.image(ref_image_data, caption="Reference Image", use_container_width=True)
                st.image(specified_img, caption="Specified Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_image_statistics(self, image_processor):
        st.header("Image Statistics")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="statistics_upload")
        if uploaded_file is not None:
            try:
                image_data = uploaded_file.read()
                mean, std_dev = image_processor.process_image_statistics(image_data)
                st.image(image_data, caption="Uploaded Image", use_container_width=True)
                st.write(f"Mean Intensity: {mean:.2f}")
                st.write(f"Standard Deviation: {std_dev:.2f}")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")