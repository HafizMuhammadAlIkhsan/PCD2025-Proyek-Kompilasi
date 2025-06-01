import streamlit as st
import numpy as np
import cv2
import os
from io import BytesIO
from PIL import Image
import time

class ImageUI:
    def display_task1_rgb_split(self, rgb_processor):
        st.header("Task 1: Image Upload & RGB Split")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="task1_rgb_upload")
        if uploaded_file is not None:
            try:
                image_data = uploaded_file.read()
                rgb_array = rgb_processor.process_rgb_split(image_data)
                st.image(image_data, caption="Uploaded Image", use_column_width=True)
                st.write("RGB Channel Arrays:")
                st.json(rgb_array)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_task2_arithmetic_operations(self, advanced_processor):
        st.header("Task 2: Arithmetic Operations")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="task2_arithmetic_upload")
        operation = st.selectbox("Select Operation", ["add", "subtract", "max", "min", "inverse"])
        value = st.slider("Value (for add/subtract/max/min)", 0, 255, 50) if operation != "inverse" else None
        if uploaded_file is not None:
            try:
                image_data = uploaded_file.read()
                result_img = advanced_processor.process_arithmetic_operation(image_data, operation, value)
                st.image(image_data, caption="Original Image", use_column_width=True)
                st.image(result_img, caption="Modified Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_task2_logic_operations(self, advanced_processor):
        st.header("Task 2: Logic Operations")
        uploaded_file1 = st.file_uploader("Upload first image", type=["jpg", "jpeg", "png"], key="task2_logic_upload1")
        operation = st.selectbox("Select Operation", ["not", "and", "xor"])
        uploaded_file2 = st.file_uploader("Upload second image (required for AND/XOR)", type=["jpg", "jpeg", "png"], key="task2_logic_upload2") if operation in ["and", "xor"] else None
        if uploaded_file1 is not None and (operation == "not" or (operation in ["and", "xor"] and uploaded_file2 is not None)):
            try:
                image_data1 = uploaded_file1.read()
                image_data2 = uploaded_file2.read() if uploaded_file2 else None
                result_img = advanced_processor.process_logic_operation(image_data1, image_data2, operation)
                st.image(image_data1, caption="First Image", use_column_width=True)
                if image_data2:
                    st.image(image_data2, caption="Second Image", use_column_width=True)
                st.image(result_img, caption="Result Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_task2_grayscale_conversion(self, advanced_processor):
        st.header("Task 2: Grayscale Conversion")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="task2_grayscale_upload")
        if uploaded_file is not None:
            try:
                image_data = uploaded_file.read()
                gray_img = advanced_processor.process_grayscale_conversion(image_data)
                st.image(image_data, caption="Original Image", use_column_width=True)
                st.image(gray_img, caption="Grayscale Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_task2_histogram_generation(self, advanced_processor):
        st.header("Task 2: Histogram Generation")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="task2_histogram_upload")
        if uploaded_file is not None:
            try:
                image_data = uploaded_file.read()
                grayscale_hist, color_hist = advanced_processor.process_histogram_generation(image_data)
                st.image(image_data, caption="Uploaded Image", use_column_width=True)
                st.image(grayscale_hist, caption="Grayscale Histogram", use_column_width=True)
                st.image(color_hist, caption="Color Histogram", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_task2_histogram_equalization(self, advanced_processor):
        st.header("Task 2: Histogram Equalization")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="task2_equalize_upload")
        if uploaded_file is not None:
            try:
                image_data = uploaded_file.read()
                equalized_img = advanced_processor.process_histogram_equalization(image_data)
                st.image(image_data, caption="Original Image", use_column_width=True)
                st.image(equalized_img, caption="Equalized Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_task2_histogram_specification(self, advanced_processor):
        st.header("Task 2: Histogram Specification")
        uploaded_file = st.file_uploader("Upload source image", type=["jpg", "jpeg", "png"], key="task2_specify_upload")
        ref_file = st.file_uploader("Upload reference image", type=["jpg", "jpeg", "png"], key="task2_specify_ref_upload")
        if uploaded_file is not None and ref_file is not None:
            try:
                image_data = uploaded_file.read()
                ref_image_data = ref_file.read()
                specified_img = advanced_processor.process_histogram_specification(image_data, ref_image_data)
                st.image(image_data, caption="Source Image", use_column_width=True)
                st.image(ref_image_data, caption="Reference Image", use_column_width=True)
                st.image(specified_img, caption="Specified Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_task2_image_statistics(self, advanced_processor):
        st.header("Task 2: Image Statistics")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="task2_statistics_upload")
        if uploaded_file is not None:
            try:
                image_data = uploaded_file.read()
                mean, std_dev = advanced_processor.process_image_statistics(image_data)
                st.image(image_data, caption="Uploaded Image", use_column_width=True)
                st.write(f"Mean Intensity: {mean:.2f}")
                st.write(f"Standard Deviation: {std_dev:.2f}")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_task3_frequency_operations(self, frequency_processor):
        st.header("Task 3: Convolution & Frequency Operations")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="task3_frequency_upload")
        operation = st.selectbox("Select Operation", ["convolution", "padding", "filter", "fourier", "noise_reduction"])
        param = None
        if operation == "convolution":
            param = st.selectbox("Select Kernel Type", ["average", "sharpen", "edge"], key="task3_convolution_kernel")
        elif operation == "padding":
            param = st.number_input("Padding Size", min_value=1, max_value=100, value=10, step=1, key="task3_padding_size")
        elif operation == "filter":
            param = st.selectbox("Select Filter Type", ["low", "high", "band"], key="task3_filter_type")
        if uploaded_file is not None and st.button("Process", key="task3_process_button"):
            try:
                image_data = uploaded_file.read()
                processed_img = frequency_processor.process_frequency_operation(image_data, operation, param)
                st.image(image_data, caption="Original Image", use_column_width=True)
                st.image(processed_img, caption="Processed Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    def display_task4_add_faces(self, face_processor):
        st.header("Tambah Wajah Baru ke Dataset")
        new_person = st.text_input("Masukkan nama orang baru:", key="task4_new_person")
        if st.button("Tambahkan Wajah Baru", key="task4_capture_button"):
            if not new_person:
                st.warning("Silakan masukkan nama orang terlebih dahulu.")
            else:
                try:
                    with st.spinner("Mengaktifkan kamera, mohon tunggu..."):
                        result = face_processor.capture_faces(new_person)
                    st.success(f"{result['count']} gambar telah ditambahkan ke dataset '{new_person}'.")

                    st.markdown("### Preview Gambar yang Ditambahkan:")
                    cols = st.columns(len(result['images']))
                    for i, img_path in enumerate(result['images']):
                        with cols[i]:
                            st.image(img_path, caption=f"Gambar {i+1}", width=200)

                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error capturing faces: {str(e)}")

    def display_task4_process_dataset(self, face_processor):
        st.header("Proses Gambar dari Dataset")
        dataset_names = face_processor.get_dataset_names()

        if not dataset_names:
            st.info("Belum ada data di folder dataset.")
            return

        selected_person = st.selectbox("Pilih nama orang:", dataset_names, key="task4_select_person")
        process_option = st.selectbox(
            "Pilih opsi pemrosesan:",
            ["Tambah Noise Salt and Pepper", "Hilangkan Noise", "Tajamkan Gambar"],
            key="task4_process_option"
        )

        if st.button("Proses Semua Gambar", key="task4_process_button"):
            try:
                with st.spinner("Memproses semua gambar..."):
                    processed_images = face_processor.process_dataset_images(selected_person, process_option)
                st.success(f"Semua gambar {selected_person} telah diproses ({process_option}) dan disimpan.")

                st.markdown("### Preview Gambar Hasil:")
                cols = st.columns(len(processed_images))
                for i, img_path in enumerate(processed_images):
                    with cols[i]:
                        st.image(img_path, caption=f"Hasil {i+1}", width=200)

            except Exception as e:
                st.error(f"Error processing dataset: {str(e)}")

    def display_task5_freeman_chain_code(self, shape_processor):
        st.header("Task 5: Freeman Chain Code")
        uploaded_file = st.file_uploader("Pilih citra (PNG/JPG)", type=["png", "jpg", "jpeg"], key="task5_freeman_upload")
        threshold_value = st.slider("Nilai Threshold Binarisasi", 0, 255, 127, key="task5_freeman_threshold")
        if uploaded_file is not None:
            with st.spinner("Memproses Kode Rantai Freeman..."):
                try:
                    fig, chain_code_str = shape_processor.process_freeman_chain_code(uploaded_file, threshold_value)
                    if fig is not None:
                        st.pyplot(fig)
                        st.subheader("Detail Kode Rantai")
                        st.text(chain_code_str)
                    else:
                        st.error(chain_code_str)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        else:
            st.info("Silakan unggah citra untuk analisis Freeman Chain Code.")

    def display_task5_canny_edge_detection(self, shape_processor):
        st.header("Task 5: Canny Edge Detection")
        uploaded_file = st.file_uploader("Pilih citra (PNG/JPG)", type=["png", "jpg", "jpeg"], key="task5_canny_upload")
        low_threshold = st.slider("Low Threshold", 0, 255, 50, key="task5_canny_low")
        high_threshold = st.slider("High Threshold", 0, 255, 150, key="task5_canny_high")
        if uploaded_file is not None:
            with st.spinner("Memproses Deteksi Tepi Canny..."):
                try:
                    fig, error_msg = shape_processor.process_canny_edge(uploaded_file, low_threshold, high_threshold)
                    if fig is not None:
                        st.pyplot(fig)
                    else:
                        st.error(error_msg)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        else:
            st.info("Silakan unggah citra untuk analisis Canny Edge Detection.")

    def display_task5_integral_projection_default(self, shape_processor):
        st.header("Task 5: Integral Projection (Default)")
        use_default_image = st.checkbox("Gunakan Citra Teks Default", value=True, key="task5_default_image")
        uploaded_file = None
        if not use_default_image:
            uploaded_file = st.file_uploader("Pilih citra (PNG/JPG)", type=["png", "jpg", "jpeg"], key="task5_integral_default_upload")
        if use_default_image or uploaded_file is not None:
            with st.spinner("Memproses Proyeksi Integral..."):
                try:
                    fig, error_msg = shape_processor.process_integral_projection_default(uploaded_file)
                    if fig is not None:
                        st.pyplot(fig)
                    else:
                        st.error(error_msg)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        else:
            st.info("Silakan unggah citra atau centang 'Gunakan Citra Teks Default' untuk analisis.")

    def display_task5_integral_projection_otsu(self, shape_processor):
        st.header("Task 5: Integral Projection (Otsu)")
        uploaded_file = st.file_uploader("Pilih citra (PNG/JPG)", type=["png", "jpg", "jpeg"], key="task5_integral_otsu_upload")
        if uploaded_file is not None:
            with st.spinner("Memproses Proyeksi Integral dengan Otsu..."):
                try:
                    fig, error_msg = shape_processor.process_integral_projection_otsu(uploaded_file)
                    if fig is not None:
                        st.pyplot(fig)
                    else:
                        st.error(error_msg)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        else:
            st.info("Silakan unggah citra untuk analisis Proyeksi Integral dengan Otsu.")