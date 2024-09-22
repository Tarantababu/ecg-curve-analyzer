import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image
from scipy import integrate
import matplotlib.pyplot as plt
import time
import io

def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        st.write(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@st.cache_data
def load_image(file):
    with Image.open(file) as img:
        return np.array(img.convert('L'))  # Convert to grayscale

@log_time
def detect_black_curve(image):
    threshold = np.mean(image) * 0.7
    binary = image < threshold
    y_positions = np.argmax(binary[::-1], axis=0)
    return binary.shape[0] - y_positions

@log_time
def calculate_area(x, y):
    return integrate.simpson(y=y, x=x)

st.set_page_config(layout="wide")
st.title('Interactive ECG Curve Analyzer')

uploaded_file = st.file_uploader("Choose an ECG image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded ECG Image', use_column_width=True)

    col1, col2 = st.columns(2)

    with col1:
        x1 = st.number_input("X coordinate", value=100, min_value=0)
        y1 = st.number_input("Y coordinate", value=100, min_value=0)

    with col2:
        width = st.number_input("Width", value=200, min_value=1)
        height = st.number_input("Height", value=150, min_value=1)

    if st.button('Analyze ECG Curve'):
        start_time = time.time()
        
        cropped_image = image[int(y1):int(y1 + height), int(x1):int(x1 + width)]
        
        col1, col2 = st.columns(2)

        with col1:
            st.image(cropped_image, caption='Cropped ECG Region', use_column_width=True)

        with col2:
            y_positions = detect_black_curve(cropped_image)

            if len(y_positions) == 0:
                st.error("No ECG curve detected. Please check the selected region.")
            else:
                x = np.arange(len(y_positions))
                inverted_y_positions = np.max(y_positions) - y_positions

                fig, ax = plt.subplots()
                ax.plot(x, y_positions, 'b-', label='Selected ECG Curve')
                ax.plot(x, inverted_y_positions, 'r-', label='Inverted ECG Curve')
                ax.set_xlabel('Sample')
                ax.set_ylabel('Amplitude')
                ax.set_title('ECG Curves')
                ax.legend()
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                st.image(buf, caption='ECG Curves', use_column_width=True)
                plt.close(fig)

                normal_area = calculate_area(x, y_positions)
                inverted_area = calculate_area(x, inverted_y_positions)

                st.write(f'Area under the normal curve: {normal_area:.2f}')
                st.write(f'Area under the inverted curve: {inverted_area:.2f}')

        end_time = time.time()
        st.write(f"Total analysis time: {end_time - start_time:.2f} seconds")

st.write("App loaded successfully")
