import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image
from scipy import integrate
from skimage import color

# Function to load the image using PIL
def load_image(file_path):
    return Image.open(file_path)

# Function to crop the image based on the bounding box
def crop_image(image, bbox):
    x1, y1, width, height = bbox
    return image.crop((x1, y1, x1 + width, y1 + height))

# Function to detect the black curve in the ECG
def detect_black_curve(image):
    print("Detecting black curve...")
    image = np.array(image)

    # Convert RGBA to RGB if necessary
    if image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]  # Discard the alpha channel

    if image.ndim == 3:
        # Convert to HSV color space
        hsv = color.rgb2hsv(image)
        # Create a mask for black color
        black_mask = (hsv[:, :, 2] < 0.2)  # Value channel < 0.2
    else:
        # If it's already grayscale, use a simple threshold
        black_mask = image < 0.2

    # Find the lowest black pixel for each column
    y_positions = np.argmax(black_mask[::-1], axis=0)
    return black_mask.shape[0] - y_positions

# Function to calculate the area under the curve using Simpson's rule
def calculate_area(x, y):
    return integrate.simps(y, x)

# Streamlit Web App Interface
st.title('Interactive ECG Curve Analyzer')

# File uploader for ECG image
uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded ECG Image', use_column_width=True)

    # Allow user to draw a bounding box on the image
    with st.expander("Select Region to Analyze"):
        x1 = st.number_input("X1", min_value=0, value=0)
        y1 = st.number_input("Y1", min_value=0, value=0)
        width = st.number_input("Width", min_value=1, value=100)
        height = st.number_input("Height", min_value=1, value=100)

        selected_region = (x1, y1, width, height)
        cropped_image = crop_image(image, selected_region)

        st.image(cropped_image, caption='Cropped ECG Region', use_column_width=True)

        # Analyze the cropped ECG curve
        y_positions = detect_black_curve(cropped_image)
        x = np.arange(len(y_positions))

        # Plot both normal and inverted ECG curves
        fig_curve = px.line(x=x, y=y_positions, labels={'x': 'Sample', 'y': 'Amplitude'}, title='Selected ECG Curve')
        st.plotly_chart(fig_curve)

        # Calculate both normal and inverted areas
        normal_area = calculate_area(x, y_positions)
        inverted_y_positions = np.max(y_positions) - y_positions
        inverted_area = calculate_area(x, inverted_y_positions)

        st.write(f'Calculated area under the normal curve: {normal_area:.2f}')
        st.write(f'Calculated area under the inverted curve: {inverted_area:.2f}')

# Footer information
st.markdown("---")
st.write("This application allows you to upload an ECG image, select a region, and analyze the curve.")
