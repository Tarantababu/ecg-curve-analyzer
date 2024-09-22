import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image
from skimage import color

# Function to load image using PIL
def load_image(file_path):
    return Image.open(file_path)

# Function to detect black curve in the image
def detect_black_curve(image):
    image = np.array(image)
    
    if image.ndim == 3 and image.shape[2] == 4:  # Check if the image has 4 channels (RGBA)
        image = Image.fromarray(image).convert('RGB')  # Convert RGBA to RGB
        image = np.array(image)  # Convert back to numpy array

    if image.ndim == 3:
        # Convert to HSV color space
        hsv = color.rgb2hsv(image)
        # Create a mask for black color
        black_mask = (hsv[:, :, 2] < 0.2)  # Value channel < 0.2 for detecting dark/black pixels
        # Detect y-positions of the curve (the first black pixel in each column)
        y_positions = np.argmax(black_mask, axis=0)
        return y_positions
    else:
        st.error("Invalid image format! Image must be RGB.")
        return None

# Function to calculate the area under the curve using trapezoidal rule
def calculate_area(x, y):
    return np.trapz(y, x)

# Streamlit Web App Interface
st.title('Interactive ECG Curve Analyzer')

# Upload image section
uploaded_image = st.file_uploader("Upload an ECG Image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Load and display the image
    image = load_image(uploaded_image)
    st.image(image, caption='Uploaded ECG Image', use_column_width=True)

    # Cropped image (for demonstration, let's assume we crop it as necessary)
    cropped_image = image.crop((50, 50, 150, 150))  # Example cropping, adjust as necessary
    st.image(cropped_image, caption='Cropped ECG Region', use_column_width=True)

    # Analyze the cropped ECG curve
    y_positions = detect_black_curve(cropped_image)
    if y_positions is not None:
        x = np.arange(len(y_positions))

        # Plot the normal ECG curve
        fig_curve = px.line(x=x, y=y_positions, title="Selected ECG Curve")
        st.plotly_chart(fig_curve)

        # Calculate the area under the normal and inverted curves
        normal_area = calculate_area(x, y_positions)
        inverted_y_positions = np.max(y_positions) - y_positions
        inverted_area = calculate_area(x, inverted_y_positions)

        # Plot the inverted ECG curve
        fig_inverted_curve = px.line(x=x, y=inverted_y_positions, title="Inverted ECG Curve")
        st.plotly_chart(fig_inverted_curve)

        # Display the calculated areas
        st.write(f"Calculated area under the normal curve: {normal_area}")
        st.write(f"Calculated area under the inverted curve: {inverted_area}")
