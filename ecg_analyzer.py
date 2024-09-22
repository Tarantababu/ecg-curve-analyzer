import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image, ImageDraw
from scipy import integrate
import matplotlib.pyplot as plt

@st.cache_data
def load_image(file):
    image = Image.open(file)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return np.array(image)

def crop_image(image, bbox):
    x1, y1, width, height = bbox
    return image[int(y1):int(y1 + height), int(x1):int(x1 + width)]

def draw_bounding_box(image, bbox):
    img_with_box = Image.fromarray(image)
    draw = ImageDraw.Draw(img_with_box)
    x1, y1, width, height = bbox
    x2, y2 = x1 + width, y1 + height
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    return img_with_box

def detect_black_curve(image):
    if len(image.shape) == 3:
        # Convert to grayscale
        gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        gray = image
    
    # Threshold to create a binary image
    threshold = np.mean(gray) * 0.7
    binary = gray < threshold
    
    # Find the lowest black pixel in each column
    y_positions = np.argmax(binary[::-1], axis=0)
    return binary.shape[0] - y_positions

def calculate_area(x, y):
    return integrate.simpson(y, x)

st.title('Interactive ECG Curve Analyzer')

uploaded_file = st.file_uploader("Choose an ECG image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded ECG Image', use_column_width=True)

    fig = px.imshow(image)
    fig.update_layout(dragmode='drawrect', title="Draw a rectangle to select the ECG region for analysis")
    selected_region = st.plotly_chart(fig)

    bbox_input = st.text_input("Enter selected region's bounding box as x, y, width, height", "100, 100, 200, 150")

    if st.button('Analyze ECG Curve'):
        try:
            x1, y1, width, height = map(int, bbox_input.split(","))
            
            if width <= 0 or height <= 0 or x1 < 0 or y1 < 0:
                raise ValueError("Invalid bounding box dimensions.")
            
            img_with_box = draw_bounding_box(image, (x1, y1, width, height))
            st.image(img_with_box, caption=f'Bounding Box: (x={x1}, y={y1}, width={width}, height={height})', use_column_width=True)

            cropped_image = crop_image(image, (x1, y1, width, height))

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

                    fig_curve, ax = plt.subplots()
                    ax.plot(x, y_positions, 'b-', label='Selected ECG Curve')
                    ax.plot(x, inverted_y_positions, 'r-', label='Selected ECG Curve (Inverted)')
                    ax.fill_between(x, y_positions, np.max(y_positions), alpha=0.3, color='b')
                    ax.fill_between(x, inverted_y_positions, np.max(inverted_y_positions), alpha=0.3, color='r')
                    ax.set_xlabel('Sample')
                    ax.set_ylabel('Amplitude')
                    ax.set_title('Selected ECG Curves')
                    ax.legend()
                    st.pyplot(fig_curve)

                    normal_area = calculate_area(x, y_positions)
                    inverted_area = calculate_area(x, inverted_y_positions)

                    st.write(f'Calculated area under the normal curve: {normal_area:.2f}')
                    st.write(f'Calculated area under the inverted curve: {inverted_area:.2f}')

        except ValueError as e:
            st.error(f"Invalid bounding box input: {e}")
