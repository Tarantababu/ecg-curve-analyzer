import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from scipy import integrate

@st.cache_data
def load_image(file):
    with Image.open(file) as img:
        return np.array(img.convert('L'))  # Convert to grayscale

def detect_black_curve(image):
    threshold = np.mean(image) * 0.7
    binary = image < threshold
    y_positions = np.argmax(binary[::-1], axis=0)
    return binary.shape[0] - y_positions

def calculate_area(x, y):
    return integrate.simpson(y=y, x=x)

st.set_page_config(layout="wide")
st.title('Interactive ECG Curve Analyzer')

uploaded_file = st.file_uploader("Choose an ECG image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    
    # Create an interactive image with Plotly
    fig = go.Figure(go.Image(z=image))
    
    fig.update_layout(
        title="Original ECG Image (Hover to see coordinates)",
        width=800,
        height=600,
    )

    # Add hover information
    fig.update_traces(
        hoverinfo="x+y",
        hovertemplate="X: %{x}<br>Y: %{y}"
    )

    # Display the interactive image
    st.plotly_chart(fig)

    col1, col2 = st.columns(2)

    with col1:
        x1 = st.number_input("X coordinate", value=100, min_value=0, max_value=image.shape[1]-1)
        y1 = st.number_input("Y coordinate", value=100, min_value=0, max_value=image.shape[0]-1)

    with col2:
        width = st.number_input("Width", value=200, min_value=1, max_value=image.shape[1]-x1)
        height = st.number_input("Height", value=150, min_value=1, max_value=image.shape[0]-y1)

    if st.button('Analyze ECG Curve'):
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

                normal_area = calculate_area(x, y_positions)
                inverted_area = calculate_area(x, inverted_y_positions)

                fig = go.Figure()
                
                # Original curve
                fig.add_trace(go.Scatter(x=x, y=y_positions, 
                                         fill='tozeroy', 
                                         fillcolor='rgba(0, 0, 255, 0.3)', 
                                         line=dict(color='blue'),
                                         name=f'Normal Curve (Area: {normal_area:.2f})'))
                
                # Inverted curve
                fig.add_trace(go.Scatter(x=x, y=inverted_y_positions, 
                                         fill='tozeroy', 
                                         fillcolor='rgba(255, 0, 0, 0.3)', 
                                         line=dict(color='red'),
                                         name=f'Inverted Curve (Area: {inverted_area:.2f})'))

                fig.update_layout(
                    title='ECG Curves with Area Calculation',
                    xaxis_title='Sample',
                    yaxis_title='Amplitude',
                    legend=dict(x=0, y=1, traceorder='normal'),
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

                st.write(f'Area under the normal curve: {normal_area:.2f}')
                st.write(f'Area under the inverted curve: {inverted_area:.2f}')
