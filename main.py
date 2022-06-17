from re import M
import streamlit as st
from usimage import ImageSegmenter


model = ImageSegmenter()
st.write("# Online Ultrasound Segmentation")
uploaded_file = st.file_uploader("Upload your image", type='png')
model = ImageSegmenter()

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write("Input image:")
    st.image(bytes_data)
    if st.button('Submit'):
        segments = model(bytes_data).numpy()
        st.write('Segmentation result')
        st.image(segments)