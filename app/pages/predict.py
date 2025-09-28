import streamlit as st
from PIL import Image

from enumerations.models import Models

from predict import predict

st.set_page_config(page_title="Image Propagation Visualizer", layout="wide")
st.title("Image Propagation Visualizer")

model_options = [m["model_id"] for m in Models.get_all_models()]

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
selected_model = st.selectbox("Select Model", options=model_options)

if uploaded_file is not None and selected_model:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running prediction..."):
        result = predict(image, selected_model)

    st.success("Result:")
    st.write(result)
