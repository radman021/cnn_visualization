import streamlit as st
from enumerations.models import Models
from predict import predict
from logger import Logger
from PIL import Image

st.set_page_config(page_title="CNN Visualizer", layout="wide")

st.title("CNN Layers Visualizer")

model_options = [m["model_id"] for m in Models.get_all_models()]

logger = Logger("cli").get_logger()
logger.info(f"Available models: {model_options}")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
selected_model = st.selectbox("Select Model", options=model_options)

if uploaded_file is not None and selected_model:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        result = predict(image, selected_model)

    st.success("Result:")
    st.write(result)
