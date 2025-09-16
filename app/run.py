import streamlit as st

st.set_page_config(page_title="Resnet Visualizer", layout="wide")
st.title("ResNet Visualizer")

st.write(
    """
    This application provides an interactive environment for exploring ResNet strcture.

    From the sidebar, you can navigate to the following modules:

    - **Visualizing Output by Layers**: inspect the intermediate feature maps produced by each convolutional layer for a given input image.
    - **Visualizing Filter Weights**: view convolutional filters at each layer of the network.
    - **Filters as Linear Combinations**: examine how higher-level filters can be expressed as linear combinations of lower-level filters
    """
)
