import streamlit as st
import torch

from enumerations.models import Models

from maps.model_paths import ModelPaths

from filters_visualizer import FiltersVisualizer
from folder_tree_builder import build_folder_tree


st.set_page_config(page_title="Filters Visualizer", layout="wide")
st.title("Filters Visualizer")

model_options = [m["model_id"] for m in Models.get_all_models()]

selected_model = st.selectbox("Select Model", options=model_options)


if selected_model:
    folder_tree = build_folder_tree(
        f"storage/{Models[selected_model].value}/filters_weight_viz"
    )
else:
    folder_tree = None

if not folder_tree:
    if st.button("Visualize filters") and selected_model:
        st.info("Extracting and saving all filters...")

        model_enum = Models[selected_model]
        paths = ModelPaths.get_model_paths()
        model = torch.load(paths[model_enum], map_location=torch.device("cpu"))

        base_dir = f"storage/{model_enum.value}/filters_weight_viz"

        fv = FiltersVisualizer(model)

        with st.spinner(f"Saving filter weights for model {model_enum.value}..."):
            fv.save_all_filters_weights(base_dir=base_dir)

        st.success(f"All filters saved in {base_dir}.")

        st.rerun()
else:
    blocks = list(folder_tree.keys())
    selected_block = st.selectbox("Choose a block", options=blocks)

    if selected_block:
        subfolders = list(folder_tree[selected_block].keys())
        selected_subfolder = st.selectbox("Choose a layer", options=subfolders)

        if selected_subfolder:
            filter_files = folder_tree[selected_block][selected_subfolder]
            filter_names = [f.name for f in filter_files]
            selected_filter = st.selectbox(
                "Choose a filter from layer", options=filter_names
            )

            if selected_filter:
                img_path = next(f for f in filter_files if f.name == selected_filter)
                import matplotlib.image as mpimg

                img = mpimg.imread(img_path)
                st.image(
                    img,
                    caption=f"{selected_block}/{selected_subfolder}/{selected_filter}",
                )
