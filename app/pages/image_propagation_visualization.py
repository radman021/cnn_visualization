import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.image as mpimg
from pathlib import Path

from enumerations.models import Models
from maps.model_paths import ModelPaths
from layer_output_visualizer import LayerOutputVisualizer
from folder_tree_builder import build_folder_tree


st.set_page_config(page_title="Image Propagation Visualizer", layout="wide")
st.title("Image Propagation Visualizer")

model_options = [m["model_id"] for m in Models.get_all_models()]
selected_model = st.selectbox("Select Model", options=model_options)

if selected_model:
    model_enum = Models[selected_model]
    base_storage = Path(f"storage/{model_enum.value}/image_propagation")
else:
    base_storage = None

if selected_model:
    if base_storage.exists():
        image_run_folders = [d for d in base_storage.iterdir() if d.is_dir()]
    else:
        image_run_folders = []

    run_names = [f.name for f in image_run_folders]
    selected_run = st.selectbox("Choose a saved image run", options=run_names)
else:
    selected_run = None

if selected_run:
    run_path = base_storage / selected_run
    folder_tree = build_folder_tree(run_path)

    if folder_tree:
        blocks = list(folder_tree.keys())
        selected_block = st.selectbox("Choose a block", options=blocks)

        if selected_block:
            subfolders = list(folder_tree[selected_block].keys())
            selected_subfolder = st.selectbox("Choose a layer", options=subfolders)

            if selected_subfolder:
                feature_files = folder_tree[selected_block][selected_subfolder]
                feature_names = [f.name for f in feature_files]
                selected_feature = st.selectbox(
                    "Choose a feature map channel", options=feature_names
                )

                if selected_feature:
                    img_path = next(
                        f for f in feature_files if f.name == selected_feature
                    )
                    img = mpimg.imread(img_path)
                    st.image(
                        img,
                        caption=f"{selected_run}/{selected_block}/{selected_subfolder}/{selected_feature}",
                    )

st.markdown("---")
st.markdown("### Run a new image through the model")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None and selected_model:
    if st.button("Run and save feature maps"):
        paths = ModelPaths.get_model_paths()
        model = torch.load(paths[model_enum], map_location=torch.device("cpu"))
        model.eval()

        pil_image = Image.open(uploaded_file).convert("RGB")
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        input_tensor = transform(pil_image).unsqueeze(0)

        layer_output_vis = LayerOutputVisualizer(model)
        layer_output_vis.register_hooks()

        image_name = Path(uploaded_file.name).stem

        with st.spinner("Visualizing outputs by layers..."):
            saved_dir = layer_output_vis.save_layer_outputs(
                input_tensor,
                base_storage=base_storage,
                image_name=image_name,
            )
        st.success(f"Feature maps saved in {saved_dir}.")
        st.rerun()
