import gradio as gr
from models_manager import ModelsManager


def greet(name):
    return f"Hello {name}!"


if __name__ == "__main__":

    ModelsManager.init_all_models()

    ui = gr.Interface(
        fn=greet,
        inputs="text",
        outputs="text",
    )

    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
    )
