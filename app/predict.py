from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from torchvision.models import ResNet18_Weights

from enumerations.models import Models
from maps.model_paths import ModelPaths

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

imagenet_classes = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]


def predict(image, model_choice: str):
    if image is None:
        return "Please upload an image."

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    model_enum = Models[model_choice]
    paths = ModelPaths.get_model_paths()

    model = torch.load(paths[model_enum], map_location=torch.device("cpu"))
    model.eval()

    img_t = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    class_id = torch.argmax(probs).item()
    confidence = probs[class_id].item()

    class_labels = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
    class_name = class_labels[class_id]

    return f"Prediction: {class_name} (confidence: {confidence:.2f})"
