import os
import torch
from torchvision import models

from config import Config
from enumerations.models import Models
from maps.model_path import model_path


class ModelsManager:

    _model_builder = {
        Models.RESNET_18: lambda: models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        ),
        Models.RESNET_34: lambda: models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1
        ),
        Models.RESNET_50: lambda: models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        ),
    }

    @classmethod
    def init_all_models(cls):
        os.makedirs(Config.base_models_folder, exist_ok=True)

        for model in cls._model_builder:
            if model_path[model]:
                continue

            save_path = os.path.join(Config.base_models_folder, f"{model.value}.pth")

            model = cls._model_builder[model]()
            torch.save(model, save_path)

            model_path[model] = save_path

        return model_path
