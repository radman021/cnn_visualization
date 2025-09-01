import os
import torch
from torchvision import models
from copy import deepcopy

from config import Config
from logger import Logger
from enumerations.models import Models
from maps.model_paths import ModelPaths


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
        logger = Logger("cli").get_logger()
        os.makedirs(Config.base_models_folder, exist_ok=True)

        model_paths = ModelPaths.get_model_paths()

        for model_enum in cls._model_builder:

            if os.path.exists(model_paths.get(model_enum)):
                logger.info(
                    f"Model {model_enum.value} already exists on path: {model_paths[model_enum]}"
                )
                continue

            logger.info(f"Model {model_enum.value} isn't present, downloading it...")
            save_path = model_enum.value

            model_instance = cls._model_builder[model_enum]()
            torch.save(model_instance, model_paths[model_enum])
            model_paths[model_enum.value] = save_path

            logger.info(f"Successfully downloaded model {model_enum.value}.")
