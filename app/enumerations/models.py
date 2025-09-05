from enum import Enum


class Models(Enum):
    RESNET_18 = "ResNet-18"
    RESNET_34 = "ResNet-34"
    RESNET_50 = "ResNet-50"

    @staticmethod
    def get_all_models():
        return [{"model_id": model.name, "model_name": model.value} for model in Models]
