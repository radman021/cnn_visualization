from enumerations.models import Models


class ModelPaths:
    _model_paths = {
        Models.RESNET_18: "app/models/resnet-18.pth",
        Models.RESNET_34: "app/models/resnet-34.pth",
        Models.RESNET_50: "app/models/resnet-50.pth",
    }

    @classmethod
    def get_model_paths(cls):
        return cls._model_paths
