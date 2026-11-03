from .base01 import BaseModel_01

from .heart01 import HeartNetwork_01

MODEL_REGISTRY = {"base01": BaseModel_01, "heart01": HeartNetwork_01}


def build_model(model_config):
    model_name = model_config["name"]
    model_class = MODEL_REGISTRY[model_name]
    kwargs = {"in_channels": 1, "out_channels": 4}

    return model_class(**kwargs)
