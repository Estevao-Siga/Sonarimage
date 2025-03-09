# This file makes the models directory a proper Python package
from .model_loader import (
    ModelLoader,
    download_model,
    load_model,
    verify_model
)
from .YOLOv8 import (
    YOLOv8,
    get_yolov8
)
from .resnet import (
    ResNetModel,
    get_resnet
)

__all__ = [
    'ModelLoader',
    'download_model',
    'load_model',
    'verify_model',
    'YOLOv8',
    'get_yolov8',
    'ResNetModel',
    'get_resnet'
]
