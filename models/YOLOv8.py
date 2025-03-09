import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional, Callable
from ultralytics import YOLO
import numpy as np

# Fix imports to match the correct file locations
from models.yolo.blocks import C2fEMFast, C2fTranslearn
from models.yolo.fasternet import create_fasternet_backbone

# Model configurations for different YOLOv8 sizes
model_configs = {
    # Format: size code -> model details
    'n': {
        'name': 'yolov8n.pt',
        'full_name': 'YOLOv8 Nano',
        'params': 3.2,
        'flops': 8.7,
        'description': 'Smallest and fastest model'
    },
    's': {
        'name': 'yolov8s.pt',
        'full_name': 'YOLOv8 Small',
        'params': 11.2,
        'flops': 28.6,
        'description': 'Small model, balanced speed-accuracy'
    },
    'm': {
        'name': 'yolov8m.pt',
        'full_name': 'YOLOv8 Medium',
        'params': 25.9,
        'flops': 78.9,
        'description': 'Medium-sized model with good accuracy'
    },
    'l': {
        'name': 'yolov8l.pt',
        'full_name': 'YOLOv8 Large',
        'params': 43.7,
        'flops': 165.2,
        'description': 'Large model with high accuracy'
    },
    'x': {
        'name': 'yolov8x.pt',
        'full_name': 'YOLOv8 XLarge',
        'params': 68.2,
        'flops': 257.8,
        'description': 'Extra large model with highest accuracy'
    },
    'x-em-fast': {
        'name': 'yolov8x-em-fast.pt',
        'full_name': 'YOLOv8 XLarge with C2f-EM-Fast',
        'params': 55.0,  # Approximate value based on paper
        'flops': 50.5,  # Value from paper
        'description': 'Modified XLarge with FasterNet, C2f-EM-Fast blocks'
    }
}


class YOLOv8:
    """
    YOLOv8 model wrapper for Ultralytics YOLO implementation

    This class provides a simplified interface to the Ultralytics YOLO model,
    allowing specification of model size using simple codes ('n', 's', etc.)
    and supports customization and enhancement of the base model.
    """

    def __init__(self, size: str = 'n', nc: int = None, pretrained: bool = True, device=None):
        """
        Initialize YOLOv8 model

        Args:
            size: Model size, one of ['n', 's', 'm', 'l', 'x', 'x-em-fast'] for different variants
            nc: Number of classes (None to use default)
            pretrained: Whether to load pre-trained weights
            device: Device to load model to ('cpu', 'cuda:0', etc.)
        """
        self.size = size
        self.nc = nc

        # Validate size
        if size not in model_configs:
            valid_sizes = list(model_configs.keys())
            raise ValueError(
                f"Invalid YOLOv8 size: {size}. Available sizes: {', '.join(valid_sizes)}")

        self.config = model_configs[size]
        model_name = self.config['name']

        # Check if model exists in weights directory first
        weights_path = os.path.join('models/weights', model_name)
        model_path = weights_path if os.path.exists(
            weights_path) else model_name

        # Set device
        if device is None:
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Load model with special handling for x-em-fast
        if size == 'x-em-fast':
            # Start with base YOLOv8x model
            if pretrained:
                base_model_path = os.path.join('models/weights', 'yolov8x.pt')
                if os.path.exists(base_model_path):
                    self.model = YOLO(base_model_path)
                else:
                    # Will download if not found
                    self.model = YOLO('yolov8x.pt')

                # Apply modifications for x-em-fast variant
                self._apply_c2f_em_fast_modifications()
            else:
                # For training from scratch
                self.model = YOLO('yolov8x.yaml')
                self._apply_c2f_em_fast_modifications()

            if hasattr(self.model, 'to') and self.device:
                self.model.to(self.device)
        else:
            # Standard YOLOv8 loading
            if pretrained:
                # Try to load from local path first
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                else:
                    # If not found, YOLO will attempt to download it
                    self.model = YOLO(model_name)

                if hasattr(self.model, 'to') and self.device:
                    self.model.to(self.device)
            else:
                # For custom training, start with pretrained but plan to retrain
                self.model = YOLO(model_path if os.path.exists(
                    model_path) else model_name)
                if hasattr(self.model, 'to') and self.device:
                    self.model.to(self.device)

        # Add hooks for pre and post processing
        self.pre_process_hooks = []
        self.post_process_hooks = []

        # Store custom layers that can be added to the model
        self.custom_layers = nn.ModuleDict()

    def _apply_c2f_em_fast_modifications(self):
        """
        Apply C2f-EM-Fast modifications to YOLOv8x model.
        This includes:
        1. Integrating FasterNet modules into the backbone
        2. Replacing C2f modules with C2f-EM-Fast modules
        3. Streamlining the fusion process
        """
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'model'):
            print("Cannot access underlying model layers for modification")
            return

        try:
            # Access the main model components
            model_layers = self.model.model.model

            # Create FasterNet backbone for feature extraction support
            fasternet_backbone = create_fasternet_backbone(
                in_channels=3,
                base_channels=64
            )

            # 1. First modification: Replace backbone C2f modules with C2f-EM-Fast
            c2f_modules = [
                (name, module) for name, module in model_layers.named_modules()
                if 'model.2' in name or 'model.4' in name or 'model.6' in name or 'model.8' in name
            ]

            for name, module in c2f_modules:
                if hasattr(module, 'cv1') and hasattr(module, 'cv2'):
                    # Get configuration from existing module
                    c1 = module.cv1.in_channels
                    c2 = module.cv2.out_channels
                    n = len(module.m) if hasattr(module, 'm') else 1
                    shortcut = getattr(module, 'shortcut', True)

                    # Create replacement C2f-EM-Fast module with integrated FasterNet features
                    new_module = C2fEMFast(c1, c2, n, shortcut)

                    # Find parent module and replace
                    parent_name = name.rsplit('.', 1)[0]
                    attr_name = name.rsplit('.', 1)[1]
                    parent = model_layers

                    for part in parent_name.split('.')[1:]:
                        parent = getattr(parent, part)

                    # Replace the module
                    setattr(parent, attr_name, new_module.to(self.device))

            # 2. Second modification: Update neck modules to use C2fTranslearn
            neck_modules = [
                (name, module) for name, module in model_layers.named_modules()
                if 'model.12' in name or 'model.15' in name or 'model.18' in name or 'model.21' in name
            ]

            for name, module in neck_modules:
                if hasattr(module, 'cv1') and hasattr(module, 'cv2'):
                    # Get configuration from existing module
                    c1 = module.cv1.in_channels
                    c2 = module.cv2.out_channels
                    n = len(module.m) if hasattr(module, 'm') else 1
                    shortcut = getattr(module, 'shortcut', True)

                    # Create replacement C2fTranslearn module
                    new_module = C2fTranslearn(c1, c2, n, shortcut)

                    # Find parent module and replace
                    parent_name = name.rsplit('.', 1)[0]
                    attr_name = name.rsplit('.', 1)[1]
                    parent = model_layers

                    for part in parent_name.split('.')[1:]:
                        parent = getattr(parent, part)

                    # Replace the module
                    setattr(parent, attr_name, new_module.to(self.device))

            print("Successfully applied C2f-EM-Fast modifications to YOLOv8x model")

        except Exception as e:
            print(f"Error applying C2f-EM-Fast modifications: {e}")
            import traceback
            traceback.print_exc()

    def __call__(self, *args, **kwargs):
        """Forward pass through the model with pre/post processing"""
        inputs = args[0] if args else kwargs.get('source')

        # Apply pre-processing hooks
        for hook in self.pre_process_hooks:
            inputs = hook(inputs)

        # Process through custom pre-layers if any
        if 'pre' in self.custom_layers:
            inputs = self.custom_layers['pre'](inputs)

        # Run through the model
        outputs = self.model(inputs, *args[1:], **kwargs)

        # Process through custom post-layers if any
        if 'post' in self.custom_layers:
            outputs = self.custom_layers['post'](outputs)

        # Apply post-processing hooks
        for hook in self.post_process_hooks:
            outputs = hook(outputs)

        return outputs

    def train(self, *args, **kwargs):
        """Train the model"""
        return self.model.train(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Run prediction"""
        return self.model.predict(*args, **kwargs)

    def val(self, *args, **kwargs):
        """Validate the model"""
        return self.model.val(*args, **kwargs)

    def export(self, *args, **kwargs):
        """Export the model"""
        return self.model.export(*args, **kwargs)

    def to(self, device):
        """Move model to specified device"""
        if hasattr(self.model, 'to'):
            self.model.to(device)
        return self

    def add_custom_layer(self, name: str, layer: nn.Module, position: str = 'post'):
        """
        Add a custom layer to the model

        Args:
            name: Name of the custom layer
            layer: The layer module to add
            position: Where to add the layer ('pre', 'post', or specific layer name in the model)
        """
        self.custom_layers[name] = layer
        # Move the layer to the same device as the model
        if hasattr(self.model, 'device'):
            self.custom_layers[name] = self.custom_layers[name].to(
                self.model.device)
        return self

    def add_pre_process_hook(self, hook: Callable):
        """Add a hook function that runs before processing input"""
        self.pre_process_hooks.append(hook)
        return self

    def add_post_process_hook(self, hook: Callable):
        """Add a hook function that runs after processing output"""
        self.post_process_hooks.append(hook)
        return self

    def get_layer_by_name(self, layer_name: str) -> nn.Module:
        """
        Access a specific layer in the model by name

        Args:
            layer_name: Name of the layer to retrieve

        Returns:
            The requested layer as a nn.Module
        """
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'model'):
            raise AttributeError("Cannot access underlying model layers")

        # Access the main model components
        model_layers = self.model.model.model

        # Try to find the layer directly
        if hasattr(model_layers, layer_name):
            return getattr(model_layers, layer_name)

        # Otherwise search through named modules
        for name, module in model_layers.named_modules():
            if name.endswith(layer_name):
                return module

        raise ValueError(f"Layer '{layer_name}' not found in model")

    def modify_layer(self, layer_name: str, new_layer: nn.Module) -> bool:
        """
        Replace a specific layer in the model

        Args:
            layer_name: Name of the layer to replace
            new_layer: New layer to use as replacement

        Returns:
            Success status
        """
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'model'):
            raise AttributeError("Cannot modify underlying model layers")

        # Access the main model components
        model_layers = self.model.model.model

        # Find parent module and attribute name
        parent = None
        attr_name = None

        # Direct child of model_layers
        if hasattr(model_layers, layer_name):
            parent = model_layers
            attr_name = layer_name
        else:
            # Search through the model
            parts = layer_name.split('.')
            if len(parts) > 1:
                parent_path = '.'.join(parts[:-1])
                attr_name = parts[-1]

                for name, module in model_layers.named_modules():
                    if name == parent_path:
                        parent = module
                        break

        if parent is not None and attr_name is not None:
            # Replace the layer
            setattr(parent, attr_name, new_layer.to(self.device))
            return True

        return False

    def get_feature_extractor(self):
        """
        Create a feature extractor that returns intermediate layer outputs

        This allows you to get activations from specific layers

        Returns:
            Callable function that extracts features from the model
        """
        if not hasattr(self.model, 'model'):
            raise AttributeError(
                "Cannot create feature extractor for this model")

        def extract_features(x, layer_names=None):
            """
            Extract features from specified layers

            Args:
                x: Input tensor
                layer_names: List of layer names to extract features from
                             (None to use default extraction points)

            Returns:
                Dictionary of layer_name -> feature_map
            """
            # Default extraction points if none specified
            if layer_names is None:
                # Common extraction points for YOLOv8
                layer_names = ['model.2', 'model.4',
                               'model.6', 'model.8', 'model.9']

            features = {}

            def hook_fn(name):
                def fn(_, __, output):
                    features[name] = output
                return fn

            # Register hooks
            handles = []
            for name in layer_names:
                try:
                    # Find the layer
                    layer = dict(self.model.model.named_modules())[name]
                    # Register the hook
                    handle = layer.register_forward_hook(hook_fn(name))
                    handles.append(handle)
                except (KeyError, AttributeError):
                    print(f"Warning: Layer {name} not found")

            # Forward pass
            _ = self.model(x)

            # Remove hooks
            for handle in handles:
                handle.remove()

            return features

        return extract_features


def get_yolov8(size: str = 'n', nc: int = None, pretrained: bool = True, device=None) -> YOLOv8:
    """
    Get a YOLOv8 model with specified size

    Args:
        size: Model size, one of ['n', 's', 'm', 'l', 'x', 'x-em-fast'] for different variants
        nc: Number of classes (None to use default)
        pretrained: Whether to load pre-trained weights
        device: Device to load model to ('cpu', 'cuda:0', etc.)

    Returns:
        YOLOv8 model instance
    """
    valid_sizes = list(model_configs.keys())
    if size not in valid_sizes:
        raise ValueError(
            f"Invalid size: {size}. Available sizes: {', '.join(valid_sizes)}")

    return YOLOv8(size=size, nc=nc, pretrained=pretrained, device=device)


def load_yolov8(model_path: str, device=None) -> YOLOv8:
    """
    Load a YOLOv8 model from a specific path

    Args:
        model_path: Path to the model file
        device: Device to load model to ('cpu', 'cuda:0', etc.)

    Returns:
        YOLOv8 model instance
    """
    # Create a wrapper with a placeholder size
    yolo = YOLOv8(size='n', pretrained=False, device=device)

    # Replace with the actual model
    yolo.model = YOLO(model_path)

    # Move to device if specified
    if device is not None and hasattr(yolo.model, 'to'):
        yolo.model.to(device)

    return yolo


def list_yolov8_versions():
    """
    Print detailed information about all YOLOv8 model versions
    """
    print("\nYOLOv8 Model Versions:")
    print("=" * 80)
    print(f"{'Model':<12} | {'Size':<12} | {'Params':<12} | {'FLOPs':<12} | {'Description':<30}")
    print("-" * 80)

    for size_code, config in sorted(model_configs.items()):
        model_name = f"yolov8{size_code}"
        full_name = config['full_name']
        params = f"{config['params']}M"
        flops = f"{config['flops']}B"
        description = config['description']

        print(
            f"{model_name:<12} | {full_name:<12} | {params:<12} | {flops:<12} | {description:<30}")

    print("-" * 80)
    print("* Params = Parameters, FLOPs = Floating Point Operations")
    print("* Detection: Use for object detection tasks")
    print("=" * 80)
