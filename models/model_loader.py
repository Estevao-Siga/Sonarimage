import os
import torch
import logging
import gc
import tempfile
from typing import Dict, List, Tuple, Optional, Union

# Import model implementations
from models.YOLOv8 import get_yolov8, load_yolov8, model_configs
from models.resnet import get_resnet

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelLoader")


class MemoryEfficientModel:
    """Helper class to load models with memory efficiency"""

    def __init__(self, model_builder, device, fp16=False):
        self.model_builder = model_builder
        self.device = device
        self.fp16 = fp16

    def load_model(self):
        """Load model with memory considerations"""
        # Clear cache before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create the model
        model = self.model_builder()

        # Half precision if requested
        if self.fp16 and self.device != 'cpu':
            model = model.half()

        # Move to device
        model = model.to(self.device)
        return model


class ModelLoader:
    """Unified model loader for supported architectures"""

    def __init__(self, device=None, arch='yolov8', pretrained=True, fp16=False):
        """
        Initialize the model loader

        Args:
            device: Device to load models onto ('cuda:0', 'cpu', etc.) or None for auto detection
            arch: Default architecture ('yolov8', 'resnet', etc.)
            pretrained: Whether to load pretrained weights 
            fp16: Whether to use half precision (FP16)
        """
        # Set device
        if device is None:
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.arch = arch
        self.pretrained = pretrained
        self.fp16 = fp16

        # Define model architectures and their parameters
        self.models_dict = {
            # YOLO model - unified implementation
            'yolov8': {
                'func': get_yolov8,
                'params': ['size', 'nc', 'pretrained', 'device'],
                'available_sizes': list(model_configs.keys()),
                'description': 'YOLOv8 object detection model'
            },
            # ResNet models
            'resnet': {
                'func': get_resnet,
                'params': ['depth', 'pretrained', 'input_channels', 'num_classes', 'robust_method'],
                'available_sizes': [18, 34, 50, 101, 152],
                'description': 'ResNet image classification model'
            }
        }

        logger.info(
            f"ModelLoader initialized with models: {', '.join(self.models_dict.keys())}")

        # Log device info
        if torch.cuda.is_available():
            logger.info(
                f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            if self.fp16:
                logger.info("FP16 precision enabled")
        else:
            logger.info("Using CPU for computation")

    def _format_model_name(self, model_name, depth_or_size):
        """Format model name with depth/size in a filename-friendly way"""
        if isinstance(depth_or_size, (list, dict)):
            # For multiple depths/sizes
            if isinstance(depth_or_size, dict):
                values = depth_or_size.get(model_name, [])
            else:
                values = depth_or_size
            return f"{model_name}_{'_'.join(map(str, values))}"
        return f"{model_name}_{depth_or_size}"

    def get_latest_checkpoint(self, model_name_with_size, dataset_name, task_name='detect'):
        """Find the latest checkpoint for a model and dataset"""
        # First try in results directory (for YOLOv8)
        checkpoint_dir = f"results/{dataset_name}/{model_name_with_size}/weights"

        if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
            checkpoints = [f for f in os.listdir(
                checkpoint_dir) if f.endswith('.pt')]
            if checkpoints:
                # Prefer best.pt or last.pt if available
                if 'best.pt' in checkpoints:
                    return os.path.join(checkpoint_dir, 'best.pt')
                elif 'last.pt' in checkpoints:
                    return os.path.join(checkpoint_dir, 'last.pt')

                # Otherwise get the most recent file
                checkpoints.sort(key=lambda x: os.path.getmtime(
                    os.path.join(checkpoint_dir, x)), reverse=True)
                return os.path.join(checkpoint_dir, checkpoints[0])

        # Try traditional checkpoint directory structure
        checkpoint_dir = f"out/{task_name}/{dataset_name}/{model_name_with_size}/save_model"
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(
                checkpoint_dir) if f.startswith(f"best_{model_name_with_size}")]
            if checkpoints:
                checkpoints.sort(key=lambda x: os.path.getmtime(
                    os.path.join(checkpoint_dir, x)), reverse=True)
                return os.path.join(checkpoint_dir, checkpoints[0])

        logger.warning(
            f"No checkpoints found for {model_name_with_size} with dataset {dataset_name}")
        return None

    def get_model(self, model_name=None, depth_or_size=None, input_channels=3, num_classes=None,
                  task_name=None, dataset_name=None):
        """
        Get a model with specified architecture and parameters

        Args:
            model_name: Architecture name ('yolov8', 'resnet', etc.)
            depth_or_size: Model depth/size ('n', 's', 'm', 'l', 'x' for YOLOv8 or 18, 34, 50, 101, 152 for ResNet)
            input_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes
            task_name: Task name for checkpoint loading
            dataset_name: Dataset name for checkpoint loading

        Returns:
            Tuple (model, model_name_with_size) or list of tuples for multiple depths/sizes
        """
        model_name = model_name or self.arch

        if model_name not in self.models_dict:
            raise ValueError(
                f"Model '{model_name}' not recognized. Available models: {', '.join(self.models_dict.keys())}")

        model_entry = self.models_dict[model_name]
        model_func = model_entry['func']
        model_params = model_entry['params']
        available_sizes = model_entry['available_sizes']

        # Handle YOLOv8 case where num_classes is 'nc'
        if model_name == 'yolov8' and num_classes is not None:
            kwargs_base = {'nc': num_classes,
                           'pretrained': self.pretrained, 'device': self.device}
        else:
            kwargs_base = {'num_classes': num_classes,
                           'pretrained': self.pretrained, 'input_channels': input_channels}

        # Handle multiple depths/sizes
        if isinstance(depth_or_size, (list, dict)):
            # Extract the values to use
            if isinstance(depth_or_size, dict):
                values = depth_or_size.get(model_name, [])
                if not values:
                    raise ValueError(
                        f"No sizes specified for {model_name} in {depth_or_size}")
            else:
                values = depth_or_size

            # Create models for each size
            models_and_names = []

            for value in values:
                if value not in available_sizes and str(value) not in available_sizes:
                    logger.warning(
                        f"Size '{value}' may not be supported for {model_name}. Available sizes: {available_sizes}")

                # Set size/depth parameter
                if model_name == 'yolov8':
                    kwargs = {**kwargs_base, 'size': value}
                elif model_name == 'resnet':
                    kwargs = {**kwargs_base, 'depth': value}
                else:
                    kwargs = {**kwargs_base, 'depth': value}

                # Filter kwargs to only include supported parameters
                filtered_kwargs = {k: v for k,
                                   v in kwargs.items() if k in model_params}

                # Format model name with size
                model_name_with_size = self._format_model_name(
                    model_name, value)

                # Load model using the specific model function
                model = self._create_or_load_model(
                    model_func, filtered_kwargs, model_name_with_size, task_name, dataset_name)

                models_and_names.append((model, model_name_with_size))

            return models_and_names

        else:
            # Single depth/size case
            # Use default size if not specified
            if depth_or_size is None:
                if model_name == 'yolov8':
                    depth_or_size = 'n'  # Nano by default
                elif model_name == 'resnet':
                    depth_or_size = 50  # ResNet-50 by default

            # Set size/depth parameter
            if model_name == 'yolov8':
                kwargs = {**kwargs_base, 'size': depth_or_size}
            elif model_name == 'resnet':
                kwargs = {**kwargs_base, 'depth': depth_or_size}
            else:
                kwargs = {**kwargs_base, 'depth': depth_or_size}

            # Filter kwargs to only include supported parameters
            filtered_kwargs = {k: v for k,
                               v in kwargs.items() if k in model_params}

            # Format model name with size
            model_name_with_size = self._format_model_name(
                model_name, depth_or_size)

            # Create or load the model
            model = self._create_or_load_model(
                model_func, filtered_kwargs, model_name_with_size, task_name, dataset_name)

            return [(model, model_name_with_size)]

    def _create_or_load_model(self, model_func, kwargs, model_name_with_size, task_name, dataset_name):
        """Create a new model or load from checkpoint"""
        checkpoint_path = None

        # Try to find checkpoint if task_name and dataset_name are provided
        if task_name and dataset_name:
            checkpoint_path = self.get_latest_checkpoint(
                model_name_with_size, dataset_name, task_name)

        # If checkpoint found, load from checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                logger.info(
                    f"Loading model from checkpoint: {checkpoint_path}")

                # For YOLOv8 models, use the load_yolov8 function
                if 'yolov8' in model_name_with_size.lower():
                    model = load_yolov8(checkpoint_path, device=self.device)
                else:
                    # For other models, create and then load weights
                    model = model_func(**kwargs)
                    checkpoint = torch.load(
                        checkpoint_path, map_location='cpu')

                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        checkpoint = checkpoint['model']

                    # Remove "module." prefix if present (from DataParallel)
                    if hasattr(checkpoint, 'items'):
                        new_state_dict = {}
                        for k, v in checkpoint.items():
                            new_k = k.replace("module.", "") if k.startswith(
                                "module.") else k
                            new_state_dict[new_k] = v
                        checkpoint = new_state_dict

                    # Load state dict
                    model.load_state_dict(checkpoint, strict=False)
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                # Fallback to creating a new model
                model = model_func(**kwargs)
        else:
            # Create a new model
            model = model_func(**kwargs)

        # Convert to half precision if requested
        if self.fp16 and self.device != 'cpu' and hasattr(model, 'half'):
            try:
                model = model.half()
                logger.info("Using half precision (FP16)")
            except Exception as e:
                logger.warning(f"Failed to convert model to FP16: {e}")

        # Ensure model is on the correct device
        if hasattr(model, 'to'):
            model = model.to(self.device)

        # Free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model

    def list_available_models(self):
        """List all available models and their configurations"""
        print("\nAvailable Models:")
        print("=" * 80)

        for model_name, model_info in self.models_dict.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 40)
            print(f"Description: {model_info['description']}")

            # Show detailed info for each size/variant
            print(f"\nAvailable variants:")

            # Use model-specific configurations when available
            if model_name == 'yolov8':
                for size_code in model_info['available_sizes']:
                    config = model_configs.get(size_code, {})
                    full_name = config.get('full_name', f"YOLOv8 {size_code}")
                    params = config.get('params', 'N/A')
                    flops = config.get('flops', 'N/A')
                    desc = config.get('description', "")
                    print(f"  - {model_name}{size_code}: {full_name}")
                    print(f"    Parameters: {params}M, FLOPs: {flops}B")
                    print(f"    {desc}")
            else:
                for size_code in model_info['available_sizes']:
                    print(f"  - {model_name}-{size_code}")
                    # Add more details if available for other model types

        print("\n" + "=" * 80)
        print("Note: Custom models like YOLOv8x-EM-Fast incorporate specialized blocks for enhanced detection")
        print(
            "      Use --arch yolov8 --depth x-em-fast to select the modified architecture")
        print("=" * 80)

    def load_pretrained_model(self, model_name, dataset_name=None, depth_or_size=None,
                              input_channels=3, num_classes=None, task_name=None):
        """
        Load a pretrained model with the specified configuration

        Args:
            model_name: Model architecture name
            dataset_name: Dataset name for checkpoint loading
            depth_or_size: Model depth/size
            input_channels: Number of input channels
            num_classes: Number of output classes
            task_name: Task name for checkpoint loading

        Returns:
            Loaded model
        """
        model, _ = self.get_model(
            model_name=model_name,
            depth_or_size=depth_or_size,
            input_channels=input_channels,
            num_classes=num_classes,
            task_name=task_name,
            dataset_name=dataset_name
        )[0]

        # Use DataParallel for multi-GPU setup
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            logger.info(
                f"Using DataParallel with {torch.cuda.device_count()} GPUs")

        return model


# === Added standalone functions for backward compatibility ===

def download_model(model_name='yolov8n.pt', output_dir='models/weights'):
    """Use download_model from YOLOv8 module"""
    from models.YOLOv8 import download_model as yolo_download
    return yolo_download(model_name, output_dir)


def verify_model(model_path):
    """Wrapper function for backward compatibility"""
    try:
        _ = torch.load(model_path, map_location="cpu")
        return True
    except Exception:
        return False


def load_model(model_path='results/steve/weights/best.pt', device=None):
    """Use the appropriate model loader based on file type"""
    if model_path.endswith('.pt'):
        # For YOLO models
        from models.YOLOv8 import load_yolov8
        return load_yolov8(model_path, device)
    else:
        # Generic model loading as fallback
        loader = ModelLoader(device=device)
        return loader.load_model(model_path, device)

