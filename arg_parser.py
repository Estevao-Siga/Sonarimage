import argparse


def create_argument_parser():
    """
    Create the argument parser with all command line options organized in groups
    Supports both hyphen-separated (--train-batch) and underscore-separated (--train_batch) argument styles

    Returns:
        parser: Configured ArgumentParser object
    """
    parser = argparse.ArgumentParser(
        description='Underwater Object Detection using YOLOv8')

    # Create argument groups for better organization

    # Operation mode group
    mode_group = parser.add_argument_group('Operation Mode')
    mode_group.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'all'],
                            help='Mode to run (train, test, all)')
    mode_group.add_argument('--list-datasets', '--list_datasets', dest='list_datasets', action='store_true',
                            help='List all available datasets')
    mode_group.add_argument('--list-models', '--list_models', dest='list_models', action='store_true',
                            help='List all available models')
    mode_group.add_argument('--visualize-data', '--visualize_data', dest='visualize_data', action='store_true',
                            help='Visualize training data')
    mode_group.add_argument('--fast', '--fast-mode', dest='fast_mode', action='store_true',
                            help='Enable fast training mode with optimized parameters')

    # Dataset selection group
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument('--data', '--dataset', dest='dataset', type=str, default='seabedok',
                               help='Dataset name to use (use --list_datasets to see available options)')

    # Model selection group
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--arch', type=str, default='yolov8',
                             help='Model architecture (yolov8, resnet, etc.)')
    model_group.add_argument('--depth', '--size', dest='depth_or_size', type=str, default=None,
                             help='Model depth/size (n, s, m, l, x for YOLOv8 or 18, 34, 50, 101, 152 for ResNet)')
    model_group.add_argument('--model-path', '--model_path', dest='model_path', type=str, default=None,
                             help='Path to model weights for testing (defaults to best weights from training)')

    # Training parameters group
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--epochs', type=int, default=100,
                             help='Number of training epochs')
    train_group.add_argument('--train-batch', '--train_batch', '--batch', dest='batch', type=int, default=8,
                             help='Training batch size')
    train_group.add_argument('--lr', type=float, default=0.01,
                             help='Learning rate')
    train_group.add_argument('--auto-batch', '--auto_batch', dest='auto_batch', action='store_true',
                             help='Automatically adjust batch size based on model and available memory')
    train_group.add_argument('--workers', type=int, default=4,
                             help='Number of worker threads for data loading')
    train_group.add_argument('--seed', type=int, default=42,
                             help='Random seed for reproducibility')
    train_group.add_argument('--cache', action='store_true',
                             help='Cache images in RAM for faster training')
    train_group.add_argument('--patience', '--early_stopping', dest='patience', type=int, default=10,
                             help='Patience for early stopping (stop after N epochs with no improvement)')
    train_group.add_argument('--image-weights', '--image_weights', dest='image_weights', action='store_true',
                             help='Use weighted image selection for imbalanced datasets')

    # Hardware configuration group
    hw_group = parser.add_argument_group('Hardware Configuration')
    hw_group.add_argument('--gpu-ids', '--gpu_ids', '--device', dest='device', type=str, default=None,
                          help='Device to run on (comma-separated GPU ids like 0,1 or just 0 for single GPU)')
    hw_group.add_argument('--fp16', action='store_true', default=True,
                          help='Use mixed precision training (FP16)')

    return parser


def parse_args():
    """
    Parse command line arguments

    Returns:
        args: Parsed arguments
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    Display help information when this script is run directly
    """
    parser = create_argument_parser()
    parser.print_help()
