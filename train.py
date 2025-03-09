import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import cv2
import shutil
# Change the import to use the model loader instead
from models.model_loader import download_model, verify_model


def setup_font_properties():
    """Set up custom font properties for plotting"""
    plt.rcParams.update({
        'font.size': 25,
        'font.family': 'Times New Roman',
        'font.weight': 'bold',
    })


def relocate_model_files(source_dir='.', target_dir='models/weights'):
    """
    Relocate any model files that may have been downloaded to the source directory
    during training operations

    Args:
        source_dir: Directory to search for model files (default: current directory)
        target_dir: Directory to move model files to (default: models/weights)
    """
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Find all model files (*.pt) in the source directory
    for file in os.listdir(source_dir):
        if file.endswith('.pt') and os.path.isfile(os.path.join(source_dir, file)):
            # Skip files that are already in a subdirectory containing 'model' or 'weight'
            if ('model' in source_dir.lower() or 'weight' in source_dir.lower()) and target_dir != source_dir:
                continue

            source_path = os.path.join(source_dir, file)
            target_path = os.path.join(target_dir, file)

            # Check if file already exists in target directory
            if os.path.exists(target_path):
                print(f"File {file} already exists in {target_dir}")
                continue

            try:
                print(f"Moving {file} from {source_dir} to {target_dir}")
                shutil.move(source_path, target_path)
            except Exception as e:
                print(f"Error moving {file}: {e}")


def train_model(data_path, img_size, model=None, epochs=300, seed=42, batch=8, workers=4,
                dataset_name='seabedok', model_name='yolov8n', device=None,
                cache=True, amp=True, image_weights=False, lr=0.01, patience=10):
    """
    Train the YOLOv8 model

    Args:
        data_path: Path to the YAML file
        img_size: Tuple of (height, width, channels)
        model: Model object to train (takes precedence over model_name)
        epochs: Number of training epochs
        seed: Random seed for reproducibility
        batch: Batch size
        workers: Number of worker threads for data loading
        dataset_name: Name of the dataset (for saving results)
        model_name: Name of the model architecture (for saving results)
        device: Device to use (None=auto, 0=first GPU, etc.)
        cache: Whether to cache images in RAM for faster training
        amp: Whether to use automatic mixed precision
        image_weights: DEPRECATED - no longer used (kept for backward compatibility)
        lr: Learning rate
        patience: Number of epochs to wait for improvement before early stopping

    Returns:
        trained_model: The trained YOLO model
    """
    # Create directory structure based on dataset and model name
    # Format: results/{dataset_name}/{model_name}/
    results_dir = os.path.join('results', dataset_name, model_name)
    weights_dir = os.path.join(results_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # Create other results directories
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'visualizations'), exist_ok=True)

    # Define custom naming for the best weights file
    best_weights_name = f"best_{model_name}_{dataset_name}_{epochs}_{batch}_{lr}.pt"
    best_weights_path = os.path.join(weights_dir, best_weights_name)

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Check if CUDA is available and set device
    if device is None:
        if torch.cuda.is_available():
            print(
                f"CUDA is available. Found {torch.cuda.device_count()} GPU(s).")
            device = 0  # Use first GPU by default
            torch.cuda.set_device(device)
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            print("CUDA is not available. Using CPU.")
            device = 'cpu'

    # Free up GPU memory if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Look for model files in root directory and move them to models/weights
    relocate_model_files()

    # Training the model - explicitly specify device
    print(
        f"Starting training with {epochs} epochs, batch size {batch}, and {workers} workers on device {device}...")
    try:
        # Remove image_weights parameter which is no longer supported
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=img_size,
            seed=seed,
            batch=batch,
            workers=workers,
            name=model_name,  # Use just the model name
            # Project is dataset name
            project=os.path.join('results', dataset_name),
            exist_ok=True,    # Overwrite existing files
            device=device,
            cache=cache,         # Cache images in RAM for faster training
            amp=amp,             # Use automatic mixed precision - IMPORTANT for speed
            close_mosaic=10,     # Disable mosaic augmentation for final epochs
            cos_lr=True,         # Use cosine learning rate scheduler
            optimizer='AdamW',   # Use AdamW optimizer which often works better than SGD
            pretrained=True,     # Use pretrained weights
            lr0=lr,              # Initial learning rate
            lrf=lr/100,          # Final learning rate ratio
            patience=patience,   # Early stopping patience
            # Disable most plots to save time in fast mode
            plots=False if patience < 10 else True,
            save_period=-1       # Only save final and best models
        )

        # After training, process and save the best weights properly
        yolo_best_weights = os.path.join(
            'results', dataset_name, model_name, 'weights', 'best.pt')

        if os.path.exists(yolo_best_weights):
            # Check if the source and destination are different before copying
            if os.path.abspath(yolo_best_weights) != os.path.abspath(best_weights_path):
                # Copy and rename best weights with our custom naming
                shutil.copy(yolo_best_weights, best_weights_path)
                print(f"Saved best weights as {best_weights_path}")
            else:
                print(f"Best weights already exist at {best_weights_path}")

            # Don't attempt to create another copy with the name 'best.pt' if it's already the source file
            # This was causing the SameFileError

            # Remove any epoch-specific weight files to save space
            for file in os.listdir(weights_dir):
                if file.startswith('epoch') and file.endswith('.pt'):
                    try:
                        os.remove(os.path.join(weights_dir, file))
                        print(f"Removed intermediate checkpoint: {file}")
                    except Exception as e:
                        print(f"Could not remove file {file}: {e}")

        # Check again for any model files downloaded during training
        relocate_model_files()

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(
                f"CUDA out of memory error. Your batch size ({batch}) is too large for your GPU.")
            if batch > 1:
                new_batch = max(1, batch // 2)
                print(
                    f"Trying again with a smaller batch size of {new_batch}...")
                # Free up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Check for any model files downloaded before retry
                relocate_model_files()

                # Try again with half the batch size - fix the name variable
                results = model.train(
                    data=data_path,
                    epochs=epochs,
                    imgsz=img_size,
                    seed=seed,
                    batch=new_batch,
                    workers=workers,
                    name=model_name,  # Fixed: using model_name instead of name
                    # Consistent with primary code path
                    project=os.path.join('results', dataset_name),
                    device=device,
                    lr0=lr,
                    lrf=lr/100
                    # image_weights parameter removed
                )

                # Check again after retry
                relocate_model_files()

            else:
                print("Your GPU doesn't have enough memory even with batch size of 1.")
                print(
                    "Try using a smaller model (yolov8n.pt, yolov8s.pt), or reduce image size.")
                raise
        else:
            raise

    print(f"Training completed. Model saved in {results_dir}")
    if os.path.exists(best_weights_path):
        print(f"Best model path: {best_weights_path}")

    # Final check for any model files downloaded during training
    relocate_model_files()

    return model


def visualize_training_metrics(results_csv_path=None, dataset_name=None, model_name=None):
    """Visualize training metrics from the results CSV file"""
    # Determine the CSV path based on parameters
    if results_csv_path is None:
        if dataset_name is None or model_name is None:
            print(
                "Error: Must provide either results_csv_path or both dataset_name and model_name")
            return
        results_csv_path = os.path.join(
            'results', dataset_name, model_name, 'results.csv')

    if not os.path.exists(results_csv_path):
        print(f"Error: Results file not found at {results_csv_path}")
        return

    # Get the output directory from the CSV path
    output_dir = os.path.dirname(results_csv_path)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Read the results CSV file
    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip()

    # Create subplots using seaborn
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 35))

    # Plot the columns using seaborn
    sns.lineplot(x='epoch', y='train/box_loss', data=df, ax=axs[0, 0])
    sns.lineplot(x='epoch', y='train/cls_loss', data=df, ax=axs[0, 1])
    sns.lineplot(x='epoch', y='train/dfl_loss', data=df, ax=axs[1, 0])
    sns.lineplot(x='epoch', y='metrics/precision(B)', data=df, ax=axs[1, 1])
    sns.lineplot(x='epoch', y='metrics/recall(B)', data=df, ax=axs[2, 0])
    sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=df, ax=axs[2, 1])
    sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=df, ax=axs[3, 0])
    sns.lineplot(x='epoch', y='val/box_loss', data=df, ax=axs[3, 1])
    sns.lineplot(x='epoch', y='val/cls_loss', data=df, ax=axs[4, 0])
    sns.lineplot(x='epoch', y='val/dfl_loss', data=df, ax=axs[4, 1])

    # Set titles and axis labels for each subplot
    axs[0, 0].set(title='Train Box Loss')
    axs[0, 1].set(title='Train Class Loss')
    axs[1, 0].set(title='Train DFL Loss')
    axs[1, 1].set(title='Metrics Precision (B)')
    axs[2, 0].set(title='Metrics Recall (B)')
    axs[2, 1].set(title='Metrics mAP50 (B)')
    axs[3, 0].set(title='Metrics mAP50-95 (B)')
    axs[3, 1].set(title='Validation Box Loss')
    axs[4, 0].set(title='Validation Class Loss')
    axs[4, 1].set(title='Validation DFL Loss')

    # Add suptitle and subheader
    plt.suptitle('Training Metrics and Loss', fontsize=16)

    # Adjust top margin to make space for suptitle
    plt.subplots_adjust(top=0.6)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save plot to the plots directory in the same folder as the CSV
    output_path = os.path.join(plots_dir, 'training_metrics.png')
    plt.savefig(output_path)
    print(f"Training metrics plot saved to {output_path}")
    plt.show()


def test_model(model_path='results/steve/weights/best.pt', conf=0.25, split='test'):
    """Test the model on test dataset after training"""
    # Import here to avoid circular import
    import test

    # Load model
    model = YOLO(model_path)

    # Evaluate model
    metrics = test.evaluate_model(model, conf, split)

    # Visualize some predictions
    test.visualize_predictions(
        model, 'dataset/seabedok/test/images', num_samples=12)

    return metrics


def train_and_test(data_path, img_size, epochs=300, batch=8, test_after_training=True):
    """Train and optionally test the model"""
    # Train the model
    model = train_model(data_path, img_size, epochs=epochs, batch=batch)

    # Visualize training metrics
    visualize_training_metrics()

    # Test if requested
    if test_after_training:
        metrics = test_model()

    return model


if __name__ == "__main__":
    # This allows the script to be run directly for debugging or testing
    from arg_parser import parse_args

    args = parse_args()
    setup_font_properties()

    # Apply fast mode if enabled
    if hasattr(args, 'fast_mode') and args.fast_mode:
        from main import apply_fast_mode_settings
        args = apply_fast_mode_settings(args)

    if args.dataset and args.model:
        from data_loader import get_data_paths, get_class_mappings, create_yaml_file

        # Get paths and prepare data
        paths = get_data_paths(args.dataset)
        yaml_path = create_yaml_file(args.dataset)

        # Train model
        model = train_model(
            yaml_path,
            (640, 640, 3),  # Default size if not specified
            model_name=args.model,
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
            seed=args.seed,
            device=args.device
        )

        # Visualize results if available
        results_csv = f"results/{args.dataset}_model/results.csv"
        if os.path.exists(results_csv):
            visualize_training_metrics(results_csv)
    else:
        print("Dataset and model must be specified when running train.py directly")
        print("Example: python train.py --dataset seabedok --model yolov8n.pt")
