import os
import sys
import torch
import shutil
from arg_parser import parse_args
from data_loader import (get_data_paths, get_class_mappings, visualize_image_with_annotation_bboxes,
                         get_image_dimensions, create_yaml_file, list_available_datasets,
                         verify_dataset_structure)
# Fix import statement - move to top-level import to ensure it's available
from models.model_loader import ModelLoader, download_model, load_model
from train import setup_font_properties, train_model, visualize_training_metrics
from test import run_test


def relocate_model_files(source_dir='.', target_dir='models/weights'):
    """
    Relocate any model files that may have been downloaded to the source directory

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


def apply_fast_mode_settings(args):
    """
    Apply optimized settings for fast training mode

    Args:
        args: Command-line arguments

    Returns:
        Updated arguments
    """
    print("=" * 60)
    print("🚀 FAST TRAINING MODE ENABLED 🚀")

    # Use smaller model for faster training
    if args.arch == 'yolov8' and (args.depth_or_size is None or args.depth_or_size in ['l', 'x']):
        args.depth_or_size = 'n'  # Nano model - 18x smaller than X model
        print(f"Using YOLOv8n (nano) model for faster training")
    elif args.arch == 'resnet' and (args.depth_or_size is None or int(args.depth_or_size) > 34):
        args.depth_or_size = '18'  # Use ResNet18 instead of larger models
        print(f"Using ResNet18 for faster training")

    # Optimize batch size if not explicitly set by user
    if args.batch == 8:  # Default value
        args.batch = 16

    # Always enable FP16 and cache in fast mode
    args.fp16 = True
    args.cache = True
    args.patience = 5  # Early stopping

    # Adjust epochs if still at default
    if args.epochs == 100:  # Default value
        args.epochs = 30

    # Set workers based on CPU cores
    args.workers = min(8, os.cpu_count() or 4)

    print(
        f"Optimized settings: batch={args.batch}, workers={args.workers}, epochs={args.epochs}")
    print("Mixed precision (FP16) and image caching enabled")
    print(f"Early stopping after {args.patience} epochs without improvement")
    print("=" * 60)

    return args


def main():
    """Main entry point for the underwater object detection script"""
    try:
        # Parse command line arguments using the centralized argument parser
        args = parse_args()

        # Apply fast mode settings if enabled
        if args.fast_mode:
            args = apply_fast_mode_settings(args)

        # Create only the base results directory and models/weights
        # Remove redundant subdirectories that aren't model-specific
        os.makedirs('results', exist_ok=True)
        os.makedirs('models/weights', exist_ok=True)

        # Check for and relocate any model files in root directory
        relocate_model_files()

        # Show dataset list if requested
        if args.list_datasets:
            list_available_datasets()
            return 0

        # Show model list if requested
        if args.list_models:
            loader = ModelLoader()  # Now ModelLoader is imported at the top level
            loader.list_available_models()
            return 0

        # Verify dataset structure
        print(f"Verifying dataset '{args.dataset}'...")
        if not verify_dataset_structure(args.dataset):
            print(
                f"Warning: Dataset '{args.dataset}' has some issues. Proceeding anyway.")

        # Get paths and class mappings
        paths = get_data_paths(args.dataset)
        mappings = get_class_mappings(args.dataset)

        # Print class mappings
        print('Index to Label Mapping:', mappings['Idx2Label'])
        print('Label to Index Mapping:', mappings['Label2Index'])

        # Visualize training data if requested
        if args.visualize_data:
            visualize_image_with_annotation_bboxes(
                paths['train_images'],
                paths['train_labels'],
                mappings['Idx2Label']
            )

        # Set font properties
        setup_font_properties()

        # Get image dimensions
        try:
            height, width, channels = get_image_dimensions(
                paths['train_images'])
            print(
                f'The image has dimensions {height}x{width} and {channels} channels')
        except Exception as e:
            print(f"Error getting image dimensions: {e}")
            print("Using default dimensions: 640x640x3")
            height, width, channels = 640, 640, 3

        # Create or verify YAML file
        yaml_path = create_yaml_file(args.dataset)
        print(f"Using YAML file: {yaml_path}")
        print("Note: Using absolute paths in YAML file to avoid path duplication issues")

        # Parse device setting
        if args.device:
            if args.device.lower() == 'cpu':
                device = 'cpu'
                print("Using CPU for computation")
            else:
                try:
                    # Handle comma-separated GPU IDs (for DataParallel)
                    if ',' in args.device:
                        device_ids = [int(x) for x in args.device.split(',')]
                        device = f'cuda:{device_ids[0]}'  # Primary GPU
                        print(f"Using GPUs: {device_ids}")
                    else:
                        device = f'cuda:{int(args.device)}'
                        print(f"Using GPU: {device}")
                except ValueError:
                    print(
                        f"Invalid device specification: {args.device}. Using default.")
                    device = None
        else:
            device = None  # Auto-detect

        # Initialize the model loader
        model_loader = ModelLoader(
            device=device,
            arch=args.arch,
            pretrained=True,
            fp16=args.fp16
        )

        # Run based on mode
        if args.mode in ['train', 'all']:
            # Count the number of classes from the dataset yaml
            num_classes = len(mappings['classes'])
            print(f"Training with {num_classes} classes")

            # Recommend smaller model if using YOLOv8x (only if not fast mode)
            if not args.fast_mode and args.arch == 'yolov8' and args.depth_or_size == 'x':
                print("\n⚠️ WARNING: YOLOv8x is very large and training will be slow.")
                print(
                    "Consider using a smaller model with --depth n/s/m for faster training")
                print("or use --fast mode for optimized training settings.")
                print(
                    "Training will continue with YOLOv8x as requested, but may take a long time.\n")
                if torch.cuda.is_available():
                    mem_free = torch.cuda.get_device_properties(
                        0).total_memory / (1024**3)
                    print(f"Available GPU memory: {mem_free:.2f} GB\n")

                # Adjust parameters for better performance with large model
                workers = min(args.workers * 2, os.cpu_count() or 8)
                amp = True
            else:
                workers = args.workers
                amp = args.fp16

            # Get model from the model loader
            model_info = model_loader.get_model(
                depth_or_size=args.depth_or_size,
                num_classes=num_classes,
                input_channels=channels
            )

            # Unpack the model and its name from the model_info
            model, model_name = model_info[0]

            # Extract the architecture and size from model_name (e.g., "yolov8_n" -> "yolov8n")
            if "_" in model_name:
                arch, size = model_name.split("_", 1)
                clean_model_name = f"{arch}{size}"
            else:
                clean_model_name = model_name

            print(f"Model loaded: {clean_model_name}")

            # Train the model
            print(
                f"Starting training for {args.epochs} epochs with batch size {args.batch}")
            trained_model = train_model(
                yaml_path,
                (height, width, channels),
                model=model,
                epochs=args.epochs,
                batch=args.batch,
                workers=workers,
                dataset_name=args.dataset,  # Pass dataset name separately
                model_name=clean_model_name,  # Pass model name separately
                device=device,
                cache=args.cache,
                amp=amp,
                lr=args.lr,
                patience=args.patience
            )

            # Visualize training metrics
            # Use the correctly structured path based on dataset and model name
            results_csv = os.path.join(
                'results', args.dataset, clean_model_name, 'results.csv')
            if os.path.exists(results_csv):
                visualize_training_metrics(results_csv_path=results_csv)
            else:
                print(f"Training results not found at {results_csv}")

            # Test if requested
            if args.mode == 'all':
                # Use new weights path format
                best_weights = os.path.join(
                    'results', args.dataset, clean_model_name,
                    'weights', f"best_{clean_model_name}_{args.dataset}_{args.epochs}_{args.batch}_{args.lr}.pt"
                )

                # Fall back to generic best.pt if specific naming doesn't exist
                if not os.path.exists(best_weights):
                    best_weights = os.path.join(
                        'results', args.dataset, clean_model_name, 'weights', 'best.pt'
                    )

                if os.path.exists(best_weights):
                    print(
                        f"Testing trained model with best weights: {best_weights}")
                    metrics = run_test(best_weights, device)
                else:
                    print(f"Best weights not found at {best_weights}")

            print("Training process completed")

        # Modify test mode to use new directory structure
        if args.mode in ['test'] or (args.mode == 'all' and not os.path.exists(
                os.path.join('results', args.dataset, model_name, 'weights', 'best.pt'))):
            # Determine model path for testing
            if args.model_path:
                test_model_path = args.model_path
            else:
                # Try to find best weights based on new structure
                clean_model_name = args.arch + (args.depth_or_size or "n")

                # Try specific naming first
                best_weights = os.path.join(
                    'results', args.dataset, clean_model_name,
                    'weights', f"best_{clean_model_name}_{args.dataset}_{args.epochs}_{args.batch}_{args.lr}.pt"
                )

                # Fall back to generic best.pt
                if not os.path.exists(best_weights):
                    best_weights = os.path.join(
                        'results', args.dataset, clean_model_name, 'weights', 'best.pt'
                    )

                if os.path.exists(best_weights):
                    test_model_path = best_weights
                else:
                    print("No model path specified and no trained model found.")
                    print(
                        "Please specify a model path with --model-path or train a model first.")
                    return 1

            # Check if model exists
            if not os.path.exists(test_model_path):
                print(f"Error: Model not found at {test_model_path}")
                return 1

            print(f"Starting model evaluation using {test_model_path}")
            metrics = run_test(test_model_path, device)
            print("Testing completed successfully")

        print("Processing complete!")

        # Final check and relocation of any model files
        relocate_model_files()

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
