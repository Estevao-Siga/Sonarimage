import os
import random
import cv2
import matplotlib.pyplot as plt
import yaml
import shutil
from pathlib import Path


class DatasetRegistry:
    """Registry for managing multiple datasets"""

    # Dictionary of supported datasets with their metadata
    DATASETS = {
        'seabedok': {
            'classes': ['plane', 'wreck ship'],
            'description': 'Seabed object detection dataset',
            'default_path': 'dataset/seabedok'
        },


    }

    @classmethod
    def get_dataset_names(cls):
        """Get list of all registered datasets"""
        return list(cls.DATASETS.keys())

    @classmethod
    def get_dataset_info(cls, dataset_name):
        """Get information about a specific dataset"""
        if dataset_name not in cls.DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' is not registered. Available datasets: {', '.join(cls.get_dataset_names())}")

        return cls.DATASETS[dataset_name]

    @classmethod
    def add_dataset(cls, name, classes, description, default_path):
        """Add a new dataset to the registry"""
        if name in cls.DATASETS:
            raise ValueError(f"Dataset '{name}' already exists in registry")

        cls.DATASETS[name] = {
            'classes': classes,
            'description': description,
            'default_path': default_path
        }

        print(f"Added new dataset '{name}' with {len(classes)} classes")
        return cls.DATASETS[name]


def get_data_paths(dataset_name='seabedok', base_path=None):
    """
    Define the paths to the images and labels directories for the specified dataset

    Args:
        dataset_name: Name of the dataset to use
        base_path: Base directory for the dataset, overrides the default if provided

    Returns:
        Dictionary with dataset paths
    """
    # Get dataset info from registry
    try:
        dataset_info = DatasetRegistry.get_dataset_info(dataset_name)
    except ValueError as e:
        print(f"Error: {e}")
        print("Defaulting to 'seabedok' dataset")
        dataset_info = DatasetRegistry.get_dataset_info('seabedok')
        dataset_name = 'seabedok'

    # Determine base path
    if base_path is None:
        base_path = dataset_info['default_path']

    # Ensure the dataset directory exists
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset directory not found: {base_path}")

    # Construct all necessary paths
    train_images = os.path.join(base_path, 'train', 'images')
    train_labels = os.path.join(base_path, 'train', 'labels')

    val_images = os.path.join(base_path, 'valid', 'images')
    val_labels = os.path.join(base_path, 'valid', 'labels')

    test_images = os.path.join(base_path, 'test', 'images')
    test_labels = os.path.join(base_path, 'test', 'labels')

    yaml_path = os.path.join(base_path, 'data.yaml')

    # Validate directories exist
    paths = {
        'base_path': base_path,
        'train_images': train_images,
        'train_labels': train_labels,
        'val_images': val_images,
        'val_labels': val_labels,
        'test_images': test_images,
        'test_labels': test_labels,
        'yaml_path': yaml_path
    }

    # Check if directories exist and report missing ones
    missing_dirs = []
    for path_name, path_value in paths.items():
        if 'images' in path_name or 'labels' in path_name:
            if not os.path.exists(path_value):
                missing_dirs.append(path_name)

    if missing_dirs:
        print(
            f"Warning: The following directories are missing for dataset '{dataset_name}':")
        for dir_name in missing_dirs:
            print(f"  - {dir_name}: {paths[dir_name]}")

    return paths


def get_class_mappings(dataset_name='seabedok'):
    """Get class mappings for the specified dataset"""
    try:
        dataset_info = DatasetRegistry.get_dataset_info(dataset_name)
        classes = dataset_info['classes']
    except ValueError:
        print(f"Dataset '{dataset_name}' not found. Using default classes.")
        classes = ['unknown']

    Idx2Label = {idx: label for idx, label in enumerate(classes)}
    Label2Index = {label: idx for idx, label in Idx2Label.items()}

    return {
        'classes': classes,
        'Idx2Label': Idx2Label,
        'Label2Index': Label2Index
    }


def visualize_image_with_annotation_bboxes(image_dir, label_dir, Idx2Label, num_samples=12, dataset_name=None):
    """
    Visualize sample images with corresponding annotation bounding boxes

    Args:
        image_dir: Directory containing the images
        label_dir: Directory containing the label files
        Idx2Label: Dictionary mapping class indices to class names
        num_samples: Number of sample images to visualize
        dataset_name: Dataset name for saving results
    """
    # Determine where to save visualizations
    if dataset_name:
        viz_dir = os.path.join('results', dataset_name,
                               'dataset_visualization')
    else:
        viz_dir = os.path.join('results', 'dataset_visualization')
    os.makedirs(viz_dir, exist_ok=True)

    # Check if directories exist
    if not os.path.exists(image_dir):
        print(f"Warning: Image directory not found at {image_dir}")
        return

    if not os.path.exists(label_dir):
        print(f"Warning: Label directory not found at {label_dir}")
        return

    # Get list of all the image files in the directory
    try:
        image_files = sorted(os.listdir(image_dir))
    except Exception as e:
        print(f"Error reading image directory: {e}")
        return

    # Choose random image files or less if there aren't enough
    sample_count = min(num_samples, len(image_files))
    if sample_count == 0:
        print(f"No images found in {image_dir}")
        return

    sample_image_files = random.sample(image_files, sample_count)

    # Set up the plot
    rows = (sample_count + 2) // 3  # Calculate rows needed
    fig, axs = plt.subplots(rows, 3, figsize=(15, 5*rows))

    # Handle case with a single row
    if rows == 1:
        axs = [axs]  # Make axs indexable for single row

    # Loop over the random images and plot the bounding boxes
    for i, image_file in enumerate(sample_image_files):
        row = i // 3
        col = i % 3

        # Load the image
        image_path = os.path.join(image_dir, image_file)
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load the labels for this image
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)

            if not os.path.exists(label_path):
                print(f"Warning: Label file not found: {label_path}")
                axs[row][col].imshow(image)
                axs[row][col].set_title(f"{image_file} (No labels)")
                axs[row][col].axis('off')
                continue

            with open(label_path, 'r') as f:
                # Loop over the labels and plot the bounding boxes
                for label in f:
                    parts = label.split()
                    if len(parts) != 5:
                        print(f"Warning: Invalid label format in {label_path}")
                        continue

                    class_id, x_center, y_center, width, height = map(
                        float, parts)
                    h, w, _ = image.shape
                    x_min = int((x_center - width/2) * w)
                    y_min = int((y_center - height/2) * h)
                    x_max = int((x_center + width/2) * w)
                    y_max = int((y_center + height/2) * h)

                    # Ensure coordinates are within image bounds
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)

                    # Draw bounding box
                    cv2.rectangle(image, (x_min, y_min),
                                  (x_max, y_max), (0, 255, 0), 2)

                    # Add label if class_id is valid
                    if int(class_id) in Idx2Label:
                        cv2.putText(image, Idx2Label[int(class_id)],
                                    (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1, color=(255, 255, 255), thickness=2)
                    else:
                        print(
                            f"Warning: Invalid class ID {int(class_id)} in {label_path}")

            axs[row][col].imshow(image)
            axs[row][col].axis('off')

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")

    # Hide any unused subplots
    for i in range(sample_count, rows * 3):
        row = i // 3
        col = i % 3
        axs[row][col].axis('off')

    # Save the visualization
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'data_visualization.png'))
    plt.show()


def get_image_dimensions(image_dir, sample_idx=None):
    """
    Get dimensions of a sample image from the directory

    Args:
        image_dir: Directory containing images
        sample_idx: Index of the sample image to use, random if None

    Returns:
        Tuple (height, width, channels) of the image
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = os.listdir(image_dir)
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    # Choose a specific image or a random one
    if sample_idx is not None and sample_idx < len(image_files):
        image_path = os.path.join(image_dir, image_files[sample_idx])
    else:
        # Try multiple images in case some are corrupted
        for i in range(min(10, len(image_files))):
            idx = random.randint(0, len(image_files)-1)
            image_path = os.path.join(image_dir, image_files[idx])
            image = cv2.imread(image_path)
            if image is not None:
                height, width, channels = image.shape
                return height, width, channels

    # If we get here, we couldn't find a valid image in our random samples
    # Try sequentially
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            height, width, channels = image.shape
            return height, width, channels

    # If we still can't find a valid image
    raise ValueError(f"Could not read any valid images from {image_dir}")


def create_yaml_file(dataset_name='seabedok', output_dir=None):
    """
    Create YAML file for training based on dataset info

    Args:
        dataset_name: Name of the dataset to use
        output_dir: Output directory for the YAML file, uses dataset dir if None

    Returns:
        Path to the created YAML file
    """
    try:
        dataset_info = DatasetRegistry.get_dataset_info(dataset_name)
        classes = dataset_info['classes']
    except ValueError as e:
        print(f"Error: {e}")
        print("Defaulting to 'seabedok' dataset")
        dataset_info = DatasetRegistry.get_dataset_info('seabedok')
        classes = dataset_info['classes']
        dataset_name = 'seabedok'

    # Get absolute paths to avoid path duplication issues
    base_path = os.path.abspath(dataset_info['default_path'])
    train_path = os.path.join(base_path, 'train', 'images')
    val_path = os.path.join(base_path, 'valid', 'images')
    test_path = os.path.join(base_path, 'test', 'images')

    # Verify that these paths exist
    for path, name in [(train_path, "Training"), (val_path, "Validation"), (test_path, "Test")]:
        if not os.path.exists(path):
            print(f"Warning: {name} images directory not found at {path}")

    # Set output path
    if output_dir is None:
        output_path = os.path.join(base_path, 'data.yaml')
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{dataset_name}_data.yaml')

    # Create yaml content with absolute paths
    data = {
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': len(classes),
        'names': classes
    }

    # Write yaml file
    with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"Created YAML file at: {output_path}")
    print(f"Using absolute paths in YAML file to avoid path duplication issues")
    return output_path


def verify_dataset_structure(dataset_name='seabedok'):
    """
    Verify that the dataset has the expected directory structure and contains data

    Args:
        dataset_name: Name of the dataset to verify

    Returns:
        True if dataset structure is valid, False otherwise
    """
    try:
        paths = get_data_paths(dataset_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

    # Check if all necessary directories exist and contain data
    required_dirs = [
        ('train_images', 'Training images'),
        ('train_labels', 'Training labels'),
        ('val_images', 'Validation images'),
        ('val_labels', 'Validation labels'),
        ('test_images', 'Test images'),
        ('test_labels', 'Test labels')
    ]

    all_valid = True
    for dir_key, dir_desc in required_dirs:
        dir_path = paths[dir_key]
        if not os.path.exists(dir_path):
            print(f"Warning: {dir_desc} directory not found at {dir_path}")
            all_valid = False
            continue

        files = os.listdir(dir_path)
        if not files:
            print(f"Warning: {dir_desc} directory is empty at {dir_path}")
            all_valid = False

    # Check if image and label files correspond
    for img_dir_key, lbl_dir_key in [('train_images', 'train_labels'),
                                     ('val_images', 'val_labels'),
                                     ('test_images', 'test_labels')]:
        img_dir = paths[img_dir_key]
        lbl_dir = paths[lbl_dir_key]

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue

        img_files = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
        lbl_files = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir)}

        missing_labels = img_files - lbl_files
        if missing_labels:
            print(
                f"Warning: {len(missing_labels)} images in {img_dir_key} have no corresponding labels")
            if len(missing_labels) < 10:
                print(f"Missing labels for: {', '.join(list(missing_labels))}")
            all_valid = False

    return all_valid


def list_available_datasets():
    """List all available datasets with their descriptions"""
    print("\nAvailable Datasets:")
    print("=" * 60)
    print(f"{'Dataset Name':<15} | {'Classes':<25} | {'Description':<20}")
    print("-" * 60)

    for name, info in DatasetRegistry.DATASETS.items():
        classes_str = ", ".join(info['classes'])
        if len(classes_str) > 23:
            classes_str = classes_str[:20] + "..."

        print(f"{name:<15} | {classes_str:<25} | {info['description']:<20}")

    print("=" * 60)


if __name__ == "__main__":
    # Use the centralized argument parser
    from arg_parser import create_argument_parser

    # Create a parser with our base arguments
    parser = create_argument_parser()

    # Add data-specific arguments
    data_group = parser.add_argument_group('Dataset Tools')
    data_group.add_argument('--list', action='store_true',
                            help='List all available datasets')
    data_group.add_argument('--verify', metavar='DATASET',
                            help='Verify dataset structure')
    data_group.add_argument('--visualize', metavar='DATASET',
                            help='Visualize dataset samples')
    data_group.add_argument('--create-yaml', metavar='DATASET',
                            help='Create YAML file for dataset')

    args = parser.parse_args()

    if args.list or args.list_datasets:
        list_available_datasets()

    if args.verify:
        result = verify_dataset_structure(args.verify)
        if result:
            print(f"Dataset '{args.verify}' structure is valid")
        else:
            print(f"Dataset '{args.verify}' has structural issues")

    if args.visualize:
        paths = get_data_paths(args.visualize)
        mappings = get_class_mappings(args.visualize)
        visualize_image_with_annotation_bboxes(
            paths['train_images'],
            paths['train_labels'],
            mappings['Idx2Label']
        )

    if args.create_yaml:
        yaml_path = create_yaml_file(args.create_yaml)
        print(f"Created YAML file at: {yaml_path}")
