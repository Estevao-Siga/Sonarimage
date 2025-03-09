import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from PIL import Image
import torch
from models.model_loader import load_model


def evaluate_model(model, conf=0.25, split='test', dataset_name=None, model_name=None):
    """Evaluate the model on the test dataset"""
    # Determine output directory
    if dataset_name and model_name:
        eval_dir = os.path.join('results', dataset_name,
                                model_name, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        metrics_file = os.path.join(eval_dir, 'metrics.txt')
    else:
        # Fallback for backward compatibility
        eval_dir = os.path.dirname(os.path.dirname(model.ckpt))
        metrics_file = os.path.join(eval_dir, 'metrics.txt')

    # Evaluate the model
    metrics = model.val(conf=conf, split=split)

    # Save metrics to file
    with open(metrics_file, 'w') as f:
        f.write(f"Mean Average Precision @.5:.95 : {metrics.box.map}\n")
        f.write(f"Mean Average Precision @ .50   : {metrics.box.map50}\n")
        f.write(f"Mean Average Precision @ .75   : {metrics.box.map75}\n")
        f.write(f"Precision                      : {metrics.box.precision}\n")
        f.write(f"Recall                         : {metrics.box.recall}\n")

    return metrics


def visualize_metrics(metrics, dataset_name=None, model_name=None):
    """Visualize evaluation metrics"""
    # Determine output directory
    if dataset_name and model_name:
        eval_dir = os.path.join('results', dataset_name,
                                model_name, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
    else:
        # For backward compatibility, use a temporary directory
        import tempfile
        eval_dir = tempfile.mkdtemp(prefix="eval_metrics_")

    # Check if metrics attributes exist
    try:
        mAP50 = metrics.box.map50
        mAP75 = metrics.box.map75
        precision = metrics.box.precision
        recall = metrics.box.recall
        mAP = metrics.box.map

        # First visualization - mAP50, mAP75, Precision, Recall
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(
            x=['mAP50', 'mAP75', 'Precision', 'Recall'],
            y=[mAP50, mAP75, precision, recall]
        )
        ax.set_title('YOLO Evaluation')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')

        # Add values on top of the bars
        for p in ax.patches:
            ax.annotate('{:.3f}'.format(p.get_height()),
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='bottom')

        plt.savefig(os.path.join(eval_dir, 'eval_metrics1.png'))
        plt.show()

        # Second visualization - mAP, mAP50, mAP75
        plt.figure(figsize=(8, 12))
        ax = sns.barplot(x=['mAP', 'mAP50', 'mAP75'], y=[mAP, mAP50, mAP75])
        ax.set_title('Evaluation Metrics')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')

        # Add values on top of the bars
        for p in ax.patches:
            ax.annotate('{:.3f}'.format(p.get_height()),
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='bottom')

        plt.savefig(os.path.join(eval_dir, 'eval_metrics2.png'))
        plt.show()

        # Print metrics
        print(f"Mean Average Precision @.5:.95 : {mAP}")
        print(f"Mean Average Precision @ .50   : {mAP50}")
        print(f"Mean Average Precision @ .75   : {mAP75}")

    except AttributeError as e:
        print(f"Attribute Error: {e}")
        print(dir(metrics.box))


def predict_detection(model, image_path):
    """Perform detection on a single image"""
    # Read the image
    image = cv2.imread(image_path)

    # Pass the image through the detection model and get the result
    detect_result = model(image)

    # Plot the detections
    detect_image = detect_result[0].plot()

    # Convert the image to RGB format
    detect_image = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)

    return detect_image


def visualize_original_images(test_images_dir, num_samples=12, dataset_name=None, model_name=None):
    """Visualize original test images"""
    # Determine output directory
    if dataset_name and model_name:
        viz_dir = os.path.join('results', dataset_name,
                               model_name, 'test_visualizations')
        os.makedirs(viz_dir, exist_ok=True)
    else:
        # For backward compatibility
        viz_dir = os.path.join(
            'results', dataset_name or 'unknown', 'test_visualizations')
        os.makedirs(viz_dir, exist_ok=True)

    # Get list of all the image files in the test directory
    image_files = sorted(os.listdir(test_images_dir))

    # Choose random image files from the list
    sample_image_files = random.sample(image_files, num_samples)

    # Set up the plot
    fig, axs = plt.subplots(4, 3, figsize=(12, 15))

    # Loop over the random images
    for i, image_file in enumerate(sample_image_files):
        img_path = os.path.join(test_images_dir, image_file)
        img = Image.open(img_path)

        row, col = i // 3, i % 3
        axs[row, col].imshow(img)
        axs[row, col].set_title(image_file)
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'original_images.png'))
    plt.show()


def visualize_images_with_borders(model, test_images_dir, num_samples=8, dataset_name=None, model_name=None):
    """Visualize predictions with borders in a canvas"""
    # Determine output directory
    if dataset_name and model_name:
        viz_dir = os.path.join('results', dataset_name,
                               model_name, 'test_visualizations')
        os.makedirs(viz_dir, exist_ok=True)
    else:
        # For backward compatibility
        viz_dir = os.path.join(
            'results', dataset_name or 'unknown', 'test_visualizations')
        os.makedirs(viz_dir, exist_ok=True)

    # Get list of all the image files in the test directory
    image_files = sorted(os.listdir(test_images_dir))

    # Choose random image files from the list
    sample_image_files = random.sample(image_files, num_samples)

    # Set up the canvas size
    img_height, img_width = 600, 450
    border_thickness = 1
    outer_border_thickness = 10

    # Create a blank canvas (2 rows, 4 columns)
    canvas_height = (img_height * 2) + \
        (border_thickness * 1) + outer_border_thickness
    canvas_width = (img_width * 4) + (border_thickness * 3) + \
        outer_border_thickness
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Loop over the random images and plot the detections
    for i, image_file in enumerate(sample_image_files):
        # Load the current image and run object detection
        image_path = os.path.join(test_images_dir, image_file)
        detect_image = predict_detection(model, image_path)
        detect_image = cv2.resize(detect_image, (img_width, img_height))

        # Draw a border around the individual image
        cv2.rectangle(detect_image, (0, 0), (img_width - 1,
                      img_height - 1), (0, 0, 0), border_thickness)

        # Calculate where to place the image on the canvas
        row = i // 4
        col = i % 4
        y_offset = row * (img_height + border_thickness) + \
            outer_border_thickness
        x_offset = col * (img_width + border_thickness) + \
            outer_border_thickness

        # Place the image on the canvas
        canvas[y_offset:y_offset + img_height,
               x_offset:x_offset + img_width] = detect_image

    # Draw an outer border around the entire canvas
    cv2.rectangle(canvas, (0, 0), (canvas_width - 1,
                  canvas_height - 1), (0, 0, 0), outer_border_thickness)

    # Display the final canvas
    plt.figure(figsize=(12, 8))
    plt.imshow(canvas)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'detection_canvas.png'))
    plt.show()


def visualize_predictions(model, test_images_dir, num_samples=12, dataset_name=None, model_name=None):
    """Visualize predictions on test images"""
    # Determine output directory
    if dataset_name and model_name:
        viz_dir = os.path.join('results', dataset_name,
                               model_name, 'test_visualizations')
        os.makedirs(viz_dir, exist_ok=True)
    else:
        # For backward compatibility
        viz_dir = os.path.join(
            'results', dataset_name or 'unknown', 'test_visualizations')
        os.makedirs(viz_dir, exist_ok=True)

    # Get list of all the image files in the test directory
    image_files = sorted(os.listdir(test_images_dir))

    # Choose random image files from the list
    sample_image_files = random.sample(image_files, num_samples)

    # Set up the plot
    fig, axs = plt.subplots(4, 3, figsize=(15, 20))

    # Loop over the random images and plot the detections
    for i, image_file in enumerate(sample_image_files):
        row, col = i // 3, i % 3

        # Load the current image and run object detection
        image_path = os.path.join(test_images_dir, image_file)
        detect_image = predict_detection(model, image_path)

        axs[row, col].imshow(detect_image)
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'predictions.png'))
    plt.show()


def run_test(model_path, device=None, dataset_name=None, model_name=None):
    """Run all test functions"""
    # Load model
    model = load_model(model_path, device=device)

    # Try to extract dataset and model name from path if not provided
    if dataset_name is None or model_name is None:
        path_parts = os.path.normpath(model_path).split(os.sep)
        if len(path_parts) >= 4 and path_parts[-3] == 'results':
            dataset_name = path_parts[-2]
            model_name = path_parts[-1].split(
                '_')[0] if '_' in path_parts[-1] else 'unknown'

    # Default values if extraction fails
    dataset_name = dataset_name or 'unknown_dataset'
    model_name = model_name or 'unknown_model'

    # Create test results dir within the model directory
    test_output_dir = os.path.join(
        'results', dataset_name, model_name, 'test_results')
    os.makedirs(test_output_dir, exist_ok=True)

    # Evaluate model
    metrics = evaluate_model(
        model, dataset_name=dataset_name, model_name=model_name)

    # Visualize metrics
    visualize_metrics(metrics, dataset_name=dataset_name,
                      model_name=model_name)

    # Visualize predictions
    test_images_dir = 'dataset/seabedok/test/images'
    visualize_original_images(
        test_images_dir, dataset_name=dataset_name, model_name=model_name)
    visualize_predictions(model, test_images_dir,
                          dataset_name=dataset_name, model_name=model_name)
    visualize_images_with_borders(
        model, test_images_dir, dataset_name=dataset_name, model_name=model_name)

    return metrics


if __name__ == "__main__":
    # This allows the script to be run directly for debugging or testing
    from arg_parser import parse_args

    args = parse_args()

    if args.model_path:
        print(f"Testing model from: {args.model_path}")
        metrics = run_test(args.model_path, args.device)
    else:
        # If not specified, try to find a trained model
        if args.dataset:
            best_weights = f"results/{args.dataset}_model/weights/best.pt"
            if os.path.exists(best_weights):
                print(
                    f"Found model for dataset {args.dataset}: {best_weights}")
                metrics = run_test(best_weights, args.device)
            else:
                print(f"No model found for dataset {args.dataset}")
                print("Specify a model path with --model-path")
        else:
            print("No model path specified. Use --model-path to specify a model.")
