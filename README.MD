# Enhanced Sonar Scan Image Detection and Classification with a Modified YOLOv8-x Algorithm

## Introduction

This project implements a modified YOLOv8-x algorithm to enhance the effectiveness and precision of sonar scan image detection and classification. Traditional approaches struggle with the complexities of underwater environments, low resolution, and limited data samples. Our approach incorporates a lightweight FasterNet into the YOLOv8-x backbone, streamlines the fusion process, and introduces a faster and lighter C2f-Em-Fast block.

## Dataset Setup

1. Download the Seabed Objects-KLSG dataset from [dataset](https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset)
2. Process the dataset as described in our paper
3. Place the dataset in the following structure:

```
Sonarimage/
└── dataset/
    └── seabedok/
        ├── data.yaml
        ├── train/
        ├── valid/
        └── test/
```

## Usage

### Training the Model

To train the model with the default configuration (YOLOv8-x with C2f-Em-Fast block):

```bash
python main.py --data seabedok --epochs 300 --train_batch 16 --lr 0.002 --gpu_ids 0 --arch yolov8 --depth x-em-fast
```

### Model Size Options

You can experiment with different model sizes by changing the `--depth` parameter:

```bash
# Nano model (smallest)
python main.py --data seabedok --epochs 300 --train_batch 8 --lr 0.001 --gpu_ids 0 --arch yolov8 --depth n

# Medium model
python main.py --data seabedok --epochs 300 --train_batch 8 --lr 0.001 --gpu_ids 0 --arch yolov8 --depth m

# Large model
python main.py --data seabedok --epochs 300 --train_batch 8 --lr 0.001 --gpu_ids 0 --arch yolov8 --depth l

# X-Large model with our C2f-Em-Fast block (best performance)
python main.py --data seabedok --epochs 300 --train_batch 8 --lr 0.001 --gpu_ids 0 --arch yolov8 --depth x-em-fast
```

## Requirements

[List any dependencies or requirements here]

## Citation

If you use this work in your research, please cite:

```
[Citation information for the paper]
```
