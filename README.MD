# Enhanced Sonar Scan Image Detection and Classification with a Modified YOLOv8-x Algorithm

## Introduction

This project implements a modified YOLOv8-x algorithm to enhance the effectiveness and precision of sonar scan image detection and classification. Traditional approaches struggle with the complexities of underwater environments, low resolution, and limited data samples. Our approach replaces selected C2f blocks with C2f-EM-Fast, embedding FasterNet-style partial convolution (via FasterNetBlock) and efficient multi-scale attention (EMA) to reduce compute while preserving detection quality.

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

### Optional: Seabed multi-class (6 classes)

If you want to train/evaluate on the 6-class seabed dataset (fish, human, mine, plane, seafloor, ship), place it in YOLO format under:

```
Sonarimage/
└── dataset/
    └── seabed/
        ├── data.yaml
        ├── train/
        ├── valid/
        └── test/
```

Public sources used to assemble the multi-class dataset (see manuscript for details):

- https://universe.roboflow.com/yoondongjin/sonar-bdxkj
- https://universe.roboflow.com/dae-hyeok-lee/side-scan-sonar

## Usage

### Training the Model

To train the model with the default configuration (YOLOv8-x with C2f-Em-Fast block):

```bash
python main.py --data seabedok --epochs 300 --train_batch 16 --lr 0.002 --gpu_ids 0 --arch yolov8 --depth x-em-fast
```

To train on the 6-class seabed dataset:

```bash
python main.py --data seabed --epochs 300 --train_batch 16 --lr 0.002 --gpu_ids 0 --arch yolov8 --depth x-em-fast
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

- Python 3.10+ recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Citation

If you use this work in your research, please cite:

```
[Citation information for the paper to be added]
```
