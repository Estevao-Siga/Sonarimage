

```bash
# . List available datasets
python main.py --list-datasets

#  Train a model using the default dataset (seabedok)
python main.py --mode train --batch 8 --epochs 100 --model yolov8n.pt

# Train a model with a specific dataset
python main.py --mode train --dataset seabedok --batch 8 --epochs 100 --model yolov8n.pt


#  Train and test in one go
python main.py --mode all --dataset seabedok --batch 8 --epochs 100 --model yolov8n.pt

# Test a previously trained model
python main.py --mode test --dataset seabedok --model-path results/seabedok_model/weights/best.pt

#  Visualize training data before running
python main.py --mode train --dataset seabedok --visualize-data

#  Download a specific YOLOv8 model
python data_loader.py --download-model yolov8n.pt
```