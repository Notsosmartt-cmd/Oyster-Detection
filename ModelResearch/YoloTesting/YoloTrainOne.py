# This program trains a basic yolo model into our own custom model after its been trained
from ultralytics import YOLO

# Configuration (adjust as needed)
TRAIN_CONFIG = {
    "data": "REU_Oyster_2024_Improved-2/data.yaml",
    "epochs": 200,
    "imgsz": 960,
    "batch": 16,
    "patience": 20,
    "amp": True,
    "optimize": True,
    "project": "oysterTrainedModels",
    "exist_ok": True
}

# Load model
model_path = "models/yolo11m.pt"
model_name= "yolo11mOyster"
model = YOLO(model_path)

# Train with unique run name
results = model.train(
    **TRAIN_CONFIG,
    name=model_name  # Saves runs in oysterTrainedModels/model_name
)