import os
from ultralytics import YOLO

# Base Configuration (adjust as needed)
TRAIN_CONFIG = {
    "data": "REU_Oyster_2024_Improved-2/data.yaml",
    "epochs": 200,
    "imgsz": 960,  # Default image size
    "batch": 16,
    "patience": 20,
    "amp": True,
    "optimize": True,
    "project": "oysterTrainedModels",
    "exist_ok": True
}

# Get all .pt models in the models folder
model_dir = "modelsM"
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]

if not model_files:
    print(f"No .pt models found in {model_dir}!")
    exit()

print(f"Found {len(model_files)} models to train:")
for i, model_file in enumerate(model_files, 1):
    model_path = os.path.join(model_dir, model_file)
    model_name = os.path.splitext(model_file)[0]  # Extract model name without extension

    print(f"\n{'=' * 50}")
    print(f"TRAINING MODEL {i}/{len(model_files)}: {model_name}")
    print(f"{'=' * 50}")

    # Clone base config and adjust imgsz for 'yollo11m'
    model_config = TRAIN_CONFIG.copy()
    if model_name == 'yolo11m':
        model_config['imgsz'] = 640  # Override image size for this specific model

    try:
        # Load model
        model = YOLO(model_path)

        # Train with unique run name
        results = model.train(
            **model_config,
            name=model_name  # Saves runs in oysterTrainedModels/model_name
        )

        print(f"\n✅ Successfully trained {model_name}")
    except Exception as e:
        print(f"\n❌ Training failed for {model_name}: {str(e)}")

print("\nAll models processed!")