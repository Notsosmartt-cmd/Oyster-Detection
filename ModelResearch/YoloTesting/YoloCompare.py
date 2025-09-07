from ultralytics import YOLO
import pandas as pd
import os

"""
  Nick's version of model comparison. His setup requires explicit naming of each model with the names = [] variable
  This can cause accidental mixing between the models and their actual name. 
  The s and n sized model names were switched so incorrect data was shown
"""

##Validate Your Model(s) on testing data

# List of model paths (update with yours)
model_paths = [
    "oysterTrainedModels/yolo11mOyster/weights/best.pt",
    "oysterTrainedModels/yolo11nOyster/weights/best.pt",
    "oysterTrainedModels/yolo11sOyster/weights/best.pt"
]

# Location of your data.yaml file
# Swap as needed, full path as was just copied/pasted from file explorer
data_yaml = "REU_Oyster_2024_Improved-2/data.yaml"


results_list = []
count = 0
names = ["YOLOv11m", "YOLOv11s", "YOLOv11n"]
for path in model_paths:
    print(f"Validating model: {path}")
    model = YOLO(path)
    results = model.val(data=data_yaml, save=False)

    results_list.append({
        "Run Name": names[count],
        "Precision": round(results.box.p.mean().item(), 3),
        "Recall": round(results.box.r.mean().item(), 3),
        "mAP@0.5": round(results.box.map50.mean().item(), 3),
        "mAP@0.5:0.95": round(results.box.map.mean().item(), 3)
    })
    count += 1

# Create and display DataFrame to neatly display metrics for all 3 models
df = pd.DataFrame(results_list)
df_sorted = df.sort_values(by="mAP@0.5", ascending=False).reset_index(drop=True)
print("\n📊 Model Comparison:")
print(df_sorted)
