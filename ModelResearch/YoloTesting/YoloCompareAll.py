import multiprocessing
from multiprocessing import freeze_support

from ultralytics import YOLO
import pandas as pd
import os
from datetime import datetime

# Configuration
RESULTS_DIR = "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
"""
Updated Yolo model Comparison.
Old Nick uses Nick's model training configuration, but transferred to yolo11
New Merged Uses the Merged data from Nick and Other groups with an updated training config
New Nick uses a new training configuration with only Nick's

Old Nick used a 640 image resolution for m sized models due to memory constraints
New Merged and New Nick uses 960 resolution for all models including m sized, but downsizes the batch size
    for m models to alleviate memory.

All models are sorted by the dataset used and mAP@0.5 in ascending order

Added Speed Metrics:
- Preprocess: Image preprocessing time (ms)
- Inference: Model inference time (ms)
- Postprocess: Result processing time (ms)
- Total: Total processing time per image (ms)
- FPS: Frames processed per second
"""


def evaluate_models(models, data_yaml, deviceChoice='cuda'):
    """Evaluate all models on a single dataset and return results DataFrame."""
    results_list = []
    dataset_name = os.path.basename(os.path.dirname(data_yaml))

    print(f"\n{'=' * 50}")
    print(f"Evaluating models on dataset: {dataset_name}")
    print(f"{'=' * 50}")

    for path in models:
        # Extract model name from path - FIXED
        model_type = os.path.basename(os.path.dirname(os.path.dirname(path)))
        # Replace 'yolo' with 'YOLOv' to preserve full identifier
        model_name = model_type.replace('yolo', 'YOLOv')

        print(f"\nValidating: {model_name} ({path})")
        model = YOLO(path)
        model = model.to(deviceChoice)
        try:
            # Run validation and get detailed results
            results = model.val(data=data_yaml, save=False,device=deviceChoice)

            # Extract speed metrics (in milliseconds)
            preprocess_time = results.speed.get('preprocess', 0)
            inference_time = results.speed.get('inference', 0)
            postprocess_time = results.speed.get('postprocess', 0)
            #Compute total time per image:
            total_time = preprocess_time + inference_time + postprocess_time

            # Calculate FPS
            fps = 1000 / total_time if total_time > 0 else 0

            results_list.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Precision": round(results.box.p.mean().item(), 3),
                "Recall": round(results.box.r.mean().item(), 3),
                "mAP@0.5": round(results.box.map50.mean().item(), 3),
                "mAP@0.5:0.95": round(results.box.map.mean().item(), 3),
                "Preprocess (ms/img)": round(preprocess_time, 2),
                "Inference (ms/img)": round(inference_time, 2),
                "Postprocess (ms/img)": round(postprocess_time, 2),
                "Total (ms/img)": round(total_time, 2),
                "FPS": round(fps, 1)
            })
            # Print success message with key metrics
            print(
                f"✅ Validation successful! P: {results_list[-1]['Precision']}, R: {results_list[-1]['Recall']}, "
                f"mAP50: {results_list[-1]['mAP@0.5']}, FPS: {results_list[-1]['FPS']}")
        except Exception as e:
            print(f"❌ Validation failed for {model_name}: {str(e)}")

    return pd.DataFrame(results_list)

def main():

    # List of model paths to evaluate
    model_paths = [
    "oysterTrainedModels/yolo11mMergedData960/weights/best.pt",  # NewMerged
    "oysterTrainedModels/yolo11nMergedData960/weights/best.pt",
    "oysterTrainedModels/yolo11sMergedData960/weights/best.pt",
    "oysterTrainedModels/yolo11mNickData960/weights/best.pt",  # NewNick
    "oysterTrainedModels/yolo11nNickData960/weights/best.pt",
    "oysterTrainedModels/yolo11sNickData960/weights/best.pt",
    "oysterTrainedModels/yolo8mMerged/weights/best.pt",  # Yolo8Merged
    "oysterTrainedModels/yolo8sMerged/weights/best.pt",
    "oysterTrainedModels/yolo8nMerged/weights/best.pt",
    "oysterTrainedModels/yolo8mNick/weights/best.pt",  # Yolo8Nick
    "oysterTrainedModels/yolo8sNick/weights/best.pt",
    "oysterTrainedModels/yolo8nNick/weights/best.pt",
    "oysterTrainedModels/yolo8nMerged8Batch/weights/best.pt", #yolotest
    "oysterTrainedModels/yolo8nMerged16Batch/weights/best.pt",
    "oysterTrainedModels/yolo8sMerged8Batch/weights/best.pt"

]

    # List of datasets to evaluate on
    data_yaml_files = [
    "REU_Oyster_2024_Improved-2/data.yaml",
    "MergedData/data.yaml"
]

# Process all datasets
    all_results = []
    dataset_results = []  # Store results for each dataset

    for data_yaml in data_yaml_files:
        if not os.path.exists(data_yaml):
            print(f"⚠️ Dataset not found: {data_yaml}")
            continue

        df = evaluate_models(model_paths, data_yaml)
        all_results.append(df)

        # Store dataset name and sorted results for later printing
        dataset_name = os.path.basename(os.path.dirname(data_yaml))
        sorted_df = df.sort_values(by="mAP@0.5:0.95", ascending=False)
        dataset_results.append((dataset_name, sorted_df))

# Print all dataset results at the end
    for dataset_name, sorted_df in dataset_results:
        print(f"\n📊 RESULTS FOR {dataset_name} (sorted by mAP@0.5:0.95):")
        print(sorted_df.to_string(index=False))
        print("-" * 80)

# Combine and save all results to single file
    if all_results:
        master_df = pd.concat(all_results).sort_values(
         by=["Dataset", "mAP@0.5:0.95"],
            ascending=[True, False]
    )

    # Create timestamped filename in dedicated folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(RESULTS_DIR, f"model_comparison_{timestamp}.csv")

        # Save to CSV
        master_df.to_csv(csv_filename, index=False)
        print(f"\n🔥 All results saved to: {os.path.abspath(csv_filename)}")

        # Print final master table
        print("\n🌟 FINAL COMPARISON ACROSS ALL DATASETS:")
        print(master_df.to_string(index=False))

        # Print speed comparison
        print("\n🚀 SPEED COMPARISON (sorted by FPS):")
        speed_df = master_df.sort_values(by="FPS", ascending=False)
        print(speed_df[["Dataset", "Model", "FPS", "Total (ms/img)", "Inference (ms/img)"]].to_string(index=False))
    else:
        print("\n❌ No results generated (check dataset paths).")



if __name__=='__main__':
    multiprocessing.freeze_support()
    main()

