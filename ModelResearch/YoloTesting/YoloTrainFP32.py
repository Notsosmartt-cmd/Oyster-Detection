"""Full-precision oyster baseline for one YOLO11 size, same naming as the existing runs
(oysterTrainedModels/yolo11<size>MergedData960). Needed for l and x, which have no baseline yet.

    python YoloTrainFP32.py --size l
    python YoloTrainFP32.py --size x --batch 4
"""
import argparse
import os
import sys

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=list("nsmlx"))
    ap.add_argument("--data", default="MergedData/data.yaml")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=None, help="default 16 (n,s), 8 (m,l), 4 (x); -1 = auto (uses ~60% of VRAM)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    batch = args.batch if args.batch is not None else {"n": 16, "s": 16, "m": 8, "l": 8, "x": 4}[args.size]
    weights = f"models/yolo11{args.size}.pt"
    name = f"yolo11{args.size}MergedData960"
    if not os.path.exists(weights):
        sys.exit(f"{weights} not found")
    print(f"weights={weights} name={name} data={args.data} epochs={args.epochs} batch={batch} imgsz={args.imgsz}")
    if args.dry_run:
        return

    YOLO(weights).train(
        data=args.data, epochs=args.epochs, patience=args.patience, imgsz=args.imgsz, batch=batch,
        workers=args.workers, seed=0, amp=True, project="oysterTrainedModels", name=name, exist_ok=False,
    )


if __name__ == "__main__":  # required on Windows (dataloader workers re-import this file)
    main()
