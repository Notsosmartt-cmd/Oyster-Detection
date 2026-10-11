"""Compare full-precision vs ternary YOLO11 models on the same split.

    python YoloCompareTernary.py                       # sizes n s m l x, splits val+test
    python YoloCompareTernary.py --sizes n s --splits test

Looks for, per size:
    FP32   oysterTrainedModels/yolo11<size>MergedData960/weights/best.pt
    ternary oysterTrainedModels/yolo11<size>Ternary<policy><Mode>/weights/best_baked.pt   (override: --ternary-glob)
Missing models are skipped. Writes evaluation_results/ternary/ternary_comparison.csv.
"""
import argparse
import csv
import glob
import os

from ultralytics import YOLO

import ternary_yolo as T

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", nargs="+", default=list("nsmlx"))
ap.add_argument("--splits", nargs="+", default=["val", "test"])
ap.add_argument("--data", default="MergedData/data.yaml")
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--project", default="oysterTrainedModels")
ap.add_argument("--ternary-glob", default="yolo11{size}Ternary*",
                help="run-folder pattern per size; every match is evaluated")
ap.add_argument("--out", default="evaluation_results/ternary/ternary_comparison.csv")
args = ap.parse_args()


def evaluate(path, split):
    m = YOLO(path)
    r = m.val(data=args.data, split=split, imgsz=args.imgsz, batch=args.batch, verbose=False,
              plots=False, project="evaluation_results/ternary/val_runs", name="tmp", exist_ok=True)
    return m, r.box.map50, r.box.map, r.box.mp, r.box.mr


def candidates(size):
    yield "fp32", f"{args.project}/yolo11{size}MergedData960/weights/best.pt"
    for d in sorted(glob.glob(os.path.join(args.project, args.ternary_glob.format(size=size)))):
        yield os.path.basename(d), os.path.join(d, "weights", "best_baked.pt")  # QAT runs
    for d in sorted(glob.glob(os.path.join(args.project, f"yolo11{size}CatQ*"))):
        if not d.endswith("_smoke"):
            yield os.path.basename(d), os.path.join(d, "weights", "catq_baked.pt")  # CAT-Q runs


rows = []
if __name__ == "__main__":
    for size in args.sizes:
        for label, path in candidates(size):
            if not os.path.exists(path):
                print(f"skip {label}: {path} not found")
                continue
            for split in args.splits:
                m, map50, map5095, p, r = evaluate(path, split)
                rep = T.ternary_report(m.model)
                rows.append({
                    "size": size, "run": label, "split": split,
                    "mAP50": round(map50, 4), "mAP50-95": round(map5095, 4),
                    "precision": round(p, 4), "recall": round(r, 4),
                    "ternary_layers": f"{rep['ternary_layers']}/{rep['conv_layers']}",
                    "ternary_pct_of_conv_weights": round(rep["ternary_pct"], 1),
                    "zero_pct_of_ternary": round(rep["zero_pct_of_ternary"], 1),
                    "file_mb": round(os.path.getsize(path) / 1e6, 1),
                    "packed_mb_est": round(rep["packed_mb_est"], 1) if rep["ternary_layers"] else "",
                })
                print(rows[-1])

    if rows:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")
    else:
        print("nothing evaluated")
