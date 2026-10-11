"""CAT-Q ternary calibration of an oyster-trained YOLO11 model (see TERNARY_PLAN.md section 11).

Weights stay frozen; a few learnable factors per conv output channel are fitted on a small set of
training images so each ternary block reproduces the full-precision block. No labels are used.

    python YoloCatQ.py --size n --dry-run          # what would be converted, no calibration
    python YoloCatQ.py --size n --smoke            # 16 images, 2 epochs (~1 min pipeline check)
    python YoloCatQ.py --size n --eval             # real run + mAP on the val split
    python YoloCatQ.py --size x --num-images 64 --propagate

Output: oysterTrainedModels/yolo11<size>CatQ<policy>/weights/catq_baked.pt  (plain YOLO checkpoint)
Then optionally refine with QAT:  python YoloTrainTernary.py --size n --init-from-catq
"""
import argparse
import copy
import os
import sys

import torch
from ultralytics import YOLO

import catq_yolo as C
import ternary_yolo as T


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=list("nsmlx"))
    ap.add_argument("--source", default=None,
                    help="default oysterTrainedModels/yolo11<size>MergedData960/weights/best.pt")
    ap.add_argument("--data", default="MergedData/data.yaml")
    ap.add_argument("--images", default=None, help="calibration image folder (default: the data.yaml train split)")
    ap.add_argument("--num-images", type=int, default=128)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--policy", default="A", choices=["A", "B"])
    ap.add_argument("--keep-depthwise", action="store_true")
    ap.add_argument("--epochs", type=int, default=60, help="calibration epochs per block")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.8, help="fraction of steps in the softened stage")
    ap.add_argument("--s-max", type=float, default=30.0)
    ap.add_argument("--propagate", action="store_true",
                    help="feed each block the output of the already-ternary blocks (2x cache memory)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--project", default="oysterTrainedModels")
    ap.add_argument("--eval", action="store_true", help="run val on the data.yaml val split afterwards")
    ap.add_argument("--smoke", action="store_true", help="16 images, 2 epochs, imgsz 320")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    source = a.source or f"oysterTrainedModels/yolo11{a.size}MergedData960/weights/best.pt"
    if not os.path.exists(source):
        sys.exit(f"Start weights not found: {source}\nTrain the FP32 baseline first (YoloTrainFP32.py --size {a.size}).")
    if a.smoke:
        a.num_images, a.epochs, a.imgsz = 16, 2, 320
    name = a.name or f"yolo11{a.size}CatQ{a.policy}" + ("_smoke" if a.smoke else "")
    dst = os.path.join(a.project, name, "weights", "catq_baked.pt")

    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float()

    preview = T.convert(copy.deepcopy(model), a.policy, "absmean", a.keep_depthwise)
    print(f"\nsource : {source}\noutput : {dst}")
    print(f"convert: {preview['ternary_layers']}/{preview['conv_layers']} conv layers, "
          f"{preview['ternary_pct']:.1f}% of conv weights (policy {a.policy})")
    print(f"calib  : {a.num_images} images @ {a.imgsz}, {a.epochs} epochs/block, lr {a.lr}, batch {a.batch}, "
          f"gamma {a.gamma}, s_max {a.s_max}, propagate={a.propagate}, device {a.device}")
    if a.dry_run:
        return
    if a.device == "cpu":
        print("WARNING: running on CPU (no CUDA in this Python). Use C:\\Program Files\\Python312\\python.exe.")

    images_dir = a.images or C.resolve_images_dir(a.data, "train")
    images = C.load_calibration_images(images_dir, a.num_images, a.imgsz, a.seed)
    print(f"loaded {images.shape[0]} calibration images from {images_dir}")

    baked, stats = C.calibrate(
        model, images, policy=a.policy, keep_depthwise=a.keep_depthwise, epochs=a.epochs, lr=a.lr,
        batch=a.batch, gamma=a.gamma, s_max=a.s_max, propagate=a.propagate, device=a.device, seed=a.seed)

    rep = T.ternary_report(baked)
    mean_i = sum(s["loss_init"] for s in stats) / max(len(stats), 1)
    mean_f = sum(s["loss_final"] for s in stats) / max(len(stats), 1)
    print(f"\nternary: {rep['ternary_layers']}/{rep['conv_layers']} layers, {rep['ternary_pct']:.1f}% of conv weights, "
          f"{rep['zero_pct_of_ternary']:.1f}% zeros, ~{rep['packed_mb_est']:.1f} MB packed")
    print(f"mean block loss (hard ternary): before calibration {mean_i:.3e} -> after {mean_f:.3e}")
    meta = {"policy": a.policy, "num_images": a.num_images, "epochs": a.epochs, "propagate": a.propagate,
            "gamma": a.gamma, "s_max": a.s_max, "stats": stats}
    C.save_catq_checkpoint(source, baked, dst, meta)
    print(f"saved {dst}")

    if a.eval:
        r = YOLO(dst).val(data=a.data, split="val", imgsz=a.imgsz, batch=a.batch, plots=False, verbose=False,
                          project=os.path.join("evaluation_results", "ternary", "val_runs"), name=name, exist_ok=True)
        print(f"val: mAP50 {r.box.map50:.4f}  mAP50-95 {r.box.map:.4f}")


if __name__ == "__main__":
    main()
