"""Fine-tune an oyster-trained YOLO11 model with ternary conv weights (see TERNARY_PLAN.md).

Examples (run from this folder):
    python YoloTrainTernary.py --size n --dry-run     # show what would be converted, no training
    python YoloTrainTernary.py --size n --smoke       # ~1 minute pipeline check (1 epoch, tiny subset)
    python YoloTrainTernary.py --size n               # real run
    python YoloTrainTernary.py --size x --batch 4 --policy A

Output: oysterTrainedModels/<name>/weights/{best,last}.pt   (training form, needs ternary_yolo.py)
        oysterTrainedModels/<name>/weights/best_baked.pt     (plain YOLO checkpoint, ternary values)
"""
import argparse
import copy
import os
import sys

from ultralytics import YOLO

import ternary_yolo as T


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=list("nsmlx"))
    ap.add_argument("--source", default=None,
                    help="start weights (default: oysterTrainedModels/yolo11<size>MergedData960/weights/best.pt)")
    ap.add_argument("--from-coco", action="store_true",
                    help="start from models/yolo11<size>.pt (COCO) instead of an oyster-trained model; not recommended")
    ap.add_argument("--init-from-catq", action="store_true",
                    help="start QAT from the CAT-Q checkpoint (YoloCatQ.py output): the ternary weights and "
                         "scales are reproduced exactly, then QAT refines them. Forces --ramp-epochs 0.")
    ap.add_argument("--data", default="MergedData/data.yaml")
    ap.add_argument("--policy", default="A", choices=["A", "B"], help="A: backbone+neck, B: A + head inner convs")
    ap.add_argument("--mode", default="absmean", choices=["absmean", "twn"])
    ap.add_argument("--keep-depthwise", action="store_true", help="leave depthwise convs full precision")
    ap.add_argument("--ramp-epochs", type=int, default=10, help="epochs to blend FP -> ternary (0 = off)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=None, help="default 16 (n,s), 8 (m,l), 4 (x); -1 = auto (uses ~60% of VRAM)")
    ap.add_argument("--lr0", type=float, default=5e-4)
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--project", default="oysterTrainedModels")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 2%% of data, imgsz 320, batch 4")
    ap.add_argument("--dry-run", action="store_true", help="load + convert + print, do not train")
    return ap.parse_args()


def main():
    args = parse_args()
    batch = args.batch if args.batch is not None else {"n": 16, "s": 16, "m": 8, "l": 8, "x": 4}[args.size]

    if args.init_from_catq:
        source = args.source or f"oysterTrainedModels/yolo11{args.size}CatQ{args.policy}/weights/catq_baked.pt"
        if args.ramp_epochs:
            print("--init-from-catq: weights are already ternary, setting --ramp-epochs 0")
            args.ramp_epochs = 0
    elif args.from_coco:
        source = f"models/yolo11{args.size}.pt"
    else:
        source = args.source or f"oysterTrainedModels/yolo11{args.size}MergedData960/weights/best.pt"
    if not os.path.exists(source):
        sys.exit(
            f"Start weights not found: {source}\n"
            + (f"Run python YoloCatQ.py --size {args.size} --policy {args.policy} first."
               if args.init_from_catq else
               f"Train the full-precision oyster baseline first (python YoloTrainFP32.py --size {args.size}), "
               f"or pass --source / --from-coco.")
        )

    name = args.name or (f"yolo11{args.size}Ternary{args.policy}{args.mode.capitalize()}"
                         + ("_fromCatQ" if args.init_from_catq else ""))
    if args.smoke:
        name += "_smoke"

    model = YOLO(source)
    kw = dict(policy=args.policy, mode=args.mode, keep_depthwise=args.keep_depthwise)

    # Preview on a throwaway copy; the trainer rebuilds the model and the callback converts that one.
    preview = T.convert(copy.deepcopy(model.model), **kw)
    print(f"\nsource        : {source}")
    print(f"run name      : {name}")
    print(f"ternary layers: {preview['ternary_layers']}/{preview['conv_layers']} conv layers, "
          f"{preview['ternary_pct']:.1f}% of conv weights (policy {args.policy}, {args.mode})")
    print(f"batch {batch}, imgsz {args.imgsz}, epochs {args.epochs}, lr0 {args.lr0}, ramp {args.ramp_epochs}")
    if args.dry_run:
        return

    T.add_ternary_callbacks(model, ramp_epochs=args.ramp_epochs, init_from_ternary=args.init_from_catq, **kw)

    train_kw = dict(
        data=args.data, epochs=args.epochs, patience=args.patience, imgsz=args.imgsz, batch=batch,
        optimizer=args.optimizer, lr0=args.lr0, weight_decay=args.weight_decay, workers=args.workers,
        seed=args.seed, amp=True, project=args.project, name=name, exist_ok=False,
    )
    if args.smoke:
        train_kw.update(epochs=1, fraction=0.02, imgsz=320, batch=4, workers=2, patience=0)

    model.train(**train_kw)
    best = str(model.trainer.best)
    try:
        T.bake_checkpoint(best, os.path.join(os.path.dirname(best), "best_baked.pt"))
    except Exception as e:  # keep the training result even if baking fails
        print(f"bake failed: {e}\nRun: python -c \"import ternary_yolo as T; T.bake_checkpoint('{best}', 'best_baked.pt')\"")


if __name__ == "__main__":  # required on Windows (dataloader workers re-import this file)
    main()
