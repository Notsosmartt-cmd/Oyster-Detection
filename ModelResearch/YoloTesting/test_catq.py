"""Correctness checks for catq_yolo.py and the CAT-Q -> QAT handoff. CPU, no dataset needed, ~2 min.

    python test_catq.py            # yolo11n (default)
    python test_catq.py --size s
"""
import argparse
import copy
import math
import os
import tempfile

import torch
import torch.nn as nn
from ultralytics import YOLO

import catq_yolo as C
import ternary_yolo as T

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="n", choices=list("nsmlx"))
ap.add_argument("--weights", default=None)
args = ap.parse_args()
path = args.weights or f"models/yolo11{args.size}.pt"

torch.manual_seed(0)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def fresh():
    return copy.deepcopy(YOLO(path).model).float().eval()


def out_of(m, x):
    with torch.no_grad():
        o = m(x)
    return o[0] if isinstance(o, (list, tuple)) else o


x = torch.rand(1, 3, 160, 160)

# 1. soft ternarizer limits
v = torch.linspace(-2, 2, 101)
check("soft_ternary ~ identity for tiny s", (C.soft_ternary(v, 1e-3, 0.5) - v).abs().max() < 1e-3)
hard = torch.sign(v) * (v.abs() > 0.5).float()
check("soft_ternary ~ hard ternary for s=30 (away from the thresholds)",
      ((C.soft_ternary(v, 30, 0.5) - hard).abs() * ((v.abs() - 0.5).abs() > 0.2)).max() < 1e-3)

# 2. conversion matches the QAT policy
a = fresh(); convs = C.convert_catq(a, "A")
q = fresh(); rq = T.convert(q, "A")
check("same layers as the QAT policy A", len(convs) == rq["ternary_layers"], f"{len(convs)} layers")
check("all original weights frozen, only 3 factors per conv trainable",
      sum(p.requires_grad for p in a.parameters()) == 0 and sum(1 for c in convs for _ in (c.dmu, c.dalpha, c.ddelta)) == 3 * len(convs))
b = fresh(); cb = C.convert_catq(b, "B")
check("policy B converts more layers than A", len(cb) > len(convs), f"A={len(convs)} B={len(cb)}")

# 3. hard forward == baked forward, baked is ternary per channel
for c in convs:
    c.catq_hard = True
hard_out = out_of(a, x)
baked = C.bake_catq(copy.deepcopy(a)).eval()
check("baked output equals hard CAT-Q output", (hard_out - out_of(baked, x)).abs().max().item() < 1e-4,
      f"{(hard_out - out_of(baked, x)).abs().max().item():.2e}")
rep = T.ternary_report(baked)
check("baked model: <=3 values per channel in every converted layer", rep["ternary_layers"] == len(convs),
      f"{rep['ternary_layers']} layers, {rep['zero_pct_of_ternary']:.1f}% zeros")
check("no CatQConv2d / extra params left after bake",
      not any(isinstance(m, C.CatQConv2d) for m in baked.modules())
      and not any(n.endswith(("dmu", "dalpha", "ddelta")) for n, _ in baked.named_parameters()))
check("baked state_dict keys equal the original model's",
      set(baked.state_dict().keys()) == set(fresh().state_dict().keys()))

# 4. gradients reach the three factors (soft stage and hard/straight-through stage)
for hard_stage in (False, True):
    g = fresh(); gc = C.convert_catq(g, "A")
    for c in gc:
        c.catq_hard, c.catq_s = hard_stage, 10.0
        for p in (c.dmu, c.dalpha, c.ddelta):
            p.requires_grad_(True)
    first_block_convs = [c for c in g.model[1].modules() if isinstance(c, C.CatQConv2d)]
    xin = torch.rand(2, g.model[0].conv.out_channels, 40, 40)
    # run block 1 in isolation on its own input
    inp = g.model[0](torch.rand(2, 3, 80, 80))
    g.model[1](inp).pow(2).mean().backward()
    ok = all(c.dmu.grad is not None and c.dalpha.grad is not None and c.ddelta.grad is not None
             and c.ddelta.grad.abs().sum() > 0 and c.dalpha.grad.abs().sum() > 0 for c in first_block_convs)
    check(f"gradients reach dmu/dalpha/ddelta ({'hard STE' if hard_stage else 'soft'} stage)", ok,
          f"{len(first_block_convs)} convs")

# 5. calibration end to end on random images (functional test; real-data quality is checked by --smoke)
imgs = torch.rand(8, 3, 160, 160).half()
for propagate in (False, True):
    m, stats = C.calibrate(fresh(), imgs, policy="A", epochs=2, batch=4, propagate=propagate, device="cpu",
                           log=lambda *_: None)
    r = T.ternary_report(m)
    finite = all(math.isfinite(s["loss_init"]) and math.isfinite(s["loss_final"]) for s in stats)
    check(f"calibrate(propagate={propagate}) runs, finite losses, ternary result",
          finite and r["ternary_layers"] == len(convs) and len(stats) > 0,
          f"{len(stats)} blocks, mean loss {sum(s['loss_init'] for s in stats) / len(stats):.2e} -> "
          f"{sum(s['loss_final'] for s in stats) / len(stats):.2e}")
cal = m

# 6. checkpoint round trip through plain YOLO()
with tempfile.TemporaryDirectory() as td:
    dst = os.path.join(td, "catq.pt")
    C.save_catq_checkpoint(path, cal, dst, {"test": True})
    y = YOLO(dst)
    res = y.predict(torch.rand(1, 3, 160, 160), imgsz=160, verbose=False, device="cpu")
    check("CAT-Q checkpoint loads with plain YOLO() and predicts", len(res) == 1)
    check("loaded checkpoint is ternary", T.ternary_report(y.model)["ternary_layers"] == len(convs))

# 7. CAT-Q -> QAT handoff reproduces the CAT-Q network exactly (both quantizer modes)
ref = out_of(cal, x)
for mode in ("absmean", "twn"):
    h = copy.deepcopy(cal).eval()
    T.convert(h, "A", mode)
    n_init = T.init_latent_from_ternary(h)
    d = (out_of(h, x) - ref).abs().max().item()
    check(f"handoff {mode}: QAT forward == CAT-Q baked forward", d < 1e-3, f"{n_init} layers, max diff {d:.2e}")

# 8. policy B QAT on a policy-A CAT-Q model: the A layers are preserved, the extra layers stay FP-latent
h = copy.deepcopy(cal).eval()
T.convert(h, "B", "absmean")
n_init = T.init_latent_from_ternary(h)
a_layers = {n for n, _ in T._eligible(fresh(), "A", False)}
ok = all((m.quantized_weight() - dict(cal.named_modules())[n].weight).abs().max() < 1e-5
         for n, m in h.named_modules() if n in a_layers)
check("handoff B-over-A keeps CAT-Q layers exact", ok and n_init == len(convs), f"{n_init} layers initialised")

print(f"\n{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
