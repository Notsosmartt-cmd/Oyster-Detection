"""Quick correctness checks for ternary_yolo.py. No training, runs on CPU in about a minute.

    python test_ternary.py            # yolo11n (default)
    python test_ternary.py --size s
"""
import argparse
import copy
import os
import tempfile
from types import SimpleNamespace

import torch
import torch.nn as nn
from ultralytics import YOLO

import ternary_yolo as T

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="n", choices=list("nsmlx"))
ap.add_argument("--weights", default=None, help="checkpoint to test (default models/yolo11<size>.pt)")
args = ap.parse_args()
path = args.weights or f"models/yolo11{args.size}.pt"

torch.manual_seed(0)
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def fresh():
    return copy.deepcopy(YOLO(path).model).float().eval()


x = torch.rand(1, 3, 320, 320)


def out_of(m):
    with torch.no_grad():
        o = m(x)
    return o[0] if isinstance(o, (list, tuple)) else o


base = fresh()
n_conv = sum(isinstance(m, nn.Conv2d) for m in base.modules())

# 1. policy A / B coverage
a = fresh(); ra = T.convert(a, "A")
b = fresh(); rb = T.convert(b, "B")
first = next(c for c in a.model[0].modules() if isinstance(c, nn.Conv2d))
check("first conv stays full precision", not isinstance(first, T.TernaryConv2d))
detect = a.model[-1]
bare_out = [s[-1] for s in list(detect.cv2) + list(detect.cv3)]
check("Detect output convs stay full precision (A and B)",
      not any(isinstance(c, T.TernaryConv2d) for c in bare_out)
      and not any(isinstance(c, T.TernaryConv2d) for c in [s[-1] for s in list(b.model[-1].cv2) + list(b.model[-1].cv3)]))
check("DFL conv untouched", not isinstance(b.model[-1].dfl.conv, T.TernaryConv2d))
check("policy B converts more layers than A", rb["ternary_layers"] > ra["ternary_layers"],
      f"A={ra['ternary_layers']} B={rb['ternary_layers']} of {n_conv} conv layers")
check("report matches layers converted",
      ra["ternary_layers"] == sum(isinstance(m, T.TernaryConv2d) for m in a.modules()))
print(f"        A: {ra['ternary_pct']:.1f}% of conv weights ternary, {ra['zero_pct_of_ternary']:.1f}% zeros, "
      f"~{ra['packed_mb_est']:.1f} MB packed")
print(f"        B: {rb['ternary_pct']:.1f}% of conv weights ternary")

# 2. ternary-form forward == baked forward
tern_out = out_of(a)
baked = T.bake(copy.deepcopy(a)).eval()
diff = (tern_out - out_of(baked)).abs().max().item()
check("baked output equals ternary-form output", diff < 1e-4, f"max diff {diff:.2e}")
check("ternary changes the output vs. the FP model (quantization is active)",
      (tern_out - out_of(base)).abs().max().item() > 1e-3)

# 3. <= 3 unique values per channel in baked ternary layers, FP layers untouched
rep = T.ternary_report(baked)
check("baked model: ternary layer count unchanged by bake", rep["ternary_layers"] == ra["ternary_layers"],
      f"{rep['ternary_layers']} layers")
check("baked model has no TernaryConv2d left", not any(isinstance(m, T.TernaryConv2d) for m in baked.modules()))
check("FP baseline reports 0 ternary layers", T.ternary_report(base)["ternary_layers"] == 0)

# 4. TWN mode works and is ternary too
w = fresh(); rw = T.convert(w, "A", mode="twn")
check("twn mode ternary", rw["ternary_layers"] == ra["ternary_layers"] and out_of(w).shape == tern_out.shape)

# 5. BN fusing: train-form fuse == baked fuse, and per-channel ternary structure survives fusing
f_t = copy.deepcopy(a).eval().fuse(verbose=False)
f_b = copy.deepcopy(baked).eval().fuse(verbose=False)
d = (out_of(f_t) - out_of(f_b)).abs().max().item()
check("fused ternary-form == fused baked", d < 1e-3, f"max diff {d:.2e}")
d2 = (out_of(f_t) - tern_out).abs().max().item()
check("fusing keeps ternary behaviour (vs unfused)", d2 < 1e-3, f"max diff {d2:.2e}")
check("fused model still ternary per channel", T.ternary_report(f_b)["ternary_layers"] >= ra["ternary_layers"] - 1,
      f"{T.ternary_report(f_b)['ternary_layers']} layers")

# 6. gradients reach latent weights through the STE
g = copy.deepcopy(a).train()
for p in g.parameters():
    p.requires_grad_(True)


def flat_tensors(o):  # train-mode Detect output is a list or a dict (e.g. 'feats') depending on version
    if torch.is_tensor(o):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from flat_tensors(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from flat_tensors(v)


loss = sum(t.float().mean() for t in flat_tensors(g(x)))
loss.backward()
tc = [m for m in g.modules() if isinstance(m, T.TernaryConv2d)]
check("STE: all ternary layers receive gradients",
      all(m.weight.grad is not None and m.weight.grad.abs().sum() > 0 for m in tc), f"{len(tc)} layers")

# 7. gradual quantization blend
bl = copy.deepcopy(a).eval()
T.set_blend(bl, 0.0)
check("blend=0 equals full precision", (out_of(bl) - out_of(base)).abs().max().item() < 1e-5)
T.set_blend(bl, 1.0)
check("blend=1 equals ternary", (out_of(bl) - tern_out).abs().max().item() < 1e-5)

# 8. trainer callbacks, driven by a fake trainer
trainer = SimpleNamespace(model=fresh().train(), ema=SimpleNamespace(ema=fresh().eval()), epoch=0)
on_pre, on_epoch = T.make_callbacks(policy="A", ramp_epochs=4)
on_pre(trainer)
check("callback converts model and EMA",
      all(any(isinstance(m, T.TernaryConv2d) for m in mdl.modules()) for mdl in (trainer.model, trainer.ema.ema)))
on_epoch(trainer)
blends = {m.tern_blend for m in trainer.model.modules() if isinstance(m, T.TernaryConv2d)}
ema_blends = {m.tern_blend for m in trainer.ema.ema.modules() if isinstance(m, T.TernaryConv2d)}
check("ramp sets blend 0.25 on the training model at epoch 0", blends == {0.25}, str(blends))
check("EMA/validation model stays fully ternary during the ramp", ema_blends == {1.0}, str(ema_blends))

# 9. pickle round trip keeps the ternary class (what trainer checkpoints rely on)
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "m.pt")
    torch.save(a, p)
    r = torch.load(p, weights_only=False)
    check("pickle round trip keeps TernaryConv2d", sum(isinstance(m, T.TernaryConv2d) for m in r.modules()) == len(tc))

    # 10. bake_checkpoint -> plain YOLO() load -> predict
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["model"] = copy.deepcopy(a).half()
    ckpt["ema"] = None
    src, dst = os.path.join(td, "src.pt"), os.path.join(td, "baked.pt")
    torch.save(ckpt, src)
    T.bake_checkpoint(src, dst)
    y = YOLO(dst)
    res = y.predict(torch.rand(1, 3, 320, 320), imgsz=320, verbose=False, device="cpu")
    check("baked checkpoint loads with plain YOLO() and predicts", len(res) == 1)
    check("baked checkpoint has ternary layers", T.ternary_report(y.model)["ternary_layers"] == ra["ternary_layers"])

print(f"\n{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
