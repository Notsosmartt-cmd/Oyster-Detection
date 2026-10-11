"""CAT-Q for YOLO11: a convolutional port of the CAT-Q ternary quantizer (arXiv 2606.26650).

CAT-Q is *not* naive post-training quantization. The pretrained weights stay frozen, but a few
learnable factors per weight group are optimised by gradient descent on a small calibration set so
that the ternary block reproduces the full-precision block's output:

  Learnable modulation (LM), per output channel (the "group"):
      mu  = mu0 + d_mu * a0           mu0 = channel mean,  a0 = channel mean |w - mu0|
      a   = d_alpha * a0
      W^  = (W - mu) / a
      T   = Q(W^; delta)  in {-1,0,+1},  delta = d_delta * 0.5
      deployed weight = a * T          (mu is dropped, like TWN)
  Softened ternarization (ST), over the calibration steps (t from 0 to 1):
      t <= gamma : forward uses the smooth f(W^; s, delta) = [tanh(s(W^-delta)) + tanh(s(W^+delta))] / (2 tanh s),
                   s rising linearly from ~0 (identity) to s_max = 30 (nearly hard ternary)
      t >  gamma : forward is hard ternary; gradients flow through f at s = s_max (straight-through)
  Loss: MSE between the ternary block output and the full-precision block output.

Deviations from the paper (it targets LLMs), all forced by the architecture:
  * groups are conv output channels, calibration data are images (labels unused), not text;
  * "sliding window over layers" becomes one YOLO top-level block (C3k2, SPPF, ...) at a time, since
    YOLO has skip/concat connections that make windows across blocks awkward;
  * stage-2 gradient handling is my reading of "keep updating with the last stage-1 gradient".
See TERNARY_PLAN.md section 11.
"""
import copy
import glob
import math
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Detect

import ternary_yolo as T

_DIMS = (1, 2, 3)


def soft_ternary(x, s, delta):
    """Smooth ternarizer: identity-like for small s, hard {-1,0,1} for large s."""
    s = max(float(s), 1e-3)
    return (torch.tanh(s * (x - delta)) + torch.tanh(s * (x + delta))) / (2 * math.tanh(s))


class CatQConv2d(nn.Conv2d):
    """Conv2d with CAT-Q learnable modulation. Made by class swap (see convert_catq); adds the
    parameters dmu/dalpha/ddelta and the buffers mu0/alpha0 (shape O,1,1,1). The weights stay frozen."""

    catq_s = 1e-3  # softness (stage 1)
    catq_s_max = 30.0
    catq_hard = False  # stage 2: hard forward, straight-through gradient

    def ternary_parts(self):
        w = self.weight.detach().float()
        mu = self.mu0 + self.dmu.clamp(-1, 1) * self.alpha0
        alpha = self.dalpha.clamp_min(1e-3) * self.alpha0
        what = (w - mu) / alpha
        delta = self.ddelta.clamp_min(1e-3) * 0.5
        return what, alpha, delta

    def quantized_weight(self):
        what, alpha, delta = self.ternary_parts()
        if self.catq_hard:
            hard = torch.sign(what) * (what.abs() > delta).float()
            soft = soft_ternary(what, self.catq_s_max, delta)
            t = soft + (hard - soft).detach()
        else:
            t = soft_ternary(what, self.catq_s, delta)
        return (alpha * t).to(self.weight.dtype)

    def forward(self, x):
        return self._conv_forward(x, self.quantized_weight(), self.bias)


def convert_catq(model, policy="A", keep_depthwise=False, s_max=30.0):
    """Swap eligible convs (same policy as the QAT path) to CatQConv2d in place. Weights are frozen
    afterwards; only the three factors per channel are trainable."""
    convs = []
    for name, c in T._eligible(model, policy, keep_depthwise):
        w = c.weight.detach().float()
        mu0 = w.mean(dim=_DIMS, keepdim=True)
        alpha0 = (w - mu0).abs().mean(dim=_DIMS, keepdim=True).clamp_min(1e-8)
        c.__class__ = CatQConv2d
        c.register_buffer("mu0", mu0)
        c.register_buffer("alpha0", alpha0)
        for nm, init in (("dmu", 0.0), ("dalpha", 1.0), ("ddelta", 1.0)):
            c.register_parameter(nm, nn.Parameter(torch.full_like(mu0, init)))
        c.catq_s, c.catq_s_max, c.catq_hard, c.catq_name = 1e-3, s_max, False, name
        convs.append(c)
    for p in model.parameters():
        p.requires_grad_(False)
    return convs


def _hard_weight(m):
    what, alpha, delta = m.ternary_parts()
    return alpha * torch.sign(what) * (what.abs() > delta).float()


def bake_catq(model):
    """In place: CatQConv2d -> plain nn.Conv2d holding alpha*T. Returns the model."""
    for m in model.modules():
        if isinstance(m, CatQConv2d):
            with torch.no_grad():
                w = _hard_weight(m).to(m.weight.dtype)
            m.weight.data = w
            for nm in ("dmu", "dalpha", "ddelta"):
                del m._parameters[nm]
            for nm in ("mu0", "alpha0"):
                del m._buffers[nm]
            m.__class__ = nn.Conv2d
            for attr in ("catq_s", "catq_s_max", "catq_hard", "catq_name"):
                m.__dict__.pop(attr, None)
    for p in model.parameters():
        p.requires_grad_(True)
    return model


# ----------------------------------------------------------------------------- data

def resolve_images_dir(data_yaml, split="train"):
    import yaml
    from pathlib import Path
    d = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    root = Path(data_yaml).resolve().parent
    entry = str(d[split])
    stripped = entry
    while stripped.startswith(("../", "./")):
        stripped = stripped.split("/", 1)[1]
    base = Path(d["path"]) if d.get("path") else root
    for cand in (base / entry, root / entry, root / stripped, base / stripped):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"cannot find '{split}' images for {data_yaml} (entry {entry!r})")


def load_calibration_images(images_dir, n, imgsz, seed=0):
    """n random images, letterboxed to imgsz x imgsz, RGB in [0,1], as a CPU fp16 tensor (n,3,H,W)."""
    files = sorted(f for ext in ("jpg", "jpeg", "png", "bmp") for f in glob.glob(os.path.join(str(images_dir), f"*.{ext}")))
    if not files:
        raise FileNotFoundError(f"no images in {images_dir}")
    random.Random(seed).shuffle(files)
    out = []
    for f in files:
        im = cv2.imread(f)
        if im is None:
            continue
        h, w = im.shape[:2]
        r = imgsz / max(h, w)
        im = cv2.resize(im, (max(1, round(w * r)), max(1, round(h * r))), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
        top, left = (imgsz - im.shape[0]) // 2, (imgsz - im.shape[1]) // 2
        canvas[top:top + im.shape[0], left:left + im.shape[1]] = im
        out.append(torch.from_numpy(canvas[..., ::-1].copy()).permute(2, 0, 1))
        if len(out) == n:
            break
    return (torch.stack(out).float() / 255).half()


# ----------------------------------------------------------------------------- block streaming

def _flatten(o):
    if torch.is_tensor(o):
        return [o]
    if isinstance(o, dict):
        return [t for v in o.values() for t in _flatten(v)]
    if isinstance(o, (list, tuple)):
        return [t for v in o for t in _flatten(v)]
    return []


def _slice(inp, idx):
    return [_slice(v, idx) for v in inp] if isinstance(inp, (list, tuple)) else inp[idx]


def _to_dev(inp, dev):
    return [_to_dev(v, dev) for v in inp] if isinstance(inp, (list, tuple)) else inp.to(dev).float()


def _forward(m, xin):
    """Run one top-level block. Detect is run in training mode so it returns raw feature maps."""
    if isinstance(m, Detect):
        was = m.training
        m.training = True
        try:
            return _flatten(m(xin))
        finally:
            m.training = was
    return [m(xin)]


@torch.no_grad()
def _apply(m, inp, n, bs, dev):
    outs = None
    for b in range(0, n, bs):
        idx = slice(b, b + bs)
        flat = _forward(m, _to_dev(_slice(inp, idx), dev))
        if outs is None:
            outs = [torch.empty((n,) + tuple(t.shape[1:]), dtype=torch.half) for t in flat]
        for k, t in enumerate(flat):
            outs[k][idx] = t.half().cpu()
    return outs  # always a list of tensors (one for normal blocks)


def _gather(m, x_prev, y):
    f = m.f
    if f == -1:
        return x_prev
    if isinstance(f, int):
        return y[f]
    return [x_prev if j == -1 else y[j] for j in f]


def _loss(outs, tgts):
    return sum(F.mse_loss(o.float(), t.float()) for o, t in zip(outs, tgts))


def _train_block(m, convs, inp, tgt, n, epochs, lr, bs, gamma, s_max, dev, seed):
    params = [p for c in convs for p in (c.dmu, c.dalpha, c.ddelta)]
    for p in params:
        p.requires_grad_(True)

    def eval_hard(max_batches=4):  # hard-ternary loss on the first few batches
        for c in convs:
            c.catq_hard = True
        tot = cnt = 0.0
        with torch.no_grad():
            for b in range(0, min(n, bs * max_batches), bs):
                idx = slice(b, b + bs)
                tot += _loss(_forward(m, _to_dev(_slice(inp, idx), dev)), _to_dev(_slice(tgt, idx), dev)).item()
                cnt += 1
        return tot / max(cnt, 1)

    loss_init = eval_hard()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    total = epochs * math.ceil(n / bs)
    step = 0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for b in range(0, n, bs):
            t = step / total
            hard = t > gamma
            for c in convs:
                c.catq_hard = hard
                c.catq_s = max(s_max * min(t / gamma, 1.0), 1e-3)
            idx = perm[b:b + bs]
            loss = _loss(_forward(m, _to_dev(_slice(inp, idx), dev)), _to_dev(_slice(tgt, idx), dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
    loss_final = eval_hard()
    for p in params:
        p.requires_grad_(False)
    return loss_init, loss_final


def calibrate(model, images, policy="A", keep_depthwise=False, epochs=60, lr=1e-3, batch=8, gamma=0.8,
              s_max=30.0, propagate=False, device="cuda", seed=0, log=print):
    """CAT-Q calibration of a DetectionModel (in place). Returns (baked_model, per-block stats).

    images: (N,3,H,W) fp16 CPU tensor in [0,1]. propagate=True feeds each block the output of the
    already-calibrated ternary blocks (error-aware, 2x the activation cache); False feeds every block
    its full-precision input."""
    dev = torch.device(device)
    model = model.float().eval().to(dev)
    fp = copy.deepcopy(model)  # untouched reference: source of inputs and targets
    convs_all = convert_catq(model, policy, keep_depthwise, s_max)
    n = images.shape[0]
    blocks, fp_blocks = list(model.model), list(fp.model)
    save = set(model.save)

    last_use = {}
    for k, m in enumerate(blocks):
        for j in ([m.f] if isinstance(m.f, int) else m.f):
            if j != -1:
                last_use[j] = k

    x_fp, y_fp = images, {}
    x_q, y_q = images, {}
    stats = []
    t0 = time.time()
    for i, (m, m_fp) in enumerate(zip(blocks, fp_blocks)):
        inp_fp = _gather(m, x_fp, y_fp)
        tgt = _apply(m_fp, inp_fp, n, batch, dev)
        convs = [c for c in m.modules() if isinstance(c, CatQConv2d)]
        inp_q = _gather(m, x_q, y_q) if propagate else inp_fp
        if convs:
            li, lf = _train_block(m, convs, inp_q, tgt, n, epochs, lr, batch, gamma, s_max, dev, seed + i)
            stats.append({"block": i, "type": type(m).__name__, "layers": len(convs), "loss_init": li, "loss_final": lf})
            log(f"[{time.time() - t0:6.0f}s] block {i:2d} {type(m).__name__:8s} {len(convs):2d} convs  "
                f"hard-ternary loss {li:.3e} -> {lf:.3e}")
        if isinstance(m, Detect):
            break
        x_fp = tgt[0]
        if i in save:
            y_fp[i] = x_fp
        if propagate:
            for c in convs:
                c.catq_hard = True
            x_q = _apply(m, inp_q, n, batch, dev)[0]
            if i in save:
                y_q[i] = x_q
        for j in [j for j, k in last_use.items() if k == i]:
            y_fp.pop(j, None)
            y_q.pop(j, None)

    bake_catq(model)
    del fp
    return model, stats


def save_catq_checkpoint(src, model, dst, meta=None):
    """Write a plain Ultralytics checkpoint (loads with YOLO(dst)) from the baked model, keeping the
    source checkpoint's metadata (names, train_args, ...)."""
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    out = dict(ckpt, model=copy.deepcopy(model).cpu().half(), ema=None, optimizer=None, updates=None)
    if meta:
        out["catq"] = meta
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    torch.save(out, dst)
    return dst
