"""Ternary-weight conv layers for Ultralytics YOLO11.

Every converted Conv2d keeps its full-precision "latent" weight for training, but the forward pass
uses a per-output-channel ternary version  alpha * {-1, 0, +1}  (straight-through estimator).
See TERNARY_PLAN.md for the design.

Typical use (done for you by YoloTrainTernary.py):

    model = YOLO("oysterTrainedModels/yolo11nMergedData960/weights/best.pt")
    add_ternary_callbacks(model, policy="A")
    model.train(...)
    bake_checkpoint(model.trainer.best, "best_baked.pt")

Importing this module also makes Ultralytics' BN fusing ternary-aware (see _ternary_aware_fuse).
The module must be importable (same folder) whenever a training-form checkpoint is unpickled.
"""
import copy

import torch
import torch.nn as nn
from ultralytics.nn import tasks as _tasks
from ultralytics.nn.modules import Conv, Detect
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model

__all__ = [
    "TernaryConv2d", "quantize_weight", "convert", "set_blend", "bake", "bake_checkpoint",
    "ternary_report", "make_callbacks", "add_ternary_callbacks",
]

_DIMS = (1, 2, 3)  # reduce over everything except the output-channel axis


def quantize_weight(w, mode="absmean"):
    """Return the ternary version of conv weight `w` (shape O,I/g,kH,kW): per channel alpha*{-1,0,1}."""
    w32 = w.detach().float()
    if mode == "absmean":  # BitNet b1.58 style
        alpha = w32.abs().mean(dim=_DIMS, keepdim=True).clamp_min(1e-8)
        return (w32 / alpha).round().clamp(-1, 1) * alpha
    if mode == "twn":  # Ternary Weight Networks
        absw = w32.abs()
        delta = 0.7 * absw.mean(dim=_DIMS, keepdim=True)
        mask = (absw > delta).float()
        alpha = (absw * mask).sum(dim=_DIMS, keepdim=True) / mask.sum(dim=_DIMS, keepdim=True).clamp_min(1)
        return torch.sign(w32) * mask * alpha
    raise ValueError(f"unknown ternary mode {mode!r} (use 'absmean' or 'twn')")


class TernaryConv2d(nn.Conv2d):
    """nn.Conv2d whose forward uses ternary weights. Instances are made by swapping __class__
    (see convert), so there are no constructor arguments and no new parameters or buffers."""

    tern_mode = "absmean"  # class-level defaults so swapped instances need no extra state
    tern_blend = 1.0  # 0 = full precision, 1 = fully ternary (used for gradual quantization)

    def quantized_weight(self):
        return quantize_weight(self.weight, self.tern_mode).to(self.weight.dtype)

    def forward(self, x):
        w = self.weight
        q = self.quantized_weight()
        if self.tern_blend < 1.0:
            wd = w.detach().to(q.dtype)
            q = wd + self.tern_blend * (q - wd)
        if torch.is_grad_enabled() and w.requires_grad:
            q = w + (q - w).detach()  # straight-through estimator
        return self._conv_forward(x, q, self.bias)


def _bake_conv(m):
    """In place: make a TernaryConv2d a plain nn.Conv2d holding its quantized weight."""
    with torch.no_grad():
        m.weight.data = m.quantized_weight()
    m.__class__ = nn.Conv2d
    for attr in ("tern_mode", "tern_blend", "tern_name"):
        m.__dict__.pop(attr, None)


def _ternary_aware_fuse(conv, bn, _orig=_tasks.fuse_conv_and_bn):
    """Ultralytics fuse() folds BN into conv.weight (in place in this version). For a TernaryConv2d
    that would fold into the latent full-precision weights, so bake the conv first. Fusing is an
    inference-only, destructive step anyway (it also deletes the BN layer)."""
    if isinstance(conv, TernaryConv2d):
        _bake_conv(conv)
    return _orig(conv, bn)


if not getattr(_tasks.fuse_conv_and_bn, "_ternary_aware", False):
    _ternary_aware_fuse._ternary_aware = True
    _tasks.fuse_conv_and_bn = _ternary_aware_fuse


def _eligible(model, policy, keep_depthwise):
    """List of (name, Conv2d) to convert. Policy A: backbone+neck. Policy B: A + head inner convs.
    Never: first conv, Detect output 1x1 convs, DFL conv."""
    if policy not in ("A", "B"):
        raise ValueError("policy must be 'A' or 'B'")
    detect = next((m for m in model.modules() if isinstance(m, Detect)), None)
    head_ids = {id(c) for c in detect.modules() if isinstance(c, nn.Conv2d)} if detect is not None else set()
    head_inner = {id(m.conv) for m in detect.modules() if isinstance(m, Conv)} if detect is not None else set()
    first = next(c for c in model.model[0].modules() if isinstance(c, nn.Conv2d))
    out = []
    for name, c in model.named_modules():
        if not isinstance(c, nn.Conv2d) or isinstance(c, TernaryConv2d) or c is first:
            continue
        if id(c) in head_ids and (policy == "A" or id(c) not in head_inner):
            continue  # head output convs and DFL are never converted; inner head convs only in B
        if keep_depthwise and c.groups > 1 and c.groups == c.in_channels:
            continue
        out.append((name, c))
    return out


def convert(model, policy="A", mode="absmean", keep_depthwise=False):
    """Swap eligible Conv2d layers to TernaryConv2d in place. Returns a summary dict."""
    for name, c in _eligible(model, policy, keep_depthwise):
        c.__class__ = TernaryConv2d
        c.tern_mode = mode
        c.tern_blend = 1.0
        c.tern_name = name
    return ternary_report(model)


def set_blend(model, blend):
    for m in model.modules():
        if isinstance(m, TernaryConv2d):
            m.tern_blend = float(blend)


def bake(model):
    """In place: replace each TernaryConv2d by a plain nn.Conv2d holding the quantized weights."""
    for m in model.modules():
        if isinstance(m, TernaryConv2d):
            _bake_conv(m)
    return model


def _max_unique_per_channel(w):
    flat = w.detach().float().reshape(w.shape[0], -1).sort(dim=1).values
    return (1 + (flat[:, 1:] != flat[:, :-1]).sum(dim=1)).max().item() if flat.shape[1] > 1 else 1


def ternary_report(model):
    """Count conv layers whose every output channel has <= 3 distinct weight values (ternary).
    Works on training-form models (uses the quantized weight) and on baked / fused ones.
    Layers with fewer than 4 weights per channel are not counted (trivially <= 3 values)."""
    n_layers = n_tern = 0
    w_total = w_tern = w_zero = 0
    for m in model.modules():
        if not isinstance(m, nn.Conv2d):
            continue
        # TernaryConv2d (QAT) and catq_yolo.CatQConv2d both expose quantized_weight()
        w = m.quantized_weight() if hasattr(m, "quantized_weight") else m.weight
        n_layers += 1
        w_total += w.numel()
        if w[0].numel() >= 4 and _max_unique_per_channel(w) <= 3:
            n_tern += 1
            w_tern += w.numel()
            w_zero += (w == 0).sum().item()
    other = sum(p.numel() for p in model.parameters()) - w_tern
    return {
        "conv_layers": n_layers, "ternary_layers": n_tern,
        "conv_weights": w_total, "ternary_weights": w_tern,
        "ternary_pct": 100.0 * w_tern / max(w_total, 1),
        "zero_pct_of_ternary": 100.0 * w_zero / max(w_tern, 1),
        # 2 bits per ternary weight + fp16 for everything else (scales, BN, biases, FP layers)
        "packed_mb_est": (w_tern * 2 / 8 + other * 2) / 1e6,
    }


def bake_checkpoint(src, dst):
    """Load a (training-form) checkpoint, bake it, and save a plain Ultralytics checkpoint."""
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    model = bake(copy.deepcopy(ckpt.get("ema") or ckpt["model"]))
    out = dict(ckpt, model=model, ema=None, optimizer=None, updates=None)
    torch.save(out, dst)
    report = ternary_report(model)
    LOGGER.info(f"baked {src} -> {dst}: {report['ternary_layers']}/{report['conv_layers']} ternary conv layers")
    return report


def init_latent_from_ternary(model):
    """Handoff CAT-Q -> QAT. A CAT-Q checkpoint holds weights alpha*{-1,0,1} per channel. Call this
    right after convert(): for every TernaryConv2d whose weights are already ternary, rescale the
    latent weight so the QAT quantizer reproduces exactly the same alpha and T:
      absmean: alpha' = mean|W'| = alpha only if W' = alpha*T/frac_nonzero  (then round(W'/alpha') = T)
      twn:     W' = alpha*T already gives alpha' = alpha.
    Layers that are not ternary yet (e.g. policy B on a policy-A CAT-Q model) are left untouched."""
    n = 0
    for m in model.modules():
        if not isinstance(m, TernaryConv2d) or m.weight[0].numel() < 4:
            continue
        w = m.weight.data
        if _max_unique_per_channel(w) > 3:
            continue
        if m.tern_mode == "absmean":
            frac = (w != 0).float().mean(dim=_DIMS, keepdim=True)
            m.weight.data = torch.where(frac > 0, w / frac.clamp_min(1e-8), w)
        n += 1
    return n


def make_callbacks(policy="A", mode="absmean", keep_depthwise=False, ramp_epochs=0, init_from_ternary=False):
    """Return (on_pretrain_routine_end, on_train_epoch_start) trainer callbacks."""

    def on_pretrain_routine_end(trainer):
        # EMA and optimizer already exist here; params are shared objects, so a class swap is safe.
        for tag, mdl in (("model", trainer.model), ("ema", trainer.ema.ema)):
            s = convert(unwrap_model(mdl), policy, mode, keep_depthwise)
            if init_from_ternary:
                k = init_latent_from_ternary(unwrap_model(mdl))
                LOGGER.info(f"ternary[{tag}]: initialised {k} already-ternary layers from the CAT-Q checkpoint")
            LOGGER.info(
                f"ternary[{tag}] policy={policy} mode={mode}: {s['ternary_layers']}/{s['conv_layers']} "
                f"conv layers, {s['ternary_pct']:.1f}% of conv weights"
            )

    def on_train_epoch_start(trainer):
        # The blend only softens the *training* forward pass. The EMA copy is what gets validated and
        # saved as best.pt/last.pt, so it always stays fully ternary (blend 1.0): the validation score
        # is then the real ternary score and best.pt can be baked/fused without changing the model.
        if ramp_epochs > 0:
            b = min(1.0, (trainer.epoch + 1) / ramp_epochs)
            set_blend(unwrap_model(trainer.model), b)
            LOGGER.info(f"ternary blend (training forward only) = {b:.2f}; validation/EMA stays fully ternary")

    return on_pretrain_routine_end, on_train_epoch_start


def add_ternary_callbacks(yolo, **kw):
    on_pre, on_epoch = make_callbacks(**kw)
    yolo.add_callback("on_pretrain_routine_end", on_pre)
    yolo.add_callback("on_train_epoch_start", on_epoch)
