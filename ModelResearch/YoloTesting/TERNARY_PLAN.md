# Ternary YOLO11 plan (n, s, m, l, x)

Goal: fine-tune the oyster-trained YOLO11 models so every (eligible) conv weight is one of
{-α, 0, +α} (ternary, ~1.58 bit), then measure how much accuracy it costs per model size.

Status: **code is staged and unit-tested; no model has been trained yet.** See section 10 for the
files and the commands to run. Facts below marked (checked) were verified on this machine.

## 0. What exists (checked)

| Item | State |
|---|---|
| Base COCO weights | `models/yolo11{n,s,m,l,x}.pt` all present (l and x downloaded for this plan) |
| Oyster-trained FP32 | `oysterTrainedModels/yolo11{n,s,m}MergedData960` and `...NickData960`, each with `weights/best.pt`. **No l or x yet.** |
| Datasets | `MergedData/` (3 classes: Indeterminate, Closed, Open) and `REU_Oyster_2024_Improved-2/` |
| Python | `C:\Program Files\Python312\python.exe`, ultralytics 8.3.228, torch 2.9.1+cu130 |
| GPU | RTX 5060 Ti, 16 GB |

Conv size per model (checked): n 2.6M params / 88 Conv2d, s 9.5M / 88, m 20.1M / 113,
l 25.4M / 174, x 57.0M / 174. Essentially all parameters sit in Conv2d layers, so quantizing
convs covers the model.

## 1. Environment fix (your console error)

`pip.exe` is blocked ("Access is denied") in your conda `(base)` shell. Do not fight it; call pip
through Python instead:

```
python -m pip install ultralytics
```

If that is blocked too, use the interpreter that already has everything:

```
& "C:\Program Files\Python312\python.exe" -m pip list
```

Run all training from `ModelResearch/YoloTesting` with that same interpreter. Check which one a shell
uses with `python -c "import sys; print(sys.executable)"`.

## 2. Key idea: train ternary, then "bake" it

* **Training form.** Replace each eligible `nn.Conv2d` with a ternary-aware conv. It keeps a latent
  full-precision weight `W`. Each forward pass computes `W_q = α * round_clip(W/α)` per output
  channel and convolves with `W_q`. Gradients bypass the rounding (straight-through estimator) and
  update `W`.
* **Quantizer (default): absmean, as in BitNet b1.58.** `α = mean(|W|)` per output channel,
  `W_q = α * clip(round(W/α), -1, 1)`. No threshold hyperparameter.
* **Ablation: Ternary Weight Networks.** `Δ = 0.7*mean(|W|)`, `α = mean(|W| where |W|>Δ)`.
* **Weight-only first.** Activations stay FP16/FP32. Activation quantization is a later, separate
  experiment.
* **Bake step.** After training, write a normal Ultralytics checkpoint whose conv weights are the
  already-quantized `W_q` in plain `nn.Conv2d`. This model:
  * loads/validates/exports like any YOLO model (no custom class needed to open it),
  * is numerically the same as the ternary model,
  * survives BN fusing: BN scale is a per-channel scalar, so each channel is still
    {-a, 0, +a} times a scalar. Verify by counting unique weight values per channel (must be <= 3).

Why bake: Ultralytics' `fuse()` (`nn/tasks.py:219`, calls `fuse_conv_and_bn` at line 231) folds BN
into `conv.weight`. On a training-form model that would fold into the **latent FP weights** and
silently drop the ternarization. `final_eval` loads `best.pt` through AutoBackend, which fuses.
Fix as built: `ternary_yolo.py` replaces `ultralytics.nn.tasks.fuse_conv_and_bn` with a wrapper that
bakes a `TernaryConv2d` first (this Ultralytics version fuses in place), so fusing is correct even
on training-form checkpoints. Tested: fused train-form == fused baked, max diff 0.

## 3. Which layers

| Layer group | Stage A (start here) | Stage B (ablation) |
|---|---|---|
| First conv (`model.0.conv`) | FP | FP |
| Backbone + neck convs | **ternary** | ternary |
| Detect head inner convs (`cv2.*.0/1`, `cv3.*`) | FP | ternary |
| Detect head final 1x1 convs (box/cls outputs) | FP | FP |
| DFL conv (fixed, no grad) | skip | skip |
| BatchNorm, biases | FP | FP |

Depthwise convs (`groups == out_channels`, used in YOLO11 cls branch) have tiny per-channel
fan-in; if they hurt, keep them FP and report it.

Report % of weights ternary for each variant, so the "ternary" claim is honest.

## 4. Hooking into Ultralytics training (checked in trainer.py)

In `_setup_train` the order is: EMA created (`trainer.py:334`), optimizer built (`:345`), then
`on_pretrain_routine_end` fires (`:358`). So:

* Register an `on_pretrain_routine_end` callback.
* In it, convert convs in **both** `trainer.model` and `trainer.ema.ema` by swapping the class in
  place (`m.__class__ = TernaryConv2d`, add attributes). Parameters stay the same objects, so the
  optimizer keeps working with no rebuild.
* Checkpoints (`save_model`, `:584`) pickle a deepcopy of the EMA model by class reference, so the
  class must be importable: keep it in `ternary_yolo.py` next to the training script and run from
  there. The baked checkpoint (section 2) avoids this dependency.
* Single GPU on Windows: no DDP concerns. Do not use `torch.compile`.

## 5. Training recipe (starting values, to be tuned on n first)

* **Start from the oyster FP32 `best.pt`, not COCO weights.** Ternarizing straight from COCO then
  adapting to oysters mixes two losses together and is harder.
* `optimizer=AdamW`, `lr0` around 5e-4 (lower than a from-scratch run), short warmup.
* Low weight decay: with STE, decay on latent weights pulls them toward 0 and inflates sparsity.
* Optional gradual quantization: blend `W` and `W_q` with factor 0 to 1 over the first ~10 epochs.
* 100 to 150 epochs, `patience=30`, `imgsz=960` to match the existing oyster runs, mosaic closing
  at the end (default `close_mosaic=10`).
* The FP->ternary blend applies to the training forward pass only. The EMA copy (what is validated
  and saved as best.pt) is always fully ternary. Found by the first smoke run: blending the EMA made
  in-training validation score a hybrid model while the final evaluation scored the fully ternary
  one (0.0283 vs 0.0001 mAP50), and best.pt would have picked a hybrid early-ramp epoch. After the
  fix, in-training, final-eval and baked-checkpoint scores agree (0.000125 / 0.000109 / 0.000116 on
  the 1-epoch smoke model).
* Same data/augmentation as the FP32 baseline so the comparison is fair.
* Dataset note: Ultralytics drops images whose labels are invalid (class 3 in a 3-class set, or
  coordinates > 1). In `MergedData`: 22 of 257 sampled train images and 133 of 638 val images are
  dropped (val scores use 505 images). Same for every model, but worth cleaning the labels.
* Later improvement: distillation from the FP32 model as teacher (not built into Ultralytics; extra
  work).

## 6. Baselines still needed

* **l and x FP32 oyster runs do not exist.** Train them first (same config as the existing
  `...MergedData960` runs). x at imgsz 960 will probably not fit batch 16 in 16 GB: lower the batch
  (try 4 to 8) or use `batch=-1` auto-batch, and record what you used.
* Re-evaluate every FP32 model with one script on the **same split** (`val` and `test`) so every
  number in the final table comes from one pipeline.

## 7. Work order

1. Env fix (section 1). Confirm `python -c "import ultralytics, torch; print(torch.cuda.is_available())"`.
2. Write `ternary_yolo.py`: `TernaryConv2d`, `convert(model, policy)`, `bake(model)`,
   `count_unique(model)`.
3. **Unit checks on n:** converted model forward runs; baked model output equals ternary-form output
   (max abs diff ~1e-4 or less); baked model has <= 3 unique values per channel.
4. `YoloTrainTernary.py`: loads `best.pt`, installs callback, trains, bakes `best.pt` to
   `oysterTrainedModels/<name>Ternary/weights/best_baked.pt`.
5. Run **n** Stage A. Tune lr, epochs, absmean vs TWN.
6. Run n Stage B. Decide the layer policy.
7. Run s, m, then train l/x FP32 baselines, then l, x ternary.
8. `YoloCompareTernary.py`: table of mAP50, mAP50-95, precision, recall, % ternary weights, zero %,
   sizes.

## 8. Metrics to report per model

* mAP50 and mAP50-95 on val and test, FP32 vs ternary, and the delta.
* Fraction of weights that are 0 / +1 / -1 per model.
* Size: FP32 file, FP16, and theoretical packed (about 2 bits per ternary weight plus FP scales).
  For reference, x has 56.9M conv weights, so about 14 MB packed at 2 bits, versus 109 MB for the
  FP16 `.pt` shown above.
* PyTorch latency. Expect **no speedup**: fake-quantized ternary still runs FP convs.

## 9. Expectations and limits

* Hypothesis to test, not a result: larger models (l, x) lose less accuracy than n because they
  have more redundancy. That is why the full sweep is worth running.
* No published ternary YOLO11 result was found in earlier research, so there is no reference number
  to match. Treat n/s accuracy drops as an open question.
* **Browser:** TensorFlow.js has no ternary ops. A baked model exports to TFJS as ordinary float
  weights, so there is no download-size or speed gain on the website unless you write a custom
  2-bit packer and unpacker. Real gains need ternary-aware inference (custom CPU kernels, FPGA).
  For the site itself, int8 YOLO11n/s is still the practical route.
* Windows file locks and a 16 GB GPU make x the slowest and riskiest run. Do it last.

## 10. What is staged and how to run it

All files are in `ModelResearch/YoloTesting/`. Run from that folder with the Python 3.12 that has
ultralytics (`python -m pip install ultralytics` if a shell lacks it).

| File | Purpose |
|---|---|
| `ternary_yolo.py` | `TernaryConv2d`, `convert`, `bake`, `bake_checkpoint`, `ternary_report`, trainer callbacks, fuse patch |
| `test_ternary.py` | 22 checks, CPU, ~1 min, no training. Passed on COCO n, oyster n, COCO x |
| `YoloTrainTernary.py` | ternary fine-tune of one size; `--dry-run`, `--smoke`; bakes `best_baked.pt` afterwards |
| `YoloTrainFP32.py` | full-precision oyster baseline (needed for l and x) |
| `YoloCompareTernary.py` | FP32 vs QAT vs CAT-Q table -> `evaluation_results/ternary/ternary_comparison.csv` |
| `catq_yolo.py`, `YoloCatQ.py`, `test_catq.py` | CAT-Q calibration (section 11) |
| `models/yolo11{n,s,m,l,x}.pt`, `models/v8/` | base weights |

Order:

```
python test_ternary.py                                # sanity (already passing)
python YoloTrainTernary.py --size n --smoke           # FIRST: ~1 min end-to-end check of the trainer hook
python YoloTrainTernary.py --size n                   # real run, policy A, absmean
python YoloTrainTernary.py --size n --policy B        # ablation
python YoloTrainTernary.py --size n --mode twn        # ablation
python YoloTrainTernary.py --size s ; ... --size m
python YoloTrainFP32.py --size l ; python YoloTrainFP32.py --size x      # baselines
python YoloTrainTernary.py --size l ; python YoloTrainTernary.py --size x
python YoloCompareTernary.py
```

Not verified yet: the real Ultralytics training loop with the hook (AMP, EMA, checkpoint save,
final eval). Everything around it is unit-tested with a fake trainer; `--smoke` is the first thing
to run. Defaults: AdamW lr0 5e-4, weight decay 1e-4, 100 epochs, patience 30, 10-epoch FP->ternary
blend, imgsz 960, batch 16/16/8/8/4 for n/s/m/l/x (chosen from the measurements in section 14).

## 11. CAT-Q implementation (convolutional port)

Files: `catq_yolo.py` (library), `YoloCatQ.py` (CLI), `test_catq.py` (18 checks, pass on ultralytics
8.3.228 and 8.4.175). Run order per size:

```
python YoloCatQ.py --size n --smoke                 # 16 images, 2 epochs, ~5 s on the GPU (already run once: works)
python YoloCatQ.py --size n --eval                  # real calibration (128 images @960, 60 epochs/block) + val mAP
python YoloTrainTernary.py --size n --init-from-catq   # optional: QAT refinement starting from CAT-Q
```

What it does (see the `catq_yolo.py` docstring for the formulas):
* Frozen pretrained weights. Per conv output channel three learnable factors (shift `d_mu`, scale
  `d_alpha`, threshold `d_delta`), optimised with AdamW on a small set of training images (labels not
  used) so the ternary block output matches the FP block output (MSE).
* Softened ternarization: tanh-based smooth ternarizer whose sharpness ramps 0 to 30 over the first
  80% of steps, then hard ternary with straight-through gradients through the sharp tanh.
* Result: per channel `alpha * {-1,0,+1}`, mu dropped; saved as a normal checkpoint
  `oysterTrainedModels/yolo11<size>CatQ<policy>/weights/catq_baked.pt`.
* Same layer policy A/B as QAT, so results are comparable. `YoloCompareTernary.py` picks both up.

Deviations from the LLM paper, forced by YOLO: groups are output channels; calibration data are
images; reconstruction is per top-level YOLO block (C3k2, SPPF, ...) instead of a sliding window
across layers (YOLO has skip/concat links); `--propagate` feeds each block the already-ternary
output of earlier blocks (error-aware, 2x cache memory), default feeds FP inputs. The stage-2
gradient rule is my interpretation of the paper's wording.

Hybrid: `--init-from-catq` makes QAT start exactly at the CAT-Q solution (latent weights rescaled
so the QAT quantizer reproduces the same scales and ternary values; verified, max diff ~1e-4) and
forces the FP->ternary ramp off, since the weights are already ternary. A policy-B QAT run on a
policy-A CAT-Q model keeps the A layers exact and ternarizes the rest from scratch.

**Measured result (yolo11n, policy A, 128 images @960, 60 epochs/block, 2026-10-11): CAT-Q alone
does not work on YOLO11n.** Calibration cut mean block MSE from 2.118 to 0.336, but val mAP50 and
mAP50-95 are both 0.0000. Diagnosis (not a file/bake bug): FP32 gives 92 detections at conf 0.25 on 16
val images, the CAT-Q checkpoint gives 0 even at conf 0.001. End-to-end, relative error vs the FP model
is already 0.31 after the first ternary conv (cosine 0.83) and compounds to cosine 0.23 to 0.55 in the
neck. `--propagate` (error-aware inputs, 20 epochs) also gives mAP 0. Re-estimating BN statistics
afterwards makes the model fire everywhere (confidence 1.0), i.e. not a fix. Conclusion: with frozen
weights, 1.58-bit conv layers of a nano detector cannot be recovered by output-matching calibration;
the weights themselves must be trained on the detection loss (QAT). CAT-Q remains useful only as a
possible initialiser for QAT (`--init-from-catq`), and that is untested. The paper's own LLM results
lose roughly 4 to 11 points at this bit-width even with far more redundancy. Calibration cache lives in CPU
RAM (fp16): about 128 images x all saved feature maps; this PC has 64 GB, so n/s/m/l are fine,
use `--num-images 64` for x at 960 if memory gets tight.

## 14. Measured time and memory (RTX 5060 Ti 16 GB, 64 GB RAM, 2026-10-11)

Synthetic forward+backward, AMP, 960 px, random input (no dataloader, loss, validation or optimizer
state; Ultralytics adds roughly 10 to 25% on top). 12,853 training images.

| size | batch | peak GPU GB | s/iter (QAT) | GPU-only epoch | 100 epochs |
|---|---|---|---|---|---|
| n | 16 / 32 | 4.7 / 9.3 | 0.13 / 0.25 | 1.7 min | ~2.8 h |
| s | 16 / 32 | 8.5 / **16.7 (thrash)** | 0.24 / **5.8** | 3.2 min | ~5.3 h |
| m | 8 / 16 | 8.8 / **17.2 (thrash)** | 0.27 / **5.9** | 7.2 min | ~12 h |
| l | 4 / 8 | 5.9 / 11.3 | 0.18 / 0.35 | 9.3 min | ~15.5 h |
| x | 2 / 4 / 8 | 5.2 / 9.3 / **17.3 (thrash)** | 0.18 / 0.33 / **10.4** | 17.9 min (bs 4) | ~30 h |

* Ternary (QAT) forward overhead is about 0 to 5% versus full precision.
* **VRAM is the training bottleneck.** Peak scales linearly with batch. Once peak reaches ~16 GB
  Windows spills into shared system memory without erroring and the step gets 20 to 50x slower. Keep
  the synthetic peak at or below ~11 to 12 GB. `--batch -1` (Ultralytics AutoBatch, ~60% of VRAM) is a
  safe way to find the largest batch.
* Real epochs add dataloader, augmentation and validation time on top of the GPU-only numbers.
* CAT-Q (yolo11n, 128 images @960, batch 8): 82 to 87 s per 20 epochs (propagate 113 s), so 60 epochs is
  about 4 to 4.5 min. GPU peak 1.16 GB allocated, GPU utilization only ~40% (60% propagate): it is
  not memory-bound; it is limited by per-step overhead and CPU-to-GPU copies. Process RAM peaked at
  4.7 GB (7.3 GB propagate) of 64 GB.

## 15. Results so far (yolo11n, policy A, absmean, QAT from the FP32 oyster model)

Run `yolo11nTernaryAAbsmean`: 68 epochs in 2.6 h (early stop, best epoch 38, patience 30). Baked
`best_baked.pt` has 62/88 ternary conv layers (83.4% of conv weights, 38.3% of them zero).
Table from `evaluation_results/ternary/ternary_comparison.csv`, imgsz 960:

| model | split | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|
| FP32 | val | 0.817 | 0.673 | 0.797 | 0.814 |
| FP32 | test | 0.525 | 0.373 | 0.756 | 0.489 |
| ternary QAT | val | 0.460 | 0.285 | 0.534 | 0.442 |
| ternary QAT | test | 0.370 | 0.220 | 0.595 | 0.350 |
| CAT-Q only | val / test | 0.000 | 0.000 | 0 | 0 |

* On the test split ternary QAT keeps about 70% of mAP50 (-15.5 points) and 59% of mAP50-95
  (-15.3 points). On val the gap looks much larger (-36 points), but the FP32 val score (0.82) is far
  above its test score (0.52), which suggests val images resemble training images (Roboflow-style
  augmented copies); treat test as the honest number.
* Size: file is 5.6 MB (plain fp16 container); ~1.4 MB only with custom 2-bit packing.
* Open ideas, none tried: knowledge distillation from the FP32 teacher, longer ramp / larger
  patience, TWN mode, larger sizes (hypothesis: they lose less).

## 12. Done criteria

* Baked ternary checkpoint for each of n, s, m, l, x loads with plain `YOLO(path)`.
* `count_unique` confirms <= 3 distinct values per channel in every ternary layer.
* One results table (section 8) generated by one script.

## 13. Method choice and which Python to use

* **What this plan is:** quantization-aware training (QAT) fine-tuning of a pretrained oyster model:
  latent FP weights, ternary forward pass, straight-through gradients, 10-epoch FP->ternary blend.
  It is *not* training from scratch and *not* post-training quantization.
* **Three routes:** (1) from scratch (BitNet-style, random init; most expensive; no `--scratch`
  option built yet), (2) QAT fine-tune from pretrained (this plan; `--from-coco` is the variant that
  starts from COCO weights), (3) post-training quantization with a small calibration set (CAT-Q is
  this kind). Nobody has published a ternary YOLO comparison of these, so which wins here is open.
* **CAT-Q** (arXiv 2606.26650, ICML 2026, code github.com/IntelChina-AI/BitTern) is an LLM
  post-training method: pretrained Llama/Qwen, 512 text samples, learns a per-group mean shift,
  scale and threshold (learnable modulation) and a tanh-softened ternarizer, via sliding-window
  output reconstruction over transformer layers. No vision results. Not usable as-is on YOLO; a
  convolutional port is now implemented (section 11).
* **No naive PTQ.** Plain "round the pretrained weights to ternary and stop" is not offered as a
  result anywhere. Both implemented methods learn something: QAT trains all weights, CAT-Q learns
  per-channel calibration factors against the FP model's block outputs.
* **Python:** train with `C:\Program Files\Python312\python.exe` (torch 2.9.1+cu130, sees the GPU).
  The conda `(base)` Python 3.13 has torch 2.14.1 **without CUDA** (CPU only) and ultralytics 8.4.175.
  `test_ternary.py` passes 22/22 on both ultralytics versions, but the real training loop has not
  been run on either yet (`--smoke` first).
