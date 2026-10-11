# Browser benchmark: quantized n/s models (TensorFlow.js, WebGL)

Run 2026-10-11 in a desktop browser (AMD RX 6700 XT), TF.js 4.22, 960x960 input, same preprocessing/NMS as the website
(IoU 0.5 class-agnostic NMS, 0.40 score cutoff for P/R/F1; mAP50 uses all detections >= 0.05).
Data: 120 images each from MergedData valid and test (every Nth image). GT boxes = bounding box of each label polygon.
This is NOT comparable to Ultralytics' own mAP (different NMS, tiny sample). Differences of ~0.01-0.02 are noise.

| Model | Download | Weights in RAM | Peak tensors | Inference (median) | Pipeline / img | Test F1 | Test mAP50 | Valid F1 | Valid mAP50 |
|---|---|---|---|---|---|---|---|---|---|
| v8n int8 (deployed) | 3.0 MB | 11.8 MB | 88 MB | 29 ms | 36 ms | 0.590 | 0.450 | 0.698 | 0.525 |
| v8n f32 | 11.8 MB | 11.8 MB | 88 MB | 29 ms | 32 ms | 0.587 | 0.466 | 0.679 | 0.519 |
| 11n f32 | 10.2 MB | 10.2 MB | 87 MB | 27 ms | 30 ms | 0.622 | 0.485 | 0.693 | 0.527 |
| 11n u8 (new) | 2.6 MB | 10.2 MB | 87 MB | 27 ms | 30 ms | 0.609 | 0.475 | 0.686 | 0.525 |
| v8s f32 | 42.8 MB | 42.8 MB | 162 MB | 60 ms | 63 ms | 0.646 | 0.486 | 0.731 | 0.551 |
| v8s u8 (new) | 10.7 MB | 42.8 MB | 162 MB | 60 ms | 63 ms | 0.645 | 0.479 | 0.732 | 0.550 |
| v8s u8 (existing public/yolo8Squant8) | 10.7 MB | 42.8 MB | 162 MB | 61 ms | 63 ms | 0.630 | 0.456 | 0.745 | 0.551 |
| 11s f32 | 36.3 MB | 36.3 MB | 155 MB | 54 ms | 58 ms | 0.644 | 0.518 | 0.714 | 0.549 |
| 11s u8 (new) | 9.1 MB | 36.3 MB | 155 MB | 55 ms | 57 ms | 0.639 | 0.516 | 0.703 | 0.546 |

Quantized files: `yolo11nMerged_uint8_web_model`, `yolo11sMerged_uint8_web_model`, `yolo8sMerged_uint8_web_model`
(made with `quantize_tfjs_uint8.py`, weights-only uint8; the browser expands weights back to floats, so runtime memory is unchanged).
