"""Static int8 (QDQ) quantization of the YOLO11n ONNX model, CPU only, low priority.
Variants:  A = every Conv/MatMul quantized      B = same, but the detection head's final convs + DFL kept in float."""
import ctypes, os, sys, glob, time
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL: be nice to the training run

import numpy as np
from PIL import Image
import onnx
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, CalibrationMethod,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

SRC = r"B:\Github\Oyster-Detection\ModelResearch\YoloTesting\oysterTrainedModels\yolo11nMergedData960\weights\best.onnx"
TRAIN = r"B:\Github\Oyster-Detection\ModelResearch\YoloTesting\MergedData\train\images"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ortsite", "models")
N_CAL = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SIZE = 960


def prep(path):
    """Same preprocessing as the website: pad to square (bottom/right), resize to 960, /255, CHW."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = max(w, h)
    sq = Image.new("RGB", (s, s), (0, 0, 0))
    sq.paste(im, (0, 0))
    a = np.asarray(sq.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    return np.ascontiguousarray(a.transpose(2, 0, 1))[None]


class Reader(CalibrationDataReader):
    def __init__(self, files):
        self.files, self.i = files, 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        x = prep(self.files[self.i]); self.i += 1
        return {"images": x}

    def rewind(self):
        self.i = 0


files = sorted(glob.glob(os.path.join(TRAIN, "*.jpg")))
step = max(1, len(files) // N_CAL)
cal = files[::step][:N_CAL]
print(f"calibration images: {len(cal)} (every {step}th of {len(files)} training images)", flush=True)

pre = os.path.join(OUT, "yolo11n_pre.onnx")
t = time.time()
quant_pre_process(SRC, pre, skip_symbolic_shape=False)
print(f"pre-processed in {time.time()-t:.1f}s", flush=True)

m = onnx.load(pre)
head_final = [n.name for n in m.graph.node if n.op_type == "Conv" and n.name.startswith("/model.23/")
              and (n.name.endswith(".2/Conv") or "/dfl/" in n.name)]
print("kept in float for variant B:", len(head_final), head_final[:8], flush=True)

variants = [
    ("C", head_final, dict(quant_format=QuantFormat.QOperator, calibrate_method=CalibrationMethod.MinMax, extra_options={})),
    ("D", head_final, dict(quant_format=QuantFormat.QDQ, calibrate_method=CalibrationMethod.Percentile, extra_options={"CalibPercentile": 99.999})),
]
for tag, exclude, kw in variants:
    t = time.time()
    dst = os.path.join(OUT, f"yolo11n_int8_{tag}.onnx")
    quantize_static(pre, dst, Reader(cal), activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, nodes_to_exclude=exclude, op_types_to_quantize=["Conv", "MatMul"], **kw)
    print(f"variant {tag}: {os.path.getsize(dst)/1048576:.2f} MB in {time.time()-t:.1f}s", flush=True)
print("done", flush=True)
