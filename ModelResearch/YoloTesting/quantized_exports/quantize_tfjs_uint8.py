"""Weight-only uint8 quantization of a TF.js graph model (same on-disk format tensorflowjs_converter
--quantize_uint8 produces). CPU/numpy only; reads the float web model, writes a NEW folder."""
import json, math, os, shutil, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
SHARD = 4 * 1024 * 1024
m = json.load(open(os.path.join(src, "model.json")))
assert len(m["weightsManifest"]) == 1, "expected a single weight group"
group = m["weightsManifest"][0]
blob = b"".join(open(os.path.join(src, p), "rb").read() for p in group["paths"])

out, off, worst = [], 0, 0.0
nq = 0
for w in group["weights"]:
    n = math.prod(w["shape"]) if w["shape"] else 1
    size = n * 4  # float32 and int32 are both 4 bytes
    raw = blob[off:off + size]; off += size
    if w["dtype"] != "float32":
        out.append(raw); continue
    x = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    lo, hi = min(float(x.min()), 0.0), max(float(x.max()), 0.0)   # range must include 0
    if hi == lo:                                                   # constant tensor
        scale, zmin, q = 1.0, float(x.min()), np.zeros(n, dtype=np.uint8)
    else:
        scale = (hi - lo) / 255.0
        zp = round(-lo / scale)                                    # nudge so 0.0 is exactly representable
        zmin = -zp * scale
        q = np.clip(np.round(x / scale) + zp, 0, 255).astype(np.uint8)
    deq = q.astype(np.float64) * scale + zmin
    worst = max(worst, float(np.abs(deq - x).max() / max(scale, 1e-30)))   # in units of one quant step
    w["quantization"] = {"dtype": "uint8", "min": zmin, "scale": scale, "original_dtype": "float32"}
    out.append(q.tobytes()); nq += 1
assert off == len(blob), (off, len(blob))

data = b"".join(out)
os.makedirs(dst, exist_ok=False)          # refuse to overwrite anything that exists
shards = [data[i:i + SHARD] for i in range(0, len(data), SHARD)]
group["paths"] = [f"group1-shard{i+1}of{len(shards)}.bin" for i in range(len(shards))]
for p, s in zip(group["paths"], shards):
    open(os.path.join(dst, p), "wb").write(s)
json.dump(m, open(os.path.join(dst, "model.json"), "w"))
meta = os.path.join(src, "metadata.yaml")
if os.path.exists(meta): shutil.copy(meta, os.path.join(dst, "metadata.yaml"))
print(f"quantized {nq} float32 tensors; {len(blob)/1048576:.2f} MB -> {len(data)/1048576:.2f} MB in {len(shards)} shard(s)")
print(f"worst per-weight error = {worst:.3f} quantization steps (0.5 is the theoretical max)")
