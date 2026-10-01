"""Step 4: embedded deployment (ONNX export, quantization, size, latency).

Plan fixed before the measurements:
- CNN: static int8 (QDQ, per-channel weights), calibrated on 128 training
  measurements drawn at random (seed 0); 5 splits x 5 seeds;
- PCA (k = 5): float32, float16 (buffers), dynamic int8 (MatMul weights);
  "int8 weights" variant, added after the first measurements (dynamic int8
  kept a float32 copy of the components), evaluated with the same protocol;
  threshold recalibrated on the healthy validation measurements for each variant;
- test set already opened in Step 3: nothing is chosen on it, only the degradation is measured;
- latency: ONNX Runtime, batch of 1, 1 thread, 300 inferences after 30 warmup runs.
Run (from the repository root): .venv/bin/python step04_embedded_export/embedded_export.py
"""
import json
import logging
import platform
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ogw.anomaly import PCADetector
from ogw.data import Normalizer, load_meta, load_signals
from ogw.embedded import (NormalizedCNN, PCAModule, export, latency, quantize_dynamic, quantize_static,
                          run)
from ogw.models import CNN1D
from ogw.splits import by_temperature, position_folds

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
RES = Path(__file__).resolve().parents[1] / "results"
OUT, ONNX = RES / "step04_embedded_export", RES / "step04_embedded_export" / "onnx"
ONNX.mkdir(parents=True, exist_ok=True)
meta, X = load_meta(), load_signals()
splits = [*position_folds(meta), by_temperature(meta)]
shape = X.shape[1:]


def size_kb(path):
    return sum(p.stat().st_size for p in Path(path).parent.glob(Path(path).name + "*")) / 1024


def cpu():
    for l in open("/proc/cpuinfo"):
        if l.startswith("model name"):
            return l.split(":", 1)[1].strip()
    return platform.processor()


# --- 1. CNN: export fidelity and int8 quantization ---------------------------
cnn = []
for d in splits:
    scale = Normalizer().fit(X[d.train]).scale
    calib = X[np.random.default_rng(0).choice(d.train, 128, replace=False)]
    y = d.target[d.test]
    for g in range(5):
        m = CNN1D()
        m.load_state_dict(torch.load(RES / "step03_cnn_vs_pca" / "models" / f"cnn_{d.name}_g{g}.pt"))
        mod = NormalizedCNN(m, scale)
        f32 = export(mod, ONNX / f"cnn_{d.name}_g{g}.onnx", shape, ("logit",))
        i8 = quantize_static(f32, ONNX / f"cnn_{d.name}_g{g}_int8.onnx", calib)
        with torch.no_grad():
            s_t = torch.cat([mod(torch.from_numpy(X[d.test][i:i + 64])) for i in range(0, len(y), 64)]).numpy()
        s32, s8 = run(f32, X[d.test]), run(i8, X[d.test])
        cnn.append(dict(split=d.name, seed=g, onnx_diff=float(np.abs(s32 - s_t).max()),
                        auc_torch=float(roc_auc_score(y, s_t)), auc_f32=float(roc_auc_score(y, s32)),
                        auc_int8=float(roc_auc_score(y, s8)),
                        decision_agreement=float(((s32 > 0) == (s8 > 0)).mean()),
                        correlation=float(np.corrcoef(s32, s8)[0, 1])))
        print(f"CNN {d.name:14s} g{g}: ONNX diff {cnn[-1]['onnx_diff']:.1e}, AUC f32 "
              f"{cnn[-1]['auc_f32']:.3f} int8 {cnn[-1]['auc_int8']:.3f}, agreement {cnn[-1]['decision_agreement']:.3f}",
              flush=True)

# --- 2. PCA: float32, float16, int8 variants ----------------------------------
healthy1 = (meta.damaged == 0) & (meta.cycle == 1)
pca_models = {}
e, v = np.flatnonzero(healthy1 & (meta.ramp == 0)), np.flatnonzero(healthy1 & (meta.ramp == 1))
pca_models["position"] = (PCADetector(5).fit(X, e, v), v)
dt = splits[-1]
se = dt.train[meta.damaged[dt.train] == 0]
sv = dt.validation[meta.damaged[dt.validation] == 0]
pca_models["temperature"] = (PCADetector(5).fit(X, se, sv), sv)

pca = []
for key, (det, healthy_val) in pca_models.items():
    f32 = export(PCAModule(det), ONNX / f"pca_{key}.onnx", shape, ("indicator", "faulty"))
    f16 = export(PCAModule(det, torch.float16), ONNX / f"pca_{key}_f16.onnx", shape, ("indicator", "faulty"))
    i8 = quantize_dynamic(f32, ONNX / f"pca_{key}_int8.onnx")
    # added after the measurements of the initial plan: dynamic int8 kept the float32 weights
    q8 = export(PCAModule(det, int8_weights=True), ONNX / f"pca_{key}_int8_weights.onnx", shape,
                ("indicator", "faulty"))
    variants = {"python (masked projection)": det.scores, "ONNX float32": lambda idx, c=f32: run(c, X[idx]),
                "ONNX float16": lambda idx, c=f16: run(c, X[idx]),
                "ONNX dynamic int8": lambda idx, c=i8: run(c, X[idx]),
                "ONNX int8 weights": lambda idx, c=q8: run(c, X[idx])}
    for variant_name, f in variants.items():
        threshold = np.quantile(f(healthy_val), 0.95)
        for d in [x for x in splits if (x.name == "temperature") == (key == "temperature")]:
            y, s = d.target[d.test], f(d.test)
            pca.append(dict(model=key, variant=variant_name, split=d.name, auc=float(roc_auc_score(y, s)),
                            fpr=float((s[y == 0] > threshold).mean()), tpr=float((s[y == 1] > threshold).mean())))
            print(f"PCA {variant_name:28s} {d.name:14s} AUC {pca[-1]['auc']:.3f} FPR {pca[-1]['fpr']:.3f} "
                  f"TPR {pca[-1]['tpr']:.3f}", flush=True)

# --- 3. Size and latency -----------------------------------------------------------
x1 = X[splits[0].test[:1]]
files = {"CNN float32": ONNX / "cnn_position_D04_g0.onnx", "CNN int8": ONNX / "cnn_position_D04_g0_int8.onnx",
         "PCA float32": ONNX / "pca_position.onnx", "PCA float16": ONNX / "pca_position_f16.onnx",
         "PCA dynamic int8": ONNX / "pca_position_int8.onnx",
         "PCA int8 weights": ONNX / "pca_position_int8_weights.onnx"}
perf = {}
for name, c in files.items():
    med, p95 = latency(c, x1)
    perf[name] = dict(size_kb=size_kb(c), latency_ms=med, latency_p95_ms=p95)
    print(f"{name:12s} {perf[name]['size_kb']:8.1f} KB  {med:.3f} ms (p95 {p95:.3f})", flush=True)

(OUT / "measurements.json").write_text(json.dumps(dict(cnn=cnn, pca=pca, perf=perf, cpu=cpu()), ensure_ascii=False,
                                                  indent=1))

# --- 4. Summary ---------------------------------------------------------------------
L = ["# Step 4: embedded deployment (40 kHz)", "",
     f"Generated by `step04_embedded_export/embedded_export.py`. Latencies measured on {cpu()}, ONNX Runtime, "
     "batch of 1, 1 thread: relative orders of magnitude, **not** microcontroller latencies.", "",
     "## Size and latency", "", "| model | ONNX file | median latency | 95th percentile |", "|---|---|---|---|"]
for name, p in perf.items():
    L.append(f"| {name} | {p['size_kb']:.0f} KB | {p['latency_ms']:.3f} ms | {p['latency_p95_ms']:.3f} ms |")
L += ["", "## CNN: export fidelity and effect of int8 (test, 5 seeds)", "",
      "| split | max ONNX/PyTorch diff | AUC float32 | AUC int8 | decision agreement | logit correlation |",
      "|---|---|---|---|---|---|"]
for d in splits:
    b = [x for x in cnn if x["split"] == d.name]
    mean = lambda k: np.mean([x[k] for x in b])
    L.append(f"| {d.name} | {max(x['onnx_diff'] for x in b):.1e} | {mean('auc_f32'):.3f} | {mean('auc_int8'):.3f} "
             f"| {mean('decision_agreement'):.3f} | {mean('correlation'):.4f} |")
L += ["", "## PCA: variants (threshold recalibrated on validation for each one)", "",
      "| variant | AUC (4 positions) | FPR / TPR (4 positions) | AUC temperature | FPR / TPR temperature |",
      "|---|---|---|---|---|"]
for variant_name in dict.fromkeys(a["variant"] for a in pca):
    p = [a for a in pca if a["variant"] == variant_name and a["model"] == "position"]
    t = [a for a in pca if a["variant"] == variant_name and a["model"] == "temperature"][0]
    L.append(f"| {variant_name} | {np.mean([a['auc'] for a in p]):.3f} (min {min(a['auc'] for a in p):.3f}) "
             f"| {np.mean([a['fpr'] for a in p]):.3f} / {np.mean([a['tpr'] for a in p]):.3f} "
             f"| {t['auc']:.3f} | {t['fpr']:.3f} / {t['tpr']:.3f} |")
(OUT / "summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
