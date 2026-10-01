"""Step 5: prepares the data shared by both environments (.venv and .venv-aidge).

Writes data/step05_inputs.npz: for the position_D04 fold (CNN model seed 0)
and the "position" PCA model:
  calib      128 training measurements (seed 0), for quantization;
  test       64 test measurements (32 healthy, 32 damaged, seed 0) and their labels;
  ort_*      reference ONNX Runtime outputs (float32 and int8) on these measurements.
Also writes into results/step05_aidge_export/ (not versioned): cnn_folded.onnx (folded
normalization), cnn_folded_2d.onnx (exact 2D rewrite), pca_no_check*.onnx.
Run (from the repository root): .venv/bin/python step05_aidge_export/prepare_data.py
"""
import numpy as np
import torch

from ogw.anomaly import PCADetector
from ogw.data import ROOT, Normalizer, load_meta, load_signals
from ogw.embedded import PCAModule, export, fold_normalization, pad_channels, run, to_2d
from ogw.models import CNN1D
from ogw.splits import position_folds

ONNX = ROOT / "results" / "step04_embedded_export" / "onnx"
meta, X = load_meta(), load_signals()
d = position_folds(meta)[0]
assert d.name == "position_D04"
rng = np.random.default_rng(0)
calib = rng.choice(d.train, 128, replace=False)
y_test = d.target[d.test]
test = np.r_[rng.choice(d.test[y_test == 0], 32, replace=False), rng.choice(d.test[y_test == 1], 32, replace=False)]
# CNN with the normalization folded into the first convolution (Aidge 0.10.1 graph adaptation
# fails on the division by a (66, 1) scale, see the README, section "C++ export with Aidge")
m = CNN1D()
m.load_state_dict(torch.load(ROOT / "results" / "step03_cnn_vs_pca" / "models" / "cnn_position_D04_g0.pt"))
folded = fold_normalization(m, Normalizer().fit(X[d.train]).scale)
(ROOT / "results" / "step05_aidge_export").mkdir(parents=True, exist_ok=True)
S5 = ROOT / "results" / "step05_aidge_export"
f_folded = export(folded, S5 / "cnn_folded.onnx", X.shape[1:], ("logit",))
# exact 2D rewrite (Aidge 0.10.1 quantization does not handle MaxPooling1D)
export(to_2d(folded, X.shape[2]), S5 / "cnn_folded_2d.onnx", (X.shape[1], 1, X.shape[2]), ("logit",))
# Step 9: cascaded global maximum (works around the error of wide int8 pooling)
export(to_2d(folded, X.shape[2], cascaded_global_max=True), S5 / "cnn_folded_2d_cascade.onnx",
       (X.shape[1], 1, X.shape[2]), ("logit",))
export(to_2d(folded, X.shape[2], cascaded_global_max="small"), S5 / "cnn_folded_2d_small.onnx",
       (X.shape[1], 1, X.shape[2]), ("logit",))
# Step 9: channels padded to multiples of 16 (input 66 -> 80, output 1 -> 16)
export(pad_channels(to_2d(folded, X.shape[2])), S5 / "cnn_folded_2d_16.onnx", (80, 1, X.shape[2]), ("logit",))
# PCA without quality check in the graph (the Aidge export templates ignore the boolean type)
healthy1 = (meta.damaged == 0) & (meta.cycle == 1)
det = PCADetector(5).fit(X, np.flatnonzero(healthy1 & (meta.ramp == 0)), np.flatnonzero(healthy1 & (meta.ramp == 1)))
f_pca = export(PCAModule(det, quality_check=False), S5 / "pca_no_check.onnx", X.shape[1:], ("indicator",))
f_pca8 = export(PCAModule(det, int8_weights=True, quality_check=False), S5 / "pca_no_check_int8_weights.onnx",
                X.shape[1:], ("indicator",))
np.savez(ROOT / "data" / "step05_inputs.npz", ort_cnn_folded=run(f_folded, X[test]),
         ort_pca_nc=run(f_pca, X[test]), ort_pca_nc_q8=run(f_pca8, X[test]),
         faulty_test=det.faulty[test],
         calib=X[calib], test=X[test], y_test=d.target[test], idx_test=test,
         ort_cnn_f32=run(ONNX / "cnn_position_D04_g0.onnx", X[test]),
         ort_cnn_int8=run(ONNX / "cnn_position_D04_g0_int8.onnx", X[test]),
         ort_pca_f32=run(ONNX / "pca_position.onnx", X[test]))
print("written:", ROOT / "data" / "step05_inputs.npz")
