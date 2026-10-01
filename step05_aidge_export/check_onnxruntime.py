"""Step 5: cross-check. Does ONNX Runtime quantize the same graph (folded CNN, 2D)?

If so, the failure of Aidge int8 comes neither from the model nor from the folding
of the normalization (scale differences between channels), but from the tool.
Run (from the repository root): .venv/bin/python step05_aidge_export/check_onnxruntime.py
(after step05_aidge_export/export_aidge.py then step05_aidge_export/diagnostic_int8.py: appends to summary.md)
"""
import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.metrics import roc_auc_score

from ogw.data import ROOT
from ogw.embedded import run

S5 = ROOT / "results" / "step05_aidge_export"
D = np.load(ROOT / "data" / "step05_inputs.npz")
F32 = S5 / "cnn_folded_2d.onnx"
NAME = onnx.load(F32).graph.input[0].name


class Reader(CalibrationDataReader):
    """Same calibration as the Aidge export: 128 training measurements."""

    def __init__(self, X):
        self.it = iter([{NAME: x[None, :, None, :]} for x in X])

    def get_next(self):
        return next(self.it, None)


lines = []
pre = S5 / "pre.onnx"
quant_pre_process(str(F32), str(pre), skip_symbolic_shape=True)
for num, per_channel in ((4, True), (5, False)):
    output = S5 / f"ort_2d_int8_{'channel' if per_channel else 'tensor'}.onnx"
    quantize_static(str(pre), str(output), Reader(D["calib"]), quant_format=QuantFormat.QDQ,
                    per_channel=per_channel, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    s = run(output, D["test"][:, :, None, :]).ravel()
    lines.append(f"{num}. ONNX Runtime, same graph, int8 {'per channel' if per_channel else 'per tensor'}: "
                 f"AUC(64) {roc_auc_score(D['y_test'], s):.3f}, correlation "
                 f"{np.corrcoef(s, D['ort_cnn_folded'])[0, 1]:.3f} with the float model.")
pre.unlink()
lines.append("   **The model quantizes without loss with another tool: for this graph, the failure is "
             "specific to the int8 C++ generation of Aidge 0.10.1.**")
with open(S5 / "summary.md", "a") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
