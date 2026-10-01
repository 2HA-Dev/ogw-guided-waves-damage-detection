"""ONNX export, quantization, size and latency measurement.

The exported graphs take preprocessed signals as input (centred, decimated,
filtered: pipeline of ogw.data, outside the graph) and include the per-path
normalization learned during training.
"""
import copy
import os
import time

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType
from onnxruntime.quantization import quantize_dynamic as _ort_quantize_dynamic
from onnxruntime.quantization import quantize_static as _ort_quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch import nn


class NormalizedCNN(nn.Module):
    """CNN preceded by its per-path normalization (scale learned on the training set)."""

    def __init__(self, model, scale):
        super().__init__()
        self.model = model.eval()
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return self.model(x / self.scale)


def fold_normalization(model, scale):
    """CNN equivalent to NormalizedCNN(model, scale), without a division operator.

    conv(x / s) = conv'(x) with w'[o, i, k] = w[o, i, k] / s[i]: the per-path
    normalization is absorbed into the weights of the first convolution.
    """
    m = copy.deepcopy(model).eval()
    conv = m.features[0]
    s = torch.as_tensor(np.asarray(scale, dtype=np.float32)).reshape(1, -1, 1)
    with torch.no_grad():
        conv.weight /= s
    return m


def _prime_factors(n):
    f, d = [], 2
    while n > 1:
        while n % d == 0:
            f.append(d)
            n //= d
        d += 1
    return f


def to_2d(model, L, cascaded_global_max=False):
    """Exact rewriting of the 1D CNN in 2D (height 1), for tools that only support
    2D operators (Aidge 0.10.1 quantization): Conv1d -> Conv2d (1, k),
    MaxPool1d -> MaxPool2d (1, 2), global maximum -> MaxPool2d (1, final width).
    Input (batch, 66, 1, L). No retraining: same weights, same output.
    cascaded_global_max: the global maximum (final width, 102 to 820 points)
    is replaced by successive MaxPool2d with kernels equal to the prime
    factors of the width (102 = 2 × 3 × 17), identical result; works around
    the error of the Aidge 0.10.1 int8 kernel on a wide pooling.
    cascaded_global_max="small": kernels of width 3 at most, with rounding
    up (ceil_mode), to avoid any wide pooling.
    """
    layers, width = [], L
    for m in model.features:
        if isinstance(m, nn.Conv1d):
            c = nn.Conv2d(m.in_channels, m.out_channels, (1, m.kernel_size[0]), padding=(0, m.padding[0]))
            c.weight.data = m.weight.data.unsqueeze(2).clone()
            c.bias.data = m.bias.data.clone()
            layers.append(c)
        elif isinstance(m, nn.BatchNorm1d):
            b = nn.BatchNorm2d(m.num_features)
            b.load_state_dict(m.state_dict())
            layers.append(b)
        elif isinstance(m, nn.MaxPool1d):
            layers.append(nn.MaxPool2d((1, m.kernel_size)))
            width //= m.kernel_size
        else:
            layers.append(copy.deepcopy(m))
    lin = model.head[-1]
    if cascaded_global_max == "small":
        # kernels of width 3 (2 at the end), rounding up: no value is
        # ignored, no padding (102 -> 34 -> 12 -> 4 -> 2 -> 1)
        n = width
        while n > 1:
            k = 3 if n > 2 else 2
            layers.append(nn.MaxPool2d((1, k), ceil_mode=True))
            n = -(-n // k)
    elif cascaded_global_max:
        layers += [nn.MaxPool2d((1, f)) for f in _prime_factors(width)]
    else:
        layers.append(nn.MaxPool2d((1, width)))
    layers += [nn.Flatten(), copy.deepcopy(lin)]
    return nn.Sequential(*layers).eval()


def pad_channels(model_2d, multiple=16):
    """Exact workaround for the Aidge 0.10.1 int8 bug (wrong convolution when its channels
    are not multiples of 16, see results/step09_aidge_int8_case): the first convolution
    receives extra zero input channels (zero weights) and the final layer extra zero
    outputs; the logit is output 0. The input must be padded with zeros up to
    in_channels (66 -> 80).
    """
    m = copy.deepcopy(model_2d)
    conv = next(c for c in m if isinstance(c, nn.Conv2d))
    c_in = -(-conv.in_channels // multiple) * multiple
    w = torch.zeros(conv.out_channels, c_in, *conv.kernel_size)
    w[:, :conv.in_channels] = conv.weight.data
    new = nn.Conv2d(c_in, conv.out_channels, conv.kernel_size, padding=conv.padding)
    new.weight.data, new.bias.data = w, conv.bias.data.clone()
    lin = m[-1]
    n_out = -(-lin.out_features // multiple) * multiple
    nl = nn.Linear(lin.in_features, n_out)
    nl.weight.data.zero_(); nl.bias.data.zero_()
    nl.weight.data[:lin.out_features], nl.bias.data[:lin.out_features] = lin.weight.data, lin.bias.data
    layers = list(m)
    layers[layers.index(conv)], layers[-1] = new, nl
    return nn.Sequential(*layers).eval()


class PCAModule(nn.Module):
    """PCA detector as a graph: (batch, 66, L) -> (indicator, faulty paths).

    Difference with ogw.anomaly: the projection covers all paths (no linear
    system solved per measurement, which standard ONNX operators lack); the
    faulty paths are only excluded from the maximum.

    dtype: storage of the buffers (float32 or float16). int8_weights: mean and
    components stored in symmetric int8 (one scale per path for the mean, one
    per component), dequantized at compute time (weight-only quantization).
    quality_check=False: neither boolean output nor boolean mask (faulty paths
    to be handled outside the graph), for tools without a boolean type.
    """

    def __init__(self, det, dtype=torch.float32, int8_weights=False, quality_check=True):
        super().__init__()
        self.quality_check = quality_check
        buffers = dict(norm=det.norm.scale, mean=det.mean, V=det.components,
                       res_scale=det.scale, energy=det.typical_energy)
        for name, v in buffers.items():
            self.register_buffer(name, torch.as_tensor(np.asarray(v, dtype=np.float32)).to(dtype))
        self.int8_weights = int8_weights
        if int8_weights:
            c = det.norm.scale.shape[0]
            mu = self.mean.float().reshape(c, -1)
            V = self.V.float()
            del self.mean, self.V
            for name, w in (("mu", mu), ("V", V)):              # scale per path / per component
                q, sc = self._quantize(w)
                self.register_buffer(f"q_{name}", q)
                self.register_buffer(f"s_{name}", sc)

    def _quantize(self, w):
        s = w.abs().amax(1, keepdim=True) / 127
        return torch.round(w / s).to(torch.int8), s

    def _weights(self):
        if self.int8_weights:
            return (self.q_mu.float() * self.s_mu).reshape(-1), self.q_V.float() * self.s_V
        return self.mean.float(), self.V.float()

    def forward(self, x):
        n, c, L = x.shape
        faulty = (x ** 2).sum(-1) < 0.3 * self.energy.float()
        xn = x / self.norm.float()
        mean, V = self._weights()
        f = xn.reshape(n, c * L) - mean
        r = (f - (f @ V.T) @ V).reshape(n, c, L)
        res = (r ** 2).sum(-1) / (xn ** 2).sum(-1).clamp_min(1e-12) / self.res_scale.float()
        if not self.quality_check:          # quality check outside the graph (no boolean)
            return res.amax(1)
        res = torch.where(faulty, torch.zeros_like(res), res)
        return res.amax(1), faulty


def export(module, path, shape, outputs=("output",)):
    x = torch.zeros(1, *shape)
    prog = torch.onnx.export(module.eval(), (x,), dynamo=True, input_names=["x"],
                             output_names=list(outputs),
                             dynamic_shapes=({0: torch.export.Dim("batch")},), verbose=False)
    prog.save(str(path))
    return path


class _Reader(CalibrationDataReader):
    def __init__(self, X):
        self.it = iter([{"x": X[i:i + 1]} for i in range(len(X))])

    def get_next(self):
        return next(self.it, None)


def quantize_static(path_f32, path_i8, X_calib):
    """Static int8 (activations and weights), QDQ format, per-channel weights."""
    pre = str(path_i8).replace(".onnx", "_pre.onnx")
    quant_pre_process(str(path_f32), pre, skip_symbolic_shape=True)
    _ort_quantize_static(pre, str(path_i8), _Reader(X_calib), quant_format=QuantFormat.QDQ,
                         per_channel=True, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    os.remove(pre)
    return path_i8


def quantize_dynamic(path_f32, path_i8):
    """Dynamic int8: MatMul weights in int8, activations quantized on the fly."""
    _ort_quantize_dynamic(str(path_f32), str(path_i8), weight_type=QuantType.QInt8)
    return path_i8


def session(path, threads=1):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def run(path, X, batch=64):
    s = session(path, threads=4)
    return np.concatenate([s.run(None, {"x": X[i:i + batch]})[0] for i in range(0, len(X), batch)])


def latency(path, x, n=300, warmup=30):
    """Latency of one inference (batch of 1, 1 thread): median and 95th percentile, in milliseconds."""
    s = session(path, threads=1)
    for _ in range(warmup):
        s.run(None, {"x": x})
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        s.run(None, {"x": x})
        t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1e3), float(np.percentile(t, 95) * 1e3)
