import numpy as np
import torch

from ogw.anomaly import PCADetector
from ogw.embedded import (PCAModule, NormalizedCNN, run, export, quantize_dynamic,
                          quantize_static, fold_normalization, to_2d,
                          pad_channels)
from ogw.models import CNN1D
from tests.test_anomaly import dataset


def test_cnn_onnx_faithful_and_int8_close(tmp_path):
    torch.manual_seed(0)
    m = CNN1D(4, widths=(8, 8), kernel=5)
    X = np.random.default_rng(0).normal(size=(64, 4, 100)).astype(np.float32)
    scale = np.full((4, 1), 2.0, dtype=np.float32)
    mod = NormalizedCNN(m, scale)
    f32 = export(mod, tmp_path / "cnn.onnx", (4, 100))
    ref = mod(torch.from_numpy(X)).detach().numpy()
    assert np.abs(run(f32, X) - ref).max() < 1e-4
    i8 = quantize_static(f32, tmp_path / "cnn_i8.onnx", X[:32])
    assert np.corrcoef(run(i8, X), ref)[0, 1] > 0.95


def test_pca_module_identical_to_detector(tmp_path):
    X, y = dataset()
    det = PCADetector(k=2).fit(X, np.arange(60), np.arange(60, 90))
    mod = PCAModule(det)
    s, faulty = mod(torch.from_numpy(X[90:]))
    assert not faulty.any()
    assert np.allclose(s.numpy(), det.scores(np.arange(90, 120)), rtol=1e-3)
    f32 = export(mod, tmp_path / "pca.onnx", X.shape[1:], outputs=("indicator", "faulty"))
    assert np.allclose(run(f32, X[90:]), s.numpy(), rtol=1e-3)
    i8 = quantize_dynamic(f32, tmp_path / "pca_i8.onnx")
    assert run(i8, X[90:]).min() > det.threshold      # damage still detected


def test_pca_float16(tmp_path):
    X, _ = dataset()
    det = PCADetector(k=2).fit(X, np.arange(60), np.arange(60, 90))
    s32 = PCAModule(det)(torch.from_numpy(X))[0].numpy()
    s16 = PCAModule(det, torch.float16)(torch.from_numpy(X))[0].numpy()
    assert np.corrcoef(s32, s16)[0, 1] > 0.99


def test_pca_int8_weights(tmp_path):
    X, _ = dataset()
    det = PCADetector(k=2).fit(X, np.arange(60), np.arange(60, 90))
    mod = PCAModule(det, int8_weights=True)
    assert mod.q_V.dtype == torch.int8 and not hasattr(mod, "V")
    s = mod(torch.from_numpy(X))[0].numpy()
    assert np.corrcoef(s, PCAModule(det)(torch.from_numpy(X))[0].numpy())[0, 1] > 0.99
    f = export(mod, tmp_path / "pca_q.onnx", X.shape[1:], outputs=("indicator", "faulty"))
    assert np.allclose(run(f, X), s, rtol=1e-3)


def test_normalization_folding():
    torch.manual_seed(0)
    m = CNN1D(4, widths=(8, 8), kernel=5).eval()
    scale = np.array([[0.5], [2.0], [3.0], [1.0]], dtype=np.float32)
    x = torch.randn(6, 4, 100)
    expected = NormalizedCNN(m, scale)(x)
    assert torch.allclose(fold_normalization(m, scale)(x), expected, atol=1e-5)
    assert torch.allclose(NormalizedCNN(m, scale)(x), expected)          # the original is not modified


def test_exact_2d_rewrite():
    torch.manual_seed(0)
    m = CNN1D(4, widths=(8, 8, 16), kernel=5).eval()
    for bn in [x for x in m.modules() if isinstance(x, torch.nn.BatchNorm1d)]:
        bn.running_mean.uniform_(-1, 1); bn.running_var.uniform_(0.5, 2)   # non-trivial BatchNorm
    x = torch.randn(3, 4, 100)
    assert torch.allclose(to_2d(m, 100)(x.unsqueeze(2)).squeeze(1), m(x), atol=1e-5)


def test_cascaded_global_max_exact():
    torch.manual_seed(0)
    m = CNN1D(4, widths=(8, 8, 8, 8), kernel=5).eval()     # 3 MaxPool(2): 820 -> 102 = 2 x 3 x 17
    x = torch.randn(3, 4, 820)
    a = to_2d(m, 820)(x.unsqueeze(2))
    b = to_2d(m, 820, cascaded_global_max=True)(x.unsqueeze(2))
    assert torch.allclose(a, b)
    assert sum(isinstance(c, torch.nn.MaxPool2d) for c in to_2d(m, 820, True)) == 6
    c = to_2d(m, 820, cascaded_global_max="small")
    assert torch.allclose(a, c(x.unsqueeze(2)))
    assert max(k.kernel_size[1] for k in c if isinstance(k, torch.nn.MaxPool2d)) == 3


def test_pad_channels_exact():
    torch.manual_seed(0)
    m2 = to_2d(CNN1D(66, widths=(32, 32, 64, 64)).eval(), 820)
    x = torch.randn(2, 66, 1, 820)
    c = pad_channels(m2)
    xc = torch.cat([x, torch.zeros(2, 14, 1, 820)], dim=1)
    assert torch.allclose(c(xc)[:, 0], m2(x)[:, 0], atol=1e-5)
    assert c[0].in_channels == 80 and c[-1].out_features == 16
