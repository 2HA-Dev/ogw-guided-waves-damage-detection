import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ogw.data import Normalizer, OGWSignals, load_signals, ogw_file, preprocess

FS = 10e6


def sine(f, n=13108, amp=1.0):
    return amp * np.sin(2 * np.pi * f * np.arange(n) / FS)


def test_preprocess_removes_offset():
    y = preprocess(np.full((1, 13108), 0.5))         # DC offset only
    assert np.abs(y).max() < 1e-6


def test_preprocess_keeps_useful_band():
    y = preprocess(sine(40e3)[None])                 # 40 kHz, q = 16
    assert y.shape == (1, 820) and y.dtype == np.float32
    middle = y[0, 100:-100]                          # away from the filter edge effects
    assert abs(middle.std() * np.sqrt(2) - 1.0) < 0.02   # amplitude preserved


@pytest.mark.parametrize("f", [2e3, 150e3, 2e6])
def test_preprocess_rejects_out_of_band(f):
    # 2 kHz: slow drift; 150 kHz: above 2f; 2 MHz: beyond Nyquist (312 kHz)
    y = preprocess(sine(f)[None])
    assert y[0, 100:-100].std() < 0.05


def test_normalizer_uses_training_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3, 100)).astype(np.float32)
    X[5:] *= 100                                     # "test" with a very different scale
    n = Normalizer().fit(X[:5])
    assert np.allclose(n.apply(X[:5]).std(axis=(0, 2)), 1, atol=1e-5)
    assert n.scale.shape == (3, 1)


def test_dataset_two_modes():
    X = np.zeros((4, 66, 50), dtype=np.float32)
    target = np.array([0, 1, 0, 1])
    ds = OGWSignals(X, target, indices=[1, 3])
    assert len(ds) == 2 and ds[0][0].shape == (66, 50) and ds[0][1] == 1
    ds = OGWSignals(X, target, indices=[1, 3], per_path=True, paths=range(10))
    assert len(ds) == 20 and ds[0][0].shape == (1, 50)
    xb, yb = next(iter(DataLoader(ds, batch_size=8)))
    assert xb.shape == (8, 1, 50) and yb.dtype == torch.float32


@pytest.mark.skipif(not ogw_file().exists(), reason="data missing")
def test_real_cache():
    X = load_signals()
    assert X.shape == (966, 66, 820) and X.dtype == np.float32
    assert np.isfinite(X).all()
    assert np.abs(X.mean(axis=-1)).max() < 1e-3      # centred signals
