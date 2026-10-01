import numpy as np
import torch

from ogw.splits import Split
from ogw.training import Config, train
from ogw.models import CNN1D, n_mac, n_parameters


def test_shape_and_cost():
    m = CNN1D(66)
    assert m(torch.zeros(3, 66, 820)).shape == (3,)
    assert m(torch.zeros(2, 66, 500)).shape == (2,)          # any length accepted
    assert n_parameters(m) < 100_000
    # first layer: 820 outputs x 32 channels x 66 inputs x 7 = 12.1 M
    assert n_mac(m, (66, 820)) > 820 * 32 * 66 * 7


def synthetic_dataset(n=240, c=4, L=128, seed=0):
    """Class 1: an echo on channel 2. Temperatures simulated by a delay."""
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    y = rng.integers(0, 2, n)
    X = np.zeros((n, c, L), dtype=np.float32)
    for i in range(n):
        r = rng.uniform(0, 5)
        for k in range(c):
            X[i, k] = np.sin(2 * np.pi * (t - r) / 16) * np.exp(-((t - 30 - r) / 10) ** 2)
        if y[i]:
            X[i, 2] += 0.3 * np.exp(-((t - 90) / 5) ** 2)
    X += rng.normal(0, 0.02, X.shape).astype(np.float32)
    idx = rng.permutation(n)
    return X, Split("synthetic", idx[:160], idx[160:200], idx[200:], y)


def test_learns_a_simple_problem():
    X, d = synthetic_dataset()
    cfg = Config(epochs=30, patience=10, widths=(8, 8), batch=16)
    _, report = train(X, d, seed=0, cfg=cfg, evaluate_test=True)
    assert report["auc_test"] > 0.95
    h = report["history"]
    assert h[report["best_epoch"]]["val_loss"] == min(e["val_loss"] for e in h)


def test_test_not_evaluated_by_default():
    X, d = synthetic_dataset()
    _, report = train(X, d, seed=0, cfg=Config(epochs=2, widths=(4,)))
    assert "auc_test" not in report and "scores_test" not in report


def test_reproducible():
    X, d = synthetic_dataset()
    cfg = Config(epochs=3, widths=(4,))
    a = train(X, d, seed=1, cfg=cfg, evaluate_test=True)[1]["scores_test"]
    b = train(X, d, seed=1, cfg=cfg, evaluate_test=True)[1]["scores_test"]
    assert np.allclose(a, b)
