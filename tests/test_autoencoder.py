import numpy as np
import torch

from ogw.autoencoder import AEConv, AEDense, AEDetector
from tests.test_anomaly import dataset


def test_shapes():
    for A in (AEConv, AEDense):
        m = A(5, channels=66, L=820)
        assert m(torch.zeros(2, 66, 820)).shape == (2, 66, 820)
    assert sum(p.numel() for p in AEConv(5).parameters()) < 80_000


def test_detects_without_seeing_damage():
    X, y = dataset(L=200)
    torch.set_num_threads(2)
    d = AEDetector(AEDense, latent=2, epochs=600, patience=200).fit(X, np.arange(60), np.arange(60, 90))
    assert d.scores(np.arange(90, 120)).min() > d.threshold
