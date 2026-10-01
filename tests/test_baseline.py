"""Tests of the baseline indicator on a small synthetic dataset.

"Temperature" simulated by a delay of the wave packet; "damage" by a weak echo
on a few paths. Two healthy cycles (1 and 2), each with a heating then a
cooling ramp, as in OGW.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from ogw.baseline import BaselineIndicator

L, NT = 400, 6
rng = np.random.default_rng(0)


def packet(delay, amp=1.0):
    t = np.arange(L)
    u = (t - 50 - delay) / 40
    return amp * np.where((u >= 0) & (u <= 1), np.sin(np.pi * u) ** 2, 0) * np.sin(2 * np.pi * (t - delay) / 10)


def dataset():
    X, lab, cyc, rmp, T = [], [], [], [], []
    for label, cycle in (("udam", 1), ("udam", 2), ("dam_D04", 1)):
        temps = np.r_[np.linspace(20, 60, 21), np.linspace(59, 20, 20)]
        for k, temp in enumerate(temps):
            x = np.stack([packet(temp / 4 + p % 3) for p in range(NT)]) + rng.normal(0, 0.01, (NT, L))
            if label != "udam":
                x[:2] += np.stack([packet(temp / 4 + 150, 0.2)] * 2)   # damage on paths 0 and 1
            X.append(x); lab.append(label); cyc.append(cycle); rmp.append(int(k > 20)); T.append(temp)
    lab = np.array(lab)
    meta = SimpleNamespace(label=lab, cycle=np.array(cyc), ramp=np.array(rmp),
                           damaged=(lab != "udam").astype(int), T_setpoint=np.array(T),
                           T_measured=np.array(T) + 0.3)
    return np.array(X, dtype=np.float32), meta


X, meta = dataset()
lib = np.flatnonzero((meta.label == "udam") & (meta.cycle == 1))
test = np.flatnonzero((meta.label != "udam") | (meta.cycle == 2))


@pytest.mark.parametrize("choice", ["setpoint", "measured", "optimal"])
def test_separates_healthy_and_damaged(choice):
    ind = BaselineIndicator(choice=choice).fit(X, meta, lib)
    s = ind.scores(test)
    y = meta.damaged[test]
    assert s[y == 1].min() > s[y == 0].max()


def test_never_its_own_baseline():
    ind = BaselineIndicator().fit(X, meta, lib)
    # a measurement compared with itself would give a zero residual
    assert np.all(ind.calibration_scores > 0)
    assert np.all(ind.residuals(lib[:5]) > 0)


def test_faulty_path_ignored():
    Xd = X.copy()
    healthy2 = test[meta.damaged[test] == 0]
    Xd[healthy2[3], 4] = 0                           # path 4 empty on a healthy measurement
    Xd[lib[7], 5] = 0                                # path 5 empty on a baseline
    ind = BaselineIndicator().fit(Xd, meta, lib)
    assert ind.faulty[healthy2[3], 4] and ind.faulty[lib[7], 5]
    assert ind.faulty.sum() == 2
    s = ind.scores(test)
    y = meta.damaged[test]
    assert np.isfinite(s).all()
    assert s[y == 1].min() > s[y == 0].max()         # no false alarm


def test_healthy_library_only():
    with pytest.raises(AssertionError):
        BaselineIndicator().fit(X, meta, np.r_[lib, test[-1]])


def test_damage_located_on_the_right_paths():
    ind = BaselineIndicator().fit(X, meta, lib)
    r = ind.residuals(np.flatnonzero(meta.damaged == 1))
    assert set(np.argsort(np.median(r, 0))[-2:]) == {0, 1}
