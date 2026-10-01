import itertools

import numpy as np
from scipy.signal import hilbert

from ogw.localization import (DEFECTS, TRANSDUCERS, RAPID, DelayAndSum, calibrate_velocity, error,
                              damage_index)

PATHS = np.array(list(itertools.combinations(range(12), 2)))    # 66 paths, same order as OGW


def distance_to_path(p, a, b):
    a, b, p = map(np.asarray, (a, b, p))
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return np.hypot(*(p - (a + t * (b - a))))


def synthetic_di(p, width=20.0):
    """Strong indices on the paths passing close to p (model of a scattering defect)."""
    d = np.array([distance_to_path(p, TRANSDUCERS[i], TRANSDUCERS[j]) for i, j in PATHS])
    return np.exp(-(d / width) ** 2)


def test_geometry():
    assert len(PATHS) == 66 and TRANSDUCERS.shape == (12, 2)
    assert tuple(PATHS[0]) == (0, 1) and tuple(PATHS[-1]) == (10, 11)


def test_rapid_interior_positions():
    rapid = RAPID(PATHS)
    for p in [DEFECTS["D12"], DEFECTS["D16"], (250, 250), (100, 150)]:
        assert error(rapid.localize(synthetic_di(p)), p)[0] < 25


def test_rapid_blind_along_an_edge_path():
    # known limitation: D24 lies on the vertical path T1-T7, which almost nothing crosses
    assert error(RAPID(PATHS).localize(synthetic_di(DEFECTS["D24"])), DEFECTS["D24"])[0] > 100


def scattered_signals(p, v=1.3, t0=90.0, fs=625e3, L=820):
    """Synthetic residuals: a wave packet at the time of flight a -> p -> b on each path."""
    t = np.arange(L) / fs * 1e6
    s = np.zeros((len(PATHS), L))
    for n, (i, j) in enumerate(PATHS):
        tv = (np.hypot(*(TRANSDUCERS[i] - p)) + np.hypot(*(TRANSDUCERS[j] - p))) / v + t0
        s[n] = np.exp(-((t - tv) / 30) ** 2) * np.sin(2 * np.pi * 0.04 * (t - tv))
    return s


def test_delay_and_sum_localizes_even_at_the_edge():
    das = DelayAndSum(PATHS, v=1.3, t0=90.0, fs=625e3, L=820)
    for p in [*DEFECTS.values(), (250, 250)]:
        env = np.abs(hilbert(scattered_signals(np.array(p)), axis=-1))
        assert error(das.localize(env), p)[0] < 15


def test_calibrate_velocity():
    fs, L, v, t0 = 625e3, 820, 1.3, 90.0
    t = np.arange(L) / fs * 1e6
    d = np.hypot(*(TRANSDUCERS[PATHS[:, 0]] - TRANSDUCERS[PATHS[:, 1]]).T)
    X = np.stack([np.exp(-((t - (dk / v + t0)) / 25) ** 2) * np.sin(2 * np.pi * 0.04 * t) for dk in d])[None]
    v_est, t0_est, _ = calibrate_velocity(X, PATHS, fs)
    assert abs(v_est - v) < 0.05 and abs(t0_est - t0) < 10


def test_healthy_index_zero():
    r = np.array([0.5, 1.0, np.nan, 3.0])
    assert np.allclose(damage_index(r), [0, 0, 0, 2])
