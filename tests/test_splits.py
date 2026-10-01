import numpy as np
import pytest

from ogw.splits import (VAL_BAND, POSITIONS, drift_control, by_temperature,
                        nested_folds, position_folds, check)
from ogw.data import load_meta, ogw_file

pytestmark = pytest.mark.skipif(not ogw_file().exists(), reason="data missing")


@pytest.fixture(scope="module")
def meta():
    return load_meta()


def test_classes(meta):
    assert len(meta) == 966
    assert (meta.damaged == 0).sum() == 322
    for p in POSITIONS:
        assert (meta.position == p).sum() == 161
    assert len(np.unique(meta.T_setpoint)) == 81  # setpoint: 20 to 60 °C in steps of 0.5
    # each measurement campaign: 81 measurements on the heating ramp (20 to 60 °C), 80 on the cooling ramp
    for lab in np.unique(meta.label):
        for c in np.unique(meta.cycle[meta.label == lab]):
            m = (meta.label == lab) & (meta.cycle == c)
            assert (meta.ramp[m] == 0).sum() == 81 and (meta.ramp[m] == 1).sum() == 80


@pytest.mark.parametrize("i", range(4))
def test_position_folds(meta, i):
    d = position_folds(meta)[i]
    check(d, meta)
    p = POSITIONS[i]
    # the test set contains the whole position p and the whole healthy cycle 2, and nothing else
    expected = np.flatnonzero((meta.position == p) | ((meta.damaged == 0) & (meta.cycle == 2)))
    assert np.array_equal(np.sort(d.test), expected)
    # every measurement is used once and only once
    assert np.array_equal(np.sort(np.r_[d.train, d.validation, d.test]), np.arange(966))
    # validation = temperature band, training = outside the band
    Tv, Te = meta.T_measured[d.validation], meta.T_measured[d.train]
    assert np.all((Tv >= VAL_BAND[0]) & (Tv <= VAL_BAND[1]))
    assert not np.any((Te >= VAL_BAND[0]) & (Te <= VAL_BAND[1]))
    # both classes are present everywhere
    for idx in (d.train, d.validation, d.test):
        assert set(d.target[idx]) == {0, 1}


def test_temperature(meta):
    d = by_temperature(meta)
    check(d, meta)
    Tt, Te = meta.T_measured[d.test], meta.T_measured[d.train]
    assert Tt.min() > 45 and Tt.max() < 55
    assert not np.any((Te >= Tt.min()) & (Te <= Tt.max()))


def test_drift(meta):
    d = drift_control(meta)
    check(d, meta)
    all_ = np.r_[d.train, d.validation, d.test]
    assert np.all(meta.damaged[all_] == 0)
    assert set(d.target[all_]) == {0, 1}


def test_nested_folds(meta):
    folds = nested_folds(meta)
    assert len(folds) == 12
    for d in folds:
        check(d, meta)
        pt, pv = d.name.split("_")[1:]
        # nothing from the outer test set is used; the validation position is absent from training
        outer = np.flatnonzero((meta.position == pt) | ((meta.damaged == 0) & (meta.cycle == 2)))
        assert not np.intersect1d(np.r_[d.train, d.validation], outer).size
        assert set(meta.position[d.train]) == set(POSITIONS) - {pt, pv} | {""}
        assert (meta.position[d.validation] == pv).sum() == 161
        # healthy validation measurements: cooling ramp of cycle 1, same temperatures as training
        healthy_val = d.validation[meta.damaged[d.validation] == 0]
        assert len(healthy_val) == 80 and np.all(meta.ramp[healthy_val] == 1)


def test_check_detects_a_leak(meta):
    d = position_folds(meta)[0]
    d.train = np.r_[d.train, d.test[:1]]
    with pytest.raises(AssertionError):
        check(d, meta)
