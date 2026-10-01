"""Train / validation / test splits without data leakage.

The split unit is always the measurement: the 66 paths of a measurement go
together into the same set (they share temperature, date and state).

- by_position (main, 4 folds): test = a damage position never seen
  + healthy cycle 2; the rest (3 positions + healthy cycle 1) is divided between
  training and validation according to a temperature band.
- by_temperature (secondary): test = a temperature band never seen,
  all classes together.
The bands refer to the measured temperature of the plate (T_measured).
- drift_control: healthy cycle 1 against healthy cycle 2. If a model can tell
  them apart, it can also learn the date rather than the damage (all healthy
  measurements precede all damaged measurements).
"""
from dataclasses import dataclass

import numpy as np

POSITIONS = ("D04", "D12", "D16", "D24")
VAL_BAND = (38.0, 42.0)
TEST_BAND = (48.0, 52.0)


@dataclass
class Split:
    name: str
    train: np.ndarray          # measurement indices
    validation: np.ndarray
    test: np.ndarray
    target: np.ndarray         # label per measurement (N,); -1 = measurement not used


def _band(meta, band):
    return (meta.T_measured >= band[0]) & (meta.T_measured <= band[1])


def _idx(mask):
    return np.flatnonzero(mask)


def by_position(meta, test_position, val_band=VAL_BAND):
    healthy = meta.damaged == 0
    test = (meta.position == test_position) | (healthy & (meta.cycle == 2))
    pool = ((meta.damaged == 1) & ~test) | (healthy & (meta.cycle == 1))
    val = pool & _band(meta, val_band)
    return Split(f"position_{test_position}", _idx(pool & ~val), _idx(val),
                 _idx(test), meta.damaged.copy())


def position_folds(meta, val_band=VAL_BAND):
    return [by_position(meta, p, val_band) for p in POSITIONS]


def by_position_nested(meta, test_position, val_position):
    """Nested validation: asks the test question (position never seen) without touching the test.

    The outer test (test_position + healthy cycle 2) is set aside. Validation =
    the whole val_position + cooling ramp of healthy cycle 1; training = the two
    other positions + heating ramp of healthy cycle 1. Healthy and damaged
    validation measurements thus cover the same temperatures (a temperature band
    reserved for the healthy validation measurements alone would confound
    temperature and class).
    """
    outer = by_position(meta, test_position)
    healthy1 = (meta.damaged == 0) & (meta.cycle == 1)
    val = (meta.position == val_position) | (healthy1 & (meta.ramp == 1))
    pool = ((meta.damaged == 1) & (meta.position != test_position)) | healthy1
    return Split(f"nested_{test_position}_{val_position}", _idx(pool & ~val),
                 _idx(val), outer.test, meta.damaged.copy())


def nested_folds(meta):
    return [by_position_nested(meta, pt, pv)
            for pt in POSITIONS for pv in POSITIONS if pv != pt]


def by_temperature(meta, test_band=TEST_BAND, val_band=VAL_BAND):
    test = _band(meta, test_band)
    val = _band(meta, val_band)
    return Split("temperature", _idx(~test & ~val), _idx(val), _idx(test),
                 meta.damaged.copy())


def drift_control(meta, test_band=TEST_BAND, val_band=VAL_BAND):
    healthy = meta.damaged == 0
    target = np.where(healthy, meta.cycle - 1, -1)          # 0 = cycle 1, 1 = cycle 2
    test = healthy & _band(meta, test_band)
    val = healthy & _band(meta, val_band)
    return Split("drift", _idx(healthy & ~test & ~val), _idx(val), _idx(test), target)


def check(d, meta):
    """Raises an AssertionError if the split shows an obvious leak."""
    sets = {"train": d.train, "validation": d.validation, "test": d.test}
    for a in sets:
        for b in sets:
            if a < b:
                assert not np.intersect1d(sets[a], sets[b]).size, f"{a} and {b} overlap"
    for name, idx in sets.items():
        assert idx.size, f"{name} empty"
        assert np.all(d.target[idx] >= 0), f"{name} contains unlabelled measurements"
    if d.name.startswith("position_"):
        p = d.name.removeprefix("position_")
        assert not np.any(meta.position[np.r_[d.train, d.validation]] == p)
        assert not np.any(np.isin(np.r_[d.train, d.validation],
                                  _idx((meta.damaged == 0) & (meta.cycle == 2))))
    if d.name.startswith("nested_"):
        pt, pv = d.name.split("_")[1:]
        assert not np.any(np.isin(meta.position[d.train], [pt, pv]))
        assert not np.any(meta.position[d.validation] == pt)
    if d.name in ("temperature", "drift"):
        Tt, Te = meta.T_measured[d.test], meta.T_measured[np.r_[d.train, d.validation]]
        assert not np.any((Te >= Tt.min()) & (Te <= Tt.max())), \
            "a temperature of the test range appears in training"


def summary(d, meta):
    """Markdown table: number of measurements per class and per set."""
    classes = ["udam c1", "udam c2", *POSITIONS]
    def key(i):
        return f"udam c{meta.cycle[i]}" if meta.damaged[i] == 0 else meta.position[i]
    lines = [f"**{d.name}**", "", "| set | " + " | ".join(classes) + " | total | share target=1 |",
             "|---" * (len(classes) + 3) + "|"]
    for name, idx in (("training", d.train), ("validation", d.validation), ("test", d.test)):
        n = [sum(key(i) == c for i in idx) for c in classes]
        lines.append(f"| {name} | " + " | ".join(map(str, n))
                     + f" | {len(idx)} | {d.target[idx].mean():.2f} |")
    return "\n".join(lines)
