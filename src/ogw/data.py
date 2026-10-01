"""Loading and preprocessing of the OGW signals (dataset no. 2, one excitation frequency).

Selected pipeline (justified in the README, section "Data and preprocessing"):
  1. centring: removal of the mean of each signal (DC offset: 7 % of the RMS on average, Step 1);
  2. decimation with an anti-aliasing filter (factor q, 10 MHz -> 10/q MHz);
  3. band-pass filter [f/2, 2f] around the excitation frequency f
     (removes the low-frequency drift, which dominated the residual between measurements);
  4. per-path scaling, statistics estimated on the training set only.

Temperature: use T_measured (sensors on the plate), not the climate chamber
setpoint, whose offset to the actual temperature varies from one measurement
campaign to another.
"""
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.signal import butter, decimate, sosfiltfilt
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
F_KHZ = 40        # selected frequency (see the README, section "Frequency choice")
Q = 16            # 10 MHz -> 625 kHz, i.e. about 16 points per period at 40 kHz (15.6)


def q_for(f_kHz):
    """Decimation factor keeping about 16 samples per period (16 at 40 kHz)."""
    return max(1, round(10e6 / (16 * f_kHz * 1e3)))


def ogw_file(f_kHz=F_KHZ):
    return ROOT / "data" / f"ogw_f{f_kHz}kHz.h5"


@dataclass
class Meta:
    """Metadata of the N measurements, in file order (class, then date)."""

    label: np.ndarray       # udam, dam_D04, dam_D12, dam_D16, dam_D24
    damaged: np.ndarray     # 0 healthy, 1 damaged
    position: np.ndarray    # "" (healthy), "D04", "D12", "D16", "D24"
    cycle: np.ndarray       # 1 or 2 for healthy, 1 for damaged
    T_setpoint: np.ndarray  # chamber setpoint rounded to 0.5 °C (grid 20 to 60)
    T_measured: np.ndarray  # mean of the two plate sensors (°C)
    date: np.ndarray        # days (MATLAB datenum)
    ramp: np.ndarray        # 0 heating ramp (20 -> 60 °C), 1 cooling ramp, within each cycle
    fs: float               # original sampling frequency (Hz)
    f_kHz: int              # excitation frequency
    paths: np.ndarray       # (66, 2): transmitter, receiver

    def __len__(self):
        return len(self.label)


def _ramps(label, cycle, date, T):
    """0 up to the temperature maximum of each cycle (heating ramp), 1 afterwards."""
    ramp = np.zeros(len(label), dtype=np.int64)
    for lab, c in set(zip(label, cycle)):
        idx = np.flatnonzero((label == lab) & (cycle == c))
        idx = idx[np.argsort(date[idx])]
        ramp[idx[np.argmax(T[idx]) + 1:]] = 1
    return ramp


def load_meta(f_kHz=F_KHZ):
    with h5py.File(ogw_file(f_kHz), "r") as f:
        label = f["label"].asstr()[()]
        cycle, date, T_ctc = f["cycle"][()].astype(np.int64), f["timestamp"][()], f["T_ctc"][()]
        return Meta(
            label=label,
            damaged=f["damaged"][()].astype(np.int64),
            position=np.array([l.removeprefix("dam_") if l != "udam" else "" for l in label]),
            cycle=cycle,
            T_setpoint=np.round(T_ctc * 2) / 2,
            T_measured=(f["T_meas_1"][()] + f["T_meas_2"][()]) / 2,
            date=date,
            ramp=_ramps(label, cycle, date, T_ctc),
            fs=float(f.attrs["fs"]),
            f_kHz=int(f.attrs["f_kHz"]),
            paths=f.attrs["channels"][()].astype(np.int64),
        )


def preprocess(x, f_kHz=F_KHZ, q=Q, fs=10e6):
    """Centres, decimates then band-pass filters along the last axis (float32)."""
    x = x.astype(np.float64)
    x -= x.mean(axis=-1, keepdims=True)
    if q > 1:
        x = decimate(x, q, ftype="fir", zero_phase=True, axis=-1)
    sos = butter(4, [f_kHz * 500, f_kHz * 2000], btype="band", fs=fs / q, output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def load_signals(f_kHz=F_KHZ, q=None, block_size=50):
    """Preprocessed signals (N, 66, L), cached in data/ on the first call."""
    q = q_for(f_kHz) if q is None else q
    file = ogw_file(f_kHz)
    cache = file.with_name(f"{file.stem}_q{q}_pb.npy")
    if cache.exists() and cache.stat().st_mtime > file.stat().st_mtime:
        return np.load(cache)
    with h5py.File(file, "r") as f:
        raw, fs = f["catch"], float(f.attrs["fs"])
        X = np.concatenate([preprocess(raw[i:i + block_size], f_kHz, q, fs)
                            for i in range(0, raw.shape[0], block_size)])
    np.save(cache, X)
    return X


def faulty_paths(X, typical_energy, threshold=0.3):
    """(N, 66) boolean: nearly empty path, energy < threshold × typical energy of the path.

    Acquisition quality check, independent of the labels. typical_energy
    (66,): median of the per-path energy over the training measurements. In
    OGW at 40 kHz, 11 (measurement, path) pairs are empty, all on path 8-9.
    """
    return (X.astype(np.float64) ** 2).sum(-1) < threshold * typical_energy


class Normalizer:
    """Divides each path by its standard deviation, estimated on the training measurements.

    One scale per path (not per signal): the relative amplitudes between
    measurements, which carry the damage information, are preserved.
    """

    def fit(self, X):
        self.scale = X.std(axis=(0, 2))[:, None].astype(np.float32)  # (66, 1)
        return self

    def apply(self, X):
        return X / self.scale


class OGWSignals(Dataset):
    """Sample = one measurement (66 paths as channels) or one (measurement, path) pair.

    X: normalized signals (N, 66, L); target: label per measurement (N,);
    indices: selected measurements (from a split of ogw.splits).
    """

    def __init__(self, X, target, indices, per_path=False, paths=None):
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.paths = list(range(X.shape[1])) if paths is None else list(paths)
        self.per_path = per_path
        if per_path:
            self.index = [(m, t) for m in indices for t in self.paths]
        else:
            self.index = [(m, None) for m in indices]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        m, t = self.index[i]
        x = self.X[m, t:t + 1] if self.per_path else self.X[m, self.paths]
        return x, self.target[m]
