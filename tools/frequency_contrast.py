"""Damage / healthy variability contrast per frequency.

The raw signals carry a DC offset that would bias the comparison, so each
signal is centred, then band-pass filtered in [f/2, 2f], before comparison.

For each frequency f and each temperature T (heating ramp of the cycle, 22 to
60 °C in steps of 5), baseline = healthy cycle 1 measurement whose measured
temperature (T_meas_1) is closest. The setpoint T_ctc is not suitable: its
offset from the measured temperature varies with the measurement campaign
(T_meas_1 minus T_ctc in metadata/ogw_index.csv: +1.15 °C on average for
healthy cycle 1, +0.57 °C for cycle 2, +0.26 to +0.34 °C for the damaged
campaigns; Step 1 reports +1.11 and +0.55 °C with the mean of the two plate
sensors read from the signal files). Relative residual per path
R(a) = ||a - ref||² / ||ref||². Compared quantities:
  - R(healthy cycle 2): variability between two days, without damage;
  - R(damage Dxx): damage effect plus the same variability;
  - R(healthy c1 at T + 0.5 °C): floor (neighbouring temperature, same cycle).
Median over the temperatures; then median and maximum over the 66 paths.

Usage (prints a Markdown table, saved as results/step01_exploration/frequency_contrast.md):
    python tools/frequency_contrast.py > results/step01_exploration/frequency_contrast.md
"""
import csv
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FREQS = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
TEMPS = [22.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
DAMAGES = ["dam_D04", "dam_D12", "dam_D16", "dam_D24"]

rows = list(csv.DictReader(open(ROOT / "metadata" / "ogw_index.csv")))
by = defaultdict(list)
for r in rows:
    by[r["label"]].append(r)
for v in by.values():
    v.sort(key=lambda r: float(r["timestamp"]))
groups = {"c1": by["udam"][:161], "c2": by["udam"][161:], **{d: by[d] for d in DAMAGES}}


def closest(group, T, heating=True):
    candidates = group[:81] if heating else group
    return min(candidates, key=lambda r: abs(float(r["T_meas_1"]) - T))


def compute(f):
    zips = {p.stem.replace("OGW_CFRP_Temperature_", ""): zipfile.ZipFile(p) for p in RAW.glob("*.zip")}
    sos = butter(4, [f * 500, f * 2000], btype="band", fs=10e6, output="sos")

    def signal(r):
        lab = r["label"]
        with zips[lab].open(f"OGW_CFRP_Temperature_{lab}/{r['folder']}/pc_f{f}kHz.h5") as fh, \
                h5py.File(fh, "r") as h:
            x = h["pitchcatch/catch"][()].astype(np.float64)
        x -= x.mean(-1, keepdims=True)
        return sosfiltfilt(sos, x, axis=-1)

    R = defaultdict(list)
    for T in TEMPS:
        for g in ["c2", *DAMAGES]:
            r = closest(groups[g], T)
            ref = signal(closest(groups["c1"], float(r["T_meas_1"]), heating=False))
            R[g].append(((signal(r) - ref) ** 2).sum(-1) / (ref ** 2).sum(-1))
        r = closest(groups["c1"], T)
        ref, neighbour = signal(r), signal(closest(groups["c1"], float(r["T_meas_1"]) + 0.5))
        R["floor"].append(((neighbour - ref) ** 2).sum(-1) / (ref ** 2).sum(-1))
    return f, {k: np.median(np.stack(v), 0) for k, v in R.items()}   # (66,) per group


if __name__ == "__main__":
    with Pool(len(FREQS)) as pool:
        res = dict(pool.map(compute, FREQS))
    print("| kHz | floor | c2 against c1 | damages (median of 4) | damage / c2: median over paths | damage / c2: max over paths |")
    print("|---|---|---|---|---|---|")
    for f in FREQS:
        r = res[f]
        d = np.median(np.stack([r[k] for k in DAMAGES]), 0)
        ratio = np.stack([r[k] / r["c2"] for k in DAMAGES])
        print(f"| {f} | {np.median(r['floor']):.4f} | {np.median(r['c2']):.4f} | "
              f"{np.median(d):.4f} | {np.median(ratio):.2f} | {np.median(ratio.max(1)):.2f} |")
