"""Step 7: the complete pipeline at the 12 excitation frequencies (40 to 260 kHz).

Same protocol at every frequency, fixed before the runs:
- preprocessing: centring, decimation q = q_for(f) (about 16 points per period),
  band-pass [f/2, 2f];
- PCA detection: k chosen per fold (smallest k reaching the best validation
  AUC: cooling ramp of healthy cycle 1 + positions outside the test set),
  threshold at the 95 % quantile of the healthy validation measurements;
- Step 2 baseline (optimal baseline, maximum over the paths);
- delay and sum localization, v and t0 calibrated on the healthy direct
  arrivals (end of crosstalk: 5 / f + 25 µs);
- multi-frequency fusion of the localization: sum, over the 12 frequencies, of
  the maps of each measurement divided by their maximum.
Run (from the repository root): .venv/bin/python step07_frequencies/frequencies.py  (option --from-json: figure and summary only, from measurements.json)
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from sklearn.metrics import roc_auc_score

from ogw.anomaly import PCADetector
from ogw.splits import POSITIONS, by_temperature, position_folds
from ogw.data import load_meta, load_signals, q_for
from ogw.localization import DEFECTS, DelayAndSum, calibrate_velocity, error
from ogw.baseline import BaselineIndicator

OUT = Path(__file__).resolve().parents[1] / "results" / "step07_frequencies"
FREQS = list(range(40, 261, 20))
K_CANDIDATES = (1, 2, 3, 5, 10, 20, 40)
FUSION_STEP = 5.0


def evaluate(f):
    meta, X = load_meta(f), load_signals(f)
    fs = meta.fs / q_for(f)
    healthy1 = (meta.damaged == 0) & (meta.cycle == 1)
    train, val = np.flatnonzero(healthy1 & (meta.ramp == 0)), np.flatnonzero(healthy1 & (meta.ramp == 1))
    b = dict(f_kHz=f, q=q_for(f), L=int(X.shape[-1]), pca={}, baseline={}, localization={})

    # 1. PCA detection, splits by position
    dets = {k: PCADetector(k).fit(X, train, val) for k in K_CANDIDATES}
    s_val_healthy = {k: d.scores(val) for k, d in dets.items()}
    for d in position_folds(meta):
        p = d.name.removeprefix("position_")
        vd = np.flatnonzero((meta.damaged == 1) & (meta.position != p))
        aucs = {k: roc_auc_score(np.r_[np.zeros(len(val)), np.ones(len(vd))],
                                 np.r_[s_val_healthy[k], dets[k].scores(vd)]) for k in K_CANDIDATES}
        k = min(k for k in K_CANDIDATES if aucs[k] == max(aucs.values()))
        s, y = dets[k].scores(d.test), meta.damaged[d.test]
        b["pca"][d.name] = dict(k=k, auc_val=aucs[k], auc=roc_auc_score(y, s),
                                fpr=float((s[y == 0] > dets[k].threshold).mean()),
                                tpr=float((s[y == 1] > dets[k].threshold).mean()))
    # temperature split: most frequent k above
    ks = [v["k"] for v in b["pca"].values()]
    k = max(set(ks), key=ks.count)
    dt = by_temperature(meta)
    se, sv = (dt.train[meta.damaged[dt.train] == 0],
              dt.validation[meta.damaged[dt.validation] == 0])
    det_t = PCADetector(k).fit(X, se, sv)
    s, y = det_t.scores(dt.test), meta.damaged[dt.test]
    b["pca"]["temperature"] = dict(k=k, auc=roc_auc_score(y, s), fpr=float((s[y == 0] > det_t.threshold).mean()),
                                   tpr=float((s[y == 1] > det_t.threshold).mean()))

    # 2. Step 2 baseline
    for d in [*position_folds(meta), dt]:
        lib = np.intersect1d(np.flatnonzero(meta.damaged == 0), np.r_[d.train, d.validation])
        ind = BaselineIndicator("optimal", "max").fit(X, meta, lib)
        s, y = ind.scores(d.test), d.target[d.test]
        b["baseline"][d.name] = dict(auc=roc_auc_score(y, s), fpr=float((s[y == 0] > ind.threshold).mean()))

    # 3. Delay and sum localization (validation PCA detector, most frequently selected k)
    det = dets[max(set(ks), key=ks.count)]
    v, t0, rms = calibrate_velocity(X[train].astype(np.float64), meta.paths, fs,
                                    t_min_us=5 / (f * 1e3) * 1e6 + 25)
    level = np.sqrt((det.residual_signals(val) ** 2).mean(axis=(0, 2)))
    das = DelayAndSum(meta.paths, v, t0, fs, X.shape[-1])
    das_f = DelayAndSum(meta.paths, v, t0, fs, X.shape[-1], step=FUSION_STEP)
    b["localization"] = dict(v=v, t0=t0, rms=rms, errors={})
    maps = {}
    for p in POSITIONS:
        idx = np.flatnonzero(meta.position == p)
        env = np.abs(hilbert(det.residual_signals(idx), axis=-1)) / level[None, :, None]
        b["localization"]["errors"][p] = error(das.localize(env), DEFECTS[p]).tolist()
        c = np.stack([das_f.map(e) for e in env]).astype(np.float32)
        maps[p] = c / c.reshape(len(c), -1).max(1)[:, None, None]
    print(f"{f} kHz done", flush=True)
    return b, maps, (das_f.x, das_f.y)


def plot(reports, fusion):
    med = lambda e: float(np.median(e))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, p in enumerate(POSITIONS):
        ax[0].plot(FREQS, [b["pca"][f"position_{p}"]["auc"] for b in reports], "o-", color=f"C{k}", label=f"PCA {p}")
        ax[1].plot(FREQS, [med(b["localization"]["errors"][p]) for b in reports], "o-", color=f"C{k}", label=p)
        ax[1].axhline(med(fusion[p]), color=f"C{k}", ls="--", lw=0.8)
    ax[0].plot(FREQS, [b["pca"]["temperature"]["auc"] for b in reports], "ks--", label="PCA temperature")
    ax[0].set_ylim(0.2, 1.02); ax[0].set_xlabel("frequency (kHz)"); ax[0].set_ylabel("test AUC")
    ax[0].set_title("PCA detection versus frequency"); ax[0].legend(fontsize=7)
    ax[1].set_yscale("log"); ax[1].set_xlabel("frequency (kHz)"); ax[1].set_ylabel("median error (mm)")
    ax[1].set_title("Localization: per frequency (dots), fusion of the 12 (dashes)"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "01_frequencies.png", dpi=110); plt.close(fig)


def write_summary(reports, fusion):
    med = lambda e: float(np.median(e))
    L = ["# Step 7: the complete pipeline at the 12 frequencies", "",
         "Generated by `step07_frequencies/frequencies.py` (same protocol at every frequency, fixed in advance).", "",
         "## Detection (test AUC; PCA: false alarms at the threshold in parentheses)", "",
         "| kHz | k | PCA D04 | PCA D12 | PCA D16 | PCA D24 | PCA temperature | baseline (4 positions, min) | baseline temperature |",
         "|---" * 9 + "|"]
    for b in reports:
        a, r = b["pca"], b["baseline"]
        L.append(f"| {b['f_kHz']} | {a['position_D04']['k']} | "
                 + " | ".join(f"{a[f'position_{p}']['auc']:.3f} ({a[f'position_{p}']['fpr']:.0%})" for p in POSITIONS)
                 + f" | {a['temperature']['auc']:.3f} ({a['temperature']['fpr']:.0%}) | "
                 f"{min(r[f'position_{p}']['auc'] for p in POSITIONS):.3f} | {r['temperature']['auc']:.3f} |")
    L += ["", "## Delay and sum localization (median error, mm)", "",
          "| kHz | v (mm/µs) | t0 (µs) | " + " | ".join(POSITIONS) + " | all |", "|---" * 8 + "|"]
    for b in reports:
        e = b["localization"]["errors"]
        L.append(f"| {b['f_kHz']} | {b['localization']['v']:.3f} | {b['localization']['t0']:.0f} | "
                 + " | ".join(f"{med(e[p]):.0f}" for p in POSITIONS)
                 + f" | {med(np.concatenate([e[p] for p in POSITIONS])):.0f} |")
    L.append(f"| **fusion of the 12** | | | " + " | ".join(f"**{med(fusion[p]):.0f}**" for p in POSITIONS)
             + f" | **{med(np.concatenate([fusion[p] for p in POSITIONS])):.0f}** |")

    L += ["", "## Leakage-free frequency selection (PCA, validation AUC of each fold only)", "",
          "| fold | best validation AUC | frequencies reaching it (kHz) | test AUC at 40 kHz |", "|---" * 4 + "|"]
    for p in POSITIONS:
        n = f"position_{p}"
        best = max(b["pca"][n]["auc_val"] for b in reports)
        tied = [b["f_kHz"] for b in reports if b["pca"][n]["auc_val"] == best]
        at40 = next(b for b in reports if b["f_kHz"] == 40)["pca"][n]["auc"]
        L.append(f"| {p} | {best:.3f} | {', '.join(map(str, tied))} | {at40:.3f} |")
    bad = [b for b in reports if b["localization"]["v"] <= 0 or b["localization"]["t0"] < 0]
    if bad:
        L += ["", "Physically meaningless velocity calibrations (negative velocity or negative t0): "
              + "; ".join(f"{b['f_kHz']} kHz (v = {b['localization']['v']:.3f} mm/µs, t0 = {b['localization']['t0']:.0f} µs)"
                          for b in bad)
              + ". The localization errors at these frequencies rest on a failed calibration."]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["--from-json"]:     # figure and summary only, from measurements.json
        m = json.loads((OUT / "measurements.json").read_text())
        plot(m["frequencies"], m["fusion"])
        write_summary(m["frequencies"], m["fusion"])
        sys.exit()
    with Pool(len(FREQS)) as pool:
        outputs = pool.map(evaluate, FREQS)
    reports = [s[0] for s in outputs]
    gx, gy = outputs[0][2]

    # Multi-frequency fusion of the localization
    fusion = {}
    for p in POSITIONS:
        total = sum(s[1][p] for s in outputs)
        k = total.reshape(len(total), -1).argmax(1)
        fusion[p] = error(np.stack([gx.ravel()[k], gy.ravel()[k]], axis=1), DEFECTS[p]).tolist()
    (OUT / "measurements.json").write_text(json.dumps(dict(frequencies=reports, fusion=fusion), ensure_ascii=False))

    write_summary(reports, fusion)
    plot(reports, fusion)
