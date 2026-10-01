"""Step 3: anomaly detection by PCA (learned on healthy measurements only).

Protocol, fixed before opening the test set:
- position splits: PCA fitted on the heating ramp of healthy cycle 1, validation
  = cooling ramp of healthy cycle 1 (healthy) + the damage positions outside the fold's test set;
- choice of k: the smallest k reaching the best validation AUC, per fold;
- temperature split: PCA on the healthy training measurements, validation on
  the healthy measurements of the validation band; k taken from the previous choice
  (no damage outside the test set available to validate it otherwise);
- threshold: 95 % quantile of the healthy validation indicators.

Run (from the repository root, with PYTHONPATH=src):
  python step03_cnn_vs_pca/evaluate_pca.py --phase validation
  then                                     --phase test
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from ogw.anomaly import PCADetector
from ogw.splits import POSITIONS, by_temperature, position_folds, check
from ogw.data import load_meta, load_signals

OUT = Path(__file__).resolve().parents[1] / "results" / "step03_cnn_vs_pca"
K_CANDIDATES = (1, 2, 3, 5, 10, 20, 40)


def auc(y, s):
    return float(roc_auc_score(y, s))


def main(phase):
    OUT.mkdir(parents=True, exist_ok=True)
    meta, X = load_meta(), load_signals()
    healthy1 = (meta.damaged == 0) & (meta.cycle == 1)
    train, healthy_val = np.flatnonzero(healthy1 & (meta.ramp == 0)), np.flatnonzero(healthy1 & (meta.ramp == 1))
    reports, k_choice = [], {}
    for d in position_folds(meta):
        check(d, meta)
        p = d.name.removeprefix("position_")
        val_dmg = np.flatnonzero((meta.damaged == 1) & (meta.position != p))
        v_idx = np.r_[healthy_val, val_dmg]
        v_y = meta.damaged[v_idx]
        assert not np.intersect1d(np.r_[train, v_idx], d.test).size, "leakage into the test set"
        aucs = {}
        for k in K_CANDIDATES:
            det = PCADetector(k).fit(X, train, healthy_val)
            aucs[k] = auc(v_y, det.scores(v_idx))
        k = min(k for k in K_CANDIDATES if aucs[k] == max(aucs.values()))
        k_choice[d.name] = k
        print(f"{d.name:16s} val AUC per k: " + ", ".join(f"{k}:{a:.3f}" for k, a in aucs.items())
              + f"  -> k = {k}")
        b = dict(split=d.name, k=k, auc_val=aucs[k], aucs_val=aucs)
        if phase == "test":
            det = PCADetector(k).fit(X, train, healthy_val)
            s, y = det.scores(d.test), meta.damaged[d.test]
            fpr, tpr, _ = roc_curve(y, s)
            b.update(auc_test=auc(y, s), tpr_fpr5=float(np.interp(0.05, fpr, tpr)),
                     fpr_threshold=float((s[y == 0] > det.threshold).mean()),
                     tpr_threshold=float((s[y == 1] > det.threshold).mean()),
                     scores_test=s.tolist(), cost=det.cost(), n_train=det.n_train)
        reports.append(b)

    d = by_temperature(meta)
    check(d, meta)
    k = max(set(k_choice.values()), key=list(k_choice.values()).count)
    ht = d.train[meta.damaged[d.train] == 0]
    hv = d.validation[meta.damaged[d.validation] == 0]
    det = PCADetector(k).fit(X, ht, hv)
    b = dict(split=d.name, k=k, auc_val=auc(meta.damaged[d.validation], det.scores(d.validation)))
    print(f"{d.name:16s} k = {k} (carried over), val AUC {b['auc_val']:.3f}")
    if phase == "test":
        s, y = det.scores(d.test), meta.damaged[d.test]
        fpr, tpr, _ = roc_curve(y, s)
        b.update(auc_test=auc(y, s), tpr_fpr5=float(np.interp(0.05, fpr, tpr)),
                 fpr_threshold=float((s[y == 0] > det.threshold).mean()),
                 tpr_threshold=float((s[y == 1] > det.threshold).mean()),
                 scores_test=s.tolist(), cost=det.cost(), n_train=det.n_train)
    reports.append(b)

    (OUT / f"pca_{phase}.json").write_text(json.dumps(reports, ensure_ascii=False))
    if phase == "test":
        print("\nsplit              k   test AUC   TPR at FPR 5 %   threshold: FPR   threshold: TPR")
        for b in reports:
            print(f"{b['split']:16s} {b['k']:3d}   {b['auc_test']:.3f}      {b['tpr_fpr5']:.3f}"
                  f"          {b['fpr_threshold']:.3f}            {b['tpr_threshold']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["validation", "test"], required=True)
    main(ap.parse_args().phase)
