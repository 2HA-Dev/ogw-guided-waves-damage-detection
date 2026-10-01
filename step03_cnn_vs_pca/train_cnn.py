"""Step 3: 1D CNN, 6 splits x 5 seeds, in two phases.

  --phase validation : trains and reports only the validation (tuning),
                       including 12 nested folds (a validation position
                       never seen, the external test set left out);
  --phase test       : frozen configuration, evaluates the test set once,
                       saves the models (results/step03_cnn_vs_pca/models/, not versioned).

Run (from the repository root, with PYTHONPATH=src):
  python step03_cnn_vs_pca/train_cnn.py --phase validation
  python step03_cnn_vs_pca/train_cnn.py --phase test
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from ogw.splits import (drift_control, by_temperature, nested_folds, position_folds,
                        check)
from ogw.data import load_meta, load_signals
from ogw.training import Config, train

OUT = Path(__file__).resolve().parents[1] / "results" / "step03_cnn_vs_pca"
SEEDS = range(5)
CONFIGS = {"default": Config(),
           "regularized": Config(widths=(16, 16, 32, 32), dropout=0.5, weight_decay=1e-3)}
CFG = None              # chosen by --config; frozen before the test phase (see the README, section "Protocol")


def splits(meta, nested=False):
    items = [*position_folds(meta), by_temperature(meta), drift_control(meta)]
    return {d.name: d for d in items + (nested_folds(meta) if nested else [])}


def _init(config_name):
    global X, SPL, CFG
    CFG = CONFIGS[config_name]
    torch.set_num_threads(2)
    meta = load_meta()
    X, SPL = load_signals(), splits(meta, nested=True)
    for d in SPL.values():
        check(d, meta)


def _run(task):
    name, seed, test = task
    model, report = train(X, SPL[name], seed, CFG, evaluate_test=test)
    if test:
        (OUT / "models").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), OUT / "models" / f"cnn_{name}_g{seed}.pt")
    print(f"{name:16s} seed {seed}: best epoch {report['best_epoch']:2d}, "
          f"val AUC {report['auc_val']:.3f}" + (f", test AUC {report['auc_test']:.3f}" if test else ""),
          flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["validation", "test"], required=True)
    ap.add_argument("--config", choices=list(CONFIGS), default="default")
    args = ap.parse_args()
    phase = args.phase
    OUT.mkdir(parents=True, exist_ok=True)
    # validation: the nested folds (position never seen, without touching the test set) in addition
    names = list(splits(load_meta(), nested=phase == "validation"))
    tasks = [(n, g, phase == "test") for n in names for g in SEEDS]
    with Pool(len(tasks), initializer=_init, initargs=(args.config,)) as pool:
        reports = pool.map(_run, tasks)
    (OUT / f"reports_{phase}_{args.config}.json").write_text(json.dumps(reports, ensure_ascii=False))
    print("\nsplit              val AUC (mean ± std)" + ("   test AUC (mean ± std)" if phase == "test" else ""))
    for n in names:
        b = [x for x in reports if x["split"] == n]
        v = np.array([x["auc_val"] for x in b])
        line = f"{n:16s}   {v.mean():.3f} ± {v.std():.3f}"
        if phase == "test":
            t = np.array([x["auc_test"] for x in b])
            line += f"        {t.mean():.3f} ± {t.std():.3f} (min {t.min():.3f})"
        print(line + f"   best epochs {[x['best_epoch'] for x in b]}")
