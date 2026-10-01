"""Training of a model on a split, without data leakage.

- normalization learned on the training set only;
- early stopping on the validation loss, best state restored;
- the test set is evaluated only on explicit request (evaluate_test=True), once
  the configuration has been frozen on the validation set.
"""
import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from ogw.data import Normalizer, OGWSignals
from ogw.models import CNN1D


@dataclass
class Config:
    epochs: int = 80
    patience: int = 15
    batch: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4      # L2 regularization (AdamW)
    dropout: float = 0.2            # dropout before the final layer
    widths: tuple = (32, 32, 64, 64)
    kernel: int = 7


def predict(model, X, batch=64):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(X[i:i + batch])) for i in range(0, len(X), batch)]).numpy()


def _auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def train(X, d, seed, cfg=Config(), evaluate_test=False):
    """X: preprocessed signals (N, 66, L); d: Split. Returns (model, report)."""
    torch.manual_seed(seed)
    Xn = Normalizer().fit(X[d.train]).apply(X).astype(np.float32)
    y = d.target.astype(np.float32)
    ytr = y[d.train]
    loader = DataLoader(OGWSignals(Xn, y, d.train), batch_size=cfg.batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    # imbalanced classes: positives are weighted to balance the loss
    weight = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    model = CNN1D(X.shape[1], cfg.widths, cfg.kernel, cfg.dropout)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    Xv, yv = torch.from_numpy(Xn[d.validation]), torch.from_numpy(y[d.validation])

    history, best, wait, best_epoch = [], np.inf, 0, 0
    state = copy.deepcopy(model.state_dict())
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            sv = model(Xv)
            pv = loss_fn(sv, yv).item()
        history.append(dict(epoch=epoch, train_loss=total / len(d.train),
                            val_loss=pv, auc_val=_auc(yv.numpy(), sv.numpy())))
        if pv < best - 1e-4:
            best, wait, best_epoch = pv, 0, epoch
            state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(state)

    sv = predict(model, Xn[d.validation])
    report = dict(split=d.name, seed=seed, config=asdict(cfg), best_epoch=best_epoch,
                  history=history, auc_val=_auc(y[d.validation], sv))
    if evaluate_test:
        st = predict(model, Xn[d.test])
        report.update(scores_test=st.tolist(), auc_test=_auc(y[d.test], st))
    return model, report
