"""Training-free damage indicator: residual to a healthy baseline.

Principle (temperature compensation by baseline selection):
  1. a library of healthy measurements, taken only outside the test set;
  2. for each evaluated measurement, the baseline is chosen in the library
     (closest in setpoint, in measured temperature, or with minimal residual);
  3. relative residual per path r_p = ||x_p - ref_p||² / ||ref_p||², divided by
     its typical level in the absence of damage (calibration);
  4. indicator = maximum (or mean) over the valid paths. An empty path
     (faulty acquisition, in the measurement or in its baseline) is excluded:
     it is a sensor failure, not damage.

Calibration on the library alone, without seeing the test set: each healthy
measurement of the library is evaluated against the measurements of the other
runs (different measurement campaign, cycle, heating/cooling ramp), never against itself nor
its immediate neighbours. The decision threshold is the 95 % quantile of these
healthy indicators, i.e. a target false alarm rate of 5 %.
"""
import numpy as np
import torch

from ogw.data import faulty_paths

CHOICES = ("setpoint", "measured", "optimal")
AGGREGATIONS = ("max", "mean")


class BaselineIndicator:
    def __init__(self, choice="optimal", aggregation="max", quantile=0.95):
        assert choice in CHOICES and aggregation in AGGREGATIONS
        self.choice, self.aggregation, self.quantile = choice, aggregation, quantile

    def fit(self, X, meta, library):
        """X: signals (N, 66, L); library: indices of healthy measurements outside the test set."""
        lib = np.asarray(library)
        assert np.all(meta.damaged[lib] == 0), "the library must be healthy"
        self.X, self.meta, self.library = X, meta, lib
        E = (X.astype(np.float64) ** 2).sum(-1)
        self.faulty = faulty_paths(X, np.median(E[lib], axis=0))             # (N, 66)
        if self.choice == "optimal":   # distances from all measurements to the library
            Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float64).reshape(len(X), -1))
            self._D = torch.cdist(Xt, Xt[lib]).numpy()                         # (N, n_lib)
        run = np.array([hash((meta.label[i], meta.cycle[i], meta.ramp[i])) for i in lib])
        r = np.stack([self._residual(i, run != run[k]) for k, i in enumerate(lib)])
        self.scale = np.nanmedian(r, axis=0)                                  # (66,)
        self.calibration_scores = self._aggregate(r / self.scale)
        self.threshold = np.quantile(self.calibration_scores, self.quantile)
        return self

    def _baseline(self, i, mask):
        """Baseline of measurement i among the library members where mask is true."""
        m, cand = self.meta, np.flatnonzero(mask)
        if self.choice == "setpoint":
            key = np.abs(m.T_setpoint[self.library[cand]] - m.T_setpoint[i])
        elif self.choice == "measured":
            key = np.abs(m.T_measured[self.library[cand]] - m.T_measured[i])
        else:
            key = self._D[i, cand]
        return self.library[cand[np.argmin(key)]]

    def _residual(self, i, mask):
        mask = mask & (self.library != i)          # never its own baseline
        j = self._baseline(i, mask)
        r = ((self.X[i] - self.X[j]) ** 2).sum(-1) / (self.X[j] ** 2).sum(-1)
        return np.where(self.faulty[i] | self.faulty[j], np.nan, r)

    def _aggregate(self, r):
        return np.nanmax(r, -1) if self.aggregation == "max" else np.nanmean(r, -1)

    def residuals(self, indices):
        """Normalized residuals per path (n, 66); NaN on faulty paths."""
        all_ = np.ones(len(self.library), dtype=bool)
        return np.stack([self._residual(i, all_) for i in indices]) / self.scale

    def scores(self, indices):
        return self._aggregate(self.residuals(indices))
