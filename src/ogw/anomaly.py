"""PCA anomaly detection, trained on healthy measurements only.

Idea: healthy signals, despite temperature, lie close to a low-dimensional
subspace (5 components are enough on OGW at 40 kHz). A measurement is
projected onto this subspace; what remains (the residual) is what the healthy
model does not explain. No damage is seen during training: the method does
not depend on the damage position.

Indicator: relative residual per path, divided by its typical level on healthy
validation measurements, maximum over the valid paths. Threshold: 95 %
quantile of the healthy validation indicators. Faulty (empty) paths are
excluded from the projection and from the indicator.
"""
import numpy as np

from ogw.data import Normalizer, faulty_paths


class PCADetector:
    def __init__(self, k=5, quantile=0.95):
        self.k, self.quantile = k, quantile

    def fit(self, X, healthy_train, healthy_val):
        self.X = X
        E = (X.astype(np.float64) ** 2).sum(-1)
        self.typical_energy = np.median(E[healthy_train], axis=0)                 # (66,)
        self.faulty = faulty_paths(X, self.typical_energy)                         # (N, 66)
        # training measurements with an empty path are left out of the fit
        clean = np.asarray(healthy_train)[~self.faulty[healthy_train].any(1)]
        self.norm = Normalizer().fit(X[clean])
        F = self._flatten(clean)
        self.mean = F.mean(0)
        _, _, Vt = np.linalg.svd(F - self.mean, full_matrices=False)
        self.components = Vt[:self.k]                                              # (k, 66 L)
        self.n_train = len(clean)
        r = self._raw_residuals(healthy_val)
        self.scale = np.nanmedian(r, axis=0)
        self.healthy_val_scores = np.nanmax(r / self.scale, axis=1)
        self.threshold = np.quantile(self.healthy_val_scores, self.quantile)
        return self

    def _flatten(self, idx):
        return self.norm.apply(self.X[idx]).reshape(len(idx), -1).astype(np.float64)

    def _raw_residuals(self, idx):
        idx = np.asarray(idx)
        F = self._flatten(idx) - self.mean
        L = self.X.shape[-1]
        R = np.empty_like(F)
        for n, i in enumerate(idx):
            valid = np.repeat(~self.faulty[i], L)
            V = self.components[:, valid]
            z = np.linalg.solve(V @ V.T, V @ F[n, valid])      # projection on the valid paths
            R[n] = F[n] - z @ self.components
        R = R.reshape(len(idx), self.X.shape[1], L)
        Xn = self._flatten(idx).reshape(R.shape)
        with np.errstate(divide="ignore", invalid="ignore"):   # empty paths, set to NaN
            r = (R ** 2).sum(-1) / (Xn ** 2).sum(-1)
        r[self.faulty[idx]] = np.nan
        return r

    def residual_signals(self, idx):
        """Projection residuals over time (n, 66, L), in normalized units;
        faulty paths set to zero (projection on the valid paths)."""
        idx = np.asarray(idx)
        F = self._flatten(idx) - self.mean
        L = self.X.shape[-1]
        R = np.empty_like(F)
        for n, i in enumerate(idx):
            valid = np.repeat(~self.faulty[i], L)
            V = self.components[:, valid]
            z = np.linalg.solve(V @ V.T, V @ F[n, valid])
            R[n] = (F[n] - z @ self.components) * valid
        return R.reshape(len(idx), self.X.shape[1], L)

    def residuals(self, idx):
        """Normalized residuals per path (n, 66); NaN on faulty paths."""
        return self._raw_residuals(idx) / self.scale

    def scores(self, idx):
        return np.nanmax(self.residuals(idx), axis=1)

    def cost(self):
        """Memory (stored values) and multiply-accumulates per measurement."""
        d = self.components.shape[1]
        return dict(values=(self.k + 1) * d + 2 * self.X.shape[1], mac=2 * self.k * d)
