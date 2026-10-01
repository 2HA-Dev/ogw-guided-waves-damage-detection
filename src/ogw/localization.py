"""Damage localization from per-path damage indices.

Geometry (Moll et al., Scientific Data 6, 191, 2019, tables 1 and 2):
500 × 500 mm plate; transducer T(k+1) = index k of the HDF5 file.

RAPID (reconstruction algorithm for probabilistic inspection of damage):
each transmitter-receiver path (a, b) contributes to the map with its damage
index DI, weighted by an elliptical window centred on the path:
  R(p) = (|p - a| + |p - b|) / |a - b| ,   w(p) = max(0, (β - R) / (β - 1)).
The estimated position is the maximum of the map. No training: the method
depends only on the geometry and on β (set to 1.05 before the experiments).
Limitation (checked on synthetic data): damage lying on an edge path is not
located along that path, for lack of paths crossing it; normalization by the
coverage (option) makes the problem worse.

Delay and sum: also uses the time of flight. The wave scattered by damage at p
reaches path (a, b) at
  t(p) = (|a - p| + |p - b|) / v + t0 ;
the image sums, over the paths, the envelope of the residual signal at that time.
v and t0 are calibrated on healthy measurements (direct arrivals).
"""
import numpy as np
from scipy.signal import hilbert

TRANSDUCERS = np.array([[450, 470], [370, 470], [290, 470], [210, 470], [130, 470], [50, 470],
                        [450, 30], [370, 30], [290, 30], [210, 30], [130, 30], [50, 30]], dtype=float)
DEFECTS = {"D04": (65.0, 400.0), "D12": (195.0, 330.0), "D16": (335.0, 260.0), "D24": (450.0, 190.0)}
SIDE = 500.0


def damage_index(residuals):
    """DI per path: excess of the normalized residual above its typical healthy level (1)."""
    return np.clip(np.nan_to_num(residuals, nan=1.0) - 1.0, 0.0, None)


class RAPID:
    def __init__(self, paths, beta=1.05, step=2.0, normalize=False):
        self.beta, self.paths, self.normalize = beta, np.asarray(paths), normalize
        g = np.arange(step / 2, SIDE, step)
        self.x, self.y = np.meshgrid(g, g)                              # grid (ny, nx) in mm
        a, b = TRANSDUCERS[self.paths[:, 0]], TRANSDUCERS[self.paths[:, 1]]
        da = np.hypot(self.x[None] - a[:, 0, None, None], self.y[None] - a[:, 1, None, None])
        db = np.hypot(self.x[None] - b[:, 0, None, None], self.y[None] - b[:, 1, None, None])
        R = (da + db) / np.hypot(*(a - b).T)[:, None, None]
        self.W = np.clip((beta - R) / (beta - 1.0), 0.0, None)          # (n_paths, ny, nx)
        coverage = self.W.sum(0)
        # outside every ellipse, the map is 0 (no division by zero)
        self.inv_coverage = np.where(coverage > 0, 1.0 / np.maximum(coverage, 1e-12), 0.0)

    def map(self, di):
        c = np.tensordot(di, self.W, axes=1)
        return c * self.inv_coverage if self.normalize else c

    def localize(self, di):
        """di: (n, n_paths) or (n_paths,) -> estimated positions (n, 2) in mm."""
        di = np.atleast_2d(di)
        maps = np.tensordot(di, self.W, axes=1)
        if self.normalize:
            maps = maps * self.inv_coverage
        maps = maps.reshape(len(di), -1)
        k = maps.argmax(1)
        return np.stack([self.x.ravel()[k], self.y.ravel()[k]], axis=1)


def error(estimated, position):
    return np.hypot(*(np.asarray(estimated) - np.asarray(position)).T)


def calibrate_velocity(X_healthy, paths, fs, t_min_us=150.0, min_length=250.0):
    """Velocity (mm/µs) and delay t0 (µs) by regression t = d / v + t0 on the
    direct arrivals of the long paths (first envelope maximum above 30 % of
    the maximum, after the crosstalk). X_healthy: (n, 66, L), healthy measurements."""
    paths = np.asarray(paths)
    d = np.hypot(*(TRANSDUCERS[paths[:, 0]] - TRANSDUCERS[paths[:, 1]]).T)
    t = np.arange(X_healthy.shape[-1]) / fs * 1e6
    env = np.abs(hilbert(X_healthy, axis=-1))
    env[..., t < t_min_us] = 0
    arrival = np.empty(env.shape[:2])
    for n, k in np.ndindex(*env.shape[:2]):
        e = env[n, k]
        j = int(np.argmax(e > 0.3 * e.max()))
        while j + 1 < len(e) and e[j + 1] >= e[j]:
            j += 1
        arrival[n, k] = t[j]
    ok = d > min_length
    A = np.c_[np.tile(d[ok], len(X_healthy)), np.ones(ok.sum() * len(X_healthy))]
    (inv_v, t0), *_ = np.linalg.lstsq(A, arrival[:, ok].ravel(), rcond=None)
    rms = float(np.sqrt(np.mean((A @ [inv_v, t0] - arrival[:, ok].ravel()) ** 2)))
    return 1.0 / inv_v, t0, rms


class DelayAndSum:
    def __init__(self, paths, v, t0, fs, L, step=2.0):
        self.paths = np.asarray(paths)
        g = np.arange(step / 2, SIDE, step)
        self.x, self.y = np.meshgrid(g, g)
        a, b = TRANSDUCERS[self.paths[:, 0]], TRANSDUCERS[self.paths[:, 1]]
        da = np.hypot(self.x[None] - a[:, 0, None, None], self.y[None] - a[:, 1, None, None])
        db = np.hypot(self.x[None] - b[:, 0, None, None], self.y[None] - b[:, 1, None, None])
        k = np.rint(((da + db) / v + t0) * 1e-6 * fs).astype(np.int64)
        self.valid = k < L
        self.k = np.minimum(k, L - 1)                                   # (n_paths, ny, nx)

    def map(self, env):
        """env: (n_paths, L) residual envelopes -> map (ny, nx)."""
        idx = np.arange(len(self.paths))[:, None, None]
        return np.where(self.valid, env[idx, self.k], 0.0).sum(0)

    def localize(self, env):
        env = np.asarray(env)[None] if np.ndim(env) == 2 else np.asarray(env)
        pos = []
        for e in env:
            c = self.map(e).ravel()
            j = int(c.argmax())
            pos.append((self.x.ravel()[j], self.y.ravel()[j]))
        return np.array(pos)
