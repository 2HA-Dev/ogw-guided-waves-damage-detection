import numpy as np

from ogw.anomaly import PCADetector


def dataset(n=120, c=5, L=200, seed=0):
    """Healthy: 2 shapes mixed according to a "temperature"; damaged: + an echo."""
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    a, b = np.sin(2 * np.pi * t / 20), np.cos(2 * np.pi * t / 20)
    T = rng.uniform(0, 1, n)
    X = np.stack([np.stack([(1 + k) * (np.cos(Ti) * a + np.sin(Ti) * b) for k in range(c)]) for Ti in T])
    X = X + rng.normal(0, 0.01, X.shape)
    y = np.zeros(n, dtype=int)
    y[-30:] = 1
    X[-30:, 3] += 0.3 * np.exp(-((t - 150) / 6) ** 2)
    return X.astype(np.float32), y


def test_detects_without_seeing_damage():
    X, y = dataset()
    d = PCADetector(k=2).fit(X, np.arange(60), np.arange(60, 90))
    s = d.scores(np.arange(90, 120))            # damaged, never seen
    assert s.min() > d.threshold
    assert np.argmax(np.nanmedian(d.residuals(np.arange(90, 120)), 0)) == 3   # right path


def test_empty_path_ignored():
    X, y = dataset()
    X[70, 1] = 0                                # empty path on a healthy validation measurement
    X[10, 2] = 0                                # and on a training measurement
    d = PCADetector(k=2).fit(X, np.arange(60), np.arange(60, 90))
    assert d.n_train == 59                      # the faulty training measurement is left out
    assert np.isfinite(d.healthy_val_scores).all()
    assert d.scores([70])[0] < d.scores(np.arange(90, 120)).min()


def test_cost():
    X, _ = dataset()
    d = PCADetector(k=2).fit(X, np.arange(60), np.arange(60, 90))
    assert d.cost()["mac"] == 2 * 2 * 5 * 200
