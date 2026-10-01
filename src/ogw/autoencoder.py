"""Deep autoencoders trained on healthy measurements.

Same role as PCA (ogw.anomaly): reconstruct the healthy signal; what is not
reconstructed is suspicious. Same indicator: relative residual per path,
divided by its typical level on the healthy validation measurements, maximum
over the valid paths; threshold at the 95 % quantile of the healthy validation
measurements.

- AEConv: compact convolutional model (59,524 to 85,900 parameters for a latent size of 2 to 10),
  designed for embedded use;
- AEDense: fully connected encoder and decoder; the last layer
  (64 -> 66 × L) gives the decoder the capacity to reproduce the signals finely.
"""
import copy

import numpy as np
import torch
from torch import nn

from ogw.data import Normalizer, faulty_paths


class AEConv(nn.Module):
    def __init__(self, latent, channels=66, L=820):
        super().__init__()
        self.L = L
        self.enc = nn.Sequential(
            nn.Conv1d(channels, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 16, 7, stride=2, padding=3), nn.ReLU())
        with torch.no_grad():
            self.feature_shape = self.enc(torch.zeros(1, channels, L)).shape[1:]
        n = int(np.prod(self.feature_shape))
        self.to_code, self.from_code = nn.Linear(n, latent), nn.Linear(latent, n)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(16, 32, 7, stride=2, padding=3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 32, 7, stride=2, padding=3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, channels, 7, stride=2, padding=3, output_padding=1))

    def forward(self, x):
        z = self.to_code(self.enc(x).flatten(1))
        y = self.dec(torch.relu(self.from_code(z)).view(-1, *self.feature_shape))
        return y[..., :self.L] if y.shape[-1] >= self.L else nn.functional.pad(y, (0, self.L - y.shape[-1]))


class AEDense(nn.Module):
    def __init__(self, latent, channels=66, L=820, hidden=64):
        super().__init__()
        d = channels * L
        self.feature_shape = (channels, L)
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(self, x):
        return self.dec(self.enc(x)).view(-1, *self.feature_shape)


class AEDetector:
    def __init__(self, architecture, latent, seed=0, epochs=3000, patience=200, lr=1e-3):
        self.architecture, self.latent, self.seed = architecture, latent, seed
        self.epochs, self.patience, self.lr = epochs, patience, lr

    def fit(self, X, healthy_train, healthy_val):
        torch.manual_seed(self.seed)
        E = (X.astype(np.float64) ** 2).sum(-1)
        self.faulty = faulty_paths(X, np.median(E[healthy_train], axis=0))
        clean = np.asarray(healthy_train)[~self.faulty[healthy_train].any(1)]
        self.norm = Normalizer().fit(X[clean])
        self.Xn = torch.from_numpy(self.norm.apply(X).astype(np.float32))
        xe = self.Xn[clean]
        val_clean = np.asarray(healthy_val)[~self.faulty[healthy_val].any(1)]
        xv = self.Xn[val_clean]
        self.model = self.architecture(self.latent, X.shape[1], X.shape[2])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best, wait, state = np.inf, 0, copy.deepcopy(self.model.state_dict())
        for ep in range(self.epochs):                 # full batch: only 79 measurements
            self.model.train()
            opt.zero_grad()
            loss = ((self.model(xe) - xe) ** 2).mean()
            loss.backward()
            opt.step()
            self.model.eval()
            with torch.no_grad():
                pv = ((self.model(xv) - xv) ** 2).mean().item()
            if pv < best - 1e-7:
                best, wait, state, self.best_epoch = pv, 0, copy.deepcopy(self.model.state_dict()), ep
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.model.load_state_dict(state)
        self.val_loss, self.n_train = best, len(clean)
        r = self._raw_residuals(healthy_val)
        self.scale = np.nanmedian(r, axis=0)
        self.healthy_val_scores = np.nanmax(r / self.scale, axis=1)
        self.threshold = np.quantile(self.healthy_val_scores, 0.95)
        return self

    def _raw_residuals(self, idx):
        idx = np.asarray(idx)
        self.model.eval()
        with torch.no_grad():
            x = self.Xn[idx]
            r = (((self.model(x) - x) ** 2).sum(-1) / (x ** 2).sum(-1).clamp_min(1e-12)).numpy().astype(np.float64)
        r[self.faulty[idx]] = np.nan
        return r

    def residuals(self, idx):
        return self._raw_residuals(idx) / self.scale

    def scores(self, idx):
        return np.nanmax(self.residuals(idx), axis=1)

    def n_parameters(self):
        return sum(p.numel() for p in self.model.parameters())
