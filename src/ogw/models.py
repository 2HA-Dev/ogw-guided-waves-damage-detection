"""CNN model and measurement of its cost (parameters, multiply-accumulates)."""
import torch
from torch import nn


class CNN1D(nn.Module):
    """1D convolutional network on a whole measurement.

    Input (batch, 66 paths, L points): the paths are the channels, so the
    first layer learns filters specific to each path. Output: one logit per
    measurement (> 0: damaged). Blocks [Conv1d, BatchNorm, ReLU, MaxPool],
    then global maximum over time (echo position unknown).
    """

    def __init__(self, in_channels=66, widths=(32, 32, 64, 64), kernel=7, dropout=0.2):
        super().__init__()
        layers, c = [], in_channels
        for k, w in enumerate(widths):
            layers += [nn.Conv1d(c, w, kernel, padding=kernel // 2), nn.BatchNorm1d(w), nn.ReLU()]
            if k < len(widths) - 1:
                layers.append(nn.MaxPool1d(2))
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(c, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def n_parameters(model):
    return sum(p.numel() for p in model.parameters())


def n_mac(model, input_shape):
    """Multiply-accumulates of one inference (convolutions and linear layers)."""
    total = 0

    def hook(m, inp, out):
        nonlocal total
        if isinstance(m, nn.Conv1d):
            total += out.numel() * m.in_channels * m.kernel_size[0] // m.groups
        elif isinstance(m, nn.Linear):
            total += m.in_features * m.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Conv1d, nn.Linear))]
    state = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape))
    model.train(state)
    for h in handles:
        h.remove()
    return total
