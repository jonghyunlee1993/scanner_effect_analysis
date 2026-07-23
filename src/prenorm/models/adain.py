"""Adaptive Instance Normalization and the style->params MLP.

AdaIN normalizes a feature map per-instance/per-channel then applies (gamma, beta)
predicted from the style code z. StyleMLP maps z to all AdaIN parameters at once,
shaped (N, n_layers, 2, channels) so the generator can index per layer.
"""
import torch
import torch.nn as nn


class AdaIN(nn.Module):
    """Apply adaptive instance norm: normalize feat, then scale/shift by (gamma, beta)."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, feat, gamma, beta):
        """feat (N, C, H, W); gamma, beta (N, C)."""
        mean = feat.mean(dim=(2, 3), keepdim=True)
        var = feat.var(dim=(2, 3), keepdim=True, unbiased=False)
        norm = (feat - mean) / torch.sqrt(var + self.eps)
        return norm * gamma[:, :, None, None] + beta[:, :, None, None]


class StyleMLP(nn.Module):
    """z (N, style_dim) -> AdaIN params (N, n_layers, 2, channels)."""

    def __init__(self, style_dim, n_layers, channels, hidden=256):
        super().__init__()
        self.n_layers = n_layers
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(style_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_layers * 2 * channels),
        )

    def forward(self, z):
        return self.net(z).view(z.size(0), self.n_layers, 2, self.channels)
