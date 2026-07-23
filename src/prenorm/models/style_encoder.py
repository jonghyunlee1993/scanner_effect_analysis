"""Style encoder E: clean RGB context -> scanner-style vector.

Shallow conv stack with NO normalization (norm would strip the very style we want
to capture) -> global average pool -> linear bottleneck. Weights shared across
scanners so a dispersed set can estimate one shared scanner-style code.
"""
import torch
import torch.nn as nn


class StyleEncoder(nn.Module):
    def __init__(self, in_ch=3, style_dim=32, ndf=64, n_downsample=4):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ndf, 7), nn.ReLU(inplace=True)]
        dim = ndf
        for _ in range(n_downsample):
            out = min(dim * 2, ndf * 4)
            layers += [nn.Conv2d(dim, out, 4, 2, 1), nn.ReLU(inplace=True)]
            dim = out
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(dim, style_dim)

    def forward(self, rgb):
        """rgb (N, 3, H, W) -> z (N, style_dim)."""
        return self.fc(self.conv(rgb).flatten(1))


class SetStyleEncoder(nn.Module):
    """Encode a spatially dispersed set with masked mean pooling."""

    def __init__(self, in_ch=3, style_dim=64, ndf=64, n_downsample=4):
        super().__init__()
        self.patch_encoder = StyleEncoder(in_ch, style_dim, ndf, n_downsample)

    def forward(self, context, present=None):
        """Map ``(..., K, 3, H, W)`` clean context patches to ``(..., D)``."""
        if context.ndim < 5:
            raise ValueError("context must end with (K, 3, H, W)")
        set_shape = context.shape[:-4]
        set_size = context.shape[-4]
        patch_codes = self.patch_encoder(context.reshape(-1, *context.shape[-3:]))
        patch_codes = patch_codes.reshape(*set_shape, set_size, -1)
        if present is None:
            return patch_codes.mean(dim=-2)

        weights = present.to(device=context.device, dtype=patch_codes.dtype)
        if tuple(weights.shape) != tuple(context.shape[:-3]):
            raise ValueError("present must match the context set dimensions")
        denominator = weights.sum(dim=-1, keepdim=True)
        if torch.any(denominator == 0):
            raise ValueError("each style context must contain a present patch")
        return (patch_codes * weights[..., None]).sum(dim=-2) / denominator
