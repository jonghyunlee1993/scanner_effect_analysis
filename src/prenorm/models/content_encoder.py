"""Content encoder C: differentiable RGB/structure inputs -> spatial content.

InstanceNorm throughout suppresses residual scanner style so the content map is
appearance-free; spatial extent is preserved (downsampled) for L_pair.
"""
import torch.nn as nn


class ResBlockIN(nn.Module):
    """Residual block with InstanceNorm + ReLU."""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ContentEncoder(nn.Module):
    def __init__(self, input_channels=5, content_ch=256, n_downsample=2, n_res=4):
        super().__init__()
        base = content_ch // (2 ** n_downsample)
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(input_channels, base, 7),
                  nn.InstanceNorm2d(base), nn.ReLU(inplace=True)]
        dim = base
        for _ in range(n_downsample):
            layers += [nn.Conv2d(dim, dim * 2, 4, 2, 1),
                       nn.InstanceNorm2d(dim * 2), nn.ReLU(inplace=True)]
            dim *= 2
        for _ in range(n_res):
            layers.append(ResBlockIN(dim))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        """Map NCHW model inputs to a stride-four spatial content tensor."""
        return self.model(inputs)
