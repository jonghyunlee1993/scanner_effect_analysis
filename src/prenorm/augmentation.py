"""Scanner-independent nuisance augmentation for content inputs.

Clean RGB targets never pass through this module.  Each image receives an
independent draw so registered tuples do not share acquisition nuisance.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinate = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coordinate / sigma).square())
    kernel = kernel / kernel.sum()
    return kernel[:, None] * kernel[None, :]


def _blur_one(image: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = _gaussian_kernel(sigma, image.device, image.dtype)
    weight = kernel.expand(image.shape[0], 1, *kernel.shape)
    pad = kernel.shape[0] // 2
    padded = F.pad(image[None], (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(padded, weight, groups=image.shape[0])[0]


class DifferentiableJPEG(nn.Module):
    """Apply a compact JPEG-like 8x8 DCT quantization with straight-through rounding."""

    def __init__(self):
        super().__init__()
        coordinate = torch.arange(8, dtype=torch.float32)
        frequency = coordinate[:, None]
        dct = torch.cos(math.pi * (2 * coordinate[None] + 1) * frequency / 16)
        dct[0] /= math.sqrt(2.0)
        self.register_buffer("dct", dct * 0.5)
        self.register_buffer(
            "quantization",
            torch.tensor(
                (
                    (16, 11, 10, 16, 24, 40, 51, 61),
                    (12, 12, 14, 19, 26, 58, 60, 55),
                    (14, 13, 16, 24, 40, 57, 69, 56),
                    (14, 17, 22, 29, 51, 87, 80, 62),
                    (18, 22, 37, 56, 68, 109, 103, 77),
                    (24, 35, 55, 64, 81, 104, 113, 92),
                    (49, 64, 78, 87, 103, 121, 120, 101),
                    (72, 92, 95, 98, 112, 100, 103, 99),
                ),
                dtype=torch.float32,
            ),
        )

    def forward(self, image: torch.Tensor, quality: float) -> torch.Tensor:
        """Compress one CHW image in [0, 1]."""
        channels, height, width = image.shape
        blocks = F.unfold(image[None], kernel_size=8, stride=8)
        blocks = blocks.view(channels, 64, -1).transpose(1, 2).view(channels, -1, 8, 8)
        centered = blocks * 255.0 - 128.0
        coefficients = self.dct @ centered @ self.dct.t()
        quality = min(max(float(quality), 1.0), 100.0)
        scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
        step = (self.quantization * scale / 100.0).clamp_min(1.0)
        normalized = coefficients / step
        quantized = normalized + (normalized.round() - normalized).detach()
        restored = self.dct.t() @ (quantized * step) @ self.dct
        restored = ((restored + 128.0) / 255.0).clamp(0.0, 1.0)
        restored = restored.view(channels, -1, 64).transpose(1, 2).reshape(1, channels * 64, -1)
        return F.fold(restored, (height, width), kernel_size=8, stride=8)[0]


class ContentNuisanceAugmenter(nn.Module):
    """Apply calibrated color, illumination, PSF, resampling, noise, and JPEG nuisance."""

    def __init__(
        self,
        brightness: float = 0.12,
        contrast: float = 0.25,
        gamma: float = 0.25,
        saturation: float = 0.30,
        illumination: float = 0.10,
        blur_probability: float = 0.5,
        blur_sigma_max: float = 1.5,
        unsharp_probability: float = 0.2,
        noise: float = 0.02,
        jpeg_quality: tuple[int, int] = (35, 95),
        resample_min_scale: float = 0.75,
    ):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.saturation = saturation
        self.illumination = illumination
        self.blur_probability = blur_probability
        self.blur_sigma_max = blur_sigma_max
        self.unsharp_probability = unsharp_probability
        self.noise = noise
        self.jpeg_quality = jpeg_quality
        self.resample_min_scale = resample_min_scale
        self.jpeg = DifferentiableJPEG()

    @staticmethod
    def _uniform(image: torch.Tensor, magnitude: float) -> torch.Tensor:
        return (torch.rand((), device=image.device) * 2.0 - 1.0) * magnitude

    def _augment_one(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        x = (image + 1.0) * 0.5
        mean = x.mean(dim=(1, 2), keepdim=True)
        x = (x - mean) * (1.0 + self._uniform(x, self.contrast * strength)) + mean
        x = x + self._uniform(x, self.brightness * strength)
        exponent = 1.0 + self._uniform(x, self.gamma * strength)
        x = x.clamp(0.0, 1.0).pow(exponent)
        gray = (x * x.new_tensor((0.299, 0.587, 0.114))[:, None, None]).sum(0, keepdim=True)
        x = gray + (x - gray) * (1.0 + self._uniform(x, self.saturation * strength))

        field = torch.randn(1, 1, 4, 4, device=x.device, dtype=x.dtype)
        field = F.interpolate(field, x.shape[-2:], mode="bicubic", align_corners=False)[0]
        x = x * (1.0 + field * self.illumination * strength)

        if torch.rand((), device=x.device) < self.blur_probability * strength:
            sigma = 0.3 + float(torch.rand(())) * max(self.blur_sigma_max * strength - 0.3, 0.0)
            x = _blur_one(x, sigma)
        elif torch.rand((), device=x.device) < self.unsharp_probability * strength:
            x = x + 0.5 * strength * (x - _blur_one(x, 1.0))

        if self.resample_min_scale < 1.0 and torch.rand((), device=x.device) < 0.5 * strength:
            scale = 1.0 - float(torch.rand(())) * (1.0 - self.resample_min_scale) * strength
            size = [max(8, int(round(d * scale))) for d in x.shape[-2:]]
            x = F.interpolate(x[None], size=size, mode="bilinear", align_corners=False)
            x = F.interpolate(x, size=image.shape[-2:], mode="bilinear", align_corners=False)[0]

        if torch.rand((), device=x.device) < strength:
            low, high = self.jpeg_quality
            quality = high - float(torch.rand(())) * (high - low) * strength
            x = self.jpeg(x.clamp(0.0, 1.0), quality)
        if self.noise > 0:
            x = x + torch.randn_like(x) * self.noise * strength
        return x.clamp(0.0, 1.0) * 2.0 - 1.0

    def forward(self, rgb: torch.Tensor, strength: float) -> torch.Tensor:
        """Augment BS3HW or N3HW RGB while preserving its outer shape."""
        if strength <= 0:
            return rgb
        leading = rgb.shape[:-3]
        flat = rgb.reshape(-1, *rgb.shape[-3:])
        output = torch.stack([self._augment_one(image, strength) for image in flat])
        return output.view(*leading, *output.shape[-3:])
