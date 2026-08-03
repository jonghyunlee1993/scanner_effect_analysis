"""Frozen E5 LOSO comparator definitions and pure numerical operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


E5_VERSION = "e5_loso_comparators_v1"
SCANNERS = ("at2", "gt450", "versa", "akoya", "s60", "s360")
FOVS = (224, 256, 512)
IMAGE_CONDITIONS = ("reinhard_lab", "paired_od_affine", "frequency_calibration")
FEATURE_CONDITIONS = ("coral", "orthogonal_procrustes")
SAMPLE_STRIDE = 8
SAMPLE_OFFSET = 4
RADIAL_BINS = 72
TARGET_MPP = 0.5052
FREQUENCY_ANCHOR = (0.03, 0.10)
FREQUENCY_UNITY_BELOW = 0.10
CORAL_CANDIDATES = (0.0001, 0.001, 0.01, 0.05, 0.10, 0.20)
GAIN_CAP_CANDIDATES = (
    1.01,
    1.02,
    1.03,
    1.04,
    1.05,
    1.10,
    1.15,
    1.20,
    1.25,
    1.50,
    2.00,
    3.00,
    4.00,
)


def centered_crop(array: np.ndarray, fov: int) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim < 3 or value.shape[-3:] != (512, 512, 3):
        raise ValueError(f"expected (...,512,512,3) RGB, got {value.shape}")
    if fov not in FOVS:
        raise ValueError(f"unsupported E5 FOV: {fov}")
    start = (512 - fov) // 2
    return value[..., start : start + fov, start : start + fov, :]


def rgb8_to_rgb01(value: np.ndarray | torch.Tensor, device=None) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.shape[-1] != 3:
        raise ValueError("RGB input must have three channels last")
    return tensor.float().div_(255.0)


def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    return torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).pow(2.4),
    )


def _linear_to_srgb(rgb: torch.Tensor) -> torch.Tensor:
    positive = rgb.clamp_min(0.0)
    return torch.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * positive.pow(1.0 / 2.4) - 0.055,
    )


def rgb01_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert channel-last sRGB in [0,1] to CIE Lab using the D65 white point."""
    if rgb.shape[-1] != 3:
        raise ValueError("RGB input must have three channels last")
    linear = _srgb_to_linear(rgb)
    matrix = torch.tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=linear.dtype,
        device=linear.device,
    )
    xyz = linear @ matrix.T
    white = torch.tensor([0.95047, 1.0, 1.08883], dtype=xyz.dtype, device=xyz.device)
    scaled = xyz / white
    delta = 6.0 / 29.0
    transformed = torch.where(
        scaled > delta**3,
        scaled.clamp_min(0.0).pow(1.0 / 3.0),
        scaled / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return torch.stack(
        [
            116.0 * transformed[..., 1] - 16.0,
            500.0 * (transformed[..., 0] - transformed[..., 1]),
            200.0 * (transformed[..., 1] - transformed[..., 2]),
        ],
        dim=-1,
    )


def lab_to_rgb01(lab: torch.Tensor) -> torch.Tensor:
    """Convert channel-last CIE Lab to unclipped sRGB."""
    if lab.shape[-1] != 3:
        raise ValueError("Lab input must have three channels last")
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    stacked = torch.stack([fx, fy, fz], dim=-1)
    delta = 6.0 / 29.0
    scaled = torch.where(
        stacked > delta,
        stacked**3,
        3.0 * delta**2 * (stacked - 4.0 / 29.0),
    )
    white = torch.tensor([0.95047, 1.0, 1.08883], dtype=lab.dtype, device=lab.device)
    xyz = scaled * white
    inverse = torch.tensor(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=lab.dtype,
        device=lab.device,
    )
    return _linear_to_srgb(xyz @ inverse.T)


def sampled_pixels(rgb01: torch.Tensor) -> torch.Tensor:
    if rgb01.ndim != 5 or rgb01.shape[0] != 6 or rgb01.shape[-1] != 3:
        raise ValueError(f"expected 6xNxHxWx3 RGB, got {tuple(rgb01.shape)}")
    return rgb01[
        :,
        :,
        SAMPLE_OFFSET::SAMPLE_STRIDE,
        SAMPLE_OFFSET::SAMPLE_STRIDE,
        :,
    ].reshape(6, -1, 3)


def rgb01_to_od(rgb: torch.Tensor) -> torch.Tensor:
    return -torch.log(((rgb * 255.0 + 1.0) / 256.0).clamp_min(1.0 / 256.0))


def od_to_rgb01(od: torch.Tensor) -> torch.Tensor:
    return (torch.exp(-od) * 256.0 - 1.0) / 255.0


def moments(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    flat = values.reshape(-1, values.shape[-1]).double()
    return flat.sum(dim=0), flat.square().sum(dim=0), int(len(flat))


def mean_std(total: np.ndarray, square: np.ndarray, count: int, floor=1e-3):
    if count <= 1:
        raise ValueError("moment count must exceed one")
    mean = np.asarray(total, dtype=np.float64) / int(count)
    variance = np.asarray(square, dtype=np.float64) / int(count) - mean**2
    return mean, np.sqrt(np.maximum(variance, float(floor) ** 2))


def reinhard_lab(
    rgb01: torch.Tensor,
    source_mean: torch.Tensor,
    source_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    lab = rgb01_to_lab(rgb01)
    corrected_lab = (lab - source_mean) / source_std.clamp_min(1e-3) * target_std + target_mean
    preclip = lab_to_rgb01(corrected_lab)
    output = preclip.clamp(0.0, 1.0)
    return clipping_report(preclip, output)


def od_affine_statistics(source_rgb: torch.Tensor, target_rgb: torch.Tensor):
    source = rgb01_to_od(source_rgb).reshape(-1, 3).double()
    target = rgb01_to_od(target_rgb).reshape(-1, 3).double()
    if source.shape != target.shape:
        raise ValueError("paired OD inputs differ in shape")
    design = torch.cat([source, torch.ones_like(source[:, :1])], dim=1)
    return design.T @ design, design.T @ target, int(len(design))


def solve_od_affine(xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
    a = np.asarray(xtx, dtype=np.float64)
    b = np.asarray(xty, dtype=np.float64)
    if a.shape != (4, 4) or b.shape != (4, 3) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid OD affine sufficient statistics")
    ridge = 1e-6 * float(np.diag(a)[:3].mean())
    regularizer = np.diag([ridge, ridge, ridge, 0.0])
    return np.linalg.solve(a + regularizer, b)


def paired_od_affine(rgb01: torch.Tensor, coefficient: torch.Tensor):
    od = rgb01_to_od(rgb01)
    design = torch.cat([od, torch.ones_like(od[..., :1])], dim=-1)
    preclip = od_to_rgb01(design @ coefficient)
    output = preclip.clamp(0.0, 1.0)
    return clipping_report(preclip, output)


@dataclass(frozen=True)
class RadialGeometry:
    index: np.ndarray
    valid: np.ndarray
    counts: np.ndarray
    frequency: np.ndarray
    nyquist: float


def radial_geometry(size: int, bins: int = RADIAL_BINS, mpp: float = TARGET_MPP):
    one = np.fft.fftfreq(int(size), d=float(mpp))
    radius = np.sqrt(one[:, None] ** 2 + one[None, :] ** 2)
    nyquist = 1.0 / (2.0 * float(mpp))
    edges = np.linspace(0.0, nyquist, int(bins) + 1)
    index = np.digitize(radius.ravel(), edges) - 1
    valid = (index >= 0) & (index < bins)
    counts = np.bincount(index[valid], minlength=bins).astype(np.float64)
    return RadialGeometry(index, valid, counts, (edges[:-1] + edges[1:]) / 2.0, nyquist)


def batch_radial_power(rgb01: torch.Tensor, geometry: RadialGeometry) -> torch.Tensor:
    """Sum Hann-windowed mean-OD radial power over a channel-last image batch."""
    if rgb01.ndim != 4 or rgb01.shape[-1] != 3 or rgb01.shape[1] != rgb01.shape[2]:
        raise ValueError("radial power requires NxHxWx3 square RGB")
    size = rgb01.shape[1]
    window = torch.hann_window(size, periodic=False, dtype=rgb01.dtype, device=rgb01.device)
    window = window[:, None] * window[None, :]
    value = rgb01_to_od(rgb01).mean(dim=-1)
    value = value - value.mean(dim=(1, 2), keepdim=True)
    power = torch.fft.fft2(value * window).abs().square().sum(dim=0).double().reshape(-1)
    index = torch.as_tensor(geometry.index[geometry.valid], dtype=torch.long, device=power.device)
    valid = torch.as_tensor(geometry.valid, dtype=torch.bool, device=power.device)
    return torch.bincount(index, weights=power[valid], minlength=len(geometry.counts))


def smooth_log_curve(log_gain: np.ndarray) -> np.ndarray:
    value = np.asarray(log_gain, dtype=np.float64)
    if value.ndim != 1:
        raise ValueError("frequency gain curve must be one-dimensional")
    padded = np.pad(value, (2, 2), mode="edge")
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
    return np.convolve(padded, kernel, mode="valid")


def fitted_frequency_gain(
    source_power: np.ndarray,
    target_power: np.ndarray,
    frequency: np.ndarray,
    cap: float,
) -> np.ndarray:
    source = np.asarray(source_power, dtype=np.float64)
    target = np.asarray(target_power, dtype=np.float64)
    freq = np.asarray(frequency, dtype=np.float64)
    if source.shape != target.shape or source.shape != freq.shape or np.any(source < 0) or np.any(target < 0):
        raise ValueError("invalid frequency sufficient statistics")
    log_gain = 0.5 * (np.log(np.maximum(target, 1e-20)) - np.log(np.maximum(source, 1e-20)))
    anchor = (freq >= FREQUENCY_ANCHOR[0]) & (freq <= FREQUENCY_ANCHOR[1])
    if not np.any(anchor):
        raise ValueError("frequency grid does not cover frozen anchor")
    log_gain -= float(log_gain[anchor].mean())
    log_gain = smooth_log_curve(log_gain)
    log_gain[freq < FREQUENCY_UNITY_BELOW] = 0.0
    bound = np.log(float(cap))
    return np.exp(np.clip(log_gain, -bound, bound))


def radial_gain_map(size: int, gain: np.ndarray, frequency: np.ndarray, device, dtype):
    one_y = torch.fft.fftfreq(size, d=TARGET_MPP, device=device)
    one_x = torch.fft.rfftfreq(size, d=TARGET_MPP, device=device)
    radius = torch.sqrt(one_y[:, None] ** 2 + one_x[None, :] ** 2)
    freq = torch.as_tensor(frequency, dtype=torch.float64, device=device)
    values = torch.as_tensor(gain, dtype=torch.float64, device=device)
    positions = torch.searchsorted(freq, radius.double()).clamp(1, len(freq) - 1)
    lo = positions - 1
    hi = positions
    weight = ((radius.double() - freq[lo]) / (freq[hi] - freq[lo])).clamp(0.0, 1.0)
    mapped = values[lo] * (1.0 - weight) + values[hi] * weight
    mapped = torch.where(radius.double() > float(freq[-1]), torch.ones_like(mapped), mapped)
    mapped[0, 0] = 1.0
    return mapped.to(dtype=dtype)


def frequency_calibration(
    rgb01: torch.Tensor,
    gain: np.ndarray,
    frequency: np.ndarray,
) -> dict[str, torch.Tensor]:
    if rgb01.ndim != 4 or rgb01.shape[-1] != 3 or rgb01.shape[1] != rgb01.shape[2]:
        raise ValueError("frequency calibration requires NxHxWx3 square RGB")
    channel_first = rgb01_to_od(rgb01).permute(0, 3, 1, 2)
    size = channel_first.shape[-1]
    padding = size // 4
    padded = F.pad(channel_first, (padding, padding, padding, padding), mode="reflect")
    spatial_mean = padded.mean(dim=(-2, -1), keepdim=True)
    centered = padded - spatial_mean
    spectrum = torch.fft.rfft2(centered)
    gain_map = radial_gain_map(
        padded.shape[-1], gain, frequency, rgb01.device, rgb01.dtype
    )
    corrected = torch.fft.irfft2(spectrum * gain_map[None, None], s=padded.shape[-2:]).real
    corrected = corrected + spatial_mean
    corrected = corrected[..., padding : padding + size, padding : padding + size]
    preclip = od_to_rgb01(corrected.permute(0, 2, 3, 1))
    output = preclip.clamp(0.0, 1.0)
    return clipping_report(preclip, output)


def clipping_report(preclip: torch.Tensor, output: torch.Tensor):
    if preclip.shape != output.shape or preclip.shape[-1] != 3:
        raise ValueError("clipping report requires equal channel-last RGB tensors")
    reduce_dims = tuple(range(1, preclip.ndim))
    return {
        "preclip": preclip,
        "output": output,
        "preclip_range_fraction": ((preclip < 0.0) | (preclip > 1.0)).float().mean(dim=reduce_dims),
        "clip_pixel_mae": (output - preclip).abs().mean(dim=reduce_dims),
    }


def feature_sufficient_statistics(features: torch.Tensor):
    """Return scanner sums/Grams and five paired source-to-AT2 cross-products."""
    if features.ndim != 3 or features.shape[0] != 6:
        raise ValueError(f"expected 6xNxD features, got {tuple(features.shape)}")
    value = features.double()
    sums = value.sum(dim=1)
    grams = torch.stack([scanner.T @ scanner for scanner in value], dim=0)
    cross = torch.stack([value[index].T @ value[0] for index in range(1, 6)], dim=0)
    return sums, grams, cross, int(value.shape[1])


def centered_covariance(total: torch.Tensor, gram: torch.Tensor, count: int):
    if count <= 1:
        raise ValueError("covariance count must exceed one")
    covariance = (gram - torch.outer(total, total) / float(count)) / float(count - 1)
    return (covariance + covariance.T) * 0.5


def regularized_covariance(covariance: torch.Tensor, shrinkage: float):
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    alpha = float(shrinkage)
    if not 0.0 < alpha < 1.0:
        raise ValueError("CORAL shrinkage must lie in (0,1)")
    scale = torch.trace(covariance) / covariance.shape[0]
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("covariance has nonpositive scale")
    identity = torch.eye(covariance.shape[0], dtype=covariance.dtype, device=covariance.device)
    return (1.0 - alpha) * covariance + alpha * scale * identity


def symmetric_matrix_power(covariance: torch.Tensor, power: float):
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    scale = torch.trace(covariance) / covariance.shape[0]
    floor = 1e-7 * scale
    eigenvalues = eigenvalues.clamp_min(floor)
    return (eigenvectors * eigenvalues.pow(float(power))[None, :]) @ eigenvectors.T


def coral_transform(
    heldout: torch.Tensor,
    source_sum: torch.Tensor,
    source_gram: torch.Tensor,
    target_sum: torch.Tensor,
    target_gram: torch.Tensor,
    train_count: int,
    shrinkage: float,
):
    source_mean = source_sum / float(train_count)
    target_mean = target_sum / float(train_count)
    source_cov = regularized_covariance(
        centered_covariance(source_sum, source_gram, train_count), shrinkage
    )
    target_cov = regularized_covariance(
        centered_covariance(target_sum, target_gram, train_count), shrinkage
    )
    whitening = symmetric_matrix_power(source_cov, -0.5)
    recoloring = symmetric_matrix_power(target_cov, 0.5)
    return (heldout.double() - source_mean) @ whitening @ recoloring + target_mean


def procrustes_transform(
    heldout: torch.Tensor,
    source_sum: torch.Tensor,
    target_sum: torch.Tensor,
    cross: torch.Tensor,
    train_count: int,
):
    source_mean = source_sum / float(train_count)
    target_mean = target_sum / float(train_count)
    centered_cross = cross - torch.outer(source_sum, target_sum) / float(train_count)
    u, _, vh = torch.linalg.svd(centered_cross, full_matrices=False)
    rotation = u @ vh
    return (heldout.double() - source_mean) @ rotation + target_mean


def covariance_condition(covariance: torch.Tensor, shrinkage: float) -> float:
    regularized = regularized_covariance(covariance, shrinkage)
    eigenvalues = torch.linalg.eigvalsh(regularized)
    return float((eigenvalues[-1] / eigenvalues[0].clamp_min(1e-30)).cpu())


def uint8_from_rgb01(rgb: torch.Tensor) -> np.ndarray:
    return rgb.mul(255.0).round().clamp(0.0, 255.0).byte().cpu().numpy()
