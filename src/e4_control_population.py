"""Frozen E4 image-control definitions and pure tensor operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from prenorm.exp01.frequency import FixedLaplacianPyramid


CONTROL_VERSION = "e4_four_pfm_controls_v1"
PYRAMID_LEVELS = 4
SCANNERS = ("at2", "gt450", "versa", "akoya", "s60", "s360")
FOVS = (224, 256, 512)


@dataclass(frozen=True)
class ControlSpec:
    name: str
    family: str
    value: float
    role: str


CONTROL_SPECS = (
    ControlSpec("hf_retention_0p75", "gain", 0.75, "attenuation"),
    ControlSpec("hf_retention_0p50", "gain", 0.50, "attenuation"),
    ControlSpec("hf_retention_0p25", "gain", 0.25, "attenuation"),
    ControlSpec("hf_retention_0p00", "gain", 0.00, "complete_blur_control"),
    ControlSpec("hf_boost_1p25", "gain", 1.25, "sharpening"),
    ControlSpec("hf_boost_1p50", "gain", 1.50, "sharpening"),
    ControlSpec("hf_boost_2p00", "gain", 2.00, "sharpening"),
    ControlSpec(
        "registered_loo_hf_mean_0p25",
        "registered_loo_mix",
        0.25,
        "paired_image_space_oracle",
    ),
    ControlSpec(
        "global_train_hf_mean_1p00",
        "global_train_mix",
        1.00,
        "phase_cancellation_negative_control",
    ),
)


def condition_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in CONTROL_SPECS)


def centered_crop(array: np.ndarray, fov: int) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 5 or value.shape[0] != 6 or value.shape[2:] != (512, 512, 3):
        raise ValueError(f"expected 6×N×512×512×3 RGB, got {value.shape}")
    if fov not in FOVS:
        raise ValueError(f"FOV is outside the frozen E4 contract: {fov}")
    start = (512 - fov) // 2
    return value[:, :, start : start + fov, start : start + fov]


def uint8_to_analysis_tensor(array: np.ndarray, device=None) -> torch.Tensor:
    value = np.asarray(array)
    if value.ndim != 5 or value.shape[0] != 6 or value.shape[-1] != 3:
        raise ValueError(f"expected 6×N×H×W×3 RGB, got {value.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(value)).to(device=device)
    return tensor.permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)


def decompose_scanner_tensor(
    pyramid: FixedLaplacianPyramid,
    images: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if images.ndim != 5 or images.shape[0] != 6 or images.shape[2] != 3:
        raise ValueError(f"expected 6×N×3×H×W tensor, got {tuple(images.shape)}")
    scanners, locations = images.shape[:2]
    flat = images.reshape(scanners * locations, *images.shape[2:])
    coarse, flat_bands = pyramid.decompose(flat)
    coarse = coarse.reshape(scanners, locations, *coarse.shape[1:])
    bands = [
        band.reshape(scanners, locations, *band.shape[1:])
        for band in flat_bands
    ]
    return coarse, bands


def registered_loo_bands(bands: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    output = []
    for band in bands:
        if band.ndim != 5 or band.shape[0] != 6:
            raise ValueError("registered bands must be 6×N×C×H×W")
        output.append((band.sum(dim=0, keepdim=True) - band) / 5.0)
    return output


def slide_band_sums(bands: Sequence[torch.Tensor]) -> list[np.ndarray]:
    return [
        band.double().sum(dim=(0, 1)).cpu().numpy()
        for band in bands
    ]


def heldout_global_mean_bands(
    total_band_sums: Sequence[np.ndarray],
    heldout_band_sums: Sequence[np.ndarray],
    total_image_count: int,
    heldout_image_count: int,
    device,
) -> list[torch.Tensor]:
    if len(total_band_sums) != PYRAMID_LEVELS or len(heldout_band_sums) != PYRAMID_LEVELS:
        raise ValueError("global reference must contain four pyramid bands")
    train_count = int(total_image_count) - int(heldout_image_count)
    if train_count <= 0:
        raise ValueError("held-out reference has no training images")
    means = []
    for total, heldout in zip(total_band_sums, heldout_band_sums):
        if np.asarray(total).shape != np.asarray(heldout).shape:
            raise ValueError("total and held-out band shapes differ")
        mean = (np.asarray(total, dtype=np.float64) - np.asarray(heldout, dtype=np.float64)) / train_count
        means.append(torch.from_numpy(mean.astype(np.float32)).to(device=device))
    return means


def requested_bands(
    spec: ControlSpec,
    source_bands: Sequence[torch.Tensor],
    registered_bands: Sequence[torch.Tensor],
    global_bands: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    if not (
        len(source_bands)
        == len(registered_bands)
        == len(global_bands)
        == PYRAMID_LEVELS
    ):
        raise ValueError("E4 intervention requires four source/reference bands")
    if spec.family == "gain":
        return [spec.value * band for band in source_bands]
    if spec.family == "registered_loo_mix":
        return [
            (1.0 - spec.value) * source + spec.value * reference
            for source, reference in zip(source_bands, registered_bands)
        ]
    if spec.family == "global_train_mix":
        return [
            (1.0 - spec.value) * source
            + spec.value * reference[None, None].to(source)
            for source, reference in zip(source_bands, global_bands)
        ]
    raise ValueError(f"unknown E4 control family: {spec.family}")


def render_control(
    pyramid: FixedLaplacianPyramid,
    coarse: torch.Tensor,
    bands: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    scanners, locations = coarse.shape[:2]
    flat_coarse = coarse.reshape(scanners * locations, *coarse.shape[2:])
    flat_bands = [
        band.reshape(scanners * locations, *band.shape[2:])
        for band in bands
    ]
    preclip = pyramid.reconstruct(flat_coarse, flat_bands).reshape(
        scanners, locations, 3, *bands[0].shape[-2:]
    )
    image = preclip.clamp(-1.0, 1.0)
    clipped = (preclip < -1.0) | (preclip > 1.0)
    return {
        "preclip": preclip,
        "image": image,
        "preclip_range_fraction": clipped.float().mean(dim=(2, 3, 4)),
        "clip_pixel_mae": (image - preclip).abs().mean(dim=(2, 3, 4)),
    }


def detail_rms(
    pyramid: FixedLaplacianPyramid,
    coarse: torch.Tensor,
    bands: Sequence[torch.Tensor],
) -> torch.Tensor:
    scanners, locations = coarse.shape[:2]
    flat_coarse = coarse.reshape(scanners * locations, *coarse.shape[2:])
    flat_bands = [
        band.reshape(scanners * locations, *band.shape[2:])
        for band in bands
    ]
    detail = pyramid.reconstruct(torch.zeros_like(flat_coarse), flat_bands)
    return detail.square().mean(dim=(1, 2, 3)).sqrt().reshape(scanners, locations)


def operational_detail_rms(
    pyramid: FixedLaplacianPyramid,
    images: torch.Tensor,
) -> torch.Tensor:
    scanners, locations = images.shape[:2]
    flat = images.reshape(scanners * locations, *images.shape[2:])
    detail = pyramid.copied_detail(flat)
    return detail.square().mean(dim=(1, 2, 3)).sqrt().reshape(scanners, locations)


def analysis_tensor_to_uint8(images: torch.Tensor) -> np.ndarray:
    if images.ndim != 5 or images.shape[0] != 6 or images.shape[2] != 3:
        raise ValueError(f"expected 6×N×3×H×W tensor, got {tuple(images.shape)}")
    return (
        images.permute(0, 1, 3, 4, 2)
        .add(1.0)
        .mul(127.5)
        .round()
        .clamp(0, 255)
        .byte()
        .cpu()
        .numpy()
    )
