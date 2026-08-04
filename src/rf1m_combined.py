"""Pure operations for the E5-RF1M combined multiscale/no-harm candidate.

RF1M keeps the RF1 Reinhard base and the exact shared-OD gamut projection, but
replaces the 72-bin radial gain with three native Laplacian bands whose gains are
fitted from band energy accumulated directly on training patches, and replaces
the panel-wide cap with a strict-nested scanner-specific cap that always admits
exact identity.

The executable rules are frozen in `docs/e5_rf1m_combined_candidate_contract.md`.
No function here reads a PFM feature or any outcome.
"""

from __future__ import annotations

import numpy as np
import torch

from analyze_rf1_multiscale import PYRAMID_SIGMAS, laplacian_pyramid
from e5_comparator_population import SCANNERS, od_to_rgb01, rgb01_to_od
from e5_reinhard_residual_frequency import (
    MATERIAL_EXCURSION,
    OD_RGB8_MAX,
    RF1_GAIN_CAP_CANDIDATES,
)


RF1M_VERSION = "e5_rf1m_multiscale_noharm_v1"
RF1M_CONDITION = "reinhard_multiscale_noharm"
RF1M_SIGMAS = PYRAMID_SIGMAS
RF1M_IDENTITY_CAP = 1.0
RF1M_CAP_CANDIDATES = (RF1M_IDENTITY_CAP, *RF1_GAIN_CAP_CANDIDATES)
RF1M_ENERGY_FLOOR = 1e-20
# Amendment 1 of the RF1M contract: two standard errors, not one.  With k=1 the
# S360 outer fold 4 selection returned the strongest eligible cap and worsened
# the held-out spectrum by 34%.
RF1M_SHRINKAGE_SE = 2.0
RF1M_SPECTRUM_TOLERANCE = 1e-12


def band_energy(
    mean_od: torch.Tensor, sigmas: tuple[float, ...] = RF1M_SIGMAS
) -> torch.Tensor:
    """Per-patch mean square of each Laplacian band of a mean-OD batch.

    This is the native accumulation the contract requires: it measures energy in
    the same mean-OD domain in which the correction is applied, rather than
    approximating it from locked radial RGB statistics.
    """
    if mean_od.ndim != 3:
        raise ValueError("band energy requires a BxHxW mean-OD batch")
    if not torch.isfinite(mean_od).all():
        raise ValueError("band energy input contains non-finite values")
    bands, _ = laplacian_pyramid(mean_od, sigmas)
    return torch.stack([band.square().mean(dim=(1, 2)) for band in bands], dim=-1)


def accumulate_band_energy(
    rgb01: torch.Tensor, sigmas: tuple[float, ...] = RF1M_SIGMAS
) -> tuple[np.ndarray, int]:
    """Sum per-patch band energy over an RGB batch and report the patch count."""
    if rgb01.ndim != 4 or rgb01.shape[-1] != 3 or rgb01.shape[1] != rgb01.shape[2]:
        raise ValueError("band-energy accumulation requires NxHxWx3 square RGB")
    energy = band_energy(rgb01_to_od(rgb01).mean(dim=-1), sigmas)
    return energy.sum(dim=0).double().cpu().numpy(), int(rgb01.shape[0])


def fitted_band_gains(
    source_energy: np.ndarray,
    target_energy: np.ndarray,
    cap: float,
    sigmas: tuple[float, ...] = RF1M_SIGMAS,
) -> np.ndarray:
    """Fit one capped energy-ratio gain per source scanner and Laplacian band."""
    source = np.asarray(source_energy, dtype=np.float64)
    target = np.asarray(target_energy, dtype=np.float64)
    if source.shape != (len(SCANNERS) - 1, len(sigmas)):
        raise ValueError(f"invalid source band-energy shape: {source.shape}")
    if target.shape != (len(sigmas),):
        raise ValueError(f"invalid target band-energy shape: {target.shape}")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("band energies must be finite")
    if np.any(source < 0) or np.any(target < 0):
        raise ValueError("band energies must be non-negative")
    if not np.isfinite(cap) or cap < RF1M_IDENTITY_CAP:
        raise ValueError("gain cap must be at least one")
    ratio = np.sqrt(
        np.maximum(target, RF1M_ENERGY_FLOOR)[None, :]
        / np.maximum(source, RF1M_ENERGY_FLOOR)
    )
    return np.clip(ratio, 1.0 / float(cap), float(cap))


def candidate_band_gains(
    source_energy: np.ndarray,
    target_energy: np.ndarray,
    caps=RF1M_CAP_CANDIDATES,
    sigmas: tuple[float, ...] = RF1M_SIGMAS,
) -> np.ndarray:
    """Stack the fitted gains of every candidate cap as CxSxB."""
    values = np.asarray(caps, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.any(np.diff(values) <= 0):
        raise ValueError("candidate caps must be strictly ascending")
    if not np.isclose(values[0], RF1M_IDENTITY_CAP):
        raise ValueError("the first candidate cap must be exact identity")
    return np.stack(
        [
            fitted_band_gains(source_energy, target_energy, float(cap), sigmas)
            for cap in values
        ],
        axis=0,
    )


def choose_candidate_k_se(
    candidate_caps: np.ndarray,
    fold_base_rmse: np.ndarray,
    fold_candidate_rmse: np.ndarray,
    fold_gamut_pass: np.ndarray,
    shrinkage_se: float = RF1M_SHRINKAGE_SE,
    tolerance: float = RF1M_SPECTRUM_TOLERANCE,
) -> tuple[int, np.ndarray, np.ndarray, int, float, float, np.ndarray]:
    """Select the weakest eligible candidate within `shrinkage_se` SE of the best.

    This is the RF1 no-harm rule with the shrinkage multiplier exposed.  The
    locked branch-1 implementation uses one standard error and is left untouched;
    RF1M uses two under Amendment 1 because a single SE did not shrink far enough
    for the scanner with the largest between-slide response variance.

    Eligibility, the delta definition, the SE formula and the tie rule toward
    identity are identical to the one-SE rule.
    """
    caps = np.asarray(candidate_caps, dtype=np.float64)
    base = np.asarray(fold_base_rmse, dtype=np.float64)
    candidate = np.asarray(fold_candidate_rmse, dtype=np.float64)
    gamut = np.asarray(fold_gamut_pass, dtype=bool)
    if (
        caps.ndim != 1
        or len(caps) == 0
        or not np.isclose(caps[0], RF1M_IDENTITY_CAP)
        or np.any(np.diff(caps) <= 0)
        or base.ndim != 1
        or len(base) < 2
        or candidate.shape != (len(base), len(caps))
        or gamut.shape != candidate.shape
        or not np.isfinite(caps).all()
        or not np.isfinite(base).all()
        or not np.isfinite(candidate).all()
    ):
        raise ValueError("invalid k-SE candidate arrays")
    if not np.isfinite(shrinkage_se) or shrinkage_se < 0:
        raise ValueError("shrinkage multiplier must be finite and non-negative")
    delta = candidate - base[:, None]
    mean_delta = delta.mean(axis=0)
    nonworse = np.all(delta <= float(tolerance), axis=0)
    eligible = np.all(gamut, axis=0) & nonworse
    if not eligible[0]:
        raise RuntimeError("exact identity must pass the k-SE no-harm gate")
    eligible_indices = np.flatnonzero(eligible)
    best = int(eligible_indices[np.argmin(mean_delta[eligible_indices])])
    best_se = float(delta[:, best].std(ddof=1) / np.sqrt(len(base)))
    threshold = float(mean_delta[best] + float(shrinkage_se) * best_se)
    within = eligible & (mean_delta <= threshold + float(tolerance))
    selected = int(np.flatnonzero(within)[0])
    return selected, eligible, mean_delta, best, best_se, threshold, within


def shared_od_multiscale_many(
    rgb01: torch.Tensor,
    gains: np.ndarray,
    sigmas: tuple[float, ...] = RF1M_SIGMAS,
) -> dict[str, torch.Tensor]:
    """Render a candidate-gain grid through one shared, gamut-safe OD residual.

    The Laplacian pyramid is built once and reused for every candidate, mirroring
    `shared_od_residual_frequency_many` for the radial RF1 audit.  Returned
    tensors have candidate as the first dimension and batch as the second.
    """
    if rgb01.ndim != 4 or rgb01.shape[-1] != 3 or rgb01.shape[1] != rgb01.shape[2]:
        raise ValueError("RF1M correction requires NxHxWx3 square RGB")
    if not torch.isfinite(rgb01).all():
        raise ValueError("RF1M input contains non-finite values")
    gain_values = np.asarray(gains, dtype=np.float64)
    if gain_values.ndim != 2 or gain_values.shape[1] != len(sigmas):
        raise ValueError("candidate gains must have shape CxB")
    if not np.isfinite(gain_values).all() or np.any(gain_values <= 0):
        raise ValueError("candidate gains must be positive and finite")

    od = rgb01_to_od(rgb01)
    mean_od = od.mean(dim=-1)
    bands, base = laplacian_pyramid(mean_od, sigmas)
    stacked = torch.stack(bands, dim=0)
    gain = torch.as_tensor(gain_values, dtype=rgb01.dtype, device=rgb01.device)
    corrected = base[None] + torch.einsum("cb,bnhw->cnhw", gain, stacked)
    proposed_delta = corrected - mean_od[None]

    lower = -od.amin(dim=-1)
    upper = OD_RGB8_MAX - od.amax(dim=-1)
    projected_delta = torch.minimum(
        torch.maximum(proposed_delta, lower[None]), upper[None]
    )
    proposed_rgb = od_to_rgb01(od[None] + proposed_delta[..., None])
    projected_rgb = od_to_rgb01(od[None] + projected_delta[..., None])
    output = projected_rgb.clamp(0.0, 1.0)

    out_of_range = (proposed_rgb < 0.0) | (proposed_rgb > 1.0)
    material = (proposed_rgb < -MATERIAL_EXCURSION) | (
        proposed_rgb > 1.0 + MATERIAL_EXCURSION
    )
    projection = (projected_delta - proposed_delta).abs()
    metric_dims = (2, 3, 4)
    pixel_dims = (2, 3)
    return {
        "output": output,
        "proposed_delta": proposed_delta,
        "projected_delta": projected_delta,
        "preproject_range_fraction": out_of_range.float().mean(dim=metric_dims),
        "material_range_fraction": material.float().mean(dim=metric_dims),
        "projection_fraction": (projection > 1e-7).float().mean(dim=pixel_dims),
        "projection_rgb_mae": (projected_rgb - proposed_rgb).abs().mean(dim=metric_dims),
        "final_clamp_mae": (output - projected_rgb).abs().mean(dim=metric_dims),
    }
