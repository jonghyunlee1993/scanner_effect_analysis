"""Deterministic stain normalization for manuscript-completion Panel A.

The Macenko implementation is ported from git commit ``3085cfd``.  The
Vahadane comparator uses deterministic sparse NMF only to estimate the stain
basis; concentrations are reconstructed with a non-negative clipped least
squares projection.  Both methods expose explicit fallbacks instead of
silently returning corrupt images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.decomposition import NMF


PUBLISHED_HE = np.asarray(
    [[0.650, 0.072], [0.704, 0.990], [0.286, 0.105]], dtype=np.float64
)
PUBLISHED_HE /= np.linalg.norm(PUBLISHED_HE, axis=0, keepdims=True)

MACENKO_VERSION = "panel_a_macenko_3085cfd_v1"
VAHADANE_VERSION = "panel_a_vahadane_sparse_nmf_v1"
OD_BETA = 0.15
ANGLE_PERCENTILES = (1.0, 99.0)
CONCENTRATION_PERCENTILE = 99.0
MIN_TISSUE_PIXELS = 50


@dataclass(frozen=True)
class StainParameters:
    basis: np.ndarray
    maximum: np.ndarray
    method: str
    converged: bool = True
    iterations: int = 0
    reconstruction_error: float = 0.0


@dataclass(frozen=True)
class NormalizationResult:
    image: np.ndarray
    fallback: bool
    clipped_fraction: float
    source_parameters: StainParameters | None


def optical_density(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB values to optical density as in commit 3085cfd."""

    value = np.asarray(rgb)
    if value.shape[-1] != 3:
        raise ValueError(f"expected RGB last dimension, got {value.shape}")
    return -np.log((value.astype(np.float64) + 1.0) / 256.0)


def od_to_rgb(od: np.ndarray) -> tuple[np.ndarray, float]:
    preclip = np.exp(-np.asarray(od, dtype=np.float64)) * 256.0 - 1.0
    clipped = np.clip(preclip, 0.0, 255.0)
    fraction = float(np.mean((preclip < 0.0) | (preclip > 255.0)))
    return np.rint(clipped).astype(np.uint8), fraction


def tissue_od(rgb: np.ndarray, beta: float = OD_BETA) -> np.ndarray:
    od = optical_density(np.asarray(rgb).reshape(-1, 3))
    return od[np.isfinite(od).all(axis=1) & (od.sum(axis=1) > float(beta))]


def _order_he(basis: np.ndarray) -> np.ndarray:
    value = np.asarray(basis, dtype=np.float64).copy()
    value *= np.where(value.sum(axis=0, keepdims=True) < 0.0, -1.0, 1.0)
    value /= np.maximum(np.linalg.norm(value, axis=0, keepdims=True), 1e-12)
    direct = float(np.sum(value * PUBLISHED_HE))
    swapped = float(np.sum(value[:, ::-1] * PUBLISHED_HE))
    return value if direct >= swapped else value[:, ::-1]


def _concentrations(od: np.ndarray, basis: np.ndarray) -> np.ndarray:
    values = np.linalg.lstsq(np.asarray(basis), np.asarray(od).T, rcond=None)[0].T
    return np.maximum(values, 0.0)


def _parameters(
    basis: np.ndarray,
    od: np.ndarray,
    method: str,
    *,
    converged: bool = True,
    iterations: int = 0,
    reconstruction_error: float = 0.0,
) -> StainParameters:
    ordered = _order_he(basis)
    concentration = _concentrations(od, ordered)
    maximum = np.percentile(
        concentration, CONCENTRATION_PERCENTILE, axis=0
    ).clip(1e-4)
    if not np.isfinite(ordered).all() or not np.isfinite(maximum).all():
        raise ValueError("non-finite stain parameters")
    condition = float(np.linalg.cond(ordered))
    if np.linalg.matrix_rank(ordered) != 2 or not np.isfinite(condition) or condition >= 100.0:
        raise ValueError(f"degenerate stain basis (condition={condition})")
    return StainParameters(
        ordered.astype(np.float32), maximum.astype(np.float32), method,
        bool(converged), int(iterations), float(reconstruction_error),
    )


def fit_macenko_od(od: np.ndarray) -> StainParameters | None:
    """Fit Macenko from an already tissue-filtered OD pixel matrix."""

    od = np.asarray(od, dtype=np.float64).reshape(-1, 3)
    if len(od) < MIN_TISSUE_PIXELS:
        return None
    try:
        centered = od - od.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / float(len(centered) - 1)
        _, vectors = np.linalg.eigh(covariance)
        plane = vectors[:, -2:]
        projected = od @ plane
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        low, high = np.percentile(angles, ANGLE_PERCENTILES)
        basis = np.stack(
            [
                plane @ np.asarray([np.cos(low), np.sin(low)]),
                plane @ np.asarray([np.cos(high), np.sin(high)]),
            ],
            axis=1,
        )
        return _parameters(basis, od, "macenko")
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None


def fit_macenko(rgb: np.ndarray) -> StainParameters | None:
    """Fit the beta=.15, angular 1/99, q99 Macenko contract."""

    return fit_macenko_od(tissue_od(rgb))


def fit_vahadane_od(
    od: np.ndarray,
    *,
    seed: int = 20260917,
    max_pixels: int = 4096,
    alpha: float = 0.001,
    max_iter: int = 1000,
    tolerance: float = 1e-3,
    solver: str = "cd",
) -> StainParameters | None:
    """Fit a two-component deterministic sparse-NMF stain basis.

    Pixels are selected by a deterministic evenly spaced subsample, avoiding
    any dependence on worker scheduling. ``alpha`` and ``l1_ratio=1`` define
    the locked sparse-NMF regularization contract. Sparsity is applied to W
    (pixel concentrations) only; penalizing the two-row H stain basis can
    collapse a component at this OD scale.
    """

    od = np.asarray(od, dtype=np.float64).reshape(-1, 3)
    if len(od) < MIN_TISSUE_PIXELS:
        return None
    if len(od) > int(max_pixels):
        # A seed-dependent cyclic offset retains determinism while preventing
        # an implicit preference for the upper-left image region.
        offset = int(seed) % len(od)
        positions = (offset + np.linspace(0, len(od) - 1, int(max_pixels), dtype=int)) % len(od)
        fitted = od[positions]
    else:
        fitted = od
    try:
        solver = str(solver).lower()
        if solver not in {"cd", "mu"}:
            raise ValueError(f"unsupported NMF solver {solver}")
        model = NMF(
            n_components=2,
            init="nndsvda" if solver == "cd" else "nndsvdar",
            solver=solver,
            beta_loss="frobenius",
            tol=float(tolerance),
            max_iter=int(max_iter),
            random_state=int(seed),
            alpha_W=float(alpha),
            alpha_H=0.0,
            l1_ratio=1.0,
            shuffle=False,
        )
        model.fit(np.maximum(fitted, 0.0))
        basis = np.asarray(model.components_, dtype=np.float64).T
        return _parameters(
            basis, od, "vahadane",
            converged=int(model.n_iter_) < int(max_iter),
            iterations=int(model.n_iter_),
            reconstruction_error=float(model.reconstruction_err_),
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None


def fit_vahadane(
    rgb: np.ndarray,
    *,
    seed: int = 20260917,
    max_pixels: int = 4096,
    alpha: float = 0.001,
    max_iter: int = 1000,
    tolerance: float = 1e-3,
    solver: str = "cd",
) -> StainParameters | None:
    return fit_vahadane_od(
        tissue_od(rgb),
        seed=seed,
        max_pixels=max_pixels,
        alpha=alpha,
        max_iter=max_iter,
        tolerance=tolerance,
        solver=solver,
    )


def aggregate_reference(
    parameters: list[StainParameters | None], method: str
) -> StainParameters:
    valid = [item for item in parameters if item is not None]
    if not valid:
        raise ValueError("no valid stain references")
    if any(item.method != method for item in valid):
        raise ValueError("reference parameters mix stain methods")
    bases = np.stack([item.basis for item in valid]).astype(np.float64)
    maxima = np.stack([item.maximum for item in valid]).astype(np.float64)
    basis = bases.mean(axis=0)
    basis /= np.maximum(np.linalg.norm(basis, axis=0, keepdims=True), 1e-12)
    maximum = np.exp(np.log(np.maximum(maxima, 1e-8)).mean(axis=0))
    return StainParameters(basis.astype(np.float32), maximum.astype(np.float32), method)


def normalize(
    rgb: np.ndarray,
    target: StainParameters,
    *,
    method: str,
    seed: int = 20260917,
    vahadane_config: Mapping[str, Any] | None = None,
) -> NormalizationResult:
    image = np.asarray(rgb, dtype=np.uint8)
    source = fit_macenko(image) if method == "macenko" else fit_vahadane(image, seed=seed, **dict(vahadane_config or {}))
    if method not in {"macenko", "vahadane"}:
        raise ValueError(f"unknown stain method {method!r}")
    if target.method != method:
        raise ValueError("target stain method does not match requested method")
    if source is None:
        return NormalizationResult(image.copy(), True, 0.0, None)
    try:
        od = optical_density(image.reshape(-1, 3))
        concentration = _concentrations(od, source.basis)
        scaled = concentration * (
            target.maximum[None] / np.maximum(source.maximum[None], 1e-4)
        )
        corrected_od = scaled @ target.basis.T
        corrected, clipped = od_to_rgb(corrected_od.reshape(image.shape))
        if not np.isfinite(corrected).all():
            raise ValueError("non-finite normalized image")
        return NormalizationResult(corrected, False, clipped, source)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return NormalizationResult(image.copy(), True, 0.0, source)


def parameters_to_json(value: StainParameters) -> dict[str, Any]:
    return {
        "method": value.method,
        "basis": np.asarray(value.basis, dtype=float).tolist(),
        "maximum": np.asarray(value.maximum, dtype=float).tolist(),
        "converged": bool(value.converged),
        "iterations": int(value.iterations),
        "reconstruction_error": float(value.reconstruction_error),
    }


def parameters_from_json(value: dict[str, Any]) -> StainParameters:
    return StainParameters(
        np.asarray(value["basis"], dtype=np.float32),
        np.asarray(value["maximum"], dtype=np.float32),
        str(value["method"]),
        bool(value.get("converged", True)),
        int(value.get("iterations", 0)),
        float(value.get("reconstruction_error", 0.0)),
    )
