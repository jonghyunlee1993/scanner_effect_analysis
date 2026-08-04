"""Pure operations for the E5-RF1U unpaired multi-target condition.

RF1U keeps RF1M's Laplacian bands, shared-OD residual and exact gamut
projection, but estimates each band gain from pooled population energies and
shrinks it by its own slide-bootstrap uncertainty.  Nothing here uses paired
patches, per-slide ratios or any PFM endpoint, so the fit needs only a source
population and a target population.

The executable rules are frozen in `docs/e5_rf1u_multitarget_contract.md`.
"""

from __future__ import annotations

import numpy as np

from e5_comparator_population import SCANNERS
from rf1m_combined import RF1M_ENERGY_FLOOR, RF1M_SIGMAS


RF1U_VERSION = "e5_rf1u_unpaired_band_v1"
RF1U_CONDITION = "reinhard_unpaired_band"
RF1U_TARGETS = ("at2", "gt450", "s60")
RF1U_SHRINKAGE_SE = 2.0
RF1U_BOOTSTRAP_REPLICATES = 2000
RF1U_BOOTSTRAP_SEED = 20260803
RF1U_HARD_CAP = 4.0


def target_index(target: str) -> int:
    """Scanner index of a frozen target name."""
    if target not in RF1U_TARGETS:
        raise ValueError(f"target must be one of {RF1U_TARGETS}, got {target!r}")
    return SCANNERS.index(target)


def source_indices(target: str) -> tuple[int, ...]:
    """Scanner indices corrected toward the target, in fixed scanner order."""
    index = target_index(target)
    return tuple(value for value in range(len(SCANNERS)) if value != index)


def log_population_gain(
    target_energy: np.ndarray, source_energy: np.ndarray
) -> np.ndarray:
    """Half the log ratio of pooled band energies, per band.

    Both arguments are per-slide band energies; the estimator pools them before
    dividing, so no slide is matched to another.
    """
    target = np.asarray(target_energy, dtype=np.float64)
    source = np.asarray(source_energy, dtype=np.float64)
    if target.ndim != 2 or source.shape != target.shape:
        raise ValueError("population gain needs matching slides-by-bands arrays")
    if not (np.isfinite(target).all() and np.isfinite(source).all()):
        raise ValueError("band energies must be finite")
    if np.any(target < 0) or np.any(source < 0):
        raise ValueError("band energies must be non-negative")
    return 0.5 * (
        np.log(np.maximum(target.mean(axis=0), RF1M_ENERGY_FLOOR))
        - np.log(np.maximum(source.mean(axis=0), RF1M_ENERGY_FLOOR))
    )


def bootstrap_log_gain_se(
    target_energy: np.ndarray,
    source_energy: np.ndarray,
    replicates: int = RF1U_BOOTSTRAP_REPLICATES,
    seed: int = RF1U_BOOTSTRAP_SEED,
) -> np.ndarray:
    """Slide-bootstrap standard error of the population log gain, per band."""
    target = np.asarray(target_energy, dtype=np.float64)
    source = np.asarray(source_energy, dtype=np.float64)
    if target.ndim != 2 or source.shape != target.shape or len(target) < 2:
        raise ValueError("bootstrap needs matching slides-by-bands arrays")
    if replicates < 2:
        raise ValueError("bootstrap needs at least two replicates")
    generator = np.random.default_rng(seed)
    index = generator.integers(0, len(target), size=(replicates, len(target)))
    drawn = 0.5 * (
        np.log(np.maximum(target[index].mean(axis=1), RF1M_ENERGY_FLOOR))
        - np.log(np.maximum(source[index].mean(axis=1), RF1M_ENERGY_FLOOR))
    )
    return drawn.std(axis=0, ddof=1)


def reliability_alpha(
    log_gain: np.ndarray,
    standard_error: np.ndarray,
    shrinkage_se: float = RF1U_SHRINKAGE_SE,
) -> np.ndarray:
    """Per-band shrinkage toward identity from the gain's own uncertainty."""
    gain = np.asarray(log_gain, dtype=np.float64)
    error = np.asarray(standard_error, dtype=np.float64)
    if gain.shape != error.shape:
        raise ValueError("log gain and standard error shapes differ")
    if not np.isfinite(shrinkage_se) or shrinkage_se < 0:
        raise ValueError("shrinkage multiplier must be finite and non-negative")
    if np.any(error < 0) or not np.isfinite(error).all():
        raise ValueError("standard errors must be finite and non-negative")
    magnitude = np.abs(gain)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(magnitude > 0, error / np.maximum(magnitude, 1e-300), np.inf)
    return np.clip(1.0 - float(shrinkage_se) * ratio, 0.0, 1.0)


def shrunk_band_gains(
    target_energy: np.ndarray,
    source_energy: np.ndarray,
    shrinkage_se: float = RF1U_SHRINKAGE_SE,
    replicates: int = RF1U_BOOTSTRAP_REPLICATES,
    seed: int = RF1U_BOOTSTRAP_SEED,
    hard_cap: float = RF1U_HARD_CAP,
) -> dict[str, np.ndarray]:
    """Fit one shrunk multiplicative gain per band for a single source scanner."""
    raw = log_population_gain(target_energy, source_energy)
    error = bootstrap_log_gain_se(target_energy, source_energy, replicates, seed)
    alpha = reliability_alpha(raw, error, shrinkage_se)
    if not np.isfinite(hard_cap) or hard_cap < 1.0:
        raise ValueError("hard cap must be at least one")
    limit = float(np.log(hard_cap))
    shrunk = np.clip(alpha * raw, -limit, limit)
    return {
        "raw_log_gain": raw,
        "standard_error": error,
        "alpha": alpha,
        "log_gain": shrunk,
        "gain": np.exp(shrunk),
    }


def fitted_scanner_gains(
    target_energy: np.ndarray,
    source_energy: np.ndarray,
    shrinkage_se: float = RF1U_SHRINKAGE_SE,
    replicates: int = RF1U_BOOTSTRAP_REPLICATES,
    seed: int = RF1U_BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    """Stack `shrunk_band_gains` over every source scanner.

    `target_energy` is slides-by-bands; `source_energy` is
    slides-by-sources-by-bands in the fixed source order.
    """
    source = np.asarray(source_energy, dtype=np.float64)
    if source.ndim != 3 or source.shape[2] != len(RF1M_SIGMAS):
        raise ValueError("source energy must be slides x sources x bands")
    fitted = [
        shrunk_band_gains(
            target_energy, source[:, index], shrinkage_se, replicates, seed
        )
        for index in range(source.shape[1])
    ]
    return {name: np.stack([value[name] for value in fitted], axis=0) for name in fitted[0]}
