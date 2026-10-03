"""Feature-space correction and tissue retrieval used by the final paper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, lstsq, solve


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def paired_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1.0 - np.sum(unit(x) * unit(y), axis=-1)


def fit_affine(x: np.ndarray, y: np.ndarray, relative_alpha: float):
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    yc = y - y_mean
    gram = xc.T @ xc
    scale = np.trace(gram) / gram.shape[0]
    gram.flat[:: gram.shape[0] + 1] += relative_alpha * scale
    try:
        weight = solve(gram, xc.T @ yc, assume_a="pos", check_finite=False, overwrite_a=True)
    except LinAlgError:
        if relative_alpha != 0:
            raise
        weight = lstsq(xc, yc, check_finite=False, lapack_driver="gelsy")[0]
    return x_mean, y_mean, weight


def apply_affine(model, x: np.ndarray) -> np.ndarray:
    x_mean, y_mean, weight = model
    return (x - x_mean) @ weight + y_mean


def fit_procrustes(x: np.ndarray, y: np.ndarray):
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    yc = y - y_mean
    u, _, vt = np.linalg.svd(xc.T @ yc, full_matrices=False)
    return x_mean, y_mean, u @ vt


def fit_combat(x: np.ndarray, y: np.ndarray, test: np.ndarray) -> np.ndarray:
    from neuroCombat import neuroCombat, neuroCombatFromTraining

    # The installed neuroCombat version still uses the removed NumPy alias.
    if not hasattr(np, "int"):
        np.int = int
    data = np.concatenate([x, y], axis=0).T
    covars = pd.DataFrame({"batch": ["source"] * len(x) + ["target"] * len(y)})
    estimates = neuroCombat(data, covars, "batch", ref_batch="target")["estimates"]
    # This implementation encodes sorted training batch labels as integers (source=0).
    return neuroCombatFromTraining(test.T, np.zeros(len(test), dtype=int), estimates)[
        "data"
    ].T


def macro_retrieval(query: np.ndarray, target: np.ndarray, tissues: np.ndarray) -> float:
    scores = unit(query) @ unit(target).T
    np.fill_diagonal(scores, -np.inf)
    predicted = tissues[np.argmax(scores, axis=1)]
    counts = pd.DataFrame({"tissue": tissues, "correct": predicted == tissues}).groupby(
        "tissue", sort=True
    )["correct"].agg(["mean", "count"])
    return float(counts.loc[counts["count"] > 1, "mean"].mean())
