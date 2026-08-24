"""CORAL and orthogonal Procrustes on the PLISM core-grid embeddings.

This replaces `analyze_plism_feature_correction.py`, which ran on the sparse
sampling -- 192-384 locations per section, four E0-panel encoders -- and was
removed in the 2026-08-24 cleanup; see `docs/superseded_records_20260824.md` for
where to read it.  Two things changed and both matter to the conclusion.

**The panel.**  UNI2-h, CONCHv1.5 and H-optimus-1 -- the encoders a 2026 study
would actually deploy -- in place of ResNet50, UNI v1, CONCH v1 and Virchow2.

**The sample count.**  ~8,950 locations per section against a few hundred, which
is the whole reason for re-running this.  The report carries a qualification
against CORAL: it failed on PLISM for the two highest-dimensional encoders, and
the diagnosis offered was estimation rather than method -- a full covariance of
dimension 2560 fitted from 192-384 samples.  That was an inference, not a
measurement, because the sparse cohort had no way to add samples.  It does now,
so `--sweep` refits CORAL at growing training sizes and reports where the failure
stops.  If the diagnosis was right the curve crosses zero and flattens; if CORAL
is simply the wrong map for these embeddings it stays negative at every size.

**Fitting uses every location; evaluation uses a frozen subsample.**  Three of the
four locked endpoints are location-by-location matrices and so cost O(L^2), while
the estimate stops moving after a few hundred locations.  The subsample is
`build_plism_core_eval_locations.py`'s, shared with the image-condition arm so
the two are paired on identical pixels.

Both fits are accumulated as sufficient statistics per (section, scanner) --
count, sum, second moment, and the cross moment against the destination -- so a
leave-one-section-out split is a subtraction rather than a re-read.  The result is
identical to concatenating the training rows, and `--self-check` verifies that
against a direct fit before any PLISM number is read.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from analyze_e4_control_frontier import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COLLAPSE_POINT_THRESHOLD,
    CONTENT_NONINFERIORITY_MARGIN,
)
from analyze_plism_pfm_frontier import (
    centroid_rms,
    collapse_values,
    self_check,
    unmatched_quantile,
)
from analyze_rf1u_scanner_probe import probe_balanced_accuracy
from e4_primary_metrics import COLLAPSE_METRICS, l2_normalize
from e5_reinhard_residual_frequency import RF1_FOLDS

REFERENCE = "AT2"
RIDGE = 1e-6
# The locked PanNormal probe reads every fourth location; the same stride is kept
# here so the two probe numbers are produced by the same amount of data per unit.
PROBE_STRIDE = 4
SWEEP_SIZES = (250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/PLISM_dataset/features")
    parser.add_argument("--eval-locations", default="outputs/plism_core_eval_locations")
    parser.add_argument("--output", default="outputs/plism_core_feature_correction")
    parser.add_argument("--destination", default=REFERENCE)
    parser.add_argument("--encoders", default="", help="comma separated; default is all")
    parser.add_argument("--sweep", action="store_true", default=True)
    parser.add_argument("--no-sweep", dest="sweep", action="store_false")
    return parser.parse_args()


# --------------------------------------------------------------------- loading
def load_encoder(directory: Path) -> dict:
    """stain -> scanner -> (location ids sorted, features in that order)."""
    store: dict = {}
    for path in sorted(directory.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            meta = json.loads(handle.attrs["meta"])
            location = handle["location"][:]
            order = np.argsort(location)
            store.setdefault(meta["stain"], {})[meta["scanner"]] = (
                location[order], handle["features"][:][order])
    return store


def align(section: dict, scanners: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(scanners, locations, dim) on the location ids common to every scanner."""
    shared = None
    for scanner in scanners:
        ids = section[scanner][0]
        shared = ids if shared is None else np.intersect1d(shared, ids)
    stack = [section[s][1][np.isin(section[s][0], shared)] for s in scanners]
    return np.stack(stack), shared


# ---------------------------------------------------------------- accumulation
def target_moments(target: np.ndarray) -> dict:
    """The destination's own statistics, shared by every scanner of a section."""
    target = np.asarray(target, dtype=np.float64)
    return {"n": len(target), "sum_t": target.sum(axis=0), "tt": target.T @ target}


def source_moments(source: np.ndarray, target: np.ndarray) -> dict:
    """One scanner's statistics against the destination, on paired rows."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return {"sum_s": source.sum(axis=0), "ss": source.T @ source,
            "st": source.T @ target}


def combine(parts: list[tuple[dict, dict]]) -> dict:
    """Sum (target, source) statistics over sections into one fitting problem."""
    out = {"n": sum(t["n"] for t, _ in parts)}
    for key in ("sum_t", "tt"):
        out[key] = sum(t[key] for t, _ in parts)
    for key in ("sum_s", "ss", "st"):
        out[key] = sum(s[key] for _, s in parts)
    return out


def covariance(second: np.ndarray, total: np.ndarray, n: int) -> np.ndarray:
    """Unbiased covariance from a second moment and a sum, matching np.cov."""
    mean = total / n
    return (second - n * np.outer(mean, mean)) / (n - 1)


def matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 1e-12, None)
    return (vectors * values**power) @ vectors.T


def coral_from(stats: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimension = len(stats["sum_s"])
    eye = RIDGE * np.eye(dimension)
    source_cov = covariance(stats["ss"], stats["sum_s"], stats["n"]) + eye
    target_cov = covariance(stats["tt"], stats["sum_t"], stats["n"]) + eye
    matrix = matrix_power(source_cov, -0.5) @ matrix_power(target_cov, 0.5)
    return stats["sum_s"] / stats["n"], stats["sum_t"] / stats["n"], matrix


def procrustes_from(stats: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_mean = stats["sum_s"] / stats["n"]
    target_mean = stats["sum_t"] / stats["n"]
    cross = stats["st"] - stats["n"] * np.outer(source_mean, target_mean)
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    return source_mean, target_mean, u @ vt


FITS = {"coral": coral_from, "procrustes": procrustes_from}


def accumulator_check() -> dict:
    """Accumulated fits against a direct fit on the concatenated rows."""
    rng = np.random.default_rng(11)
    blocks = [(rng.normal(size=(40, 7)), rng.normal(size=(40, 7))) for _ in range(3)]
    stats = combine([(target_moments(t), source_moments(s, t)) for s, t in blocks])
    source = np.concatenate([b[0] for b in blocks])
    target = np.concatenate([b[1] for b in blocks])

    direct_cov = np.cov(source - source.mean(axis=0), rowvar=False) + RIDGE * np.eye(7)
    accumulated_cov = covariance(stats["ss"], stats["sum_s"], stats["n"]) + RIDGE * np.eye(7)
    direct_cross = (source - source.mean(axis=0)).T @ (target - target.mean(axis=0))
    accumulated_cross = stats["st"] - stats["n"] * np.outer(
        stats["sum_s"] / stats["n"], stats["sum_t"] / stats["n"])
    return {"covariance": float(np.abs(direct_cov - accumulated_cov).max()),
            "cross_moment": float(np.abs(direct_cross - accumulated_cross).max()),
            "mean": float(np.abs(source.mean(axis=0) - stats["sum_s"] / stats["n"]).max())}


# ------------------------------------------------------------------- endpoints
def endpoints(stack: np.ndarray, raw: np.ndarray, anchor: np.ndarray,
              scanners: list[str], raw_cache: dict) -> list[dict]:
    threshold = unmatched_quantile(anchor)
    anchor_unit = l2_normalize(anchor)
    radius, raw_radius = centroid_rms(stack), centroid_rms(raw)
    rows = []
    for index, scanner in enumerate(scanners):
        margin = (l2_normalize(stack[index]) * anchor_unit).sum(axis=1) - threshold
        raw_margin = (l2_normalize(raw[index]) * anchor_unit).sum(axis=1) - threshold
        collapse = collapse_values(stack[index])
        if scanner not in raw_cache:
            raw_cache[scanner] = collapse_values(raw[index])
        base = raw_cache[scanner]
        rows.append({"scanner": scanner, "radius": radius, "raw_radius": raw_radius,
                     "content_delta": float(margin.mean() - raw_margin.mean()),
                     **{f"collapse_{m}": collapse[m] / base[m] if base[m] > 0 else np.nan
                        for m in COLLAPSE_METRICS}})
    return rows


def bootstrap(frame: pd.DataFrame, key: list[str], probe: dict) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = []
    for values, block in frame.groupby(key):
        values = values if isinstance(values, tuple) else (values,)
        sections = sorted(block["section"].unique())
        draws = rng.integers(0, len(sections), size=(BOOTSTRAP_REPLICATES, len(sections)))
        by_section = block.groupby("section")
        radius = np.array([by_section.get_group(s)["radius"].iloc[0] for s in sections])
        raw_radius = np.array([by_section.get_group(s)["raw_radius"].iloc[0] for s in sections])
        reduction = 1.0 - radius / raw_radius
        samples = reduction[draws].mean(axis=1)
        delta = block.groupby("section")["content_delta"].mean().reindex(sections).to_numpy()
        delta_samples = delta[draws].mean(axis=1)
        by_scanner = block.groupby("scanner")[[f"collapse_{m}" for m in COLLAPSE_METRICS]].mean()
        worst = {m: float(by_scanner[f"collapse_{m}"].min()) for m in COLLAPSE_METRICS}
        entry = dict(zip(key, values))
        entry.update({
            "rr": float(reduction.mean()),
            "rr_ci_low": float(np.percentile(samples, 2.5)),
            "rr_ci_high": float(np.percentile(samples, 97.5)),
            "content_delta": float(delta.mean()),
            "content_ci_low": float(np.percentile(delta_samples, 2.5)),
            **{f"worst_{m}": v for m, v in worst.items()},
            "probe": probe.get(values, np.nan),
            "safe": bool(np.percentile(delta_samples, 2.5) > CONTENT_NONINFERIORITY_MARGIN
                         and all(v >= COLLAPSE_POINT_THRESHOLD for v in worst.values())),
            "improved": bool(np.percentile(samples, 2.5) > 0)})
        summary.append(entry)
    return pd.DataFrame(summary)


def main() -> None:
    args = parse_args()
    root = Path(args.features)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    checks = {**self_check(), **accumulator_check()}
    print("self-check against the locked estimators and a direct fit:")
    for name, value in checks.items():
        print(f"  {name:26s} |difference| = {value:.3e}")
    worst = max(checks.values())
    if worst > 1e-8:
        raise RuntimeError(f"restatement disagrees with the locked form by {worst:.3e}")
    print(f"  worst {worst:.3e} — restatement is exact\n")

    encoders = ([e.strip() for e in args.encoders.split(",") if e.strip()]
                or sorted(p.name for p in root.iterdir()
                          if p.is_dir() and p.name != "analysis"))

    rows, sweep_rows, probe_scores = [], [], {}
    for encoder in encoders:
        store = load_encoder(root / encoder)
        sections = sorted(store)
        scanners = sorted(store[sections[0]])
        destination = scanners.index(args.destination)
        dimension = store[sections[0]][scanners[0]][1].shape[1]
        print(f"{encoder}: {len(sections)} sections x {len(scanners)} scanners, dim {dimension}")

        stacks, evaluation, statistics = {}, {}, {}
        for section in sections:
            if set(store[section]) != set(scanners):
                print(f"  {section}: incomplete scanner set, skipped")
                continue
            stack, shared = align(store[section], scanners)
            wanted = json.loads(
                (Path(args.eval_locations) / encoder / f"{section}.json").read_text())
            mask = np.isin(shared, np.asarray(wanted["locations"]))
            stacks[section] = stack
            evaluation[section] = np.asarray(stack[:, mask], dtype=np.float64)
            shared_target = target_moments(stack[destination])
            statistics[section] = (shared_target, [
                source_moments(stack[i], stack[destination]) for i in range(len(scanners))])
            print(f"  {section}: {stack.shape[1]:,} shared, {int(mask.sum()):,} evaluated")
        del store

        fitted = sorted(stacks)
        total = sum(statistics[s][0]["n"] for s in fitted)
        print(f"  fitting population: {total:,} locations per scanner "
              f"({total / dimension:.1f} per dimension)\n")

        probe_features = {}
        for method, fit in FITS.items():
            for held_out in fitted:
                train = [s for s in fitted if s != held_out]
                corrected = []
                for index, scanner in enumerate(scanners):
                    if index == destination:
                        corrected.append(evaluation[held_out][index])
                        continue
                    source_mean, target_mean, matrix = fit(combine(
                        [(statistics[s][0], statistics[s][1][index]) for s in train]))
                    corrected.append(
                        (evaluation[held_out][index] - source_mean) @ matrix + target_mean)
                corrected = np.stack(corrected)
                probe_features[(method, held_out)] = corrected.astype(np.float32)
                raw_cache: dict = {}
                for entry in endpoints(corrected, evaluation[held_out],
                                       evaluation[held_out][destination], scanners, raw_cache):
                    rows.append({"encoder": encoder, "method": method,
                                 "section": held_out, **entry})
                reduction = 1.0 - rows[-1]["radius"] / rows[-1]["raw_radius"]
                print(f"  {method:11s} hold out {held_out:5s} rr={reduction:+.3f}")

        print(f"\n  scanner probe, every {PROBE_STRIDE}th evaluated location")
        for method in FITS:
            features, groups, folds = [], [], []
            for section in fitted:
                stack = probe_features[(method, section)][:, ::PROBE_STRIDE]
                fold = fitted.index(section) % RF1_FOLDS
                for index in range(stack.shape[0]):
                    features.append(stack[index])
                    groups.append(np.full(stack.shape[1], index))
                    folds.append(np.full(stack.shape[1], fold))
            scores = probe_balanced_accuracy(np.concatenate(features), np.concatenate(groups),
                                             np.concatenate(folds))
            probe_scores[(encoder, method)] = float(np.mean(scores))
            print(f"    {method:11s} {np.mean(scores):.4f}")
        del probe_features

        if args.sweep:
            print(f"\n  CORAL sample-size sweep, {encoder} (dim {dimension})")
            for size in SWEEP_SIZES:
                used = 0
                for held_out in fitted:
                    train = [s for s in fitted if s != held_out]
                    corrected = []
                    for index, scanner in enumerate(scanners):
                        if index == destination:
                            corrected.append(evaluation[held_out][index])
                            continue
                        if size:
                            take = max(1, size // len(train))
                            parts = []
                            for section in train:
                                stack = stacks[section][:, :take]
                                parts.append((target_moments(stack[destination]),
                                              source_moments(stack[index],
                                                             stack[destination])))
                        else:
                            parts = [(statistics[s][0], statistics[s][1][index])
                                     for s in train]
                        stats = combine(parts)
                        used = stats["n"]
                        source_mean, target_mean, matrix = coral_from(stats)
                        corrected.append(
                            (evaluation[held_out][index] - source_mean) @ matrix + target_mean)
                    corrected = np.stack(corrected)
                    sweep_rows.append({
                        "encoder": encoder, "dim": dimension, "requested": size,
                        "fit_samples": int(used),
                        "samples_per_dimension": used / dimension,
                        "section": held_out,
                        "rr": 1.0 - centroid_rms(corrected) / centroid_rms(evaluation[held_out])})
                block = [r for r in sweep_rows
                         if r["encoder"] == encoder and r["requested"] == size]
                mean = float(np.mean([r["rr"] for r in block]))
                print(f"    n={block[0]['fit_samples']:>7,} "
                      f"({block[0]['samples_per_dimension']:>7.1f}/dim)  rr={mean:+.4f}")
        del stacks, statistics, evaluation
        print()

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_section.csv", index=False)

    table = bootstrap(frame, ["encoder", "method"], probe_scores)
    table.to_csv(output_dir / "frontier.csv", index=False)
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(output_dir / "coral_sample_size.csv", index=False)

    pd.set_option("display.width", 240)
    print(f"\n=== feature-space correction toward {args.destination}, "
          "leave-one-section-out ===")
    print(table[["encoder", "method", "rr", "rr_ci_low", "rr_ci_high", "content_delta",
                 "worst_variance_trace", "probe", "safe", "improved"]]
          .round(4).to_string(index=False))

    (output_dir / "summary.json").write_text(json.dumps({
        "analysis": "plism_core_feature_correction",
        "features": str(root), "destination": args.destination,
        "encoders": encoders, "probe_stride": PROBE_STRIDE,
        "bootstrap": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
        "chance": 1.0 / frame["scanner"].nunique(), "ridge": RIDGE,
        "self_check_worst": worst}, indent=2))
    print(f"\nwrote {output_dir}")


if __name__ == "__main__":
    main()
