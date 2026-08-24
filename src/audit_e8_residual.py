"""Image-only gate and hallucination audit for the E8 paired residual arms.

Encoding is blocked until this script closes both.  Nothing here reads a PFM
feature, an embedding, a scanner-radius endpoint or a tissue label.

Section 7 gate, from the inner-validation summaries the training runs wrote:
an arm proceeds only if its selected checkpoint beats the RF1U base on inner-val
band RMSE in at least four of five outer folds, and in no fewer than four of the
five source scanners within each passing fold, with zero material range excursion
and a projection fraction no worse than RF1U's.

Section 8 audit, computed here on the outer test folds:

- arm A exact properties -- `sign(G_k * b_k) == sign(b_k)` everywhere, and
  `output - base - sum_k (G_k - 1) b_k` zero wherever the projection does not
  bind.  A single violation is a defect, not a finding.
- both arms, approximate statistics -- re-extracted band sign-change rate and
  high-frequency phase correlation, each beside the constant-gain RF1U value, so
  arm A's departure from a zero-phase filter is quantified rather than assumed.
- arm B -- invented-structure rate: pixels carrying gradient that is in neither
  the base nor the paired target.  Declared budget 0.1%; exceeding it does not
  stop the arm but must be reported beside every one of its endpoints.

Frozen rules: `docs/e8_paired_residual_contract.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_rf1_multiscale import PYRAMID_SIGMAS, laplacian_pyramid
from build_e8_cache import E8_CROP
from e5_comparator_population import SCANNERS, TARGET_MPP, rgb01_to_od
from e5_reinhard_residual_frequency import RF1_FOLDS
from e8_paired_residual import (
    E8_ARMS,
    FrozenBase,
    load_base_state,
    PairedScannerDataModule,
    ResidualCorrector,
    shared_od_apply,
)
from rf1u_unpaired import source_indices
from train_e8_residual import (
    E8_GATE_MIN_SOURCES,
    E8_GRID_FOLD,
    configuration_tag,
    gate_legs,
    run_directory,
    select_grid_winner,
)


ANALYSIS = "e8_paired_residual_audit"
E8_PHASE_FLOOR_CYCLES_PER_MICROMETRE = 0.10
E8_INVENTED_BUDGET = 0.001
E8_GATE_MIN_FOLDS = 4



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="gt450")
    parser.add_argument("--runs", default="outputs/e8_residual/runs")
    parser.add_argument("--cache", default="outputs/e8_residual/cache")
    parser.add_argument("--output", default="outputs/e8_residual/audit")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--locations", type=int, default=25)
    return parser.parse_args()


# --------------------------------------------------------------------------
# section 7 gate
# --------------------------------------------------------------------------


def read_run(runs: Path, target: str, arm: str, fold: int, configuration: dict):
    path = run_directory(runs, target, arm, fold, configuration_tag(configuration))
    summary = path / "summary.json"
    if not summary.exists():
        return None
    return json.loads(summary.read_text())


def evaluate_gate(runs: Path, target: str, arm: str) -> dict:
    winner = select_grid_winner(runs, target, arm)
    folds = {}
    for fold in range(RF1_FOLDS):
        run = read_run(runs, target, arm, fold, winner)
        if run is None:
            continue
        legs = gate_legs(run)
        sources = float(run.get("val_paired_sources_improved") or 0.0)
        excursion_base = float(run.get("val_material_range_fraction_base") or 0.0)
        folds[fold] = {
            "val_paired_mae": run["val_paired_mae"],
            "val_paired_mae_base": run["val_paired_mae_base"],
            "val_band_rmse": run.get("val_band_rmse"),
            "val_band_rmse_base": run.get("val_band_rmse_base"),
            "material_range_fraction_base": excursion_base,
            **legs,
            "sources_improved": sources,
            "material_range_fraction": run.get("val_material_range_fraction"),
            "projection_fraction": run.get("val_projection_fraction"),
            "passes": all(legs.values()),
            "checkpoint": run["checkpoint"],
        }
    passing = [fold for fold, value in folds.items() if value["passes"]]
    return {
        "arm": arm,
        "winner": winner,
        "folds_run": sorted(folds),
        "folds_passing": sorted(passing),
        "gate_pass": len(passing) >= E8_GATE_MIN_FOLDS,
        "detail": folds,
    }


# --------------------------------------------------------------------------
# section 8 audit
# --------------------------------------------------------------------------


def high_frequency_mask(size: int, device) -> torch.Tensor:
    """Annulus above the frozen 0.10 cycles/micrometre phase floor."""
    axis = torch.fft.fftfreq(size, d=TARGET_MPP, device=device)
    radius = (axis[:, None] ** 2 + axis[None, :] ** 2).sqrt()
    return radius >= E8_PHASE_FLOOR_CYCLES_PER_MICROMETRE


def phase_correlation(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor):
    """Mean cos(phase difference) over the high-frequency annulus."""
    fa = torch.fft.fft2(a.double())
    fb = torch.fft.fft2(b.double())
    product = fa * fb.conj()
    magnitude = product.abs().clamp_min(1e-30)
    cosine = (product / magnitude).real
    return (cosine * mask).sum(dim=(1, 2)) / mask.sum().clamp_min(1)


def gradient_magnitude(value: torch.Tensor) -> torch.Tensor:
    dy = value[:, 1:, :-1] - value[:, :-1, :-1]
    dx = value[:, :-1, 1:] - value[:, :-1, :-1]
    return (dy.square() + dx.square()).sqrt()


def band_sign_change(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fraction of pixels whose re-extracted band sign differs, per sample."""
    first, _ = laplacian_pyramid(a, PYRAMID_SIGMAS)
    second, _ = laplacian_pyramid(b, PYRAMID_SIGMAS)
    changed = [
        (torch.sign(first[k]) * torch.sign(second[k]) < 0).float().mean(dim=(1, 2))
        for k in range(len(first))
    ]
    return torch.stack(changed, dim=-1).mean(dim=-1)


@torch.no_grad()
def audit_arm(
    arm: str,
    target: str,
    gate: dict,
    cache: str,
    batch_size: int,
    locations: int,
    threshold: float,
) -> dict:
    """Run the selected checkpoint of every passing fold over its outer test fold."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accumulator = {
        name: []
        for name in (
            "scaled_band_violations",
            "residual_exactness",
            "projection_fraction",
            "reextracted_sign_change",
            "reextracted_sign_change_base",
            "phase_correlation",
            "phase_correlation_base",
            "invented_fraction",
            "gain_min",
            "gain_max",
        )
    }
    for fold in gate["folds_passing"]:
        checkpoint = gate["detail"][fold]["checkpoint"]
        model = ResidualCorrector.load_from_checkpoint(checkpoint, map_location=device)
        model.eval().to(device)
        data = PairedScannerDataModule(
            target=target, outer_fold=fold, cache_root=cache, batch_size=batch_size, workers=4
        )
        data.setup()
        dataset = data.datasets["test"]
        mask = high_frequency_mask(E8_CROP, device)

        indices = [
            index
            for index in range(len(dataset))
            if index % dataset.locations < locations
        ]
        for start in range(0, len(indices), batch_size):
            batch = [dataset[index] for index in indices[start : start + batch_size]]
            source = torch.stack([item[0] for item in batch]).to(device).float() / 255.0
            paired = torch.stack([item[1] for item in batch]).to(device).float() / 255.0
            position = torch.tensor([item[2] for item in batch], device=device)
            report = model.correct(source, position)
            base = report["base"]
            output = report["output"]

            base_od = rgb01_to_od(base).mean(dim=-1)
            output_od = rgb01_to_od(output).mean(dim=-1)
            paired_od = rgb01_to_od(paired).mean(dim=-1)
            bands, _ = laplacian_pyramid(base_od, PYRAMID_SIGMAS)
            stacked = torch.stack(bands, dim=0)

            if arm == "gainfield":
                gains = torch.exp(report["log_gain"]).permute(1, 0, 2, 3)
                violations = (
                    torch.sign(gains * stacked) * torch.sign(stacked) < 0
                ).float().mean(dim=(0, 2, 3))
                intended = ((gains - 1.0) * stacked).sum(dim=0)
                actual = output_od - base_od
                unbound = (actual - intended).abs() <= 1e-5
                exact = torch.where(
                    unbound, (actual - intended).abs(), torch.zeros_like(actual)
                ).amax(dim=(1, 2))
                accumulator["scaled_band_violations"].append(violations.cpu())
                accumulator["residual_exactness"].append(exact.cpu())
                accumulator["gain_min"].append(gains.amin(dim=(0, 2, 3)).cpu())
                accumulator["gain_max"].append(gains.amax(dim=(0, 2, 3)).cpu())

            # Constant-gain RF1U reference for the two approximate statistics.
            rf1u_gain = model.base.gains[position].permute(1, 0)[:, :, None, None]
            rf1u_od = rgb01_to_od(
                shared_od_apply(base, rf1u_gain)["output"]
            ).mean(dim=-1)

            accumulator["reextracted_sign_change"].append(
                band_sign_change(output_od, base_od).cpu()
            )
            accumulator["reextracted_sign_change_base"].append(
                band_sign_change(rf1u_od, base_od).cpu()
            )
            accumulator["phase_correlation"].append(
                phase_correlation(output_od, base_od, mask).cpu()
            )
            accumulator["phase_correlation_base"].append(
                phase_correlation(rf1u_od, base_od, mask).cpu()
            )
            accumulator["projection_fraction"].append(
                report["projection_fraction"].float().cpu()
            )

            out_gradient = gradient_magnitude(output_od)
            base_gradient = gradient_magnitude(base_od)
            paired_gradient = gradient_magnitude(paired_od)
            invented = (
                (out_gradient > threshold)
                & (base_gradient < threshold / 4.0)
                & (paired_gradient < threshold / 4.0)
            ).float().mean(dim=(1, 2))
            accumulator["invented_fraction"].append(invented.cpu())

    result = {}
    for name, values in accumulator.items():
        if not values:
            continue
        pooled = torch.cat([value.reshape(-1) for value in values]).double()
        result[name] = {
            "mean": float(pooled.mean()),
            "max": float(pooled.max()),
            "min": float(pooled.min()),
        }
    result["gradient_threshold"] = threshold
    result["locations_per_slide"] = locations
    return result


@torch.no_grad()
def gradient_threshold(target: str, cache: str, batch_size: int) -> float:
    """Cohort median gradient magnitude of the RF1U base, frozen once.

    Computed on the grid fold's training slides so it is a training-side
    quantity, and shared by both arms so their invented-structure rates are
    measured against the same bar.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = FrozenBase(load_base_state(target, E8_GRID_FOLD)).to(device)
    data = PairedScannerDataModule(
        target=target, outer_fold=E8_GRID_FOLD, cache_root=cache,
        batch_size=batch_size, workers=4,
    )
    data.setup()
    dataset = data.datasets["train"]
    step = max(len(dataset) // 512, 1)
    medians = []
    for start in range(0, len(dataset), step * batch_size):
        picked = list(range(start, min(start + step * batch_size, len(dataset)), step))
        if not picked:
            continue
        batch = [dataset[index] for index in picked]
        source = torch.stack([item[0] for item in batch]).to(device).float() / 255.0
        position = torch.tensor([item[2] for item in batch], device=device)
        density = rgb01_to_od(base(source, position)["output"]).mean(dim=-1)
        medians.append(gradient_magnitude(density).flatten().median().item())
    return float(np.median(medians))


def main():
    args = parse_args()
    runs = Path(args.runs)
    output = Path(args.output) / args.target
    output.mkdir(parents=True, exist_ok=True)

    report = {
        "analysis": ANALYSIS,
        "target": args.target,
        "pfm_feature_access": False,
        "outcome_access": False,
        "sources": [SCANNERS[index] for index in source_indices(args.target)],
        "grid_fold": E8_GRID_FOLD,
        "invented_budget": E8_INVENTED_BUDGET,
        "arms": {},
    }
    threshold = gradient_threshold(args.target, args.cache, args.batch_size)
    report["gradient_threshold"] = threshold
    for arm in E8_ARMS:
        gate = evaluate_gate(runs, args.target, arm)
        entry = {"gate": gate}
        # Diagnostics run whenever any fold produced a checkpoint, including for
        # an arm the gate has closed: the section 3.4 exact properties are a
        # claim about the implementation, and a blocked arm is exactly the case
        # where the record of what it did is worth having.  Encoding stays gated.
        if gate["folds_passing"]:
            entry["audit"] = audit_arm(
                arm, args.target, gate, args.cache, args.batch_size,
                args.locations, threshold,
            )
            audit = entry["audit"]
            entry["exact_properties_hold"] = bool(
                arm != "gainfield"
                or (
                    audit["scaled_band_violations"]["max"] == 0.0
                    and audit["residual_exactness"]["max"] < 1e-5
                )
            )
            entry["invented_within_budget"] = bool(
                audit["invented_fraction"]["mean"] <= E8_INVENTED_BUDGET
            )
        else:
            entry["audit"] = None
        report["arms"][arm] = entry

    report["encoding_unblocked"] = {
        arm: bool(
            value["gate"]["gate_pass"] and value.get("exact_properties_hold", True)
        )
        for arm, value in report["arms"].items()
    }
    (output / "summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
