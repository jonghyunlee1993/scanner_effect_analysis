"""Train one E8 paired residual arm for one target and one outer fold.

Two stages, both driven by `--task-index` so a SLURM array can run them.

- `grid`  — outer fold 0 only, the declared hyperparameter grid.  This is the
  only search permitted by the contract and it is resolved on inner-validation
  image metrics alone.
- `folds` — outer folds 1..4 with the winning configuration of each arm, read
  back from the grid stage's summaries.

Model selection touches no PFM embedding, no scanner radius, no content margin
and no tissue label.  Frozen rules: `docs/e8_paired_residual_contract.md`.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from e8_paired_residual import (
    E8_ARMS,
    E8_VERSION,
    PairedScannerDataModule,
    ResidualCorrector,
)
from e5_reinhard_residual_frequency import RF1_FOLDS


E8_SEED = 20260803
E8_GRID_FOLD = 0
E8_LAMBDA_PIXEL = (0.1, 0.3)
E8_LAMBDA_IDENTITY = (0.0, 0.03, 0.3)

# Contract section 7 gate legs.  They live here rather than in the audit so the
# grid winner is chosen by the same rule that later judges it: selecting purely
# on `paired_mae` once picked a configuration that beat the base overall but in
# only three of five source scanners, which the gate then rejected -- and that
# configuration would have been carried into every remaining fold.
E8_GATE_MIN_SOURCES = 4.0
E8_BAND_TOLERANCE = 0.05
E8_GAMUT_TOLERANCE = 0.05
E8_GAMUT_SLACK = 1e-4


def gate_legs(summary: dict) -> dict:
    """The three no-harm legs for one run, each read straight from its summary."""
    band = summary.get("val_band_rmse")
    band_base = summary.get("val_band_rmse_base")
    excursion = float(summary.get("val_material_range_fraction") or 0.0)
    excursion_base = float(summary.get("val_material_range_fraction_base") or 0.0)
    return {
        "paired_improved": bool(
            summary["val_paired_mae"] < summary["val_paired_mae_base"]
            and float(summary.get("val_paired_sources_improved") or 0.0)
            >= E8_GATE_MIN_SOURCES
        ),
        "band_non_inferior": bool(
            band is None
            or band_base is None
            or band <= band_base * (1.0 + E8_BAND_TOLERANCE)
        ),
        "gamut_non_inferior": bool(
            excursion <= excursion_base * (1.0 + E8_GAMUT_TOLERANCE) + E8_GAMUT_SLACK
        ),
    }


def gate_passes(summary: dict) -> bool:
    return all(gate_legs(summary).values())


def grid_configurations():
    """The declared search: arm A over pixel x identity, arm B over pixel.

    Arm B is listed first so that a throttled array puts the ceiling probe -- the
    reason this condition exists -- in its opening wave, and so that both arms
    are exercised before the array has committed all its slots.  An arm-major
    ordering once hid a batch-axis error on the `free` path behind a wave of
    `gainfield` runs.
    """
    values = [
        {"arm": "free", "lambda_pixel": lambda_pixel, "lambda_identity": 0.0}
        for lambda_pixel in E8_LAMBDA_PIXEL
    ]
    values.extend(
        {
            "arm": "gainfield",
            "lambda_pixel": lambda_pixel,
            "lambda_identity": lambda_identity,
        }
        for lambda_pixel in E8_LAMBDA_PIXEL
        for lambda_identity in E8_LAMBDA_IDENTITY
    )
    return values


def fold_configurations(output_root: Path, target: str):
    """Winning configuration of each arm, applied to outer folds 1..4."""
    values = []
    for arm in E8_ARMS:
        winner = select_grid_winner(output_root, target, arm)
        for fold in range(1, RF1_FOLDS):
            values.append({**winner, "outer_fold": fold})
    return values


def run_directory(root: Path, target: str, arm: str, fold: int, tag: str) -> Path:
    return Path(root) / target / arm / f"fold{fold}" / tag


def configuration_tag(configuration: dict) -> str:
    return (
        f"pix{configuration['lambda_pixel']:g}_id{configuration['lambda_identity']:g}"
    )


def select_grid_winner(output_root: Path, target: str, arm: str) -> dict:
    """Best gate-eligible configuration, then lowest inner-validation paired MAE.

    Eligibility comes first because a configuration that cannot clear the frozen
    no-harm gate on the grid fold has no business being carried into the other
    four.  Within the eligible set the ranking is the declared image statistic,
    and a tie is broken toward the larger identity weight -- that is, toward
    RF1U.  No PFM quantity enters any leg of this.
    """
    candidates = []
    for configuration in grid_configurations():
        if configuration["arm"] != arm:
            continue
        path = run_directory(
            output_root, target, arm, E8_GRID_FOLD, configuration_tag(configuration)
        ) / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        candidates.append(
            (
                not gate_passes(summary),
                summary["val_paired_mae"],
                -configuration["lambda_identity"],
                configuration,
            )
        )
    if not candidates:
        raise RuntimeError(f"no completed grid runs for {target}/{arm}")
    candidates.sort(key=lambda value: value[:3])
    return dict(candidates[0][3])


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("grid", "folds"), required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--target", default="gt450")
    parser.add_argument("--cache", default="outputs/e8_residual/cache")
    parser.add_argument("--output", default="outputs/e8_residual/runs")
    parser.add_argument("--max-epochs", type=int, default=14)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E8 training requires a CUDA device")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    output_root = Path(args.output)
    if args.stage == "grid":
        configurations = [
            {**value, "outer_fold": E8_GRID_FOLD} for value in grid_configurations()
        ]
    else:
        configurations = fold_configurations(output_root, args.target)
    if not 0 <= args.task_index < len(configurations):
        raise SystemExit(f"task index outside 0..{len(configurations) - 1}")
    configuration = configurations[args.task_index]

    pl.seed_everything(E8_SEED + args.task_index, workers=True)
    directory = run_directory(
        output_root,
        args.target,
        configuration["arm"],
        configuration["outer_fold"],
        configuration_tag(configuration),
    )
    directory.mkdir(parents=True, exist_ok=True)

    data = PairedScannerDataModule(
        target=args.target,
        outer_fold=configuration["outer_fold"],
        cache_root=args.cache,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    model = ResidualCorrector(
        arm=configuration["arm"],
        target=args.target,
        outer_fold=configuration["outer_fold"],
        lambda_pixel=configuration["lambda_pixel"],
        lambda_identity=configuration["lambda_identity"],
        learning_rate=args.learning_rate,
    )
    checkpoint = ModelCheckpoint(
        dirpath=directory,
        filename="best",
        monitor="val/paired_mae",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        gradient_clip_val=1.0,
        # Not `deterministic="warn"`: `reflection_pad2d_backward_cuda` has no
        # deterministic implementation, so the flag never bought determinism
        # here -- it only logged that fact and forced slower kernels.  Runs are
        # seeded and the seed is recorded; bitwise reproducibility is not
        # claimed, and no endpoint in this study depends on it.
        deterministic=False,
        log_every_n_steps=50,
        default_root_dir=str(directory),
        logger=CSVLogger(save_dir=str(directory), name="csv", version=""),
        callbacks=[
            checkpoint,
            EarlyStopping(monitor="val/paired_mae", mode="min", patience=args.patience),
        ],
    )
    trainer.fit(model, datamodule=data)

    metrics = {
        name: float(value)
        for name, value in trainer.callback_metrics.items()
        if np.isscalar(value) or torch.is_tensor(value)
    }
    summary = {
        "analysis": "e8_paired_residual_run",
        "e8_version": E8_VERSION,
        "stage": args.stage,
        "task_index": args.task_index,
        "target": args.target,
        "pfm_feature_access": False,
        "outcome_access": False,
        **configuration,
        "inner_fold": data.inner_fold,
        "slides": {name: len(ids) for name, ids in data.splits.items()},
        "parameters": int(sum(p.numel() for p in model.net.parameters())),
        "epochs_run": int(trainer.current_epoch),
        "val_paired_mae": float(checkpoint.best_model_score),
        "val_paired_mae_base": metrics.get("val/paired_mae_base"),
        "val_paired_sources_improved": metrics.get("val/paired_sources_improved"),
        "val_band_rmse": metrics.get("val/band_rmse"),
        "val_band_rmse_base": metrics.get("val/band_rmse_base"),
        "val_band_rmse_gain": metrics.get("val/band_rmse_gain"),
        "val_sources_improved": metrics.get("val/sources_improved"),
        "val_projection_fraction": metrics.get("val/projection_fraction"),
        "val_material_range_fraction": metrics.get("val/material_range_fraction"),
        "val_material_range_fraction_base": metrics.get("val/material_range_fraction_base"),
        "checkpoint": str(checkpoint.best_model_path),
        "seed": E8_SEED + args.task_index,
        "git_revision": git_revision(),
        "torch": torch.__version__,
        "lightning": pl.__version__,
        "node": platform.node(),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
