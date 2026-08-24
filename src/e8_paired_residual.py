"""Pure operations and Lightning modules for the E8 paired residual baseline.

E8 predicts a correction on top of the frozen RF1U output, supervised by the
registered acquisition of the same physical location on the target scanner.
Two arms with different guarantees share every other component:

- `gainfield` (arm A) predicts a spatially varying per-band log gain, so the
  output minus the base is exactly `sum_k (G_k - 1) * b_k` with `G_k > 0`
  wherever the gamut projection does not bind.  The residual is therefore a
  strictly positive pointwise rescaling of the base's own band maps: nothing is
  synthesized and no band's zero crossings move.  Hue is preserved and the same
  shared-OD gamut projection closes the pipeline.  A spatially varying gain is
  *not* a globally zero-phase filter the way constant-gain RF1U is; see
  contract section 3.4, which records what is exact and what is not.
- `free` (arm B) predicts an unconstrained three-channel OD residual.  It is the
  ceiling probe -- pix2pix's generator -- and it carries no guarantee beyond a
  per-channel OD clamp.

Both heads are zero-initialised, so at step zero either arm reproduces RF1U
exactly and training can only be reported as movement away from the analytic
solution.

Nothing in this module reads a PFM feature, an embedding, a scanner-radius
endpoint or a tissue label.  Frozen rules: `docs/e8_paired_residual_contract.md`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from analyze_rf1_multiscale import (
    PYRAMID_SIGMAS,
    _gaussian_blur_increment,
    laplacian_pyramid,
)
from build_e8_cache import E8_CROP, E8_MAX_SHIFT, E8_VALID
from build_rf1m_slide_band_energy import fold_lab_statistics
from e5_comparator_population import (
    SCANNERS,
    od_to_rgb01,
    reinhard_lab,
    rgb01_to_od,
)
from e5_reinhard_residual_frequency import (
    OD_RGB8_MAX,
    RF1_FOLDS,
    fold_assignments,
)
from extract_rf1u_features import fold_gains, load_band_energy
from rf1u_unpaired import source_indices, target_index


E8_VERSION = "e8_paired_residual_v1"
E8_ARMS = ("gainfield", "free")
E8_LOG_GAIN_LIMIT = math.log(4.0)
E8_ENERGY_WINDOW_SIGMA = 8.0
E8_ENERGY_POOL = 16
# Four orders of magnitude below the weakest band energy carried by tissue, so
# empty background cannot dominate a log-ratio loss while real structure is
# untouched.  RF1M_ENERGY_FLOOR (1e-20) is a division guard, not a loss floor.
E8_LOSS_ENERGY_FLOOR = 1e-6
E8_INPUT_OD_SCALE = 2.0
E8_INPUT_BAND_SCALE = 10.0


# --------------------------------------------------------------------------
# shared-OD composition
# --------------------------------------------------------------------------


def shared_od_apply(rgb01: torch.Tensor, gains: torch.Tensor) -> dict:
    """Apply per-band gains through one shared, gamut-projected OD residual.

    This is `analyze_rf1_multiscale.shared_od_multiscale` generalised so a gain
    may vary per sample and per pixel.  `gains` is broadcastable to
    `(len(sigmas), N, H, W)`; a constant gain reproduces the frozen operator and
    a gain of exactly one is the identity.
    """
    if rgb01.ndim != 4 or rgb01.shape[-1] != 3:
        raise ValueError("shared-OD composition requires NxHxWx3 RGB")
    od = rgb01_to_od(rgb01)
    mean_od = od.mean(dim=-1)
    bands, base = laplacian_pyramid(mean_od, PYRAMID_SIGMAS)
    stacked = torch.stack(bands, dim=0)
    corrected = base + (gains * stacked).sum(dim=0)
    proposed = corrected - mean_od
    lower = -od.amin(dim=-1)
    upper = OD_RGB8_MAX - od.amax(dim=-1)
    projected = torch.minimum(torch.maximum(proposed, lower), upper)
    preproject = od_to_rgb01(od + proposed[..., None])
    output = od_to_rgb01(od + projected[..., None]).clamp(0.0, 1.0)
    return {
        "output": output,
        "projection_fraction": ((projected - proposed).abs() > 1e-7).float().mean(
            dim=(1, 2)
        ),
        "material_range_fraction": (
            (preproject < -1.0 / 255.0) | (preproject > 1.0 + 1.0 / 255.0)
        )
        .float()
        .mean(dim=(1, 2, 3)),
    }


def free_od_apply(rgb01: torch.Tensor, residual: torch.Tensor) -> dict:
    """Add an unconstrained three-channel OD residual and clamp to the OD range."""
    if rgb01.shape != residual.shape:
        raise ValueError("free residual must match the image shape")
    od = rgb01_to_od(rgb01)
    proposed = od + residual
    clamped = proposed.clamp(0.0, OD_RGB8_MAX)
    return {
        "output": od_to_rgb01(clamped).clamp(0.0, 1.0),
        "projection_fraction": ((clamped - proposed).abs() > 1e-7)
        .float()
        .mean(dim=(1, 2, 3)),
        "material_range_fraction": (
            (proposed < -1e-3) | (proposed > OD_RGB8_MAX + 1e-3)
        )
        .float()
        .mean(dim=(1, 2, 3)),
    }


# --------------------------------------------------------------------------
# band statistics and alignment
# --------------------------------------------------------------------------


def band_maps(mean_od: torch.Tensor):
    bands, base = laplacian_pyramid(mean_od, PYRAMID_SIGMAS)
    return torch.stack(bands, dim=0), base


def local_band_energy(bands: torch.Tensor) -> torch.Tensor:
    """Gaussian-windowed local mean square of each band, shape BxNxHxW.

    Kept for the audit, where cost does not matter and a smooth window reads
    more naturally.  Training uses `pooled_band_energy`, which is the same
    statistic on a non-overlapping grid and two orders of magnitude cheaper.
    """
    squared = bands.square()
    return torch.stack(
        [
            _gaussian_blur_increment(squared[index], E8_ENERGY_WINDOW_SIGMA)
            for index in range(squared.shape[0])
        ],
        dim=0,
    )


def pooled_band_energy(bands: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Local mean square of each band on a non-overlapping window grid.

    The band maps are cropped to the residual-aligned window first, because the
    shift is defined in full-resolution pixels, and only then pooled.  A box
    window of `E8_ENERGY_POOL` pixels estimates the same local band power a
    Gaussian of comparable scale does, gives independent cells, and replaces six
    65-tap separable convolutions per step with one pooling call.
    """
    cropped = aligned_window(bands.square(), shift)
    count, batch, height, width = cropped.shape
    pooled = F.avg_pool2d(
        cropped.reshape(count * batch, 1, height, width), E8_ENERGY_POOL
    )
    return pooled.reshape(count, batch, *pooled.shape[-2:])


def global_band_energy(mean_od: torch.Tensor) -> torch.Tensor:
    """Per-patch mean square of each band, shape NxB -- the RF1U accumulation."""
    bands, _ = band_maps(mean_od)
    return bands.square().mean(dim=(2, 3)).transpose(0, 1)


def aligned_window(
    value: torch.Tensor, shift: torch.Tensor, valid: int = E8_VALID
) -> torch.Tensor:
    """Crop the `valid` window offset by an integer per-sample shift.

    `value` is `(..., N, H, W)`; `shift` is `(N, 2)` holding `(ty, tx)` under the
    convention fixed by `build_e8_cache.best_integer_shift`.  A shift of zero
    returns the centred window, which is what the target acquisition always uses.
    """
    if shift.ndim != 2 or shift.shape[-1] != 2:
        raise ValueError("shift must be an Nx2 integer tensor")
    count = value.shape[-3]
    if count != len(shift):
        raise ValueError(
            f"aligned_window expects the batch on dim -3, found {count} there "
            f"against {len(shift)} shifts; permute a channel axis out of the way "
            f"or use aligned_window_channels"
        )
    steps = torch.arange(valid, device=value.device)
    rows = (E8_MAX_SHIFT + shift[:, 0])[:, None] + steps[None, :]
    columns = (E8_MAX_SHIFT + shift[:, 1])[:, None] + steps[None, :]
    index = torch.arange(count, device=value.device)
    return value[..., index[:, None, None], rows[:, :, None], columns[:, None, :]]


def aligned_window_channels(value: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Aligned window of an NxHxWxC tensor, returned as NxCxhxw.

    `aligned_window` takes the batch on dim -3.  Permuting NHWC to NCHW puts the
    channel there instead, which silently indexes the wrong axis, so the channel
    is moved to the front and restored afterwards.
    """
    return aligned_window(value.permute(3, 0, 1, 2), shift).permute(1, 0, 2, 3)


# --------------------------------------------------------------------------
# frozen RF1U base
# --------------------------------------------------------------------------


def load_base_state(
    target: str,
    outer_fold: int,
    statistics_root: str = "outputs/e5_image_statistics",
    energy_root: str = "outputs/rf1u_multitarget/energy",
    fov: int = E8_CROP,
) -> dict:
    """Reinhard moments and shrunk RF1U band gains for one target and fold."""
    with np.load(Path(statistics_root) / f"fov_{fov}.npz") as statistics:
        values = {name: statistics[name] for name in statistics.files}
    slide_ids = [str(value) for value in values["slide_ids"]]
    assignments = fold_assignments(slide_ids)
    lab_mean, lab_std = fold_lab_statistics(values, slide_ids, assignments)
    _, _, energy = load_band_energy(Path(energy_root), target, fov)
    if [str(value) for value in energy["slide_ids"]] != slide_ids:
        raise RuntimeError("band-energy slide order differs from the E5 statistics")
    gains = fold_gains(energy, outer_fold)["gain"]
    sources = source_indices(target)
    if gains.shape != (len(sources), len(PYRAMID_SIGMAS)):
        raise RuntimeError(f"invalid RF1U gain shape: {gains.shape}")
    return {
        "slide_ids": slide_ids,
        "assignments": assignments,
        "lab_mean": lab_mean[outer_fold],
        "lab_std": lab_std[outer_fold],
        "gains": gains,
        "target_index": target_index(target),
        "source_indices": np.asarray(sources, dtype=np.int64),
    }


class FrozenBase(nn.Module):
    """RF1U toward one target for one outer fold.  Buffers only, never trained."""

    def __init__(self, state: dict):
        super().__init__()
        self.register_buffer(
            "lab_mean", torch.as_tensor(state["lab_mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "lab_std", torch.as_tensor(state["lab_std"], dtype=torch.float32)
        )
        self.register_buffer(
            "gains", torch.as_tensor(state["gains"], dtype=torch.float32)
        )
        self.register_buffer(
            "source_scanner", torch.as_tensor(state["source_indices"])
        )
        self.target_index = int(state["target_index"])

    @torch.no_grad()
    def forward(self, rgb01: torch.Tensor, source_position: torch.Tensor) -> dict:
        scanner = self.source_scanner[source_position]
        shape = (-1, 1, 1, 3)
        base = reinhard_lab(
            rgb01,
            self.lab_mean[scanner].reshape(shape),
            self.lab_std[scanner].reshape(shape),
            self.lab_mean[self.target_index].reshape(1, 1, 1, 3),
            self.lab_std[self.target_index].reshape(1, 1, 1, 3),
        )
        gain = self.gains[source_position].permute(1, 0)[:, :, None, None]
        return shared_od_apply(base["output"], gain)


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------


class ConditionedBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.first = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm_first = nn.GroupNorm(8, out_channels)
        self.second = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm_second = nn.GroupNorm(8, out_channels)
        self.film = nn.Linear(cond_dim, 2 * out_channels)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        value = F.silu(self.norm_first(self.first(value)))
        value = self.norm_second(self.second(value))
        scale, shift = self.film(condition).chunk(2, dim=-1)
        value = value * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return F.silu(value)


class ConditionedUNet(nn.Module):
    """Fully convolutional U-Net with FiLM conditioning on the source scanner."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 3,
        widths=(48, 96, 192, 384),
        sources: int = 5,
        cond_dim: int = 16,
    ):
        super().__init__()
        self.embedding = nn.Embedding(sources, cond_dim)
        self.encoder = nn.ModuleList()
        previous = in_channels
        for width in widths:
            self.encoder.append(ConditionedBlock(previous, width, cond_dim))
            previous = width
        self.decoder = nn.ModuleList()
        self.reduce = nn.ModuleList()
        for level in range(len(widths) - 1, 0, -1):
            self.reduce.append(nn.Conv2d(widths[level], widths[level - 1], 3, padding=1))
            self.decoder.append(
                ConditionedBlock(2 * widths[level - 1], widths[level - 1], cond_dim)
            )
        self.head = nn.Conv2d(widths[0], out_channels, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, value: torch.Tensor, source_position: torch.Tensor):
        condition = self.embedding(source_position)
        skips = []
        for index, block in enumerate(self.encoder):
            if index > 0:
                value = F.avg_pool2d(value, 2)
            value = block(value, condition)
            skips.append(value)
        value = skips[-1]
        for index, (reduce, block) in enumerate(zip(self.reduce, self.decoder)):
            value = F.interpolate(value, scale_factor=2.0, mode="bilinear", align_corners=False)
            value = reduce(value)
            value = block(torch.cat([value, skips[-2 - index]], dim=1), condition)
        return self.head(value)


def network_input(base_rgb01: torch.Tensor) -> torch.Tensor:
    """Seven-channel NCHW input: the base OD plus its mean-OD decomposition."""
    od = rgb01_to_od(base_rgb01)
    bands, low_pass = band_maps(od.mean(dim=-1))
    return torch.cat(
        [
            od.permute(0, 3, 1, 2) / E8_INPUT_OD_SCALE,
            bands.permute(1, 0, 2, 3) * E8_INPUT_BAND_SCALE,
            (low_pass / E8_INPUT_OD_SCALE)[:, None],
        ],
        dim=1,
    )


# --------------------------------------------------------------------------
# Lightning module
# --------------------------------------------------------------------------


class ResidualCorrector(pl.LightningModule):
    def __init__(
        self,
        arm: str,
        target: str,
        outer_fold: int,
        lambda_pixel: float = 0.1,
        lambda_identity: float = 0.0,
        lambda_gamut: float = 10.0,
        learning_rate: float = 2e-4,
        warmup_steps: int = 500,
        statistics_root: str = "outputs/e5_image_statistics",
        energy_root: str = "outputs/rf1u_multitarget/energy",
    ):
        super().__init__()
        if arm not in E8_ARMS:
            raise ValueError(f"arm must be one of {E8_ARMS}, got {arm!r}")
        self.save_hyperparameters()
        state = load_base_state(target, outer_fold, statistics_root, energy_root)
        self.base = FrozenBase(state)
        self.net = ConditionedUNet(sources=len(state["source_indices"]))
        self.register_buffer(
            "rf1u_log_gain",
            torch.log(torch.as_tensor(state["gains"], dtype=torch.float32)),
        )
        self._reset_validation()

    # -- composition -------------------------------------------------------

    def correct(self, rgb01: torch.Tensor, source_position: torch.Tensor):
        with torch.autocast("cuda", enabled=False):
            base = self.base(rgb01.float(), source_position)
            features = network_input(base["output"])
        prediction = self.net(features, source_position)
        with torch.autocast("cuda", enabled=False):
            prediction = prediction.float()
            if self.hparams.arm == "gainfield":
                log_gain = prediction.clamp(-E8_LOG_GAIN_LIMIT, E8_LOG_GAIN_LIMIT)
                report = shared_od_apply(
                    base["output"], torch.exp(log_gain).permute(1, 0, 2, 3)
                )
                report["log_gain"] = log_gain
            else:
                report = free_od_apply(
                    base["output"], prediction.permute(0, 2, 3, 1)
                )
        report["base"] = base["output"]
        report["base_projection_fraction"] = base["projection_fraction"]
        report["base_material_range_fraction"] = base["material_range_fraction"]
        return report

    # -- losses ------------------------------------------------------------

    def _losses(self, report: dict, target_rgb01: torch.Tensor, shift: torch.Tensor):
        output_od = rgb01_to_od(report["output"])
        target_od = rgb01_to_od(target_rgb01)
        output_bands, _ = band_maps(output_od.mean(dim=-1))
        target_bands, _ = band_maps(target_od.mean(dim=-1))
        output_energy = pooled_band_energy(output_bands, shift)
        target_energy = pooled_band_energy(target_bands, torch.zeros_like(shift))
        band = (
            torch.log(output_energy + E8_LOSS_ENERGY_FLOOR)
            - torch.log(target_energy + E8_LOSS_ENERGY_FLOOR)
        ).abs().mean()

        if self.hparams.arm == "gainfield":
            output_pixel = aligned_window(output_od.mean(dim=-1), shift)
            target_pixel = aligned_window(
                target_od.mean(dim=-1), torch.zeros_like(shift)
            )
        else:
            output_pixel = aligned_window_channels(output_od, shift)
            target_pixel = aligned_window_channels(target_od, torch.zeros_like(shift))
        # A shift sitting on the search boundary means the true offset may lie
        # outside the cached range, so the pixel term would be paying for
        # misregistration.  The band term is shift-robust and keeps every patch.
        interior = (shift.abs().amax(dim=1) < E8_MAX_SHIFT).to(output_pixel.dtype)
        per_sample = (output_pixel - target_pixel).abs().flatten(1).mean(dim=1)
        pixel = (per_sample * interior).sum() / interior.sum().clamp_min(1.0)

        gamut = report["material_range_fraction"].mean()
        identity = torch.zeros((), device=band.device)
        if self.hparams.arm == "gainfield" and self.hparams.lambda_identity > 0:
            anchor = self.rf1u_log_gain[report["source_position"]]
            identity = (
                report["log_gain"] - anchor[:, :, None, None]
            ).square().mean()
        total = (
            band
            + self.hparams.lambda_pixel * pixel
            + self.hparams.lambda_identity * identity
            + self.hparams.lambda_gamut * gamut
        )
        return total, {"band": band, "pixel": pixel, "gamut": gamut, "identity": identity}

    # -- steps -------------------------------------------------------------

    def _unpack(self, batch):
        source, target, position, shift = batch
        return (
            source.float().div(255.0),
            target.float().div(255.0),
            position.long(),
            shift.long(),
        )

    def training_step(self, batch, batch_index):
        source, target, position, shift = self._unpack(batch)
        report = self.correct(source, position)
        report["source_position"] = position
        loss, parts = self._losses(report, target, shift)
        self.log("train/loss", loss, prog_bar=True, batch_size=len(source))
        for name, value in parts.items():
            self.log(f"train/{name}", value, batch_size=len(source))
        return loss

    def _reset_validation(self):
        sources = 5
        bands = len(PYRAMID_SIGMAS)
        self._energy = {
            name: torch.zeros(sources, bands, dtype=torch.float64)
            for name in ("output", "base", "target")
        }
        self._counts = torch.zeros(sources, dtype=torch.float64)
        self._projection = torch.zeros(sources, dtype=torch.float64)
        self._clamp = torch.zeros(sources, dtype=torch.float64)
        self._clamp_base = torch.zeros(sources, dtype=torch.float64)
        self._paired = {
            name: torch.zeros(sources, dtype=torch.float64) for name in ("output", "base")
        }
        self._paired_counts = torch.zeros(sources, dtype=torch.float64)

    @staticmethod
    def _paired_mae(rgb01: torch.Tensor, target_od: torch.Tensor, shift: torch.Tensor):
        """Residual-aligned mean absolute OD error against the paired target.

        Three channels for both arms, so the statistic is identical whatever the
        arm predicts -- arm B may move colour and arm A may not, and the gate has
        to see both.
        """
        window = aligned_window_channels(rgb01_to_od(rgb01), shift)
        anchor = aligned_window_channels(target_od, torch.zeros_like(shift))
        return (window - anchor).abs().flatten(1).mean(dim=1)

    def on_validation_epoch_start(self):
        self._reset_validation()

    @torch.no_grad()
    def validation_step(self, batch, batch_index):
        source, target, position, shift = self._unpack(batch)
        report = self.correct(source, position)
        report["source_position"] = position
        loss, parts = self._losses(report, target, shift)
        self.log("val/loss", loss, batch_size=len(source))
        for name, value in parts.items():
            self.log(f"val/{name}", value, batch_size=len(source))

        energies = {
            "output": global_band_energy(rgb01_to_od(report["output"]).mean(dim=-1)),
            "base": global_band_energy(rgb01_to_od(report["base"]).mean(dim=-1)),
            "target": global_band_energy(rgb01_to_od(target).mean(dim=-1)),
        }
        index = position.cpu()
        for name, value in energies.items():
            self._energy[name].index_add_(0, index, value.double().cpu())
        ones = torch.ones(len(index), dtype=torch.float64)
        self._counts.index_add_(0, index, ones)
        self._projection.index_add_(
            0, index, report["projection_fraction"].double().cpu()
        )
        self._clamp.index_add_(
            0, index, report["material_range_fraction"].double().cpu()
        )
        self._clamp_base.index_add_(
            0, index, report["base_material_range_fraction"].double().cpu()
        )

        target_od = rgb01_to_od(target)
        interior = shift.abs().amax(dim=1) < E8_MAX_SHIFT
        if bool(interior.any()):
            inside = index[interior.cpu()]
            for name, image in (("output", report["output"]), ("base", report["base"])):
                errors = self._paired_mae(image[interior], target_od[interior], shift[interior])
                self._paired[name].index_add_(0, inside, errors.double().cpu())
            self._paired_counts.index_add_(
                0, inside, torch.ones(len(inside), dtype=torch.float64)
            )

    def on_validation_epoch_end(self):
        counts = self._counts.clamp_min(1.0)[:, None]
        mean = {name: value / counts for name, value in self._energy.items()}
        floor = 1e-20
        rmse = {}
        for name in ("output", "base"):
            ratio = torch.log(mean[name].clamp_min(floor)) - torch.log(
                mean["target"].clamp_min(floor)
            )
            rmse[name] = ratio.square().mean(dim=1).sqrt()
        self.log("val/band_rmse", rmse["output"].mean().item(), prog_bar=True)
        self.log("val/band_rmse_base", rmse["base"].mean().item())
        self.log(
            "val/band_rmse_gain", (rmse["base"] - rmse["output"]).mean().item()
        )
        self.log(
            "val/sources_improved",
            float((rmse["output"] < rmse["base"]).sum().item()),
        )

        paired_counts = self._paired_counts.clamp_min(1.0)
        paired = {name: value / paired_counts for name, value in self._paired.items()}
        self.log("val/paired_mae", paired["output"].mean().item(), prog_bar=True)
        self.log("val/paired_mae_base", paired["base"].mean().item())
        self.log(
            "val/paired_sources_improved",
            float((paired["output"] < paired["base"]).sum().item()),
        )
        self.log(
            "val/projection_fraction",
            (self._projection / self._counts.clamp_min(1.0)).mean().item(),
        )
        self.log(
            "val/material_range_fraction",
            (self._clamp / self._counts.clamp_min(1.0)).mean().item(),
        )
        self.log(
            "val/material_range_fraction_base",
            (self._clamp_base / self._counts.clamp_min(1.0)).mean().item(),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
        )
        total = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup = min(int(self.hparams.warmup_steps), max(total - 1, 1))

        def schedule(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
                "interval": "step",
            },
        }


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


class CachedPairs(Dataset):
    """(slide, source, location) triples served from RAM-resident 256 px crops."""

    def __init__(
        self,
        slide_ids,
        cache_root: Path,
        target: str,
        augment: bool,
        locations: int = 100,
    ):
        self.slide_ids = list(slide_ids)
        self.target_index = target_index(target)
        self.source_indices = np.asarray(source_indices(target), dtype=np.int64)
        self.locations = int(locations)
        self.augment = bool(augment)
        self.crops = [
            np.load(Path(cache_root) / "crops" / f"{value}.npy") for value in self.slide_ids
        ]
        self.shifts = [
            np.load(Path(cache_root) / "align" / target / f"{value}.npy")
            for value in self.slide_ids
        ]
        for crop, shift in zip(self.crops, self.shifts):
            if crop.shape != (6, self.locations, E8_CROP, E8_CROP, 3):
                raise ValueError(f"invalid cached crop shape: {crop.shape}")
            if shift.shape != (len(self.source_indices), self.locations, 2):
                raise ValueError(f"invalid cached shift shape: {shift.shape}")
        self.per_slide = len(self.source_indices) * self.locations

    def __len__(self) -> int:
        return len(self.slide_ids) * self.per_slide

    def __getitem__(self, index: int):
        slide, remainder = divmod(index, self.per_slide)
        position, location = divmod(remainder, self.locations)
        source = self.crops[slide][self.source_indices[position], location]
        target = self.crops[slide][self.target_index, location]
        shift = self.shifts[slide][position, location].astype(np.int64)
        if self.augment:
            code = int(np.random.randint(4))
            if code & 1:
                source, target = source[::-1], target[::-1]
                shift = np.array([-shift[0], shift[1]])
            if code & 2:
                source, target = source[:, ::-1], target[:, ::-1]
                shift = np.array([shift[0], -shift[1]])
        return (
            torch.from_numpy(np.ascontiguousarray(source)),
            torch.from_numpy(np.ascontiguousarray(target)),
            int(position),
            torch.from_numpy(np.ascontiguousarray(shift)),
        )


class PairedScannerDataModule(pl.LightningDataModule):
    """Outer test fold `f`, inner validation fold `(f + 1) % RF1_FOLDS`."""

    def __init__(
        self,
        target: str,
        outer_fold: int,
        cache_root: str = "outputs/e8_residual/cache",
        statistics_root: str = "outputs/e5_image_statistics",
        batch_size: int = 32,
        workers: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.inner_fold = (int(outer_fold) + 1) % RF1_FOLDS

    def setup(self, stage=None):
        with np.load(Path(self.hparams.statistics_root) / f"fov_{E8_CROP}.npz") as values:
            slide_ids = [str(value) for value in values["slide_ids"]]
        assignments = fold_assignments(slide_ids)
        held = {self.hparams.outer_fold, self.inner_fold}
        splits = {
            "train": [v for v in slide_ids if assignments[v] not in held],
            "val": [v for v in slide_ids if assignments[v] == self.inner_fold],
            "test": [v for v in slide_ids if assignments[v] == self.hparams.outer_fold],
        }
        root = Path(self.hparams.cache_root)
        self.datasets = {
            name: CachedPairs(ids, root, self.hparams.target, augment=(name == "train"))
            for name, ids in splits.items()
        }
        self.splits = splits

    def _loader(self, name: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            self.datasets[name],
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.workers,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=self.hparams.workers > 0,
        )

    def train_dataloader(self):
        return self._loader("train", True)

    def val_dataloader(self):
        return self._loader("val", False)

    def test_dataloader(self):
        return self._loader("test", False)
