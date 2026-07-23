"""Lightning objective for Exp-06 low-frequency harmonization."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .model import LowFrequencyHarmonizer


def _weighted(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(values).clamp(0, 1)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _per_image(value: torch.Tensor) -> torch.Tensor:
    return value.flatten(1).mean(1)


def _ssim_loss(prediction: torch.Tensor, target: torch.Tensor, window: int = 7):
    pad = window // 2
    mu_x = F.avg_pool2d(prediction, window, 1, pad)
    mu_y = F.avg_pool2d(target, window, 1, pad)
    var_x = F.avg_pool2d(prediction.square(), window, 1, pad) - mu_x.square()
    var_y = F.avg_pool2d(target.square(), window, 1, pad) - mu_y.square()
    cov = F.avg_pool2d(prediction * target, window, 1, pad) - mu_x * mu_y
    c1, c2 = 0.02**2, 0.06**2
    score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    ).clamp_min(1e-6)
    return 1.0 - score.flatten(1).mean(1)


class Exp06Module(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = LowFrequencyHarmonizer(
            levels=int(cfg.pyramid.levels),
            width=int(cfg.model.width),
            residual_scale=float(cfg.model.residual_scale),
        )

    def checkpoint_metadata(self):
        return {
            "design_version": 6,
            "pyramid_levels": int(self.cfg.pyramid.levels),
            "target_scanners": list(self.cfg.target_scanners),
            "at2_hard_bypass": True,
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["exp06_metadata"] = self.checkpoint_metadata()

    def losses(self, batch):
        source, reference = batch["source"], batch["reference"]
        n = source.shape[0]
        context = batch["context_source"].unsqueeze(0).expand(n, -1, -1, -1, -1)
        mask = batch["context_mask"].unsqueeze(0).expand(n, -1)
        result = self.model(source, context, mask, clip=True)
        target_low = self.model.pyramid.coarsest(reference)
        pair = _per_image(torch.sqrt((result["corrected_low"] - target_low).square() + 1e-6))
        ssim = _ssim_loss(result["corrected_low"], target_low)
        residual = _per_image(result["residual"].abs())
        field = result["residual"]
        tv = _per_image((field[:, :, 1:] - field[:, :, :-1]).abs())
        tv = tv + _per_image((field[:, :, :, 1:] - field[:, :, :, :-1]).abs())
        second_context, _ = self.model.correct_context_set(result["context_low"], mask)
        second_low, _ = self.model.correct_coarse(
            result["corrected_low"], second_context, mask
        )
        idem = _per_image((second_low - result["corrected_low"]).abs())
        weights = batch["q_reg"]
        terms = {
            "pair_low": _weighted(pair, weights),
            "ssim_low": _weighted(ssim, weights),
            "residual": residual.mean(),
            "smooth": tv.mean(),
            "idempotence": idem.mean(),
        }
        terms["total"] = sum(
            float(getattr(self.cfg.loss, name)) * value for name, value in terms.items()
        )
        with torch.no_grad():
            # Ignore sub-1e-6 synthesis round-off, matching the copied-band
            # float32 safety tolerance in the Exp-06 contract.
            terms["clip_fraction"] = (
                (result["preclip"] < -1.0 - 1e-6)
                | (result["preclip"] > 1.0 + 1e-6)
            ).float().mean()
            terms["raw_pair_low"] = _weighted(
                _per_image((result["source_low"] - target_low).abs()), weights
            )
        return terms

    def _step(self, batch, stage):
        terms = self.losses(batch)
        scanner = str(batch["scanner"])
        for name, value in terms.items():
            self.log(f"{stage}/{name}", value, on_step=stage == "train", on_epoch=True,
                     prog_bar=name == "total", batch_size=batch["source"].shape[0])
        self.log(f"{stage}/{scanner}_pair_low", terms["pair_low"], on_step=False,
                 on_epoch=True, batch_size=batch["source"].shape[0])
        return terms["total"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=float(self.cfg.optim.lr),
            betas=tuple(self.cfg.optim.betas), weight_decay=float(self.cfg.optim.weight_decay)
        )
