"""Validation image logging for source, anchor, canonical, and renderer outputs."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from prenorm import viz


class ImageLoggingCallback(pl.Callback):
    """Write fixed registered diagnostic grids to disk and Weights & Biases."""

    def __init__(self, cfg):
        self.every = int(cfg.image_log.every_n_epochs)
        self.sample_dir = Path(cfg.paths.sample_dir)
        self.reference_scanner = cfg.reference_scanner

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or pl_module._reference_batch is None:
            return
        if trainer.current_epoch % self.every:
            return

        batch = pl_module._reference_batch
        scanners = list(pl_module.model.scanners)
        reference_index = scanners.index(self.reference_scanner)
        epoch_dir = self.sample_dir / f"epoch{trainer.current_epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        panels = []
        for index, tuple_rgb in enumerate(batch["rgb"]):
            source = tuple_rgb.to(pl_module.device)
            targets = torch.arange(len(scanners), device=pl_module.device)
            with torch.no_grad():
                canonical = pl_module.canonicalize(source)
                rendered = pl_module.render(source, targets)
            figure = viz.scanner_grid(
                source.cpu(),
                source[reference_index].cpu(),
                canonical.cpu(),
                rendered.cpu(),
                scanners,
                f"epoch {trainer.current_epoch} — location #{index}",
            )
            image = viz.fig_to_array(figure)
            plt.close(figure)
            cv2.imwrite(str(epoch_dir / f"grid_loc{index}.png"), image[:, :, ::-1])
            panels.append(image)

        experiment = getattr(trainer.logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "log"):
            import wandb

            experiment.log(
                {"val_panels": [wandb.Image(panel, caption=f"location #{i}")
                                for i, panel in enumerate(panels)]}
            )


class MilestoneCheckpoint(pl.Callback):
    """Save exact human-numbered epoch milestones in addition to best/last."""

    def __init__(self, directory, epochs):
        self.directory = Path(directory)
        self.epochs = {int(epoch) for epoch in epochs}
        if not self.epochs or min(self.epochs) < 1:
            raise ValueError("milestone epochs must be positive")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        completed_epochs = int(trainer.current_epoch) + 1
        if completed_epochs not in self.epochs:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"epoch{completed_epochs:03d}.ckpt"
        trainer.save_checkpoint(str(path))
        print(f"[checkpoint] saved milestone after epoch {completed_epochs}: {path}")
