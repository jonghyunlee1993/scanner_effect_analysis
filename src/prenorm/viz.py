"""Matplotlib renderers for v3 canonicalization diagnostics."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import ssim


COLUMNS = ("Source", "AT2 target", "Canonical", "Target render")


def rgb01(tensor):
    """Convert CHW RGB in [-1, 1] to an HWC NumPy image in [0, 1]."""
    return ((tensor.clamp(-1, 1) + 1) * 0.5).permute(1, 2, 0).cpu().numpy()


def fig_to_array(figure):
    """Rasterize a Matplotlib figure as an RGB uint8 array."""
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    rgba = np.frombuffer(figure.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return rgba[:, :, :3].copy()


def scanner_grid(source, reference, canonical, rendered, scanners, title):
    """Show one registered coordinate across all scanners and v3 output paths."""
    scanner_count = source.shape[0]
    canonical_ssim = [
        float(ssim(canonical[i : i + 1], reference[None])) for i in range(scanner_count)
    ]
    renderer_ssim = [
        float(ssim(rendered[i : i + 1], source[i : i + 1])) for i in range(scanner_count)
    ]

    figure, axes = plt.subplots(
        scanner_count,
        len(COLUMNS),
        figsize=(10.5, scanner_count * 2.4),
        squeeze=False,
    )
    for row in range(scanner_count):
        images = (source[row], reference, canonical[row], rendered[row])
        for column, image in enumerate(images):
            axes[row, column].imshow(rgb01(image))
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(COLUMNS[column])
        axes[row, 0].set_ylabel(
            f"{scanners[row]}\ncanonical SSIM={canonical_ssim[row]:.2f}"
            f"\nrender SSIM={renderer_ssim[row]:.2f}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=50,
        )
    figure.suptitle(title)
    figure.tight_layout(rect=(0.08, 0.0, 1.0, 0.96))
    return figure


def input_channels(channels, names, title):
    """Show one differentiable input-builder output in channel order."""
    figure, axes = plt.subplots(1, len(names), figsize=(2.2 * len(names), 2.6), squeeze=False)
    for index, name in enumerate(names):
        axis = axes[0, index]
        axis.imshow(channels[index].detach().cpu(), cmap="gray", vmin=-1 if index < 3 else 0, vmax=1)
        axis.set_title(name)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    return figure
