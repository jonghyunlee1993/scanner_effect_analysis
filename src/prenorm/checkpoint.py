"""Strict versioned checkpoint loading for training resume and inference."""
from __future__ import annotations

from pathlib import Path

import torch

from .module import CHECKPOINT_FORMAT_VERSION, Phase1Module


def read_checkpoint(path, map_location="cpu"):
    """Read a Lightning checkpoint and reject pre-v3 formats immediately."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    metadata = checkpoint.get("prenorm_metadata")
    if metadata is None:
        raise ValueError("versionless and v2 checkpoints are not supported")
    if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {metadata.get('format_version')}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    return checkpoint


def validate_checkpoint_metadata(actual, expected):
    """Require semantic fields that can silently change model interpretation."""
    fields = (
        "format_version",
        "design_version",
        "scanners",
        "reference_scanner",
        "input_mode",
        "channel_order",
        "stain_reference_hash",
        "style_dim",
        "alpha",
        "phase",
    )
    mismatches = [field for field in fields if actual.get(field) != expected.get(field)]
    if mismatches:
        details = ", ".join(
            f"{field}: checkpoint={actual.get(field)!r}, config={expected.get(field)!r}"
            for field in mismatches
        )
        raise ValueError(f"checkpoint metadata mismatch: {details}")


def validate_resume_checkpoint(path, cfg):
    """Validate a resume checkpoint before Lightning restores optimizer state."""
    checkpoint = read_checkpoint(path)
    module = Phase1Module(cfg)
    validate_checkpoint_metadata(checkpoint["prenorm_metadata"], module.checkpoint_metadata())
    return checkpoint


def load_model_for_inference(path, cfg, map_location="cpu"):
    """Strictly load v3 weights and return the label-free canonicalizer core."""
    checkpoint = read_checkpoint(path, map_location=map_location)
    module = Phase1Module(cfg)
    validate_checkpoint_metadata(checkpoint["prenorm_metadata"], module.checkpoint_metadata())
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    return module.model.eval()
