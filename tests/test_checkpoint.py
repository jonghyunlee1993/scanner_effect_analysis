import copy

import pytest
import torch

from prenorm.checkpoint import load_model_for_inference, read_checkpoint
from prenorm.module import Phase1Module
from tests.helpers import make_config


def test_strict_checkpoint_roundtrip(tmp_path):
    cfg = make_config(tmp_path)
    module = Phase1Module(cfg).eval()
    path = tmp_path / "model.ckpt"
    torch.save({
        "state_dict": module.state_dict(),
        "prenorm_metadata": module.checkpoint_metadata(),
    }, path)
    model = load_model_for_inference(path, cfg)
    rgb = torch.rand(1, 3, 32, 32) * 2 - 1
    assert torch.allclose(model.canonicalize(rgb), module.canonicalize(rgb))


def test_versionless_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "old.ckpt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError, match="versionless"):
        read_checkpoint(path)


def test_scanner_order_mismatch_is_rejected(tmp_path):
    cfg = make_config(tmp_path)
    module = Phase1Module(cfg)
    path = tmp_path / "model.ckpt"
    metadata = copy.deepcopy(module.checkpoint_metadata())
    metadata["scanners"] = list(reversed(metadata["scanners"]))
    torch.save({"state_dict": module.state_dict(), "prenorm_metadata": metadata}, path)
    with pytest.raises(ValueError, match="scanners"):
        load_model_for_inference(path, cfg)
