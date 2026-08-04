import json

import numpy as np

from e4_control_population import condition_names
from extract_e4_control_features import existing_output_passes, infer_condition
from fetch_e0_pfm_checkpoints import sha256


class FakeEncoder:
    precision = __import__("torch").float32
    eval_transforms = staticmethod(
        lambda image: __import__("torch").from_numpy(np.asarray(image).copy())
        .permute(2, 0, 1)
        .float()
    )

    def __call__(self, batch):
        return batch.mean(dim=(2, 3))


def test_infer_condition_preserves_scanner_location_shape(monkeypatch):
    import torch

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    images = np.arange(6 * 100 * 8 * 8 * 3, dtype=np.uint32)
    images = (images % 255).astype(np.uint8).reshape(6, 100, 8, 8, 3)
    output = infer_condition(FakeEncoder(), images, batch_size=31, feature_dim=3)
    assert output.shape == (6, 100, 3)
    assert np.isfinite(output).all()


def test_control_shard_reuse_gate(tmp_path):
    output = tmp_path / "slide.h5"
    output.write_bytes(b"feature shard")
    model = {"encoder_id": "uni_v1", "checkpoint_sha256": "checkpoint"}
    source = {"output_sha256": "source"}
    reference = {"output_sha256": "reference"}
    summary = {
        "analysis": "e4_control_feature_shard",
        "control_version": "e4_four_pfm_controls_v1",
        "encoder_id": "uni_v1",
        "checkpoint_sha256": "checkpoint",
        "source_grid_sha256": "source",
        "global_reference_sha256": "reference",
        "conditions": list(condition_names()),
        "output_sha256": sha256(output),
        "shard_gate_pass": True,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary))
    assert existing_output_passes(output, model, source, reference)
    reference["output_sha256"] = "changed"
    assert not existing_output_passes(output, model, source, reference)

