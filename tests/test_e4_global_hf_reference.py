import json

import numpy as np

from build_e4_global_hf_reference import existing_output_passes, selected_fov
from fetch_e0_pfm_checkpoints import sha256


def test_fov_selection():
    assert selected_fov(None, 0) == 224
    assert selected_fov(None, 1) == 256
    assert selected_fov(None, 2) == 512
    assert selected_fov(256, None) == 256


def test_reference_reuse_gate(tmp_path):
    output = tmp_path / "fov_224.npz"
    np.savez_compressed(output, band_sum_0=np.zeros((3, 4, 4)))
    summary = {
        "analysis": "e4_global_hf_reference",
        "control_version": "e4_four_pfm_controls_v1",
        "fov": 224,
        "pyramid_levels": 4,
        "grid_audit_sha256": "audit",
        "output_sha256": sha256(output),
        "reference_gate_pass": True,
    }
    summary_path = tmp_path / "fov_224.summary.json"
    summary_path.write_text(json.dumps(summary))
    assert existing_output_passes(output, summary_path, "audit", 224)
    assert not existing_output_passes(output, summary_path, "changed", 224)

