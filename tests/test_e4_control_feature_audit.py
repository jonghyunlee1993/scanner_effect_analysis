import numpy as np

from audit_e4_control_features import audit_reference
from e4_control_population import CONTROL_SPECS, condition_names
from fetch_e0_pfm_checkpoints import sha256


def test_audit_population_constants():
    assert len(condition_names()) == 9
    assert 4 * 109 * len(CONTROL_SPECS) * 600 == 2_354_400


def test_reference_audit_reads_frozen_sufficient_statistics(tmp_path):
    path = tmp_path / "fov_224.npz"
    np.savez_compressed(
        path,
        band_sum_0=np.zeros((3, 224, 224)),
        band_sum_1=np.zeros((3, 112, 112)),
        band_sum_2=np.zeros((3, 56, 56)),
        band_sum_3=np.zeros((3, 28, 28)),
        image_count=np.asarray(65400),
        slide_ids=np.asarray([f"s{index}" for index in range(109)]),
    )
    summary = {
        "reference_gate_pass": True,
        "control_version": "e4_four_pfm_controls_v1",
        "output_sha256": sha256(path),
    }
    assert audit_reference(path, summary, 224)["pass"]
    summary["output_sha256"] = "changed"
    assert not audit_reference(path, summary, 224)["pass"]
