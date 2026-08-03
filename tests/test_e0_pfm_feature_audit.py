import hashlib
import json

import h5py
import numpy as np
import pandas as pd

from audit_e0_pfm_features import audit_feature_shard, build_feature_index
from extract_e0_pfm_features import FEATURE_VERSION, SCANNERS


def write_fixture(root):
    grid = root / "s1.h5"
    with h5py.File(grid, "w") as store:
        store.create_dataset("scanner", data=np.asarray(SCANNERS, dtype="S"))
        store.create_dataset("location_id", data=np.arange(100))
        store.create_dataset("replicate_id", data=np.repeat(np.arange(5), 20))
        store.create_dataset("canonical_center_x", data=np.arange(100) + 1000)
        store.create_dataset("canonical_center_y", data=np.arange(100) + 2000)
    grid.with_suffix(".summary.json").write_text(json.dumps({"output_sha256": "source-hash"}))

    feature = root / "feature.h5"
    rng = np.random.default_rng(4)
    with h5py.File(feature, "w") as store:
        store.create_dataset("features", data=rng.normal(size=(6, 100, 2)).astype(np.float32))
        with h5py.File(grid, "r") as source:
            for name in ("scanner", "location_id", "replicate_id", "canonical_center_x", "canonical_center_y"):
                store.create_dataset(name, data=source[name][:])
        store.attrs["analysis"] = "e0_pfm_feature_shard"
        store.attrs["feature_version"] = FEATURE_VERSION
        store.attrs["slide_id"] = "s1"
        store.attrs["encoder_id"] = "resnet50"
        store.attrs["feature_dim"] = 2
        store.attrs["native_fov_px"] = 256
        store.attrs["checkpoint_sha256"] = "checkpoint"
        store.attrs["source_grid_sha256"] = "source-hash"
    digest = hashlib.sha256(feature.read_bytes()).hexdigest()
    feature.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "analysis": "e0_pfm_feature_shard",
                "feature_version": FEATURE_VERSION,
                "slide_id": "s1",
                "encoder_id": "resnet50",
                "feature_dim": 2,
                "native_fov_px": 256,
                "checkpoint_sha256": "checkpoint",
                "source_grid_sha256": "source-hash",
                "output_sha256": digest,
                "shard_gate_pass": True,
            }
        )
    )
    return grid, feature


def test_feature_shard_checks_identity_finiteness_and_variance(tmp_path):
    grid, feature = write_fixture(tmp_path)
    model = {
        "encoder_id": "resnet50",
        "feature_dim": 2,
        "native_fov_px": 256,
        "checkpoint_sha256": "checkpoint",
    }
    result = audit_feature_shard(model, grid, feature, verify_sha256=True)
    assert result["status"] == "pass"
    assert result["features"] == 600
    assert result["total_variance"] > 0


def test_feature_index_has_one_row_per_scanner_location(tmp_path):
    rows = []
    for scanner in SCANNERS:
        for location_id in range(100):
            rows.append(
                {
                    "slide_id": f"s{location_id // 100}",
                    "tissue_type": "normal",
                    "scanner": scanner,
                    "location_id": location_id,
                    "replicate_id": location_id // 20,
                    "canonical_center_x": location_id,
                    "canonical_center_y": location_id + 1,
                    "pixel_source": "native_wsi_only",
                }
            )
    geometry = pd.DataFrame(rows)
    repeated = pd.concat([geometry.assign(slide_id=f"s{index}") for index in range(109)])
    models = [{"encoder_id": "resnet50", "feature_dim": 2, "native_fov_px": 256}]
    index = build_feature_index(repeated, models, tmp_path)
    assert len(index) == 65_400
    assert index["scanner_index"].min() == 0
    assert index["scanner_index"].max() == 5
