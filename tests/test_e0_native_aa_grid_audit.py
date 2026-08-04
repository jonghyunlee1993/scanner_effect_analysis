import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import audit_e0_native_aa_grid as audit_module
from audit_e0_native_aa_grid import audit_grid
from render_e0_native_aa_shard import MODEL_FOV, RENDER_VERSION, SCANNERS


def make_geometry():
    rows = []
    for scanner_index, scanner in enumerate(SCANNERS):
        for location_id in range(100):
            rows.append(
                {
                    "slide_id": "s1",
                    "scanner": scanner,
                    "location_id": location_id,
                    "replicate_id": location_id // 10,
                    "canonical_center_x": 1000 + location_id,
                    "canonical_center_y": 2000 + location_id,
                    "total_target_dx": scanner_index,
                    "total_target_dy": -scanner_index,
                    "native_path": f"/{scanner}.svs",
                    "native_to_target_m00": 1.0,
                    "native_to_target_m01": 0.0,
                    "native_to_target_m02": 0.0,
                    "native_to_target_m10": 0.0,
                    "native_to_target_m11": 1.0,
                    "native_to_target_m12": 0.0,
                }
            )
    return pd.DataFrame(rows)


def write_shard(root: Path, geometry: pd.DataFrame):
    path = root / "s1.h5"
    with h5py.File(path, "w") as store:
        store.create_dataset("rgb", data=np.ones((6, 100, 2, 2, 3), dtype=np.uint8))
        store.create_dataset("scanner", data=np.asarray(SCANNERS, dtype="S"))
        store.create_dataset("location_id", data=np.arange(100))
        first = geometry[geometry.scanner.eq("at2")].sort_values("location_id")
        for name in ("replicate_id", "canonical_center_x", "canonical_center_y"):
            store.create_dataset(name, data=first[name].to_numpy())
        store.create_dataset("native_path", data=np.asarray([f"/{s}.svs" for s in SCANNERS], dtype="S"))
        store.create_dataset("native_to_target", data=np.tile(np.eye(3), (6, 1, 1)))
        x = np.asarray([[1000 + i + s - 256 for i in range(100)] for s in range(6)])
        y = np.asarray([[2000 + i - s - 256 for i in range(100)] for s in range(6)])
        store.create_dataset("target_top_left_x", data=x)
        store.create_dataset("target_top_left_y", data=y)
        store.attrs["analysis"] = "e0_native_aa_grid_shard"
        store.attrs["render_version"] = RENDER_VERSION
        store.attrs["slide_id"] = "s1"
        store.attrs["pixel_source"] = "native_wsi_only"
        store.attrs["historical_registered_rgb_used"] = False
        store.attrs["fov"] = 512
        store.attrs["model_fov_json"] = json.dumps(MODEL_FOV)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "analysis": "e0_native_aa_grid_shard",
                "render_version": RENDER_VERSION,
                "slide_id": "s1",
                "pixel_source": "native_wsi_only",
                "output_sha256": digest,
            }
        )
    )
    return path


def test_complete_grid_passes_with_geometry_and_pixel_checks(tmp_path, monkeypatch):
    geometry = make_geometry()
    write_shard(tmp_path, geometry)
    original_assert_equal = audit_module.assert_equal

    def allow_tiny_test_rgb(observed, expected, label):
        if label != "RGB shape":
            original_assert_equal(observed, expected, label)

    monkeypatch.setattr(audit_module, "assert_equal", allow_tiny_test_rgb)
    audit, summary = audit_grid(geometry, tmp_path, verify_sha256=True)
    assert audit.status.tolist() == ["pass"]
    assert summary["grid_gate_pass"] is True


def test_grid_reports_missing_shard(tmp_path):
    _, summary = audit_grid(make_geometry(), tmp_path, verify_sha256=False)
    assert summary["grid_gate_pass"] is False
    assert summary["missing_shards"] == ["s1.h5"]
