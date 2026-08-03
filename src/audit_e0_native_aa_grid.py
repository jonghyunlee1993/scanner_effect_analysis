"""Audit the complete E0 native anti-aliased RGB grid before PFM inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from render_e0_native_aa_shard import (
    MODEL_FOV,
    RENDER_VERSION,
    SCANNERS,
    affine_from_row,
    target_top_left,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry",
        default="outputs/e0_native_geometry_final/native_geometry_manifest.csv",
    )
    parser.add_argument("--shards", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--output", default="outputs/e0_native_aa_grid/audit")
    parser.add_argument("--skip-sha256", action="store_true")
    return parser.parse_args()


def sha256(path: Path, block_size=8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def strings(values) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def assert_equal(observed, expected, label: str):
    if observed != expected:
        raise ValueError(f"{label}: observed {observed!r}, expected {expected!r}")


def audit_shard(
    geometry: pd.DataFrame,
    slide_id: str,
    shard_path: Path,
    verify_sha256: bool = True,
):
    summary_path = shard_path.with_suffix(".summary.json")
    if not shard_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"{slide_id}: missing HDF5 shard or summary")
    summary = json.loads(summary_path.read_text())
    assert_equal(summary.get("analysis"), "e0_native_aa_grid_shard", "summary analysis")
    assert_equal(summary.get("render_version"), RENDER_VERSION, "summary render version")
    assert_equal(str(summary.get("slide_id")), slide_id, "summary slide")
    assert_equal(summary.get("pixel_source"), "native_wsi_only", "summary pixel source")
    expected_hash = str(summary.get("output_sha256"))
    if len(expected_hash) != 64:
        raise ValueError(f"{slide_id}: invalid summary SHA256")
    if verify_sha256:
        assert_equal(sha256(shard_path), expected_hash, "shard SHA256")

    slide = geometry[geometry["slide_id"].astype(str).eq(slide_id)].copy()
    slide = slide.sort_values(["scanner", "location_id"])
    if len(slide) != 600:
        raise ValueError(f"{slide_id}: geometry does not contain 600 rows")

    black_patches = 0
    white_patches = 0
    with h5py.File(shard_path, "r") as store:
        assert_equal(store.attrs.get("analysis"), "e0_native_aa_grid_shard", "HDF5 analysis")
        assert_equal(store.attrs.get("render_version"), RENDER_VERSION, "HDF5 render version")
        assert_equal(str(store.attrs.get("slide_id")), slide_id, "HDF5 slide")
        assert_equal(store.attrs.get("pixel_source"), "native_wsi_only", "HDF5 pixel source")
        assert_equal(bool(store.attrs.get("historical_registered_rgb_used")), False, "registered RGB flag")
        assert_equal(int(store.attrs.get("fov")), 512, "shared FOV")
        assert_equal(
            json.loads(store.attrs.get("model_fov_json")), MODEL_FOV, "model FOV contract"
        )

        rgb = store["rgb"]
        assert_equal(rgb.shape, (6, 100, 512, 512, 3), "RGB shape")
        assert_equal(rgb.dtype, np.dtype("uint8"), "RGB dtype")
        assert_equal(strings(store["scanner"][:]), list(SCANNERS), "scanner order")
        assert_equal(store["location_id"][:].astype(int).tolist(), list(range(100)), "location IDs")

        first = slide[slide["scanner"].eq("at2")].sort_values("location_id")
        for name, dtype in (
            ("replicate_id", np.int64),
            ("canonical_center_x", np.int64),
            ("canonical_center_y", np.int64),
        ):
            observed = store[name][:].astype(dtype)
            expected = first[name].to_numpy(dtype)
            if not np.array_equal(observed, expected):
                raise ValueError(f"{slide_id}: {name} differs from final geometry")

        native_paths = strings(store["native_path"][:])
        for scanner_index, scanner in enumerate(SCANNERS):
            rows = slide[slide["scanner"].eq(scanner)].sort_values("location_id")
            paths = rows["native_path"].astype(str).unique().tolist()
            assert_equal(paths, [native_paths[scanner_index]], f"{scanner} native path")
            matrices = np.stack([affine_from_row(row) for row in rows.itertuples()])
            if not np.allclose(matrices, store["native_to_target"][scanner_index], atol=1e-10):
                raise ValueError(f"{slide_id}/{scanner}: affine differs from final geometry")
            expected_xy = np.asarray([target_top_left(row, 512) for row in rows.itertuples()])
            if not np.array_equal(store["target_top_left_x"][scanner_index], expected_xy[:, 0]):
                raise ValueError(f"{slide_id}/{scanner}: target x differs from final geometry")
            if not np.array_equal(store["target_top_left_y"][scanner_index], expected_xy[:, 1]):
                raise ValueError(f"{slide_id}/{scanner}: target y differs from final geometry")
            for location_index in range(100):
                patch = rgb[scanner_index, location_index]
                black_patches += int(not np.any(patch))
                white_patches += int(np.all(patch == 255))

    if black_patches or white_patches:
        raise ValueError(
            f"{slide_id}: constant boundary/background patches: "
            f"black={black_patches}, white={white_patches}"
        )
    return {
        "slide_id": slide_id,
        "status": "pass",
        "shard_bytes": int(shard_path.stat().st_size),
        "patches": 600,
        "black_patches": black_patches,
        "white_patches": white_patches,
        "sha256_verified": bool(verify_sha256),
        "error": "",
    }


def audit_grid(geometry: pd.DataFrame, shards: Path, verify_sha256: bool = True):
    slides = sorted(geometry["slide_id"].astype(str).unique())
    expected_files = {f"{slide_id}.h5" for slide_id in slides}
    observed_files = {path.name for path in shards.glob("*.h5")}
    unexpected = sorted(observed_files - expected_files)
    missing = sorted(expected_files - observed_files)
    temporary = sorted(str(path) for path in shards.glob(".*.tmp"))
    rows = []
    for slide_id in slides:
        try:
            rows.append(audit_shard(geometry, slide_id, shards / f"{slide_id}.h5", verify_sha256))
        except Exception as error:
            rows.append(
                {
                    "slide_id": slide_id,
                    "status": "fail",
                    "shard_bytes": 0,
                    "patches": 0,
                    "black_patches": 0,
                    "white_patches": 0,
                    "sha256_verified": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    audit = pd.DataFrame(rows)
    grid_pass = bool(
        not missing
        and not unexpected
        and not temporary
        and len(audit) == len(slides)
        and audit["status"].eq("pass").all()
    )
    summary = {
        "analysis": "e0_native_aa_grid_audit",
        "render_version": RENDER_VERSION,
        "slides_expected": len(slides),
        "slides_passing": int(audit["status"].eq("pass").sum()),
        "slides_failing": int(audit["status"].eq("fail").sum()),
        "shard_bytes": int(audit["shard_bytes"].sum()),
        "patches_expected": len(slides) * 600,
        "patches_observed": int(audit["patches"].sum()),
        "missing_shards": missing,
        "unexpected_shards": unexpected,
        "temporary_files": temporary,
        "pixel_source": "native_wsi_only",
        "grid_gate_pass": grid_pass,
    }
    return audit, summary


def main():
    args = parse_args()
    geometry = pd.read_csv(args.geometry, dtype={"slide_id": str})
    audit, summary = audit_grid(geometry, Path(args.shards), not args.skip_sha256)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "shard_audit.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["grid_gate_pass"]:
        failed = audit[audit["status"].eq("fail")]
        if not failed.empty:
            print(failed[["slide_id", "error"]].to_string(index=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
