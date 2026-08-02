"""Build one slide's six-scanner, 512 px-aware feature-location manifest.

Original Exp05 locations are retained when they pass the frozen current-route geometry
contract and the maximum model FOV (CONCH v1, 512 px). Failed slots are replaced from the
pre-feature intersection of all six curated coordinate pools in deterministic random order.
No PFM features or correction outcomes are read during selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import align, qc


SCANNERS = ("gt450", "versa", "akoya", "s60", "s360")
ALL_SCANNERS = ("at2",) + SCANNERS
PATCH_SIZE = 256
MAX_FOV = 512
SEARCH_MARGIN = 96
REFINE_RADIUS = 16
PROBE_MIN_NCC = 0.60
TILE_MIN_NCC = 0.50
MAX_DISTANCE_FROM_PRIOR = 48
MAX_RESIDUAL_SHIFT = 8.0
MAX_PAD_FRACTION = 0.01
SELECTION_SEED = 20260802


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-id", required=True)
    parser.add_argument(
        "--data-root",
        default="/mnt/isilon/oldridge_lab/batch_effects/pan_normal/registered_ref_at2_all",
    )
    parser.add_argument(
        "--registry",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/"
            "registered_ref_at2_all/registered_images/qc_registered_eval/"
            "registered_pair_summary.csv"
        ),
    )
    parser.add_argument(
        "--selected-patches",
        default="outputs/exp05_spectral_cohort_109/selected_patches.csv",
    )
    parser.add_argument(
        "--alignment-root",
        default="outputs/e0_registration_cohort_109/shards",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/e0_feature_manifest_109/shards",
    )
    parser.add_argument("--target-locations", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=3000)
    return parser.parse_args()


def curated_path(root: Path, scanner: str, slide_id: str):
    return (
        root
        / "registered_curated_features"
        / scanner
        / "20x_256px_0px_overlap"
        / "patches"
        / f"{slide_id}_patches.h5"
    )


def common_curated_coords(root: Path, slide_id: str):
    coordinate_sets = []
    for scanner in ALL_SCANNERS:
        path = curated_path(root, scanner, slide_id)
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            coords = np.asarray(handle["coords"], dtype=np.int64)
        coordinate_sets.append({(int(x), int(y)) for x, y in coords})
    common = sorted(set.intersection(*coordinate_sets))
    if not common:
        raise ValueError(f"no six-scanner common coordinate pool: {slide_id}")
    return np.asarray(common, dtype=np.int64)


def stable_rng(slide_id: str):
    digest = hashlib.sha256(f"{SELECTION_SEED}|{slide_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def fov_in_bounds(x, y, dx, dy, width, height, fov=MAX_FOV):
    center_x, center_y = int(x) + PATCH_SIZE // 2, int(y) + PATCH_SIZE // 2
    left = center_x - fov // 2 + int(dx)
    top = center_y - fov // 2 + int(dy)
    return left >= 0 and top >= 0 and left + fov <= width and top + fov <= height


def read_patch(handle, x, y, size):
    return np.asarray(
        handle.read_region((int(x), int(y)), 0, (int(size), int(size))).convert("RGB"),
        dtype=np.uint8,
    )


def read_fov(handle, x, y, dx, dy, fov=MAX_FOV):
    center_x, center_y = int(x) + PATCH_SIZE // 2, int(y) + PATCH_SIZE // 2
    return read_patch(
        handle,
        center_x - fov // 2 + int(dx),
        center_y - fov // 2 + int(dy),
        fov,
    )


def fit_slide_priors(alignment: pd.DataFrame):
    priors = {}
    rows = []
    for scanner in SCANNERS:
        frame = alignment[
            alignment["scanner"].eq(scanner)
            & alignment["global_ncc"].ge(PROBE_MIN_NCC)
            & ~alignment["global_boundary"]
        ]
        if frame.empty:
            median = (0, 0)
            coefficient = None
            info = {
                "n": 0,
                "r2_dy": np.nan,
                "r2_dx": np.nan,
                "resid_std_dy": np.nan,
                "resid_std_dx": np.nan,
                "affine": False,
            }
        else:
            median = (
                int(np.median(frame["global_dy"])),
                int(np.median(frame["global_dx"])),
            )
            probe = frame[["x", "y", "global_dy", "global_dx", "global_ncc"]].to_numpy(
                dtype=float
            )
            coefficient, info = align.fit_offset_field(probe, min_n=12, min_r2=0.5)
        priors[scanner] = (median, coefficient)
        rows.append(
            {
                "scanner": scanner,
                "n_confident": int(len(frame)),
                "median_dy": median[0],
                "median_dx": median[1],
                "model": "affine" if info["affine"] else "constant",
                **info,
            }
        )
    return priors, pd.DataFrame(rows)


def choose_integer_crop(ext, reference, x, y, prior_contract):
    median, coefficient = prior_contract
    prior_dy, prior_dx = align.prior_at(x, y, median, coefficient)
    prior_dy = int(np.clip(prior_dy, -SEARCH_MARGIN + REFINE_RADIUS, SEARCH_MARGIN - REFINE_RADIUS))
    prior_dx = int(np.clip(prior_dx, -SEARCH_MARGIN + REFINE_RADIUS, SEARCH_MARGIN - REFINE_RADIUS))
    nmap = align.ncc_map(ext, reference)
    global_dy, global_dx, global_ncc = align.peak(nmap, SEARCH_MARGIN)
    global_boundary = bool(
        abs(global_dy) >= SEARCH_MARGIN or abs(global_dx) >= SEARCH_MARGIN
    )
    global_far = bool(
        abs(global_dy - prior_dy) + abs(global_dx - prior_dx)
        > MAX_DISTANCE_FROM_PRIOR
    )
    if global_ncc >= TILE_MIN_NCC and not global_far:
        dy, dx, ncc = global_dy, global_dx, global_ncc
        mode = "global"
        boundary = global_boundary
    else:
        dy, dx, ncc = align.peak(
            nmap,
            SEARCH_MARGIN,
            around=(prior_dy, prior_dx),
            radius=REFINE_RADIUS,
        )
        mode = "local"
        boundary = bool(
            abs(dy) >= SEARCH_MARGIN
            or abs(dx) >= SEARCH_MARGIN
            or abs(dy - prior_dy) >= REFINE_RADIUS
            or abs(dx - prior_dx) >= REFINE_RADIUS
        )
    crop = align.crop_at(ext, dy, dx, PATCH_SIZE, SEARCH_MARGIN)
    quality = qc.registration_quality(reference, crop, sigma=3.0)
    content, pad = align.content_ok(crop, reference, max_pad=MAX_PAD_FRACTION)
    geometry = bool(
        content and not boundary and quality["residual_shift"] <= MAX_RESIDUAL_SHIFT
    )
    return {
        "dy": int(dy),
        "dx": int(dx),
        "ncc": float(ncc),
        "mode": mode,
        "boundary": boundary,
        "global_dy": int(global_dy),
        "global_dx": int(global_dx),
        "global_ncc": float(global_ncc),
        "global_boundary": global_boundary,
        "global_far_from_prior": global_far,
        "residual_shift": quality["residual_shift"],
        "q_reg": quality["q_reg"],
        "pad_256": pad,
        "content_256": content,
        "geometry_pass": geometry,
    }


def flatten_scanner(record, scanner, metrics):
    for key, value in metrics.items():
        record[f"{scanner}_{key}"] = value


def fov_checks(record, handles, reference_fov, x, y, width, height):
    if align.is_blank(reference_fov):
        return False, "reference_512_blank"
    record["reference_tissue_fraction_512"] = qc.tissue_fraction(reference_fov)
    record["reference_stained_fraction_512"] = float(
        align.stained_mask(reference_fov).mean()
    )
    for scanner in SCANNERS:
        dx = int(record[f"{scanner}_dx"])
        dy = int(record[f"{scanner}_dy"])
        if not fov_in_bounds(x, y, dx, dy, width, height):
            record[f"{scanner}_fov_512_pass"] = False
            record[f"{scanner}_pad_512"] = np.nan
            return False, f"{scanner}_512_out_of_bounds"
        crop = read_fov(handles[scanner], x, y, dx, dy)
        content, pad = align.content_ok(crop, reference_fov, max_pad=MAX_PAD_FRACTION)
        record[f"{scanner}_fov_512_pass"] = bool(content)
        record[f"{scanner}_pad_512"] = float(pad)
        if not content:
            return False, f"{scanner}_512_content"
    return True, "pass"


def evaluate_original(row, alignment_lookup, handles, width, height):
    x, y, patch_index = int(row.x), int(row.y), int(row.patch_index)
    record = {
        "x": x,
        "y": y,
        "source_patch_index": patch_index,
        "source_kind": "original",
        "candidate_rank": -1,
    }
    if not fov_in_bounds(x, y, 0, 0, width, height):
        return record, False, "reference_512_out_of_bounds"
    for scanner in SCANNERS:
        metrics = alignment_lookup[(patch_index, scanner)]
        values = {
            "dy": int(metrics["corrected_dy"]),
            "dx": int(metrics["corrected_dx"]),
            "ncc": float(metrics["corrected_ncc"]),
            "mode": str(metrics["corrected_mode"]),
            "boundary": bool(metrics["corrected_boundary"]),
            "global_dy": int(metrics["global_dy"]),
            "global_dx": int(metrics["global_dx"]),
            "global_ncc": float(metrics["global_ncc"]),
            "global_boundary": bool(metrics["global_boundary"]),
            "global_far_from_prior": bool(metrics["global_far_from_prior"]),
            "residual_shift": float(metrics["residual_shift"]),
            "q_reg": float(metrics["q_reg"]),
            "pad_256": float(metrics["pad_fraction"]),
            "content_256": bool(metrics["content_ok"]),
            "geometry_pass": bool(metrics["geometry_pass"]),
        }
        flatten_scanner(record, scanner, values)
        if not values["geometry_pass"]:
            return record, False, f"{scanner}_geometry"
    reference_fov = read_fov(handles["at2"], x, y, 0, 0)
    return record, *fov_checks(record, handles, reference_fov, x, y, width, height)


def evaluate_candidate(x, y, candidate_rank, priors, handles, width, height):
    x, y = int(x), int(y)
    record = {
        "x": x,
        "y": y,
        "source_patch_index": np.nan,
        "source_kind": "replacement_candidate",
        "candidate_rank": int(candidate_rank),
    }
    if not fov_in_bounds(x, y, 0, 0, width, height):
        return record, False, "reference_512_out_of_bounds"
    reference = read_patch(handles["at2"], x, y, PATCH_SIZE)
    reference_fov = read_fov(handles["at2"], x, y, 0, 0)
    if align.is_blank(reference) or align.is_blank(reference_fov):
        return record, False, "reference_blank"
    for scanner in SCANNERS:
        ext = read_patch(
            handles[scanner],
            x - SEARCH_MARGIN,
            y - SEARCH_MARGIN,
            PATCH_SIZE + 2 * SEARCH_MARGIN,
        )
        metrics = choose_integer_crop(ext, reference, x, y, priors[scanner])
        flatten_scanner(record, scanner, metrics)
        if not metrics["geometry_pass"]:
            return record, False, f"{scanner}_geometry"
    return record, *fov_checks(record, handles, reference_fov, x, y, width, height)


def main():
    args = parse_args()
    slide_id = str(args.slide_id)
    registry = pd.read_csv(args.registry, dtype={"slide_id": str})
    selected_registry = registry[registry["slide_id"].eq(slide_id)]
    if len(selected_registry) != 1:
        raise ValueError(f"expected one registry row for {slide_id}")
    registry_row = selected_registry.iloc[0]
    selected = pd.read_csv(args.selected_patches, dtype={"slide_id": str})
    selected = selected[selected["slide_id"].eq(slide_id)].sort_values("patch_index")
    if len(selected) != args.target_locations:
        raise ValueError(f"{slide_id}: expected {args.target_locations} original locations")
    alignment_path = Path(args.alignment_root) / slide_id / "alignment_patch_metrics.csv"
    alignment_frame = pd.read_csv(alignment_path, dtype={"slide_id": str})
    alignment_frame = alignment_frame[alignment_frame["branch"].eq("current")]
    if len(alignment_frame) != args.target_locations * len(SCANNERS):
        raise ValueError(f"{slide_id}: incomplete current-route alignment shard")
    alignment_lookup = {
        (int(row.patch_index), str(row.scanner)): row._asdict()
        for row in alignment_frame.itertuples(index=False)
    }
    priors, prior_frame = fit_slide_priors(alignment_frame)

    handles = {
        scanner: openslide.OpenSlide(str(registry_row[f"{scanner}_path"]))
        for scanner in ALL_SCANNERS
    }
    dimensions = {scanner: handle.dimensions for scanner, handle in handles.items()}
    if len(set(dimensions.values())) != 1:
        raise ValueError(f"registered dimensions differ: {slide_id}: {dimensions}")
    width, height = dimensions["at2"]

    audit_rows = []
    retained = []
    failed_slots = []
    try:
        for row in selected.itertuples(index=False):
            record, passed, reason = evaluate_original(
                row, alignment_lookup, handles, width, height
            )
            record.update(
                slide_id=slide_id,
                evaluation_order=len(audit_rows),
                eligible=passed,
                failure_reason=reason,
            )
            audit_rows.append(record)
            if passed:
                record["location_id"] = int(row.patch_index)
                record["replicate_id"] = int(row.patch_index) % 5
                record["retained_original"] = True
                retained.append(record.copy())
            else:
                failed_slots.append(int(row.patch_index))

        common = common_curated_coords(Path(args.data_root), slide_id)
        original_coords = {
            (int(row.x), int(row.y)) for row in selected.itertuples(index=False)
        }
        candidates = np.asarray(
            [coord for coord in common if (int(coord[0]), int(coord[1])) not in original_coords],
            dtype=np.int64,
        )
        order = stable_rng(slide_id).permutation(len(candidates))
        replacements = []
        evaluated_candidates = 0
        for candidate_rank, index_candidate in enumerate(order):
            if len(replacements) == len(failed_slots):
                break
            if evaluated_candidates >= args.max_candidates:
                break
            x, y = candidates[int(index_candidate)]
            record, passed, reason = evaluate_candidate(
                x,
                y,
                candidate_rank,
                priors,
                handles,
                width,
                height,
            )
            evaluated_candidates += 1
            record.update(
                slide_id=slide_id,
                evaluation_order=len(audit_rows),
                eligible=passed,
                failure_reason=reason,
            )
            audit_rows.append(record)
            if passed:
                slot = failed_slots[len(replacements)]
                record["location_id"] = slot
                record["replicate_id"] = slot % 5
                record["retained_original"] = False
                replacements.append(record.copy())
    finally:
        for handle in handles.values():
            handle.close()

    final_rows = retained + replacements
    final = pd.DataFrame(final_rows).sort_values("location_id")
    audit = pd.DataFrame(audit_rows)
    output = Path(args.output_root) / slide_id
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "candidate_audit.csv", index=False)
    prior_frame.insert(0, "slide_id", slide_id)
    prior_frame.to_csv(output / "prior_fit.csv", index=False)
    complete = len(final) == args.target_locations
    manifest_path = output / "feature_manifest.csv"
    if complete:
        final.insert(0, "manifest_version", "e0_integer_512_v1")
        final.insert(1, "tissue_type", "")
        final["center_x"] = final["x"].astype(int) + PATCH_SIZE // 2
        final["center_y"] = final["y"].astype(int) + PATCH_SIZE // 2
        final["max_model_fov_px"] = MAX_FOV
        final["alignment_version"] = "current_global_first_integer_v1"
        final.to_csv(manifest_path, index=False)
    else:
        manifest_path.unlink(missing_ok=True)
    summary = {
        "analysis": "e0_six_scanner_feature_manifest_slide",
        "slide_id": slide_id,
        "complete": complete,
        "target_locations": int(args.target_locations),
        "retained_original": int(len(retained)),
        "replacements_needed": int(len(failed_slots)),
        "replacements_found": int(len(replacements)),
        "common_coordinate_pool": int(len(common)),
        "replacement_candidates_evaluated": int(evaluated_candidates),
        "max_candidates": int(args.max_candidates),
        "frozen_contract": {
            "patch_size": PATCH_SIZE,
            "max_model_fov": MAX_FOV,
            "search_margin": SEARCH_MARGIN,
            "refine_radius": REFINE_RADIUS,
            "max_distance_from_prior": MAX_DISTANCE_FROM_PRIOR,
            "max_residual_shift": MAX_RESIDUAL_SHIFT,
            "max_pad_fraction": MAX_PAD_FRACTION,
            "selection_seed": SELECTION_SEED,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not complete:
        raise RuntimeError(
            f"{slide_id}: found {len(final)}/{args.target_locations} valid 512 px-aware locations"
        )


if __name__ == "__main__":
    main()
