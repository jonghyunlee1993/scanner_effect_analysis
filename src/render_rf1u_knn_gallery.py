"""Render one query patch and its nearest neighbours, per condition.

The neighbourhood table reports what fraction of a patch's neighbours share its
scanner. This renders the same computation so it can be looked at: the query, the
patches the representation считает most similar, and which scanner each came from.

Neighbours are read from the already extracted embeddings, so nothing is
re-encoded here; only the handful of patches that end up on screen are rendered.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import h5py
import numpy as np

from analyze_rf1_multiscale import laplacian_pyramid, shared_od_multiscale
from build_rf1m_cell import load_e5_statistics
from build_rf1m_slide_band_energy import fold_lab_statistics
from e4_primary_metrics import l2_normalize
from e5_comparator_population import (
    SCANNERS,
    centered_crop,
    reinhard_lab,
    rgb8_to_rgb01,
    uint8_from_rgb01,
)
from e5_comparator_population import rgb01_to_od
from e5_reinhard_residual_frequency import fold_assignments
from extract_rf1u_features import fold_gains, load_band_energy
from fetch_e0_pfm_checkpoints import sha256
from rf1m_combined import RF1M_SIGMAS
from rf1u_unpaired import RF1U_CONDITION, RF1U_VERSION, source_indices, target_index


KNN_FOV = 256
KNN_MODEL = "resnet50"
KNN_QUERY_SCANNER = "akoya"
NEIGHBOURS = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-audit", default="outputs/e0_native_aa_grid/audit/summary.json")
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--e5-statistics", default="outputs/e5_image_statistics")
    parser.add_argument("--energy", default="outputs/rf1u_multitarget/energy")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--features", default="outputs/rf1u_multitarget/features")
    parser.add_argument("--gallery", default="outputs/rf1u_multitarget/gallery")
    parser.add_argument("--output", default="outputs/rf1u_multitarget/knn_gallery")
    parser.add_argument("--models", nargs="+", default=["resnet50", "uni_v1"])
    parser.add_argument("--panel-px", type=int, default=192)
    parser.add_argument("--query-scanner", default=KNN_QUERY_SCANNER)
    parser.add_argument(
        "--query-selection",
        choices=("typical", "hardest", "fixed"),
        default="typical",
    )
    parser.add_argument("--neighbours", type=int, default=NEIGHBOURS)
    parser.add_argument("--texture-percentile", type=float, default=0.67)
    parser.add_argument("--targets", nargs="+", default=["gt450", "at2"])
    return parser.parse_args()


def png_data_uri(rgb8: np.ndarray, size: int) -> str:
    from PIL import Image

    image = Image.fromarray(rgb8)
    if size and size != image.width:
        image = image.resize((size, size), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def structure_score(rgb01) -> float:
    """Standard deviation of the mid-scale detail band of the mean OD.

    Every location on this cohort clears a stain-fraction threshold, including
    near-empty fields, so score structure instead and draw the query from the
    detailed end where a sharpness change is actually judgeable.
    """
    bands, _ = laplacian_pyramid(rgb01_to_od(rgb01).mean(dim=-1), RF1M_SIGMAS)
    return float(bands[1].std())


def select_query(
    features: np.ndarray, count: int, rule: str, eligible: np.ndarray = None
) -> int:
    """Pick the query patch by a stated rule rather than by eye.

    `typical` takes a patch sitting at the slide's median raw same-scanner
    count, so the figure shows what usually happens rather than an extreme.
    `hardest` takes the most scanner-dominated patch instead. Ties resolve by
    index in both cases, so either choice is reproducible.
    """
    scanners, locations = features.shape[0], features.shape[1]
    unit = l2_normalize(features).reshape(scanners * locations, -1)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    top = np.argsort(-similarity, axis=1)[:, :count]
    scanner_of = np.repeat(np.arange(scanners), locations)
    location_of = np.tile(np.arange(locations), scanners)
    same_scanner = (scanner_of[top] == scanner_of[:, None]).sum(axis=1)
    same_location = (location_of[top] == location_of[:, None]).sum(axis=1)
    if eligible is None:
        eligible = np.ones(len(same_scanner), dtype=bool)
    if not eligible.any():
        raise ValueError("no patch meets the tissue-content requirement")
    penalty = np.where(eligible, 0, len(same_scanner) * 10)
    if rule == "hardest":
        order = np.lexsort(
            (np.arange(len(same_scanner)), same_location, -same_scanner, penalty)
        )
        return int(order[0])
    median = np.median(same_scanner[eligible])
    order = np.lexsort(
        (
            np.arange(len(same_scanner)),
            -same_location,
            np.abs(same_scanner - median),
            penalty,
        )
    )
    return int(order[0])


def top_neighbours(features: np.ndarray, query: int, count: int):
    """Indices and cosines of the closest patches, excluding the query itself."""
    unit = l2_normalize(features).reshape(features.shape[0] * features.shape[1], -1)
    similarity = unit @ unit[query]
    similarity[query] = -np.inf
    order = np.argsort(-similarity)[:count]
    return [(int(index), float(similarity[index])) for index in order]


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("KNN gallery rendering requires a CUDA device")
    audit_sha = sha256(Path(args.grid_audit))
    _, _, statistics = load_e5_statistics(Path(args.e5_statistics), KNN_FOV, audit_sha)
    slide_ids = [str(value) for value in statistics["slide_ids"]]
    assignments = fold_assignments(slide_ids)
    lab_mean, lab_std = fold_lab_statistics(statistics, slide_ids, assignments)

    gallery = json.loads((Path(args.gallery) / "summary.json").read_text())
    slide_id = gallery["slide_id"]
    location = int(gallery["location"])
    fold = assignments[slide_id]
    with h5py.File(
        Path(args.raw) / args.models[0] / "shards" / f"{slide_id}.h5", "r"
    ) as source:
        raw_features = np.asarray(source["features"][:], dtype=np.float32)
    grid_root = Path(args.grid)
    mean_tensor = torch.as_tensor(lab_mean[fold], dtype=torch.float32, device="cuda")
    std_tensor = torch.as_tensor(lab_std[fold], dtype=torch.float32, device="cuda")

    def raw_patch(scanner_index: int, position: int):
        with h5py.File(grid_root / f"{slide_id}.h5", "r") as source:
            return rgb8_to_rgb01(
                centered_crop(source["rgb"][scanner_index, position : position + 1], KNN_FOV),
                device="cuda",
            )

    with torch.inference_mode():
        content = np.asarray(
            [
                structure_score(raw_patch(scanner_index, position))
                for scanner_index in range(len(SCANNERS))
                for position in range(100)
            ]
        )
    threshold = float(np.quantile(content, args.texture_percentile))
    eligible = content >= threshold
    print(
        f"structure filter: {int(eligible.sum())}/{len(eligible)} patches at or above "
        f"the {args.texture_percentile:.0%} percentile ({threshold:.4f})",
        flush=True,
    )
    if args.query_selection == "fixed":
        query_index = SCANNERS.index(args.query_scanner) * 100 + location
    else:
        query_index = select_query(
            raw_features, args.neighbours, args.query_selection, eligible
        )
    query_scanner, location = divmod(query_index, 100)
    print(
        f"query = {SCANNERS[query_scanner]} @ {slide_id} loc {location} "
        f"(fold {fold}, models {args.models}, selection {args.query_selection})",
        flush=True,
    )


    conditions = [("raw", None, {args.models[0]: raw_features})]
    per_model_raw = {}
    for model in args.models:
        with h5py.File(
            Path(args.raw) / model / "shards" / f"{slide_id}.h5", "r"
        ) as source:
            per_model_raw[model] = np.asarray(source["features"][:], dtype=np.float32)
    conditions = [("raw", None, per_model_raw)]
    for target in args.targets:
        bundle = {}
        for model in args.models:
            path = Path(args.features) / target / model / "shards" / f"{slide_id}.h5"
            with h5py.File(path, "r") as source:
                names = [value.decode() for value in source["condition"][:]]
                index = names.index(f"{RF1U_CONDITION}_{target}")
                bundle[model] = np.asarray(source["features"][index], dtype=np.float32)
        conditions.append((f"ours_{target}", target, bundle))

    payload = {}
    with torch.inference_mode():
        for name, target, bundle in conditions:
            if target is None:
                gains = None
                reference = None
            else:
                reference = target_index(target)
                sources = source_indices(target)
                _, _, energy = load_band_energy(Path(args.energy), target, KNN_FOV)
                fitted = fold_gains(energy, fold)
                gains = {sources[i]: fitted["gain"][i] for i in range(len(sources))}

            def render(scanner_index: int, position: int):
                patch = raw_patch(scanner_index, position)
                if gains is None or scanner_index == reference:
                    return patch
                base = reinhard_lab(
                    patch,
                    mean_tensor[scanner_index].reshape(1, 1, 1, 3),
                    std_tensor[scanner_index].reshape(1, 1, 1, 3),
                    mean_tensor[reference].reshape(1, 1, 1, 3),
                    std_tensor[reference].reshape(1, 1, 1, 3),
                )["output"]
                return shared_od_multiscale(base, gains[scanner_index], RF1M_SIGMAS)["output"]

            images = {}
            models_payload = {}
            for model in args.models:
                entries = []
                ranked = [(query_index, 1.0)] + top_neighbours(
                    bundle[model], query_index, args.neighbours
                )
                for rank, (index, cosine) in enumerate(ranked):
                    scanner_index, position = divmod(index, 100)
                    key = f"{SCANNERS[scanner_index]}_{position}"
                    if key not in images:
                        images[key] = png_data_uri(
                            uint8_from_rgb01(render(scanner_index, position))[0],
                            args.panel_px,
                        )
                    entries.append(
                        {
                            "rank": rank,
                            "key": key,
                            "scanner": SCANNERS[scanner_index],
                            "location": position,
                            "cosine": cosine,
                            "is_query": rank == 0,
                            "same_scanner": scanner_index == query_scanner,
                            "same_location": position == location,
                        }
                    )
                models_payload[model] = {
                    "entries": entries,
                    "same_scanner_neighbours": sum(
                        entry["same_scanner"] for entry in entries[1:]
                    ),
                    "same_location_neighbours": sum(
                        entry["same_location"] for entry in entries[1:]
                    ),
                }
                print(
                    f"  {name} / {model}: same-scanner "
                    f"{models_payload[model]['same_scanner_neighbours']}/{args.neighbours}, "
                    f"same-location "
                    f"{models_payload[model]['same_location_neighbours']}/{args.neighbours}",
                    flush=True,
                )
            payload[name] = {
                "condition": name,
                "target": target,
                "images": images,
                "models": models_payload,
            }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "knn_gallery.json").write_text(json.dumps(payload, indent=1) + "\n")
    summary = {
        "analysis": "rf1u_knn_gallery",
        "rf1u_version": RF1U_VERSION,
        "models": args.models,
        "panel_px": args.panel_px,
        "slide_id": slide_id,
        "location": location,
        "query_scanner": SCANNERS[query_scanner],
        "query_selection": args.query_selection,
        "texture_percentile": args.texture_percentile,
        "query_structure_score": float(content[query_index]),
        "median_structure_score": float(np.median(content)),
        "query_rule": (
            "a patch at this slide's median raw same-scanner neighbour count, so the "
            "figure shows the usual case, restricted to structurally detailed patches; "
            "ties by most same-location hits, then index"
        ),
        "heldout_fold": fold,
        "neighbours": args.neighbours,
        "conditions": list(payload),
        "fov": KNN_FOV,
        "note": (
            "Neighbours come from the already extracted embeddings of this slide's "
            "600 patches; only the displayed patches were rendered."
        ),
        "query_from_model": args.models[0],
        "counts": {
            name: {
                model: {
                    "same_scanner": entry["same_scanner_neighbours"],
                    "same_location": entry["same_location_neighbours"],
                }
                for model, entry in value["models"].items()
            }
            for name, value in payload.items()
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
