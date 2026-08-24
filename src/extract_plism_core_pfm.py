"""PFM embeddings for one PLISM WSI, at the aligned TMA-core grid locations.

One task is one (section, scanner).  Reading, rendering and encoding happen in a
single pass: the study's earlier route wrote 512 px tiles to disk first, which at
this grid density would be 92 GB of intermediates for a set of feature files
totalling under 5 GB.

**Geometry.**  Each location is read at level 0 on the scanner's own pixel grid,
at the crop position the alignment refinement actually measured — `centre_x`,
`centre_y` already carry the integer refinement shift — then reduced onto the
study's 0.5052 um/px grid with the same libvips Lanczos3 chain the rest of the
study uses, giving a 512 px tile of 258.66 um.  The encoder's own field of view
is a centre crop of that tile, at the size TRIDENT publishes for it at 20x:
256 px for UNI2-h, 512 for CONCHv1.5, 224 for H-optimus-0.  The model's eval
transform then does its own resize, exactly as in TRIDENT.

This is the one resampling step the native cohort was chosen to avoid, and it is
unavoidable: a PFM has a fixed input scale.  It is the same step PanNormal tiles
go through, so the cross-cohort comparison stays fair.  The spectral arm (E9
Arm A) still works on unresampled pixels and is untouched by this.

**Which locations.**  The lattice is at the 129.33 um patch pitch.  An encoder
whose field of view is wider than that would produce overlapping tiles, which add
compute and correlated features but no new tissue, so those encoders take every
other lattice point in each axis (`lattice_stride`).  Locations are gated on
`residual_um`; `response` is carried through rather than gated on, because a low
response means the alignment could not be *verified* there, and for one block
(S60/HRH) the cause is a soft scan rather than a misalignment -- real scanner
variation, which a batch-effect study should keep rather than silently drop.
See ALIGNMENT.md section 5.

**Conditions.**  `--condition raw` encodes the tile as read.  The correction
conditions apply the study's frozen operators to the rendered tile before the
centre crop, in memory: `reinhard_<dest>` is CIE Lab moment matching closed by
source-ray gamut projection, and `rf1u_<dest>` adds the fitted band gains through
the same shared-OD multiscale operator the deployment path uses.  Writing the
corrected tiles to disk first, as the sparse route did, would cost 640 GB per
condition on this grid.

A correction condition is restricted to the frozen evaluation subsample
(`--locations`), because the endpoints it feeds are O(L^2) in the location count
and stop moving after a few hundred.  The raw condition at those same locations
is already a subset of the full feature store, so it is not re-encoded.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE = "AT2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True,
                        help="0..90 over (section, scanner), sorted")
    parser.add_argument("--encoder-id", required=True)
    parser.add_argument("--contract", default="outputs/e9_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--core-registration", default="outputs/plism_core_registration")
    parser.add_argument("--refinement", default="outputs/plism_core_refinement")
    parser.add_argument("--wsi-dir", default="data/PLISM_dataset/original_wsi")
    parser.add_argument("--output", default="data/PLISM_dataset/features")
    parser.add_argument("--gate-um", type=float, default=1.0)
    parser.add_argument("--condition", default="raw",
                        help="raw, reinhard_<dest> or rf1u_<dest> with dest in at2, gt450")
    parser.add_argument("--locations", default="",
                        help="frozen evaluation subsample directory; empty uses every location")
    parser.add_argument("--population", default="outputs/plism_core_population_stats")
    parser.add_argument("--gains", default="outputs/plism_core_render_gains/fitted_gains.csv")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--trident-root",
                        default="/mnt/isilon/oldridge_lab/irinaz/github/env/trident")
    return parser.parse_args()


def units(core_dir: Path) -> list[tuple[str, str]]:
    """Every (section, scanner) pair, in a fixed order, so a task index is stable."""
    out = []
    for path in sorted(core_dir.glob("*.json")):
        section = json.loads(path.read_text())
        for scanner in sorted(section["scanners"]):
            out.append((section["stain"], scanner))
    return out


def sublattice(points: np.ndarray, step: float, stride: int) -> np.ndarray:
    """Mask keeping every `stride`-th lattice point in each axis.

    Indices are taken relative to the section's own minimum so the choice does
    not depend on where the lattice phase happened to fall.
    """
    if stride <= 1:
        return np.ones(len(points), dtype=bool)
    index = np.rint((points - points.min(axis=0)) / step).astype(int)
    return (index[:, 0] % stride == 0) & (index[:, 1] % stride == 0)


def build_condition(condition: str, scanner: str, population_dir: Path, gains_path: Path):
    """The per-tile transform for one condition, or None for the identity.

    Returned as a closure over a uint8 NxHxWx3 batch so the caller does not have
    to know which operators a condition is made of.
    """
    if condition == "raw":
        return None, {}

    import numpy as np
    import torch

    from render_plism_conditions import load_populations
    from rf1_gamut_reinhard import reinhard_lab_source_ray

    parts = condition.split("_")
    if len(parts) != 2 or parts[0] not in ("reinhard", "rf1u"):
        raise ValueError(f"unknown condition {condition}")
    method, destination = parts[0], parts[1].upper()

    populations = load_populations(population_dir)
    if destination not in populations:
        raise KeyError(f"no population statistics for {destination}")
    identity_colour = scanner == destination
    source, target = populations[scanner], populations[destination]
    moments = [torch.tensor([source[c][0] for c in "Lab"]),
               torch.tensor([source[c][1] for c in "Lab"]),
               torch.tensor([target[c][0] for c in "Lab"]),
               torch.tensor([target[c][1] for c in "Lab"])]

    gains, sigmas = None, None
    if method == "rf1u":
        from rf1m_combined import RF1M_SIGMAS

        table = pd.read_csv(gains_path)
        block = table.loc[(table["destination"] == parts[1]) & (table["source"] == scanner)]
        # A source that is its own destination, or a band whose gain could not be
        # estimated reliably, gets identity.  RF1U degrading to plain Reinhard is
        # a designed outcome, not a failure.
        gains = (np.ones((1, len(RF1M_SIGMAS))) if block.empty
                 else block.sort_values("band")["gain"].to_numpy()[None, :])
        sigmas = RF1M_SIGMAS

    limited, projected = [], []

    def apply(batch: np.ndarray) -> np.ndarray:
        from rf1m_combined import shared_od_multiscale_many

        rgb01 = torch.from_numpy(np.ascontiguousarray(batch)).float() / 255.0
        if not identity_colour:
            report = reinhard_lab_source_ray(rgb01, *moments)
            rgb01 = report["output"]
            limited.append(float(report["limited_pixel_fraction"].mean()))
        if gains is not None:
            report = shared_od_multiscale_many(rgb01, gains, sigmas)
            rgb01 = report["output"][0]
            projected.append(float(report["projection_fraction"].mean()))
        return (rgb01.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)

    note = {"condition": condition, "destination": destination,
            "colour_identity": bool(identity_colour),
            "gains": None if gains is None else gains[0].tolist(),
            "sigmas_px": None if sigmas is None else list(sigmas),
            "_limited": limited, "_projected": projected}
    return apply, note


def centred_crop(array: np.ndarray, fov: int) -> np.ndarray:
    height, width = array.shape[1:3]
    if fov > height or fov > width:
        raise ValueError(f"field of view {fov} exceeds the rendered tile {height}x{width}")
    top, left = (height - fov) // 2, (width - fov) // 2
    return array[:, top:top + fov, left:left + fov]


def main() -> None:
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    models = {m["encoder_id"]: m for m in contract["models"]}
    if args.encoder_id not in models:
        raise KeyError(f"{args.encoder_id} not in {args.contract}")
    model = models[args.encoder_id]

    target_mpp = float(contract["target_mpp"])
    tile_px = int(contract["tile_px"])
    tile_um = tile_px * target_mpp
    fov = int(model["native_fov_px"])
    stride = int(model["lattice_stride"])

    core_dir = Path(args.core_registration)
    pairs = units(core_dir)
    if not 0 <= args.task_index < len(pairs):
        raise IndexError(f"task index outside 0..{len(pairs) - 1}")
    stain, scanner = pairs[args.task_index]

    section = json.loads((core_dir / f"{stain}.json").read_text())
    meta = section["scanners"][scanner]
    step = float(section["lattice_step_px"])
    points = np.array(section["locations_reference_thumb_xy"], dtype=np.float64)
    keep_lattice = sublattice(points, step, stride)

    plan = pd.read_csv(Path(args.refinement) / f"{stain}.csv")
    plan = plan.loc[(plan["scanner"] == scanner) & plan["ok"]]
    plan = plan.loc[plan["residual_um"] <= args.gate_um]
    plan = plan.loc[plan["location"].map(lambda i: bool(keep_lattice[i]))]
    if args.locations:
        wanted = json.loads(
            (Path(args.locations) / args.encoder_id / f"{stain}.json").read_text())
        plan = plan.loc[plan["location"].isin(set(int(v) for v in wanted["locations"]))]
    plan = plan.sort_values("location")
    if plan.empty:
        raise RuntimeError(f"{stain}/{scanner}: no location survived the gate")

    output_dir = Path(args.output) / args.encoder_id
    if args.condition != "raw":
        output_dir = Path(args.output) / args.condition / args.encoder_id
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{stain}_{scanner}.h5"
    if destination.exists():
        print(f"{stain}/{scanner}/{args.encoder_id}: exists, skipping")
        return

    import h5py
    import pyvips
    import torch
    from PIL import Image

    transform, condition_note = build_condition(
        args.condition, scanner, Path(args.population), Path(args.gains))

    sys.path.insert(0, str(Path(args.trident_root).resolve()))
    from trident.patch_encoder_models.load import encoder_factory

    kwargs = {"weights_path": model["checkpoint_path"]}
    encoder = encoder_factory(args.encoder_id, **kwargs).eval().cuda()
    # The contract names a precision and TRIDENT's loader chooses one; if they
    # disagree the features are not the ones the contract describes.
    declared = str(model["precision"])
    actual = str(getattr(encoder, "precision", "")).replace("torch.", "")
    if actual and actual != declared:
        raise RuntimeError(
            f"{args.encoder_id}: contract says {declared}, TRIDENT builds {actual}")
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}
    if declared not in dtypes:
        raise KeyError(f"unsupported precision in the contract: {declared}")
    dtype = dtypes[declared]
    encoder = encoder.to(dtype)
    batch_size = args.batch_size or int(model["batch_size"])

    path = str(Path(args.wsi_dir) / str(meta["name"]))
    image = pyvips.Image.new_from_file(path, access="random")
    if image.hasalpha():
        image = image.flatten(background=255)

    mpp = float(meta["mpp"])
    shrink = target_mpp / mpp
    side = int(round(tile_um / mpp))
    flip = bool(plan["flip"].iloc[0])

    print(f"{stain}/{scanner}/{args.encoder_id}/{args.condition}: {len(plan)} locations "
          f"(stride {stride}), native {mpp:.5f} um/px, shrink {shrink:.3f}, "
          f"crop {fov} px = {fov * target_mpp:.2f} um, flip {flip}")

    features = np.empty((len(plan), int(model["feature_dim"])), dtype=np.float32)
    records, written = [], 0
    batch_tiles, batch_records = [], []
    out_of_bounds = 0

    def flush():
        nonlocal written
        if not batch_tiles:
            return
        stack = np.stack(batch_tiles)
        if transform is not None:
            # The correction acts on the whole rendered tile, as it does at
            # deployment; the encoder's centre crop is taken from the result.
            stack = transform(stack)
        stack = centred_crop(stack, fov)
        tensors = [encoder.eval_transforms(Image.fromarray(t, mode="RGB")) for t in stack]
        batch = torch.stack(tensors, dim=0).cuda(non_blocking=True).to(dtype)
        with torch.inference_mode():
            output = encoder(batch)
        features[written:written + len(stack)] = output.float().cpu().numpy()
        written += len(stack)
        records.extend(batch_records)
        batch_tiles.clear()
        batch_records.clear()

    for record in plan.itertuples():
        x = int(round(record.centre_x)) - side // 2
        y = int(round(record.centre_y)) - side // 2
        if x < 0 or y < 0 or x + side > image.width or y + side > image.height:
            out_of_bounds += 1
            continue
        patch = image.crop(x, y, side, side).reduce(shrink, shrink, kernel="lanczos3")
        array = patch.numpy()[:tile_px, :tile_px]
        if array.shape[:2] != (tile_px, tile_px):
            out_of_bounds += 1
            continue
        if flip:
            array = np.rot90(array, 2)
        batch_tiles.append(np.ascontiguousarray(array, dtype=np.uint8))
        batch_records.append(record)
        if len(batch_tiles) == batch_size:
            flush()
    flush()

    features = features[:written]
    if written == 0:
        raise RuntimeError(f"{stain}/{scanner}: every tile fell outside the slide")

    with h5py.File(destination, "w") as handle:
        handle.create_dataset("features", data=features, compression="lzf")
        handle.create_dataset("location", data=np.array([r.location for r in records], "int32"))
        handle.create_dataset("core", data=np.array([r.core for r in records], "int32"))
        handle.create_dataset(
            "tissue_type",
            data=np.array([str(r.tissue_type) for r in records], dtype=h5py.string_dtype()))
        handle.create_dataset("residual_um",
                              data=np.array([r.residual_um for r in records], "float32"))
        handle.create_dataset("response",
                              data=np.array([r.response for r in records], "float32"))
        handle.create_dataset("centre_x", data=np.array([r.centre_x for r in records], "float64"))
        handle.create_dataset("centre_y", data=np.array([r.centre_y for r in records], "float64"))
        handle.attrs["meta"] = json.dumps({
            "stain": stain, "scanner": scanner, "slide": str(meta["name"]),
            "encoder_id": args.encoder_id, "feature_dim": int(model["feature_dim"]),
            "native_fov_px": fov, "native_fov_um": fov * target_mpp,
            "encoder_input_px": int(model["encoder_input_px"]),
            "precision": model["precision"], "lattice_stride": stride,
            "target_mpp": target_mpp, "tile_px": tile_px, "native_mpp": mpp,
            "shrink": shrink, "flip": flip, "gate_um": args.gate_um,
            "tiles": int(written), "out_of_bounds": int(out_of_bounds),
            "condition": args.condition,
            "condition_detail": {k: v for k, v in condition_note.items()
                                 if not k.startswith("_")},
            "gamut_limited": (float(np.mean(condition_note["_limited"]))
                              if condition_note.get("_limited") else None),
            "projection_fraction": (float(np.mean(condition_note["_projected"]))
                                    if condition_note.get("_projected") else None),
            "eval_subsample": bool(args.locations),
            "contract_version": contract["contract_version"],
            "trident_commit": contract["trident_commit"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            "file_corrections": section.get("file_corrections", []),
        })

    print(f"  wrote {written} x {model['feature_dim']} -> {destination}"
          f"  ({out_of_bounds} out of bounds)")


if __name__ == "__main__":
    main()
