"""Per-location alignment refinement for one PLISM stain section.

The global transform leaves 5-9 um of residual that is local rather than global,
so each location is refined against the section's AT2 reference by phase
correlation.  The correction is stored as an **integer pixel shift**; applying it
is a different crop, never an interpolation, so the no-resampling property that
makes this cohort worth using survives intact.

The residual measured after refinement is stored with it.  Analysis gates on that
number rather than assuming the refinement worked: it does not always, and where
it fails the location is dropped for that scanner instead of quietly entering the
estimate.

The transform model is chosen per scanner from the measured residual over all 13
sections (see audit_plism_registration_models.py).  A blanket affine is wrong --
it rescues Philips and S60 and is worse than the similarity for S210 and S360.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_plism_alignment_gallery import (
    FLIPPED,
    MODEL_POLICY,
    fit_models,
    measure,
    read_raw,
    to_od,
)
from build_plism_section_registration import REFERENCE, THUMB_UM, thumbnail
from build_plism_registration_report_assets import open_full
from extract_plism_native_psd import PATCH_UM
from plism_mask_registration import gross_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", default="outputs/plism_section_registration")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--wsi-dir", default="data/PLISM_dataset/original_wsi")
    parser.add_argument("--output", default="outputs/plism_location_refinement")
    parser.add_argument("--no-guide", action="store_true",
                        help="fit keypoints unguided, as before Amendment 2")
    parser.add_argument("--chunk", type=int, default=400,
                        help="reference patches held in memory at once")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registration_dir = Path(args.registration)
    stains = sorted(p.stem for p in registration_dir.glob("*.json"))
    stain = stains[args.task_index]
    section = json.loads((registration_dir / f"{stain}.json").read_text())
    wsi_dir = Path(args.wsi_dir)

    locations = np.array(section["locations_reference_thumb_xy"], dtype=np.float64)
    cores = np.array(section["location_core"], dtype=int)
    # Replicate index runs within a core.  Cores are contiguous in the location
    # list, and a core map may give them unequal sizes, so count rather than
    # divide by a nominal per-core count.
    seen: dict[int, int] = {}
    replicates = []
    for core in cores.tolist():
        replicates.append(seen.get(core, 0))
        seen[core] = seen.get(core, 0) + 1
    replicates = np.array(replicates, dtype=int)
    # A core map carries PLISM's own tissue names; a plain section registration
    # does not, and then the column is simply absent.
    tissues = section.get("location_tissue")

    reference_meta = section["scanners"][REFERENCE]
    reference_mpp = reference_meta["mpp"]
    reference_image = open_full(str(wsi_dir / reference_meta["name"]))
    side = int(round(PATCH_UM / reference_mpp))
    window = np.outer(np.hanning(side), np.hanning(side))

    # A reference patch is 511x511 float32 = 1.04 MB.  On the pitch-1.0 lattice a
    # section carries ~8,400 locations, so preloading them all would need ~8.8 GB
    # before pyvips has read anything.  They are loaded a chunk at a time instead;
    # the number of reads is unchanged.
    present = np.zeros(len(locations), dtype=bool)
    for index, point in enumerate(locations):
        centre = point * THUMB_UM / reference_mpp
        side = int(round(PATCH_UM / reference_mpp))
        x = int(round(centre[0])) - side // 2
        y = int(round(centre[1])) - side // 2
        present[index] = (x >= 0 and y >= 0
                          and x + side <= reference_image.width
                          and y + side <= reference_image.height)

    def reference_chunk(indices):
        out = {}
        for index in indices:
            if not present[index]:
                continue
            patch = read_raw(reference_image,
                             locations[index] * THUMB_UM / reference_mpp, reference_mpp)
            if patch is not None:
                out[index] = to_od(patch)
        return out

    thumbs = {s: thumbnail(str(wsi_dir / m["name"]), m["mpp"])
              for s, m in section["scanners"].items()}
    images = {s: open_full(str(wsi_dir / m["name"]))
              for s, m in section["scanners"].items() if s != REFERENCE}
    chunks = [range(start, min(start + args.chunk, len(locations)))
              for start in range(0, len(locations), args.chunk)]

    rows = []
    # the reference needs no refinement, but carries the same bookkeeping
    for index in range(len(locations)):
        rows.append({
            "stain": stain, "scanner": REFERENCE, "location": index,
            "core": int(cores[index]), "replicate": int(replicates[index]),
            "shift_x": 0, "shift_y": 0, "residual_um": 0.0, "response": 1.0,
            "model": "identity", "ok": bool(present[index]),
            "centre_x": float(locations[index][0] * THUMB_UM / reference_mpp),
            "centre_y": float(locations[index][1] * THUMB_UM / reference_mpp),
            "mpp": reference_mpp, "flip": False,
        })

    guides, plans = {}, {}
    for scanner in sorted(section["scanners"]):
        if scanner == REFERENCE:
            continue
        meta = section["scanners"][scanner]
        model = MODEL_POLICY[scanner]

        # Mask-first: a rigid fit on the tissue silhouettes cannot settle one core
        # pitch off the way an unguided keypoint fit can.  It guides the keypoint
        # fit rather than replacing it, because it is only integer-pixel accurate.
        guide, report = None, {}
        if not args.no_guide:
            fit = gross_transform(thumbs[scanner], thumbs[REFERENCE])
            report.update(mask_dice=fit["dice"], mask_rotation=fit["rotation"],
                          mask_ok=fit["ok"])
            guide = fit["matrix"] if fit["ok"] else None
            guides[scanner] = fit

        models = fit_models(thumbs[scanner], thumbs[REFERENCE], guide=guide, report=report)
        inverse = models.get(model)
        if inverse is None and guide is not None:
            import cv2

            inverse = cv2.invertAffineTransform(np.asarray(guide, dtype=np.float64))
            report["fell_back_to_mask"] = True
        if inverse is None:
            raise RuntimeError(f"{stain}/{scanner}: {model} fit failed and no usable mask fit")
        if report:
            print(f"  {scanner:6s} guide dice={report.get('mask_dice', float('nan')):.3f} "
                  f"rot={report.get('mask_rotation', float('nan')):+7.2f} "
                  f"matches={report.get('matches', 0)} "
                  f"kept={report.get('guided_kept', -1)} "
                  f"dropped={report.get('guided_dropped', -1)}"
                  + (" FELLBACK" if report.get("fell_back_to_mask") else ""))
        plans[scanner] = {
            "meta": meta, "model": model, "image": images[scanner],
            "flip": scanner in FLIPPED, "rows": [],
            "mapped": (locations @ inverse[:, :2].T + inverse[:, 2]) * THUMB_UM / meta["mpp"],
        }

    # Locations outside, scanners inside: a reference patch is then read once and
    # measured against all six, instead of being re-read for every scanner.
    for chunk in chunks:
        references = reference_chunk(chunk)
        for scanner, plan in plans.items():
            meta, model = plan["meta"], plan["model"]
            image, flip = plan["image"], plan["flip"]
            for index in chunk:
                centre = plan["mapped"][index]
                record = {"stain": stain, "scanner": scanner, "location": index,
                          "core": int(cores[index]), "replicate": int(replicates[index]),
                          "model": model, "shift_x": 0, "shift_y": 0,
                          "residual_um": np.nan, "response": np.nan, "ok": False,
                          "centre_x": float(centre[0]), "centre_y": float(centre[1]),
                          "mpp": meta["mpp"], "flip": flip}
                reference_od = references.get(index)
                if reference_od is not None:
                    patch = read_raw(image, centre, meta["mpp"])
                    if patch is not None:
                        view = np.rot90(patch, 2) if flip else patch
                        dx, dy, _, _ = measure(reference_od, to_od(view), window, reference_mpp)
                        shift = np.array([dx, dy]) * reference_mpp / meta["mpp"]
                        if flip:
                            shift = -shift
                        shift = np.round(shift).astype(int)
                        refined = read_raw(image, centre, meta["mpp"], extra=shift)
                        if refined is not None:
                            view = np.rot90(refined, 2) if flip else refined
                            _, _, residual, response = measure(reference_od, to_od(view),
                                                               window, reference_mpp)
                            # store the crop position actually used, so extraction never
                            # re-fits RANSAC and cannot drift from what was measured here
                            record.update(shift_x=int(shift[0]), shift_y=int(shift[1]),
                                          centre_x=float(centre[0] + shift[0]),
                                          centre_y=float(centre[1] + shift[1]),
                                          residual_um=float(residual),
                                          response=float(response), ok=True)
                plan["rows"].append(record)

    for scanner, plan in plans.items():
        rows.extend(plan["rows"])
        block = [r for r in plan["rows"] if r["ok"]]
        values = np.array([r["residual_um"] for r in block]) if block else np.array([])
        responses = np.array([r["response"] for r in block]) if block else np.array([])
        locked = responses >= 0.3
        median = np.median(values[locked]) if locked.any() else float("nan")
        print(f"  {scanner:6s} {plan['model']:10s} n={len(block):5d} "
              f"median={median:6.3f} um  locked={int(locked.sum()):5d} "
              f"usable={int((locked & (values <= 1)).sum()):5d}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{stain}.csv"
    if guides:
        section.setdefault("mask_guides", {})
        for scanner, fit in guides.items():
            section["mask_guides"][scanner] = {
                "dice": fit["dice"], "rotation_deg": fit["rotation"],
                "shift_px": fit["shift"].tolist(), "ok": bool(fit["ok"]),
                "matrix_moving_to_reference": np.asarray(fit["matrix"]).tolist()}
        (registration_dir / f"{stain}.json").write_text(json.dumps(section, indent=2))

    frame = pd.DataFrame(rows)
    if tissues is not None:
        frame.insert(4, "tissue_type", frame["location"].map(dict(enumerate(tissues))))
    frame.to_csv(destination, index=False)
    print(f"{stain}: {len(locations)} locations x {len(section['scanners'])} scanners "
          f"-> {destination}")


if __name__ == "__main__":
    main()
