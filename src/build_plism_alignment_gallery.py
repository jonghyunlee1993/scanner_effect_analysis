"""Alignment-check crops for the PLISM registration report.

Produces many more samples than the first pass, each carrying the residual
offset measured on it, so alignment can be judged by eye against a number rather
than by impression.

Three things this pass fixes or adds.

*Model per scanner.*  A blanket affine is wrong: it rescues Philips (28.2 to
9.4 um) and S60 (8.9 to 6.4) but is worse than the similarity for S210 and S360,
because the extra two degrees of freedom fit keypoint noise when there is no real
anisotropy.  The policy below was chosen on measured residual.

*Per-location refinement.*  The global model leaves 5-7 um that is local, not
global.  Each location is refined by phase correlation against the reference and
the correction is applied as an **integer pixel shift**, so no analysed pixel is
interpolated.

*Orientation.*  Philips scans 180 degrees rotated.  The transform maps
coordinates, not pixels, so its tiles came out upside down in the first gallery.
They are rotated for display here, and the rotation is applied before correlation
so its residual is measured against the right orientation.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np

from build_plism_section_registration import REFERENCE, THUMB_UM, thumbnail
from build_plism_registration_report_assets import find_cores, open_full, reference_thumbnail
from extract_plism_native_psd import PATCH_UM

SCANNER_ORDER = ["AT2", "GT450", "P", "S210", "S360", "S60", "SQ"]
# Chosen on measured residual over 13 sections; see audit_plism_registration_models.py
MODEL_POLICY = {"GT450": "similarity", "P": "affine", "S210": "similarity",
                "S360": "similarity", "S60": "affine", "SQ": "similarity"}
FLIPPED = {"P"}
DISPLAY_PX = 232


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", default="outputs/plism_section_registration")
    parser.add_argument("--wsi-dir", default="data/PLISM_dataset/original_wsi")
    parser.add_argument("--output", default="outputs/plism_report_assets")
    parser.add_argument("--sections", default="MY,GIVH,HR")
    parser.add_argument("--cores", type=int, default=10)
    parser.add_argument("--compare-cores", type=int, default=6)
    return parser.parse_args()


def encode(array: np.ndarray, quality: int = 84) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


def fit_models(moving: np.ndarray, fixed: np.ndarray, guide: np.ndarray | None = None,
               report: dict | None = None) -> dict:
    """Inverse transforms (fixed -> moving) for each model.

    With a `guide` -- the rigid mask fit of plism_mask_registration -- matches
    that disagree with it are dropped before RANSAC.  On a TMA the wrong solution
    is a whole core-pitch away and internally consistent, so without this filter
    it can outvote the right one; see Amendment 2 of the contract.
    """
    import cv2

    detector = cv2.AKAZE_create()
    kp_move, desc_move = detector.detectAndCompute(moving, None)
    kp_fix, desc_fix = detector.detectAndCompute(fixed, None)
    if desc_move is None or desc_fix is None:
        return {}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_move, desc_fix, k=2)
    good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
    source = np.float32([kp_move[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    target = np.float32([kp_fix[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)

    if report is not None:
        report["matches"] = len(good)
    if guide is not None and len(good):
        from plism_mask_registration import MIN_GUIDED_MATCHES, guided_matches

        kept_source, kept_target, kept, dropped = guided_matches(source, target, guide)
        if report is not None:
            report.update(guided_kept=kept, guided_dropped=dropped)
        if kept >= MIN_GUIDED_MATCHES:
            source, target = kept_source, kept_target
        elif report is not None:
            report["guide_used_directly"] = True

    out = {}
    if len(source) >= 4:
        similarity, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                                    ransacReprojThreshold=2.0)
        affine, _ = cv2.estimateAffine2D(source, target, method=cv2.RANSAC,
                                         ransacReprojThreshold=2.0)
        for name, matrix in (("similarity", similarity), ("affine", affine)):
            if matrix is not None:
                out[name] = cv2.invertAffineTransform(matrix)
    return out


def read_raw(image, centre, mpp: float, extra=(0, 0)):
    side = int(round(PATCH_UM / mpp))
    x = int(round(centre[0])) - side // 2 + int(round(extra[0]))
    y = int(round(centre[1])) - side // 2 + int(round(extra[1]))
    if x < 0 or y < 0 or x + side > image.width or y + side > image.height:
        return None
    return image.crop(x, y, side, side).numpy()


def to_od(patch: np.ndarray) -> np.ndarray:
    return -np.log10(np.clip(patch.astype(np.float32), 1.0, 255.0) / 255.0).mean(axis=2)


def measure(reference_od: np.ndarray, patch_od: np.ndarray, window, ref_mpp: float):
    import cv2

    moving = patch_od
    if moving.shape != reference_od.shape:
        moving = cv2.resize(moving, reference_od.shape[::-1], interpolation=cv2.INTER_AREA)
    a = np.ascontiguousarray(reference_od - reference_od.mean(), dtype=np.float64)
    b = np.ascontiguousarray(moving - moving.mean(), dtype=np.float64)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    return dx, dy, float(np.hypot(dx, dy) * ref_mpp), response


def decorate(patch: np.ndarray) -> np.ndarray:
    """Common display size plus a faint centre reticle to judge correspondence by."""
    import cv2

    tile = cv2.resize(patch, (DISPLAY_PX, DISPLAY_PX), interpolation=cv2.INTER_AREA).copy()
    mid = DISPLAY_PX // 2
    arm = 14
    colour = (20, 190, 210)
    cv2.line(tile, (mid - arm, mid), (mid - 4, mid), colour, 1)
    cv2.line(tile, (mid + 4, mid), (mid + arm, mid), colour, 1)
    cv2.line(tile, (mid, mid - arm), (mid, mid - 4), colour, 1)
    cv2.line(tile, (mid, mid + 4), (mid, mid + arm), colour, 1)
    return tile


def main() -> None:
    args = parse_args()
    registration_dir = Path(args.registration)
    wsi_dir = Path(args.wsi_dir)
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]

    galleries, comparisons = [], []
    for stain in sections:
        section = json.loads((registration_dir / f"{stain}.json").read_text())
        thumb = reference_thumbnail(section, wsi_dir)
        cores = find_cores(thumb, args.cores)
        reference_meta = section["scanners"][REFERENCE]
        reference_mpp = reference_meta["mpp"]
        reference_image = open_full(str(wsi_dir / reference_meta["name"]))
        side = int(round(PATCH_UM / reference_mpp))
        window = np.outer(np.hanning(side), np.hanning(side))

        thumbs = {s: thumbnail(str(wsi_dir / m["name"]), m["mpp"])
                  for s, m in section["scanners"].items()}
        inverses = {s: fit_models(thumbs[s], thumbs[REFERENCE])
                    for s in section["scanners"] if s != REFERENCE}

        print(f"{stain}: {len(cores)} cores")
        for core in cores:
            point = np.array([[core["x"], core["y"]]])
            reference_patch = read_raw(reference_image, point[0] * THUMB_UM / reference_mpp,
                                       reference_mpp)
            if reference_patch is None:
                continue
            reference_od = to_od(reference_patch)
            row = {"stain": stain, "core": core["core"],
                   "tiles": {REFERENCE: encode(decorate(reference_patch))},
                   "residual": {REFERENCE: 0.0}, "response": {REFERENCE: 1.0}}
            compare = {"stain": stain, "core": core["core"], "tiles": {}, "residual": {}}

            for scanner in SCANNER_ORDER:
                if scanner == REFERENCE:
                    continue
                meta = section["scanners"][scanner]
                image = open_full(str(wsi_dir / meta["name"]))
                flip = scanner in FLIPPED

                variants = {}
                for label, model in (("before", "similarity"), ("after", MODEL_POLICY[scanner])):
                    inverse = inverses[scanner].get(model)
                    if inverse is None:
                        continue
                    centre = (point @ inverse[:, :2].T + inverse[:, 2])[0] * THUMB_UM / meta["mpp"]
                    patch = read_raw(image, centre, meta["mpp"])
                    if patch is None:
                        continue
                    view = np.rot90(patch, 2) if flip else patch
                    dx, dy, offset, response = measure(reference_od, to_od(view), window,
                                                       reference_mpp)
                    if label == "after":
                        # integer shift only: the refinement never interpolates a pixel.
                        # Sign fixed empirically -- +shift drives the residual to 0.06-0.08 um,
                        # -shift doubles it.
                        shift = np.array([dx, dy]) * reference_mpp / meta["mpp"]
                        if flip:
                            shift = -shift
                        refined = read_raw(image, centre, meta["mpp"], extra=shift)
                        if refined is not None:
                            patch = refined
                            view = np.rot90(patch, 2) if flip else patch
                            dx, dy, offset, response = measure(reference_od, to_od(view),
                                                              window, reference_mpp)
                    variants[label] = (view, offset, response)

                if "after" in variants:
                    view, offset, response = variants["after"]
                    row["tiles"][scanner] = encode(decorate(view))
                    row["residual"][scanner] = offset
                    row["response"][scanner] = response
                if "before" in variants and core["core"] <= args.compare_cores:
                    view, offset, _ = variants["before"]
                    compare["tiles"][f"{scanner}_before"] = encode(decorate(view))
                    compare["residual"][f"{scanner}_before"] = offset
                    if "after" in variants:
                        compare["tiles"][f"{scanner}_after"] = row["tiles"][scanner]
                        compare["residual"][f"{scanner}_after"] = row["residual"][scanner]

            galleries.append(row)
            if compare["tiles"]:
                comparisons.append(compare)
            print(f"  core {core['core']}: " +
                  " ".join(f"{s}={row['residual'].get(s, float('nan')):.1f}" for s in SCANNER_ORDER))

    payload = {
        "sections": sections,
        "reference": REFERENCE,
        "patch_um": PATCH_UM,
        "display_px": DISPLAY_PX,
        "scanner_order": SCANNER_ORDER,
        "model_policy": MODEL_POLICY,
        "flipped": sorted(FLIPPED),
        "gallery": galleries,
        "comparison": comparisons,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "alignment_gallery.json"
    destination.write_text(json.dumps(payload))
    print(f"-> {destination} ({destination.stat().st_size / 1e6:.1f} MB), "
          f"{len(galleries)} gallery rows, {len(comparisons)} comparison rows")


if __name__ == "__main__":
    main()
