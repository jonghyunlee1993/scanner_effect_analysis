"""Crops for the PLISM registration QC report.

Two galleries.  The first is the alignment evidence: several TMA cores of one
section, each read on all seven scanners at the location the registration maps
them to.  The second shows the thirteen staining conditions on one scanner.

Cores are found on the section's own AT2 reference by connected components and
labelled by position.  PLISM's 46 tissue-type names are *not* used here; they live
in the authors' non-rigidly registered canvas, and importing them looked at first
like it would need the warp this work deliberately avoids.

build_plism_core_map.py later showed it does not.  A core is 2.5 mm across on a
3.6 mm pitch, so naming one only has to be right to a millimetre, and a single
global similarity between the canvas occupancy lattice and the section's own
tissue mask is enough for that.  It carries names inward and never places a
pixel.  Anything that wants named cores should read
outputs/plism_core_registration instead of this module.

Crops are read at level 0 and resized **for display only**.  Sharpness judged by
eye from these tiles is not the measurement; the measurement is the radial
spectrum computed on unresampled pixels.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np

from extract_plism_native_psd import PATCH_UM, TISSUE_OD_FLOOR, otsu_threshold

SCANNER_ORDER = ["AT2", "GT450", "P", "S210", "S360", "S60", "SQ"]
DISPLAY_PX = 240
CORE_MIN_AREA = 1500


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", default="outputs/plism_section_registration")
    parser.add_argument("--wsi-dir", default="data/PLISM_dataset/original_wsi")
    parser.add_argument("--output", default="outputs/plism_report_assets")
    parser.add_argument("--gallery-stain", default="MY")
    parser.add_argument("--cores", type=int, default=8)
    return parser.parse_args()


def open_full(path: str):
    import pyvips

    image = pyvips.Image.new_from_file(path, access="random")
    if image.hasalpha():
        image = image.flatten(background=255)
    return image


def encode(array: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


def crop_display(image, centre_xy, mpp: float) -> np.ndarray:
    import cv2

    side = int(round(PATCH_UM / mpp))
    x = int(round(centre_xy[0])) - side // 2
    y = int(round(centre_xy[1])) - side // 2
    x = int(np.clip(x, 0, image.width - side))
    y = int(np.clip(y, 0, image.height - side))
    patch = image.crop(x, y, side, side).numpy()
    return cv2.resize(patch, (DISPLAY_PX, DISPLAY_PX), interpolation=cv2.INTER_AREA)


def reference_thumbnail(registration: dict, wsi_dir: Path) -> np.ndarray:
    """Rebuild the AT2 reference thumbnail the registration was fitted on."""
    import cv2
    import pyvips

    meta = registration["scanners"][registration["reference"]]
    path = str(wsi_dir / meta["name"])
    probe = pyvips.Image.new_from_file(path, access="sequential")
    count = int(probe.get("openslide.level-count"))
    downsamples = [float(probe.get(f"openslide.level[{i}].downsample")) for i in range(count)]
    target = registration["thumb_um_per_px"]
    level = int(np.argmin([abs(np.log(d * meta["mpp"] / target)) for d in downsamples]))
    image = pyvips.Image.new_from_file(path, level=level, access="sequential")
    if image.hasalpha():
        image = image.flatten(background=255)
    array = image.colourspace("b-w").numpy()
    if array.ndim == 3:
        array = array[..., 0]
    scale = (downsamples[level] * meta["mpp"]) / target
    return cv2.resize(array.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def find_cores(thumb: np.ndarray, want: int) -> list[dict]:
    """Connected tissue components on the reference, largest first, spread out."""
    from scipy import ndimage

    white = max(float(np.percentile(thumb, 99.0)), 1.0)
    od = -np.log10(np.clip(thumb.astype(np.float32), 1.0, None) / white)
    mask = od > max(otsu_threshold(od), TISSUE_OD_FLOOR)
    mask = ndimage.binary_opening(mask, np.ones((3, 3)))
    labels, count = ndimage.label(mask)
    if count == 0:
        raise RuntimeError("no tissue components on the reference")
    areas = ndimage.sum(mask, labels, range(1, count + 1))
    order = np.argsort(areas)[::-1]

    cores, taken = [], []
    for index in order:
        if len(cores) >= want:
            break
        if areas[index] < CORE_MIN_AREA:
            continue
        cy, cx = ndimage.center_of_mass(mask, labels, index + 1)
        if any(np.hypot(cx - tx, cy - ty) < 60 for tx, ty in taken):
            continue
        taken.append((cx, cy))
        cores.append({"core": len(cores) + 1, "x": float(cx), "y": float(cy),
                      "area_px": float(areas[index])})
    return cores


def main() -> None:
    args = parse_args()
    registration_dir = Path(args.registration)
    wsi_dir = Path(args.wsi_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    gallery = json.loads((registration_dir / f"{args.gallery_stain}.json").read_text())
    thumb = reference_thumbnail(gallery, wsi_dir)
    cores = find_cores(thumb, args.cores)
    thumb_um = gallery["thumb_um_per_px"]
    print(f"{args.gallery_stain}: {len(cores)} cores on the {gallery['reference']} reference")

    images = {}
    for scanner in SCANNER_ORDER:
        transform = gallery["transforms"][scanner]
        meta = gallery["scanners"][scanner]
        inverse = np.array(transform["inverse"], dtype=np.float64)
        full = open_full(str(wsi_dir / meta["name"]))
        for core in cores:
            reference_xy = np.array([[core["x"], core["y"]]])
            mapped = (reference_xy @ inverse[:, :2].T + inverse[:, 2])[0] * thumb_um / meta["mpp"]
            images[f"core{core['core']}_{scanner}"] = encode(crop_display(full, mapped, meta["mpp"]))
        print(f"  {scanner}: {len(cores)} crops")

    # Stain strip: the reference scanner of every section, at its own largest core.
    strip = {}
    for path in sorted(registration_dir.glob("*.json")):
        section = json.loads(path.read_text())
        stain = section["stain"]
        section_thumb = reference_thumbnail(section, wsi_dir)
        best = find_cores(section_thumb, 1)[0]
        meta = section["scanners"][section["reference"]]
        full = open_full(str(wsi_dir / meta["name"]))
        centre = np.array([best["x"], best["y"]]) * section["thumb_um_per_px"] / meta["mpp"]
        strip[stain] = encode(crop_display(full, centre, meta["mpp"]))
        print(f"  strip {stain}")

    payload = {
        "gallery_stain": args.gallery_stain,
        "reference": gallery["reference"],
        "patch_um": PATCH_UM,
        "display_px": DISPLAY_PX,
        "scanner_order": SCANNER_ORDER,
        "cores": cores,
        "images": images,
        "stain_strip": strip,
    }
    destination = output / "report_assets.json"
    destination.write_text(json.dumps(payload))
    print(f"-> {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
