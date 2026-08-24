"""Seven-scanner crop strips for named TMA cores, for the alignment report.

Every tile is read at the crop the refinement actually used -- `centre_x`,
`centre_y` and `flip` are taken straight from the refinement table, not
recomputed -- so what the reader sees is the patch the residual was measured on.
Tiles are resized for display only.

A strip is the alignment claim in visible form: one core, one tissue name, the
same field of view on seven scanners.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_plism_registration_report_assets import open_full
from extract_plism_native_psd import PATCH_UM

REFERENCE = "AT2"
SCANNER_ORDER = ["AT2", "GT450", "P", "S210", "S360", "S60", "SQ"]
DISPLAY_PX = 220


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-registration", default="outputs/plism_core_registration")
    parser.add_argument("--refinement", default="outputs/plism_core_refinement")
    parser.add_argument("--wsi-dir", default="data/PLISM_dataset/original_wsi")
    parser.add_argument("--output", default="outputs/plism_core_gallery")
    parser.add_argument("--sections", default="MY,HRH",
                        help="one clean section and one carrying a known failure")
    parser.add_argument("--cores", default="18_liver,25_cerebral_cortex,29_lung,"
                                           "45_skeletal_muscle,12_lymph_node,39_HCC")
    return parser.parse_args()


def encode(array: np.ndarray, quality: int = 82) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


def read_measured(image, record) -> np.ndarray | None:
    """The exact crop the refinement measured, at that scanner's level 0."""
    side = int(round(PATCH_UM / float(record.mpp)))
    x = int(round(float(record.centre_x))) - side // 2
    y = int(round(float(record.centre_y))) - side // 2
    if x < 0 or y < 0 or x + side > image.width or y + side > image.height:
        return None
    patch = image.crop(x, y, side, side).numpy()
    return np.rot90(patch, 2) if bool(record.flip) else patch


def decorate(patch: np.ndarray) -> np.ndarray:
    """One display size for every scanner, plus a reticle to judge correspondence by."""
    import cv2

    tile = cv2.resize(patch, (DISPLAY_PX, DISPLAY_PX), interpolation=cv2.INTER_AREA).copy()
    mid, arm = DISPLAY_PX // 2, 13
    for a, b in (((mid - arm, mid), (mid - 4, mid)), ((mid + 4, mid), (mid + arm, mid)),
                 ((mid, mid - arm), (mid, mid - 4)), ((mid, mid + 4), (mid, mid + arm))):
        cv2.line(tile, a, b, (20, 190, 210), 1)
    return tile


def main() -> None:
    args = parse_args()
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    wanted = [c.strip() for c in args.cores.split(",") if c.strip()]
    wsi_dir = Path(args.wsi_dir)

    strips = []
    for stain in sections:
        section = json.loads((Path(args.core_registration) / f"{stain}.json").read_text())
        table = pd.read_csv(Path(args.refinement) / f"{stain}.csv")
        images = {s: open_full(str(wsi_dir / m["name"]))
                  for s, m in section["scanners"].items()}

        for tissue in wanted:
            block = table.loc[(table["tissue_type"] == tissue) & table["ok"]]
            if block.empty:
                print(f"  {stain}/{tissue}: no measured location")
                continue
            # the location where the whole panel succeeded, so the strip is comparable
            counts = block.groupby("location")["scanner"].nunique()
            complete = counts.loc[counts == len(section["scanners"])]
            location = int((complete.index[0] if len(complete) else counts.idxmax()))
            row = {"stain": stain, "tissue_type": tissue, "location": location,
                   "tiles": {}, "residual": {}}
            for record in block.loc[block["location"] == location].itertuples():
                patch = read_measured(images[record.scanner], record)
                if patch is None:
                    continue
                row["tiles"][record.scanner] = encode(decorate(patch))
                row["residual"][record.scanner] = float(record.residual_um)
            if len(row["tiles"]) > 1:
                strips.append(row)
                shown = " ".join(f"{s}={row['residual'].get(s, float('nan')):.2f}"
                                 for s in SCANNER_ORDER if s in row["residual"])
                print(f"  {stain}/{tissue} loc {location}: {shown}")

    payload = {"sections": sections, "scanner_order": SCANNER_ORDER,
               "patch_um": PATCH_UM, "display_px": DISPLAY_PX, "strips": strips}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "core_gallery.json"
    destination.write_text(json.dumps(payload))
    print(f"-> {destination} ({destination.stat().st_size / 1e6:.1f} MB), {len(strips)} strips")


if __name__ == "__main__":
    main()
