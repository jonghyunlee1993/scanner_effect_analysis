"""Write the README that ships with the PLISM feature store.

The features are useless without their geometry: which pixels went into a vector,
at what scale, from which physical section, and which locations were left out.
All of that is in the per-file HDF5 attributes already; this collects it into one
page so someone opening the folder does not have to reverse-engineer it.

Run after extraction so the counts are the real ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/PLISM_dataset/features")
    parser.add_argument("--contract", default="outputs/e9_pfm_contract/checkpoint_manifest.json")
    return parser.parse_args()


def survey(directory: Path) -> dict:
    files = sorted(directory.glob("*.h5"))
    if not files:
        return {}
    total, dims, sections, scanners, oob = 0, set(), set(), set(), 0
    meta = {}
    for path in files:
        with h5py.File(path, "r") as handle:
            meta = json.loads(handle.attrs["meta"])
            total += int(handle["features"].shape[0])
            dims.add(int(handle["features"].shape[1]))
            sections.add(meta["stain"])
            scanners.add(meta["scanner"])
            oob += int(meta.get("out_of_bounds", 0))
    return {"files": len(files), "vectors": total, "dim": sorted(dims),
            "sections": len(sections), "scanners": len(scanners),
            "out_of_bounds": oob, "meta": meta}


def cross_encoder_section(root: Path) -> list[str]:
    """One table putting the encoders on the only scale that compares them."""
    analysis = root / "analysis"
    frames = {}
    for directory in sorted(p for p in analysis.iterdir() if p.is_dir()) if analysis.exists() else []:
        path = directory / "cosine_vs_reference.csv"
        if path.exists():
            frames[directory.name] = pd.read_csv(path).set_index("scanner")
    if len(frames) < 2:
        return []

    order = [e for e in ("uni_v2", "conch_v15", "hoptimus1") if e in frames]
    order += [e for e in frames if e not in order]
    scanners = list(frames[order[0]].index)

    lines = [
        "",
        "## Scanner similarity, encoder by encoder",
        "",
        "Full detail per encoder is in `analysis/<encoder>/similarity_report.html`. The one",
        "number that compares *across* encoders is **top-1 retrieval**: of all locations of a",
        "TMA core on the reference scanner, is the nearest neighbour of a given location's",
        "embedding on another scanner the matching location? It is a rank statistic, so it",
        "does not care that the encoders' cosine scales differ.",
        "",
        "**Raw cosine does not compare across encoders.** CONCHv1.5 pairs at 0.94–0.97 and",
        "UNI2-h at 0.69–0.89, which looks decisive until the null is read beside it: the same",
        "tissue at a *different* location scores 0.78 on CONCHv1.5 and 0.42 on UNI2-h. The two",
        "embedding spaces simply use different parts of the cosine range.",
        "",
        "### top-1 retrieval against " + REFERENCE_NAME,
        "",
        "| Scanner | " + " | ".join(order) + " |",
        "|---" * (len(order) + 1) + "|",
    ]
    for scanner in scanners:
        cells = " | ".join(f"{frames[e].loc[scanner, 'top1']:.3f}" for e in order)
        lines.append(f"| `{scanner}` | {cells} |")
    mean = " | ".join(f"**{frames[e]['top1'].mean():.3f}**" for e in order)
    lines.append(f"| **mean** | {mean} |")

    lines += [
        "",
        "### Two things this shows",
        "",
        "**No encoder clusters by scanner.** Silhouette by scanner is within ±0.02 of zero for",
        "all three, and in the embedding plots the seven machines cover the same ground. The",
        "batch effect is invisible at the level of global structure — and plainly visible in",
        "retrieval, where the worst scanner misses roughly one match in ten. A study that",
        "checks only for scanner clusters would conclude there is nothing here.",
        "",
        "**The scanners do not rank the same way for every encoder.** CONCHv1.5 and",
        "H-optimus-1 agree almost completely (Spearman +0.94); UNI2-h is the one that departs",
        "(+0.43 and +0.26 against the other two). `SQ` is third-best on CONCHv1.5 and",
        "H-optimus-1 but last on UNI2-h; `GT450` is the reverse. Only the extremes are common",
        "to all three — `S360` best everywhere, Philips near the bottom on two of three. Take",
        "a single ranking of \"which scanner is hardest\" from one encoder and it will not",
        "transfer.",
        "",
    ]
    return lines


REFERENCE_NAME = "AT2"


def main() -> None:
    args = parse_args()
    root = Path(args.features)
    contract = json.loads(Path(args.contract).read_text())
    models = {m["encoder_id"]: m for m in contract["models"]}

    rows = []
    for encoder_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        stats = survey(encoder_dir)
        if stats:
            rows.append((encoder_dir.name, stats))

    target_mpp = contract["target_mpp"]
    tile_px = contract["tile_px"]

    lines = [
        "# PLISM — pathology foundation model features",
        "",
        "Patch embeddings for the native PLISM WSIs in `../original_wsi/`, taken at the "
        "aligned TMA-core grid described in `../ALIGNMENT.md`.",
        "",
        "## What one vector is",
        "",
        f"Every vector comes from one square of tissue **{tile_px * target_mpp:.2f} µm** "
        f"across, read at level 0 on that scanner's own pixel grid and reduced onto the "
        f"study's shared grid of **{tile_px} px at {target_mpp} µm/px** (libvips "
        "Lanczos3). The encoder then takes a centre crop of that tile at the size "
        "TRIDENT publishes for it at 20×, and the model's own eval transform resizes "
        "that crop to the model input.",
        "",
        "| Encoder | dim | crop from the tile | = µm | model input | lattice stride | precision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for encoder_id, stats in rows:
        model = models.get(encoder_id, {})
        lines.append(
            f"| `{encoder_id}` | {model.get('feature_dim','?')} | "
            f"{model.get('native_fov_px','?')} px | "
            f"{model.get('native_fov_um', float('nan')):.2f} | "
            f"{model.get('encoder_input_px','?')} px | "
            f"{model.get('lattice_stride','?')} | {model.get('precision','?')} |")

    lines += [
        "",
        "The 0.5052 µm/px grid is AT2's own 20× rather than a nominal 0.5; the 1.04% "
        "difference is far below the 19% spread of native pixel sizes across the panel "
        "(0.220–0.262 µm/px).",
        "",
        "**Lattice stride.** The grid locations sit on a 129.33 µm pitch, which is the "
        "patch size the alignment was measured on. An encoder whose field of view is "
        "wider than that pitch would produce overlapping tiles — more compute and "
        "correlated features, no new tissue — so it takes every other location in each "
        "axis instead. `uni_v2` at 129.33 µm tiles the grid exactly and uses stride 1; "
        "`conch_v15` at 258.66 µm uses stride 2, one quarter of the locations.",
        "",
        "**Resampling.** This is the one resampling step the native cohort was chosen to "
        "avoid, and it is unavoidable here: a foundation model has a fixed input scale. "
        "The spectral work in `../ALIGNMENT.md` and E9 Arm A still runs on unresampled "
        "pixels and is unaffected.",
        "",
        "## Contents",
        "",
        "| Encoder | files | vectors | dim | sections × scanners |",
        "|---|---:|---:|---:|---|",
    ]
    for encoder_id, stats in rows:
        lines.append(f"| `{encoder_id}` | {stats['files']} | {stats['vectors']:,} | "
                     f"{', '.join(str(d) for d in stats['dim'])} | "
                     f"{stats['sections']} × {stats['scanners']} |")

    lines += [
        "",
        "One file per (section, scanner), named `{stain}_{scanner}.h5`.",
        "",
        "### Datasets in each file",
        "",
        "| Name | Shape | Meaning |",
        "|---|---|---|",
        "| `features` | (N, dim) | float32 embedding |",
        "| `location` | (N,) | index into the section's grid, shared across scanners |",
        "| `core` | (N,) | TMA core 1–46 |",
        "| `tissue_type` | (N,) | PLISM's own name, e.g. `18_liver` |",
        "| `residual_um` | (N,) | alignment residual at that location |",
        "| `response` | (N,) | phase-correlation response; see the gating note below |",
        "| `centre_x`, `centre_y` | (N,) | crop centre in that scanner's level-0 pixels |",
        "",
        "`attrs['meta']` holds the geometry, the encoder identity, the checkpoint "
        "sha256, the TRIDENT commit, and any file correction applied to that slide.",
        "",
        "### Matching across scanners",
        "",
        "`location` is the same integer for the same physical place on all seven "
        "scanners of a section, so a scanner contrast is content-matched by taking the "
        "intersection of `location` values. It is **not** comparable across sections — "
        "sections are serial cuts, so their grids are independent. Use `tissue_type` to "
        "align across sections at the core level.",
        "",
        "### Gating",
        "",
        "Locations were kept when `residual_um <= 1.0`. `response` is carried through "
        "rather than gated on. A low response means the alignment could not be "
        "*verified* at that location, and that has more than one cause — for the "
        "`HRH`/`S60` block it is a soft scan rather than a misalignment, which is real "
        "scanner variation a batch-effect study should keep rather than silently drop. "
        "Filter on `response >= 0.3` if a stricter content match is wanted; §2.1 and §5 "
        "of `../ALIGNMENT.md` give the numbers.",
        "",
        "### Known defects carried in",
        "",
        "`GIVH_SQ.ndpi` and `HRH_SQ.ndpi` hold each other's section in the published "
        "dataset. **The correction is applied here** — a file named `GIVH_SQ.h5` "
        "contains features of the GIVH section, read from `HRH_SQ.ndpi`. Each file's "
        "`meta.file_corrections` records this. `HRH_S60.ndpi` is out of focus and is "
        "included as-is. See `../PROVENANCE.md` and `../ALIGNMENT.md` §5.",
        *cross_encoder_section(root),
        "",
        "## Provenance",
        "",
        f"- TRIDENT commit `{contract['trident_commit']}`, contract "
        f"`{contract['contract_version']}`.",
        "- Checkpoints and their sha256 are in `outputs/e9_pfm_contract/checkpoint_manifest.json` "
        "of the analysis repository.",
        "- One conda environment per encoder: they cannot share one, because UNI2-h needs "
        "timm's 1.0 line while TRIDENT's H-Optimus loader asserts timm 0.9.16.",
        "- Code: `src/fetch_e9_pfm_checkpoints.py`, `src/extract_plism_core_pfm.py`.",
        "- Cite PLISM: Ochi, M., Komura, D., Onoyama, T. et al. *Sci Data* **11**, 330 (2024).",
        "",
    ]
    destination = root / "README.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"-> {destination}")
    for encoder_id, stats in rows:
        print(f"   {encoder_id}: {stats['files']} files, {stats['vectors']:,} vectors, "
              f"{stats['out_of_bounds']} out of bounds")


if __name__ == "__main__":
    main()
