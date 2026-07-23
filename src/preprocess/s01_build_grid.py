"""s01 -- build the per-slide canonical sampling grid.

The grid is the INTERSECTION of all four scanners' curated coord sets, restricted to
coords whose margin-extended read window stays inside the level-0 frame. Previously only
the reference's coords were used, which seeded the store with tiles akoya never scanned.
grid.build_grid validates the 256-lattice and cross-scanner level-0 dimension agreement.
Writes one parquet per slide (columns slide_id, x, y) under cfg.paths.qc_dir/grid/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import _common
from utils import grid
from utils.config import load_config, resolve_config_path


def main():
    cfg = load_config(resolve_config_path())
    reg = _common.load_registry(cfg)
    slides = _common.slide_list(cfg, reg)
    out = Path(cfg.paths.qc_dir) / "grid"
    out.mkdir(parents=True, exist_ok=True)
    margin = int(getattr(cfg.patch, "margin", 0))
    for sid in slides:
        # External-scanner validation can lack a curated-coordinate file for the
        # source scanner.  In that case use the AT2 lattice and let s03's
        # registration/tissue gates decide per-tile eligibility.  Training keeps
        # the stricter all-scanner intersection by default.
        intersect = bool(getattr(getattr(cfg, "grid", {}), "intersect_scanners", True))
        g = grid.build_grid(cfg, sid, reg.loc[sid], intersect=intersect, margin=margin)
        c = g["coords"]
        pd.DataFrame({"slide_id": sid, "x": c[:, 0], "y": c[:, 1]}).to_parquet(
            out / f"{sid}.parquet", index=False)
        print(f"[s01] {sid}: ref {g['n_ref']} -> all-scanner intersection {g['n_intersect']} "
              f"({g['n_intersect']/max(g['n_ref'],1)*100:.1f}%) -> in-bounds@margin{margin} "
              f"{g['n_final']} coords")
    print(f"[s01] wrote grids for {len(slides)} slides -> {out}")


if __name__ == "__main__":
    main()
