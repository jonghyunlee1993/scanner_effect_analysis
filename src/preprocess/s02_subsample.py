"""s02 -- spatial-stratified subsample of the candidate grid.

Reads the per-slide grid from s01 and deterministically picks candidates spread across a
budget.grid_cells spatial grid. Two sizing policies:

  budget.grid_fraction set -> ceil(grid_fraction * len(grid)), i.e. a fixed FRACTION of the
      slide's coords. Scales coverage with slide size and pulls in rare tissue types.
  otherwise                -> ceil(budget.tiles_per_slide * budget.oversample), the old
      fixed-count behaviour.

Only ~34% of candidates survive s03's QC (dominated by min_tissue_frac on the reference),
so the candidate pool must exceed the tile budget. Writes one candidate parquet per slide
(columns slide_id, x, y) under cfg.paths.qc_dir/candidates/. Re-run only this stage when
the budget changes.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import _common
from utils.config import load_config, resolve_config_path


def main():
    cfg = load_config(resolve_config_path())
    reg = _common.load_registry(cfg)
    slides = _common.slide_list(cfg, reg)
    gdir = Path(cfg.paths.qc_dir) / "grid"
    out = Path(cfg.paths.qc_dir) / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    frac = getattr(cfg.budget, "grid_fraction", None)
    total = 0
    for sid in slides:
        coords = pd.read_parquet(gdir / f"{sid}.parquet")[["x", "y"]].to_numpy()
        target = (math.ceil(frac * len(coords)) if frac
                  else math.ceil(cfg.budget.tiles_per_slide * cfg.budget.oversample))
        rng = np.random.default_rng(_common.slide_seed(cfg, sid))
        sel = coords[_common.spatial_sample(coords, target, cfg.budget.grid_cells, rng)]
        total += len(sel)
        pd.DataFrame({"slide_id": sid, "x": sel[:, 0], "y": sel[:, 1]}).to_parquet(
            out / f"{sid}.parquet", index=False)
        print(f"[s02] {sid}: {len(sel)} candidates (target {target}, grid {len(coords)}, "
              f"{len(sel)/max(len(coords),1)*100:.1f}%)")
    print(f"[s02] wrote candidates for {len(slides)} slides ({total} total) -> {out}")


if __name__ == "__main__":
    main()
