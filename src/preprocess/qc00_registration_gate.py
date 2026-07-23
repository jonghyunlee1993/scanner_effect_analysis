"""qc00 -- registration-quality gate (read-only on the WSIs / store).

Samples cfg.qc.gate_tiles_per_slide tiles per slide, reads all scanners at each
location, and computes per-scanner registration quality vs the reference scanner
(NCC / SSIM / residual phase shift) plus two image-space baselines on gray:
test-retest cosine (re-read upper bound) and within-scanner cosine across
distinct patches (dynamic range). Writes reg_quality.csv (per tile-scanner) and
reg_quality_summary.csv (per-scanner median/IQR of ncc,ssim and % below the
configured thresholds) under cfg.paths.qc_dir. Use this to confirm go/no-go and
to set qc.min_ncc / focus thresholds. No PFM/ResNet (evaluation stays sealed).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pandas as pd

import _common
from utils import grid, qc
from utils.config import load_config, resolve_config_path
from utils.wsi_io import open_wsi, read_patch


def _gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _gray_cos(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a.dot(b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def gate_slide(cfg, slide_id, row):
    """Return a list of per (tile, scanner) metric dicts for one slide."""
    scanners, ref, size = cfg.scanners, cfg.reference_scanner, cfg.patch.size
    coords, _ = grid.load_coords(grid.coords_path(cfg, ref, slide_id))
    rng = np.random.default_rng(_common.slide_seed(cfg, slide_id))
    sel = coords[_common.spatial_sample(coords, cfg.qc.gate_tiles_per_slide, cfg.budget.grid_cells, rng)]
    handles = {s: open_wsi(row[f"{s}_path"]) for s in scanners}
    grays = {s: [] for s in scanners}
    per = {s: [] for s in scanners}
    for x, y in sel:
        patch = {s: read_patch(handles[s], x, y, size) for s in scanners}
        gref = _gray(patch[ref])
        for s in scanners:
            p = patch[s]
            g = _gray(p)
            grays[s].append(g)
            if s == ref:
                ncc, ssim, ps = 1.0, 1.0, 0.0
            else:
                ncc = qc.local_ncc(patch[ref], p)
                ssim = qc.local_ssim(patch[ref], p)
                ps = qc.phase_shift(patch[ref], p)
            tr = _gray_cos(g, _gray(read_patch(handles[s], x, y, size)))  # re-read upper bound
            per[s].append(dict(slide_id=slide_id, x=int(x), y=int(y), scanner=s,
                               ncc=ncc, ssim=ssim, phase_shift=ps,
                               focus=qc.focus_score(p), test_retest_cos=tr))
    for s in scanners:
        handles[s].close()
    rows = []
    for s in scanners:
        gs = grays[s]
        for i, d in enumerate(per[s]):
            d["within_scanner_cos"] = _gray_cos(gs[i], gs[(i + 1) % len(gs)]) if len(gs) > 1 else 1.0
            rows.append(d)
    return rows


def summarize(df, cfg):
    g = df.groupby("scanner")
    out = pd.DataFrame({
        "n": g["ncc"].size(),
        "ncc_median": g["ncc"].median(),
        "ncc_iqr": g["ncc"].quantile(0.75) - g["ncc"].quantile(0.25),
        "ssim_median": g["ssim"].median(),
        "ssim_iqr": g["ssim"].quantile(0.75) - g["ssim"].quantile(0.25),
        "phase_shift_median": g["phase_shift"].median(),
        "test_retest_median": g["test_retest_cos"].median(),
        "within_scanner_median": g["within_scanner_cos"].median(),
        "pct_below_min_ncc": g["ncc"].apply(lambda s: float((s < cfg.qc.min_ncc).mean() * 100)),
        "pct_below_min_ssim": g["ssim"].apply(lambda s: float((s < cfg.qc.min_ssim).mean() * 100)),
    })
    return out.reset_index()


def main():
    cfg = load_config(resolve_config_path())
    reg = _common.load_registry(cfg)
    slides = _common.slide_list(cfg, reg)
    qc_dir = Path(cfg.paths.qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in slides:
        r = gate_slide(cfg, sid, reg.loc[sid])
        rows.extend(r)
        print(f"[qc00] {sid}: {len(r)} tile-scanner rows")
    df = pd.DataFrame(rows)
    df.to_csv(qc_dir / "reg_quality.csv", index=False)
    summarize(df, cfg).to_csv(qc_dir / "reg_quality_summary.csv", index=False)
    print(f"[qc00] wrote {len(df)} rows for {len(slides)} slides -> {qc_dir}")


if __name__ == "__main__":
    main()
