"""s04 -- merge the s03 sidecar metadata into the global parquet index.

Reads every per-slide sidecar, builds the flat per-(slide,tuple) index via
index.build_index, assigns a slide-level train/val/test split (explicit slide
lists when split.policy == explicit, otherwise deterministic fractions), and
writes cfg.paths.output_index.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from utils.config import load_config, resolve_config_path
from utils.index import assign_split, build_index, write_index


def main():
    cfg = load_config(resolve_config_path())
    scanners = cfg.scanners
    side_dir = Path(cfg.paths.qc_dir) / "sidecar"
    files = sorted(side_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"no sidecar parquet under {side_dir}; run s03 first")

    records = []
    for fp in files:
        records.extend(pd.read_parquet(fp).to_dict("records"))
    df = build_index(records, scanners)

    if cfg.split.policy == "explicit":
        sl = cfg.split.slides
        explicit = {"train": list(sl.train), "val": list(sl.val), "test": list(sl.test)}
        df = assign_split(df, "explicit", explicit=explicit)
    else:
        fractions = {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test}
        df = assign_split(df, "by_slide", fractions=fractions, seed=cfg.split.seed)

    out = Path(cfg.paths.output_index)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_index(df, str(out))
    by_split = df.groupby("split")["slide_id"].nunique().to_dict()
    print(f"[s04] index rows {len(df)}; slides {df['slide_id'].nunique()}; split-slides {by_split} -> {out}")


if __name__ == "__main__":
    main()
