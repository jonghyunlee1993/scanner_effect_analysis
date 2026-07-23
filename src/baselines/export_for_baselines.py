"""Export the precomputed store to the folder formats expected by the vendored
pytorch-CycleGAN-and-pix2pix repo, so baselines train on the SAME tiles as our method
(apples-to-apples). The vendored stock datasets are used unchanged.

  aligned  (pix2pix):  {out}/{split}/{slide}_{tid}.png = [src | ref] side-by-side
  unaligned(cyclegan): {out}/{split}A/*.png (src domain), {out}/{split}B/*.png (ref domain)

src = input scanner, ref = target/reference scanner (default at2). Store/index paths come
from $PRENORM_CONFIG; experiment params via flags (run interactively or from a .sh).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import h5py
import numpy as np

from utils import store
from utils.config import load_config
from utils.index import read_index


def _rgb(handle, scanner, tid):
    return store.decode(handle[scanner]["rgb"][tid])[..., ::-1]  # RGB -> BGR for cv2.imwrite


def export(cfg, mode, src, ref, split, out):
    df = read_index(cfg.paths.output_index)
    df = df[(df["split"] == split) & df["kept"]]
    store_dir, out = Path(cfg.paths.output_store), Path(out)
    n = 0
    for slide, g in df.groupby("slide_id"):
        with h5py.File(store_dir / f"{slide}.h5", "r") as h:
            for _, r in g.iterrows():
                tid = int(r["tuple_id"])
                if mode == "aligned":
                    if not (
                        r[f"present_{src}"]
                        and r[f"geom_ok_{src}"]
                        and r[f"present_{ref}"]
                        and r[f"geom_ok_{ref}"]
                    ):
                        continue
                    d = out / split
                    d.mkdir(parents=True, exist_ok=True)
                    ab = np.concatenate([_rgb(h, src, tid), _rgb(h, ref, tid)], axis=1)
                    cv2.imwrite(str(d / f"{slide}_{tid}.png"), ab)
                    n += 1
                else:
                    for scanner, suffix in ((src, "A"), (ref, "B")):
                        if not r[f"present_{scanner}"]:
                            continue
                        d = out / f"{split}{suffix}"
                        d.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(d / f"{slide}_{tid}.png"), _rgb(h, scanner, tid))
                        n += scanner == src
    print(f"[export] {mode} {src}->{ref} split={split}: {n} src tiles -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["aligned", "unaligned"], default="aligned")
    ap.add_argument("--src", default="gt450", help="input scanner")
    ap.add_argument("--ref", default="at2", help="target/reference scanner")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(
        os.environ.get("PRENORM_CONFIG", "configs/main/phase_0_preprocessing_10slide_v3.yaml")
    )
    export(cfg, a.mode, a.src, a.ref, a.split, a.out)


if __name__ == "__main__":
    main()
