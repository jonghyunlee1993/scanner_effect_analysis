"""Check the correction-condition features: contract, finiteness, and pairing.

The pairing check is the one that matters here.  Every condition of a section
must sit on exactly the frozen evaluation locations, because the endpoints
contrast a corrected stack against the raw one location by location; if a
condition silently kept a different set, the contrast would be between different
tissue and nothing downstream would notice.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

root = Path("outputs/plism_core_condition_features")
frozen = Path("outputs/plism_core_eval_locations")
raw = Path("data/PLISM_dataset/features")
contract = json.loads(Path("outputs/e9_pfm_contract/checkpoint_manifest.json").read_text())
models = {m["encoder_id"]: m for m in contract["models"]}

bad, rows = [], []
for condition in sorted(p.name for p in root.iterdir() if p.is_dir()):
    for encoder in sorted(models):
        files = sorted((root / condition / encoder).glob("*.h5"))
        dim = models[encoder]["feature_dim"]
        total, nonfinite, off_grid, identical = 0, 0, 0, 0
        for path in files:
            with h5py.File(path, "r") as handle:
                x = handle["features"][:]
                meta = json.loads(handle.attrs["meta"])
                location = handle["location"][:]
            if x.shape[1] != dim:
                bad.append(f"{condition}/{encoder}/{path.name}: dim {x.shape[1]} != {dim}")
            if meta.get("condition") != condition:
                bad.append(f"{path.name}: meta condition {meta.get('condition')}")
            if str(meta["precision"]) != models[encoder]["precision"]:
                bad.append(f"{path.name}: precision {meta['precision']}")
            if meta["checkpoint_sha256"] != models[encoder]["checkpoint_sha256"]:
                bad.append(f"{path.name}: checkpoint sha mismatch")
            count = int((~np.isfinite(x)).sum())
            if count:
                nonfinite += count
                bad.append(f"{path.name}: {count} non-finite")
            wanted = set(json.loads(
                (frozen / encoder / f"{meta['stain']}.json").read_text())["locations"])
            if not set(int(v) for v in location) <= wanted:
                off_grid += 1
                bad.append(f"{path.name}: locations outside the frozen evaluation set")
            # the correction must actually have changed the tile
            with h5py.File(raw / encoder / f"{meta['stain']}_{meta['scanner']}.h5", "r") as h:
                rl, rf = h["location"][:], h["features"][:]
            mask = np.isin(rl, location)
            base = rf[mask][np.argsort(rl[mask])]
            here = x[np.argsort(location)]
            if base.shape == here.shape and np.allclose(base, here, atol=1e-6):
                identical += 1
                if not meta["condition_detail"].get("colour_identity"):
                    bad.append(f"{path.name}: identical to raw but not an identity condition")
            total += x.shape[0]
        rows.append((condition, encoder, len(files), total, nonfinite, off_grid, identical))

print(f"{'condition':16s} {'encoder':11s} {'files':>5s} {'vectors':>10s} "
      f"{'nonfinite':>9s} {'off-grid':>8s} {'==raw':>6s}")
for r in rows:
    print(f"{r[0]:16s} {r[1]:11s} {r[2]:5d} {r[3]:10,d} {r[4]:9d} {r[5]:8d} {r[6]:6d}")
print(f"\ntotal vectors {sum(r[3] for r in rows):,}")
print("PROBLEMS:", len(bad))
for entry in bad[:12]:
    print("  ", entry)
sys.exit(1 if bad else 0)
