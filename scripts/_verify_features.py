"""Check every feature file: shape against the contract, finite values, shared grid."""
import h5py, json, numpy as np, sys
from pathlib import Path
root = Path("data/PLISM_dataset/features")
contract = json.loads(Path("outputs/e9_pfm_contract/checkpoint_manifest.json").read_text())
models = {m["encoder_id"]: m for m in contract["models"]}
bad = []
for enc in sorted(models):
    files = sorted((root / enc).glob("*.h5"))
    dim = models[enc]["feature_dim"]
    tot = 0; nonfinite = 0; per_section = {}
    for f in files:
        with h5py.File(f, "r") as h:
            x = h["features"][:]
            m = json.loads(h.attrs["meta"])
            if x.shape[1] != dim: bad.append(f"{f.name}: dim {x.shape[1]} != {dim}")
            if m["encoder_id"] != enc: bad.append(f"{f.name}: meta encoder {m['encoder_id']}")
            if str(m["precision"]) != models[enc]["precision"]:
                bad.append(f"{f.name}: precision {m['precision']}")
            if m["checkpoint_sha256"] != models[enc]["checkpoint_sha256"]:
                bad.append(f"{f.name}: checkpoint sha mismatch")
            nf = int((~np.isfinite(x)).sum())
            if nf: nonfinite += nf; bad.append(f"{f.name}: {nf} non-finite")
            tot += x.shape[0]
            per_section.setdefault(m["stain"], []).append((m["scanner"], set(h["location"][:].tolist())))
    # the grid must be shared: within a section every scanner sees the same locations
    mismatched = 0
    for stain, entries in per_section.items():
        sets = [s for _, s in entries]
        if len(set(map(frozenset, sets))) != 1: mismatched += 1
    print(f"{enc:10s} {len(files):3d} files  {tot:9,d} vectors  dim {dim:5d}  "
          f"non-finite {nonfinite}  sections with unequal grids {mismatched}/13")
print()
print("PROBLEMS:", len(bad))
for b in bad[:10]: print("  ", b)
sys.exit(1 if bad else 0)
