"""Exp-07: how much does the image-level low-band correction move the UNI
(foundation-model) EMBEDDING toward AT2, does it reduce embedding-level scanner
separability, and does that survive when the correction is driven by the
source-only fingerprint estimate instead of the paired oracle affine?

Reuses the sealed UNI evaluator (`load_uni`/`embed_uni`/`scanner_probe` in
`eval_report.py`).  For held-out test tuples we build full-resolution renderings
of each source tile (only the low band changes; copied detail is identical),
embed them plus the paired AT2 tile with UNI, and report per rendering variant:

  * paired cosine to the tile's OWN registered AT2 (higher = closer),
  * scanner_probe balanced accuracy (4-class {at2,gt450,versa,s60}, 3-class).

Variants: raw, local (oracle contract-A affine + local projection), and -- if
``--estimated-affines`` is given -- est_local (per-(scanner,slide) fingerprint
estimate from ``eval_exp07_scanner_params.py`` + local projection).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prenorm.exp06.data import BalancedPairDataset
from prenorm.exp06.frequency import FixedLaplacianPyramid
from prenorm.exp07.projection import local_range_project
from utils.config import load_config
from eval_report import load_uni, embed_uni, scanner_probe


def apply_affine(low, affine):
    """affine [4,3] (shared) or [N,4,3] (per-item)."""
    x = low.permute(0, 2, 3, 1)
    x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
    if affine.dim() == 2:
        out = x @ affine
    else:
        out = torch.einsum("nhwk,nkc->nhwc", x, affine)
    return out.permute(0, 3, 1, 2)


@torch.no_grad()
def corrected_output(pyramid, source, affine, steps):
    low, _ = pyramid.decompose(source)
    corrected_unsafe = apply_affine(low, affine)
    return local_range_project(pyramid, source, corrected_unsafe, steps=steps)["output"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp06a_lf16_10slide.yaml")
    parser.add_argument("--baselines", default="outputs/exp06_baselines")
    parser.add_argument("--estimated-affines", default="outputs/exp07_stage1/estimated_affines.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-per-scanner", type=int, default=300)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    pyramid = FixedLaplacianPyramid(int(cfg.pyramid.levels))
    dataset = BalancedPairDataset(cfg, args.split, False)
    rng = np.random.default_rng(args.seed)
    model, size, mean, std = load_uni(device)

    scanners = [str(s) for s in cfg.target_scanners]
    baseline = {s: torch.tensor(json.loads(
        (Path(args.baselines) / f"{s}.json").read_text())["affine_matrix_with_bias"],
        dtype=torch.float32) for s in scanners}

    est_path = Path(args.estimated_affines)
    est = json.loads(est_path.read_text()) if est_path.exists() else None
    variants = ["raw", "local"] + (["est_local"] if est else [])

    def emb(images):
        return embed_uni(model, images, size, mean, std, device,
                         batch_size=args.batch_size).numpy()

    store = {s: {v: [] for v in variants} for s in scanners}
    for s in scanners:
        store[s]["at2"] = []
        store[s]["loc"] = []
    loc_ids = {}

    for scanner in scanners:
        idx = [i for i, r in enumerate(dataset.records) if r["scanner"] == scanner]
        if len(idx) > args.max_per_scanner:
            idx = sorted(rng.choice(idx, args.max_per_scanner, replace=False).tolist())
        for start in range(0, len(idx), args.batch_size):
            items = [dataset[i] for i in idx[start:start + args.batch_size]]
            source = torch.stack([it["source"] for it in items])
            reference = torch.stack([it["reference"] for it in items])
            store[scanner]["raw"].append(emb(source))
            store[scanner]["local"].append(
                emb(corrected_output(pyramid, source, baseline[scanner], args.steps)))
            if est:
                per_item = torch.stack([
                    torch.tensor(est[scanner][it["slide_id"]], dtype=torch.float32)
                    for it in items])
                store[scanner]["est_local"].append(
                    emb(corrected_output(pyramid, source, per_item, args.steps)))
            store[scanner]["at2"].append(emb(reference))
            for it in items:
                key = (it["slide_id"], int(it["tuple_id"]))
                loc_ids.setdefault(key, len(loc_ids))
                store[scanner]["loc"].append(loc_ids[key])

    for s in scanners:
        for v in variants + ["at2"]:
            store[s][v] = np.concatenate(store[s][v])
        store[s]["loc"] = np.asarray(store[s]["loc"])

    def paired_cos(a, b):
        return float((a * b).sum(1).mean())
    paired = {s: {v: paired_cos(store[s][v], store[s]["at2"]) for v in variants}
              for s in scanners}

    def probe(kind, classes):
        X, y, g = [], [], []
        for s in scanners:
            if s not in classes:
                continue
            X.append(store[s][kind]); y += [s] * len(store[s][kind]); g += store[s]["loc"].tolist()
        if "at2" in classes:
            seen, aX, ag = set(), [], []
            for s in scanners:
                for e, l in zip(store[s]["at2"], store[s]["loc"]):
                    if l in seen:
                        continue
                    seen.add(l); aX.append(e); ag.append(l)
            X.append(np.asarray(aX)); y += ["at2"] * len(aX); g += ag
        return scanner_probe(np.concatenate(X), np.asarray(y), np.asarray(g))

    tasks = {}
    for name, classes in (("4class", ["at2"] + scanners), ("3class", scanners)):
        tasks[name] = {"chance": 1.0 / len(classes),
                       **{v: probe(v, classes) for v in variants}}

    report = {"split": args.split, "variants": variants,
              "paired_cosine_to_at2": paired, "probe": tasks}
    print("\n=== UNI paired cosine to own AT2 (higher = closer) ===")
    for s in scanners:
        print(f"  {s:6s}  " + "  ".join(f"{v}={paired[s][v]:.3f}" for v in variants))
    print("\n=== UNI scanner separability (balanced acc; lower = more invariant) ===")
    for name, t in tasks.items():
        print(f"  {name} (chance {t['chance']:.3f}): "
              + "  ".join(f"{v}={t[v]:.3f}" for v in variants))

    out_dir = Path(cfg.paths.repo) / "outputs" / "exp07_stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "uni_embedding.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n[exp07-uni] wrote {out_dir/'uni_embedding.json'}")


if __name__ == "__main__":
    main()
