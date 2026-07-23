"""Exp-07 high-frequency probe: which pyramid band carries the batch effect a
foundation model reads, and can correcting it buy UNI invariance?

The registered AT2 pair lets us TRANSPLANT the true AT2 detail bands at the same
location -- an oracle high-frequency correction with zero hallucination -- and
measure the ceiling before investing in any generative synthesis.  We embed a
spectrum of band manipulations with UNI and report scanner separability and
paired cosine to AT2 for each:

  raw          source unchanged
  low_corr     our low-band affine+local correction, source detail copied (Track-1)
  low_oracle   AT2 low transplanted, source detail kept  (perfect low fix; control)
  hi_oracle    source low kept, AT2 detail transplanted   (perfect high fix; PROBE 1)
  both_oracle  our low correction + AT2 detail            (both tracks)
  atten{λ}     our low correction, source detail scaled by λ in {0.75,0.5,0.25,0}
               (deterministic blur frontier; λ=0 = pure low band)

Together with the low-band scanner-BACC (coarse features), this locates the PFM
batch effect on the frequency axis.
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
    x = low.permute(0, 2, 3, 1)
    x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
    return (x @ affine).permute(0, 3, 1, 2)


ATTEN = [0.75, 0.5, 0.25, 0.0]


@torch.no_grad()
def variants(pyramid, source, reference, affine, steps):
    low_s, bands_s = pyramid.decompose(source)
    low_a, bands_a = pyramid.decompose(reference)
    corrected_unsafe = apply_affine(low_s, affine)
    low_c = local_range_project(pyramid, source, corrected_unsafe, steps=steps)["coarse"]
    out = {
        "raw": source,
        "low_corr": pyramid.reconstruct(low_c, bands_s),
        "low_oracle": pyramid.reconstruct(low_a, bands_s),
        "hi_oracle": pyramid.reconstruct(low_s, bands_a),
        "both_oracle": pyramid.reconstruct(low_c, bands_a),
    }
    for lam in ATTEN:
        out[f"atten{lam:.2f}"] = pyramid.reconstruct(low_c, [lam * b for b in bands_s])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp06a_lf16_10slide.yaml")
    parser.add_argument("--baselines", default="outputs/exp06_baselines")
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
    names = ["raw", "low_corr", "low_oracle", "hi_oracle", "both_oracle"] + \
            [f"atten{lam:.2f}" for lam in ATTEN]

    def emb(images):
        return embed_uni(model, images, size, mean, std, device,
                         batch_size=args.batch_size).numpy()

    store = {s: {n: [] for n in names + ["at2", "loc"]} for s in scanners}
    loc_ids = {}
    for scanner in scanners:
        idx = [i for i, r in enumerate(dataset.records) if r["scanner"] == scanner]
        if len(idx) > args.max_per_scanner:
            idx = sorted(rng.choice(idx, args.max_per_scanner, replace=False).tolist())
        for start in range(0, len(idx), args.batch_size):
            items = [dataset[i] for i in idx[start:start + args.batch_size]]
            source = torch.stack([it["source"] for it in items])
            reference = torch.stack([it["reference"] for it in items])
            v = variants(pyramid, source, reference, baseline[scanner], args.steps)
            for n in names:
                store[scanner][n].append(emb(v[n]))
            store[scanner]["at2"].append(emb(reference))
            for it in items:
                key = (it["slide_id"], int(it["tuple_id"]))
                loc_ids.setdefault(key, len(loc_ids))
                store[scanner]["loc"].append(loc_ids[key])

    for s in scanners:
        for n in names + ["at2"]:
            store[s][n] = np.concatenate(store[s][n])
        store[s]["loc"] = np.asarray(store[s]["loc"])

    def paired_cos(a, b):
        return float((a * b).sum(1).mean())
    paired = {n: {s: paired_cos(store[s][n], store[s]["at2"]) for s in scanners}
              for n in names}

    def probe(name, classes):
        X, y, g = [], [], []
        for s in scanners:
            if s not in classes:
                continue
            X.append(store[s][name]); y += [s] * len(store[s][name]); g += store[s]["loc"].tolist()
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
    for tname, classes in (("4class", ["at2"] + scanners), ("3class", scanners)):
        tasks[tname] = {"chance": 1.0 / len(classes),
                        **{n: probe(n, classes) for n in names}}

    report = {"split": args.split, "variants": names,
              "paired_cosine_to_at2": paired, "probe": tasks}
    print("\n=== UNI scanner separability by band manipulation (lower = invariant) ===")
    print(f"  {'variant':12s} {'4class':>7s} {'3class':>7s}   {'meanCosAT2':>10s}")
    for n in names:
        cos = float(np.mean([paired[n][s] for s in scanners]))
        print(f"  {n:12s} {tasks['4class'][n]:7.3f} {tasks['3class'][n]:7.3f}   {cos:10.3f}")
    print(f"  (chance 4class={tasks['4class']['chance']:.3f}  3class={tasks['3class']['chance']:.3f})")

    out_dir = Path(cfg.paths.repo) / "outputs" / "exp07_stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "uni_bands.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n[exp07-uni-bands] wrote {out_dir/'uni_bands.json'}")


if __name__ == "__main__":
    main()
