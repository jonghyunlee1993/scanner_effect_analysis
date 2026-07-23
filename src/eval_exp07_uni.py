"""Exp-07: how much does the image-level low-band correction move the UNI
(foundation-model) EMBEDDING toward AT2, does it reduce embedding-level scanner
separability, and does that survive when the correction is driven by the
source-only fingerprint estimate instead of the paired oracle affine?

Reuses the sealed UNI evaluator (`load_uni`/`embed_uni`/`scanner_probe` in
`eval_report.py`).  For held-out test tuples we build full-resolution renderings
of each source tile (only the low band changes; copied detail is identical),
embed them plus the paired AT2 tile with UNI, and report per rendering variant:

  * paired cosine to the tile's OWN registered AT2 (higher = closer),
  * scanner_probe balanced accuracy separately on each physical sampling lattice:
    internal_v3 {at2,gt450,versa} and external_s60_v1 {at2,s60}.

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
from prenorm.data.identity import (
    lattice_task_name,
    location_key,
    matched_indices_by_lattice,
)
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
    scanners = [str(s) for s in cfg.target_scanners]
    selections = matched_indices_by_lattice(
        dataset.records, scanners, args.max_per_scanner, rng
    )
    model, size, mean, std = load_uni(device)

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

    for selection in selections.values():
        for scanner in selection["scanners"]:
            idx = selection["indices"][scanner]
            for start in range(0, len(idx), args.batch_size):
                items = [dataset[i] for i in idx[start:start + args.batch_size]]
                source = torch.stack([it["source"] for it in items])
                reference = torch.stack([it["reference"] for it in items])
                store[scanner]["raw"].append(emb(source))
                store[scanner]["local"].append(
                    emb(
                        corrected_output(
                            pyramid, source, baseline[scanner], args.steps
                        )
                    )
                )
                if est:
                    per_item = torch.stack([
                        torch.tensor(est[scanner][it["slide_id"]], dtype=torch.float32)
                        for it in items])
                    store[scanner]["est_local"].append(
                        emb(
                            corrected_output(
                                pyramid, source, per_item, args.steps
                            )
                        )
                    )
                store[scanner]["at2"].append(emb(reference))
                store[scanner]["loc"].extend(
                    location_key(item).group_token() for item in items
                )

    for s in scanners:
        for v in variants + ["at2"]:
            store[s][v] = np.concatenate(store[s][v])
        store[s]["loc"] = np.asarray(store[s]["loc"])

    def paired_cos(a, b):
        return float((a * b).sum(1).mean())
    paired = {s: {v: paired_cos(store[s][v], store[s]["at2"]) for v in variants}
              for s in scanners}

    def probe(kind, lattice_scanners):
        X, y, groups = [], [], []
        for scanner in lattice_scanners:
            X.append(store[scanner][kind])
            y += [scanner] * len(store[scanner][kind])
            groups += store[scanner]["loc"].tolist()

        seen, at2_embedding, at2_groups = set(), [], []
        for scanner in lattice_scanners:
            for embedding, group in zip(store[scanner]["at2"], store[scanner]["loc"]):
                if group in seen:
                    continue
                seen.add(group)
                at2_embedding.append(embedding)
                at2_groups.append(group)
        X.append(np.asarray(at2_embedding))
        y += ["at2"] * len(at2_embedding)
        groups += at2_groups
        return scanner_probe(np.concatenate(X), np.asarray(y), np.asarray(groups))

    tasks = {}
    for lattice_id, selection in selections.items():
        lattice_scanners = selection["scanners"]
        task_name = lattice_task_name(lattice_id, lattice_scanners)
        classes = ["at2", *lattice_scanners]
        tasks[task_name] = {
            "lattice_id": lattice_id,
            "classes": classes,
            "chance": 1.0 / len(classes),
            "n_locations": len(selection["keys"]),
            **{
                variant: probe(variant, lattice_scanners)
                for variant in variants
            },
        }

    report = {
        "split": args.split,
        "variants": variants,
        "sample_identity": {
            "location_key": ["lattice_id", "slide_id", "tuple_id"],
            "cross_lattice_joint_probe": False,
        },
        "paired_cosine_to_at2": paired,
        "probe": tasks,
    }
    print("\n=== UNI paired cosine to own AT2 (higher = closer) ===")
    for s in scanners:
        print(f"  {s:6s}  " + "  ".join(f"{v}={paired[s][v]:.3f}" for v in variants))
    print("\n=== UNI scanner separability (balanced acc; lower = more invariant) ===")
    for name, task in tasks.items():
        values = "  ".join(f"{v}={task[v]:.3f}" for v in variants)
        print(
            f"  {name} (n={task['n_locations']}, chance {task['chance']:.3f}): "
            f"{values}"
        )

    out_dir = Path(cfg.paths.repo) / "outputs" / "exp07_stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "uni_embedding.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(f"\n[exp07-uni] wrote {out_dir/'uni_embedding.json'}")


if __name__ == "__main__":
    main()
