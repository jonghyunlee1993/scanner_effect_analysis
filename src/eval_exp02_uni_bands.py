"""Exp-02 high-frequency probe: which pyramid band carries the batch effect a
foundation model reads, and can correcting it buy UNI invariance?

The registered AT2 pair lets us TRANSPLANT the true AT2 detail bands at the same
location with zero coefficient-estimation error.  Mixing low and high bands from
different scanners can nevertheless leave the valid image range, so every
variant is audited before UNI's defensive clamp.  We embed the manipulations and
report scanner separability and paired cosine to AT2 for each:

  raw          source unchanged
  low_corr     our low-band affine+local correction, source detail copied (Track-1)
  low_oracle   AT2 low transplanted, source detail kept  (perfect low fix; control)
  hi_oracle    source low kept, AT2 detail transplanted   (paired-band probe)
  corr_low_at2_high  our low correction + AT2 detail      (hybrid, not full oracle)
  full_oracle  exact paired AT2 image                      (positive control)
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

from prenorm.exp01.data import BalancedPairDataset
from prenorm.exp01.frequency import FixedLaplacianPyramid
from prenorm.exp02.projection import local_range_project
from prenorm.embedding import embed_uni, load_uni, scanner_probe
from prenorm.data.identity import (
    lattice_task_name,
    location_key,
    matched_indices_by_lattice,
)
from utils.config import load_config


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
        "corr_low_at2_high": pyramid.reconstruct(low_c, bands_a),
        # Exact reference, intentionally bypassing a decompose/reconstruct
        # round-trip. This must collapse a properly paired scanner probe to chance.
        "full_oracle": reference,
    }
    for lam in ATTEN:
        out[f"atten{lam:.2f}"] = pyramid.reconstruct(low_c, [lam * b for b in bands_s])
    return out


@torch.no_grad()
def range_audit(images, tolerance=1e-6):
    """Per-image range violations before UNI's defensive input clamp."""
    excess = (images.abs() - 1.0).clamp_min(0)
    outside = excess > tolerance
    return {
        "pixel_fraction": outside.float().flatten(1).mean(1).cpu().numpy(),
        "max_excess": excess.flatten(1).amax(1).cpu().numpy(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp01a_lf16_10slide.yaml")
    parser.add_argument("--baselines", default="outputs/exp01_baselines")
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
    names = [
        "raw",
        "low_corr",
        "low_oracle",
        "hi_oracle",
        "corr_low_at2_high",
        "full_oracle",
        *(f"atten{lam:.2f}" for lam in ATTEN),
    ]

    def emb(images):
        return embed_uni(model, images, size, mean, std, device,
                         batch_size=args.batch_size).numpy()

    store = {s: {n: [] for n in names + ["at2", "loc"]} for s in scanners}
    range_values = {
        scanner: {
            name: {"pixel_fraction": [], "max_excess": []}
            for name in names
        }
        for scanner in scanners
    }
    for selection in selections.values():
        for scanner in selection["scanners"]:
            idx = selection["indices"][scanner]
            for start in range(0, len(idx), args.batch_size):
                items = [dataset[i] for i in idx[start:start + args.batch_size]]
                source = torch.stack([it["source"] for it in items])
                reference = torch.stack([it["reference"] for it in items])
                generated = variants(
                    pyramid, source, reference, baseline[scanner], args.steps
                )
                for name in names:
                    audit = range_audit(generated[name])
                    store[scanner][name].append(emb(generated[name]))
                    for metric, values in audit.items():
                        range_values[scanner][name][metric].extend(values.tolist())
                store[scanner]["at2"].append(emb(reference))
                store[scanner]["loc"].extend(
                    location_key(item).group_token() for item in items
                )

    for s in scanners:
        for n in names + ["at2"]:
            store[s][n] = np.concatenate(store[s][n])
        store[s]["loc"] = np.asarray(store[s]["loc"])

    def paired_cos(a, b):
        return float((a * b).sum(1).mean())
    paired = {n: {s: paired_cos(store[s][n], store[s]["at2"]) for s in scanners}
              for n in names}

    def probe(name, lattice_scanners):
        X, y, groups = [], [], []
        for scanner in lattice_scanners:
            X.append(store[scanner][name])
            y += [scanner] * len(store[scanner][name])
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
            **{name: probe(name, lattice_scanners) for name in names},
        }

    range_report = {}
    for scanner in scanners:
        range_report[scanner] = {}
        for name in names:
            pixel_fraction = np.asarray(
                range_values[scanner][name]["pixel_fraction"], dtype=np.float64
            )
            max_excess = np.asarray(
                range_values[scanner][name]["max_excess"], dtype=np.float64
            )
            range_report[scanner][name] = {
                "mean_pixel_fraction_outside": float(pixel_fraction.mean()),
                "fraction_images_outside": float((pixel_fraction > 0).mean()),
                "max_excess": float(max_excess.max()),
            }

    report = {
        "split": args.split,
        "variants": names,
        "sample_identity": {
            "location_key": ["lattice_id", "slide_id", "tuple_id"],
            "cross_lattice_joint_probe": False,
        },
        "paired_cosine_to_at2": paired,
        "probe": tasks,
        "range_audit_pre_embedding_clamp": range_report,
    }
    print("\n=== UNI scanner separability by band manipulation (lower = invariant) ===")
    header = "  " + "variant".ljust(22)
    for task_name in tasks:
        header += f" {task_name:>30s}"
    header += "   meanCosAT2"
    print(header)
    for name in names:
        row = f"  {name:22s}"
        for task in tasks.values():
            row += f" {task[name]:30.3f}"
        cosine = float(np.mean([paired[name][scanner] for scanner in scanners]))
        print(f"{row}   {cosine:10.3f}")
    print("  chance: " + "  ".join(
        f"{task_name}={task['chance']:.3f}" for task_name, task in tasks.items()
    ))

    out_dir = Path(cfg.paths.repo) / "outputs" / "exp02_stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "uni_bands.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(f"\n[exp02-uni-bands] wrote {out_dir/'uni_bands.json'}")


if __name__ == "__main__":
    main()
