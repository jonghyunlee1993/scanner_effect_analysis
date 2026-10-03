#!/usr/bin/env python3
"""RV16 stage 3 (CPU): gate, primary endpoints, decision rule and secondary endpoints.

Protocol: ``scanner_composition_pilot_protocol.md``. Reads ``embeddings/<model>.h5`` from
``scanner_composition_embed.py``. Statistics: location means within slide and direction, the
five directions equally weighted within slide, mean over the 21 held-out slides; 95% CIs from
2,000 bootstrap resamples of held-out slides (seed 20260924), one resample matrix shared by all
statistics and models so that differences are paired.

Primary endpoints are divided by each model's between-tissue distance (protocol amendment D1).

Outputs (``results/scanner_composition_pilot/metrics/``): ``model_statistics.csv``,
``arm_effects.csv``, ``decision.json``, ``summary.md``, ``endpoints_by_epoch.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
REVISION = Path(__file__).resolve().parent
for _path in (PROJECT, PROJECT / "src", REVISION):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import corrected_embeddings_common as C  # noqa: E402

OUTPUT = REVISION / "results/scanner_composition_pilot"
ARMS = ("A", "B", "A1", "B1")
EPOCHS = (10, 25, 50, 100)
PRIMARY_EPOCH = 25  # amendment D2
CONFIRM_EPOCH = 50  # amendment D2
TARGETS = ("versa", "akoya", "gt450", "s360", "s60")
SCANNERS = ("at2",) + TARGETS
GATE_SCANNERS = ("versa", "akoya", "s360", "s60")
BANDS = ("low_mid", "mid", "high")
DOSES = (0.25, 0.5)
K = 15
N_BOOT = 2000
SEED = 20260924
EXPECTED = {"P1": None, "P2": -1, "P3": 1}
LABELS = {"P1": "High minus low-mid shift (dose 0.25)",
          "P2": "High-band shift, increase minus decrease (dose 0.25)",
          "P3": "GT450: colour + frequency gain over Reinhard"}


# ----------------------------------------------------------------------------- loading
def load(root: Path):
    held = pd.read_csv(root / "embeddings/held_out_locations.csv", dtype={"slide_id": str})
    bank = pd.read_csv(root / "embeddings/bank_locations.csv", dtype={"slide_id": str})
    models = {}
    names = None
    for tag in ["init_e000"] + [f"{a}_e{e:03d}" for a in ARMS for e in EPOCHS]:
        path = root / "embeddings" / f"{tag}.h5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as store:
            models[tag] = {"held": np.asarray(store["held_out"], dtype=np.float32),
                           "bank": np.asarray(store["bank"], dtype=np.float32)}
            current = C.decode(store["condition_names"][:])
        if names is not None and names != current:
            raise ValueError("condition order differs between models")
        names = current
    band = pd.concat([pd.read_csv(p, dtype={"slide_id": str}) for p in sorted((root / "render").glob("*_band.csv.gz"))],
                     ignore_index=True)
    return held, bank, models, names, band


# ----------------------------------------------------------------------------- per-location values
def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - np.sum(a * b, axis=-1)


def location_values(feat: np.ndarray, names: list[str], held: pd.DataFrame, band: pd.DataFrame) -> pd.DataFrame:
    """One row per (location, direction, statistic) with the location-level value."""
    idx = {name: i for i, name in enumerate(names)}
    rows = []
    position = {(s, int(l)): i for i, (s, l) in enumerate(zip(held.slide_id, held.location_index))}
    for scanner in TARGETS:
        base = feat[:, idx[f"reinhard_to_{scanner}"]]
        target = feat[:, idx[f"target_{scanner}"]]
        base_distance = distance(base, target)
        frame = {"slide_id": held.slide_id.to_numpy(), "location_index": held.location_index.to_numpy(), "scanner": scanner}
        for dose in DOSES:
            for b in BANDS:
                for sign, tag in ((-1, "dec"), (1, "inc")):
                    changed = feat[:, idx[f"band_{b}_{tag}_d{dose:g}_to_{scanner}"]]
                    rows.append(pd.DataFrame({**frame, "statistic": f"shift_{b}_{tag}_d{dose:g}",
                                              "value": distance(base, changed)}))
                    rows.append(pd.DataFrame({**frame, "statistic": f"gain_{b}_{tag}_d{dose:g}",
                                              "value": base_distance - distance(changed, target)}))
        source = feat[:, idx["source_at2"]]
        raw = distance(source, target)
        for method in ("reinhard", "frequency", "combined"):
            rows.append(pd.DataFrame({**frame, "statistic": f"gain_{method}",
                                      "value": raw - distance(feat[:, idx[f"{method}_to_{scanner}"]], target)}))
        rows.append(pd.DataFrame({**frame, "statistic": "increment_combined_over_reinhard",
                                  "value": distance(base, target) - distance(feat[:, idx[f"combined_to_{scanner}"]], target)}))
        rows.append(pd.DataFrame({**frame, "statistic": "raw_target_distance", "value": raw}))
    values = pd.concat(rows, ignore_index=True)
    # target-direction flag of each high-band row (RV02 definition) for the target-direction gain
    like = band[band.dose_fraction.eq(0.25) & band.band.eq("high")]
    like = like.assign(statistic=np.where(like.sign > 0, "gain_high_inc_d0.25", "gain_high_dec_d0.25"))
    like = like[like.target_like][["slide_id", "location_index", "scanner", "statistic"]]
    target_dir = values.merge(like, on=["slide_id", "location_index", "scanner", "statistic"], how="inner")
    target_dir = target_dir.assign(statistic="target_gain_high_d0.25")
    if len(target_dir) != len(held) * len(TARGETS):
        raise ValueError(f"target-direction rows {len(target_dir)}")
    return pd.concat([values, target_dir], ignore_index=True)


def derived(values: pd.DataFrame) -> pd.DataFrame:
    """Location-level composite statistics (computed before averaging; all linear)."""
    wide = values.pivot_table(index=["slide_id", "location_index", "scanner"], columns="statistic", values="value")
    out = {}
    for dose in DOSES:
        d = f"d{dose:g}"
        for b in BANDS:
            out[f"shift_{b}_{d}"] = (wide[f"shift_{b}_inc_{d}"] + wide[f"shift_{b}_dec_{d}"]) / 2
        out[f"P1_{d}"] = out[f"shift_high_{d}"] - out[f"shift_low_mid_{d}"]
        out[f"P2_{d}"] = wide[f"shift_high_inc_{d}"] - wide[f"shift_high_dec_{d}"]
    frame = pd.DataFrame(out)
    frame["increment_combined_over_reinhard"] = wide["increment_combined_over_reinhard"]
    for column in ("gain_reinhard", "gain_frequency", "gain_combined", "raw_target_distance", "target_gain_high_d0.25"):
        frame[column] = wide[column]
    return frame.reset_index().melt(id_vars=["slide_id", "location_index", "scanner"], var_name="statistic")


def slide_vectors(long: pd.DataFrame, slides: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    """(statistic, direction) -> slide vector; direction 'all' = equal-weight mean of the five."""
    by = long.groupby(["statistic", "scanner", "slide_id"]).value.mean()
    vectors = {}
    for statistic in long.statistic.unique():
        part = by.loc[statistic]
        per_scanner = {s: part.loc[s].reindex(slides).to_numpy(float) for s in TARGETS}
        for s, v in per_scanner.items():
            vectors[(statistic, s)] = v
        vectors[(statistic, "all")] = np.mean(np.stack(list(per_scanner.values())), axis=0)
    return vectors


def normalizer(pair_means: np.ndarray, slide_tissue: np.ndarray, draws: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean distance over slide pairs of different tissue types, for all slides and every resample."""
    different = slide_tissue[:, None] != slide_tissue[None, :]

    def value(index):
        sub, mask = pair_means[np.ix_(index, index)], different[np.ix_(index, index)]
        return float(sub[mask].mean()) if mask.any() else float("nan")

    return value(np.arange(len(slide_tissue))), np.array([value(d) for d in draws])


# ----------------------------------------------------------------------------- tissue kNN
def knn_correct(query: np.ndarray, query_tissue: np.ndarray, bank: np.ndarray, bank_tissue: np.ndarray) -> np.ndarray:
    sims = query @ bank.T
    order = np.argsort(-sims, axis=1)[:, :K]
    correct = np.empty(len(query), dtype=float)
    for i in range(len(query)):
        labels = bank_tissue[order[i]]
        scores = {}
        for rank, label in enumerate(labels):
            count, weight = scores.get(label, (0, 0.0))
            scores[label] = (count + 1, weight + sims[i, order[i, rank]])
        winner = max(scores.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
        correct[i] = float(winner == query_tissue[i])
    return correct


def balanced(correct: np.ndarray, held: pd.DataFrame, slides: np.ndarray, draws: np.ndarray):
    """Balanced accuracy over held-out tissues: slide accuracies averaged within tissue, then over tissues."""
    frame = held.assign(correct=correct).groupby("slide_id").agg(acc=("correct", "mean"), tissue=("tissue_type", "first"))
    frame = frame.reindex(slides)
    acc, tissue = frame.acc.to_numpy(float), frame.tissue.to_numpy()
    codes, inverse = np.unique(tissue, return_inverse=True)

    def value(index):
        a, t = acc[index], inverse[index]
        sums = np.bincount(t, weights=a, minlength=len(codes))
        counts = np.bincount(t, minlength=len(codes))
        present = counts > 0
        return float(np.mean(sums[present] / counts[present]))

    point = value(np.arange(len(slides)))
    boot = np.array([value(d) for d in draws])
    return point, boot


# ----------------------------------------------------------------------------- main
def summarize(point: float, boot: np.ndarray) -> dict:
    """Point estimate and 95% percentile CI; resamples without a defined value (e.g. no pair of
    different tissue types) are dropped and counted."""
    boot = np.asarray(boot, dtype=float)
    finite = np.isfinite(boot)
    if not finite.any():
        return {"mean": float(point), "ci_low": float("nan"), "ci_high": float("nan"), "undefined_resamples": int(boot.size)}
    return {"mean": float(point), "ci_low": float(np.quantile(boot[finite], 0.025)),
            "ci_high": float(np.quantile(boot[finite], 0.975)), "undefined_resamples": int((~finite).sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--output-name", default="metrics", help="subdirectory for this analysis run")
    args = parser.parse_args()
    root = args.output_root
    out = root / args.output_name
    held, bank, models, names, band = load(root)
    slides = np.asarray(sorted(held.slide_id.unique()))
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(slides), size=(N_BOOT, len(slides)))
    held_tissue = held.tissue_type.to_numpy()
    bank_tissue = bank.tissue_type.to_numpy()
    raw_index = {s: names.index("source_at2" if s == "at2" else f"target_{s}") for s in SCANNERS}
    slide_codes = pd.Categorical(held.slide_id, categories=slides).codes
    slide_tissue = held.groupby("slide_id").tissue_type.first().reindex(slides).to_numpy()

    vectors: dict[str, dict] = {}
    records = []
    for tag, feats in models.items():
        long = derived(location_values(feats["held"], names, held, band))
        vec = slide_vectors(long, slides)
        # between-tissue distance (amendment D1): raw AT2 and raw GT450 held-out embeddings, location
        # pairs of different tissue types, averaged over the two scanners; slide-pair means allow
        # recomputation in every bootstrap resample.
        at2 = feats["held"][:, raw_index["at2"]]
        pair_means = np.zeros((len(slides), len(slides)))
        for s in ("at2", "gt450"):
            x = feats["held"][:, raw_index[s]]
            d = 1.0 - x @ x.T
            pair_means += pd.DataFrame(d, index=slide_codes, columns=slide_codes).groupby(level=0).mean().T.groupby(level=0).mean().reindex(index=range(len(slides)), columns=range(len(slides))).to_numpy() / 2
        between, between_boot = normalizer(pair_means, slide_tissue, draws)
        # normalized AT2-GT450 distance of raw pairs
        pair = distance(at2, feats["held"][:, raw_index["gt450"]])
        vec[("raw_at2_gt450_distance", "gt450")] = pd.Series(pair).groupby(held.slide_id.to_numpy()).mean().reindex(slides).to_numpy()
        # tissue kNN per scanner (query held-out, bank pretraining slides, same scanner)
        knn = {}
        for s in SCANNERS:
            correct = knn_correct(feats["held"][:, raw_index[s]], held_tissue, feats["bank"][:, SCANNERS.index(s)], bank_tissue)
            knn[s] = balanced(correct, held, slides, draws)
        gate_point = float(np.mean([knn[s][0] for s in GATE_SCANNERS]))
        gate_boot = np.mean(np.stack([knn[s][1] for s in GATE_SCANNERS]), axis=0)
        vectors[tag] = {"vec": vec, "between": between, "between_boot": between_boot, "knn": knn,
                        "gate": (gate_point, gate_boot)}
        for (statistic, scanner), v in vec.items():
            records.append({"model": tag, "statistic": statistic, "direction": scanner,
                            **summarize(v.mean(), v[draws].mean(axis=1))})
        for s in SCANNERS:
            records.append({"model": tag, "statistic": "tissue_knn_balanced", "direction": s, **summarize(*knn[s])})
        records.append({"model": tag, "statistic": "gate_knn_unseen_scanners", "direction": "versa+akoya+s360+s60",
                        **summarize(gate_point, gate_boot)})
        records.append({"model": tag, "statistic": "between_tissue_distance", "direction": "at2+gt450",
                        **summarize(between, between_boot)})
        print(f"{tag}: between {between:.4f}, gate kNN {gate_point:.3f}", flush=True)
    model_stats = pd.DataFrame(records)
    C.write_frame(out / "model_statistics.csv", model_stats)

    # endpoint vectors
    def endpoint(tag: str, name: str) -> np.ndarray:
        vec = vectors[tag]["vec"]
        if name == "P1":
            return vec[("P1_d0.25", "all")]
        if name == "P2":
            return vec[("P2_d0.25", "all")]
        if name == "P3":
            return vec[("increment_combined_over_reinhard", "gt450")]
        raise KeyError(name)

    def level(tag: str, name: str, scale: str) -> tuple[float, np.ndarray]:
        """Endpoint point estimate and bootstrap values; 'norm' divides by between-tissue distance."""
        v = endpoint(tag, name)
        point, boot = v.mean(), v[draws].mean(axis=1)
        if scale == "norm":
            point, boot = point / vectors[tag]["between"], boot / vectors[tag]["between_boot"]
        return float(point), boot

    effects = []
    available_epochs = [e for e in EPOCHS if all(f"{a}_e{e:03d}" in vectors for a in ARMS)]
    for epoch in available_epochs:
        t = {a: f"{a}_e{epoch:03d}" for a in ARMS}
        for scale in ("norm", "raw"):
            for name in ("P1", "P2", "P3"):
                for label, (x, y) in {"arm_seed0": ("B", "A"), "arm_seed1": ("B1", "A1"),
                                      "noise_at2": ("A1", "A"), "noise_gt450": ("B1", "B")}.items():
                    px, bx = level(t[x], name, scale)
                    py, by = level(t[y], name, scale)
                    effects.append({"epoch": epoch, "scale": scale, "endpoint": name, "comparison": label,
                                    "pair": f"{x}-{y}", **summarize(px - py, bx - by)})
                for arm in ARMS:
                    effects.append({"epoch": epoch, "scale": scale, "endpoint": name, "comparison": f"level_{arm}",
                                    "pair": arm, **summarize(*level(t[arm], name, scale))})
    if "init_e000" in vectors:
        for name in ("P1", "P2", "P3"):
            effects.append({"epoch": 0, "scale": "raw", "endpoint": name, "comparison": "level_init", "pair": "init",
                            **summarize(*level("init_e000", name, "raw"))})
    effects = pd.DataFrame(effects)
    C.write_frame(out / "arm_effects.csv", effects)

    # gate and decision at the primary epoch
    decision = {"primary_epoch": PRIMARY_EPOCH, "available_epochs": available_epochs}
    if PRIMARY_EPOCH in available_epochs and "init_e000" in vectors:
        init_point, init_boot = vectors["init_e000"]["gate"]
        gate = {}
        for arm in ARMS:
            point, boot = vectors[f"{arm}_e{PRIMARY_EPOCH:03d}"]["gate"]
            gate[arm] = summarize(point - init_point, boot - init_boot)
            gate[arm]["passes"] = gate[arm]["ci_low"] > 0
        decision["gate"] = {"improvement_over_step0": gate, "passes": all(g["passes"] for g in gate.values())}
        primary = rule(effects, PRIMARY_EPOCH)
        decision["primary"] = primary
        decision["go"] = bool(decision["gate"]["passes"] and any(p["reproduced"] for p in primary.values()))
        if CONFIRM_EPOCH in available_epochs:
            confirm = rule(effects, CONFIRM_EPOCH)
            for name, p in primary.items():
                confirm[name]["confirms_primary"] = bool(p["reproduced"] and confirm[name]["reproduced"]
                                                         and confirm[name]["sign"] == p["sign"])
            decision["confirmation"] = {"epoch": CONFIRM_EPOCH, "endpoints": confirm}
    C.write_json(out / "decision.json", decision)
    write_summary(out, model_stats, effects, decision, vectors)
    plot(out, effects)
    print(json.dumps({"status": "complete", "go": decision.get("go")}), flush=True)


def rule(effects: pd.DataFrame, epoch: int) -> dict:
    """Pre-specified decision rule on the normalized endpoints at one epoch."""
    if True:
        primary = {}
        at = effects[effects.epoch.eq(epoch) & effects.scale.eq("norm")].set_index(["endpoint", "comparison"])
        for name in ("P1", "P2", "P3"):
            a0, a1 = at.loc[(name, "arm_seed0")], at.loc[(name, "arm_seed1")]
            n0, n1 = at.loc[(name, "noise_at2")], at.loc[(name, "noise_gt450")]
            same_sign = np.sign(a0["mean"]) == np.sign(a1["mean"])
            excludes = all(r["ci_low"] > 0 or r["ci_high"] < 0 for r in (a0, a1))
            above_noise = min(abs(a0["mean"]), abs(a1["mean"])) > max(abs(n0["mean"]), abs(n1["mean"]))
            sign = int(np.sign(a0["mean"])) if same_sign else 0
            primary[name] = {"label": LABELS[name], "arm_effect_seed0": float(a0["mean"]), "arm_effect_seed1": float(a1["mean"]),
                             "noise_at2": float(n0["mean"]), "noise_gt450": float(n1["mean"]),
                             "same_sign": bool(same_sign), "both_ci_exclude_zero": bool(excludes),
                             "above_noise": bool(above_noise), "reproduced": bool(same_sign and excludes and above_noise),
                             "sign": sign, "expected_sign": EXPECTED[name],
                             "matches_expectation": None if EXPECTED[name] is None else bool(sign == EXPECTED[name])}
    return primary


def write_summary(out: Path, stats: pd.DataFrame, effects: pd.DataFrame, decision: dict, vectors: dict) -> None:
    fmt = lambda r: f"{r['mean']:+.5f} [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]"  # noqa: E731
    lines = ["# RV16 scanner-composition pilot: results", "",
             "Protocol: `analysis/revision/scanner_composition_pilot_protocol.md`. Generated by `scanner_composition_metrics.py`.", ""]
    if "gate" in decision:
        lines += ["## Gate (tissue kNN on unseen scanners, improvement over step 0)", "",
                  "| Arm | improvement [95% CI] | passes |", "| --- | --- | --- |"]
        for arm, g in decision["gate"]["improvement_over_step0"].items():
            lines.append(f"| {arm} | {fmt(g)} | {g['passes']} |")
        lines += ["", f"Gate passes: **{decision['gate']['passes']}**", "",
                  f"## Primary endpoints (epoch {decision['primary_epoch']})", "",
                  "| Endpoint | B−A | B′−A′ | A′−A | B′−B | reproduced | sign vs expectation |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, p in decision["primary"].items():
            lines.append(f"| {name}: {p['label']} | {p['arm_effect_seed0']:+.5f} | {p['arm_effect_seed1']:+.5f} | "
                         f"{p['noise_at2']:+.5f} | {p['noise_gt450']:+.5f} | {p['reproduced']} | "
                         f"{'n/a' if p['expected_sign'] is None else p['matches_expectation']} |")
        lines += ["", f"**Decision: {'GO' if decision['go'] else 'NO-GO'}** (epoch {decision['primary_epoch']})", ""]
        if "confirmation" in decision:
            lines += [f"## Confirmation look (epoch {decision['confirmation']['epoch']})", "",
                      "| Endpoint | B−A | B′−A′ | A′−A | B′−B | reproduced | confirms primary |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for name, p in decision["confirmation"]["endpoints"].items():
                lines.append(f"| {name} | {p['arm_effect_seed0']:+.5f} | {p['arm_effect_seed1']:+.5f} | "
                             f"{p['noise_at2']:+.5f} | {p['noise_gt450']:+.5f} | {p['reproduced']} | {p['confirms_primary']} |")
            lines.append("")
    lines += ["## Arm effects with CIs, every epoch (norm = divided by between-tissue distance; primary)", "",
              "| epoch | scale | endpoint | comparison | mean [95% CI] |", "| --- | --- | --- | --- | --- |"]
    for r in effects[~effects.comparison.str.startswith("level")].to_dict("records"):
        lines.append(f"| {r['epoch']} | {r['scale']} | {r['endpoint']} | {r['comparison']} ({r['pair']}) | {fmt(r)} |")
    lines += ["", "## Levels per model (raw; divide by between-tissue for the primary scale)", "",
              "| model | between-tissue | gate kNN | P1 | P2 | P3 |", "| --- | --- | --- | --- | --- | --- |"]
    for tag, v in vectors.items():
        vec = v["vec"]
        lines.append(f"| {tag} | {v['between']:.4f} | {v['gate'][0]:.3f} | {vec[('P1_d0.25', 'all')].mean():+.5f} | "
                     f"{vec[('P2_d0.25', 'all')].mean():+.5f} | {vec[('increment_combined_over_reinhard', 'gt450')].mean():+.5f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def plot(out: Path, effects: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"A": "#3B6EA8", "A1": "#8FB3DE", "B": "#C8553D", "B1": "#E9A08F", "init": "#666666"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, name in zip(axes, ("P1", "P2", "P3")):
        part = effects[effects.endpoint.eq(name) & effects.comparison.str.startswith("level") & effects.scale.eq("norm")]
        for arm in ("A", "A1", "B", "B1"):
            rows = part[part.pair.eq(arm)].sort_values("epoch")
            ax.errorbar(rows.epoch, rows["mean"], yerr=[rows["mean"] - rows.ci_low, rows.ci_high - rows["mean"]],
                        marker="o", ms=4, lw=1.4, capsize=2, color=colors[arm], label=arm)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(LABELS[name] + " / between-tissue", fontsize=9)
        ax.set_xlabel("epoch")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "endpoints_by_epoch.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
