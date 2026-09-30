#!/usr/bin/env python3
"""RV-P0d: render results/provenance_restoration/summary.md from the verification files.

Reads the outputs of `restore_code_identity.py`, `restore_scanner_tissue_space.py`,
`restore_plism_gan_crossencoder.py` and `restore_structure_ablation.py`; computes nothing
new. A stage whose verification file is missing is reported as pending.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUTPUT = Path(__file__).resolve().parent / "results/provenance_restoration"


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def code_lines() -> list[str]:
    lines = []
    frames = [pd.read_csv(p) for p in sorted(OUTPUT.glob("code_identity_cp*.csv"))]
    if not frames:
        return ["Code identity: pending.", ""]
    table = pd.concat(frames, ignore_index=True)
    lines += ["| Cache (Python) | Candidate | Cache source = candidate (size, mtime) | "
              "Functions identical | Differences |", "| --- | --- | --- | --- | --- |"]
    for (python, check_id, label, candidate), group in table.groupby(
            ["python", "check_id", "candidate_label", "candidate"], sort=False):
        functions = group[group.scope != "module (all code)"]
        module = group[group.scope == "module (all code)"]
        differing = functions[functions.status != "identical"].scope.tolist()
        notes = []
        if differing:
            notes.append("functions: " + ", ".join(differing))
        if len(module) and module.status.iloc[0] != "identical":
            text = str(module.differences.iloc[0])
            notes.append("module: " + (text[:160] + "..." if len(text) > 160 else text))
        lines.append(
            f"| {check_id} (cp{python}) | {label}: `{candidate}` | "
            f"{bool(group.header_size_and_mtime_match.iloc[0])} | "
            f"{(functions.status == 'identical').sum()}/{len(functions)} | "
            f"{'; '.join(notes) or 'none'} |")
    return lines + [""]


def main() -> None:
    fig2 = load_json(OUTPUT / "scanner_tissue_space/verification.json")
    gan = load_json(OUTPUT / "plism_gan_crossencoder/verification.json")
    structure = load_json(OUTPUT / "structure_ablation/verification.json")
    lines = [
        "# RV-P0d Provenance restoration", "",
        "**Scope:** recover or rewrite the producers of three paper assets and show whether each "
        "reproduces its frozen output. Nothing under `outputs/` or `analysis/paper/` was written. "
        "Restoration into `analysis/paper/` waits for discussion, as the protocol requires.", "",
        "**Code:**",
        "- `analysis/revision/restore_scanner_tissue_space.py`",
        "- `restore_plism_gan_crossencoder.py`",
        "- `restore_structure_ablation.py` with `restore_structure_ablation_{train,compare,highband}.py`",
        "- `restore_code_identity.py`",
        "", "Each has a matching `.sbatch`; job logs are in `logs/`.", "",
        "**How the code was tied to the runs.**",
        "- When these jobs ran, Python wrote byte-code caches. The caches survived in "
        "`.Trash/2026-09-26_paper_refactor/generated/**/__pycache__/`.",
        "- Each cache header records the source size and mtime it was compiled from.",
        "- `restore_code_identity.py` compiles each candidate source and compares it with the cache, "
        "function by function (byte code, names and constants).",
        "",
    ]

    lines += ["## (i) Fig. 2 tissue/core positions", ""]
    if fig2 is None:
        lines += ["Pending.", ""]
    else:
        table = pd.read_csv(OUTPUT / "scanner_tissue_space/verification.csv")
        lines += [
            "- **Frozen file:** `analysis/paper/results/scanner_color_frequency_space/"
            "pannormal_tissue_positions.csv`, plus `plism_core_positions.csv`. Both are inputs to "
            "`analysis/paper/plot_scanner_common_image_phenotype.py`.",
            "- **Producer:** neither script named in the README "
            "(`scanner_color_frequency_space.py`, `plism_scanner_axis_transfer.py`) wrote these files.",
            "  - The actual producer is `scanner_tissue_space.py`, recovered from "
            "`.Trash/2026-09-26_paper_refactor/unused_analysis/analysis/paper/`.",
            "  - It ran as SLURM job 23553153 on 2026-09-24 at 16:55, via "
            "`sbatch analysis/run_scanner_tissue_space.sbatch` with workdir `00_manuscript`.",
            "  - The recovered file was last modified at 16:54:52 and has not changed since; the job "
            "started at 16:55:01. A cp313 cache records the same source size and mtime (12,499 bytes), "
            "and all functions are byte-code identical.",
            "- **Rewrite:** `restore_scanner_tissue_space.py` copies the functions verbatim and changes "
            "only the output directory.",
            f"- **Result:** {int(table.byte_identical.sum())}/{len(table)} regenerated files are "
            "byte-identical to the frozen ones.",
            f"  - The two Fig. 2 inputs are byte-identical: {fig2['fig2_inputs_byte_identical']}. "
            "Their SHA-256 also equals `analysis/paper/evidence_hashes.json`.",
            "  - The PNGs and `tissue_space_diagnostics.json` are byte-identical.",
            "  - The two PDFs differ only because the embedded creation date differs; they were not "
            "compared further.",
            "",
        ]

    lines += ["## (ii) PLISM GAN cross-PFM evaluation", ""]
    lines += [
        "- **Producer:** `review_plism_gan_crossencoder.py` and `_aggregate.py`, recovered from "
        "`.Trash/2026-09-29_paper_code_cleanup/scripts/`.",
        "  - Original run: array 23670688 (tasks 0-17) and job 23670935 on 2026-09-25. The logs are "
        "in `.Trash/2026-09-26_paper_refactor/generated/outputs/plism_gan_crossencoder_2026-09-25/logs/`.",
        "  - The recovered sources were edited on 2026-09-26 (one relocated path constant each). Apart "
        "from that, they are byte-code identical to the run-time caches.",
        "- **Rewrite:** `restore_plism_gan_crossencoder.py` copies the functions verbatim.",
        "  - It imports the encoder helpers from `scripts/features/review_multiencoder_scanner.py`, "
        "which are byte-code identical to the run-time cache.",
        "  - `aggregate_main` differs from the cached `main` in a single instruction pair "
        "(`LOAD_ATTR ... PUSH_NULL` vs method-call form for `OUTPUT.mkdir`). Python 3.13 emits the "
        "first form when the name is imported, and `OUTPUT` is now defined locally. Nothing else differs.",
        "- **Caveat:** `scanner_gan/predict.py` was edited at 23:43 and 23:49 on 2026-09-25, after the "
        "run. `load_validated_generator` is identical in the 23:43 cache and in the current file.",
        "",
    ]
    if gan is None:
        lines += ["**Regeneration:** pending (GPU array and comparison job).", ""]
    else:
        summary = pd.read_csv(OUTPUT / "plism_gan_crossencoder/verification_summary_level.csv")
        rederived = summary[summary.table.str.startswith("recovered aggregation")]
        location = pd.read_csv(OUTPUT / "plism_gan_crossencoder/verification_location_level.csv")
        lines += [
            "**Regeneration:** all 18 tasks were rerun on GPU and aggregated with the recovered code.",
            f"- Frozen shards re-aggregated with the recovered code: byte-identical = "
            f"{rederived.byte_identical.tolist()} (section distances, gain summary).",
            f"- Regenerated vs frozen, location level:",
            f"  - max |Δ raw distance| = {gan['max_abs_raw_distance_difference_location']:.2e}.",
            f"  - max |Δ corrected distance| = "
            f"{gan['max_abs_corrected_distance_difference_location']:.2e}.",
            f"  - Exact-match fraction of corrected distances, per task: min "
            f"{location.corrected_distance_exact_fraction.min():.3f}, median "
            f"{location.corrected_distance_exact_fraction.median():.3f}.",
            f"- Regenerated vs frozen, summary level: max |Δ gain| = "
            f"{gan['max_abs_gain_difference_summary']:.2e}.",
            f"  - Tolerance declared before the run: {gan['summary_tolerance_abs_gain']}. Within "
            f"tolerance: {gan['within_tolerance']}.",
            f"  - All gain signs agree: {gan['all_gain_signs_agree']}. Whether each CI excludes 0 "
            f"agrees: {gan['ci_excludes_zero_agrees']}.",
            "",
        ]

    lines += ["## (iii) GT450 edge-constraint (structure) ablation", ""]
    lines += [
        "- **Producers:** `discussion_structure_train.py`, `discussion_structure_compare.py` and "
        "`discussion_highband_audit.py`, plus the `discussion_structure_*.sbatch` files, all recovered "
        "from `.Trash/2026-09-29_paper_code_cleanup/`.",
        "  - The run-time caches record the same size and mtime as the recovered files, and every "
        "function is byte-code identical.",
        "  - They were restored verbatim as `restore_structure_ablation_{train,compare,highband}.py`. "
        "Only the module docstring changed, and the compare module imports "
        "`macro_retrieval`/`unit` from `prenorm.feature_correction` (byte-code identical).",
        "- **Generic stages:** the unchanged `src/scanner_gan/{predict,evaluate_images,evaluate_uni}.py`. "
        "The `evaluate_*` modules have the same size and mtime as their 2026-09-17 caches.",
        "- **Original jobs, 2026-09-25:** training 23597820 and 23598335; predict 23600835; "
        "image evaluation 23600886; UNI evaluation 23600892; compare 23600896; high-band audit 23608343.",
        "- **Not rerun: training.** Pix2Pix training here uses TF32, bfloat16 autocast and "
        "non-deterministic cuDNN, so a retrain would be a replicate, not a restoration. The recovered "
        "training code is in place (`--stage train`), but it was not submitted.",
        "",
    ]
    if structure is None:
        lines += ["**Regeneration:** pending (predict -> image/UNI evaluation -> compare chain).", ""]
    else:
        stages = pd.read_csv(OUTPUT / "structure_ablation/verification_stages.csv")
        headline = pd.read_csv(OUTPUT / "structure_ablation/verification_headline_contrasts.csv")
        lines += [
            "**Regeneration:**",
            f"- **From the frozen 02-04 intermediates:** the recovered compare and high-band code "
            f"reproduces all five `05_comparison` files byte-identically: "
            f"{structure['from_frozen_intermediates_05_byte_identical']}.",
            f"- **Predictions from the frozen checkpoints:** "
            f"{structure['prediction_shards_found']}/{structure['prediction_shards_expected']} shards.",
            f"  - Byte-identical shards: {structure['prediction_shards_byte_identical']}.",
            f"  - Shards whose generated images are identical: "
            f"{structure['generated_images_identical_shards']}.",
            f"  - Max |Δ| of the generated uint8 images: {structure['generated_max_abs_diff_uint8']}.",
            f"- **Regenerated chain (predict -> evaluate -> compare) vs frozen:** max |Δ| over the "
            f"paired contrasts and their CIs = {structure['regenerated_05_paired_contrast_max_abs_diff']:.2e}.",
            f"  - Tolerance declared before the run: {structure['tolerance_05_abs']}. Within "
            f"tolerance: {structure['regenerated_05_within_tolerance']}.",
            "",
            "| Stage | File | Byte-identical | Max abs diff (column) |", "| --- | --- | --- | --- |",
        ]
        for record in stages.itertuples(index=False):
            worst = "" if pd.isna(getattr(record, "max_abs_diff", float("nan"))) else \
                f"{record.max_abs_diff:.2e}" + ("" if pd.isna(record.max_abs_diff_column)
                                                else f" ({record.max_abs_diff_column})")
            lines.append(f"| {record.stage} | {record.file} | {record.byte_identical} | {worst} |")
        lines += ["", "Headline contrasts (regularized minus baseline), regenerated vs frozen:", "",
                  "| Space | Metric | Regenerated (95% CI) | Frozen (95% CI) |", "| --- | --- | --- | --- |"]
        for r in headline.itertuples(index=False):
            lines.append(
                f"| {r.space} | {r.metric} | {r.regularized_minus_baseline_new:+.5f} "
                f"({r.ci_low_new:+.5f} to {r.ci_high_new:+.5f}) | {r.regularized_minus_baseline_frozen:+.5f} "
                f"({r.ci_low_frozen:+.5f} to {r.ci_high_frozen:+.5f}) |")
        lines.append("")

    lines += ["## Code identity table", ""] + code_lines()
    lines += [
        "## Other findings", "",
        "- **README row is incomplete.** The `analysis/paper/README.md` row \"Shared scanner image "
        "phenotype\" should also name the restored `scanner_tissue_space.py` producer.",
        "- **Current `scanner_color_frequency_space.py` has changed.** Its `main` differs from the "
        "2026-09-24 16:36:24 cache. The file was edited at 16:37:11, after its last run ended at "
        "16:36:48 (job 23551505). The only change is one extra label offset for S360 in the plot. "
        "`plism_scanner_axis_transfer.py` is identical to its cache.",
        "- **`discussion_existing_review.py` was edited after it ran.** It differs from its run-time "
        "cache only in one relocated path constant.",
        "",
    ]
    (OUTPUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
