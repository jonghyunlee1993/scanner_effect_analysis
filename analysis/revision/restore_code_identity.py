#!/usr/bin/env python3
"""RV-P0d / RV06: link recovered or current source files to the code that actually ran.

The producers of several frozen outputs were moved to `.Trash/` or renamed after they
ran. Python byte-code caches written at run time survive in `.Trash/.../__pycache__/`.
For each producer this script

1. reads the source mtime and size stored in the cache header (the source version the
   cache was compiled from), and
2. compiles the candidate source (recovered file, current renamed file, or the
   `analysis/revision/restore_*.py` rewrite) and compares code objects function by
   function: byte code, names, variables, flags and constants, ignoring file names and
   line numbers.

A cache can only be unmarshalled by the Python version that wrote it, so the SLURM job
runs this script twice: with the project `cpath` Python 3.9 (cp39 caches, used by the
analysis jobs) and with Python 3.13 (cp313 caches). Only the standard library is used.
Outputs: `results/provenance_restoration/code_identity_cp<ver>.csv` and a text file with
the disassembly differences.
"""

from __future__ import annotations

import csv
import datetime as dt
import difflib
import dis
import marshal
import struct
import sys
import types
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
REVISION = Path(__file__).resolve().parent
OUTPUT = REVISION / "results/provenance_restoration"
TRASH_REFACTOR = ".Trash/2026-09-26_paper_refactor/generated"
TRASH_CLEANUP = ".Trash/2026-09-29_paper_code_cleanup"
VERSION = f"{sys.version_info.major}{sys.version_info.minor}"

# (check id, asset, cache stem path without ".cpython-XY.pyc", candidate sources, note)
# Each candidate is (label, path, functions to compare or None for all functions).
CHECKS = [
    ("scanner_tissue_space", "P0d(i) Fig. 2 tissue/core positions",
     f"{TRASH_REFACTOR}/analysis/paper/__pycache__/scanner_tissue_space",
     [("recovered", ".Trash/2026-09-26_paper_refactor/unused_analysis/analysis/paper/scanner_tissue_space.py", None),
      ("restore", "analysis/revision/restore_scanner_tissue_space.py",
       ("rank_summary", "variance_partition", "main"))]),
    ("scanner_color_frequency_space", "P0d(i) context: axis summary producer named in analysis/paper/README.md",
     f"{TRASH_REFACTOR}/analysis/paper/__pycache__/scanner_color_frequency_space",
     [("current", "analysis/paper/scanner_color_frequency_space.py", None)]),
    ("plism_scanner_axis_transfer", "P0d(i) context: PLISM axis producer named in analysis/paper/README.md",
     f"{TRASH_REFACTOR}/analysis/paper/__pycache__/plism_scanner_axis_transfer",
     [("current", "analysis/paper/plism_scanner_axis_transfer.py", None)]),
    ("review_plism_gan_crossencoder", "P0d(ii)/RV06(b) PLISM GAN cross-PFM extraction",
     f"{TRASH_REFACTOR}/scripts/__pycache__/review_plism_gan_crossencoder",
     [("recovered", f"{TRASH_CLEANUP}/scripts/review_plism_gan_crossencoder.py", None),
      ("restore", "analysis/revision/restore_plism_gan_crossencoder.py",
       ("load_inputs", "checkpoint_rows", "run"))]),
    ("review_plism_gan_crossencoder_aggregate", "P0d(ii)/RV06(b) PLISM GAN cross-PFM aggregation",
     f"{TRASH_REFACTOR}/scripts/__pycache__/review_plism_gan_crossencoder_aggregate",
     [("recovered", f"{TRASH_CLEANUP}/scripts/review_plism_gan_crossencoder_aggregate.py", None),
      ("restore", "analysis/revision/restore_plism_gan_crossencoder.py",
       ("seed", "load", "section_values", "verify_baseline", "summarize", "aggregate_main"))]),
    ("review_multiencoder_scanner", "P0d(ii) encoder loading/embedding helper",
     f"{TRASH_REFACTOR}/scripts/__pycache__/review_multiencoder_scanner",
     [("current", "scripts/features/review_multiencoder_scanner.py",
       ("load_encoder", "embed", "cosine_distance"))]),
    ("scanner_gan_predict", "P0d(ii)/(iii) generator loading and prediction (cache is the 2026-09-25 23:43 edit, after both runs)",
     f"{TRASH_REFACTOR}/src/scanner_gan/__pycache__/predict",
     [("current", "src/scanner_gan/predict.py", None)]),
    ("discussion_structure_train", "P0d(iii) GT450 edge-constraint training",
     f"{TRASH_REFACTOR}/src/manuscript_completion/__pycache__/discussion_structure_train",
     [("recovered", f"{TRASH_CLEANUP}/src/manuscript_completion/discussion_structure_train.py", None),
      ("restore", "analysis/revision/restore_structure_ablation_train.py", None)]),
    ("discussion_structure_compare", "P0d(iii) paired slide contrasts and tissue retrieval",
     f"{TRASH_REFACTOR}/src/manuscript_completion/__pycache__/discussion_structure_compare",
     [("recovered", f"{TRASH_CLEANUP}/src/manuscript_completion/discussion_structure_compare.py", None),
      ("restore", "analysis/revision/restore_structure_ablation_compare.py", None)]),
    ("discussion_highband_audit", "P0d(iii) high-band amplitude audit",
     f"{TRASH_REFACTOR}/src/manuscript_completion/__pycache__/discussion_highband_audit",
     [("recovered", f"{TRASH_CLEANUP}/src/manuscript_completion/discussion_highband_audit.py", None),
      ("restore", "analysis/revision/restore_structure_ablation_highband.py", None)]),
    ("discussion_feature_compare", "RV06(c) internal feature correction (renamed feature_uni_v1.py; helpers moved to prenorm.feature_correction)",
     f"{TRASH_REFACTOR}/src/manuscript_completion/__pycache__/discussion_feature_compare",
     [("current", "src/manuscript_completion/feature_uni_v1.py", ("main",)),
      ("current", "src/prenorm/feature_correction.py",
       ("unit", "paired_distance", "fit_affine", "apply_affine", "fit_procrustes",
        "fit_combat", "macro_retrieval"))]),
    ("discussion_feature_external", "RV06(c) PLISM transfer of feature maps (renamed feature_uni_external.py)",
     f"{TRASH_REFACTOR}/src/manuscript_completion/__pycache__/discussion_feature_external",
     [("current", "src/manuscript_completion/feature_uni_external.py", None)]),
    ("discussion_feature_panel40", "RV06(c) 40-location training panel (renamed feature_panel40.py)",
     f"{TRASH_REFACTOR}/src/manuscript_completion/__pycache__/discussion_feature_panel40",
     [("current", "src/manuscript_completion/feature_panel40.py", None)]),
    ("discussion_existing_review", "RV06(d) GT450 direction contrasts (CycleGAN internal contrast)",
     f"{TRASH_REFACTOR}/src/manuscript_completion/__pycache__/discussion_existing_review",
     [("recovered", f"{TRASH_CLEANUP}/src/manuscript_completion/discussion_existing_review.py", None)]),
    ("pfm_problem", "RV06(c) cohort/fold contract and 20-location embedding loader",
     f"{TRASH_REFACTOR}/src/manuscript_completion/__pycache__/pfm_problem",
     [("current", "src/manuscript_completion/pfm_problem.py", None)]),
    ("frequency_aggregate", "RV06(a) aggregation. cp39 cache = 2026-09-18 22:23 EDT source (v5, after the matched run); cp313 cache = 2026-09-17 23:56 EDT source, 5 min before the v3 freeze used by the matched run",
     f"{TRASH_REFACTOR}/src/manuscript_completion/__pycache__/frequency_aggregate",
     [("current", "src/manuscript_completion/frequency_aggregate.py", None)]),
    ("perturbation", "RV06(a) symmetric filters. cp39 cache = 2026-09-18 22:18 EDT source (v4/v5); cp313 cache = 2026-09-17 23:56 EDT source (pre-v3 freeze)",
     f"{TRASH_REFACTOR}/src/manuscript_completion/__pycache__/perturbation",
     [("current", "src/manuscript_completion/perturbation.py", None)]),
]

ATTRIBUTES = ("co_code", "co_names", "co_varnames", "co_argcount", "co_kwonlyargcount",
              "co_posonlyargcount", "co_freevars", "co_cellvars", "co_flags")


def read_cache(path: Path) -> tuple[types.CodeType, float, int]:
    data = path.read_bytes()
    if struct.unpack("<I", data[4:8])[0] != 0:
        raise ValueError(f"hash-based cache not supported: {path}")
    mtime, size = struct.unpack("<II", data[8:16])
    return marshal.loads(data[16:]), float(mtime), int(size)


def functions(code: types.CodeType, prefix: str = "") -> dict[str, types.CodeType]:
    found = {}
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            found[prefix + const.co_name] = const
            found.update(functions(const, prefix + const.co_name + "."))
    return found


def differences(a: types.CodeType, b: types.CodeType, path: str = "") -> list[str]:
    diffs = [path + name for name in ATTRIBUTES
             if getattr(a, name, None) != getattr(b, name, None)]
    consts_a, consts_b = list(a.co_consts), list(b.co_consts)
    if len(consts_a) != len(consts_b):
        return diffs + [path + "n_consts"]
    for x, y in zip(consts_a, consts_b):
        if isinstance(x, types.CodeType) and isinstance(y, types.CodeType):
            diffs += differences(x, y, path + x.co_name + ".")
        elif x != y:
            diffs.append(path + f"const {x!r:.120} -> {y!r:.120}")
    return diffs


def listing(code: types.CodeType) -> list[str]:
    lines = []
    for instruction in dis.get_instructions(code):
        lines.append(f"{instruction.opname} {instruction.argrepr}")
    return lines


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, diff_text = [], []
    for check_id, asset, stem, candidates in CHECKS:
        cache = PROJECT / f"{stem}.cpython-{VERSION}.pyc"
        if not cache.is_file():
            continue
        cached, cache_mtime, cache_size = read_cache(cache)
        cached_functions = functions(cached)
        for label, relative, selected in candidates:
            source = PROJECT / relative
            base = {
                "python": VERSION, "check_id": check_id, "asset": asset,
                "cache": str(cache.relative_to(PROJECT)),
                "cache_source_mtime": dt.datetime.fromtimestamp(cache_mtime).isoformat(),
                "cache_source_bytes": cache_size, "candidate_label": label,
                "candidate": relative,
            }
            if not source.is_file():
                rows.append({**base, "scope": "file", "status": "candidate_missing",
                             "differences": ""})
                continue
            text = source.read_text()
            stat = source.stat()
            base.update({
                "candidate_mtime": dt.datetime.fromtimestamp(int(stat.st_mtime)).isoformat(),
                "candidate_bytes": stat.st_size,
                "header_size_and_mtime_match": bool(stat.st_size == cache_size
                                                    and int(stat.st_mtime) == int(cache_mtime)),
            })
            compiled = compile(text, str(source), "exec", dont_inherit=True)
            compiled_functions = functions(compiled)
            if selected is None:
                module_diff = differences(cached, compiled)
                rows.append({**base, "scope": "module (all code)",
                             "status": "identical" if not module_diff else "differs",
                             "differences": " | ".join(module_diff)})
                names = [name for name in cached_functions if "." not in name]
            else:
                names = list(selected)
            for name in names:
                # restore_plism_gan_crossencoder.py renames the aggregate's main.
                cached_name = {"aggregate_main": "main"}.get(name, name)
                old = cached_functions.get(cached_name)
                new = compiled_functions.get(name)
                if old is None or new is None:
                    rows.append({**base, "scope": name, "status": "function_missing",
                                 "differences": f"cached={old is not None} candidate={new is not None}"})
                    continue
                diff = differences(old, new, "")
                status = "identical" if not diff else "differs"
                rows.append({**base, "scope": name, "status": status,
                             "differences": " | ".join(diff)})
                if diff:
                    body = difflib.unified_diff(listing(old), listing(new),
                                                fromfile=f"{cache.name}:{cached_name}",
                                                tofile=f"{relative}:{name}", lineterm="", n=2)
                    diff_text.append("\n".join(body))
    columns = ["python", "check_id", "asset", "cache", "cache_source_mtime", "cache_source_bytes",
               "candidate_label", "candidate", "candidate_mtime", "candidate_bytes",
               "header_size_and_mtime_match", "scope", "status", "differences"]
    with (OUTPUT / f"code_identity_cp{VERSION}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    (OUTPUT / f"code_identity_cp{VERSION}_disassembly_diffs.txt").write_text(
        "\n\n".join(diff_text) + "\n")
    for row in rows:
        print(row["check_id"], row["candidate_label"], row["scope"], row["status"],
              row["differences"][:160])


if __name__ == "__main__":
    main()
