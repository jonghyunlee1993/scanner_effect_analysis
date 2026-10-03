# Records of removed work

Analyses, files, and intermediate results that are not part of the current manuscript
were removed from the repository. This folder keeps what is needed to know what they
were and what they showed. Nothing here is used to build the manuscript.

| Record | Content |
| --- | --- |
| [`removed_2026-10-02.tsv`](removed_2026-10-02.tsv) | every path removed on 2026-10-02, with its size, category, and reason |
| [`removed_analyses/`](removed_analyses/) | summaries and small result tables of the removed analyses |
| [`restore_maps/`](restore_maps/) | path maps of the earlier cleanups (2026-09-26 trash, 2026-09-29 code cleanup, relocations, former `scripts/` entry points); the trash itself was deleted on 2026-10-02, so these maps now record history only |
| [`paper_notes/`](paper_notes/) | dated review notes behind the band, discussion follow-up, feature, and GAN cross-encoder analyses |

## Analyses removed on 2026-10-02

Protocols and outcomes are also described in `analysis/revision/README.md`.

| Analysis | Question | Outcome | Why removed |
| --- | --- | --- | --- |
| RV15 reference encoders | Is PLIP's scanner-organized representation inherited from CLIP, and where do CLIP and ResNet50 fall on the sensitivity–robustness trend? | PLIP's same-scanner neighborhoods were largely inherited from CLIP; every reference encoder lay below the eight-model robustness trend (`removed_analyses/reference_encoders_rv15/summary.md`) | not reported in the manuscript |
| RV16 scanner-composition pilot | Does the scanner composition of pretraining data change band sensitivity (ViT-S/14 trained from scratch)? | NO-GO with exploratory signals (`removed_analyses/scanner_composition_pilot/`) | pilot; superseded by RV18 and RV19 |
| RV17 scanner-mixture pilot | Do endpoints follow the AT2 share of pretraining data (ViT-Tiny, four shares)? | GO for two endpoints, single seed (`removed_analyses/scanner_mixture/`) | pilot; superseded by RV18 and RV19 |
| RV19 epochs 2–3 | Does the Virchow2 band ratio change after more epochs (pre-registered contingency)? | the band ratio stayed flat with the AT2 share at epochs 2–3, as at epoch 1 (`removed_analyses/virchow2_continued_e3/`) | the manuscript reports epoch 1 only |

## Other removals on 2026-10-02

- Unused checkpoints of the continued-pretraining runs: `teacher_e000.pt` (identical
  backbone to the released weights) and, for UNI, `teacher_e002.pt` and
  `teacher_e003.pt`, with their embeddings. The analyzed `teacher_e001.pt` were kept.
- Pre-assembly shards of the corrected-image embeddings (`corrected_embeddings/parts`);
  the assembled embeddings were kept.
- Smoke-test and throughput-benchmark outputs, the Hugging Face model cache, execution
  logs (the empty `logs/` folders stay because SLURM writes there), old LaTeX builds,
  Python caches, and the previous `.Trash/`.
- The manuscript audit tooling of the previous manuscript version (`scripts/checks/`,
  `analysis/paper/artifact_map.json`, hash locks, figure verification), `tests/`, and
  `archive/`. These checked figure and table hashes of a manuscript version that no
  longer exists.
- Renderers of figures that the revision replaced, wrappers and protocols of RV15–RV17
  (library modules still imported by RV18 were kept), and unused modules of
  `src/prenorm/`.

Frozen GAN, stain-normalization, and PLISM results in `outputs/` were kept in full,
including all saved checkpoints, because the checkpoint manifests reference them.
`analysis/revision/corrected_embeddings_finalize.py` compares against
`00_manuscript/tables/table_cross_pfm_correction.tex`, which was removed from the
manuscript; that table is in the manuscript repository's history (commit `27b9d90`).
