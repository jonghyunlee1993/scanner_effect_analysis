# Scanner Spectrum experiment index

The active analysis project restarts experiment numbering at one. Numbers from
the archived canonicalizer-modeling phase are historical labels and are not
part of this active sequence.

| Active experiment | Question | Previous working label | Status |
|---|---|---|---|
| Exp-01 | How much low-frequency scanner variation is correctable by simple affine transforms with explicit clipping? | Exp-06 | Complete; learned LF path retired, affine retained |
| Exp-02 | Does pairwise image-space correction move PFM embeddings, and what happens under matched detail damage? | Exp-07 | Pilot complete; population/multi-PFM confirmation next |
| Exp-03 | Is there a single content-preserving image space that satisfies all scanners and PFMs? | New | Planned |
| Exp-04 | Can feature-space correction achieve invariance without image hallucination? | New | Reserved |

Active paths follow this numbering:

```text
configs/experiments/exp01*.yaml
src/prenorm/exp01/
src/prenorm/exp02/
src/audit_exp01_stage0.py
src/eval_exp01_baselines.py
src/eval_exp02*.py
scripts/exp01/
scripts/exp02/
outputs/exp01*
outputs/exp02*
```

The retired learned Exp-01 checkpoints keep their historical metadata and are
not part of the active analysis path.

The current scientific synthesis, defensible claims, Pix2Pix falsification gate
and publication plan are recorded in
[`research_synthesis.md`](research_synthesis.md). Retired model and diagnostic
paths are summarized in [`retired_attempts.md`](retired_attempts.md); their
historical implementation is recoverable from Git or the existing local
`archived/` directories.
