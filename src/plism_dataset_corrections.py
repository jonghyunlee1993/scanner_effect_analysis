"""Corrections to the published PLISM file labelling.

PLISM ships `original_wsi/{stain}_{scanner}.{ext}`, and the stain in the file
name is taken to name the physical section.  For two files it does not.

**`GIVH_SQ.ndpi` holds the HRH section and `HRH_SQ.ndpi` holds the GIVH
section.**  The two are swapped.

How this was established, since it is a claim about someone else's published
data.  Within a stain the seven scanners image one physical section, so patches
must correspond.  For these two files they do not, and four explanations were
excluded in turn:

* *Wrong core.*  Excluded -- the tissue-mask fit of the whole section outline
  returns Dice 0.93, and the keypoint transform agrees with it to 40-52 um
  against a 3600 um core pitch.  The transform is on the right core.
* *Merely a large offset.*  Excluded -- normalised cross-correlation over
  +/-800 um finds a best peak of 0.10-0.27 whose runner-up elsewhere in the
  window is 0.88-0.99 of it.  That is a noise field, not a displaced match.
* *Scale error.*  Excluded -- sweeping 0.4-2.5x, the best correlation is 0.17 at
  1.90x, against 0.92 at 1.00x for a healthy block.
* *Blur.*  Excluded -- high-band power and patch contrast are normal, and
  normalised cross-correlation is robust to blur in any case.

What remained was that the pixels are a different section, and matching each SQ
file against the AT2 reference of all thirteen sections said which:
`GIVH_SQ` matches AT2/HRH at 0.747 and its own label at 0.111; `HRH_SQ` matches
AT2/GIVH at 0.831 and its own label at 0.130; the other eleven sections score
0.09-0.32.  Repeated over eight cores this is unanimous, 16 of 16, while S360 run
through the identical comparison is correct 16 of 16 (0.824-0.983 against its own
section).  Six of the seven scanners agree with each other under the published
labels and only SQ disagrees, so the two SQ files are the mislabelled ones.

Applying the swap is what makes the 13 x 7 design complete again: without it,
SQ contributes nothing at all to two of the thirteen sections.

Kept switchable so the dataset as published stays reproducible.
"""

from __future__ import annotations

# (stain, scanner) -> the file that actually holds that section
FILE_CORRECTIONS = {
    ("GIVH", "SQ"): "HRH_SQ.ndpi",
    ("HRH", "SQ"): "GIVH_SQ.ndpi",
}

EVIDENCE = "NCC 0.747/0.831 to the other section vs 0.111/0.130 to its own, 16/16 cores"


def corrected_name(stain: str, scanner: str, published: str) -> tuple[str, bool]:
    """The file that really holds this section, and whether it differs."""
    replacement = FILE_CORRECTIONS.get((stain, scanner))
    if replacement is None or replacement == published:
        return published, False
    return replacement, True


# A block that is correctly labelled and correctly registered, but whose pixels
# are not what the scanner normally produces.  `HRH_S60` is out of focus: patches
# sit 4-11 um from the reference and blur-tolerant NCC confirms the tissue at
# 0.46-0.82, while high-band power is down five- to tenfold.  That is a real
# property of the scan, so it stays in the dataset and in the alignment report.
#
# It is excluded from *sharpness* statistics, and only those, because the
# endpoint of this study is high-band power: a per-(section, scanner) sharpness
# number that includes this block measures a focus failure rather than the
# scanner.  The exclusion is at block level, never per patch -- rejecting
# individual patches on how detailed they look is the detail-based selection
# section 2.1 of the contract forbids, and it would remove the evidence along
# with the problem.
DEFECTIVE_BLOCKS = {("HRH", "S60"): "out-of-focus scan; high-band power down 5-10x"}


def parse_exclusions(text: str) -> dict:
    """`--exclude HRH:S60,GIV:P` -> {(stain, scanner): reason}, `default` -> the table."""
    if not text:
        return {}
    if text.strip() == "default":
        return dict(DEFECTIVE_BLOCKS)
    out = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        stain, _, scanner = item.partition(":")
        if not stain or not scanner:
            raise ValueError(f"expected stain:scanner, got {item!r}")
        out[(stain, scanner)] = DEFECTIVE_BLOCKS.get((stain, scanner), "excluded on request")
    return out


def drop_blocks(frame, exclusions: dict, stain_column: str = "stain",
                scanner_column: str = "scanner"):
    """Remove whole (section, scanner) blocks from a table, and say what went."""
    if not exclusions:
        return frame, []
    removed = []
    keep = frame.index == frame.index  # all True, without importing numpy here
    for (stain, scanner), reason in exclusions.items():
        hit = (frame[stain_column] == stain) & (frame[scanner_column] == scanner)
        count = int(hit.sum())
        if count:
            removed.append({"stain": stain, "scanner": scanner,
                            "rows": count, "reason": reason})
            keep = keep & ~hit
    return frame[keep], removed
