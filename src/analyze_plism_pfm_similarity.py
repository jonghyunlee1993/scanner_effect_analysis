"""How much of a PFM embedding is tissue and how much is scanner.

The alignment work makes a stronger question possible than the usual one.  Within
a stain section the seven scanners image the *same physical section*, and the grid
gives every location an identifier shared across all seven, so a scanner contrast
here is **content-matched by construction**: two embeddings of literally the same
tissue, differing only in the machine that imaged it.  Cosine similarity between
them is therefore a direct read on scanner invariance, not a mixture of scanner
and biology.

Three statistics, per (tissue, scanner pair):

*paired*   cos(f_A(loc), f_B(loc)) -- the same place on two scanners.  1.0 would
           mean the encoder cannot tell the machines apart at all.
*null*     cos(f_A(loc), f_B(loc')), loc' a different location of the same tissue.
           This is the floor: what two embeddings of the same tissue type score
           when they are *not* the same place.  Without it a paired cosine of 0.95
           is uninterpretable, because foundation-model embeddings are not
           zero-centred and score high on everything.
*top-1*    of all locations of that tissue on scanner B, is the nearest neighbour
           of f_A(loc) the matching location?  A retrieval framing of the same
           question, and the one that degrades visibly when scanner dominates.

The gap between paired and null is the quantity of interest.  Large gap: the
embedding encodes which place this is, and the scanner is a small perturbation.
Small gap: the embedding is dominated by something other than the tissue at hand.

Figures are small multiples rather than one seven-colour scatter.  That is not a
stylistic choice: a categorical palette cannot separate seven classes at
all-pairs on a scatter, and highlighting one scanner against the common cloud
answers "does this machine sit apart" more directly anyway.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REFERENCE = "AT2"
SCANNER_ORDER = ["AT2", "GT450", "P", "S210", "S360", "S60", "SQ"]

# Validated palette (see the dataviz skill's reference instance).  Only two
# categorical slots are ever on screen at once, and both clear every gate.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
MUTED = "#d8d7d2"
# Sequential blue, light -> dark, for magnitude
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

HIGHLIGHT_TISSUES = ["18_liver", "25_cerebral_cortex", "29_lung",
                     "45_skeletal_muscle", "12_lymph_node", "26_thyroid"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-id", required=True)
    parser.add_argument("--features", default="data/PLISM_dataset/features")
    parser.add_argument("--contract", default="outputs/e9_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--output", default="data/PLISM_dataset/features/analysis")
    parser.add_argument("--umap-per-tissue", type=int, default=3,
                        help="locations sampled per (section, tissue) for the embedding")
    parser.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args()


def load_section(directory: Path, stain: str) -> dict:
    """Unit-normed features per scanner, plus the location index they share."""
    out = {}
    for path in sorted(directory.glob(f"{stain}_*.h5")):
        scanner = path.stem.split("_", 1)[1]
        with h5py.File(path, "r") as handle:
            features = handle["features"][:].astype(np.float32)
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            out[scanner] = {
                "f": features / np.maximum(norms, 1e-8),
                "loc": handle["location"][:].astype(np.int64),
                "tissue": np.array([t.decode() if isinstance(t, bytes) else str(t)
                                    for t in handle["tissue_type"][:]]),
            }
    return out


def section_statistics(block: dict, rng: np.random.Generator) -> list[dict]:
    """Paired, null and top-1 for every scanner pair and tissue in one section."""
    scanners = [s for s in SCANNER_ORDER if s in block]
    # locations present on every scanner: the content-matched set
    shared = set(block[scanners[0]]["loc"].tolist())
    for scanner in scanners[1:]:
        shared &= set(block[scanner]["loc"].tolist())
    shared = np.array(sorted(shared))
    if len(shared) == 0:
        return []

    index, tissue_of = {}, None
    for scanner in scanners:
        order = {int(v): i for i, v in enumerate(block[scanner]["loc"])}
        index[scanner] = np.array([order[int(v)] for v in shared])
        if tissue_of is None:
            tissue_of = block[scanner]["tissue"][index[scanner]]

    rows = []
    for tissue in np.unique(tissue_of):
        mask = tissue_of == tissue
        count = int(mask.sum())
        if count < 4:      # a null needs somewhere to permute to
            continue
        permutation = rng.permutation(count)
        # a derangement, so "a different location" is never the same one
        while np.any(permutation == np.arange(count)) and count > 1:
            clash = np.flatnonzero(permutation == np.arange(count))
            permutation[clash] = permutation[np.roll(clash, 1)]
            if len(clash) == 1:
                swap = (clash[0] + 1) % count
                permutation[[clash[0], swap]] = permutation[[swap, clash[0]]]

        vectors = {s: block[s]["f"][index[s][mask]] for s in scanners}
        for i, a in enumerate(scanners):
            for b in scanners[i + 1:]:
                left, right = vectors[a], vectors[b]
                paired = np.einsum("ij,ij->i", left, right)
                null = np.einsum("ij,ij->i", left, right[permutation])
                similarity = left @ right.T
                top1 = float((similarity.argmax(axis=1) == np.arange(count)).mean())
                rows.append({
                    "tissue_type": str(tissue), "scanner_a": a, "scanner_b": b,
                    "n": count,
                    "paired_mean": float(paired.mean()),
                    "paired_median": float(np.median(paired)),
                    "null_mean": float(null.mean()),
                    "separation": float(paired.mean() - null.mean()),
                    "top1": top1,
                })
    return rows


def collect_umap_sample(block: dict, stain: str, per_tissue: int,
                        rng: np.random.Generator) -> tuple[np.ndarray, pd.DataFrame]:
    scanners = [s for s in SCANNER_ORDER if s in block]
    shared = set(block[scanners[0]]["loc"].tolist())
    for scanner in scanners[1:]:
        shared &= set(block[scanner]["loc"].tolist())
    shared = np.array(sorted(shared))
    if len(shared) == 0:
        return np.empty((0, 0)), pd.DataFrame()

    order = {int(v): i for i, v in enumerate(block[scanners[0]]["loc"])}
    base = np.array([order[int(v)] for v in shared])
    tissue_of = block[scanners[0]]["tissue"][base]

    chosen = []
    for tissue in np.unique(tissue_of):
        pool = np.flatnonzero(tissue_of == tissue)
        take = min(per_tissue, len(pool))
        chosen.extend(pool[rng.choice(len(pool), size=take, replace=False)].tolist())
    chosen = np.array(sorted(chosen))
    if len(chosen) == 0:
        return np.empty((0, 0)), pd.DataFrame()

    stack, labels = [], []
    for scanner in scanners:
        position = {int(v): i for i, v in enumerate(block[scanner]["loc"])}
        rows = np.array([position[int(v)] for v in shared[chosen]])
        stack.append(block[scanner]["f"][rows])
        labels.append(pd.DataFrame({
            "stain": stain, "scanner": scanner,
            "location": shared[chosen],
            "tissue_type": block[scanner]["tissue"][rows],
        }))
    return np.concatenate(stack), pd.concat(labels, ignore_index=True)


# ------------------------------------------------------------------ figures


def figure_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_SOFT, "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "axes.edgecolor": "#e4e2dd", "axes.linewidth": 0.8,
        "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def encode(figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode()


def small_multiples(plt, coords: np.ndarray, labels: pd.DataFrame, column: str,
                    values: list, title: str) -> str:
    """One panel per class: the whole cloud in grey, that class in one hue.

    A seven-colour scatter cannot be made colourblind-safe at all-pairs, and this
    reads better regardless -- each panel answers "where does this one sit".
    """
    columns = 4 if len(values) > 3 else 3
    rows = int(np.ceil(len(values) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.0 * rows),
                                squeeze=False)
    for axis in axes.ravel():
        axis.set_visible(False)
    for position, value in enumerate(values):
        axis = axes[position // columns][position % columns]
        axis.set_visible(True)
        axis.scatter(coords[:, 0], coords[:, 1], s=1.5, c=MUTED, linewidths=0, rasterized=True)
        mask = (labels[column] == value).to_numpy()
        axis.scatter(coords[mask, 0], coords[mask, 1], s=2.4, c=SERIES_1,
                     linewidths=0, rasterized=True)
        axis.set_title(f"{value}   n={int(mask.sum()):,}", color=INK, loc="left")
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    return encode(figure)


def heatmap(plt, frame: pd.DataFrame, value: str, title: str, label: str) -> str:
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    height = max(2.4, 0.22 * len(frame.index))
    figure, axis = plt.subplots(figsize=(1.0 + 0.62 * len(frame.columns), height))
    image = axis.imshow(frame.to_numpy(), aspect="auto", cmap=cmap)
    axis.set_xticks(range(len(frame.columns)), frame.columns, rotation=0)
    axis.set_yticks(range(len(frame.index)), frame.index, fontsize=7)
    axis.set_title(title, color=INK, loc="left")
    for spine in axis.spines.values():
        spine.set_visible(False)
    bar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    bar.set_label(label, color=INK_SOFT)
    bar.outline.set_visible(False)
    figure.tight_layout()
    return encode(figure)


def dumbbell(plt, frame: pd.DataFrame, title: str) -> str:
    figure, axis = plt.subplots(figsize=(6.4, 0.42 * len(frame) + 1.9))
    y = np.arange(len(frame))
    axis.hlines(y, frame["null_mean"], frame["paired_mean"], color=MUTED, linewidth=2)
    axis.scatter(frame["null_mean"], y, s=46, c=SERIES_2, zorder=3, label="different location, same tissue")
    axis.scatter(frame["paired_mean"], y, s=46, c=SERIES_1, zorder=3, label="same location")
    for position, row in enumerate(frame.itertuples()):
        axis.annotate(f"{row.paired_mean:.3f}", (row.paired_mean, position),
                      xytext=(6, 0), textcoords="offset points", va="center",
                      fontsize=8, color=INK)
        axis.annotate(f"{row.null_mean:.3f}", (row.null_mean, position),
                      xytext=(-6, 0), textcoords="offset points", va="center",
                      ha="right", fontsize=8, color=INK_SOFT)
    axis.set_yticks(y, frame.index)
    axis.set_xlabel("cosine similarity")
    axis.set_title(title, color=INK, loc="left")
    # outside the axes: inside, it collided with the lowest row's value label
    axis.legend(frameon=False, fontsize=8, ncol=2,
                loc="lower center", bbox_to_anchor=(0.5, 1.04))
    axis.margins(x=0.16, y=0.10)
    figure.tight_layout()
    return encode(figure)


# ------------------------------------------------------------------ main


def main() -> None:
    args = parse_args()
    directory = Path(args.features) / args.encoder_id
    files = sorted(directory.glob("*.h5"))
    if not files:
        raise RuntimeError(f"no features under {directory}")
    stains = sorted({p.stem.split("_", 1)[0] for p in files})
    contract = json.loads(Path(args.contract).read_text())
    model = {m["encoder_id"]: m for m in contract["models"]}[args.encoder_id]

    rng = np.random.default_rng(args.seed)
    records, embeddings, labels = [], [], []
    for stain in stains:
        block = load_section(directory, stain)
        if len(block) < 2:
            continue
        rows = section_statistics(block, rng)
        for row in rows:
            row["stain"] = stain
        records.extend(rows)
        vectors, frame = collect_umap_sample(block, stain, args.umap_per_tissue, rng)
        if len(frame):
            embeddings.append(vectors)
            labels.append(frame)
        print(f"  {stain}: {len(block)} scanners, {len(rows)} (tissue, pair) cells")

    detail = pd.DataFrame(records)
    if detail.empty:
        raise RuntimeError("no comparable locations found")

    output = Path(args.output) / args.encoder_id
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "cosine_by_tissue_scanner_section.csv", index=False)

    def weighted(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        def block(group):
            weight = group["n"].to_numpy()
            return pd.Series({
                "n": int(weight.sum()),
                "paired_mean": float(np.average(group["paired_mean"], weights=weight)),
                "null_mean": float(np.average(group["null_mean"], weights=weight)),
                "separation": float(np.average(group["separation"], weights=weight)),
                "top1": float(np.average(group["top1"], weights=weight)),
            })
        return frame.groupby(keys).apply(block, include_groups=False).reset_index()

    by_pair = weighted(detail, ["scanner_a", "scanner_b"])
    by_tissue = weighted(detail, ["tissue_type"])
    by_pair.to_csv(output / "cosine_by_scanner_pair.csv", index=False)
    by_tissue.to_csv(output / "cosine_by_tissue.csv", index=False)

    # everything against the study's reference scanner
    versus = detail.loc[(detail["scanner_a"] == REFERENCE) | (detail["scanner_b"] == REFERENCE)].copy()
    versus["scanner"] = np.where(versus["scanner_a"] == REFERENCE,
                                 versus["scanner_b"], versus["scanner_a"])
    by_scanner = weighted(versus, ["scanner"]).set_index("scanner")
    by_scanner = by_scanner.reindex([s for s in SCANNER_ORDER if s in by_scanner.index])
    by_scanner.to_csv(output / "cosine_vs_reference.csv")

    # ---- embedding
    coords, frame, silhouette = None, pd.DataFrame(), {}
    if embeddings:
        stack = np.concatenate(embeddings)
        frame = pd.concat(labels, ignore_index=True)
        print(f"  UMAP on {stack.shape[0]:,} x {stack.shape[1]}")
        import umap
        from sklearn.metrics import silhouette_score

        coords = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                           random_state=args.seed).fit_transform(stack)
        frame["umap_x"], frame["umap_y"] = coords[:, 0], coords[:, 1]
        frame.to_csv(output / "umap.csv", index=False)
        # measured in the embedding space itself, not on the 2-D projection
        sample = rng.choice(len(stack), size=min(6000, len(stack)), replace=False)
        for name, column in (("scanner", "scanner"), ("tissue", "tissue_type")):
            silhouette[name] = float(silhouette_score(
                stack[sample], frame[column].to_numpy()[sample], metric="cosine"))
        print(f"  silhouette: scanner {silhouette['scanner']:+.4f}  "
              f"tissue {silhouette['tissue']:+.4f}")

    # ---- figures
    plt = figure_style()
    figures = {}
    if coords is not None:
        present = [s for s in SCANNER_ORDER if s in set(frame["scanner"])]
        figures["umap_scanner"] = small_multiples(
            plt, coords, frame, "scanner", present,
            "Embedding by scanner — each panel highlights one machine against the whole cloud")
        tissues = [t for t in HIGHLIGHT_TISSUES if t in set(frame["tissue_type"])]
        if tissues:
            figures["umap_tissue"] = small_multiples(
                plt, coords, frame, "tissue_type", tissues,
                "Embedding by tissue — six of the forty-six cores, same cloud")

    matrix = pd.DataFrame(index=SCANNER_ORDER, columns=SCANNER_ORDER, dtype=float)
    for row in by_pair.itertuples():
        matrix.loc[row.scanner_a, row.scanner_b] = row.paired_mean
        matrix.loc[row.scanner_b, row.scanner_a] = row.paired_mean
    np.fill_diagonal(matrix.values, 1.0)
    matrix = matrix.dropna(how="all").dropna(axis=1, how="all")
    figures["pair_matrix"] = heatmap(
        plt, matrix, "paired", "Paired cosine between scanners, same physical location",
        "cosine")

    wide = versus.pivot_table(index="tissue_type", columns="scanner",
                              values="paired_mean", aggfunc="mean")
    wide = wide[[s for s in SCANNER_ORDER if s in wide.columns]].sort_index()
    figures["tissue_matrix"] = heatmap(
        plt, wide, "paired", f"Paired cosine against {REFERENCE}, by TMA core", "cosine")

    figures["dumbbell"] = dumbbell(
        plt, by_scanner, f"Same location vs different location of the same tissue, against {REFERENCE}")

    html = render_html(args.encoder_id, model, by_scanner, by_tissue, detail,
                       silhouette, figures, len(files))
    (output / "similarity_report.html").write_text(html)
    print(f"-> {output}/similarity_report.html")
    for name in sorted(p.name for p in output.iterdir()):
        print(f"   {name}")


def table(frame: pd.DataFrame, formats: dict | None = None, index_name: str = "") -> str:
    formats = formats or {}
    head = f"<th>{index_name}</th>" if index_name else ""
    head += "".join(f"<th>{c}</th>" for c in frame.columns)
    body = []
    for label, row in frame.iterrows():
        cells = f"<th scope=row>{label}</th>" if index_name else ""
        for column in frame.columns:
            value = row[column]
            spec = formats.get(column)
            cells += f"<td>{format(value, spec) if spec and pd.notna(value) else value}</td>"
        body.append(f"<tr>{cells}</tr>")
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def render_html(encoder_id, model, by_scanner, by_tissue, detail,
                silhouette, figures, files) -> str:
    worst = by_tissue.nsmallest(8, "separation")[
        ["tissue_type", "n", "paired_mean", "null_mean", "separation", "top1"]]
    best = by_tissue.nlargest(5, "separation")[
        ["tissue_type", "n", "paired_mean", "null_mean", "separation", "top1"]]
    fmt = {"paired_mean": ".4f", "null_mean": ".4f", "separation": ".4f", "top1": ".3f"}

    scanner_rows = by_scanner.copy()
    scanner_rows.index.name = "Scanner"

    sil = ""
    if silhouette:
        sil = (f"<div class=\"cards\">"
               f"<div class=\"card\"><div class=\"name\">silhouette · scanner</div>"
               f"<div class=\"value\">{silhouette['scanner']:+.4f}</div>"
               f"<div class=\"sub\">near 0 means the machines do not form clusters</div></div>"
               f"<div class=\"card\"><div class=\"name\">silhouette · tissue</div>"
               f"<div class=\"value\">{silhouette['tissue']:+.4f}</div>"
               f"<div class=\"sub\">higher means the embedding organises by tissue</div></div>"
               f"</div>")

    images = "".join(
        f'<figure><img alt="{name}" src="data:image/png;base64,{data}"></figure>'
        for name, data in figures.items())

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{encoder_id} scanner similarity</title>
<style>
:root {{ --bg:#fbfaf8; --panel:#fff; --ink:#1a1a1c; --muted:#6b6b73; --line:#e4e2dd; --accent:#2f6f5e; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#17171a; --panel:#1f1f23; --ink:#ecebe8; --muted:#9b9aa2; --line:#33333a; --accent:#7fc4ae; }} }}
:root[data-theme="dark"] {{ --bg:#17171a; --panel:#1f1f23; --ink:#ecebe8; --muted:#9b9aa2; --line:#33333a; --accent:#7fc4ae; }}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:3rem 1.25rem 6rem}}
h1{{font-size:clamp(1.5rem,3vw,2.1rem);margin:0 0 .3rem;letter-spacing:-.01em}}
h2{{font-size:1.2rem;margin:2.8rem 0 .5rem}}
p{{max-width:74ch;margin:.7rem 0}} .lede{{color:var(--muted);max-width:74ch}}
.cards{{display:grid;gap:.7rem;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));margin:1.4rem 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.85rem .95rem}}
.card .name{{font-size:.76rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}}
.card .value{{font-size:1.5rem;font-variant-numeric:tabular-nums;margin-top:.15rem}}
.card .sub{{font-size:.76rem;color:var(--muted);margin-top:.25rem}}
.scroll{{overflow-x:auto;margin:1rem 0;border:1px solid var(--line);border-radius:10px;background:var(--panel)}}
table{{border-collapse:collapse;width:100%;font-size:.86rem}}
th,td{{padding:.42rem .6rem;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}}
thead th{{background:var(--panel);border-bottom:1px solid var(--line);color:var(--muted);font-weight:600}}
tbody th{{text-align:left;font-weight:500}}
tbody tr+tr td,tbody tr+tr th{{border-top:1px solid var(--line)}}
figure{{margin:1.4rem 0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.7rem}}
figure img{{width:100%;height:auto;display:block;border-radius:6px}}
.note{{border-left:3px solid var(--accent);padding:.1rem 0 .1rem 1rem;margin:1.3rem 0;color:var(--muted)}}
.note strong{{color:var(--ink)}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.88em;background:var(--line);padding:.1em .35em;border-radius:4px}}
</style></head><body><div class="wrap">

<h1>{encoder_id} — scanner similarity across TMA cores</h1>
<p class="lede">How much of this encoder's embedding is the tissue and how much is the
machine that imaged it. {files} feature files, {model['feature_dim']}-d,
crop {model['native_fov_px']} px = {model['native_fov_um']:.2f} µm at 20×, {model['precision']}.</p>

<div class="note"><strong>The comparison is content-matched.</strong> Within a stain section
the seven scanners image the same physical section, and the aligned grid gives every
location an identifier shared across all seven. A scanner contrast here is therefore two
embeddings of <em>literally the same tissue</em>, differing only in the machine — not a
mixture of scanner and biology.</div>

{sil}

<h2>Against the reference scanner</h2>
<p><b>paired</b> is the same physical location on two machines. <b>null</b> is a
<em>different</em> location of the same tissue — the floor, and the reason a raw cosine is
uninterpretable on its own: these embeddings are not zero-centred and score high on
everything. <b>separation</b> is the gap. <b>top-1</b> asks whether the matching location is
the nearest neighbour among all locations of that tissue.</p>
{table(scanner_rows, fmt, "Scanner")}

<h2>Figures</h2>
<p>Each embedding panel highlights one class against the whole cloud. A seven-colour
scatter cannot be made colourblind-safe at all-pairs, and this reads better regardless —
the question is whether a machine occupies its own territory, which one panel answers
directly.</p>
{images}

<h2>Cores where the encoder separates least</h2>
<p>Low separation means the embedding of a location is not much closer to the same place
on another scanner than to a different place in the same core. That can be scanner
sensitivity, or a core so homogeneous that one location looks like any other.</p>
{table(worst.set_index("tissue_type"), fmt, "TMA core")}

<h2>…and most</h2>
{table(best.set_index("tissue_type"), fmt, "TMA core")}

<h2>Files</h2>
<p><code>cosine_vs_reference.csv</code>, <code>cosine_by_scanner_pair.csv</code>,
<code>cosine_by_tissue.csv</code>,
<code>cosine_by_tissue_scanner_section.csv</code> (every cell, {len(detail):,} rows),
<code>umap.csv</code> (coordinates and labels, to re-plot without recomputing).</p>
<p class="lede">Alignment that makes this possible: <code>../../ALIGNMENT.md</code>.
Encoder provenance: <code>outputs/e9_pfm_contract/checkpoint_manifest.json</code>,
checkpoint sha256 <code>{model['checkpoint_sha256'][:16]}…</code>.</p>

</div></body></html>
"""


if __name__ == "__main__":
    main()
