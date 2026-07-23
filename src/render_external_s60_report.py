"""Render a joint internal-held-out and external-S60 validation report."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image
from sklearn.decomposition import PCA

from utils.store import decode


INTERNAL = Path("outputs/phase1_v3_10slide_ab08_versa_pair05_50ep/eval")
EXTERNAL = Path("outputs/external_s60_10slide_ab08_v1")
STORE = Path("data/external_s60_10slide_store_v1")
COLORS = {"ID · at2": "#2563eb", "ID · gt450": "#dc2626", "ID · versa": "#16a34a",
          "ID · akoya": "#9333ea", "OOD · S60 raw": "#d97706",
          "OOD · S60 canonical": "#ea580c", "OOD · paired AT2": "#0e7490",
          "OOD · paired AT2 canonical": "#0891b2"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--internal", type=Path, default=INTERNAL)
    p.add_argument("--external", type=Path, default=EXTERNAL)
    p.add_argument("--store", type=Path, default=STORE)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def png(fig, output):
    fig.tight_layout(); fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)


def joint_umap(internal, external, output, workers):
    cache = output / "joint_umap_coordinates.npz"
    internal_embeddings = internal / "embeddings.npz"
    external_embeddings = external / "uni_embeddings.npz"
    old, ext = np.load(internal_embeddings), np.load(external_embeddings)
    scanners = np.char.add("ID · ", np.array(["at2", "gt450", "versa", "akoya"])[old["scanner_id"]])
    canonical_at2 = ext["standard_at2"] if "standard_at2" in ext else ext["paired_at2"]
    names = np.concatenate([scanners, np.full(len(ext["raw_s60"]), "OOD · S60 raw"),
                            np.full(len(ext["paired_at2"]), "OOD · paired AT2"), scanners,
                            np.full(len(ext["standard_s60"]), "OOD · S60 canonical"),
                            np.full(len(canonical_at2), "OOD · paired AT2 canonical")])
    representations = np.concatenate([np.full(len(old["raw"]), "raw"), np.full(len(ext["raw_s60"]), "raw"),
                                      np.full(len(ext["paired_at2"]), "raw"), np.full(len(old["canonical"]), "canonical"),
                                      np.full(len(ext["standard_s60"]), "canonical"),
                                      np.full(len(canonical_at2), "canonical")])
    vectors = np.concatenate([old["raw"], ext["raw_s60"], ext["paired_at2"], old["canonical"],
                              ext["standard_s60"], canonical_at2])
    source_mtime = max(internal_embeddings.stat().st_mtime_ns,
                       external_embeddings.stat().st_mtime_ns)
    xy = None
    if cache.exists() and cache.stat().st_mtime_ns >= source_mtime:
        cached = np.load(cache)
        candidate = cached["xy"]
        cached_count = int(cached["n_vectors"]) if "n_vectors" in cached else len(candidate)
        if cached_count == len(vectors) and candidate.shape == (len(vectors), 2):
            xy = candidate
    if xy is None:
        import umap
        n_components = min(50, len(vectors), vectors.shape[1])
        reduced = PCA(n_components=n_components, svd_solver="randomized", random_state=1234).fit_transform(vectors)
        n_neighbors = min(30, len(vectors) - 1)
        xy = umap.UMAP(n_neighbors=n_neighbors, min_dist=.15, metric="cosine", random_state=1234, n_jobs=workers).fit_transform(reduced)
        np.savez_compressed(cache, xy=xy, n_vectors=np.int64(len(vectors)))
    rng = np.random.default_rng(1234); fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=170, sharex=True, sharey=True)
    for ax, rep, title in zip(axes, ("raw", "canonical"), ("Input UNI space", "Canonical-output UNI space")):
        selected_rep = representations == rep
        for label in dict.fromkeys(names[selected_rep]):
            idx = np.flatnonzero(selected_rep & (names == label)); idx = rng.choice(idx, min(len(idx), 5000), replace=False)
            is_ood = str(label).startswith("OOD")
            ax.scatter(xy[idx, 0], xy[idx, 1], s=7 if is_ood else 2.5,
                       alpha=.46 if is_ood else .28, color=COLORS[label],
                       marker="x" if is_ood else "o", linewidths=.45 if is_ood else 0,
                       label=label, rasterized=True)
        ax.set_title(title, loc="left", weight="bold"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    legend_handles = [
        Line2D([0], [0], linestyle="none", marker="x" if label.startswith("OOD") else "o",
               markersize=6, markeredgecolor=color,
               markerfacecolor="none" if label.startswith("OOD") else color)
        for label, color in COLORS.items()
    ]
    axes[1].legend(legend_handles, COLORS.keys(), frameon=False,
                   fontsize=8, ncol=2, loc="best")
    png(fig, output / "joint_umap.png")


def read_pair(row, store):
    with h5py.File(store / f"{row.slide_id}.h5", "r") as f:
        return decode(f["s60"]["rgb"][int(row.tuple_id)]), decode(f["at2"]["rgb"][int(row.tuple_id)])


def retrieval(external, store, output, queries=None, filename="external_retrieval.png"):
    data, frame = np.load(external / "uni_embeddings.npz"), pd.read_csv(external / "per_tile.csv")
    # These deterministic examples also have saved four-panel input/generated
    # images, allowing the standard query thumbnail to match its embedding.
    if queries is None:
        queries = np.array([0, 20, 39], dtype=int)
    queries = np.asarray(queries, dtype=int)
    roles = [("raw_s60", "OOD · S60 raw"), ("standard_s60", "OOD · S60 canonical")]
    fig, axes = plt.subplots(len(queries) * len(roles), 6, figsize=(13, len(queries) * len(roles) * 2.5), dpi=160)
    for qi, query in enumerate(queries):
        source, paired = read_pair(frame.iloc[query], store)
        for ri, (key, label) in enumerate(roles):
            row = qi * len(roles) + ri; similarity = data[key][query] @ data["paired_at2"].T
            top = np.argsort(similarity)[-5:][::-1]
            query_image = source if key == "raw_s60" else np.asarray(Image.open(external / "samples" / f"{query:03d}.png"))[:, 256:512]
            axes[row, 0].imshow(query_image); axes[row, 0].set_title(f"{label} query\nslide {frame.iloc[query].slide_id}", fontsize=8)
            for spine in axes[row, 0].spines.values():
                spine.set_color("#2156d7"); spine.set_linewidth(2.5)
            for rank, index in enumerate(top, 1):
                _, at2 = read_pair(frame.iloc[int(index)], store); hit = index == query
                axes[row, rank].imshow(at2); axes[row, rank].set_title(f"AT2 NN{rank}\ncos {similarity[index]:.3f}", fontsize=8)
                for spine in axes[row, rank].spines.values(): spine.set_color("#14805e" if hit else "#c2414b"); spine.set_linewidth(2.5)
            for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("OOD S60 retrieval · green = co-located paired AT2", weight="bold")
    png(fig, output / filename)


def joint_retrieval(internal_plot, external, store, output):
    """Append one raw/canonical S60 query pair below the existing ID grid."""
    s60_plot = output / "s60_retrieval_rows.png"
    retrieval(external, store, output, queries=[0], filename=s60_plot.name)
    top, bottom = Image.open(internal_plot).convert("RGB"), Image.open(s60_plot).convert("RGB")
    if bottom.width != top.width:
        height = round(bottom.height * top.width / bottom.width)
        bottom = bottom.resize((top.width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (top.width, top.height + bottom.height), "white")
    canvas.paste(top, (0, 0)); canvas.paste(bottom, (0, top.height))
    canvas.save(output / "joint_retrieval.png")


def report(internal, external):
    m, old = json.loads((external / "metrics.json").read_text()), json.loads((internal / "metrics.json").read_text())
    samples = "".join(f'<figure><img src="samples/{i:03d}.png"><figcaption>S60 raw · standard · AT2 recon · paired AT2</figcaption></figure>' for i in range(12))
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>AB08 external S60 validation</title><style>
body{{font:15px/1.55 system-ui,sans-serif;margin:0;color:#18212f;background:#f3f6fa}}header{{padding:42px max(24px,calc((100% - 1100px)/2));background:#102f52;color:white}}main{{max-width:1100px;margin:auto;padding:24px}}section{{background:white;border:1px solid #dce3ec;border-radius:14px;padding:20px;margin:14px 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #dce3ec;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f5f7fa}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}img{{max-width:100%}}.gallery{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}figure{{margin:0;border:1px solid #dce3ec}}figcaption{{padding:7px;font-size:12px;color:#687386}}.good{{color:#14805e;font-weight:700}}.warn{{color:#c2414b;font-weight:700}}@media(max-width:700px){{.grid,.gallery{{grid-template-columns:1fr}}}}</style></head><body>
<header><div>PRENORM · JOINT INTERNAL + EXTERNAL VALIDATION</div><h1>AB08: held-out scanners plus external S60</h1><p>Internal test evaluation is shown alongside an independently scanned S60 cohort mapped to the AT2 canvas and locally refined at patch level.</p></header><main>
<section><h2>Scope and headline</h2><div class="grid"><table><tr><th>Internal held-out</th><td>{int(old['n_images']):,} images / {int(old['n_locations']):,} locations</td></tr><tr><th>Internal UNI cosine</th><td>raw {old['raw_same_location_cosine']:.3f} → canonical {old['canonical_same_location_cosine']:.3f}</td></tr><tr><th>External S60</th><td>{m['n_tiles']:,} paired patches / 10 slides</td></tr><tr><th>External UNI cosine to paired AT2</th><td class="warn">raw {m['uni_cosine_raw_s60_to_at2']:.3f} → standard {m['uni_cosine_standard_s60_to_at2']:.3f}</td></tr></table><p><b>Interpretation:</b> image-space matching improves externally, but UNI similarity and focus decline. The external result therefore supports AT2-style color/low-frequency matching, not yet robust preservation of S60 morphology.</p></div></section>
<section><h2>1 · Joint UMAP: existing scanners + S60</h2><img src="joint_umap.png"><p>Left: scanner input embeddings. Right: canonical-output embeddings. Points are capped at 5,000 per class for legibility; the fitted projection uses all embeddings.</p></section>
<section><h2>2 · Image-space external validation</h2><table><tr><th>Metric vs paired AT2</th><th>Raw S60</th><th>Standard / AT2 recon</th></tr><tr><td>Robust image distance ↓</td><td>{m['raw_s60_distance_to_at2']:.5f}</td><td class="good">{m['standard_distance_to_at2']:.5f}</td></tr><tr><td>SSIM ↑</td><td>{m['raw_s60_ssim_to_at2']:.5f}</td><td class="good">{m['standard_ssim_to_at2']:.5f}</td></tr><tr><td>UNI cosine ↑</td><td>{m['uni_cosine_raw_s60_to_at2']:.5f}</td><td class="warn">{m['uni_cosine_standard_s60_to_at2']:.5f}</td></tr><tr><td>Focus score</td><td>{m['raw_s60_focus']:.5f}</td><td>{m['standard_focus']:.5f}</td></tr></table><p>For this α=0 model, standard and AT2 reconstruction are identical; there is no learned S60 decoder.</p></section>
<section><h2>3 · Cross-domain retrieval</h2><img src="external_retrieval.png"><p>Each S60 query retrieves among all external paired AT2 patches in frozen UNI space. Green borders identify the exact co-located AT2 match; red indicates another location.</p></section>
<section><h2>4 · Inputs and generated images</h2><div class="gallery">{samples}</div></section>
<section><h2>Artifacts</h2><p><a href="metrics.json">aggregate metrics</a> · <a href="per_tile.csv">per-tile metrics</a> · <a href="uni_embeddings.npz">external UNI embeddings</a> · <a href="../phase1_v3_10slide_ab08_versa_pair05_50ep/eval/report.html">original internal report</a></p></section>
</main></body></html>'''


def main():
    a = parse_args(); a.external.mkdir(parents=True, exist_ok=True)
    joint_umap(a.internal, a.external, a.external, a.workers)
    retrieval(a.external, a.store, a.external)
    (a.external / "report.html").write_text(report(a.internal, a.external))
    print(f"[external-report] rendered {a.external / 'report.html'}")


if __name__ == "__main__": main()
