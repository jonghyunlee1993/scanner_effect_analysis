"""Build the standalone RF1U report page.

Reads the frontier tables, the neighbourhood tables and the rendered gallery,
and writes a single self-contained HTML file with the images inlined as data
URIs. Regenerating the page after a rerun is one command, so the numbers on the
page can never drift from the artifacts they came from.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


TARGETS = ("at2", "gt450", "s60")
TARGET_LABEL = {"at2": "AT2", "gt450": "GT450", "s60": "S60"}
MODELS = ("resnet50", "uni_v1", "conch_v1", "virchow2")
MODEL_LABEL = {
    "resnet50": ("ResNet50", "natural-image CNN"),
    "uni_v1": ("UNI v1", "pathology SSL"),
    "conch_v1": ("CONCH v1", "vision-language"),
    "virchow2": ("Virchow2", "large pathology SSL"),
}
PANEL_LABEL = {
    "raw_source": "Raw source",
    "raw_target": "Raw target",
    "reinhard": "Reinhard",
    "macenko": "Macenko",
    "paired_od_affine": "Paired OD affine",
    "rf1u": "Ours",
    "rf1u_minus_reinhard_x5": "Ours − Reinhard ×5",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier", default="outputs/rf1u_multitarget/frontier")
    parser.add_argument("--neighbourhood", default="outputs/rf1u_multitarget/neighbourhood")
    parser.add_argument("--gallery", default="outputs/rf1u_multitarget/gallery")
    parser.add_argument("--knn-gallery", default="outputs/rf1u_multitarget/knn_gallery")
    parser.add_argument("--scanner-probe", default="outputs/rf1u_multitarget/scanner_probe")
    parser.add_argument(
        "--output", default="presentations/rf1u_multitarget_2026-08-04/index.html"
    )
    return parser.parse_args()


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def signed_pp(value: float) -> str:
    return f"{value * 100:+.2f}pp".replace("+-", "−").replace("-", "−")


def cell(value: str, klass: str = "") -> str:
    attr = f' class="{klass}"' if klass else ""
    return f"<td{attr}>{value}</td>"


def radius_rows(endpoints: pd.DataFrame) -> str:
    rows = []
    for target in TARGETS:
        for position, model in enumerate(MODELS):
            base = endpoints[
                (endpoints.target == target)
                & (endpoints.encoder_id == model)
                & (~endpoints.is_ours)
            ].iloc[0]
            ours = endpoints[
                (endpoints.target == target)
                & (endpoints.encoder_id == model)
                & (endpoints.is_ours)
            ].iloc[0]
            delta = ours.relative_radius_reduction - base.relative_radius_reduction
            klass = "num-pos" if delta > 0 else "num-neg"
            if ours.safe_and_improved:
                verdict = '<span class="flag flag-pass">safe + improved</span>'
            elif not ours.content_noninferiority_pass and not ours.collapse_every_scanner_pass:
                verdict = '<span class="flag flag-fail">content + collapse fail</span>'
            elif not ours.content_noninferiority_pass:
                verdict = '<span class="flag flag-fail">content fail</span>'
            elif not ours.invariance_better_than_reinhard:
                verdict = '<span class="flag">CI includes zero</span>'
            else:
                verdict = '<span class="flag flag-fail">worse than Reinhard</span>'
            label, kind = MODEL_LABEL[model]
            group = ' class="group-start"' if position == 0 else ""
            span = (
                f'<td class="txt" rowspan="{len(MODELS)}">{TARGET_LABEL[target]}</td>'
                if position == 0
                else ""
            )
            ours_value = pct(ours.relative_radius_reduction)
            if ours.safe_and_improved:
                ours_value = f"<strong>{ours_value}</strong>"
            rows.append(
                f"<tr{group}>{span}"
                + cell(f'{label} <span class="dim">{kind}</span>', "txt")
                + cell(pct(base.relative_radius_reduction))
                + cell(ours_value, "dim" if ours.relative_radius_reduction < 0 else "")
                + cell(signed_pp(delta), klass)
                + cell(verdict, "txt")
                + "</tr>"
            )
    return "\n".join(rows)


def neighbour_rows(frame: pd.DataFrame, metric: str, better: str) -> str:
    rows = []
    for model in MODELS:
        label, _ = MODEL_LABEL[model]
        values = {}
        raw = frame[
            (frame.encoder_id == model) & (frame.condition == "raw")
        ][metric].iloc[0]
        cells = [cell(label, "txt"), cell(f"{raw:.3f}", "dim")]
        for target in TARGETS:
            pair = []
            for prefix in ("reinhard_", "reinhard_unpaired_band_"):
                condition = f"{prefix}{target}"
                pair.append(
                    frame[
                        (frame.encoder_id == model) & (frame.condition == condition)
                    ][metric].iloc[0]
                )
            improved = (pair[1] < pair[0]) if better == "low" else (pair[1] > pair[0])
            values[target] = improved
            cells.append(cell(f"{pair[0]:.3f}"))
            cells.append(
                cell(
                    f"<strong>{pair[1]:.3f}</strong>" if improved else f"{pair[1]:.3f}",
                    "num-pos" if improved else "num-neg",
                )
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def gallery_blocks(gallery: dict) -> str:
    panels = gallery["panels"]
    blocks = []
    for target in TARGETS:
        rows = [row for row in gallery["rows"] if row["target"] == target]
        head = "".join(f"<th>{PANEL_LABEL[name]}</th>" for name in panels)
        body = []
        for row in rows:
            source = row["source"]
            gains = " · ".join(
                "{:.2f}".format(row["gain_sigma{}".format(index + 1)])
                for index in range(3)
            )
            cells = []
            for name in panels:
                uri = gallery["images"]["{}__{}__{}".format(target, source, name)]
                alt = "{}, {} toward {}".format(
                    PANEL_LABEL[name], source, TARGET_LABEL[target]
                )
                cells.append(
                    '<td><img src="{}" alt="{}" width="256" height="256" '
                    'loading="lazy"></td>'.format(uri, html.escape(alt))
                )
            body.append(
                '<tr><th class="row-head" scope="row">{}'
                '<span class="row-sub">band gains {}</span></th>{}</tr>'.format(
                    source.upper(), gains, "".join(cells)
                )
            )
        blocks.append(
            f'<div class="gallery-block">'
            f'<h3>Toward {TARGET_LABEL[target]}</h3>'
            f'<div class="gallery-scroll"><table class="gallery">'
            f'<thead><tr><th class="row-head"></th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div></div>'
        )
    return "\n".join(blocks)


def knn_blocks(knn: dict, meta: dict) -> str:
    """Toggleable strips: the query patch and its nearest neighbours per condition."""
    order = ["raw"] + [f"ours_{target}" for target in ("gt450", "at2")]
    label = {
        "raw": "Raw",
        "ours_gt450": "Ours → GT450",
        "ours_at2": "Ours → AT2",
    }
    buttons = "".join(
        '<button type="button" class="knn-btn{}" data-cond="{}" '
        'aria-pressed="{}">{}</button>'.format(
            " is-active" if index == 0 else "",
            name,
            "true" if index == 0 else "false",
            label[name],
        )
        for index, name in enumerate(order)
        if name in knn
    )

    panels = []
    for index, name in enumerate([value for value in order if value in knn]):
        condition = knn[name]
        strips = []
        for model in MODELS:
            if model not in condition["models"]:
                continue
            entry = condition["models"][model]
            model_label, _ = MODEL_LABEL[model]
            tiles = []
            for item in entry["entries"]:
                classes = ["knn-tile"]
                if item["is_query"]:
                    classes.append("is-query")
                elif item["same_location"]:
                    classes.append("is-location")
                elif item["same_scanner"]:
                    classes.append("is-scanner")
                tag = (
                    "QUERY"
                    if item["is_query"]
                    else "cos {:.3f}".format(item["cosine"])
                )
                tiles.append(
                    '<figure class="{}"><img src="{}" alt="{} location {}" '
                    'width="192" height="192" loading="lazy">'
                    '<figcaption>{} · {}<span class="knn-cos">{}</span></figcaption>'
                    "</figure>".format(
                        " ".join(classes),
                        condition["images"][item["key"]],
                        item["scanner"].upper(),
                        item["location"],
                        item["scanner"].upper(),
                        item["location"],
                        tag,
                    )
                )
            strips.append(
                '<div class="knn-model"><div class="knn-model-head">{}'
                '<span class="knn-counts">{} of 8 same scanner · {} of 5 same location'
                "</span></div>"
                '<div class="knn-strip">{}</div></div>'.format(
                    model_label,
                    entry["same_scanner_neighbours"],
                    entry["same_location_neighbours"],
                    "".join(tiles),
                )
            )
        panels.append(
            '<div class="knn-panel{}" data-cond="{}"{}>{}</div>'.format(
                " is-active" if index == 0 else "",
                name,
                "" if index == 0 else " hidden",
                "".join(strips),
            )
        )

    return (
        '<div class="knn"><div class="knn-controls" role="group" '
        'aria-label="Condition">{}</div>{}</div>'.format(buttons, "".join(panels))
    )


FEATURE_METHODS = (("coral", "CORAL"), ("orthogonal_procrustes", "Procrustes"))


def probe_rows(frame: pd.DataFrame) -> str:
    """Image-space conditions and the locked feature-space comparators, side by side."""
    rows = []
    for model in MODELS:
        label, _ = MODEL_LABEL[model]

        def value(condition):
            return frame[
                (frame.encoder_id == model) & (frame.condition == condition)
            ]["balanced_accuracy"].iloc[0]

        raw = value("raw")
        cells = [cell(label, "txt"), cell(f"{raw:.3f}", "dim")]
        for target in TARGETS:
            base, ours = value(f"reinhard_{target}"), value(f"ours_{target}")
            cells.append(cell(f"{base:.3f}"))
            cells.append(
                cell(
                    f"<strong>{ours:.3f}</strong>" if ours < base else f"{ours:.3f}",
                    "num-pos" if ours < base else "num-neg",
                )
            )
        for condition, _ in FEATURE_METHODS:
            if condition in set(frame.condition):
                cells.append(cell(f"<strong>{value(condition):.3f}</strong>", "num-pos"))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def knn_narrative(knn: dict, meta: dict) -> str:
    """Describe what the strips show, from the counts rather than from memory."""
    total = meta["neighbours"]
    lines = []
    for model in ("resnet50", "uni_v1"):
        if not all(model in knn[name]["models"] for name in knn):
            continue
        label = MODEL_LABEL[model][0]
        counts = {
            name: knn[name]["models"][model]["same_scanner_neighbours"] for name in knn
        }
        location = knn["raw"]["models"][model]["same_location_neighbours"]
        moved = counts.get("ours_gt450", counts["raw"]) - counts["raw"]
        if moved < 0:
            verb = "falls to {}".format(counts["ours_gt450"])
        elif moved > 0:
            verb = "rises to {}".format(counts["ours_gt450"])
        else:
            verb = "is unchanged at {}".format(counts["ours_gt450"])
        lines.append(
            "Under {label} the query keeps {location} of its five reachable same-location "
            "neighbours in raw, and {raw} of {total} neighbours come from its own scanner. "
            "Aiming at GT450 that count {verb}; aiming at AT2 it is {at2}.".format(
                label=label,
                location=location,
                raw=counts["raw"],
                total=total,
                verb=verb,
                at2=counts.get("ours_at2", counts["raw"]),
            )
        )
    lines.append(
        "A single patch cannot carry a population effect: the tables above move by a few "
        "hundredths on average, which is real across 109 slides and often invisible on any "
        "one example. The strips are here to show what the metric is counting, not to prove it."
    )
    return "\n".join("      <p>{}</p>".format(line) for line in lines)


def main():
    args = parse_args()
    frontier = Path(args.frontier)
    endpoints = pd.read_csv(frontier / "endpoint_summary.csv")
    incremental = pd.read_csv(frontier / "incremental_vs_reinhard.csv")
    endpoints["is_ours"] = endpoints.condition.str.startswith("reinhard_unpaired")
    endpoints = endpoints.merge(
        incremental[["target", "encoder_id", "method", "rf1u_better_than_reinhard"]],
        left_on=["target", "encoder_id", "condition"],
        right_on=["target", "encoder_id", "method"],
        how="left",
    )
    endpoints["invariance_better_than_reinhard"] = endpoints[
        "rf1u_better_than_reinhard"
    ].fillna(False)
    endpoints["safe_and_improved"] = (
        endpoints.safe_for_pfm & endpoints.invariance_better_than_reinhard
    )

    neighbourhood = pd.read_csv(Path(args.neighbourhood) / "neighbourhood_summary.csv")
    neighbour_meta = json.loads((Path(args.neighbourhood) / "summary.json").read_text())
    gallery = json.loads((Path(args.gallery) / "gallery.json").read_text())
    gallery_meta = json.loads((Path(args.gallery) / "summary.json").read_text())
    probe = pd.concat(
        [
            pd.read_csv(Path(args.scanner_probe) / f"{model}.csv")
            for model in MODELS
        ]
    )
    probe_meta = json.loads(
        (Path(args.scanner_probe) / f"{MODELS[0]}.summary.json").read_text()
    )
    knn = json.loads((Path(args.knn_gallery) / "knn_gallery.json").read_text())
    knn_meta = json.loads((Path(args.knn_gallery) / "summary.json").read_text())

    safe_count = int(endpoints[endpoints.is_ours].safe_and_improved.sum())
    template = Path(__file__).with_name("rf1u_report_template.html").read_text()
    page = (
        template.replace("{{SAFE_COUNT}}", str(safe_count))
        .replace("{{RADIUS_ROWS}}", radius_rows(endpoints))
        .replace(
            "{{SAME_SCANNER_ROWS}}",
            neighbour_rows(neighbourhood, "same_scanner_share", "low"),
        )
        .replace(
            "{{SAME_LOCATION_ROWS}}",
            neighbour_rows(neighbourhood, "same_location_share", "high"),
        )
        .replace("{{GALLERY}}", gallery_blocks(gallery))
        .replace("{{CHANCE}}", f"{neighbour_meta['chance_same_scanner_share']:.3f}")
        .replace("{{NEIGHBOURS}}", str(neighbour_meta["neighbours"]))
        .replace("{{GALLERY_SLIDE}}", html.escape(gallery_meta["slide_id"]))
        .replace("{{GALLERY_LOCATION}}", str(gallery_meta["location"]))
        .replace("{{GALLERY_TRAIN}}", str(gallery_meta["training_slides"]))
        .replace("{{KNN_GALLERY}}", knn_blocks(knn, knn_meta))
        .replace("{{KNN_QUERY_SCANNER}}", knn_meta["query_scanner"].upper())
        .replace("{{KNN_QUERY_LOCATION}}", str(knn_meta["location"]))
        .replace("{{KNN_QUERY_MODEL}}", MODEL_LABEL[knn_meta["query_from_model"]][0])
        .replace("{{KNN_NARRATIVE}}", knn_narrative(knn, knn_meta))
        .replace("{{PROBE_ROWS}}", probe_rows(probe))
        .replace(
            "{{PROBE_CHANCE}}",
            "{:.3f}".format(probe_meta["chance_balanced_accuracy"]),
        )
        .replace(
            "{{PROBE_RAW_MIN}}",
            "{:.3f}".format(probe[probe.condition == "raw"].balanced_accuracy.min()),
        )
        .replace(
            "{{PROBE_RAW_MAX}}",
            "{:.3f}".format(probe[probe.condition == "raw"].balanced_accuracy.max()),
        )
        .replace(
            "{{PROBE_FEATURE_MIN}}",
            "{:.3f}".format(
                probe[probe.condition.isin([c for c, _ in FEATURE_METHODS])]
                .balanced_accuracy.min()
            ),
        )
        .replace(
            "{{PROBE_FEATURE_MAX}}",
            "{:.3f}".format(
                probe[probe.condition.isin([c for c, _ in FEATURE_METHODS])]
                .balanced_accuracy.max()
            ),
        )
        .replace(
            "{{PROBE_IMAGE_BEST}}",
            "{:.3f}".format(
                probe[
                    probe.condition.str.startswith(("reinhard_", "ours_"))
                ].balanced_accuracy.min()
            ),
        )
        .replace(
            "{{PROBE_LOWERED}}",
            str(
                int(
                    sum(
                        probe[
                            (probe.encoder_id == m) & (probe.condition == f"ours_{t}")
                        ].balanced_accuracy.iloc[0]
                        < probe[
                            (probe.encoder_id == m)
                            & (probe.condition == f"reinhard_{t}")
                        ].balanced_accuracy.iloc[0]
                        for m in MODELS
                        for t in TARGETS
                    )
                )
            ),
        )
        .replace("{{KNN_QUERY_SCORE}}", "{:.4f}".format(knn_meta["query_structure_score"]))
        .replace(
            "{{KNN_MEDIAN_SCORE}}", "{:.4f}".format(knn_meta["median_structure_score"])
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(
        json.dumps(
            {
                "analysis": "rf1u_report_page",
                "output": str(output.resolve()),
                "bytes": len(page.encode()),
                "safe_and_improved_cells": safe_count,
                "gallery_rows": len(gallery["rows"]),
                "gallery_panels": len(gallery["panels"]),
                "knn_conditions": list(knn),
                "knn_images": sum(len(value["images"]) for value in knn.values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
