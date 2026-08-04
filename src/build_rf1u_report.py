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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
