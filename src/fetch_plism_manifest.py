"""Build a download manifest for the PLISM figshare+ collection.

Collection: https://doi.org/10.25452/figshare.plus.c.6773925 (CC BY 4.0, Ochi et al. 2024)

    24988074  PLISM-original  91 native WSIs, 180.8 GB, .ndpi/.svs/.tiff  <- external validation target
    23614422  PLISM-wsi       91 tar.gz of registered 512 px patches, 129.3 GB
    23590791  PLISM-sm        smartphone subset, 19.2 GB, not used here

No authentication is required.  The figshare ``download_url`` answers 302 with a
presigned S3 URL whose ``X-Amz-Expires`` is 10 seconds, so the redirect must be
followed immediately and a presigned URL must never be cached.  The manifest
therefore stores only the stable ndownloader URL.  HEAD is refused by S3; sizes
and checksums come from the API instead.

Native WSI names are ``{stain}_{scanner}.{ext}`` over 13 stains x 7 scanners.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd


API = "https://api.figshare.com/v2/articles/{article}/files?page_size=1000"

ARTICLES = {
    "original": 24988074,
    "registered": 23614422,
    "smartphone": 23590791,
}

# 40x native pixel size, Ochi et al. 2024 Table 1.  Used downstream to decide the
# common physical frequency band; kept here so the manifest is self-describing.
NATIVE_MPP = {
    "S60": 0.220,
    "S210": 0.220,
    "SQ": 0.221,
    "S360": 0.229,
    "P": 0.250,
    "AT2": 0.253,
    "GT450": 0.262,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", choices=sorted(ARTICLES), default="original")
    parser.add_argument("--output", default="data/PLISM_dataset/manifest_{article}.csv")
    return parser.parse_args()


def fetch_files(article_id: int) -> list[dict]:
    request = urllib.request.Request(
        API.format(article=article_id),
        headers={"User-Agent": "prenorm-plism-fetch/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if not payload:
        raise RuntimeError(f"figshare returned no files for article {article_id}")
    return payload


def split_name(name: str) -> tuple[str, str, str]:
    """``GIVH_AT2.svs`` -> (stain, scanner, ext); non-WSI members return blanks."""
    stem, _, ext = name.rpartition(".")
    parts = stem.split("_")
    if len(parts) != 2:
        return "", "", ext
    return parts[0], parts[1], ext


def main() -> None:
    args = parse_args()
    article_id = ARTICLES[args.article]
    files = fetch_files(article_id)

    rows = []
    for index, entry in enumerate(sorted(files, key=lambda item: item["name"])):
        stain, scanner, ext = split_name(entry["name"])
        rows.append(
            {
                "task_index": index,
                "article_id": article_id,
                "file_id": entry["id"],
                "name": entry["name"],
                "stain": stain,
                "scanner": scanner,
                "ext": ext,
                "native_mpp": NATIVE_MPP.get(scanner, float("nan")),
                "size_bytes": entry["size"],
                "md5": entry.get("supplied_md5") or entry.get("computed_md5") or "",
                "download_url": entry["download_url"],
            }
        )

    frame = pd.DataFrame(rows)
    missing = frame["md5"].eq("").sum()
    if missing:
        raise RuntimeError(f"{missing} files carry no checksum; refusing to write manifest")

    output = Path(args.output.format(article=args.article))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    total_gb = frame["size_bytes"].sum() / 1e9
    print(f"article {article_id} ({args.article})")
    print(f"  files          {len(frame)}")
    print(f"  total          {total_gb:.1f} GB")
    if frame["scanner"].astype(bool).all():
        crossed = pd.crosstab(frame["stain"], frame["scanner"])
        print(f"  design         {crossed.shape[0]} stains x {crossed.shape[1]} scanners")
        print(f"  fully crossed  {bool((crossed.to_numpy() == 1).all())}")
    print(f"  manifest       {output}")
    print(f"  array range    0-{len(frame) - 1}")


if __name__ == "__main__":
    main()
