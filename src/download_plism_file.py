"""Download one PLISM file from the manifest, resumable and checksum-verified.

Idempotent: a file already present with the manifest checksum is skipped, so the
array can be resubmitted after a partial run without re-downloading anything.

The figshare ndownloader URL answers 302 with a presigned S3 URL that expires in
10 seconds, so every attempt re-resolves the redirect.  ``curl -C -`` does that
on each retry, which is why the transfer is delegated to curl rather than to
urllib.  S3 refuses HEAD, so size is checked against the manifest, not the wire.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/PLISM_dataset/manifest_original.csv")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-dir", default="data/PLISM_dataset/original_wsi")
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="checksum what is on disk and exit non-zero if it does not match",
    )
    return parser.parse_args()


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    matches = manifest.loc[manifest["task_index"] == args.task_index]
    if len(matches) != 1:
        raise KeyError(f"task index {args.task_index} absent from {args.manifest}")
    row = matches.iloc[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / str(row["name"])
    partial = target.with_suffix(target.suffix + ".part")

    if target.exists():
        if target.stat().st_size == int(row["size_bytes"]) and md5sum(target) == row["md5"]:
            print(f"skip {target.name}: already complete and verified")
            return
        print(f"{target.name} present but does not verify; re-downloading")
        target.unlink()

    if args.verify_only:
        raise SystemExit(f"{target.name} missing or unverified")

    expected_gb = int(row["size_bytes"]) / 1e9
    print(f"downloading {target.name} ({expected_gb:.2f} GB) from file {row['file_id']}")

    command = [
        "curl",
        "--location",
        "--continue-at",
        "-",
        "--retry",
        str(args.retries),
        "--retry-delay",
        "15",
        "--retry-all-errors",
        "--connect-timeout",
        "60",
        # figshare stalls rather than failing; drop below 100 kB/s for 5 min and retry
        "--speed-limit",
        "102400",
        "--speed-time",
        "300",
        "--silent",
        "--show-error",
        "--output",
        str(partial),
        str(row["download_url"]),
    ]
    subprocess.run(command, check=True)

    actual_size = partial.stat().st_size
    if actual_size != int(row["size_bytes"]):
        raise RuntimeError(
            f"{target.name}: size {actual_size} != manifest {int(row['size_bytes'])}"
        )
    actual_md5 = md5sum(partial)
    if actual_md5 != row["md5"]:
        raise RuntimeError(f"{target.name}: md5 {actual_md5} != manifest {row['md5']}")

    partial.rename(target)
    print(f"ok {target.name}: {actual_size} bytes, md5 {actual_md5}")


if __name__ == "__main__":
    main()
