"""Small provenance helpers shared by release and artifact tooling."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file SHA-256 without loading the whole artifact into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
