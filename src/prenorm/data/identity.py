"""Stable sample identity for registered multi-lattice scanner data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


INTERNAL_LATTICE_ID = "internal_v3"
EXTERNAL_S60_LATTICE_ID = "external_s60_v1"
# The stored identifier remains unchanged for artifact compatibility. New
# analyses use the neutral name because this lattice is fitted/evaluated with
# the same train/test protocol as the internal scanner lattice.
S60_LATTICE_ID = EXTERNAL_S60_LATTICE_ID


@dataclass(frozen=True, order=True)
class LocationKey:
    """A registered location, namespaced by its physical sampling lattice."""

    lattice_id: str
    slide_id: str
    tuple_id: int

    def group_token(self) -> str:
        """A stable one-dimensional group label for sklearn splitters."""
        return f"{self.lattice_id}\x1f{self.slide_id}\x1f{self.tuple_id}"


def location_key(item: Mapping) -> LocationKey:
    """Build a location key and fail closed if lattice identity is absent."""
    lattice_id = str(item.get("lattice_id", "")).strip()
    if not lattice_id:
        raise ValueError("sample identity requires a non-empty lattice_id")
    return LocationKey(
        lattice_id=lattice_id,
        slide_id=str(item["slide_id"]),
        tuple_id=int(item["tuple_id"]),
    )


def scanners_by_lattice(
    records: Sequence[Mapping], scanners: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Group scanners by lattice, requiring one stable lattice per scanner."""
    scanner_order = [str(scanner) for scanner in scanners]
    observed: dict[str, set[str]] = {scanner: set() for scanner in scanner_order}
    for record in records:
        scanner = str(record["scanner"])
        if scanner in observed:
            observed[scanner].add(location_key(record).lattice_id)

    groups: dict[str, list[str]] = {}
    for scanner in scanner_order:
        lattices = observed[scanner]
        if len(lattices) != 1:
            raise ValueError(
                f"scanner {scanner!r} must have exactly one lattice, found {sorted(lattices)}"
            )
        lattice_id = next(iter(lattices))
        groups.setdefault(lattice_id, []).append(scanner)
    return {lattice: tuple(group) for lattice, group in groups.items()}


def matched_location_indices(
    records: Sequence[Mapping],
    scanners: Sequence[str],
    max_locations: int | None,
    rng: np.random.Generator,
) -> tuple[list[LocationKey], dict[str, list[int]]]:
    """Select the same location keys for every scanner in one lattice."""
    scanner_order = [str(scanner) for scanner in scanners]
    by_scanner: dict[str, dict[LocationKey, int]] = {
        scanner: {} for scanner in scanner_order
    }
    for index, record in enumerate(records):
        scanner = str(record["scanner"])
        if scanner not in by_scanner:
            continue
        key = location_key(record)
        if key in by_scanner[scanner]:
            raise ValueError(f"duplicate record for {scanner}/{key}")
        by_scanner[scanner][key] = index

    common = set.intersection(*(set(values) for values in by_scanner.values()))
    keys = sorted(common)
    if not keys:
        raise ValueError(f"no common locations for scanners {scanner_order}")
    if max_locations is not None:
        if max_locations <= 0:
            raise ValueError("max_locations must be positive")
        if len(keys) > max_locations:
            chosen = np.sort(rng.choice(len(keys), max_locations, replace=False))
            keys = [keys[int(index)] for index in chosen]

    indices = {
        scanner: [by_scanner[scanner][key] for key in keys]
        for scanner in scanner_order
    }
    return keys, indices


def matched_indices_by_lattice(
    records: Sequence[Mapping],
    scanners: Sequence[str],
    max_locations: int | None,
    rng: np.random.Generator,
) -> dict[str, dict]:
    """Return matched per-scanner indices separately for every lattice."""
    selections = {}
    for lattice_id, lattice_scanners in scanners_by_lattice(records, scanners).items():
        keys, indices = matched_location_indices(
            records, lattice_scanners, max_locations, rng
        )
        selections[lattice_id] = {
            "scanners": lattice_scanners,
            "keys": keys,
            "indices": indices,
        }
    return selections


def lattice_task_name(lattice_id: str, scanners: Sequence[str]) -> str:
    classes = "_".join(("at2", *(str(scanner) for scanner in scanners)))
    return f"{lattice_id}__{classes}"
