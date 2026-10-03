"""Data contract and loaders for the paired scanner-to-AT2 Pix2Pix baseline.

The canonical cache contains one HDF5 shard per physical slide.  Every row in
``images`` is a six-scanner, same-location tuple with shape
``(scanner, 256, 256, RGB)``.  This module keeps HDF5 access lazy and local to a
DataLoader worker; it never copies the 117 GB cache.

The primary translation direction is deliberately explicit: ``source_scanner``
is one of GT450, VERSA, or AKOYA and the fixed target is AT2.  Scanner positions
are looked up from each shard's ``scanner_names`` dataset rather than assumed.

Registration convention
-----------------------
For scanner ``s``, define ``q_s = gradient_shift_xy_s - applied_shift_xy_s``.
A pixel in the stored scanner crop at coordinate ``p`` corresponds to an AT2
pixel at ``p - q_s``.  Consequently, when the source is scanner ``s`` and the
target is scanner ``t``, the target crop must start at
``q_t - q_s`` relative to the source crop.  For the primary ``s -> AT2`` task,
this is ``applied_shift_xy_s - gradient_shift_xy_s``.  HDF5 shifts are stored
as ``(x, y)``; NumPy slices are always constructed as ``(y, x)`` here.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterator, NamedTuple, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


PRIMARY_SOURCE_SCANNERS = ("gt450", "versa", "akoya")
PRIMARY_TARGET_SCANNER = "at2"
DEFAULT_VALID_CROP_SIZE = 252
INDEX_COLUMNS = (
    "sample_index",
    "slide_id",
    "tissue_type",
    "fold",
    "cache_path",
    "location_index",
    "source_index",
    "scanner_order",
)


def _stable_seed(*values: object) -> int:
    token = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)


def _decode_names(values: np.ndarray) -> tuple[str, ...]:
    names = tuple(
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    )
    normalized = tuple(name.strip().lower() for name in names)
    if not normalized or any(not name for name in normalized):
        raise ValueError(f"invalid scanner_names: {names!r}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"duplicate scanner_names: {names!r}")
    return normalized


def read_scanner_names(cache_path: str | Path) -> tuple[str, ...]:
    """Read and validate the ordered scanner names from one cache shard."""

    path = Path(cache_path)
    with h5py.File(path, "r") as store:
        if "scanner_names" not in store:
            raise KeyError(f"{path}: missing scanner_names")
        names = _decode_names(np.asarray(store["scanner_names"][:]))
        if "images" not in store or store["images"].ndim != 5:
            raise ValueError(
                f"{path}: images must have (location, scanner, y, x, rgb) shape"
            )
        if int(store["images"].shape[1]) != len(names):
            raise ValueError(f"{path}: scanner_names/images scanner dimension mismatch")
    return names


def scanner_lookup(scanner_names: Sequence[str]) -> dict[str, int]:
    """Return a case-insensitive name-to-column lookup with duplicate checks."""

    names = tuple(str(name).strip().lower() for name in scanner_names)
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate scanner names: {names!r}")
    return {name: index for index, name in enumerate(names)}


def verify_scanner_names(
    cache_path: str | Path,
    expected: Sequence[str],
) -> tuple[str, ...]:
    """Verify that a shard has exactly the expected scanner order."""

    actual = read_scanner_names(cache_path)
    expected_normalized = tuple(str(name).strip().lower() for name in expected)
    if actual != expected_normalized:
        raise ValueError(
            f"{Path(cache_path)}: scanner order {actual!r} != {expected_normalized!r}"
        )
    return actual


def _read_frame(
    value: str | Path | pd.DataFrame, *, slide_ids: bool = True
) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    dtype = {"slide_id": str} if slide_ids else None
    return pd.read_csv(Path(value), dtype=dtype)


def assign_split_roles(
    sample_index: pd.DataFrame,
    test_fold: int,
    validation_fold: int | None = None,
) -> pd.DataFrame:
    """Attach mutually exclusive slide-level train/validation/test roles.

    By contract, validation is the fold immediately after the outer test fold.
    The explicit ``validation_fold`` argument exists for audits and must be
    different from ``test_fold``.
    """

    test_fold = int(test_fold)
    validation_fold = (
        (test_fold + 1) % 5 if validation_fold is None else int(validation_fold)
    )
    if test_fold == validation_fold:
        raise ValueError("test and validation folds must differ")
    folds = pd.to_numeric(sample_index["fold"], errors="raise").astype(int)
    available = set(folds.unique().tolist())
    if test_fold not in available or validation_fold not in available:
        raise ValueError(
            f"requested test/validation folds {(test_fold, validation_fold)} not in {sorted(available)}"
        )
    result = sample_index.copy()
    result["split_role"] = np.where(
        folds == test_fold,
        "test",
        np.where(folds == validation_fold, "validation", "train"),
    )
    # A slide is the inferential and leakage-control unit.  Assert that a slide
    # has not somehow received multiple fold labels in a hand-edited index.
    roles_per_slide = result.groupby("slide_id", sort=False)["split_role"].nunique()
    if int(roles_per_slide.max()) != 1:
        offenders = roles_per_slide[roles_per_slide != 1].index.tolist()
        raise ValueError(f"slides assigned to multiple split roles: {offenders[:5]}")
    return result


def build_sample_index(
    cohort: str | Path | pd.DataFrame,
    folds: str | Path | pd.DataFrame | None = None,
    output_path: str | Path | None = None,
    *,
    expected_scanners: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build the location-level index from the frozen cohort.

    Scanner order is inferred from the first shard unless ``expected_scanners``
    is supplied, then verified against every shard.  The resulting order is
    stored in every row so a saved CSV remains self-describing.
    """

    cohort_frame = _read_frame(cohort)
    required = {"slide_id", "tissue_type", "cache_path", "pair_count"}
    missing = required - set(cohort_frame.columns)
    if missing:
        raise ValueError(f"cohort is missing columns: {sorted(missing)}")
    if cohort_frame["slide_id"].duplicated().any():
        raise ValueError("cohort contains duplicate slide_id values")

    if folds is not None:
        fold_frame = _read_frame(folds)
        if set(fold_frame.columns) < {"slide_id", "fold"}:
            raise ValueError("fold table must contain slide_id and fold")
        if fold_frame["slide_id"].duplicated().any():
            raise ValueError("fold table contains duplicate slide_id values")
        if "fold" in cohort_frame:
            cohort_frame = cohort_frame.drop(columns="fold")
        cohort_frame = cohort_frame.merge(
            fold_frame[["slide_id", "fold"]],
            on="slide_id",
            how="left",
            validate="one_to_one",
        )
    if "fold" not in cohort_frame or cohort_frame["fold"].isna().any():
        raise ValueError("every cohort slide must have a fold assignment")

    canonical = (
        tuple(str(name).strip().lower() for name in expected_scanners)
        if expected_scanners is not None
        else None
    )
    pieces: list[pd.DataFrame] = []
    next_index = 0
    for row in cohort_frame.itertuples(index=False):
        path = Path(str(row.cache_path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        names = read_scanner_names(path)
        if canonical is None:
            canonical = names
        if names != canonical:
            raise ValueError(
                f"{path}: scanner order {names!r} != canonical {canonical!r}"
            )
        lookup = scanner_lookup(names)
        required_scanners = {PRIMARY_TARGET_SCANNER, *PRIMARY_SOURCE_SCANNERS}
        if not required_scanners.issubset(lookup):
            raise ValueError(
                f"{path}: missing required scanners {sorted(required_scanners - set(lookup))}"
            )
        with h5py.File(path, "r") as store:
            n_locations = int(store["images"].shape[0])
            if n_locations != int(row.pair_count):
                raise ValueError(
                    f"{path}: cache has {n_locations} rows but cohort records {int(row.pair_count)}"
                )
            source_indices = np.asarray(store["source_index"][:], dtype=np.int64)
            for required_dataset in (
                "alignment/gradient_shift_xy",
                "applied_shift_xy",
            ):
                if required_dataset not in store:
                    raise KeyError(f"{path}: missing {required_dataset}")
        if source_indices.shape != (n_locations,):
            raise ValueError(
                f"{path}: invalid source_index shape {source_indices.shape}"
            )
        pieces.append(
            pd.DataFrame(
                {
                    "sample_index": np.arange(
                        next_index, next_index + n_locations, dtype=np.int64
                    ),
                    "slide_id": str(row.slide_id),
                    "tissue_type": str(row.tissue_type),
                    "fold": int(row.fold),
                    "cache_path": str(path),
                    "location_index": np.arange(n_locations, dtype=np.int64),
                    "source_index": source_indices,
                    "scanner_order": ",".join(names),
                }
            )
        )
        next_index += n_locations
    if not pieces:
        raise ValueError("cohort contains no slides")
    result = pd.concat(pieces, ignore_index=True)
    if result["sample_index"].duplicated().any():
        raise AssertionError("internal error: duplicate sample_index")

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        result.to_csv(temporary, index=False)
        temporary.replace(path)
        manifest = {
            "rows": int(len(result)),
            "slides": int(result["slide_id"].nunique()),
            "scanner_names": list(canonical or ()),
            "primary_source_scanners": list(PRIMARY_SOURCE_SCANNERS),
            "target_scanner": PRIMARY_TARGET_SCANNER,
        }
        manifest_path = path.with_suffix(path.suffix + ".json")
        manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest_tmp.replace(manifest_path)
    return result


def load_sample_index(
    path: str | Path,
    *,
    verify_caches: bool = False,
) -> pd.DataFrame:
    """Load and structurally validate a location-level sample index."""

    frame = pd.read_csv(Path(path), dtype={"slide_id": str, "cache_path": str})
    missing = set(INDEX_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"sample index is missing columns: {sorted(missing)}")
    for column in ("sample_index", "fold", "location_index", "source_index"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    if frame["sample_index"].duplicated().any():
        raise ValueError("sample_index values are not unique")
    if frame[["slide_id", "location_index"]].duplicated().any():
        raise ValueError("(slide_id, location_index) values are not unique")
    orders = frame["scanner_order"].drop_duplicates().tolist()
    if len(orders) != 1:
        raise ValueError(f"sample index contains multiple scanner orders: {orders}")
    names = tuple(str(orders[0]).split(","))
    scanner_lookup(names)
    if verify_caches:
        for cache_path in frame["cache_path"].drop_duplicates():
            verify_scanner_names(cache_path, names)
    return frame


class WorkerLocalH5Handles:
    """Small per-process LRU of read-only HDF5 handles.

    Handles are discarded after fork/pickle and reopened in the consuming
    worker.  This avoids sharing an h5py handle across DataLoader processes.
    """

    def __init__(
        self, expected_scanners: Sequence[str], max_open_files: int = 8
    ) -> None:
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.expected_scanners = tuple(
            str(name).strip().lower() for name in expected_scanners
        )
        self.max_open_files = int(max_open_files)
        self._pid = os.getpid()
        self._handles: OrderedDict[str, h5py.File] = OrderedDict()

    def _ensure_process(self) -> None:
        if self._pid != os.getpid():
            # In a forked worker, inherited HDF5 identifiers must not be used.
            self._handles = OrderedDict()
            self._pid = os.getpid()

    def get(self, cache_path: str | Path) -> h5py.File:
        self._ensure_process()
        key = str(Path(cache_path).resolve())
        handle = self._handles.pop(key, None)
        if handle is None:
            handle = h5py.File(key, "r")
            names = _decode_names(np.asarray(handle["scanner_names"][:]))
            if names != self.expected_scanners:
                handle.close()
                raise ValueError(
                    f"{key}: scanner order {names!r} != {self.expected_scanners!r}"
                )
        self._handles[key] = handle
        while len(self._handles) > self.max_open_files:
            _, oldest = self._handles.popitem(last=False)
            oldest.close()
        return handle

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_pid"] = -1
        return state

    def __del__(self) -> None:
        self.close()


def residual_offset_xy(
    store: h5py.File,
    location_index: int,
    scanner_index: int,
) -> np.ndarray:
    """Return ``gradient - applied`` residual for one scanner in ``(x, y)``."""

    gradient = np.asarray(
        store["alignment/gradient_shift_xy"][location_index, scanner_index],
        dtype=np.int64,
    )
    applied = np.asarray(
        store["applied_shift_xy"][location_index, scanner_index], dtype=np.int64
    )
    if gradient.shape != (2,) or applied.shape != (2,):
        raise ValueError("shift arrays must contain an (x, y) pair")
    return gradient - applied


def target_crop_shift_xy(
    store: h5py.File,
    location_index: int,
    source_scanner_index: int,
    target_scanner_index: int,
) -> np.ndarray:
    """Return target-crop offset relative to the central source crop.

    The general expression is ``q_target - q_source``.  For the primary
    scanner-to-AT2 direction this equals ``applied_source - gradient_source``.
    """

    source_residual = residual_offset_xy(store, location_index, source_scanner_index)
    target_residual = residual_offset_xy(store, location_index, target_scanner_index)
    return target_residual - source_residual


def aligned_valid_crop_slices(
    target_shift_xy: Sequence[int],
    patch_shape: Sequence[int] = (256, 256),
    crop_size: int = DEFAULT_VALID_CROP_SIZE,
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """Build source/target ``(y, x)`` slices for a registered valid crop."""

    height, width = map(int, patch_shape[:2])
    crop_size = int(crop_size)
    if crop_size <= 0 or crop_size > height or crop_size > width:
        raise ValueError(f"invalid crop_size {crop_size} for patch {(height, width)}")
    shift = np.asarray(target_shift_xy, dtype=np.int64)
    if shift.shape != (2,):
        raise ValueError("target_shift_xy must be a two-element (x, y) vector")
    dx, dy = map(int, shift)  # HDF5 convention is x then y.
    source_y = (height - crop_size) // 2
    source_x = (width - crop_size) // 2
    target_y = source_y + dy
    target_x = source_x + dx
    if (
        target_y < 0
        or target_x < 0
        or target_y + crop_size > height
        or target_x + crop_size > width
    ):
        raise ValueError(
            f"target shift {(dx, dy)} is outside the valid {crop_size}px crop for {(height, width)}"
        )
    return (
        (slice(source_y, source_y + crop_size), slice(source_x, source_x + crop_size)),
        (slice(target_y, target_y + crop_size), slice(target_x, target_x + crop_size)),
    )


def crop_aligned_pair(
    source: np.ndarray,
    target: np.ndarray,
    target_shift_xy: Sequence[int],
    crop_size: int = DEFAULT_VALID_CROP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop two HWC images so target pixels align to the central source crop."""

    source = np.asarray(source)
    target = np.asarray(target)
    if source.ndim != 3 or target.ndim != 3 or source.shape[:2] != target.shape[:2]:
        raise ValueError("source and target must be same-sized HWC arrays")
    source_slices, target_slices = aligned_valid_crop_slices(
        target_shift_xy, source.shape[:2], crop_size
    )
    sy, sx = source_slices
    ty, tx = target_slices
    return (
        np.ascontiguousarray(source[sy, sx, ...]),
        np.ascontiguousarray(target[ty, tx, ...]),
    )


def apply_d4(image: np.ndarray, transform: int) -> np.ndarray:
    """Apply one of eight square symmetries to an HWC image."""

    value = np.asarray(image)
    if value.ndim != 3 or value.shape[0] != value.shape[1]:
        raise ValueError("D4 augmentation requires a square HWC image")
    transform = int(transform)
    if transform < 0 or transform >= 8:
        raise ValueError("D4 transform must be in 0..7")
    if transform >= 4:
        value = np.flip(value, axis=1)  # reflection across the vertical axis (x -> -x)
    value = np.rot90(value, k=transform % 4, axes=(0, 1))
    return np.ascontiguousarray(value)


def apply_synchronized_d4(
    images: Sequence[np.ndarray], transform: int
) -> tuple[np.ndarray, ...]:
    """Apply exactly the same D4 geometry to all paired image views."""

    return tuple(apply_d4(image, transform) for image in images)


def build_slide_balanced_epoch(
    sample_index: pd.DataFrame,
    epoch: int,
    *,
    locations_per_slide: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Return deterministic row positions for one slide-balanced pass.

    Each slide contributes ``locations_per_slide`` rows.  For slide ``s``, the
    conceptual stream is a concatenation of independently shuffled complete
    cycles through its locations.  Adjacent epochs therefore continue that
    stream without replacement until a cycle is exhausted.
    """

    if locations_per_slide <= 0:
        raise ValueError("locations_per_slide must be positive")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    frame = sample_index.reset_index(drop=True)
    if frame.empty:
        raise ValueError("cannot balance an empty sample index")
    selected_by_slide: dict[str, np.ndarray] = {}
    stream_start = int(epoch) * int(locations_per_slide)
    stream_stop = stream_start + int(locations_per_slide)
    for slide_id, group in frame.groupby("slide_id", sort=True):
        positions = group.index.to_numpy(dtype=np.int64)
        if len(positions) == 0:
            raise AssertionError("empty slide group")
        pieces = []
        first_cycle = stream_start // len(positions)
        last_cycle = (stream_stop - 1) // len(positions)
        for cycle in range(first_cycle, last_cycle + 1):
            rng = np.random.default_rng(_stable_seed(seed, slide_id, "cycle", cycle))
            pieces.append(rng.permutation(positions))
        stream = np.concatenate(pieces)
        offset = stream_start - first_cycle * len(positions)
        selected_by_slide[str(slide_id)] = stream[offset : offset + locations_per_slide]

    # Interleave slides for better I/O/GAN batch mixing while keeping the exact
    # per-slide exposure fixed.  Every round uses its own deterministic order.
    slides = np.asarray(sorted(selected_by_slide), dtype=object)
    output = []
    for draw in range(locations_per_slide):
        rng = np.random.default_rng(_stable_seed(seed, "slide-order", epoch, draw))
        for slide_id in rng.permutation(slides):
            output.append(int(selected_by_slide[str(slide_id)][draw]))
    result = np.asarray(output, dtype=np.int64)
    expected = sample_index["slide_id"].nunique() * locations_per_slide
    if len(result) != expected:
        raise AssertionError(
            f"balanced epoch has {len(result)} rows, expected {expected}"
        )
    return result


class EpochIndex(NamedTuple):
    position: int
    epoch: int


class UnpairedEpochIndex(NamedTuple):
    source_position: int
    target_position: int
    epoch: int


def build_unpaired_slide_balanced_epoch(
    sample_index: pd.DataFrame,
    epoch: int,
    *,
    locations_per_slide: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Return independent, slide-balanced source and target row positions.

    Every slide contributes exactly ``locations_per_slide`` observations to
    each domain. Within each draw, target slides are a cyclic derangement of
    the source-slide order, so a training pair never shares a physical slide
    and therefore can never expose the same-location correspondence.
    """

    frame = sample_index.reset_index(drop=True)
    slides = frame["slide_id"].astype(str)
    n_slides = int(slides.nunique())
    if n_slides < 2:
        raise ValueError("unpaired sampling requires at least two physical slides")
    source = build_slide_balanced_epoch(
        frame,
        epoch,
        locations_per_slide=locations_per_slide,
        seed=seed,
    )
    target = build_slide_balanced_epoch(
        frame,
        epoch,
        locations_per_slide=locations_per_slide,
        seed=_stable_seed(seed, "independent-target-stream"),
    )
    pairs: list[tuple[int, int]] = []
    for draw in range(locations_per_slide):
        start = draw * n_slides
        stop = start + n_slides
        source_block = source[start:stop]
        target_block = target[start:stop]
        source_order = slides.iloc[source_block].to_numpy(dtype=object)
        target_lookup = {
            str(slides.iloc[int(position)]): int(position)
            for position in target_block
        }
        if len(target_lookup) != n_slides:
            raise AssertionError("target block is not slide balanced")
        offset = 1 + (
            _stable_seed(seed, "target-slide-rotation", epoch, draw)
            % (n_slides - 1)
        )
        rotated_target_slides = np.roll(source_order, -int(offset))
        for source_position, target_slide in zip(source_block, rotated_target_slides):
            target_position = target_lookup[str(target_slide)]
            if str(slides.iloc[int(source_position)]) == str(target_slide):
                raise AssertionError("unpaired sampler produced a same-slide pair")
            pairs.append((int(source_position), int(target_position)))
    result = np.asarray(pairs, dtype=np.int64)
    expected = n_slides * int(locations_per_slide)
    if result.shape != (expected, 2):
        raise AssertionError(
            f"unpaired epoch has shape {result.shape}, expected {(expected, 2)}"
        )
    source_counts = slides.iloc[result[:, 0]].value_counts().sort_index()
    target_counts = slides.iloc[result[:, 1]].value_counts().sort_index()
    if not (source_counts == locations_per_slide).all() or not (
        target_counts == locations_per_slide
    ).all():
        raise AssertionError("unpaired epoch lost exact slide balance")
    return result


class SlideBalancedSampler(Sampler[EpochIndex]):
    """PyTorch sampler for deterministic, epoch-aware slide balancing."""

    def __init__(
        self,
        sample_index: pd.DataFrame,
        locations_per_slide: int = 100,
        seed: int = 0,
    ) -> None:
        self.sample_index = sample_index.reset_index(drop=True)[["slide_id"]].copy()
        self.locations_per_slide = int(locations_per_slide)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[EpochIndex]:
        positions = build_slide_balanced_epoch(
            self.sample_index,
            self.epoch,
            locations_per_slide=self.locations_per_slide,
            seed=self.seed,
        )
        return iter(EpochIndex(int(position), self.epoch) for position in positions)

    def __len__(self) -> int:
        return int(self.sample_index["slide_id"].nunique()) * self.locations_per_slide


class UnpairedSlideBalancedSampler(Sampler[UnpairedEpochIndex]):
    """Epoch-aware independent-domain sampler with exact slide balance."""

    def __init__(
        self,
        sample_index: pd.DataFrame,
        locations_per_slide: int = 100,
        seed: int = 0,
    ) -> None:
        self.sample_index = sample_index.reset_index(drop=True)[
            ["slide_id", "location_index"]
        ].copy()
        self.locations_per_slide = int(locations_per_slide)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[UnpairedEpochIndex]:
        pairs = build_unpaired_slide_balanced_epoch(
            self.sample_index,
            self.epoch,
            locations_per_slide=self.locations_per_slide,
            seed=self.seed,
        )
        return iter(
            UnpairedEpochIndex(int(source), int(target), self.epoch)
            for source, target in pairs
        )

    def __len__(self) -> int:
        return int(self.sample_index["slide_id"].nunique()) * self.locations_per_slide


class PairMetadata(NamedTuple):
    slide_id: str
    tissue_type: str
    fold: int
    location_index: int
    source_index: int
    source_scanner: str
    target_scanner: str
    mapping: str
    residual_convention: str
    source_gradient_shift_x: int
    source_gradient_shift_y: int
    source_applied_shift_x: int
    source_applied_shift_y: int
    target_crop_shift_x: int
    target_crop_shift_y: int


def _to_normalized_tensor(image: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(array).to(dtype=torch.float32).div_(127.5).sub_(1.0)


class PairedScannerDataset(Dataset[dict[str, object]]):
    """Lazy paired scanner dataset for Pix2Pix.

    Returned tensors are normalized to ``[-1, 1]``.  ``source_image`` and
    ``target_image`` are full 256px registered-grid views.  ``source_valid`` is
    the central 252px source crop and ``target_valid`` is the residual-aligned
    AT2 crop.  Train the generator on ``source_image``, then central-crop its
    output and compare/condition the discriminator with the ``*_valid`` pair.
    """

    def __init__(
        self,
        sample_index: pd.DataFrame,
        source_scanner: str,
        *,
        target_scanner: str = PRIMARY_TARGET_SCANNER,
        split_role: str | None = None,
        augment: bool = False,
        seed: int = 0,
        crop_size: int = DEFAULT_VALID_CROP_SIZE,
        max_open_files: int = 8,
        allow_bidirectional_reference_pair: bool = False,
    ) -> None:
        source_scanner = str(source_scanner).strip().lower()
        target_scanner = str(target_scanner).strip().lower()
        if source_scanner == target_scanner:
            raise ValueError("source_scanner and target_scanner must differ")
        if allow_bidirectional_reference_pair:
            allowed = {
                ("gt450", "at2"),
                ("at2", "gt450"),
            }
            if (source_scanner, target_scanner) not in allowed:
                raise ValueError(
                    "bidirectional reference-pair mode only permits "
                    f"GT450<->AT2, got {source_scanner}->{target_scanner}"
                )
        else:
            if target_scanner != PRIMARY_TARGET_SCANNER:
                raise ValueError(
                    f"the formal Pix2Pix target is fixed to {PRIMARY_TARGET_SCANNER!r}, "
                    f"got {target_scanner!r}"
                )
            if source_scanner not in PRIMARY_SOURCE_SCANNERS:
                raise ValueError(
                    "primary Pix2Pix source must be one of "
                    f"{PRIMARY_SOURCE_SCANNERS}, got {source_scanner!r}"
                )
        frame = sample_index.copy()
        if split_role is not None:
            if "split_role" not in frame:
                raise ValueError(
                    "split_role filtering requires assign_split_roles output"
                )
            frame = frame[frame["split_role"] == str(split_role)].copy()
        if frame.empty:
            raise ValueError("paired dataset contains no samples")
        missing = set(INDEX_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"sample index is missing columns: {sorted(missing)}")
        orders = frame["scanner_order"].drop_duplicates().tolist()
        if len(orders) != 1:
            raise ValueError(f"dataset has multiple scanner orders: {orders}")
        self.scanner_names = tuple(
            name.strip().lower() for name in str(orders[0]).split(",")
        )
        lookup = scanner_lookup(self.scanner_names)
        if source_scanner not in lookup or target_scanner not in lookup:
            raise ValueError(
                f"mapping {source_scanner}->{target_scanner} unavailable in {self.scanner_names}"
            )
        self.source_scanner = source_scanner
        self.target_scanner = target_scanner
        self.source_scanner_index = lookup[source_scanner]
        self.target_scanner_index = lookup[target_scanner]
        self.sample_index = frame.reset_index(drop=True)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.crop_size = int(crop_size)
        self.handles = WorkerLocalH5Handles(
            self.scanner_names, max_open_files=max_open_files
        )

    def __len__(self) -> int:
        return len(self.sample_index)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for direct integer indexing (EpochIndex is worker-safe)."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __getitem__(
        self, item: int | EpochIndex | tuple[int, int]
    ) -> dict[str, object]:
        if isinstance(item, EpochIndex):
            position, epoch = item
        elif isinstance(item, tuple) and len(item) == 2:
            position, epoch = map(int, item)
        else:
            position, epoch = int(item), self.epoch
        row = self.sample_index.iloc[int(position)]
        handle = self.handles.get(str(row.cache_path))
        location = int(row.location_index)
        source = np.asarray(
            handle["images"][location, self.source_scanner_index], dtype=np.uint8
        )
        target = np.asarray(
            handle["images"][location, self.target_scanner_index], dtype=np.uint8
        )
        shift_xy = target_crop_shift_xy(
            handle,
            location,
            self.source_scanner_index,
            self.target_scanner_index,
        )
        source_gradient_xy = np.asarray(
            handle["alignment/gradient_shift_xy"][location, self.source_scanner_index],
            dtype=np.int64,
        )
        source_applied_xy = np.asarray(
            handle["applied_shift_xy"][location, self.source_scanner_index],
            dtype=np.int64,
        )
        source_residual_xy = source_gradient_xy - source_applied_xy
        target_residual_xy = residual_offset_xy(
            handle, location, self.target_scanner_index
        )
        source_valid, target_valid = crop_aligned_pair(
            source, target, shift_xy, crop_size=self.crop_size
        )
        transform = 0
        if self.augment:
            transform = (
                _stable_seed(
                    self.seed,
                    epoch,
                    str(row.slide_id),
                    location,
                    self.source_scanner,
                    self.target_scanner,
                    "d4",
                )
                % 8
            )
            source, target, source_valid, target_valid = apply_synchronized_d4(
                (source, target, source_valid, target_valid), transform
            )
        metadata = PairMetadata(
            slide_id=str(row.slide_id),
            tissue_type=str(row.tissue_type),
            fold=int(row.fold),
            location_index=location,
            source_index=int(row.source_index),
            source_scanner=self.source_scanner,
            target_scanner=self.target_scanner,
            mapping=f"{self.source_scanner}->{self.target_scanner}",
            residual_convention=(
                "target_crop_xy=q_target-q_source; q=gradient_shift_xy-applied_shift_xy"
            ),
            source_gradient_shift_x=int(source_gradient_xy[0]),
            source_gradient_shift_y=int(source_gradient_xy[1]),
            source_applied_shift_x=int(source_applied_xy[0]),
            source_applied_shift_y=int(source_applied_xy[1]),
            target_crop_shift_x=int(shift_xy[0]),
            target_crop_shift_y=int(shift_xy[1]),
        )
        return {
            "source_image": _to_normalized_tensor(source),
            "target_image": _to_normalized_tensor(target),
            "source_valid": _to_normalized_tensor(source_valid),
            "target_valid": _to_normalized_tensor(target_valid),
            "target_crop_shift_xy": torch.as_tensor(shift_xy.copy(), dtype=torch.int64),
            "source_gradient_shift_xy": torch.as_tensor(
                source_gradient_xy.copy(), dtype=torch.int64
            ),
            "source_applied_shift_xy": torch.as_tensor(
                source_applied_xy.copy(), dtype=torch.int64
            ),
            "source_residual_offset_xy": torch.as_tensor(
                source_residual_xy.copy(), dtype=torch.int64
            ),
            "target_residual_offset_xy": torch.as_tensor(
                target_residual_xy.copy(), dtype=torch.int64
            ),
            "d4_transform": int(transform),
            "metadata": metadata,
        }

    def close(self) -> None:
        self.handles.close()


class UnpairedScannerDataset(Dataset[dict[str, object]]):
    """Lazy two-domain dataset that withholds same-location correspondence.

    The sampler supplies independent source and target row positions. Source
    and target D4 transforms are also independent. No registration shift or
    paired crop is read because neither is available to an unpaired method.
    """

    def __init__(
        self,
        sample_index: pd.DataFrame,
        source_scanner: str,
        target_scanner: str,
        *,
        split_role: str | None = None,
        augment: bool = False,
        seed: int = 0,
        max_open_files: int = 8,
    ) -> None:
        source_scanner = str(source_scanner).strip().lower()
        target_scanner = str(target_scanner).strip().lower()
        if source_scanner == target_scanner:
            raise ValueError("source_scanner and target_scanner must differ")
        frame = sample_index.copy()
        if split_role is not None:
            if "split_role" not in frame:
                raise ValueError(
                    "split_role filtering requires assign_split_roles output"
                )
            frame = frame[frame["split_role"] == str(split_role)].copy()
        if frame.empty:
            raise ValueError("unpaired dataset contains no samples")
        missing = set(INDEX_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"sample index is missing columns: {sorted(missing)}")
        orders = frame["scanner_order"].drop_duplicates().tolist()
        if len(orders) != 1:
            raise ValueError(f"dataset has multiple scanner orders: {orders}")
        self.scanner_names = tuple(
            name.strip().lower() for name in str(orders[0]).split(",")
        )
        lookup = scanner_lookup(self.scanner_names)
        if source_scanner not in lookup or target_scanner not in lookup:
            raise ValueError(
                f"domains {source_scanner}/{target_scanner} unavailable in "
                f"{self.scanner_names}"
            )
        self.source_scanner = source_scanner
        self.target_scanner = target_scanner
        self.source_scanner_index = lookup[source_scanner]
        self.target_scanner_index = lookup[target_scanner]
        self.sample_index = frame.reset_index(drop=True)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.handles = WorkerLocalH5Handles(
            self.scanner_names, max_open_files=max_open_files
        )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(
        self, item: UnpairedEpochIndex | tuple[int, int, int]
    ) -> dict[str, object]:
        if isinstance(item, UnpairedEpochIndex):
            source_position, target_position, epoch = item
        elif isinstance(item, tuple) and len(item) == 3:
            source_position, target_position, epoch = map(int, item)
        else:
            raise TypeError(
                "UnpairedScannerDataset requires "
                "(source_position, target_position, epoch) indices"
            )
        source_row = self.sample_index.iloc[int(source_position)]
        target_row = self.sample_index.iloc[int(target_position)]
        source_slide = str(source_row.slide_id)
        target_slide = str(target_row.slide_id)
        if source_slide == target_slide:
            raise ValueError("unpaired source and target must come from different slides")
        source_location = int(source_row.location_index)
        target_location = int(target_row.location_index)
        source_handle = self.handles.get(str(source_row.cache_path))
        target_handle = self.handles.get(str(target_row.cache_path))
        source = np.asarray(
            source_handle["images"][source_location, self.source_scanner_index],
            dtype=np.uint8,
        )
        target = np.asarray(
            target_handle["images"][target_location, self.target_scanner_index],
            dtype=np.uint8,
        )
        source_transform = 0
        target_transform = 0
        if self.augment:
            source_transform = _stable_seed(
                self.seed,
                epoch,
                source_slide,
                source_location,
                self.source_scanner,
                "source-d4",
            ) % 8
            target_transform = _stable_seed(
                self.seed,
                epoch,
                target_slide,
                target_location,
                self.target_scanner,
                "target-d4",
            ) % 8
            source = apply_d4(source, source_transform)
            target = apply_d4(target, target_transform)
        return {
            "source_image": _to_normalized_tensor(source),
            "target_image": _to_normalized_tensor(target),
            "source_position": int(source_position),
            "target_position": int(target_position),
            "source_slide_id": source_slide,
            "target_slide_id": target_slide,
            "source_location_index": source_location,
            "target_location_index": target_location,
            "source_d4_transform": int(source_transform),
            "target_d4_transform": int(target_transform),
            "accidental_same_slide": False,
            "accidental_same_location": False,
        }

    def close(self) -> None:
        self.handles.close()


def build_locked_evaluation_index(
    sample_index: pd.DataFrame,
    metrics_root: str | Path,
) -> pd.DataFrame:
    """Select the frozen 40 locations per slide with the existing chooser."""

    # Lazy import prevents the larger analysis module from becoming a DataLoader
    # worker import dependency unless this one-time index builder is requested.
    from augmentation_ood_study import choose_locations

    root = Path(metrics_root)
    pieces = []
    for slide_id, group in sample_index.groupby("slide_id", sort=True):
        metrics_path = root / str(slide_id) / "location_metrics.h5"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        locations = np.asarray(choose_locations(metrics_path), dtype=np.int64)
        if len(locations) != 40 or len(np.unique(locations)) != 40:
            raise ValueError(
                f"{slide_id}: locked chooser returned {len(locations)} non-unique/invalid rows"
            )
        locked = group[group["location_index"].isin(locations)].copy()
        if len(locked) != 40 or set(locked["location_index"].astype(int)) != set(
            locations.tolist()
        ):
            raise ValueError(f"{slide_id}: locked locations do not match sample index")
        pieces.append(locked)
    if not pieces:
        raise ValueError("cannot lock evaluation locations from an empty index")
    result = pd.concat(pieces, ignore_index=True)
    result["evaluation_locked"] = True
    return result


class LockedEvaluationDataset(PairedScannerDataset):
    """Paired dataset restricted to the pre-existing 40-location image panel."""

    def __init__(
        self,
        sample_index: pd.DataFrame,
        metrics_root: str | Path,
        source_scanner: str,
        *,
        target_scanner: str = PRIMARY_TARGET_SCANNER,
        split_role: str | None = None,
        crop_size: int = DEFAULT_VALID_CROP_SIZE,
        max_open_files: int = 8,
        allow_bidirectional_reference_pair: bool = False,
    ) -> None:
        locked = build_locked_evaluation_index(sample_index, metrics_root)
        super().__init__(
            locked,
            source_scanner,
            target_scanner=target_scanner,
            split_role=split_role,
            augment=False,
            seed=0,
            crop_size=crop_size,
            max_open_files=max_open_files,
            allow_bidirectional_reference_pair=allow_bidirectional_reference_pair,
        )


__all__ = [
    "DEFAULT_VALID_CROP_SIZE",
    "INDEX_COLUMNS",
    "PRIMARY_SOURCE_SCANNERS",
    "PRIMARY_TARGET_SCANNER",
    "EpochIndex",
    "LockedEvaluationDataset",
    "PairMetadata",
    "PairedScannerDataset",
    "SlideBalancedSampler",
    "UnpairedEpochIndex",
    "UnpairedScannerDataset",
    "UnpairedSlideBalancedSampler",
    "WorkerLocalH5Handles",
    "aligned_valid_crop_slices",
    "apply_d4",
    "apply_synchronized_d4",
    "assign_split_roles",
    "build_locked_evaluation_index",
    "build_sample_index",
    "build_slide_balanced_epoch",
    "build_unpaired_slide_balanced_epoch",
    "crop_aligned_pair",
    "load_sample_index",
    "read_scanner_names",
    "residual_offset_xy",
    "scanner_lookup",
    "target_crop_shift_xy",
    "verify_scanner_names",
]
