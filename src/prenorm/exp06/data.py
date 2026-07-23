"""Balanced AT2-target pair loading across the internal and S60 stores."""

from __future__ import annotations

import functools
import os
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from prenorm.data.transforms import rgb_to_tensor, tuple_geometric_aug
from prenorm.data.identity import INTERNAL_LATTICE_ID, EXTERNAL_S60_LATTICE_ID
from utils import store


def slide_split_map(cfg) -> dict[str, str]:
    return {
        slide: split
        for split in ("train", "val", "test")
        for slide in getattr(cfg.split.slides, split)
    }


def load_exp06_indices(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    internal = pd.read_parquet(cfg.paths.internal_index)
    s60 = pd.read_parquet(cfg.paths.s60_index)
    mapping = slide_split_map(cfg)
    expected = set(mapping)
    for name, frame in (("internal", internal), ("S60", s60)):
        unknown = set(frame.slide_id.unique()).difference(expected)
        if unknown:
            raise ValueError(f"{name} index has slides absent from Exp-06 split: {sorted(unknown)}")
    s60 = s60.copy()
    s60["split"] = s60["slide_id"].map(mapping)
    return internal, s60


class BalancedPairDataset(Dataset):
    """A flat view of valid AT2-target pairs from two independent lattices."""

    def __init__(self, cfg, split: str, train: bool):
        internal, s60 = load_exp06_indices(cfg)
        self.cfg = cfg
        self.train = bool(train)
        self.aug = tuple(getattr(cfg.loader, "aug", ())) if train else ()
        self.handle_cache_size = int(getattr(cfg.loader, "handle_cache", 4))
        self._handles: OrderedDict[tuple[str, str], h5py.File] = OrderedDict()
        self._rng = None
        self.records: list[dict] = []
        sources = {
            "gt450": (internal, str(cfg.paths.internal_store), INTERNAL_LATTICE_ID),
            "versa": (internal, str(cfg.paths.internal_store), INTERNAL_LATTICE_ID),
            "s60": (s60, str(cfg.paths.s60_store), EXTERNAL_S60_LATTICE_ID),
        }
        for scanner in cfg.target_scanners:
            frame, store_dir, lattice_id = sources[str(scanner)]
            frame = frame[frame["split"] == split]
            pair_ok = (
                frame["present_at2"].astype(bool)
                & frame["geom_ok_at2"].astype(bool)
                & frame[f"present_{scanner}"].astype(bool)
                & frame[f"geom_ok_{scanner}"].astype(bool)
            )
            for row in frame[pair_ok].sort_values(["slide_id", "tile_id"]).to_dict("records"):
                self.records.append({
                    "scanner": str(scanner),
                    "lattice_id": lattice_id,
                    "store_dir": store_dir,
                    "slide_id": str(row["slide_id"]),
                    "tuple_id": int(row["tuple_id"]),
                    "tile_id": int(row["tile_id"]),
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "tissue_density": float(row["tissue_density"]),
                    "q_reg": float(row[f"q_reg_{scanner}"]),
                })
        self.group_to_indices: dict[tuple[str, str], list[int]] = {}
        for index, record in enumerate(self.records):
            key = (record["scanner"], record["slide_id"])
            self.group_to_indices.setdefault(key, []).append(index)
        self.scanner_to_slides = {
            str(scanner): sorted({slide for target, slide in self.group_to_indices if target == scanner})
            for scanner in cfg.target_scanners
        }
        missing = [scanner for scanner, slides in self.scanner_to_slides.items() if not slides]
        if missing:
            raise ValueError(f"no valid {split} pairs for: {', '.join(missing)}")

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_rng"] = None
        return state

    def _handle(self, store_dir: str, slide_id: str):
        key = (store_dir, slide_id)
        handle = self._handles.get(key)
        if handle is not None:
            self._handles.move_to_end(key)
            return handle
        path = os.path.join(store_dir, f"{slide_id}.h5")
        handle = h5py.File(path, "r")
        if int(handle.attrs.get("schema_version", -1)) != 3:
            handle.close()
            raise ValueError(f"expected v3 RGB store: {path}")
        self._handles[key] = handle
        if len(self._handles) > self.handle_cache_size:
            self._handles.popitem(last=False)[1].close()
        return handle

    @staticmethod
    def _rgb(handle, scanner: str, tuple_id: int):
        value = np.asarray(handle[scanner]["rgb"][tuple_id])
        image = store.decode(value) if value.ndim == 1 else value
        return np.ascontiguousarray(image)

    def _get_rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2**32))
        return self._rng

    def __getitem__(self, index):
        row = self.records[int(index)]
        handle = self._handle(row["store_dir"], row["slide_id"])
        source = self._rgb(handle, row["scanner"], row["tuple_id"])
        reference = self._rgb(handle, "at2", row["tuple_id"])
        if self.train and self.aug:
            source, reference = tuple_geometric_aug(
                [source, reference], self._get_rng(), self.aug
            )
        return {
            "source": rgb_to_tensor(source),
            "reference": rgb_to_tensor(reference),
            **{key: row[key] for key in (
                "scanner", "lattice_id", "slide_id", "tuple_id", "tile_id", "x", "y",
                "tissue_density", "q_reg"
            )},
        }


class BalancedScannerBatchSampler(Sampler[list[int]]):
    """Yield query-disjoint same-slide batches with exact scanner balance."""

    def __init__(self, dataset: BalancedPairDataset, cfg, *, batches: int, train: bool):
        self.dataset = dataset
        self.context = int(cfg.context.k)
        self.queries = int(cfg.loader.queries_per_slide)
        self.min_distance = float(cfg.context.min_distance)
        self.spatial_bins = int(cfg.context.spatial_bins)
        self.batches = int(batches)
        self.train = bool(train)
        self.seed = int(cfg.runtime.seed) + (0 if train else 100_000)
        self.iteration = 0
        self.scanners = [str(scanner) for scanner in cfg.target_scanners]
        required = self.context + self.queries
        for key, indices in dataset.group_to_indices.items():
            if len(indices) < required:
                raise ValueError(f"{key} has only {len(indices)} pairs; need {required}")

    def __len__(self):
        return self.batches

    def _dispersed_context(self, candidates, rng):
        rows = self.dataset.records
        bins: dict[tuple[int, int], list[int]] = {}
        xy = np.asarray([(rows[i]["x"], rows[i]["y"]) for i in candidates], dtype=np.float64)
        lo, hi = xy.min(0), xy.max(0)
        span = np.maximum(hi - lo, 1.0)
        coords = np.minimum(((xy - lo) / span * self.spatial_bins).astype(int), self.spatial_bins - 1)
        for index, cell in zip(candidates, coords):
            bins.setdefault(tuple(cell), []).append(index)
        cells = list(bins)
        rng.shuffle(cells)
        chosen = []
        for cell in cells:
            chosen.append(int(rng.choice(bins[cell])))
            if len(chosen) == self.context:
                return chosen
        remaining = list(set(candidates).difference(chosen))
        chosen.extend(rng.choice(remaining, self.context - len(chosen), replace=False).tolist())
        return chosen

    def _sample(self, scanner, slide, rng):
        indices = self.dataset.group_to_indices[(scanner, slide)]
        rows = self.dataset.records
        for _ in range(64):
            context = self._dispersed_context(indices, rng)
            context_xy = np.asarray([(rows[i]["x"], rows[i]["y"]) for i in context])
            eligible = []
            for index in indices:
                if index in context:
                    continue
                point = np.asarray((rows[index]["x"], rows[index]["y"]))
                if np.sqrt(((context_xy - point) ** 2).sum(1)).min() >= self.min_distance:
                    eligible.append(index)
            if len(eligible) >= self.queries:
                query = rng.choice(eligible, self.queries, replace=False).tolist()
                return context + query
        raise ValueError(f"cannot draw separated context/query batch for {scanner}/{slide}")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.iteration)
        self.iteration += int(self.train)
        scanner_order = []
        while len(scanner_order) < self.batches:
            cycle = list(self.scanners)
            if self.train:
                rng.shuffle(cycle)
            scanner_order.extend(cycle)
        for scanner in scanner_order[:self.batches]:
            slide = str(rng.choice(self.dataset.scanner_to_slides[scanner]))
            yield self._sample(scanner, slide, rng)


def collate_pair_batch(batch, context: int):
    context_items, query_items = batch[:context], batch[context:]
    scanners = {item["scanner"] for item in batch}
    lattices = {item["lattice_id"] for item in batch}
    slides = {item["slide_id"] for item in batch}
    if len(scanners) != 1 or len(lattices) != 1 or len(slides) != 1:
        raise ValueError(
            "an Exp-06 batch must contain one scanner, one lattice, and one slide"
        )
    return {
        "context_source": torch.stack([item["source"] for item in context_items]),
        "context_mask": torch.ones(len(context_items), dtype=torch.bool),
        "source": torch.stack([item["source"] for item in query_items]),
        "reference": torch.stack([item["reference"] for item in query_items]),
        "q_reg": torch.tensor([item["q_reg"] for item in query_items], dtype=torch.float32),
        "scanner": next(iter(scanners)),
        "lattice_id": next(iter(lattices)),
        "slide_id": next(iter(slides)),
        "tuple_id": torch.tensor([item["tuple_id"] for item in query_items]),
        "tile_id": torch.tensor([item["tile_id"] for item in query_items]),
        "x": torch.tensor([item["x"] for item in query_items]),
        "y": torch.tensor([item["y"] for item in query_items]),
    }


class Exp06DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_ds = BalancedPairDataset(self.cfg, "train", True)
        self.val_ds = BalancedPairDataset(self.cfg, "val", False)
        self.test_ds = BalancedPairDataset(self.cfg, "test", False)

    def _loader(self, dataset, batches, train):
        sampler = BalancedScannerBatchSampler(dataset, self.cfg, batches=batches, train=train)
        workers = int(self.cfg.loader.num_workers)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=functools.partial(collate_pair_batch, context=int(self.cfg.context.k)),
            num_workers=workers,
            persistent_workers=workers > 0,
            generator=torch.Generator().manual_seed(
                int(self.cfg.runtime.seed) + (0 if train else 10_000)
            ),
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, int(self.cfg.loader.batches_per_epoch), True)

    def val_dataloader(self):
        return self._loader(self.val_ds, int(self.cfg.loader.val_batches), False)

    def test_dataloader(self):
        return self._loader(self.test_ds, int(self.cfg.loader.test_batches), False)
