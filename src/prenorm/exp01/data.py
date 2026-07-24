"""Balanced AT2-target pair loading across the internal and S60 stores."""

from __future__ import annotations

import os
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from prenorm.data.transforms import rgb_to_tensor, tuple_geometric_aug
from prenorm.data.identity import INTERNAL_LATTICE_ID, S60_LATTICE_ID
from utils import store


def slide_split_map(cfg) -> dict[str, str]:
    return {
        slide: split
        for split in ("train", "val", "test")
        for slide in getattr(cfg.split.slides, split)
    }


def load_exp01_indices(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    internal = pd.read_parquet(cfg.paths.internal_index)
    s60 = pd.read_parquet(cfg.paths.s60_index)
    mapping = slide_split_map(cfg)
    expected = set(mapping)
    for name, frame in (("internal", internal), ("S60", s60)):
        unknown = set(frame.slide_id.unique()).difference(expected)
        if unknown:
            raise ValueError(f"{name} index has slides absent from Exp-01 split: {sorted(unknown)}")
    s60 = s60.copy()
    s60["split"] = s60["slide_id"].map(mapping)
    return internal, s60


class BalancedPairDataset(Dataset):
    """A flat view of valid AT2-target pairs from two independent lattices."""

    def __init__(self, cfg, split: str, train: bool):
        internal, s60 = load_exp01_indices(cfg)
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
            "akoya": (internal, str(cfg.paths.internal_store), INTERNAL_LATTICE_ID),
            "s60": (s60, str(cfg.paths.s60_store), S60_LATTICE_ID),
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
