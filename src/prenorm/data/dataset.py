"""Registered multi-scanner RGB tuple dataset for the v3 store."""

import os
from collections import OrderedDict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import store
from utils.index import read_index

from .transforms import rgb_to_tensor, tuple_geometric_aug


class PairedTupleDataset(Dataset):
    """Read one clean, registered RGB patch per scanner at a shared coordinate."""

    def __init__(self, index_path, store_dir, scanners, cfg, split, train):
        frame = read_index(index_path)
        self.scanners = list(scanners)
        self._validate_index(frame)

        frame = frame[frame["split"] == split].copy()
        if "kept" in frame.columns:
            frame = frame[frame["kept"]]
        frame = frame.sort_values(["slide_id", "tile_id"]).reset_index(drop=True)

        self.rows = frame.to_dict("records")
        self.store_dir = store_dir
        self.train = bool(train)
        self.handle_cache_size = int(getattr(cfg.loader, "handle_cache", 4))
        self.aug = tuple(getattr(cfg.loader, "aug", ()))
        self.nuclei_store = getattr(getattr(cfg, "paths", None), "nuclei_store", None)

        # Samplers use row positions so dataset reads stay independent of index labels.
        self.slide_to_rows = {}
        for row_index, row in enumerate(self.rows):
            self.slide_to_rows.setdefault(row["slide_id"], []).append(row_index)

        self._handles = OrderedDict()
        self._nuclei_handles = OrderedDict()
        self._rng = None

    def _validate_index(self, frame):
        required = {
            "split", "slide_id", "tuple_id", "tile_id", "x", "y", "tissue_density"
        }
        for scanner in self.scanners:
            required.update(
                {f"present_{scanner}", f"geom_ok_{scanner}", f"q_reg_{scanner}"}
            )
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"v3 index is missing required columns: {', '.join(missing)}")

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        """Discard process-local HDF5 state when DataLoader workers are spawned."""
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_nuclei_handles"] = OrderedDict()
        state["_rng"] = None
        return state

    def _get_handle(self, slide_id):
        handle = self._handles.get(slide_id)
        if handle is not None:
            self._handles.move_to_end(slide_id)
            return handle

        path = os.path.join(self.store_dir, f"{slide_id}.h5")
        handle = h5py.File(path, "r")
        if int(handle.attrs.get("schema_version", -1)) != 3:
            handle.close()
            raise ValueError(f"Expected v3 RGB store: {path}")
        self._handles[slide_id] = handle
        if len(self._handles) > self.handle_cache_size:
            self._handles.popitem(last=False)[1].close()
        return handle

    def _get_nuclei_handle(self, slide_id):
        handle = self._nuclei_handles.get(slide_id)
        if handle is not None:
            self._nuclei_handles.move_to_end(slide_id)
            return handle
        path = os.path.join(self.nuclei_store, f"{slide_id}.h5")
        handle = h5py.File(path, "r")
        if int(handle.attrs.get("schema_version", -1)) != 1:
            handle.close()
            raise ValueError(f"Expected StarDist nuclei store: {path}")
        self._nuclei_handles[slide_id] = handle
        if len(self._nuclei_handles) > self.handle_cache_size:
            self._nuclei_handles.popitem(last=False)[1].close()
        return handle

    def _get_rng(self):
        """Create a deterministic worker-local NumPy stream from the PyTorch seed."""
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2**32))
        return self._rng

    @staticmethod
    def _read_rgb(handle, scanner, tuple_id):
        """Read RGB while deliberately ignoring any extra datasets in the group."""
        value = np.asarray(handle[scanner]["rgb"][tuple_id])
        rgb = store.decode(value) if value.ndim == 1 else value
        return np.ascontiguousarray(rgb)

    def __getitem__(self, index):
        row = self.rows[index]
        handle = self._get_handle(row["slide_id"])
        tuple_id = int(row["tuple_id"])
        images = [self._read_rgb(handle, scanner, tuple_id) for scanner in self.scanners]
        labels = None
        if self.nuclei_store is not None:
            nuclei = self._get_nuclei_handle(row["slide_id"])
            labels = [np.asarray(nuclei[scanner]["labels"][tuple_id]) for scanner in self.scanners]

        if self.train:
            combined = images if labels is None else images + labels
            combined = tuple_geometric_aug(combined, self._get_rng(), self.aug)
            images = combined[:len(self.scanners)]
            if labels is not None:
                labels = combined[len(self.scanners):]

        item = {
            "rgb": torch.stack([rgb_to_tensor(image) for image in images]),
            "present": torch.tensor(
                [bool(row[f"present_{scanner}"]) for scanner in self.scanners]
            ),
            "geom_ok": torch.tensor(
                [bool(row[f"geom_ok_{scanner}"]) for scanner in self.scanners]
            ),
            "q_reg": torch.tensor(
                [float(row[f"q_reg_{scanner}"]) for scanner in self.scanners],
                dtype=torch.float32,
            ),
            "slide_id": row["slide_id"],
            "tile_id": int(row["tile_id"]),
            "x": int(row["x"]),
            "y": int(row["y"]),
            "tissue_density": float(row["tissue_density"]),
        }
        if labels is not None:
            item["nuclei_labels"] = torch.stack(
                [torch.from_numpy(label.astype(np.int32))[None] for label in labels]
            )
        return item
