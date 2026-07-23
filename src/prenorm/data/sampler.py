"""Spatially dispersed Phase 1 batch sampling and collation."""

import numpy as np
import torch


def _rank_quartiles(values):
    """Assign balanced density quartiles without failing on duplicate values."""
    order = np.argsort(values, kind="stable")
    quartiles = np.empty(len(values), dtype=np.int64)
    quartiles[order] = np.minimum(np.arange(len(values)) * 4 // len(values), 3)
    return quartiles


def _spatial_labels(x, y, bins_per_axis):
    """Map slide coordinates to a regular slide-relative spatial grid."""
    def axis_bin(values):
        span = values.max() - values.min()
        if span == 0:
            return np.zeros(len(values), dtype=np.int64)
        scaled = (values - values.min()) / span
        return np.minimum((scaled * bins_per_axis).astype(np.int64), bins_per_axis - 1)

    return axis_bin(y) * bins_per_axis + axis_bin(x)


class SpatialTupleBatchSampler:
    """Yield ``P`` slides by ``K`` dispersed tuple coordinates per batch.

    Each K-coordinate group covers at least ``spatial_bins`` grid cells. Tissue
    density quartiles are visited round-robin where the spatial constraint allows.
    """

    def __init__(
        self,
        dataset,
        slides_per_batch,
        patches_per_slide,
        spatial_bins=4,
        seed=0,
        batches_per_epoch=None,
    ):
        self.slide_to_rows = dataset.slide_to_rows
        self.rows = dataset.rows
        self.spb = int(slides_per_batch)
        self.pps = int(patches_per_slide)
        self.spatial_bins = int(spatial_bins)
        self.seed = int(seed)
        self.epoch = 0
        self.batches_per_epoch = batches_per_epoch

        if self.pps < self.spatial_bins:
            raise ValueError("patches_per_slide must cover the requested spatial bins")

        self._metadata = {}
        for slide_id, row_indices in self.slide_to_rows.items():
            x = np.asarray([self.rows[index]["x"] for index in row_indices], dtype=np.float64)
            y = np.asarray([self.rows[index]["y"] for index in row_indices], dtype=np.float64)
            density = np.asarray(
                [self.rows[index]["tissue_density"] for index in row_indices],
                dtype=np.float64,
            )
            spatial = _spatial_labels(x, y, self.spatial_bins)
            self._metadata[slide_id] = {
                "rows": np.asarray(row_indices, dtype=np.int64),
                "spatial": spatial,
                "density": _rank_quartiles(density),
            }

        self.slides = [
            slide_id
            for slide_id, metadata in self._metadata.items()
            if len(metadata["rows"]) >= self.pps
            and len(np.unique(metadata["spatial"])) >= self.spatial_bins
        ]
        if not self.slides:
            raise ValueError("No slide can provide the requested dispersed tuple group")

        self._groups_per_slide = {
            slide_id: max(1, len(self._metadata[slide_id]["rows"]) // self.pps)
            for slide_id in self.slides
        }
        self._group_count = sum(self._groups_per_slide.values())

    def set_epoch(self, epoch):
        """Select the deterministic sampling stream for an epoch."""
        self.epoch = int(epoch)

    def __len__(self):
        if self.batches_per_epoch is not None:
            return int(self.batches_per_epoch)
        return max(1, self._group_count // self.spb)

    def _sample_group(self, slide_id, rng):
        metadata = self._metadata[slide_id]
        available = list(range(len(metadata["rows"])))
        selected = []
        used_spatial = set()
        density_offset = int(rng.integers(0, 4))

        for position in range(self.pps):
            remaining_slots = self.pps - position
            missing_spatial = self.spatial_bins - len(used_spatial)
            candidates = available

            # Reserve the final required slots for as-yet unseen spatial strata.
            if missing_spatial >= remaining_slots:
                candidates = [
                    index
                    for index in candidates
                    if metadata["spatial"][index] not in used_spatial
                ]

            target_quartile = (density_offset + position) % 4
            density_matches = [
                index
                for index in candidates
                if metadata["density"][index] == target_quartile
            ]
            if density_matches:
                candidates = density_matches

            # Prefer a new spatial stratum while preserving the density rotation.
            unseen = [
                index
                for index in candidates
                if metadata["spatial"][index] not in used_spatial
            ]
            if unseen and missing_spatial > 0:
                candidates = unseen

            chosen = candidates[int(rng.integers(0, len(candidates)))]
            available.remove(chosen)
            selected.append(int(metadata["rows"][chosen]))
            used_spatial.add(int(metadata["spatial"][chosen]))

        return selected

    def _draw_slides(self, rng):
        replace = len(self.slides) < self.spb
        chosen = rng.choice(self.slides, size=self.spb, replace=replace)
        return chosen.tolist()

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        if self.batches_per_epoch is not None:
            for _ in range(len(self)):
                batch = []
                for slide_id in self._draw_slides(rng):
                    batch.extend(self._sample_group(slide_id, rng))
                yield batch
            return

        group_slides = [
            slide_id
            for slide_id in self.slides
            for _ in range(self._groups_per_slide[slide_id])
        ]
        rng.shuffle(group_slides)
        required = len(self) * self.spb
        if len(group_slides) < required:
            group_slides.extend(self._draw_slides(rng)[: required - len(group_slides)])

        for start in range(0, required, self.spb):
            batch = []
            for slide_id in group_slides[start : start + self.spb]:
                batch.extend(self._sample_group(slide_id, rng))
            yield batch


def collate_phase1(batch, scanners, ref_idx):
    """Stack raw Phase 1 tensors and attach same-slide group identifiers."""
    slide_groups = {}
    group_ids = []
    for item in batch:
        slide_groups.setdefault(item["slide_id"], len(slide_groups))
        group_ids.append(slide_groups[item["slide_id"]])

    result = {
        "rgb": torch.stack([item["rgb"] for item in batch]),
        "present": torch.stack([item["present"] for item in batch]),
        "geom_ok": torch.stack([item["geom_ok"] for item in batch]),
        "q_reg": torch.stack([item["q_reg"] for item in batch]),
        "slide_group": torch.tensor(group_ids, dtype=torch.long),
        "scanners": list(scanners),
        "ref_idx": int(ref_idx),
        "slide_id": [item["slide_id"] for item in batch],
        "tile_id": torch.tensor([item["tile_id"] for item in batch], dtype=torch.long),
        "x": torch.tensor([item["x"] for item in batch], dtype=torch.long),
        "y": torch.tensor([item["y"] for item in batch], dtype=torch.long),
        "tissue_density": torch.tensor(
            [item["tissue_density"] for item in batch], dtype=torch.float32
        ),
    }
    if "nuclei_labels" in batch[0]:
        result["nuclei_labels"] = torch.stack([item["nuclei_labels"] for item in batch])
    return result
