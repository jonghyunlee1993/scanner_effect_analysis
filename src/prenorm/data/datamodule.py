"""Lightning data module for Phase 1 registered RGB tuples."""

import functools

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import PairedTupleDataset
from .sampler import SpatialTupleBatchSampler, collate_phase1


class PrenormDataModule(pl.LightningDataModule):
    """Build split datasets and spatially stratified Phase 1 loaders."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        holdout = getattr(cfg.experiment, "holdout_scanner", None)
        self.scanners = [scanner for scanner in cfg.scanners if scanner != holdout]
        self.ref_idx = self.scanners.index(cfg.reference_scanner)
        self.train_ds = self.val_ds = self.test_ds = None

    def _dataset(self, split, train):
        return PairedTupleDataset(
            self.cfg.paths.output_index,
            self.cfg.paths.output_store,
            self.scanners,
            self.cfg,
            split,
            train,
        )

    def setup(self, stage=None):
        self.train_ds = self._dataset("train", train=True)
        self.val_ds = self._dataset("val", train=False)
        self.test_ds = self._dataset("test", train=False)

    def _loader(self, dataset, batches_per_epoch=None):
        sampler = SpatialTupleBatchSampler(
            dataset,
            self.cfg.loader.slides_per_batch,
            self.cfg.loader.patches_per_slide,
            spatial_bins=getattr(self.cfg.loader, "spatial_bins", 4),
            seed=self.cfg.runtime.seed,
            batches_per_epoch=batches_per_epoch,
        )
        collate_fn = functools.partial(
            collate_phase1,
            scanners=self.scanners,
            ref_idx=self.ref_idx,
        )
        workers = int(self.cfg.loader.num_workers)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=workers,
            persistent_workers=workers > 0,
        )

    def train_dataloader(self):
        return self._loader(
            self.train_ds,
            getattr(self.cfg.loader, "batches_per_epoch", None),
        )

    def val_dataloader(self):
        return self._loader(self.val_ds)

    def test_dataloader(self):
        return self._loader(self.test_ds)
