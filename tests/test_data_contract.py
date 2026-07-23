from types import SimpleNamespace

import numpy as np
import pytest

from prenorm.data.dataset import PairedTupleDataset
from prenorm.data.sampler import SpatialTupleBatchSampler, collate_phase1
from utils.index import build_index, write_index
from utils.store import write_slide


def build_v3_fixture(tmp_path):
    scanners = ["at2", "other"]
    records = []
    sidecars = []
    for index in range(8):
        rgb = np.full((16, 16, 3), 40 + index, np.uint8)
        records.append({
            "coords": (index % 4 * 256, index // 4 * 256),
            "tile_id": index,
            "present": [1, 1],
            "geom_ok": [1, 1],
            "q_reg": [1.0, 0.5],
            "at2": {"rgb": rgb},
            "other": {"rgb": rgb + 1},
        })
        sidecars.append({
            "slide_id": "slide",
            "tuple_id": index,
            "tile_id": index,
            "x": index % 4 * 256,
            "y": index // 4 * 256,
            "tissue_density": index / 8,
            "present": [True, True],
            "geom_ok": [True, True],
            "q_reg": [1.0, 0.5],
        })
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    write_slide(store_dir / "slide.h5", "slide", scanners, records, {})
    frame = build_index(sidecars, scanners)
    frame["split"] = "train"
    index_path = tmp_path / "index.parquet"
    write_index(frame, index_path)
    return scanners, store_dir, index_path


def test_dataset_and_collate_return_only_v3_contract(tmp_path):
    scanners, store_dir, index_path = build_v3_fixture(tmp_path)
    cfg = SimpleNamespace(loader=SimpleNamespace(handle_cache=1, aug=[]))
    dataset = PairedTupleDataset(index_path, store_dir, scanners, cfg, "train", False)
    item = dataset[0]
    assert set(item) == {
        "rgb", "present", "geom_ok", "q_reg", "slide_id", "tile_id",
        "x", "y", "tissue_density",
    }
    assert item["rgb"].shape == (2, 3, 16, 16)
    batch = collate_phase1([dataset[0], dataset[1]], scanners, 0)
    assert batch["rgb"].shape == (2, 2, 3, 16, 16)
    assert batch["slide_group"].tolist() == [0, 0]


def test_legacy_valid_is_rejected():
    with pytest.raises(ValueError, match="legacy 'valid'"):
        build_index([{
            "slide_id": "s", "tuple_id": 0, "tile_id": 0, "x": 0, "y": 0,
            "tissue_density": 0.5, "valid": [True], "present": [True],
            "geom_ok": [True], "q_reg": [1.0],
        }], ["at2"])


def test_spatial_sampler_is_dispersed_and_epoch_reproducible():
    rows = []
    for y in range(4):
        for x in range(4):
            rows.append({"x": x, "y": y, "tissue_density": (x + y) / 8})
    dataset = SimpleNamespace(slide_to_rows={"slide": list(range(16))}, rows=rows)
    sampler = SpatialTupleBatchSampler(
        dataset, slides_per_batch=1, patches_per_slide=8, spatial_bins=2, seed=3,
        batches_per_epoch=1,
    )
    first = next(iter(sampler))
    sampler.set_epoch(0)
    second = next(iter(sampler))
    assert first == second
    coordinates = {(rows[index]["x"], rows[index]["y"]) for index in first}
    assert len(coordinates) == 8
