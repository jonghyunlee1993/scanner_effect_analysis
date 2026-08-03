import h5py
import numpy as np
import pandas as pd

from analyze_e5_rf1_tissue_probe import paired_summary
from analyze_e7_tissue_probe import BOOTSTRAP_SEED, bootstrap_weights
from build_e5_rf1_tissue_probe_rows import load_queries
from e5_reinhard_residual_frequency import RF1_CONDITION


def test_rf1_tissue_query_loader_preserves_raw_and_hybrid_identity(tmp_path):
    raw_path = tmp_path / "raw.h5"
    feature_path = tmp_path / "rf1.h5"
    raw = np.arange(6 * 100 * 4, dtype=np.float32).reshape(6, 100, 4) + 1
    locations = np.asarray([f"location_{index}".encode() for index in range(100)])
    with h5py.File(raw_path, "w") as target:
        target.create_dataset("features", data=raw)
        target.create_dataset("location_id", data=locations)
    hybrid = raw[None].copy()
    hybrid[:, 1:] += 2
    with h5py.File(feature_path, "w") as target:
        target.create_dataset("features", data=hybrid)
        target.create_dataset("location_id", data=locations)
        target.create_dataset("condition", data=np.asarray([RF1_CONDITION.encode()]))
    observed = load_queries(raw_path, feature_path, 4)
    assert observed.shape == (2, 6, 100, 4)
    assert np.array_equal(observed[0], raw)
    assert np.array_equal(observed[1], hybrid[0])


def test_rf1_tissue_paired_summary_uses_slide_paired_delta():
    base = pd.DataFrame(
        {
            "slide_id": ["a", "b", "c", "d"],
            "tissue_type": ["x", "x", "y", "y"],
            "slides_in_tissue": [2, 2, 2, 2],
        }
    )
    raw = base.assign(metric=[0.1, 0.2, 0.3, 0.4])
    corrected = base.assign(metric=[0.15, 0.25, 0.35, 0.45])
    plan = bootstrap_weights(base, 2, BOOTSTRAP_SEED)
    values = paired_summary(corrected, raw, "metric", plan)
    assert np.isclose(values[0], 0.30)
    assert np.isclose(values[3], 0.25)
    assert np.isclose(values[4], 0.05)
    assert np.isclose(values[5], 0.05)
    assert np.isclose(values[6], 0.05)
