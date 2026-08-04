import numpy as np
import pandas as pd

from analyze_e7_tissue_probe import bootstrap_weights, summarize_vector
from build_e7_tissue_probe_rows import heldout_centroids, tissue_reference_sums


def test_heldout_tissue_centroid_excludes_complete_slide():
    raw_at2 = {
        "a1": np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "a2": np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        "b1": np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        "b2": np.asarray([[1.0, -1.0], [1.0, -1.0]], dtype=np.float32),
    }
    tissue_by_slide = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    tissues = ["A", "B"]
    sums = tissue_reference_sums(raw_at2, tissue_by_slide, tissues)
    centroids, target = heldout_centroids(
        "a1", raw_at2, tissue_by_slide, tissues, sums
    )
    np.testing.assert_allclose(centroids[target], [0.0, 1.0])
    np.testing.assert_allclose(centroids[1], [1.0, 0.0])


def test_hierarchical_bootstrap_point_weights_tissues_equally():
    slides = pd.DataFrame({
        "slide_id": ["a1", "a2", "b1", "b2", "b3", "b4"],
        "tissue_type": ["A", "A", "B", "B", "B", "B"],
        "slides_in_tissue": [2, 2, 4, 4, 4, 4],
    })
    plan = bootstrap_weights(slides, minimum_size=2, seed=11)
    assert np.allclose(plan[3].sum(axis=1), 1.0)
    values = slides[["slide_id", "tissue_type"]].copy()
    values["metric"] = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    point, lower, upper, _ = summarize_vector(values, "metric", plan)
    assert point == 0.5
    assert lower <= point <= upper
