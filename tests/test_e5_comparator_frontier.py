import numpy as np
import pandas as pd

from analyze_e5_comparator_frontier import CONDITIONS, summarize
from e4_primary_metrics import COLLAPSE_METRICS
from e5_comparator_population import SCANNERS


def synthetic_frames():
    invariance = []
    content = []
    collapse = []
    for slide in range(109):
        slide_id = f"slide-{slide:03d}"
        for condition_index, condition in enumerate(CONDITIONS):
            radius = 1.0 if condition == "raw" else 0.9 - 0.01 * condition_index
            invariance.append(
                {
                    "encoder_id": "resnet50",
                    "slide_id": slide_id,
                    "condition": condition,
                    "scanner_centroid_rms": radius,
                }
            )
            for scanner in SCANNERS[1:]:
                content.append(
                    {
                        "encoder_id": "resnet50",
                        "slide_id": slide_id,
                        "condition": condition,
                        "scanner": scanner,
                        "content_margin": 0.2,
                    }
                )
                for metric in COLLAPSE_METRICS:
                    collapse.append(
                        {
                            "encoder_id": "resnet50",
                            "slide_id": slide_id,
                            "condition": condition,
                            "scanner": scanner,
                            "metric": metric,
                            "ratio": 1.0,
                        }
                    )
    return pd.DataFrame(invariance), pd.DataFrame(content), pd.DataFrame(collapse)


def test_e5_summary_applies_frozen_safe_and_improved_rules():
    endpoints, content_scanner, collapse, common, decisions = summarize(*synthetic_frames())
    assert len(endpoints) == len(CONDITIONS)
    assert len(content_scanner) == len(CONDITIONS) * 5
    assert len(collapse) == len(CONDITIONS) * 5 * 3
    corrected = endpoints[endpoints["condition"] != "raw"]
    assert corrected["safe_for_pfm"].all()
    assert corrected["invariance_improved"].all()
    assert np.allclose(corrected["fidelity_constrained_rr"], corrected["relative_radius_reduction"])
    assert len(common) == len(CONDITIONS)
    assert decisions.iloc[0]["fidelity_constrained_best"] == CONDITIONS[-1]

