from types import SimpleNamespace

import numpy as np
import torch

from render_eval_report import (
    _mixed_gallery_metrics,
    mixed_retrieval_table,
    objective_formula,
)
from eval_report import scanner_purity_excluding_location


def test_mixed_gallery_metrics_distinguishes_location_from_scanner():
    # The registered counterpart is the nearest neighbor; same-scanner images
    # remain in the gallery and therefore still contribute to scanner purity.
    locations = np.repeat(np.arange(3), 2)
    scanners = np.tile(np.arange(2), 3)
    embeddings = np.eye(3, dtype=np.float32)[locations]
    embeddings += 0.01 * np.eye(2, 3, dtype=np.float32)[scanners]
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    result = _mixed_gallery_metrics(embeddings, locations, scanners, k=5)

    assert result["same_location_top1"] == 1.0
    assert result["counterpart_recall_at5"] == 1.0
    assert result["same_scanner_top1"] == 0.0
    assert np.isclose(result["same_scanner_purity_at5"], 0.4)


def test_scanner_purity_can_exclude_registered_counterparts():
    locations = np.repeat(np.arange(3), 2)
    scanners = np.tile(np.arange(2), 3)
    embeddings = np.concatenate((
        3.0 * np.eye(2, dtype=np.float32)[scanners],
        0.1 * np.eye(3, dtype=np.float32)[locations],
    ), axis=1)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    value = scanner_purity_excluding_location(
        torch.from_numpy(embeddings), locations, scanners, k=2
    )

    assert value == 1.0


def test_objective_formula_only_lists_active_resolved_terms():
    loss = SimpleNamespace(
        standard=1.0,
        translate=1.0,
        pair_start=0.0,
        pair_end=0.5,
        detail=0.0,
        at2_identity_detail=0.25,
        paired_target_detail=0.1,
        nuclei_rgb_detail=0.25,
        neighborhood_consistency=0.05,
        nuclei_reconstruction=0.0,
        variance=0.05,
        style=0.1,
        gradient_end=0.2,
        pair_scanner_weights={"gt450": 0.75, "versa": 1.0},
    )

    formula = str(objective_formula(SimpleNamespace(loss=loss)))

    assert "L<sub>AT2-id</sub>" in formula
    assert "L<sub>paired-detail</sub>" in formula
    assert "L<sub>nuc-RGB</sub>" in formula
    assert "L<sub>neighbor</sub>" in formula
    assert "L<sub>nuc-recon</sub>" not in formula
    assert "GT450=0.75" in formula
    assert "VERSA=1" in formula


def test_mixed_retrieval_table_accepts_id_only_reports():
    row = {
        "cohort": "ID 4-scanner",
        "representation": "raw",
        "n_queries": 12,
        "same_location_top1": 0.5,
        "same_location_hit_at5": 0.8,
        "counterpart_recall_at5": 0.6,
        "all_counterparts_at5": 0.4,
        "same_scanner_top1": 0.5,
        "same_scanner_purity_at5": 0.3,
        "all5_same_scanner": 0.1,
    }

    rendered = str(mixed_retrieval_table([row]))

    assert "ID 4-scanner" in rendered
    assert "OOD" not in rendered
