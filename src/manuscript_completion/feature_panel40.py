"""Read the existing 40-position raw UNI panel for feature-map fitting."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from manuscript_completion.pfm_problem import TARGET_SCANNERS


ROOT = Path("outputs/scanner_batch_effect_analysis_2026-09-17")


def feature_dir(scanner: str) -> Path:
    if scanner == "gt450":
        return ROOT / (
            "06_learned_baselines/09_bidirectional_full_training/"
            "04_uni/at2_to_gt450/feature_shards"
        )
    return ROOT / (
        "12_manuscript_completion/02_baseline_benchmark/03_pix2pix/"
        f"04_uni/at2_to_{scanner}/feature_shards"
    )


def load_panel40(cohort: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    slides = cohort.slide_id.astype(str).tolist()
    panel = np.empty((len(slides), 40, 6, 1024), dtype=np.float32)
    positions = np.empty((len(slides), 40), dtype=np.int64)
    for scanner_index, scanner in enumerate(TARGET_SCANNERS, start=1):
        root = feature_dir(scanner)
        if {p.stem for p in root.glob("*.h5")} != set(slides):
            raise ValueError(f"{scanner}: feature-shard slide population differs")
        for slide_offset, slide_id in enumerate(slides):
            with h5py.File(root / f"{slide_id}.h5", "r") as store:
                if str(store.attrs["slide_id"]) != slide_id:
                    raise ValueError(f"{scanner}/{slide_id}: H5 identity mismatch")
                if str(store.attrs["source_scanner"]) != "at2" or str(store.attrs["target_scanner"]) != scanner:
                    raise ValueError(f"{scanner}/{slide_id}: scanner identity mismatch")
                location = np.asarray(store["location_index"], dtype=np.int64)
                source = np.asarray(store["raw_source"], dtype=np.float32)
                target = np.asarray(store["real_target"], dtype=np.float32)
            if location.shape != (40,) or source.shape != (40, 1024) or target.shape != (40, 1024):
                raise ValueError(f"{scanner}/{slide_id}: unexpected feature shape")
            if scanner_index == 1:
                positions[slide_offset] = location
                panel[slide_offset, :, 0] = source
            else:
                if not np.array_equal(positions[slide_offset], location):
                    raise ValueError(f"{scanner}/{slide_id}: locations differ")
                if not np.allclose(panel[slide_offset, :, 0], source, atol=1e-6):
                    raise ValueError(f"{scanner}/{slide_id}: AT2 features differ")
            panel[slide_offset, :, scanner_index] = target
    if not np.isfinite(panel).all():
        raise ValueError("non-finite 40-position raw features")
    return panel, positions
