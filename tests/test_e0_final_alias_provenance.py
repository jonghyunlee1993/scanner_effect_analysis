import numpy as np
import pandas as pd

from export_e0_final_alias_provenance import TARGET_MPP, export_provenance


def test_export_uses_final_matrix_and_frozen_mpp_expectation():
    rows = []
    for scanner, scale in (("gt450", 0.52), ("versa", 0.54)):
        for index in range(109):
            rows.append(
                {
                    "slide_id": f"s{index}",
                    "scanner": scanner,
                    "cell_pass": True,
                    "native_to_target_m00": scale,
                    "native_to_target_m01": 0.0,
                    "native_to_target_m10": 0.0,
                    "native_to_target_m11": scale,
                    "expected_native_px_per_target_px": 1.0 / scale,
                    "explicit_aa_pre_scale": scale,
                    "geometry_route_label": "primary",
                    "native_path": f"/{scanner}/{index}.svs",
                    "native_width": 1000,
                    "native_height": 900,
                    "alignment_version": "v1",
                }
            )
    result = export_provenance(pd.DataFrame(rows))
    assert len(result) == 218
    assert np.allclose(
        result["native_mpp"],
        TARGET_MPP / result["nominal_native_px_per_target_px"],
    )
    assert np.allclose(
        result["output_px_per_native_px_min"], result["explicit_aa_pre_scale"]
    )
    assert np.allclose(result["affine_rotation_deg"], 0.0)
