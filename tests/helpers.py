from types import SimpleNamespace

import numpy as np


def make_stain_artifact(tmp_path):
    path = tmp_path / "stain.npz"
    np.savez(
        path,
        matrix=np.asarray(
            [[0.650, 0.072], [0.704, 0.990], [0.286, 0.105]], dtype=np.float32
        ),
        concentration_scale=np.asarray([2.0, 2.0], dtype=np.float32),
    )
    return path


def make_config(tmp_path, holdout=None, pair_scanner_weights=None, scanners=None):
    stain = make_stain_artifact(tmp_path)
    scanners = scanners or ["at2", "other"]
    return SimpleNamespace(
        design_version=3,
        scanners=scanners,
        reference_scanner="at2",
        paths=SimpleNamespace(stain_reference=str(stain)),
        input=SimpleNamespace(mode="full", group_dropout_p=0.3),
        model=SimpleNamespace(
            content_ch=64, style_dim=8, ngf=16, n_downsample=2, n_res=1
        ),
        augmentation=SimpleNamespace(
            brightness=0.12, contrast=0.25, gamma=0.25, saturation=0.3,
            illumination=0.1, blur_probability=0.5, blur_sigma_max=1.5,
            unsharp_probability=0.2, noise=0.02, jpeg_quality=[60, 90],
            resample_min_scale=0.75,
        ),
        registration=SimpleNamespace(q_min=0.25),
        loss=SimpleNamespace(
            standard=5.0, translate=10.0, pair_start=1.0, pair_end=10.0,
            variance=0.05, style=0.1, variance_gamma=1.0, gradient_end=0.5,
            pair_scanner_weights=pair_scanner_weights or {},
        ),
        schedule=SimpleNamespace(
            bootstrap_end=0.1, identification_end=0.4, hardening_end=0.8
        ),
        experiment=SimpleNamespace(holdout_scanner=holdout),
        runtime=SimpleNamespace(seed=7),
        image_log=SimpleNamespace(n_ref_tuples=2),
        optim=SimpleNamespace(lr=1e-4, betas=[0.5, 0.999], weight_decay=0.0),
    )
