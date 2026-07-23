import torch

from prenorm.module import Phase1Module
from tests.helpers import make_config


def synthetic_batch():
    return {
        "rgb": torch.rand(4, 2, 3, 32, 32) * 2 - 1,
        "present": torch.ones(4, 2, dtype=torch.bool),
        "geom_ok": torch.ones(4, 2, dtype=torch.bool),
        "q_reg": torch.ones(4, 2),
        "slide_group": torch.zeros(4, dtype=torch.long),
    }


def test_phase1_validation_forward_backward(tmp_path):
    module = Phase1Module(make_config(tmp_path))
    output = module._shared_step(synthetic_batch(), train=False, batch_idx=0)
    output["loss"].backward()
    assert output["standard"] >= 0
    assert output["translate_self"] >= 0
    assert output["translate_cross"] >= 0
    assert output["coverage_standard"] == 1
    assert module.B.grad is not None


def test_phase1_frequency_asymmetric_standard_reports_both_bands(tmp_path):
    cfg = make_config(tmp_path)
    cfg.loss.standard_mode = "frequency_asymmetric"
    cfg.loss.standard_lowpass_kernel = 9
    cfg.loss.standard_lowpass_sigma = 2.0
    cfg.loss.standard_high_frequency_weight = 0.25
    cfg.loss.standard_coherence_low = 0.1
    cfg.loss.standard_coherence_high = 0.3
    cfg.loss.standard_detail_energy_low = 0.5
    cfg.loss.standard_detail_energy_high = 0.8
    cfg.loss.standard_detail_q_low = 0.8
    cfg.loss.standard_detail_q_high = 0.95
    module = Phase1Module(cfg)
    output = module._shared_step(synthetic_batch(), train=False, batch_idx=0)
    output["loss"].backward()

    assert output["standard"] > 0
    assert output["standard_low"] > 0
    assert output["standard_high"] > 0
    assert output["coverage_standard"] == 1
    assert output["coverage_standard_high"] > 0
    assert module.model.G.out[1].weight.grad is not None


def test_prototype_regression_target_is_detached(tmp_path):
    module = Phase1Module(make_config(tmp_path))
    rgb = synthetic_batch()["rgb"]
    present = torch.ones(4, 2, dtype=torch.bool)
    style, _ = module._style_regression(rgb, present, torch.zeros(4, dtype=torch.long))
    style.backward()
    assert module.B.grad is None
    assert any(parameter.grad is not None for parameter in module.model.E_set.parameters())


def test_loso_does_not_create_holdout_prototype(tmp_path):
    module = Phase1Module(make_config(tmp_path, holdout="other"))
    assert module.model.scanners == ("at2",)
    assert module.B.shape[0] == 1


def test_versa_pair_weight_is_isolated_and_logged(tmp_path):
    module = Phase1Module(
        make_config(
            tmp_path,
            pair_scanner_weights={"versa": 0.2},
            scanners=["at2", "versa"],
        )
    )
    output = module._shared_step(synthetic_batch(), train=False, batch_idx=0)

    assert output["pair_versa_weight_mass_fraction"] == 1.0
    assert output["pair_versa"] >= 0
    assert output["pair_nonversa"] == 0


def test_phase1_nuclei_head_receives_stardist_supervision(tmp_path):
    cfg = make_config(tmp_path)
    cfg.model.nuclei_head = True
    cfg.loss.nuclei_reconstruction = 0.25
    cfg.loss.nuclei_boundary = 0.5
    cfg.loss.nuclei_reference_high_q = 0.9
    batch = synthetic_batch()
    batch["nuclei_labels"] = torch.zeros(4, 2, 1, 32, 32, dtype=torch.int32)
    batch["nuclei_labels"][:, :, :, 8:24, 8:24] = 1
    module = Phase1Module(cfg)
    output = module._shared_step(batch, train=False, batch_idx=0)
    output["loss"].backward()
    assert output["nuclei_reconstruction"] > 0
    assert output["coverage_nuclei"] > 0
    assert any(parameter.grad is not None for parameter in module.model.G.nuclei_head.parameters())


def test_phase1_nuclei_rgb_detail_supervises_rgb_without_auxiliary_head(tmp_path):
    cfg = make_config(tmp_path)
    cfg.loss.nuclei_rgb_detail = 0.25
    cfg.loss.nuclei_rgb_dilation = 3
    cfg.loss.nuclei_rgb_laplacian = 0.25
    batch = synthetic_batch()
    batch["nuclei_labels"] = torch.zeros(4, 2, 1, 32, 32, dtype=torch.int32)
    batch["nuclei_labels"][:, :, :, 8:24, 8:24] = 1
    module = Phase1Module(cfg)
    output = module._shared_step(batch, train=False, batch_idx=0)
    output["loss"].backward()
    assert module.model.G.nuclei_head is None
    assert output["nuclei_rgb_detail"] > 0
    assert output["coverage_nuclei_rgb_detail"] > 0
    assert module.model.G.out[1].weight.grad is not None


def test_phase1_nuclei_orientation_replaces_rgb_detail(tmp_path):
    cfg = make_config(tmp_path)
    cfg.loss.nuclei_rgb_detail = 0.0
    cfg.loss.nuclei_orientation = 0.25
    cfg.loss.nuclei_orientation_dilation = 3
    cfg.loss.nuclei_orientation_smoothing_kernel = 7
    cfg.loss.nuclei_orientation_smoothing_sigma = 1.25
    cfg.loss.nuclei_orientation_magnitude_floor = 0.1
    batch = synthetic_batch()
    batch["nuclei_labels"] = torch.zeros(4, 2, 1, 32, 32, dtype=torch.int32)
    batch["nuclei_labels"][:, :, :, 8:24, 8:24] = 1
    module = Phase1Module(cfg)
    output = module._shared_step(batch, train=False, batch_idx=0)
    output["loss"].backward()

    assert module.model.G.nuclei_head is None
    assert output["nuclei_rgb_detail"] == 0
    assert output["nuclei_orientation"] > 0
    assert output["coverage_nuclei_orientation"] > 0
    assert module.model.G.out[1].weight.grad is not None


def test_phase1_neighborhood_consistency_supervises_content(tmp_path):
    cfg = make_config(tmp_path)
    cfg.loss.neighborhood_consistency = 0.05
    cfg.loss.neighborhood_temperature = 0.1
    cfg.loss.neighborhood_pool_size = 2
    cfg.loss.neighborhood_min_q = 0.8
    module = Phase1Module(cfg)
    output = module._shared_step(synthetic_batch(), train=False, batch_idx=0)
    output["loss"].backward()

    assert output["neighborhood_consistency"] > 0
    assert output["coverage_neighborhood_consistency"] == 1
    assert output["neighborhood_ramp"] == 1
    assert any(parameter.grad is not None for parameter in module.C.parameters())
