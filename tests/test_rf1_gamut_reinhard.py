import torch

from e5_comparator_population import rgb01_to_lab
from rf1_gamut_reinhard import reinhard_lab_source_ray, source_ray_rgb_projection


def test_source_ray_keeps_valid_proposals_unchanged():
    source = torch.tensor([[[[0.2, 0.4, 0.6], [0.1, 0.2, 0.3]]]])
    proposed = torch.tensor([[[[0.3, 0.2, 0.8], [0.0, 1.0, 0.4]]]])
    report = source_ray_rgb_projection(source, proposed)
    assert torch.allclose(report["output"], proposed)
    assert torch.all(report["scale"] == 1.0)
    assert torch.all(report["final_clamp_mae"] == 0.0)


def test_source_ray_uses_one_channel_scale_and_hits_gamut_boundary():
    source = torch.tensor([[[[0.2, 0.4, 0.6]]]], dtype=torch.float64)
    proposed = torch.tensor([[[[1.8, -0.4, 0.8]]]], dtype=torch.float64)
    report = source_ray_rgb_projection(source, proposed)
    output = report["output"]
    scale = report["scale"].item()
    assert abs(scale - 0.5) < 1e-12
    assert torch.allclose(output, source + scale * (proposed - source))
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0)
    assert torch.all(report["final_clamp_mae"] == 0.0)


def test_source_ray_reinhard_identity_is_an_identity():
    generator = torch.Generator().manual_seed(20260803)
    rgb = torch.rand((2, 17, 19, 3), generator=generator) * 0.9 + 0.05
    lab = rgb01_to_lab(rgb)
    mean = lab.reshape(-1, 3).mean(0)
    std = lab.reshape(-1, 3).std(0, unbiased=False)
    report = reinhard_lab_source_ray(rgb, mean, std, mean, std)
    assert torch.max(torch.abs(report["output"] - rgb)) < 3e-5
    assert torch.all(report["final_clamp_mae"] == 0.0)
