import torch

from prenorm.exp06.frequency import FixedLaplacianPyramid
from prenorm.exp06.model import LowFrequencyHarmonizer


def test_exp06_pyramid_roundtrip_and_copied_coefficients():
    torch.manual_seed(31)
    value = torch.randn(2, 3, 64, 64)
    pyramid = FixedLaplacianPyramid(4)
    low, bands = pyramid.decompose(value)
    reconstructed = pyramid.reconstruct(low, bands)
    torch.testing.assert_close(reconstructed, value, atol=1e-6, rtol=0)
    changed = pyramid.reconstruct(low + 0.1, bands)
    assert not torch.allclose(changed, value)
    for copied, original in zip(bands, bands):
        assert torch.equal(copied, original)


def test_exp06_model_at2_bypass_is_bitwise_and_backward_is_finite():
    torch.manual_seed(37)
    model = LowFrequencyHarmonizer(levels=3, width=8, residual_scale=0.1)
    source = torch.rand(2, 3, 64, 64) * 2 - 1
    context = torch.rand(2, 4, 3, 64, 64) * 2 - 1
    mask = torch.ones(2, 4, dtype=torch.bool)
    bypass = model(source, context, mask, is_reference=torch.ones(2, dtype=torch.bool))
    assert torch.equal(bypass["output"], source)
    assert torch.count_nonzero(bypass["residual"]) == 0
    output = model(source, context, mask)
    loss = output["corrected_low"].square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_exp06_initialization_is_identity_before_clipping():
    model = LowFrequencyHarmonizer(levels=4, width=8, residual_scale=0.1).eval()
    source = torch.rand(1, 3, 64, 64) * 1.8 - 0.9
    context = torch.rand(1, 3, 3, 64, 64) * 2 - 1
    mask = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        output = model(source, context, mask, clip=False)
    torch.testing.assert_close(output["preclip"], source, atol=1e-6, rtol=0)
    assert torch.count_nonzero(output["residual"]) == 0
