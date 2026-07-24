import torch

from prenorm.exp01.frequency import FixedLaplacianPyramid


def test_exp01_pyramid_roundtrip_and_copied_coefficients():
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
