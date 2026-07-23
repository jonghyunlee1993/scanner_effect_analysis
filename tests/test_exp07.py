import torch

from prenorm.exp06.frequency import FixedLaplacianPyramid
from prenorm.exp07.projection import (
    scalar_range_alpha,
    project_correction,
    low_correction_delta,
    local_range_project,
)


def _random_case(seed, n=4, c=3, size=64):
    torch.manual_seed(seed)
    pyramid = FixedLaplacianPyramid(4)
    source = torch.rand(n, c, size, size) * 2 - 1            # x in [-1, 1]
    low, bands = pyramid.decompose(source)
    corrected_low = low + torch.randn_like(low) * 0.3        # arbitrary l'
    return pyramid, source, low, bands, corrected_low


def test_exp07_projection_keeps_output_in_range():
    pyramid, source, _, _, corrected_low = _random_case(1)
    d, _, _ = low_correction_delta(pyramid, source, corrected_low)
    output, alpha, _, _ = project_correction(source, d)
    assert output.max() <= 1.0 + 1e-6
    assert output.min() >= -1.0 - 1e-6
    assert (alpha >= 0).all() and (alpha <= 1).all()


def test_exp07_projection_preserves_copied_detail_bandwise():
    pyramid, source, _, bands, corrected_low = _random_case(2)
    d, low, _ = low_correction_delta(pyramid, source, corrected_low)
    output, alpha, _, _ = project_correction(source, d)
    # Output detail (output minus its own low-only reconstruction) must equal the
    # copied input detail: the projection never recomputes the high bands.
    corrected_low_safe = low + alpha.reshape(-1, 1, 1, 1) * (corrected_low - low)
    zeros = [torch.zeros_like(b) for b in bands]
    low_only = pyramid.reconstruct(corrected_low_safe, zeros)
    input_detail = pyramid.reconstruct(torch.zeros_like(low), bands)
    torch.testing.assert_close(output - low_only, input_detail, atol=1e-6, rtol=0)


def test_exp07_alpha_is_maximal_and_linear():
    pyramid, source, low, bands, corrected_low = _random_case(3)
    d, _, _ = low_correction_delta(pyramid, source, corrected_low)
    output, alpha, alpha_raw, _ = project_correction(source, d)
    # Linearity: x + alpha*d equals reconstruct(l + alpha*(l'-l), b).
    a = alpha.reshape(-1, 1, 1, 1)
    relit = pyramid.reconstruct(low + a * (corrected_low - low), bands)
    torch.testing.assert_close(output, relit, atol=1e-6, rtol=0)
    # Maximality: a range-binding image (alpha_raw < 1) sits on a boundary; a
    # non-binding image (alpha_raw >= 1) applies the full correction.
    gap = torch.minimum((1.0 - output).amin(dim=(1, 2, 3)),
                        (output + 1.0).amin(dim=(1, 2, 3)))
    binding = alpha_raw < 1.0
    if binding.any():
        assert gap[binding].abs().max() < 1e-5
    if (~binding).any():
        torch.testing.assert_close(
            output[~binding], (source + d)[~binding], atol=1e-6, rtol=0
        )


def test_exp07_identity_input_needs_no_projection():
    # If x is already in range and d is zero, alpha is unconstrained -> full (1).
    torch.manual_seed(4)
    x = torch.rand(3, 3, 32, 32) * 2 - 1
    d = torch.zeros_like(x)
    _, alpha, alpha_raw, _ = project_correction(x, d)
    assert torch.isinf(alpha_raw).all()
    assert torch.equal(alpha, torch.ones_like(alpha))


def test_exp07_local_projection_is_range_safe_and_detail_exact():
    pyramid, source, low, bands, corrected_low = _random_case(5)
    res = local_range_project(pyramid, source, corrected_low, steps=40)
    output = res["output"]
    assert output.max() <= 1.0 + 1e-6
    assert output.min() >= -1.0 - 1e-6
    # High-band identity: output detail equals the copied input detail exactly,
    # for the local coarse solution too (bands are never recomputed).
    zeros = [torch.zeros_like(b) for b in bands]
    low_only = pyramid.reconstruct(res["coarse"], zeros)
    input_detail = pyramid.reconstruct(torch.zeros_like(low), bands)
    torch.testing.assert_close(output - low_only, input_detail, atol=1e-6, rtol=0)
    # The returned coarse reconstructs to the reported output.
    relit = pyramid.reconstruct(res["coarse"], bands)
    torch.testing.assert_close(output, relit, atol=1e-6, rtol=0)


def test_exp07_local_projection_retains_at_least_scalar_efficacy():
    # Construct a case the scalar projection handles poorly: a uniform low-band
    # shift, but one already-saturated detail pixel that chokes the global alpha.
    torch.manual_seed(6)
    pyramid = FixedLaplacianPyramid(4)
    source = (torch.rand(4, 3, 64, 64) * 1.6 - 0.8)
    source[:, :, 0, 0] = 0.99                      # near-saturated copied detail
    low, bands = pyramid.decompose(source)
    target_low = low + 0.25                         # want a uniform brightening
    # scalar route
    d, _, _ = low_correction_delta(pyramid, source, target_low)
    _, alpha_s, _, _ = project_correction(source, d)
    scalar_coarse = low + alpha_s.reshape(-1, 1, 1, 1) * (target_low - low)
    scalar_mae = (scalar_coarse - target_low).abs().mean()
    # local route
    res = local_range_project(pyramid, source, target_low, steps=200)
    local_mae = (res["coarse"] - target_low).abs().mean()
    # Local projection must not do worse than the global scalar, and here should
    # do meaningfully better because only the saturated corner needs backing off.
    assert local_mae <= scalar_mae + 1e-4
