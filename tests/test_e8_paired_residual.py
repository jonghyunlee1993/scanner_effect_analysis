"""Guarantees and conventions of the E8 paired residual baseline.

The arm A properties asserted here are the ones the contract's section 3.4
calls exact.  The statistics it calls approximate are measured in the audit, not
asserted here, because they are nonzero for the locked RF1U method too.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from analyze_rf1_multiscale import PYRAMID_SIGMAS, laplacian_pyramid, shared_od_multiscale
from build_e8_cache import E8_CROP, E8_MAX_SHIFT, E8_VALID, best_integer_shift
from e5_comparator_population import od_to_rgb01, rgb01_to_od
from e8_paired_residual import (
    E8_ENERGY_POOL,
    ConditionedUNet,
    aligned_window,
    free_od_apply,
    global_band_energy,
    local_band_energy,
    network_input,
    pooled_band_energy,
    shared_od_apply,
)


def tissue_like(count: int, size: int, seed: int = 0) -> torch.Tensor:
    """Smooth, bounded, mid-density RGB -- closer to stained tissue than noise."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.randn(count, 1, size // 8, size // 8, generator=generator)
    smooth = F.interpolate(coarse, size=(size, size), mode="bicubic", align_corners=False)
    fine = 0.15 * torch.randn(count, 1, size, size, generator=generator)
    field = (smooth + fine).squeeze(1)
    field = (field - field.mean()) / field.std()
    density = (0.9 + 0.35 * field).clamp(0.05, 2.5)
    return od_to_rgb01(density[..., None].repeat(1, 1, 1, 3) * torch.tensor([1.0, 1.25, 0.85]))


def test_constant_gain_reproduces_the_frozen_operator():
    """A constant gain field must equal the locked shared-OD multiscale operator."""
    rgb = tissue_like(3, 64)
    gains = np.array([1.3, 0.8, 1.1])
    frozen = shared_od_multiscale(rgb, gains, PYRAMID_SIGMAS)["output"]
    generalized = shared_od_apply(
        rgb, torch.as_tensor(gains, dtype=torch.float32)[:, None, None, None]
    )["output"]
    assert torch.allclose(frozen, generalized, atol=1e-6)


@pytest.mark.parametrize(
    "apply_correction",
    [
        lambda rgb: shared_od_apply(rgb, torch.ones(3, 1, 1, 1))["output"],
        lambda rgb: free_od_apply(rgb, torch.zeros_like(rgb))["output"],
    ],
)
def test_untrained_arms_are_the_exact_identity(apply_correction):
    """A zero-initialised head must return the RF1U base bit-exactly."""
    rgb = tissue_like(3, 64)
    assert torch.allclose(apply_correction(rgb), rgb, atol=1e-6)


def test_residual_is_exactly_the_band_combination_away_from_the_projection():
    """Contract 3.4 property 1: nothing outside the base's own bands can appear."""
    rgb = tissue_like(4, 128)
    density = rgb01_to_od(rgb)
    bands, _ = laplacian_pyramid(density.mean(-1), PYRAMID_SIGMAS)
    stacked = torch.stack(bands, dim=0)
    generator = torch.Generator().manual_seed(1)
    gains = torch.rand(3, 4, 128, 128, generator=generator) * 1.4 + 0.6

    output = shared_od_apply(rgb, gains)["output"]
    actual = rgb01_to_od(output).mean(-1) - density.mean(-1)
    intended = ((gains - 1.0) * stacked).sum(dim=0)
    unbound = (actual - intended).abs() <= 1e-5
    assert unbound.float().mean() > 0.9
    assert (actual - intended).abs()[unbound].max() < 1e-5


def test_positive_gain_never_moves_a_band_zero_crossing():
    """Contract 3.4 property 2, at the most aggressive admissible gain range."""
    rgb = tissue_like(4, 128)
    bands, _ = laplacian_pyramid(rgb01_to_od(rgb).mean(-1), PYRAMID_SIGMAS)
    stacked = torch.stack(bands, dim=0)
    generator = torch.Generator().manual_seed(2)
    log_limit = float(np.log(4.0))
    gains = torch.exp(
        torch.rand(3, 4, 128, 128, generator=generator) * 2 * log_limit - log_limit
    )
    assert int(((torch.sign(gains * stacked) * torch.sign(stacked)) < 0).sum()) == 0


def test_alignment_recovers_a_known_shift_and_matches_the_loss_window():
    """The cache convention and the training crop must be the same convention."""
    generator = np.random.default_rng(3)
    source = generator.random((4, E8_CROP, E8_CROP), dtype=np.float32)
    truth = np.array([[0, 0], [2, -1], [-3, 3], [1, 2]], dtype=np.int8)
    target = np.zeros_like(source)
    for index, (row, column) in enumerate(truth):
        target[
            index,
            E8_MAX_SHIFT : E8_MAX_SHIFT + E8_VALID,
            E8_MAX_SHIFT : E8_MAX_SHIFT + E8_VALID,
        ] = source[
            index,
            E8_MAX_SHIFT + row : E8_MAX_SHIFT + row + E8_VALID,
            E8_MAX_SHIFT + column : E8_MAX_SHIFT + column + E8_VALID,
        ]

    found, _ = best_integer_shift(source, target)
    assert (found == truth).all()

    window = aligned_window(
        torch.from_numpy(source), torch.from_numpy(truth.astype(np.int64))
    )
    anchor = aligned_window(
        torch.from_numpy(target), torch.zeros(4, 2, dtype=torch.long)
    )
    assert torch.allclose(window, anchor, atol=1e-6)


def test_flip_augmentation_negates_the_shift():
    """Reversing an axis maps a window start s to size - s - length."""
    size, length = E8_CROP, E8_VALID
    for shift in (-3, -1, 0, 2, 3):
        start = E8_MAX_SHIFT + shift
        assert size - start - length == E8_MAX_SHIFT + (-shift)


def test_network_is_zero_initialised_and_shaped():
    rgb = tissue_like(3, 64)
    network = ConditionedUNet(sources=5)
    features = network_input(rgb)
    assert features.shape == (3, 7, 64, 64)
    prediction = network(features, torch.tensor([0, 3, 4]))
    assert prediction.shape == (3, 3, 64, 64)
    assert prediction.abs().max().item() == 0.0


def test_band_energy_shapes():
    rgb = tissue_like(3, 64)
    bands, _ = laplacian_pyramid(rgb01_to_od(rgb).mean(-1), PYRAMID_SIGMAS)
    assert local_band_energy(torch.stack(bands, dim=0)).shape == (3, 3, 64, 64)
    assert global_band_energy(rgb01_to_od(rgb).mean(-1)).shape == (3, 3)


def test_pooled_band_energy_tracks_the_gaussian_window():
    """The cheap training statistic must agree with the smooth audit one."""
    rgb = tissue_like(4, E8_CROP)
    bands, _ = laplacian_pyramid(rgb01_to_od(rgb).mean(-1), PYRAMID_SIGMAS)
    stacked = torch.stack(bands, dim=0)
    shift = torch.zeros(4, 2, dtype=torch.long)

    pooled = pooled_band_energy(stacked, shift)
    assert pooled.shape == (3, 4, E8_VALID // E8_ENERGY_POOL, E8_VALID // E8_ENERGY_POOL)

    smooth = aligned_window(local_band_energy(stacked), shift)
    reference = F.avg_pool2d(
        smooth.reshape(-1, 1, E8_VALID, E8_VALID), E8_ENERGY_POOL
    ).reshape(pooled.shape)
    both = torch.stack([pooled.flatten(), reference.flatten()]).log()
    correlation = torch.corrcoef(both)[0, 1]
    assert correlation > 0.95


def test_aligned_window_channels_indexes_the_batch_not_the_channel():
    """Regression: permuting NHWC to NCHW puts the channel on dim -3.

    `aligned_window` takes the batch there, so the NCHW form silently indexed the
    wrong axis whenever the batch was not exactly three.  Both arm B's pixel term
    and the paired-MAE gate metric went through that path.
    """
    from e8_paired_residual import aligned_window_channels

    count = 5
    generator = torch.Generator().manual_seed(7)
    value = torch.rand(count, E8_CROP, E8_CROP, 3, generator=generator)
    shift = torch.tensor([[0, 0], [1, -2], [-3, 3], [2, 1], [0, -1]])

    window = aligned_window_channels(value, shift)
    assert window.shape == (count, 3, E8_VALID, E8_VALID)

    for index, (row, column) in enumerate(shift.tolist()):
        expected = value[
            index,
            E8_MAX_SHIFT + row : E8_MAX_SHIFT + row + E8_VALID,
            E8_MAX_SHIFT + column : E8_MAX_SHIFT + column + E8_VALID,
            :,
        ].permute(2, 0, 1)
        assert torch.equal(window[index], expected)


def test_aligned_window_rejects_a_batch_mismatch():
    """The shape check must fire rather than index the wrong axis."""
    value = torch.rand(4, 3, E8_CROP, E8_CROP)
    with pytest.raises(ValueError, match="batch on dim -3"):
        aligned_window(value, torch.zeros(4, 2, dtype=torch.long))


def test_passing_folds_survives_the_json_string_key_round_trip():
    """Regression: JSON has no integer keys.

    `folds_passing` round-trips as a list of integers while the `detail` map
    round-trips with string keys, so indexing detail by a fold from that list
    raised KeyError and killed all four encoding tasks in seconds.
    """
    import json

    from extract_e8_residual_features import passing_folds

    report = {
        "arms": {
            "free": {
                "gate": {
                    "folds_passing": [0, 2, 4],
                    "detail": {
                        fold: {"checkpoint": f"/ckpt/fold{fold}.ckpt"} for fold in range(5)
                    },
                }
            }
        }
    }
    round_tripped = json.loads(json.dumps(report))
    assert list(round_tripped["arms"]["free"]["gate"]["detail"]) == ["0", "1", "2", "3", "4"]

    found = passing_folds(round_tripped, "free")
    assert found == {
        0: "/ckpt/fold0.ckpt",
        2: "/ckpt/fold2.ckpt",
        4: "/ckpt/fold4.ckpt",
    }
    assert all(isinstance(fold, int) for fold in found)
