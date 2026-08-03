import numpy as np
import torch

from analyze_rf1_multiscale import (
    PILOT_GAIN_CAP,
    PYRAMID_SIGMAS,
    fitted_multiscale_gains,
    laplacian_pyramid,
    laplacian_transfer,
    radial_power_to_band_energy,
    shared_od_multiscale,
)
from e5_comparator_population import radial_geometry, rgb01_to_od
from e5_reinhard_residual_frequency import OD_RGB8_MAX


def test_laplacian_pyramid_reconstructs_input():
    rng = np.random.default_rng(31)
    value = torch.tensor(rng.normal(size=(2, 32, 32)), dtype=torch.float32)
    bands, base = laplacian_pyramid(value)
    reconstructed = base + sum(bands)
    assert len(bands) == len(PYRAMID_SIGMAS)
    assert torch.allclose(reconstructed, value, atol=2e-6, rtol=0)


def test_identity_multiscale_gain_is_identity():
    rng = np.random.default_rng(32)
    rgb = torch.tensor(rng.uniform(0.05, 0.95, size=(2, 32, 32, 3)), dtype=torch.float32)
    report = shared_od_multiscale(rgb, np.ones(len(PYRAMID_SIGMAS)))
    assert torch.allclose(report["output"], rgb, atol=3e-6, rtol=0)
    assert float(report["projection_fraction"].max()) == 0.0
    assert float(report["final_clamp_mae"].max()) < 1e-8


def test_gamut_projection_preserves_shared_od_chroma():
    y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    pattern = ((x + y) % 2).float()
    rgb = torch.stack(
        [0.94 - 0.80 * pattern, 0.81 - 0.62 * pattern, 0.69 - 0.48 * pattern],
        dim=-1,
    )[None]
    report = shared_od_multiscale(rgb, np.full(len(PYRAMID_SIGMAS), 4.0))
    output = report["output"]
    source_od = rgb01_to_od(rgb)
    output_od = rgb01_to_od(output)
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
    assert torch.allclose(
        source_od[..., 0] - source_od[..., 1],
        output_od[..., 0] - output_od[..., 1],
        atol=2e-5,
        rtol=0,
    )
    assert float(output_od.min()) >= -1e-6
    assert float(output_od.max()) <= OD_RGB8_MAX + 1e-5
    assert float(report["final_clamp_mae"].max()) < 1e-8


def test_band_transfer_and_energy_are_finite_nonnegative():
    geometry = radial_geometry(256)
    transfer, base = laplacian_transfer(geometry.frequency)
    energy = radial_power_to_band_energy(
        np.linspace(1.0, 3.0, len(geometry.frequency)), geometry.frequency
    )
    assert transfer.shape == (3, 72)
    assert base.shape == (72,)
    assert np.isfinite(energy).all()
    assert np.all(energy > 0)
    assert np.all(transfer >= 0)
    assert np.all(base >= 0)
    assert np.allclose(transfer.sum(axis=0) + base, 1.0)


def test_fitted_band_gains_obey_cap_and_direction():
    geometry = radial_geometry(256)
    source = np.stack(
        [np.ones(72), np.full(72, 4.0), np.full(72, 0.25)], axis=0
    )
    target = np.full(72, 2.0)
    gains = fitted_multiscale_gains(source, target, geometry.frequency)
    assert gains.shape == (3, 3)
    assert np.all(gains >= 1.0 / PILOT_GAIN_CAP)
    assert np.all(gains <= PILOT_GAIN_CAP)
    assert np.all(gains[0] > 1.0)
    assert np.all(gains[1] < 1.0)
    assert np.all(gains[2] > 1.0)
