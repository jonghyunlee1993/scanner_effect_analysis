import numpy as np
import torch

from e5_macenko_supplement import (
    aggregate_loso_reference,
    macenko_normalize_batch,
    macenko_parameters,
)


def test_macenko_parameter_and_loso_reference_are_finite():
    generator = torch.Generator().manual_seed(13)
    stains = torch.tensor([[0.65, 0.15], [0.70, 0.95], [0.29, 0.27]], dtype=torch.float64)
    concentration = torch.rand((2, 2000), generator=generator, dtype=torch.float64) * 1.5
    od = (stains @ concentration).T
    parameters = macenko_parameters(od)
    assert parameters is not None
    fitted_stains, fitted_maximum = parameters
    assert fitted_stains.shape == (3, 2)
    assert fitted_maximum.shape == (2,)
    population_stains = np.broadcast_to(fitted_stains.numpy(), (109, 3, 2)).copy()
    population_maximum = np.broadcast_to(fitted_maximum.numpy(), (109, 2)).copy()
    target_stains, target_maximum, train_slides = aggregate_loso_reference(
        population_stains, population_maximum, 0
    )
    assert train_slides == 108
    assert np.isfinite(target_stains).all()
    assert np.isfinite(target_maximum).all()


def test_macenko_batch_falls_back_on_blank_image():
    blank = torch.ones((2, 32, 32, 3), dtype=torch.float32)
    target_stains = torch.tensor([[0.6, 0.2], [0.7, 0.9], [0.3, 0.3]])
    target_maximum = torch.ones(2)
    report = macenko_normalize_batch(blank, target_stains, target_maximum)
    assert report["fallback"].all()
    assert torch.equal(report["output"], blank)

