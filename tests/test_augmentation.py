import torch

from prenorm.augmentation import ContentNuisanceAugmenter, DifferentiableJPEG


def test_jpeg_preserves_shape_range_and_gradient():
    image = torch.rand(3, 16, 16, requires_grad=True)
    output = DifferentiableJPEG()(image, quality=60)
    assert output.shape == image.shape
    assert output.min() >= 0 and output.max() <= 1
    output.mean().backward()
    assert image.grad is not None


def test_nuisance_augmenter_preserves_outer_shape_and_clean_zero_strength():
    rgb = torch.rand(2, 3, 3, 16, 16) * 2 - 1
    augmenter = ContentNuisanceAugmenter(jpeg_quality=(80, 90))
    assert torch.equal(augmenter(rgb, strength=0.0), rgb)
    augmented = augmenter(rgb, strength=0.5)
    assert augmented.shape == rgb.shape
    assert augmented.min() >= -1 and augmented.max() <= 1
