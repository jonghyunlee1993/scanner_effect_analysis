import torch

from prenorm.losses import (
    apply_scanner_pair_weights,
    build_supervision_weights,
    coordinate_variance_loss,
    frequency_asymmetric_standard_loss,
    high_frequency_coherence,
    high_frequency_reliability,
    nuclei_masked_source_detail_loss,
    nuclei_masked_source_orientation_loss,
    nuclei_reconstruction_loss,
    reference_neighborhood_consistency_loss,
    robust_image_distance,
    sample_translation_pairs,
    spatial_pair_loss,
    target_structural_detail_loss,
    weighted_mean_with_coverage,
)


def test_supervision_weights_follow_v3_contract():
    present = torch.tensor([[True, True, True]])
    geom_ok = torch.tensor([[True, True, False]])
    q_reg = torch.tensor([[1.0, 0.5, 0.9]])
    weights = build_supervision_weights(present, geom_ok, q_reg, q_min=0.25)

    assert torch.allclose(weights["standard"], torch.tensor([[1.0, 0.625, 0.0]]))
    assert weights["cross"][0, 0, 1] == 0.625
    assert weights["cross"][0, 1, 0] == 0.625
    assert weights["cross"][0, 0, 2] == 0.0
    assert torch.equal(weights["self"], torch.ones_like(weights["self"]))


def test_zero_coverage_is_explicit_and_graph_connected():
    values = torch.tensor([1.0, 2.0], requires_grad=True)
    loss, coverage = weighted_mean_with_coverage(values, torch.zeros(2))
    loss.backward()
    assert loss == 0
    assert coverage == 0
    assert torch.equal(values.grad, torch.zeros_like(values))


def test_robust_distance_separates_equal_and_perturbed_images():
    target = torch.zeros(2, 3, 32, 32)
    equal = robust_image_distance(target, target)
    perturbed = robust_image_distance(target + 0.25, target)
    assert torch.all(perturbed > equal)


def test_high_frequency_coherence_rejects_unmatched_texture():
    torch.manual_seed(3)
    reference = torch.zeros(1, 3, 32, 32)
    reference[:, :, 8:24, 8:24] = -0.8
    noise = torch.rand_like(reference) * 2 - 1
    sources = torch.stack((reference, reference, noise), dim=1)
    coherence = high_frequency_coherence(sources, reference, kernel_size=9, sigma=2.0)

    assert coherence[0, 0] > 0.99
    assert coherence[0, 1] > 0.99
    assert coherence[0, 2] < 0.25


def test_high_frequency_reliability_uses_coherence_and_registration():
    reference = torch.zeros(1, 3, 32, 32)
    reference[:, :, 8:24, 8:24] = -0.8
    sources = torch.stack((reference, reference, reference), dim=1)
    q_reg = torch.tensor([[1.0, 1.0, 0.2]])
    reliability, coherence, energy_ratio = high_frequency_reliability(
        sources,
        reference,
        q_reg,
        reference_index=0,
        coherence_low=0.1,
        coherence_high=0.3,
        energy_low=0.5,
        energy_high=0.8,
        q_low=0.8,
        q_high=0.95,
    )

    assert reliability[0, 0] == 1
    assert reliability[0, 1] > 0.99
    assert reliability[0, 2] == 0
    assert torch.all(coherence > 0.99)
    assert torch.all(energy_ratio > 0.99)


def test_high_frequency_reliability_rejects_blurred_matching_structure():
    reference = torch.zeros(1, 3, 32, 32)
    reference[:, :, 8:24, 8:24] = -0.8
    blurred = torch.nn.functional.avg_pool2d(reference, 9, 1, 4)
    sources = torch.stack((reference, blurred), dim=1)
    reliability, coherence, energy_ratio = high_frequency_reliability(
        sources,
        reference,
        torch.ones(1, 2),
        reference_index=0,
        coherence_low=0.1,
        coherence_high=0.3,
        energy_low=0.5,
        energy_high=0.8,
        q_low=0.8,
        q_high=0.95,
    )

    assert coherence[0, 1] > 0.1
    assert energy_ratio[0, 1] < 0.5
    assert reliability[0, 1] == 0


def test_frequency_asymmetric_standard_blocks_unreliable_high_band_gradient():
    torch.manual_seed(5)
    reference = torch.rand(1, 3, 32, 32) * 2 - 1
    canonical = torch.zeros(1, 2, 3, 32, 32, requires_grad=True)
    total, low, high, low_coverage, high_coverage = (
        frequency_asymmetric_standard_loss(
            canonical,
            reference,
            low_weights=torch.zeros(1, 2),
            high_weights=torch.tensor([[1.0, 0.0]]),
            high_frequency_weight=0.25,
        )
    )
    total.backward()

    assert low == 0
    assert high > 0
    assert low_coverage == 0
    assert high_coverage == 0.5
    assert canonical.grad[0, 0].abs().sum() > 0
    assert canonical.grad[0, 1].abs().sum() == 0


def test_target_detail_is_zero_for_equal_and_penalizes_blur():
    target = torch.zeros(1, 1, 3, 32, 32)
    target[:, :, :, 8:24, 8:24] = -0.8
    prediction = target.clone().requires_grad_(True)
    equal, coverage = target_structural_detail_loss(
        prediction, target, torch.ones(1, 1)
    )
    blurred = torch.nn.functional.avg_pool2d(
        target.flatten(0, 1), 5, 1, 2
    ).view_as(target)
    soft, _ = target_structural_detail_loss(blurred, target, torch.ones(1, 1))
    equal.backward()
    assert equal < soft
    assert coverage == 1
    assert prediction.grad is not None


def test_target_detail_zero_coverage_stays_graph_connected():
    prediction = torch.zeros(1, 2, 3, 16, 16, requires_grad=True)
    value, coverage = target_structural_detail_loss(
        prediction, torch.zeros_like(prediction), torch.zeros(1, 2)
    )
    value.backward()
    assert value == 0
    assert coverage == 0
    assert prediction.grad is not None


def test_nuclei_reconstruction_uses_occupancy_and_instance_boundaries():
    labels = torch.zeros(1, 1, 1, 16, 16, dtype=torch.int32)
    labels[:, :, :, 3:8, 3:8] = 1
    labels[:, :, :, 8:13, 8:13] = 2
    logits = torch.full((1, 1, 2, 16, 16), -4.0, requires_grad=True)
    value, source, reference, coverage = nuclei_reconstruction_loss(
        logits,
        labels,
        labels[:, 0],
        torch.ones(1, 1),
        torch.zeros(1, 1),
    )
    value.backward()
    assert value == source
    assert reference == 0
    assert coverage == 0.5
    assert logits.grad is not None


def test_nuclei_masked_rgb_detail_is_zero_for_equal_and_penalizes_blur():
    source = torch.zeros(1, 1, 3, 32, 32)
    source[:, :, :, 10:22, 10:22] = -0.8
    labels = torch.zeros(1, 1, 1, 32, 32, dtype=torch.int32)
    labels[:, :, :, 10:22, 10:22] = 1
    equal = source.clone().requires_grad_(True)
    equal_value, coverage = nuclei_masked_source_detail_loss(
        equal, source, labels, torch.ones(1, 1)
    )
    blurred = torch.nn.functional.avg_pool2d(
        source.flatten(0, 1), 5, 1, 2
    ).view_as(source)
    blurred_value, _ = nuclei_masked_source_detail_loss(
        blurred, source, labels, torch.ones(1, 1)
    )
    equal_value.backward()
    assert equal_value < blurred_value
    assert coverage == 1
    assert equal.grad is not None


def test_nuclei_masked_rgb_detail_excludes_empty_masks():
    canonical = torch.ones(1, 2, 3, 16, 16, requires_grad=True)
    source = torch.zeros_like(canonical)
    labels = torch.zeros(1, 2, 1, 16, 16, dtype=torch.int32)
    value, coverage = nuclei_masked_source_detail_loss(
        canonical, source, labels, torch.ones(1, 2)
    )
    value.backward()
    assert value == 0
    assert coverage == 0
    assert canonical.grad is not None


def test_nuclei_orientation_is_polarity_invariant_and_spatially_sensitive():
    source = torch.zeros(1, 1, 3, 32, 32)
    source[:, :, :, :, 16:] = -0.8
    orthogonal = torch.zeros_like(source)
    orthogonal[:, :, :, 16:, :] = -0.8
    labels = torch.ones(1, 1, 1, 32, 32, dtype=torch.int32)
    equal = source.clone().requires_grad_(True)
    inverted = (-source).requires_grad_(True)

    equal_value, coverage = nuclei_masked_source_orientation_loss(
        equal, source, labels, torch.ones(1, 1)
    )
    inverted_value, _ = nuclei_masked_source_orientation_loss(
        inverted, source, labels, torch.ones(1, 1)
    )
    orthogonal_value, _ = nuclei_masked_source_orientation_loss(
        orthogonal, source, labels, torch.ones(1, 1)
    )
    equal_value.backward()

    assert equal_value < 0.01
    assert inverted_value < 0.01
    assert orthogonal_value > equal_value + 0.25
    assert coverage == 1
    assert equal.grad is not None


def test_nuclei_orientation_excludes_empty_masks():
    canonical = torch.rand(1, 2, 3, 16, 16, requires_grad=True)
    source = torch.rand_like(canonical)
    labels = torch.zeros(1, 2, 1, 16, 16, dtype=torch.int32)
    value, coverage = nuclei_masked_source_orientation_loss(
        canonical, source, labels, torch.ones(1, 2)
    )
    value.backward()

    assert value == 0
    assert coverage == 0
    assert canonical.grad is not None


def test_nuclei_reconstruction_validates_shapes():
    try:
        nuclei_reconstruction_loss(
            torch.zeros(1, 1, 1, 8, 8),
            torch.zeros(1, 1, 1, 8, 8),
            torch.zeros(1, 1, 8, 8),
            torch.ones(1, 1),
            torch.ones(1, 1),
        )
    except ValueError as error:
        assert "logits" in str(error)
    else:
        raise AssertionError("invalid nuclei logits were accepted")


def test_spatial_pair_loss_does_not_reduce_content_to_gap():
    content = torch.zeros(1, 2, 1, 2, 2)
    content[0, 1, 0] = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    cross = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    loss, coverage = spatial_pair_loss(content, cross)
    assert loss > 0
    assert coverage == 0.5


def test_reference_neighborhood_consistency_matches_relational_geometry():
    torch.manual_seed(11)
    reference = torch.randn(4, 3, 4, 4)
    matched = torch.stack((reference, reference.clone()), dim=1)
    permuted = torch.stack((reference, reference[[0, 2, 1, 3]]), dim=1)
    present = geom_ok = torch.ones(4, 2, dtype=torch.bool)
    q_reg = torch.ones(4, 2)
    slide_group = torch.zeros(4, dtype=torch.long)

    matched_loss, matched_coverage = reference_neighborhood_consistency_loss(
        matched, present, geom_ok, q_reg, slide_group, reference_index=0,
        temperature=0.1, pool_size=2, min_q=0.8,
    )
    permuted_loss, permuted_coverage = reference_neighborhood_consistency_loss(
        permuted, present, geom_ok, q_reg, slide_group, reference_index=0,
        temperature=0.1, pool_size=2, min_q=0.8,
    )

    assert abs(float(matched_loss)) < 1e-6
    assert permuted_loss > matched_loss + 1e-3
    assert matched_coverage == 1
    assert permuted_coverage == 1


def test_reference_neighborhood_teacher_is_detached():
    torch.manual_seed(13)
    content = torch.randn(4, 2, 3, 4, 4, requires_grad=True)
    present = geom_ok = torch.ones(4, 2, dtype=torch.bool)
    q_reg = torch.ones(4, 2)
    value, coverage = reference_neighborhood_consistency_loss(
        content,
        present,
        geom_ok,
        q_reg,
        torch.zeros(4, dtype=torch.long),
        reference_index=0,
        pool_size=2,
    )
    value.backward()

    assert value > 0
    assert coverage == 1
    assert torch.equal(content.grad[:, 0], torch.zeros_like(content.grad[:, 0]))
    assert content.grad[:, 1].abs().sum() > 0


def test_reference_neighborhood_zero_coverage_is_graph_connected():
    content = torch.randn(3, 2, 2, 4, 4, requires_grad=True)
    present = geom_ok = torch.ones(3, 2, dtype=torch.bool)
    q_reg = torch.tensor([[1.0, 0.2], [1.0, 0.3], [1.0, 0.4]])
    value, coverage = reference_neighborhood_consistency_loss(
        content,
        present,
        geom_ok,
        q_reg,
        torch.zeros(3, dtype=torch.long),
        reference_index=0,
        min_q=0.8,
    )
    value.backward()

    assert value == 0
    assert coverage == 0
    assert torch.equal(content.grad, torch.zeros_like(content.grad))


def test_scanner_pair_weight_only_reduces_pairs_involving_scanner():
    cross = torch.ones(1, 3, 3)
    cross[:, torch.arange(3), torch.arange(3)] = 0
    weighted = apply_scanner_pair_weights(
        cross, ["at2", "gt450", "versa"], {"versa": 0.2}
    )

    assert weighted[0, 0, 1] == 1.0
    assert weighted[0, 1, 0] == 1.0
    assert weighted[0, 0, 2] == 0.2
    assert weighted[0, 2, 1] == 0.2
    assert weighted[0, 2, 2] == 0.0


def test_scanner_pair_weight_validates_configuration():
    cross = torch.ones(1, 2, 2)
    try:
        apply_scanner_pair_weights(cross, ["at2", "versa"], {"versa": 1.1})
    except ValueError as error:
        assert "[0, 1]" in str(error)
    else:
        raise AssertionError("out-of-range pair weight was accepted")


def test_variance_is_measured_across_coordinates():
    content = torch.zeros(2, 2, 1, 2, 2)
    content[1] = 2.0
    weights = torch.ones(2, 2)
    slide_group = torch.zeros(2, dtype=torch.long)
    varied, coverage = coordinate_variance_loss(content, weights, slide_group, gamma=1.0)
    collapsed, _ = coordinate_variance_loss(content * 0, weights, slide_group, gamma=1.0)
    assert varied < collapsed
    assert coverage == 1


def test_translation_sampler_draws_one_self_and_one_cross_per_coordinate():
    present = torch.ones(3, 2, dtype=torch.bool)
    cross = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]] * 3)
    batch, source, target, weight, is_self = sample_translation_pairs(
        present, cross, torch.Generator().manual_seed(7)
    )
    assert batch.numel() == source.numel() == target.numel() == weight.numel() == 6
    assert is_self.sum() == 3
    assert (~is_self).sum() == 3
