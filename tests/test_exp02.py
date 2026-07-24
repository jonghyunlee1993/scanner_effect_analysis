import torch
import numpy as np

from eval_exp02_uni_bands import range_audit, variants as band_variants
from run_exp02_hf_trajectory import blue_dominant_fraction
from prenorm.data.identity import (
    EXTERNAL_S60_LATTICE_ID,
    INTERNAL_LATTICE_ID,
    LocationKey,
    location_key,
    matched_indices_by_lattice,
)
from prenorm.embedding import scanner_purity_excluding_location
from prenorm.exp01.frequency import FixedLaplacianPyramid
from prenorm.exp02.projection import (
    scalar_range_alpha,
    project_correction,
    low_correction_delta,
    local_range_project,
)
from prenorm.exp02.pairwise import (
    apply_affine,
    clipped_detail_retention_frontier,
    clipped_low_transform,
    detail_retention_frontier,
    solve_affine,
)
from prenorm.exp02.trajectory import (
    chord_direction,
    embedding_trajectory_metrics,
    gain_high_frequency,
    mix_high_frequency,
    pairwise_direction_cosine,
    spherical_mean,
)
from prenorm.metrics import per_image_ssim


def _random_case(seed, n=4, c=3, size=64):
    torch.manual_seed(seed)
    pyramid = FixedLaplacianPyramid(4)
    source = torch.rand(n, c, size, size) * 2 - 1            # x in [-1, 1]
    low, bands = pyramid.decompose(source)
    corrected_low = low + torch.randn_like(low) * 0.3        # arbitrary l'
    return pyramid, source, low, bands, corrected_low


def test_per_image_ssim_is_one_for_identical_images():
    torch.manual_seed(0)
    images = torch.rand(3, 3, 32, 32) * 2 - 1
    torch.testing.assert_close(
        per_image_ssim(images, images),
        torch.ones(3),
        atol=1e-6,
        rtol=0,
    )


def test_scanner_purity_can_exclude_registered_counterparts():
    locations = np.repeat(np.arange(3), 2)
    scanners = np.tile(np.arange(2), 3)
    embeddings = np.concatenate(
        (
            3.0 * np.eye(2, dtype=np.float32)[scanners],
            0.1 * np.eye(3, dtype=np.float32)[locations],
        ),
        axis=1,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    value = scanner_purity_excluding_location(
        torch.from_numpy(embeddings),
        locations,
        scanners,
        k=2,
    )

    assert value == 1.0


def test_exp02_projection_keeps_output_in_range():
    pyramid, source, _, _, corrected_low = _random_case(1)
    d, _, _ = low_correction_delta(pyramid, source, corrected_low)
    output, alpha, _, _ = project_correction(source, d)
    assert output.max() <= 1.0 + 1e-6
    assert output.min() >= -1.0 - 1e-6
    assert (alpha >= 0).all() and (alpha <= 1).all()


def test_exp02_projection_preserves_copied_detail_bandwise():
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


def test_exp02_alpha_is_maximal_and_linear():
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


def test_exp02_identity_input_needs_no_projection():
    # If x is already in range and d is zero, alpha is unconstrained -> full (1).
    torch.manual_seed(4)
    x = torch.rand(3, 3, 32, 32) * 2 - 1
    d = torch.zeros_like(x)
    _, alpha, alpha_raw, _ = project_correction(x, d)
    assert torch.isinf(alpha_raw).all()
    assert torch.equal(alpha, torch.ones_like(alpha))


def test_exp02_local_projection_is_range_safe_and_detail_exact():
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


def test_exp02_local_projection_retains_at_least_scalar_efficacy():
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


def test_location_key_namespaces_independent_lattices():
    internal = {
        "lattice_id": INTERNAL_LATTICE_ID,
        "slide_id": "8-12_14",
        "tuple_id": 7,
    }
    external = {
        "lattice_id": EXTERNAL_S60_LATTICE_ID,
        "slide_id": "8-12_14",
        "tuple_id": 7,
    }
    assert location_key(internal) != location_key(external)
    assert location_key(internal).group_token() != location_key(external).group_token()


def test_location_key_requires_lattice_identity():
    try:
        location_key({"slide_id": "8-12_14", "tuple_id": 7})
    except ValueError as error:
        assert "lattice_id" in str(error)
    else:
        raise AssertionError("missing lattice_id must fail closed")


def test_matched_indices_are_balanced_within_each_lattice():
    records = [
        {"scanner": "gt450", "lattice_id": INTERNAL_LATTICE_ID,
         "slide_id": "s", "tuple_id": 1},
        {"scanner": "gt450", "lattice_id": INTERNAL_LATTICE_ID,
         "slide_id": "s", "tuple_id": 2},
        {"scanner": "versa", "lattice_id": INTERNAL_LATTICE_ID,
         "slide_id": "s", "tuple_id": 2},
        {"scanner": "versa", "lattice_id": INTERNAL_LATTICE_ID,
         "slide_id": "s", "tuple_id": 3},
        {"scanner": "akoya", "lattice_id": INTERNAL_LATTICE_ID,
         "slide_id": "s", "tuple_id": 2},
        {"scanner": "s60", "lattice_id": EXTERNAL_S60_LATTICE_ID,
         "slide_id": "s", "tuple_id": 2},
        {"scanner": "s60", "lattice_id": EXTERNAL_S60_LATTICE_ID,
         "slide_id": "s", "tuple_id": 3},
    ]
    selected = matched_indices_by_lattice(
        records,
        ("gt450", "versa", "akoya", "s60"),
        None,
        np.random.default_rng(0),
    )
    internal = selected[INTERNAL_LATTICE_ID]
    external = selected[EXTERNAL_S60_LATTICE_ID]
    assert internal["keys"] == [LocationKey(INTERNAL_LATTICE_ID, "s", 2)]
    assert all(
        len(internal["indices"][scanner]) == 1
        for scanner in ("gt450", "versa", "akoya")
    )
    assert [key.tuple_id for key in external["keys"]] == [2, 3]


def test_band_full_oracle_is_the_exact_paired_reference():
    torch.manual_seed(7)
    pyramid = FixedLaplacianPyramid(3)
    source = torch.rand(2, 3, 32, 32) * 2 - 1
    reference = torch.rand(2, 3, 32, 32) * 2 - 1
    affine = torch.zeros(4, 3)
    affine[:3] = torch.eye(3)
    generated = band_variants(
        pyramid, source, reference, affine, steps=2
    )
    assert "both_oracle" not in generated
    assert "corr_low_at2_high" in generated
    assert torch.equal(generated["full_oracle"], reference)


def test_range_audit_reports_pre_clamp_violations_per_image():
    images = torch.zeros(2, 3, 2, 2)
    images[0, 0, 0, 0] = 1.25
    images[0, 1, 0, 0] = -1.10
    audit = range_audit(images)
    np.testing.assert_allclose(audit["pixel_fraction"], [2 / 12, 0])
    np.testing.assert_allclose(audit["max_excess"], [0.25, 0])


def test_pairwise_affine_recovers_known_rgb_mapping():
    torch.manual_seed(8)
    source = torch.rand(8, 3, 5, 5) * 1.2 - 0.6
    expected = torch.tensor([
        [0.90, 0.04, 0.01],
        [0.03, 1.05, 0.02],
        [0.01, 0.05, 0.95],
        [0.08, -0.03, 0.04],
    ])
    target = apply_affine(source, expected)
    fitted = solve_affine(source, target)
    torch.testing.assert_close(fitted, expected, atol=1e-5, rtol=1e-5)


def test_detail_retention_frontier_has_exact_endpoints():
    pyramid, source, low, bands, _ = _random_case(9)
    frontier = detail_retention_frontier(
        pyramid, low, bands, retention=(1.0, 0.5, 0.0)
    )
    torch.testing.assert_close(frontier[1.0], source, atol=1e-6, rtol=0)
    low_only = pyramid.reconstruct(
        low, [torch.zeros_like(band) for band in bands]
    )
    torch.testing.assert_close(frontier[0.0], low_only, atol=1e-6, rtol=0)


def test_clipped_low_transform_is_operational_not_detail_exact():
    pyramid = FixedLaplacianPyramid(3)
    source = torch.full((1, 3, 32, 32), 0.95)
    affine = torch.zeros(4, 3)
    affine[:3] = torch.eye(3)
    affine[3] = 0.2
    transformed = clipped_low_transform(
        pyramid, source, affine
    )
    assert transformed["preclip"].max() > 1.0
    assert transformed["output"].max() <= 1.0
    assert transformed["output"].min() >= -1.0
    assert not torch.equal(
        transformed["output"], transformed["preclip"]
    )


def test_clipped_detail_frontier_clips_each_requested_variant():
    pyramid, source, _, bands, corrected_low = _random_case(10)
    corrected_low = corrected_low + 2.0
    frontier = clipped_detail_retention_frontier(
        pyramid,
        corrected_low,
        bands,
        retention=(1.0, 0.0),
    )
    assert frontier["preclip"][1.0].max() > 1.0
    for image in frontier["clipped"].values():
        assert image.max() <= 1.0
        assert image.min() >= -1.0


def test_source_and_target_frontiers_use_the_same_retention_contract():
    torch.manual_seed(11)
    pyramid = FixedLaplacianPyramid(3)
    source = torch.rand(2, 3, 32, 32) * 2 - 1
    target = torch.rand(2, 3, 32, 32) * 2 - 1
    source_low, source_bands = pyramid.decompose(source)
    target_low, target_bands = pyramid.decompose(target)
    retention = (1.0, 0.5, 0.0)
    source_frontier = clipped_detail_retention_frontier(
        pyramid, source_low, source_bands, retention
    )["clipped"]
    target_frontier = clipped_detail_retention_frontier(
        pyramid, target_low, target_bands, retention
    )["clipped"]
    torch.testing.assert_close(
        source_frontier[1.0], source, atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        target_frontier[1.0], target, atol=1e-6, rtol=0
    )
    assert source_frontier.keys() == target_frontier.keys()


def test_high_frequency_gain_and_mix_have_exact_unclipped_endpoints():
    pyramid, source, low, bands, _ = _random_case(12)
    raw = gain_high_frequency(pyramid, low, bands, gain=1.0)
    torch.testing.assert_close(raw["image"], source, atol=1e-6, rtol=0)
    reference = [torch.zeros_like(band) for band in bands]
    mixed = mix_high_frequency(
        pyramid, low, bands, reference, amount=1.0
    )
    expected = pyramid.reconstruct(low, reference)
    torch.testing.assert_close(
        mixed["image"], expected.clamp(-1, 1), atol=1e-6, rtol=0
    )


def test_blue_dominant_fraction_separates_marker_cast_from_he_colors():
    marker = torch.tensor([0.10, 0.15, 0.80]).reshape(1, 3, 1, 1)
    eosin = torch.tensor([0.85, 0.45, 0.70]).reshape(1, 3, 1, 1)
    images = torch.cat((marker, eosin), dim=0) * 2.0 - 1.0
    fractions = blue_dominant_fraction(images)
    torch.testing.assert_close(fractions, torch.tensor([1.0, 0.0]))


def test_pairwise_embedding_direction_cosine_detects_opposition():
    baseline = np.zeros((3, 2))
    endpoint = np.asarray([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]])
    directions = chord_direction(endpoint, baseline)
    cosine = pairwise_direction_cosine(directions)
    np.testing.assert_allclose(
        cosine,
        [[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
    )


def test_embedding_trajectory_reports_consensus_aligned_motion():
    raw = np.asarray([[1.0, 0.0, 0.0]])
    consensus = np.asarray([[0.0, 1.0, 0.0]])
    metric = embedding_trajectory_metrics(consensus, raw, consensus)
    assert metric["direction_cosine_to_consensus"][0] > 0.99
    assert metric["consensus_cosine_gain"][0] > 0
    assert metric["orthogonal_drift"][0] < 1e-12
    mean = spherical_mean(np.concatenate((raw, consensus), axis=0))
    np.testing.assert_allclose(np.linalg.norm(mean), 1.0)
