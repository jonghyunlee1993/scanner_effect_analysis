from build_e0_feature_manifest_slide import fov_in_bounds, stable_rng


def test_fov_in_bounds_uses_canonical_256_patch_center():
    assert fov_in_bounds(128, 128, 0, 0, 1024, 1024)
    assert not fov_in_bounds(127, 128, 0, 0, 1024, 1024)
    assert not fov_in_bounds(128, 127, 0, 0, 1024, 1024)


def test_fov_in_bounds_applies_scanner_integer_offset():
    assert fov_in_bounds(128, 128, 1, 1, 1024, 1024)
    assert not fov_in_bounds(128, 128, -1, 0, 1024, 1024)
    assert not fov_in_bounds(128, 128, 0, -1, 1024, 1024)


def test_candidate_order_is_slide_stable_and_slide_specific():
    first = stable_rng("slide-a").permutation(20)
    repeated = stable_rng("slide-a").permutation(20)
    other = stable_rng("slide-b").permutation(20)
    assert (first == repeated).all()
    assert not (first == other).all()
