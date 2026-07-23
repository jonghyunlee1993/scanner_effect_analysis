"""Losses for registration-supervised Phase-1 canonicalization.

The functions in this module operate on explicit tensors and return coverage next
to weighted losses.  A zero-coverage batch is therefore visible to the trainer
instead of being hidden by a clamped denominator.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor:
    """Convert NCHW RGB to fixed linear luminance without changing its range."""
    weights = rgb.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def sobel_components(gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return horizontal and vertical Sobel responses for N1HW images."""
    kernel_x = gray.new_tensor(
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(2, 3)
    return (
        F.conv2d(gray, kernel_x, padding=1),
        F.conv2d(gray, kernel_y, padding=1),
    )


def weighted_mean_with_coverage(
    values: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a weighted mean and the fraction of candidates with positive weight."""
    weights = weights.to(dtype=values.dtype, device=values.device)
    coverage = (weights > 0).to(values.dtype).mean() if weights.numel() else values.new_zeros(())
    denominator = weights.sum()
    if weights.numel() == 0 or not bool(denominator.detach() > 0):
        return values.sum() * 0.0, coverage
    return (values * weights).sum() / denominator, coverage


def _charbonnier(error: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.sqrt(error.square() + epsilon * epsilon)


def _per_image_mean(value: torch.Tensor) -> torch.Tensor:
    return value.flatten(1).mean(dim=1)


def gaussian_lowpass(
    image: torch.Tensor,
    kernel_size: int = 9,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Apply a fixed depthwise Gaussian low-pass filter to an NCHW image."""
    if image.ndim != 4:
        raise ValueError("image must have shape (N, C, H, W)")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("low-pass kernel must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("low-pass sigma must be positive")
    if kernel_size == 1:
        return image
    radius = kernel_size // 2
    if image.shape[-2] <= radius or image.shape[-1] <= radius:
        raise ValueError("low-pass kernel is too large for the image")
    coordinate = torch.arange(
        -radius, radius + 1, dtype=image.dtype, device=image.device
    )
    kernel_1d = torch.exp(-0.5 * (coordinate / float(sigma)).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).view(
        1, 1, kernel_size, kernel_size
    )
    kernel_2d = kernel_2d.expand(image.shape[1], 1, -1, -1)
    padded = F.pad(image, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel_2d, groups=image.shape[1])


def _smooth_ramp(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    if high <= low:
        raise ValueError("reliability ramp high must be greater than low")
    position = ((value - float(low)) / float(high - low)).clamp(0.0, 1.0)
    return position.square() * (3.0 - 2.0 * position)


def high_frequency_statistics(
    source_rgb: torch.Tensor,
    reference_rgb: torch.Tensor,
    kernel_size: int = 9,
    sigma: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure high-frequency coherence and energy relative to registered AT2.

    The normalized correlation is computed on grayscale high-pass residuals
    inside tissue shared by the source and reference. Scanner-specific noise can
    have high energy while remaining incoherent with AT2. Blur can retain high
    normalized correlation while reducing residual energy, so both statistics
    are required for a useful recoverability proxy.
    """
    if source_rgb.ndim != 5:
        raise ValueError("source RGB must have shape (B, S, 3, H, W)")
    if reference_rgb.shape != source_rgb.shape[:1] + source_rgb.shape[2:]:
        raise ValueError("reference RGB must have shape (B, 3, H, W)")
    batch, scanners = source_rgb.shape[:2]
    source = source_rgb.reshape(batch * scanners, *source_rgb.shape[2:])
    reference = reference_rgb.unsqueeze(1).expand_as(source_rgb).reshape_as(source)
    source_gray = rgb_to_gray(source)
    reference_gray = rgb_to_gray(reference)
    source_high = source_gray - gaussian_lowpass(
        source_gray, kernel_size=kernel_size, sigma=sigma
    )
    reference_high = reference_gray - gaussian_lowpass(
        reference_gray, kernel_size=kernel_size, sigma=sigma
    )
    tissue = ((source_gray < 0.85) & (reference_gray < 0.85)).to(source_high.dtype)
    source_high = source_high * tissue
    reference_high = reference_high * tissue
    numerator = (source_high * reference_high).flatten(1).sum(dim=1)
    source_energy = source_high.square().flatten(1).sum(dim=1).clamp_min(1e-12).sqrt()
    reference_energy = (
        reference_high.square().flatten(1).sum(dim=1).clamp_min(1e-12).sqrt()
    )
    denominator = source_energy * reference_energy
    coherence = torch.where(
        denominator > 1e-6,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
    energy_ratio = source_energy / reference_energy
    return (
        coherence.clamp(-1.0, 1.0).view(batch, scanners),
        energy_ratio.view(batch, scanners),
    )


def high_frequency_coherence(
    source_rgb: torch.Tensor,
    reference_rgb: torch.Tensor,
    kernel_size: int = 9,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Return the registered high-frequency correlation with AT2."""
    coherence, _ = high_frequency_statistics(
        source_rgb, reference_rgb, kernel_size=kernel_size, sigma=sigma
    )
    return coherence


def high_frequency_reliability(
    source_rgb: torch.Tensor,
    reference_rgb: torch.Tensor,
    q_reg: torch.Tensor,
    reference_index: int,
    coherence_low: float,
    coherence_high: float,
    energy_low: float,
    energy_high: float,
    q_low: float,
    q_high: float,
    kernel_size: int = 9,
    sigma: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return detached detail reliability, coherence, and relative energy."""
    if q_reg.shape != source_rgb.shape[:2]:
        raise ValueError("q_reg must match the source batch/scanner axes")
    if not 0 <= reference_index < source_rgb.shape[1]:
        raise ValueError("reference index is outside the scanner axis")
    coherence, energy_ratio = high_frequency_statistics(
        source_rgb.detach(),
        reference_rgb.detach(),
        kernel_size=kernel_size,
        sigma=sigma,
    )
    coherence_weight = _smooth_ramp(coherence, coherence_low, coherence_high)
    energy_weight = _smooth_ramp(energy_ratio, energy_low, energy_high)
    registration_weight = _smooth_ramp(q_reg.detach(), q_low, q_high)
    reliability = coherence_weight * energy_weight * registration_weight
    reliability[:, reference_index] = 1.0
    coherence[:, reference_index] = 1.0
    energy_ratio[:, reference_index] = 1.0
    return reliability, coherence, energy_ratio


def robust_image_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gradient_weight: float = 0.0,
    epsilon: float = 1e-3,
    pyramid_scales: tuple[int, ...] = (1, 2, 4),
    pyramid_weights: tuple[float, ...] = (1.0, 0.5, 0.25),
) -> torch.Tensor:
    """Compute a per-image Charbonnier pyramid distance with an optional edge term."""
    if len(pyramid_scales) != len(pyramid_weights):
        raise ValueError("pyramid scales and weights must have equal length")

    total = prediction.new_zeros(prediction.shape[0])
    normalizer = float(sum(pyramid_weights))
    for scale, weight in zip(pyramid_scales, pyramid_weights):
        if scale == 1:
            pred_level, target_level = prediction, target
        else:
            pred_level = F.avg_pool2d(prediction, scale, scale)
            target_level = F.avg_pool2d(target, scale, scale)
        pixel = _per_image_mean(_charbonnier(pred_level - target_level, epsilon))
        if gradient_weight > 0:
            pred_grad = sobel_components(rgb_to_gray(pred_level))
            target_grad = sobel_components(rgb_to_gray(target_level))
            gradient = 0.5 * (
                _per_image_mean(_charbonnier(pred_grad[0] - target_grad[0], epsilon))
                + _per_image_mean(_charbonnier(pred_grad[1] - target_grad[1], epsilon))
            )
            pixel = pixel + gradient_weight * gradient
        total = total + weight * pixel
    return total / normalizer


def build_supervision_weights(
    present: torch.Tensor,
    geom_ok: torch.Tensor,
    q_reg: torch.Tensor,
    q_min: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Build standard, cross, and self weights from the v3 registration contract."""
    confidence = q_min + (1.0 - q_min) * q_reg.to(torch.float32)
    usable = present & geom_ok
    standard = usable.to(confidence.dtype) * confidence
    cross_mask = usable.unsqueeze(2) & usable.unsqueeze(1)
    scanner_count = usable.shape[-1]
    diagonal = torch.eye(scanner_count, dtype=torch.bool, device=usable.device)
    cross_mask = cross_mask & ~diagonal.view(1, scanner_count, scanner_count)
    cross_confidence = torch.minimum(confidence.unsqueeze(2), confidence.unsqueeze(1))
    cross = cross_mask.to(confidence.dtype) * cross_confidence
    return {
        "confidence": confidence,
        "standard": standard,
        "cross": cross,
        "self": present.to(confidence.dtype),
    }


def apply_scanner_pair_weights(
    cross_weights: torch.Tensor,
    scanners: list[str] | tuple[str, ...],
    scanner_weights: dict[str, float],
) -> torch.Tensor:
    """Downweight pairs involving configured scanners without changing eligibility.

    The pair factor is the minimum endpoint weight, so a pair involving one or
    more downweighted scanners receives that scanner's strongest reduction.
    Registration confidence remains owned by ``cross_weights``.
    """
    scanner_count = len(scanners)
    if cross_weights.shape[-2:] != (scanner_count, scanner_count):
        raise ValueError("cross-weight shape does not match the scanner list")
    unknown = sorted(set(scanner_weights).difference(scanners))
    if unknown:
        raise ValueError(f"pair weights configured for unknown scanners: {', '.join(unknown)}")

    endpoint = cross_weights.new_ones(scanner_count)
    for scanner, value in scanner_weights.items():
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"pair scanner weight for {scanner} must be in [0, 1]")
        endpoint[scanners.index(scanner)] = value
    pair_factor = torch.minimum(endpoint[:, None], endpoint[None, :])
    return cross_weights * pair_factor


def standard_loss(
    canonical: torch.Tensor,
    reference_rgb: torch.Tensor,
    weights: torch.Tensor,
    gradient_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervise every source canonical against the clean registered reference."""
    batch, scanners = canonical.shape[:2]
    target = reference_rgb.unsqueeze(1).expand(-1, scanners, -1, -1, -1)
    distance = robust_image_distance(
        canonical.reshape(batch * scanners, *canonical.shape[2:]),
        target.reshape(batch * scanners, *target.shape[2:]),
        gradient_weight=gradient_weight,
    ).view(batch, scanners)
    return weighted_mean_with_coverage(distance, weights)


def frequency_asymmetric_standard_loss(
    canonical: torch.Tensor,
    reference_rgb: torch.Tensor,
    low_weights: torch.Tensor,
    high_weights: torch.Tensor,
    lowpass_kernel: int = 9,
    lowpass_sigma: float = 2.0,
    high_frequency_weight: float = 0.25,
    gradient_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split AT2 supervision into universal low and reliable-view high bands.

    Every geometrically usable source contributes to the Gaussian-low-pass term.
    Only sources selected by ``high_weights`` contribute their residual detail.
    The high-pass term deliberately omits another Sobel penalty because the
    residual already isolates the frequency band that it supervises.
    """
    if canonical.ndim != 5:
        raise ValueError("canonical must have shape (B, S, 3, H, W)")
    if reference_rgb.shape != canonical.shape[:1] + canonical.shape[2:]:
        raise ValueError("reference RGB must have shape (B, 3, H, W)")
    if low_weights.shape != canonical.shape[:2] or high_weights.shape != canonical.shape[:2]:
        raise ValueError("frequency weights must match the canonical batch/scanner axes")
    if high_frequency_weight < 0:
        raise ValueError("high-frequency loss weight must be non-negative")

    batch, scanners = canonical.shape[:2]
    target = reference_rgb.unsqueeze(1).expand_as(canonical)
    prediction = canonical.reshape(batch * scanners, *canonical.shape[2:])
    target = target.reshape_as(prediction)
    prediction_low = gaussian_lowpass(
        prediction, kernel_size=lowpass_kernel, sigma=lowpass_sigma
    )
    target_low = gaussian_lowpass(
        target, kernel_size=lowpass_kernel, sigma=lowpass_sigma
    )
    low_distance = robust_image_distance(
        prediction_low,
        target_low,
        gradient_weight=gradient_weight,
    ).view(batch, scanners)
    high_distance = robust_image_distance(
        prediction - prediction_low,
        target - target_low,
        gradient_weight=0.0,
        pyramid_scales=(1,),
        pyramid_weights=(1.0,),
    ).view(batch, scanners)
    low_value, low_coverage = weighted_mean_with_coverage(low_distance, low_weights)
    high_value, high_coverage = weighted_mean_with_coverage(high_distance, high_weights)
    total = low_value + float(high_frequency_weight) * high_value
    return total, low_value, high_value, low_coverage, high_coverage


def translation_loss(
    rendered: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    gradient_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervise sampled self or ordered cross-scanner renders."""
    distance = robust_image_distance(rendered, targets, gradient_weight=gradient_weight)
    return weighted_mean_with_coverage(distance, weights)


def spatial_pair_distances(content: torch.Tensor) -> torch.Tensor:
    """Return registered scanner-pair RMS distances over complete feature maps."""
    difference = content.unsqueeze(2) - content.unsqueeze(1)
    return difference.square().flatten(3).mean(dim=3).clamp_min(1e-12).sqrt()


def spatial_pair_loss(
    content: torch.Tensor, cross_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match registered scanner content using the complete spatial feature maps."""
    return weighted_mean_with_coverage(spatial_pair_distances(content), cross_weights)


def reference_neighborhood_consistency_loss(
    content: torch.Tensor,
    present: torch.Tensor,
    geom_ok: torch.Tensor,
    q_reg: torch.Tensor,
    slide_group: torch.Tensor,
    reference_index: int,
    temperature: float = 0.1,
    pool_size: int = 4,
    min_q: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align each scanner's location-neighborhood distribution to detached AT2.

    Content maps are spatially pooled, flattened, and normalized.  For each
    same-slide group and non-reference scanner, the function compares the
    softmax distribution of pairwise location similarities with the equivalent
    detached reference-scanner distribution.  It never pulls two locations
    together directly, so biological differences remain represented by the
    reference neighborhood geometry.

    At least three common high-quality locations are required: one query and
    two non-self candidates.  Coverage is the fraction of possible
    non-reference scanner/query rows that contribute.
    """
    if content.ndim != 5:
        raise ValueError("content must have shape (B, S, C, H, W)")
    if present.shape != content.shape[:2] or geom_ok.shape != content.shape[:2]:
        raise ValueError("present and geom_ok must match content batch/scanner axes")
    if q_reg.shape != content.shape[:2]:
        raise ValueError("q_reg must match content batch/scanner axes")
    if slide_group.shape != content.shape[:1]:
        raise ValueError("slide_group must have one entry per batch location")
    if not 0 <= reference_index < content.shape[1]:
        raise ValueError("reference index is outside the scanner axis")
    if temperature <= 0:
        raise ValueError("neighborhood temperature must be positive")
    if pool_size < 1:
        raise ValueError("neighborhood pool size must be positive")
    if not 0 <= min_q <= 1:
        raise ValueError("neighborhood min_q must be in [0, 1]")

    batch, scanners = content.shape[:2]
    flat = content.reshape(batch * scanners, *content.shape[2:])
    pooled = F.adaptive_avg_pool2d(flat, (pool_size, pool_size))
    embedding = F.normalize(pooled.flatten(1), dim=1).view(batch, scanners, -1)
    usable = present & geom_ok & (q_reg >= float(min_q))

    divergences = []
    row_weights = []
    for group in torch.unique(slide_group):
        group_rows = torch.nonzero(slide_group == group, as_tuple=False).flatten()
        if group_rows.numel() < 3:
            continue
        reference_usable = usable[group_rows, reference_index]
        for scanner_index in range(scanners):
            if scanner_index == reference_index:
                continue
            common = reference_usable & usable[group_rows, scanner_index]
            selected = group_rows[common]
            if selected.numel() < 3:
                continue

            reference = embedding[selected, reference_index]
            student = embedding[selected, scanner_index]
            reference_logits = (reference @ reference.t()).detach() / float(temperature)
            student_logits = (student @ student.t()) / float(temperature)
            diagonal = torch.eye(
                selected.numel(), dtype=torch.bool, device=content.device
            )
            reference_logits = reference_logits.masked_fill(diagonal, -torch.inf)
            student_logits = student_logits.masked_fill(diagonal, -torch.inf)
            reference_log_probability = F.log_softmax(reference_logits, dim=1)
            reference_probability = reference_log_probability.exp()
            student_log_probability = F.log_softmax(student_logits, dim=1)
            divergence = (
                reference_probability
                * (reference_log_probability - student_log_probability)
            ).masked_fill(diagonal, 0.0).sum(dim=1)
            divergences.append(divergence)
            row_weights.append(q_reg[selected, scanner_index].to(divergence.dtype))

    possible_rows = batch * max(scanners - 1, 0)
    if not divergences:
        return content.sum() * 0.0, content.new_zeros(())
    divergence = torch.cat(divergences)
    weights = torch.cat(row_weights)
    loss = (divergence * weights).sum() / weights.sum().clamp_min(1e-12)
    coverage = content.new_tensor(divergence.numel() / max(possible_rows, 1))
    return loss, coverage


def reference_lowpass_pair_distances(
    content: torch.Tensor,
    reference_index: int,
    kernel_size: int = 5,
) -> torch.Tensor:
    """Return each scanner's low-pass content distance from detached AT2 content."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("pair low-pass kernel must be a positive odd integer")
    batch, scanners, channels, height, width = content.shape
    flat = content.reshape(batch * scanners, channels, height, width)
    if kernel_size > 1:
        radius = kernel_size // 2
        flat = F.avg_pool2d(
            F.pad(flat, (radius, radius, radius, radius), mode="reflect"),
            kernel_size=kernel_size,
            stride=1,
        )
    smooth = flat.view(batch, scanners, channels, height, width)
    reference = smooth[:, reference_index].detach().unsqueeze(1)
    return (smooth - reference).square().flatten(2).mean(dim=2).clamp_min(1e-12).sqrt()


def reference_lowpass_pair_loss(
    content: torch.Tensor,
    cross_weights: torch.Tensor,
    reference_index: int,
    kernel_size: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull each source toward a detached, low-pass AT2 content representation.

    Unlike the symmetric full-map pair loss, the reference representation is not
    moved toward lower-quality scanners and the comparison ignores the finest
    registration-sensitive feature variation.
    """
    rms = reference_lowpass_pair_distances(content, reference_index, kernel_size)
    weights = cross_weights[:, :, reference_index]
    return weighted_mean_with_coverage(rms, weights)


def _laplacian(gray: torch.Tensor) -> torch.Tensor:
    kernel = gray.new_tensor(
        ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
    ).view(1, 1, 3, 3)
    return F.conv2d(gray, kernel, padding=1)


def _masked_per_image(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(value.dtype)
    numerator = (value * mask).flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def _ssim_map(prediction: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
    """Differentiable local grayscale SSIM map for tensors in [-1, 1]."""
    padding = window // 2
    mu_x = F.avg_pool2d(prediction, window, 1, padding)
    mu_y = F.avg_pool2d(target, window, 1, padding)
    var_x = F.avg_pool2d(prediction.square(), window, 1, padding) - mu_x.square()
    var_y = F.avg_pool2d(target.square(), window, 1, padding) - mu_y.square()
    covariance = F.avg_pool2d(prediction * target, window, 1, padding) - mu_x * mu_y
    c1, c2 = 0.02**2, 0.06**2
    return ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    ).clamp_min(1e-6)


def structural_detail_loss(
    canonical: torch.Tensor,
    source_rgb: torch.Tensor,
    reference_rgb: torch.Tensor,
    weights: torch.Tensor,
    q_reg: torch.Tensor,
    high_q: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve foreground source detail and match reliable AT2 structure.

    Source Sobel/Laplacian responses discourage smoothing.  At high-registration-
    quality locations, AT2 Sobel/Laplacian and local SSIM additionally constrain
    the preserved structure so scanner-specific noise is not blindly copied.
    """
    batch, scanners = canonical.shape[:2]
    prediction = canonical.reshape(batch * scanners, *canonical.shape[2:])
    source = source_rgb.reshape(batch * scanners, *source_rgb.shape[2:])
    target = reference_rgb.unsqueeze(1).expand_as(canonical).reshape_as(prediction)
    pred_gray, source_gray, target_gray = map(rgb_to_gray, (prediction, source, target))

    source_mask = source_gray < 0.85
    target_mask = target_gray < 0.85
    source_grad = sobel_components(source_gray)
    pred_grad = sobel_components(pred_gray)
    source_edge = 0.5 * (
        (pred_grad[0] - source_grad[0]).abs() + (pred_grad[1] - source_grad[1]).abs()
    ) + 0.25 * (_laplacian(pred_gray) - _laplacian(source_gray)).abs()
    source_term = _masked_per_image(source_edge, source_mask).view(batch, scanners)
    source_value, source_coverage = weighted_mean_with_coverage(source_term, weights)

    target_grad = sobel_components(target_gray)
    target_edge = 0.5 * (
        (pred_grad[0] - target_grad[0]).abs() + (pred_grad[1] - target_grad[1]).abs()
    ) + 0.25 * (_laplacian(pred_gray) - _laplacian(target_gray)).abs()
    target_structure = target_edge + (1.0 - _ssim_map(pred_gray, target_gray)).clamp_min(0.0)
    target_term = _masked_per_image(target_structure, source_mask & target_mask).view(
        batch, scanners
    )
    target_weights = weights * (q_reg >= high_q).to(weights.dtype)
    target_value, target_coverage = weighted_mean_with_coverage(target_term, target_weights)
    coverage = 0.5 * (source_coverage + target_coverage)
    return 0.5 * (source_value + target_value), coverage


def target_structural_detail_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match target morphology without copying source-scanner texture.

    Inputs are ``(B, S, 3, H, W)`` and weights are ``(B, S)``.  Unlike the
    legacy detail loss, this objective contains no source-image edge term.
    """
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must share shape (B, S, 3, H, W)")
    if weights.shape != prediction.shape[:2]:
        raise ValueError("weights must match the prediction batch/scanner axes")
    batch, scanners = prediction.shape[:2]
    pred_gray = rgb_to_gray(prediction.reshape(batch * scanners, *prediction.shape[2:]))
    target_gray = rgb_to_gray(target.reshape(batch * scanners, *target.shape[2:]))
    tissue = target_gray < 0.85
    pred_grad = sobel_components(pred_gray)
    target_grad = sobel_components(target_gray)
    detail = 0.5 * (
        (pred_grad[0] - target_grad[0]).abs()
        + (pred_grad[1] - target_grad[1]).abs()
    )
    detail = detail + 0.25 * (
        _laplacian(pred_gray) - _laplacian(target_gray)
    ).abs()
    detail = detail + (1.0 - _ssim_map(pred_gray, target_gray)).clamp_min(0.0)
    per_image = _masked_per_image(detail, tissue).view(batch, scanners)
    return weighted_mean_with_coverage(per_image, weights)


def nuclei_masked_source_detail_loss(
    canonical: torch.Tensor,
    source_rgb: torch.Tensor,
    source_labels: torch.Tensor,
    weights: torch.Tensor,
    dilation: int = 3,
    laplacian_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve source geometry directly in canonical RGB around nuclei.

    The loss compares grayscale Sobel and Laplacian responses rather than RGB
    values, allowing stain appearance to change while discouraging smoothing or
    displacement of nuclear structure.  Only source StarDist masks are used;
    registered reference masks deliberately do not supervise this objective.
    """
    if canonical.shape != source_rgb.shape or canonical.ndim != 5:
        raise ValueError("canonical and source RGB must share shape (B, S, 3, H, W)")
    if source_labels.shape != canonical.shape[:2] + (1,) + canonical.shape[-2:]:
        raise ValueError("source labels must have shape (B, S, 1, H, W)")
    if weights.shape != canonical.shape[:2]:
        raise ValueError("weights must match the canonical batch/scanner axes")
    if dilation < 0:
        raise ValueError("nuclei mask dilation must be non-negative")
    if laplacian_weight < 0:
        raise ValueError("nuclei Laplacian weight must be non-negative")

    batch, scanners = canonical.shape[:2]
    prediction = canonical.reshape(batch * scanners, *canonical.shape[2:])
    source = source_rgb.reshape_as(prediction)
    labels = source_labels.reshape(batch * scanners, 1, *canonical.shape[-2:])
    mask = (labels > 0).to(prediction.dtype)
    if dilation > 0:
        kernel = 2 * dilation + 1
        mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilation)

    pred_gray, source_gray = rgb_to_gray(prediction), rgb_to_gray(source)
    pred_grad, source_grad = sobel_components(pred_gray), sobel_components(source_gray)
    detail = 0.5 * (
        (pred_grad[0] - source_grad[0]).abs()
        + (pred_grad[1] - source_grad[1]).abs()
    )
    detail = detail + float(laplacian_weight) * (
        _laplacian(pred_gray) - _laplacian(source_gray)
    ).abs()
    per_image = _masked_per_image(detail, mask).view(batch, scanners)
    contains_nuclei = mask.flatten(1).any(dim=1).view(batch, scanners)
    return weighted_mean_with_coverage(
        per_image, weights * contains_nuclei.to(weights.dtype)
    )


def nuclei_masked_source_orientation_loss(
    canonical: torch.Tensor,
    source_rgb: torch.Tensor,
    source_labels: torch.Tensor,
    weights: torch.Tensor,
    dilation: int = 3,
    smoothing_kernel: int = 7,
    smoothing_sigma: float = 1.25,
    magnitude_floor: float = 0.1,
    epsilon: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve local source orientation without matching scanner edge strength.

    Grayscale source and canonical images are Gaussian-smoothed before Sobel
    gradients are measured. Squared cosine makes the orientation unsigned, so a
    contrast-polarity reversal has zero cost. Detached source-gradient magnitude
    is used only as a soft confidence mask; it is never a reconstruction target.
    """
    if canonical.shape != source_rgb.shape or canonical.ndim != 5:
        raise ValueError("canonical and source RGB must share shape (B, S, 3, H, W)")
    if source_labels.shape != canonical.shape[:2] + (1,) + canonical.shape[-2:]:
        raise ValueError("source labels must have shape (B, S, 1, H, W)")
    if weights.shape != canonical.shape[:2]:
        raise ValueError("weights must match the canonical batch/scanner axes")
    if dilation < 0:
        raise ValueError("nuclei mask dilation must be non-negative")
    if magnitude_floor <= 0:
        raise ValueError("orientation magnitude floor must be positive")
    if epsilon <= 0:
        raise ValueError("orientation epsilon must be positive")

    batch, scanners = canonical.shape[:2]
    prediction = canonical.reshape(batch * scanners, *canonical.shape[2:])
    source = source_rgb.reshape_as(prediction)
    labels = source_labels.reshape(batch * scanners, 1, *canonical.shape[-2:])
    mask = (labels > 0).to(prediction.dtype)
    if dilation > 0:
        kernel = 2 * dilation + 1
        mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilation)

    pred_gray = gaussian_lowpass(
        rgb_to_gray(prediction), kernel_size=smoothing_kernel, sigma=smoothing_sigma
    )
    source_gray = gaussian_lowpass(
        rgb_to_gray(source), kernel_size=smoothing_kernel, sigma=smoothing_sigma
    )
    pred_x, pred_y = sobel_components(pred_gray)
    source_x, source_y = sobel_components(source_gray)
    pred_energy = pred_x.square() + pred_y.square()
    source_energy = source_x.square() + source_y.square()
    denominator = ((pred_energy + epsilon) * (source_energy + epsilon)).sqrt()
    cosine = ((pred_x * source_x + pred_y * source_y) / denominator).clamp(-1.0, 1.0)
    orientation = (1.0 - cosine.square()).clamp_min(0.0)

    source_magnitude = source_energy.detach().sqrt()
    confidence = source_magnitude / (source_magnitude + float(magnitude_floor))
    effective_mask = mask * confidence
    per_image = _masked_per_image(orientation, effective_mask).view(batch, scanners)
    contains_oriented_nuclei = (effective_mask.flatten(1).sum(dim=1) > 0).view(
        batch, scanners
    )
    return weighted_mean_with_coverage(
        per_image, weights * contains_oriented_nuclei.to(weights.dtype)
    )


def _instance_boundary(labels: torch.Tensor) -> torch.Tensor:
    """Return a one-pixel boundary map, including borders between touching nuclei."""
    if labels.ndim != 4 or labels.shape[1] != 1:
        raise ValueError("instance labels must have shape (N, 1, H, W)")
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    vertical = labels[:, :, 1:] != labels[:, :, :-1]
    horizontal = labels[:, :, :, 1:] != labels[:, :, :, :-1]
    boundary[:, :, 1:] |= vertical
    boundary[:, :, :-1] |= vertical
    boundary[:, :, :, 1:] |= horizontal
    boundary[:, :, :, :-1] |= horizontal
    return boundary & (labels > 0)


def _soft_mask_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image BCE plus soft Dice, robust to sparse foreground masks."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = _per_image_mean(bce)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).flatten(1).sum(dim=1)
    denominator = probability.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return bce + 0.5 * dice


def nuclei_reconstruction_loss(
    logits: torch.Tensor,
    source_labels: torch.Tensor,
    reference_labels: torch.Tensor,
    source_weights: torch.Tensor,
    reference_weights: torch.Tensor,
    boundary_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct StarDist nuclei occupancy and boundaries from canonical features.

    ``logits`` has occupancy and boundary channels for every canonical render.
    Source instance labels provide registration-free morphology supervision, while
    the registered AT2 labels provide a separately weighted canonical anchor.
    """
    if logits.ndim != 5 or logits.shape[2] != 2:
        raise ValueError("logits must have shape (B, S, 2, H, W)")
    if source_labels.shape != logits.shape[:2] + (1,) + logits.shape[-2:]:
        raise ValueError("source labels must match logits batch/scanner/spatial axes")
    if reference_labels.shape != logits.shape[:1] + (1,) + logits.shape[-2:]:
        raise ValueError("reference labels must have shape (B, 1, H, W)")
    if source_weights.shape != logits.shape[:2] or reference_weights.shape != logits.shape[:2]:
        raise ValueError("nuclei weights must match logits batch/scanner axes")

    batch, scanners = logits.shape[:2]
    flat_logits = logits.reshape(batch * scanners, 2, *logits.shape[-2:])
    flat_source = source_labels.reshape(batch * scanners, 1, *logits.shape[-2:])
    source_occupancy = (flat_source > 0).to(flat_logits.dtype)
    source_boundary = _instance_boundary(flat_source).to(flat_logits.dtype)
    source_error = _soft_mask_loss(flat_logits[:, :1], source_occupancy)
    source_error = source_error + float(boundary_weight) * _soft_mask_loss(
        flat_logits[:, 1:], source_boundary
    )
    source_value, source_coverage = weighted_mean_with_coverage(
        source_error.view(batch, scanners), source_weights
    )

    expanded_reference = reference_labels.unsqueeze(1).expand(
        -1, scanners, -1, -1, -1
    ).reshape_as(flat_source)
    reference_occupancy = (expanded_reference > 0).to(flat_logits.dtype)
    reference_boundary = _instance_boundary(expanded_reference).to(flat_logits.dtype)
    reference_error = _soft_mask_loss(flat_logits[:, :1], reference_occupancy)
    reference_error = reference_error + float(boundary_weight) * _soft_mask_loss(
        flat_logits[:, 1:], reference_boundary
    )
    reference_value, reference_coverage = weighted_mean_with_coverage(
        reference_error.view(batch, scanners), reference_weights
    )
    coverage = 0.5 * (source_coverage + reference_coverage)
    return source_value + reference_value, source_value, reference_value, coverage


def coordinate_variance_loss(
    content: torch.Tensor,
    standard_weights: torch.Tensor,
    slide_group: torch.Tensor,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve content variation across coordinates, never across scanners."""
    scanner_denominator = standard_weights.sum(dim=1, keepdim=True)
    valid_coordinate = scanner_denominator.squeeze(1) > 0
    safe_denominator = torch.where(
        scanner_denominator > 0,
        scanner_denominator,
        torch.ones_like(scanner_denominator),
    )
    scanner_mean = (
        content * standard_weights[:, :, None, None, None]
    ).sum(dim=1) / safe_denominator[:, :, None, None]
    vectors = scanner_mean.mean(dim=(2, 3))

    terms = []
    for group in torch.unique(slide_group):
        selected = (slide_group == group) & valid_coordinate
        if int(selected.sum()) < 2:
            continue
        std = vectors[selected].std(dim=0, unbiased=False)
        terms.append(torch.relu(gamma - std).square().mean())
    if not terms:
        return content.sum() * 0.0, content.new_zeros(())
    return torch.stack(terms).mean(), content.new_tensor(len(terms) / max(int(torch.unique(slide_group).numel()), 1))


def prototype_regression_loss(
    estimated: torch.Tensor,
    prototypes: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regress set-style estimates onto detached translated scanner prototypes."""
    mse = (estimated - prototypes.detach()).square().mean(dim=-1)
    return weighted_mean_with_coverage(mse, mask.to(mse.dtype))


def sample_translation_pairs(
    present: torch.Tensor,
    cross_weights: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one self pair and one ordered cross pair per coordinate when available.

    Returns source indices, target indices, pair weights, and a boolean identifying
    self pairs.  Coordinates without candidates simply contribute no sampled row.
    """
    source_indices: list[int] = []
    target_indices: list[int] = []
    batch_indices: list[int] = []
    pair_weights: list[torch.Tensor] = []
    is_self: list[bool] = []

    for batch_index in range(present.shape[0]):
        self_candidates = torch.nonzero(present[batch_index], as_tuple=False).flatten()
        if self_candidates.numel():
            pick = torch.randint(self_candidates.numel(), (1,), generator=generator).item()
            scanner = int(self_candidates[pick])
            batch_indices.append(batch_index)
            source_indices.append(scanner)
            target_indices.append(scanner)
            pair_weights.append(present.new_tensor(1.0, dtype=torch.float32))
            is_self.append(True)

        cross_candidates = torch.nonzero(cross_weights[batch_index] > 0, as_tuple=False)
        if cross_candidates.numel():
            pick = torch.randint(cross_candidates.shape[0], (1,), generator=generator).item()
            source, target = (int(v) for v in cross_candidates[pick])
            batch_indices.append(batch_index)
            source_indices.append(source)
            target_indices.append(target)
            pair_weights.append(cross_weights[batch_index, source, target])
            is_self.append(False)

    device = present.device
    return (
        torch.tensor(batch_indices, device=device, dtype=torch.long),
        torch.tensor(source_indices, device=device, dtype=torch.long),
        torch.tensor(target_indices, device=device, dtype=torch.long),
        torch.stack(pair_weights).to(device) if pair_weights else torch.empty(0, device=device),
        torch.tensor(is_self, device=device, dtype=torch.bool),
    )
