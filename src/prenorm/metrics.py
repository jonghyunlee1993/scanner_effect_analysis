"""Image and representation metrics used without a trainable pathology model."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure

from .losses import rgb_to_gray


def ssim(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Return mean RGB structural similarity for NCHW images."""
    return structural_similarity_index_measure(prediction, target, data_range=data_range)


def masked_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 2.0,
) -> torch.Tensor:
    """Average the SSIM map inside a soft foreground mask."""
    _, similarity = structural_similarity_index_measure(
        prediction,
        target,
        data_range=data_range,
        return_full_image=True,
    )
    similarity = similarity.mean(dim=1, keepdim=True)
    denominator = mask.sum()
    if not bool(denominator.detach() > 0):
        return similarity.sum() * 0.0
    return (similarity * mask).sum() / denominator


def standard_convergence(canonical: torch.Tensor, geom_ok: torch.Tensor) -> torch.Tensor:
    """Measure pairwise canonical SSIM for registered usable scanner views."""
    values = []
    for batch_index in range(canonical.shape[0]):
        for source in range(canonical.shape[1]):
            for target in range(source + 1, canonical.shape[1]):
                if bool(geom_ok[batch_index, source] and geom_ok[batch_index, target]):
                    values.append(
                        ssim(
                            canonical[batch_index, source][None],
                            canonical[batch_index, target][None],
                        )
                    )
    return torch.stack(values).mean() if values else canonical.sum() * 0.0


def focus_score(rgb: torch.Tensor) -> torch.Tensor:
    """Return per-image Laplacian variance on fixed luminance."""
    gray = rgb_to_gray(rgb)
    kernel = gray.new_tensor(((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0)))
    laplacian = F.conv2d(gray, kernel.view(1, 1, 3, 3), padding=1)
    return laplacian.flatten(1).var(dim=1, unbiased=False)


def effective_rank(features: torch.Tensor) -> torch.Tensor:
    """Compute entropy effective rank for a batch of feature vectors."""
    centered = features - features.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered.float())
    total = singular_values.sum()
    if not bool(total.detach() > 0):
        return features.new_zeros(())
    probability = singular_values / total
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return entropy.exp().to(features.dtype)
