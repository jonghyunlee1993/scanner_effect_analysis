"""Foundation-model embedding utilities for scanner-spectrum analyses."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score


def load_uni(device: torch.device):
    """Load the sealed UNI evaluator from the local Hugging Face cache."""
    import timm

    model = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    ).eval().to(device)
    pretrained = model.pretrained_cfg
    size = pretrained["input_size"][-1]
    mean = torch.tensor(pretrained["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(pretrained["std"], device=device).view(1, 3, 1, 1)
    return model, size, mean, std


@torch.no_grad()
def embed_uni(model, images, size, mean, std, device, batch_size=32):
    """Return normalized UNI embeddings for RGB tensors in ``[-1, 1]``."""
    embeddings = []
    for start in range(0, len(images), batch_size):
        image = images[start : start + batch_size].to(device)
        image = F.interpolate(
            (image.clamp(-1, 1) + 1) * 0.5,
            size=(size, size),
            mode="bicubic",
            align_corners=False,
        )
        embeddings.append(model((image - mean) / std).float().cpu())
    return F.normalize(torch.cat(embeddings), dim=1)


def scanner_probe(embedding, scanner_id, location_id):
    """Return location-grouped five-fold balanced scanner accuracy."""
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    classes = np.unique(scanner_id)
    group_count = len(np.unique(location_id))
    folds = min(5, group_count)
    if len(classes) < 2 or folds < 2:
        return float("nan")
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=1234,
    )
    return float(
        cross_val_score(
            classifier,
            embedding,
            scanner_id,
            groups=location_id,
            cv=splitter,
            scoring="balanced_accuracy",
        ).mean()
    )


def same_location_cosine(embedding, location_id):
    """Average cosine similarity across scanners at each registered location."""
    values = []
    for location in np.unique(location_id):
        selected = np.flatnonzero(location_id == location)
        for offset, left in enumerate(selected):
            for right in selected[offset + 1 :]:
                values.append(float((embedding[left] * embedding[right]).sum()))
    return float(np.mean(values)) if values else float("nan")


def scanner_purity_excluding_location(
    embedding,
    location_id,
    scanner_id,
    k=5,
    chunk_size=512,
):
    """Scanner purity after removing all registered same-location counterparts."""
    location = torch.as_tensor(location_id)
    scanner = torch.as_tensor(scanner_id)
    values = []
    for start in range(0, len(embedding), chunk_size):
        stop = min(start + chunk_size, len(embedding))
        similarity = embedding[start:stop] @ embedding.t()
        same_location = location[start:stop, None] == location[None, :]
        similarity = similarity.masked_fill(same_location, -torch.inf)
        available = (~same_location).sum(dim=1).min().item()
        if available < 1:
            continue
        neighbors = similarity.topk(min(k, int(available)), dim=1).indices
        values.append(
            (scanner[neighbors] == scanner[start:stop, None]).float().mean(dim=1)
        )
    return float(torch.cat(values).mean()) if values else float("nan")
