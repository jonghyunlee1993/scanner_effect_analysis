"""CPU transforms for clean registered RGB tuples."""

import numpy as np
import torch


def tuple_geometric_aug(arrays, rng, ops):
    """Apply one shared flip/rotation draw to every RGB image in a tuple."""
    do_flip = "flip" in ops and bool(rng.integers(0, 2))
    rotations = int(rng.integers(0, 4)) if "rot90" in ops else 0

    transformed = []
    for array in arrays:
        if do_flip:
            array = array[:, ::-1]
        if rotations:
            array = np.rot90(array, rotations)
        transformed.append(np.ascontiguousarray(array))
    return transformed


def rgb_to_tensor(rgb_uint8):
    """Convert an HWC uint8 RGB image to CHW float RGB in ``[-1, 1]``."""
    rgb = torch.from_numpy(rgb_uint8.astype(np.float32)).permute(2, 0, 1)
    return rgb / 127.5 - 1.0
