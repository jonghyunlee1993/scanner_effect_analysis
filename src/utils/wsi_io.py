"""WSI access. All reads are at level 0 with a fixed patch size (openslide mpp
metadata is unreliable, so we never trust it for resolution)."""
import numpy as np
import openslide


def open_wsi(path):
    return openslide.OpenSlide(str(path))


def read_patch(slide, x, y, size):
    """Read an (size x size) RGB uint8 patch at level-0 top-left (x, y).
    read_region returns RGBA; drop the alpha channel."""
    region = slide.read_region((int(x), int(y)), 0, (int(size), int(size)))
    return np.asarray(region.convert("RGB"), dtype=np.uint8)
