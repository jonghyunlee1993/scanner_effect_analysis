"""Differentiable RGB-to-content input construction."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FULL_CHANNELS = ("r", "g", "b", "edge_lp", "gray", "h", "e", "background")
STRUCTURE_CHANNELS = ("edge_lp", "gray", "h", "e", "background")
GROUPS = ("rgb", "edge", "gray", "stain", "background")


class DifferentiableInputBuilder(nn.Module):
    """Build fixed RGB and structure channels while preserving RGB gradients.

    The stain artifact must contain a ``matrix`` with shape ``(3, 2)`` and a
    two-element ``concentration_scale``. A group mask follows ``GROUPS`` order;
    a mask without a scanner axis is broadcast across a registered tuple.
    """

    def __init__(
        self,
        stain_reference_path,
        mode="full",
        edge_sigma=1.5,
        group_dropout_p=0.3,
        od_epsilon=1.0 / 255.0,
        background_brightness=0.8,
        background_saturation=0.2,
        background_temperature=0.05,
    ):
        super().__init__()
        if mode not in {"full", "structure"}:
            raise ValueError(f"unknown input mode: {mode}")

        with np.load(stain_reference_path, allow_pickle=False) as artifact:
            stain_matrix = torch.as_tensor(artifact["matrix"], dtype=torch.float32)
            concentration_scale = torch.as_tensor(
                artifact["concentration_scale"], dtype=torch.float32
            )
        if stain_matrix.shape != (3, 2) or concentration_scale.shape != (2,):
            raise ValueError("stain artifact requires matrix (3, 2) and concentration_scale (2,)")

        # The pseudoinverse is a fixed OD-space 1x1 convolution.
        stain_projection = torch.linalg.pinv(stain_matrix)
        self.register_buffer("stain_matrix", stain_matrix)
        self.register_buffer("stain_projection", stain_projection[:, :, None, None])
        self.register_buffer("concentration_scale", concentration_scale[None, :, None, None])
        self.register_buffer("gaussian_kernel", self._gaussian_kernel(edge_sigma))
        self.register_buffer("sobel_kernel", self._sobel_kernel())

        self.mode = mode
        self.edge_sigma = float(edge_sigma)
        self.group_dropout_p = float(group_dropout_p)
        self.od_epsilon = float(od_epsilon)
        self.background_brightness = float(background_brightness)
        self.background_saturation = float(background_saturation)
        self.background_temperature = float(background_temperature)

    @staticmethod
    def _gaussian_kernel(sigma):
        radius = int(math.ceil(3.0 * sigma))
        coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
        kernel_1d /= kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d[None, None]

    @staticmethod
    def _sobel_kernel():
        return torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
             [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            dtype=torch.float32,
        )[:, None] / 8.0

    @property
    def channel_names(self):
        """Return output channel names in tensor order."""
        return FULL_CHANNELS if self.mode == "full" else STRUCTURE_CHANNELS

    @property
    def out_channels(self):
        """Return the configured output channel count."""
        return len(self.channel_names)

    def sample_group_mask(self, tuple_count, device=None):
        """Sample one dropout mask per tuple, preserving a structure cue."""
        keep = torch.rand(int(tuple_count), len(GROUPS), device=device) >= self.group_dropout_p
        missing_structure = ~keep[:, (1, 2, 4)].any(dim=1)
        fallback = torch.tensor((1, 2, 4), device=device)[
            torch.randint(3, (int(missing_structure.sum()),), device=device)
        ]
        keep[missing_structure, fallback] = True
        return keep

    def _blur(self, rgb):
        radius = self.gaussian_kernel.shape[-1] // 2
        kernel = self.gaussian_kernel.to(dtype=rgb.dtype).expand(3, 1, -1, -1)
        return F.conv2d(F.pad(rgb, (radius,) * 4, mode="reflect"), kernel, groups=3)

    def _edge(self, rgb):
        blurred = self._blur(rgb)
        gray = self._luminance(blurred)
        gradients = F.conv2d(
            F.pad(gray, (1, 1, 1, 1), mode="reflect"),
            self.sobel_kernel.to(dtype=rgb.dtype),
        )
        magnitude = torch.sqrt(gradients.square().sum(dim=1, keepdim=True) + 1e-12)
        return magnitude.clamp(0.0, 1.0)

    @staticmethod
    def _luminance(rgb):
        weights = rgb.new_tensor((0.299, 0.587, 0.114))[None, :, None, None]
        return (rgb * weights).sum(dim=1, keepdim=True)

    def _stains(self, rgb):
        rgb_unit = (rgb + 1.0) * 0.5
        optical_density = -torch.log(rgb_unit.clamp_min(self.od_epsilon))
        projection = self.stain_projection.to(dtype=rgb.dtype)
        scale = self.concentration_scale.to(dtype=rgb.dtype)
        concentrations = F.relu(F.conv2d(optical_density, projection)) / scale
        return concentrations.clamp(0.0, 1.0)

    def _background(self, rgb):
        rgb_unit = (rgb + 1.0) * 0.5
        maximum = rgb_unit.amax(dim=1, keepdim=True)
        minimum = rgb_unit.amin(dim=1, keepdim=True)
        saturation = (maximum - minimum) / (maximum + self.od_epsilon)
        temperature = self.background_temperature
        bright = torch.sigmoid((maximum - self.background_brightness) / temperature)
        neutral = torch.sigmoid((self.background_saturation - saturation) / temperature)
        return bright * neutral

    @staticmethod
    def _broadcast_group_mask(group_mask, leading_shape, dtype, device):
        mask = torch.as_tensor(group_mask, dtype=dtype, device=device)
        if mask.shape[-1] != len(GROUPS):
            raise ValueError(f"group_mask must follow {GROUPS}")
        # (B, 5) is shared by all S scanners in an RGB tensor shaped (B, S, 3, H, W).
        if leading_shape and tuple(mask.shape[:-1]) == tuple(leading_shape[:-1]):
            mask = mask.unsqueeze(-2)
        return torch.broadcast_to(mask, tuple(leading_shape) + (len(GROUPS),))

    def forward(self, rgb, mode=None, group_mask=None):
        """Build full or structure-only channels from RGB in ``[-1, 1]``.

        ``rgb`` may have any leading dimensions before ``(3, H, W)``. For
        registered ``(B, S, 3, H, W)`` tuples, a ``(B, 5)`` group mask is
        automatically shared over the scanner dimension.
        """
        selected_mode = self.mode if mode is None else mode
        if selected_mode not in {"full", "structure"}:
            raise ValueError(f"unknown input mode: {selected_mode}")
        if rgb.shape[-3] != 3:
            raise ValueError("rgb must end with (3, H, W)")

        leading_shape = rgb.shape[:-3]
        height, width = rgb.shape[-2:]
        flat_rgb = rgb.reshape(-1, 3, height, width)
        edge = self._edge(flat_rgb)
        gray = ((self._luminance(flat_rgb) + 1.0) * 0.5).clamp(0.0, 1.0)
        stains = self._stains(flat_rgb)
        background = self._background(flat_rgb)
        full = torch.cat((flat_rgb, edge, gray, stains, background), dim=1)

        if group_mask is not None:
            mask = self._broadcast_group_mask(group_mask, leading_shape, full.dtype, full.device)
            mask = mask.reshape(-1, len(GROUPS))
            channel_mask = torch.cat(
                (mask[:, 0:1].expand(-1, 3), mask[:, 1:3],
                 mask[:, 3:4].expand(-1, 2), mask[:, 4:5]),
                dim=1,
            )
            full = full * channel_mask[:, :, None, None]

        output = full if selected_mode == "full" else full[:, 3:]
        return output.reshape(*leading_shape, output.shape[1], height, width)
