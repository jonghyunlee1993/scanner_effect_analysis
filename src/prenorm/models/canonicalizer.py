"""Prototype-only registration-supervised canonicalizer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .content_encoder import ContentEncoder
from .generator import Generator
from .input_builder import DifferentiableInputBuilder
from .style_encoder import SetStyleEncoder


class Canonicalizer(nn.Module):
    """Own the differentiable input, content, renderer, and scanner prototypes."""

    def __init__(
        self,
        scanners,
        reference_scanner,
        stain_reference_path,
        input_mode="full",
        content_ch=256,
        style_dim=64,
        ngf=64,
        n_downsample=2,
        n_res=4,
        canonical_skip="none",
        canonical_skip_kernel=9,
        nuclei_head=False,
    ):
        super().__init__()
        self.scanners = tuple(scanners)
        self.reference_scanner = reference_scanner
        self.reference_index = self.scanners.index(reference_scanner)
        self.input_mode = input_mode
        self.canonical_skip = str(canonical_skip)
        self.canonical_skip_kernel = int(canonical_skip_kernel)
        if self.canonical_skip not in {"none", "low_frequency_residual"}:
            raise ValueError(f"unsupported canonical skip mode: {self.canonical_skip}")
        if self.canonical_skip_kernel < 1 or self.canonical_skip_kernel % 2 == 0:
            raise ValueError("canonical_skip_kernel must be a positive odd integer")

        self.input_builder = DifferentiableInputBuilder(stain_reference_path, mode=input_mode)
        self.content_encoder = ContentEncoder(
            self.input_builder.out_channels, content_ch, n_downsample, n_res
        )
        self.generator = Generator(
            content_ch, style_dim, ngf, n_res, n_downsample, nuclei_head=nuclei_head
        )
        self.style_encoder = SetStyleEncoder(3, style_dim, ngf)
        self.prototypes = nn.Parameter(torch.empty(len(self.scanners), style_dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

    @property
    def M(self):
        """Alias for the differentiable input builder."""
        return self.input_builder

    @property
    def C(self):
        """Alias for the spatial content encoder."""
        return self.content_encoder

    @property
    def G(self):
        """Alias for the prototype-conditioned renderer."""
        return self.generator

    @property
    def B(self):
        """Alias for the learnable scanner prototype table."""
        return self.prototypes

    @property
    def E_set(self):
        """Alias for the clean-context set style encoder."""
        return self.style_encoder

    def encode(self, rgb, group_mask=None):
        """Encode an RGB batch into a spatial content map."""
        inputs = self.input_builder(rgb, group_mask=group_mask)
        leading_shape = inputs.shape[:-3]
        flat = inputs.reshape(-1, *inputs.shape[-3:])
        content = self.content_encoder(flat)
        return content.reshape(*leading_shape, *content.shape[-3:])

    def decode(self, content, target_idx):
        """Render content with one target prototype per batch element."""
        target_idx = torch.as_tensor(target_idx, dtype=torch.long, device=content.device)
        if target_idx.ndim == 0:
            target_idx = target_idx.expand(content.shape[0])
        if target_idx.shape != (content.shape[0],):
            raise ValueError("target_idx must be scalar or have one entry per content map")
        return self.generator(content, self.prototypes[target_idx])

    def canonicalize(self, rgb):
        """Render RGB through the fixed reference-scanner prototype only."""
        return self.decode_canonical(self.encode(rgb), rgb)

    def decode_canonical(self, content, source_rgb):
        """Decode AT2 style, optionally retaining source high-frequency detail.

        The residual skip permits only a low-frequency change from the source:
        ``source + LP(decoded - source)``.  It therefore tests whether the shared
        encoder/decoder bottleneck, rather than the supervision, causes blur.
        """
        decoded = self.decode(content, self.reference_index)
        if self.canonical_skip == "none":
            return decoded
        radius = self.canonical_skip_kernel // 2
        residual = decoded - source_rgb
        residual = F.avg_pool2d(
            F.pad(residual, (radius, radius, radius, radius), mode="reflect"),
            kernel_size=self.canonical_skip_kernel,
            stride=1,
        )
        return (source_rgb + residual).clamp(-1.0, 1.0)

    def decode_canonical_with_nuclei(self, content, source_rgb):
        """Decode canonical RGB and StarDist-supervised nuclei logits together."""
        style = self.prototypes[self.reference_index].expand(content.shape[0], -1)
        decoded, nuclei_logits = self.generator.forward_with_nuclei(content, style)
        if self.canonical_skip == "none":
            return decoded, nuclei_logits
        radius = self.canonical_skip_kernel // 2
        residual = F.avg_pool2d(
            F.pad(decoded - source_rgb, (radius,) * 4, mode="reflect"),
            kernel_size=self.canonical_skip_kernel,
            stride=1,
        )
        return (source_rgb + residual).clamp(-1.0, 1.0), nuclei_logits

    def render(self, rgb, target_idx):
        """Render RGB into the requested scanner prototype."""
        return self.decode(self.encode(rgb), target_idx)

    def forward(self, rgb):
        """Canonicalize RGB for standard inference calls."""
        return self.canonicalize(rgb)
