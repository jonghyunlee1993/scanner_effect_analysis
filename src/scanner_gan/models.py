"""Canonical, dependency-free PyTorch modules for Pix2Pix.

The generator maps normalized ``[-1, 1]`` images to ``[-1, 1]`` with a final
``tanh``.  The discriminator is conditional: concatenate source and target
images along the channel dimension before calling it (six channels for RGB).
"""

from __future__ import annotations

from functools import partial
import random
from typing import Callable, Iterable

import torch
from torch import Tensor, nn


NormFactory = Callable[[int], nn.Module]


def _uses_bias(norm_layer: NormFactory) -> bool:
    """Return whether convolutions should keep a bias before normalization."""

    if isinstance(norm_layer, partial):
        return norm_layer.func is nn.InstanceNorm2d
    return norm_layer is nn.InstanceNorm2d


def _validate_image_tensor(
    image: Tensor,
    *,
    channels: int,
    name: str,
    spatial_multiple: int | None = None,
) -> None:
    """Fail early on image tensors that cannot safely traverse the network."""

    if image.ndim != 4:
        raise ValueError(f"{name} must have shape (N, C, H, W); got {tuple(image.shape)}")
    if image.shape[1] != channels:
        raise ValueError(
            f"{name} must have {channels} channels; got shape {tuple(image.shape)}"
        )
    if not torch.is_floating_point(image):
        raise TypeError(f"{name} must be a floating-point tensor; got {image.dtype}")
    height, width = image.shape[-2:]
    if height < 1 or width < 1:
        raise ValueError(f"{name} must have non-empty spatial dimensions")
    if spatial_multiple is not None and (
        height < spatial_multiple
        or width < spatial_multiple
        or height % spatial_multiple
        or width % spatial_multiple
    ):
        raise ValueError(
            f"{name} spatial dimensions must be at least and divisible by "
            f"{spatial_multiple}; got {height}x{width}"
        )


class UnetSkipConnectionBlock(nn.Module):
    """One recursive U-Net block used by :class:`UnetGenerator`."""

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        *,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer: NormFactory = nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        if outer_nc <= 0 or inner_nc <= 0:
            raise ValueError("outer_nc and inner_nc must be positive")
        if outermost and innermost:
            raise ValueError("a U-Net block cannot be both outermost and innermost")
        input_nc = outer_nc if input_nc is None else input_nc
        if input_nc <= 0:
            raise ValueError("input_nc must be positive")
        if not innermost and submodule is None:
            raise ValueError("non-innermost blocks require a submodule")

        use_bias = _uses_bias(norm_layer)
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        uprelu = nn.ReLU(inplace=True)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            layers: list[nn.Module] = [downconv]
            assert submodule is not None
            layers.extend([submodule, uprelu, upconv, nn.Tanh()])
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            layers = [downrelu, downconv, uprelu, upconv, norm_layer(outer_nc)]
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            layers = [
                downrelu,
                downconv,
                norm_layer(inner_nc),
                submodule,
                uprelu,
                upconv,
                norm_layer(outer_nc),
            ]
            if use_dropout:
                layers.append(nn.Dropout(0.5))

        self.outermost = outermost
        self.model = nn.Sequential(*layers)

    def forward(self, image: Tensor) -> Tensor:
        transformed = self.model(image)
        if self.outermost:
            return transformed
        if transformed.shape[0] != image.shape[0] or transformed.shape[2:] != image.shape[2:]:
            raise RuntimeError(
                "U-Net skip shapes differ: "
                f"input={tuple(image.shape)}, transformed={tuple(transformed.shape)}"
            )
        return torch.cat((image, transformed), dim=1)


class UnetGenerator(nn.Module):
    """Canonical Pix2Pix U-Net generator; ``num_downs=8`` is U-Net-256."""

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        num_downs: int = 8,
        ngf: int = 64,
        norm_layer: NormFactory = nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        if input_nc <= 0 or output_nc <= 0 or ngf <= 0:
            raise ValueError("input_nc, output_nc, and ngf must be positive")
        if num_downs < 5:
            raise ValueError("num_downs must be at least 5")

        block: nn.Module = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            innermost=True,
            norm_layer=norm_layer,
        )
        for _ in range(num_downs - 5):
            block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                submodule=block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, submodule=block, norm_layer=norm_layer
        )
        block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, submodule=block, norm_layer=norm_layer
        )
        block = UnetSkipConnectionBlock(
            ngf, ngf * 2, submodule=block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=block,
            outermost=True,
            norm_layer=norm_layer,
        )
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.num_downs = num_downs
        self.spatial_multiple = 2**num_downs

    def forward(self, image: Tensor) -> Tensor:
        _validate_image_tensor(
            image,
            channels=self.input_nc,
            name="generator input",
            spatial_multiple=self.spatial_multiple,
        )
        output = self.model(image)
        expected = (image.shape[0], self.output_nc, *image.shape[-2:])
        if output.shape != expected:
            raise RuntimeError(
                f"generator output shape mismatch: expected {expected}, got {tuple(output.shape)}"
            )
        return output


class NLayerDiscriminator(nn.Module):
    """Conditional PatchGAN discriminator; ``n_layers=3`` has a 70x70 field."""

    def __init__(
        self,
        input_nc: int = 6,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: NormFactory = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if input_nc <= 0 or ndf <= 0:
            raise ValueError("input_nc and ndf must be positive")
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        kernel_size = 4
        padding = 1
        sequence: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf_mult = 1
        for layer in range(1, n_layers):
            previous = nf_mult
            nf_mult = min(2**layer, 8)
            sequence.extend(
                [
                    nn.Conv2d(
                        ndf * previous,
                        ndf * nf_mult,
                        kernel_size,
                        stride=2,
                        padding=padding,
                        bias=_uses_bias(norm_layer),
                    ),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        previous = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence.extend(
            [
                nn.Conv2d(
                    ndf * previous,
                    ndf * nf_mult,
                    kernel_size,
                    stride=1,
                    padding=padding,
                    bias=_uses_bias(norm_layer),
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * nf_mult, 1, kernel_size, stride=1, padding=padding),
            ]
        )
        self.model = nn.Sequential(*sequence)
        self.input_nc = input_nc
        self.n_layers = n_layers

    def forward(self, image_pair: Tensor) -> Tensor:
        _validate_image_tensor(
            image_pair,
            channels=self.input_nc,
            name="discriminator input",
        )
        minimum = 3 * 2**self.n_layers
        if min(image_pair.shape[-2:]) < minimum:
            raise ValueError(
                f"discriminator input must be at least {minimum}px per side for "
                f"n_layers={self.n_layers}; got {tuple(image_pair.shape[-2:])}"
            )
        return self.model(image_pair)


class ResnetBlock(nn.Module):
    """Residual block used by the canonical 256px CycleGAN generator."""

    def __init__(
        self,
        dim: int,
        *,
        norm_layer: NormFactory,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        use_bias = _uses_bias(norm_layer)
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.extend(
            [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
                norm_layer(dim),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, image: Tensor) -> Tensor:
        return image + self.block(image)


class ResnetGenerator(nn.Module):
    """ResNet-9 image translator used by CycleGAN at 256px resolution."""

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
        norm_layer: NormFactory = partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        ),
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        if n_blocks < 0:
            raise ValueError("n_blocks must be non-negative")
        use_bias = _uses_bias(norm_layer)
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        for level in range(2):
            multiplier = 2**level
            layers.extend(
                [
                    nn.Conv2d(
                        ngf * multiplier,
                        ngf * multiplier * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(ngf * multiplier * 2),
                    nn.ReLU(True),
                ]
            )
        for _ in range(n_blocks):
            layers.append(
                ResnetBlock(
                    ngf * 4,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                )
            )
        for level in range(2):
            multiplier = 2 ** (2 - level)
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        ngf * multiplier,
                        ngf * multiplier // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(ngf * multiplier // 2),
                    nn.ReLU(True),
                ]
            )
        layers.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, output_nc, kernel_size=7),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)
        self.input_nc = int(input_nc)

    def forward(self, image: Tensor) -> Tensor:
        _validate_image_tensor(image, channels=self.input_nc, name="generator input")
        return self.model(image)


class ImagePool:
    """CycleGAN replay buffer with a private deterministic random stream."""

    def __init__(self, pool_size: int = 50, seed: int = 0) -> None:
        if pool_size < 0:
            raise ValueError("pool_size must be non-negative")
        self.pool_size = int(pool_size)
        self.images: list[Tensor] = []
        self.random = random.Random(int(seed))

    def query(self, images: Tensor) -> Tensor:
        if self.pool_size == 0:
            return images.detach()
        returned: list[Tensor] = []
        for image in images.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image.clone())
                returned.append(image)
            elif self.random.random() > 0.5:
                index = self.random.randrange(self.pool_size)
                historical = self.images[index].clone()
                self.images[index] = image.clone()
                returned.append(historical)
            else:
                returned.append(image)
        return torch.cat(returned, dim=0)


class GANLoss(nn.Module):
    """Adversarial loss accepting discriminator logits and a real/fake flag."""

    def __init__(
        self,
        gan_mode: str = "vanilla",
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(float(target_real_label)))
        self.register_buffer("fake_label", torch.tensor(float(target_fake_label)))
        if gan_mode == "lsgan":
            self.loss: nn.Module | None = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "wgangp":
            self.loss = None
        else:
            raise ValueError(
                f"unsupported gan_mode={gan_mode!r}; use 'vanilla', 'lsgan', or 'wgangp'"
            )
        self.gan_mode = gan_mode

    def get_target_tensor(self, prediction: Tensor, target_is_real: bool) -> Tensor:
        """Return a label tensor with prediction's shape, dtype, and device."""

        label = self.real_label if target_is_real else self.fake_label
        return label.to(dtype=prediction.dtype, device=prediction.device).expand_as(prediction)

    def forward(self, prediction: Tensor, target_is_real: bool) -> Tensor:
        if prediction.ndim < 1 or not torch.is_floating_point(prediction):
            raise ValueError("prediction must be a non-scalar floating-point tensor")
        if self.gan_mode in {"lsgan", "vanilla"}:
            assert self.loss is not None
            return self.loss(prediction, self.get_target_tensor(prediction, target_is_real))
        return -prediction.mean() if target_is_real else prediction.mean()


def init_weights(
    network: nn.Module,
    init_type: str = "normal",
    init_gain: float = 0.02,
) -> nn.Module:
    """Initialize convolutional/linear layers and return ``network``."""

    if init_gain <= 0:
        raise ValueError("init_gain must be positive")
    allowed = {"normal", "xavier", "kaiming", "orthogonal"}
    if init_type not in allowed:
        raise ValueError(f"unsupported init_type={init_type!r}; choose one of {sorted(allowed)}")

    def initialize(module: nn.Module) -> None:
        class_name = module.__class__.__name__
        if hasattr(module, "weight") and ("Conv" in class_name or "Linear" in class_name):
            weight = module.weight
            if init_type == "normal":
                nn.init.normal_(weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(weight.data, a=0, mode="fan_in")
            else:
                nn.init.orthogonal_(weight.data, gain=init_gain)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias.data, 0.0)
        elif "BatchNorm2d" in class_name and getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight.data, 1.0, init_gain)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias.data, 0.0)

    network.apply(initialize)
    return network


def init_net(
    network: nn.Module,
    *,
    device: torch.device | str | None = None,
    init_type: str = "normal",
    init_gain: float = 0.02,
) -> nn.Module:
    """Optionally move, then initialize a network without wrapping it."""

    if device is not None:
        network = network.to(torch.device(device))
    return init_weights(network, init_type=init_type, init_gain=init_gain)


def count_parameters(network: nn.Module, *, trainable_only: bool = True) -> int:
    """Count scalar parameters, optionally excluding frozen parameters."""

    parameters: Iterable[nn.Parameter] = network.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def build_pix2pix_models(
    *,
    input_nc: int = 3,
    output_nc: int = 3,
    ngf: int = 64,
    ndf: int = 64,
    use_dropout: bool = False,
    norm_layer: NormFactory = nn.BatchNorm2d,
    device: torch.device | str | None = None,
    init_type: str = "normal",
    init_gain: float = 0.02,
) -> tuple[UnetGenerator, NLayerDiscriminator]:
    """Construct and initialize canonical U-Net-256 and conditional PatchGAN."""

    generator = UnetGenerator(
        input_nc=input_nc,
        output_nc=output_nc,
        num_downs=8,
        ngf=ngf,
        norm_layer=norm_layer,
        use_dropout=use_dropout,
    )
    discriminator = NLayerDiscriminator(
        input_nc=input_nc + output_nc,
        ndf=ndf,
        n_layers=3,
        norm_layer=norm_layer,
    )
    init_net(generator, device=device, init_type=init_type, init_gain=init_gain)
    init_net(discriminator, device=device, init_type=init_type, init_gain=init_gain)
    return generator, discriminator


def build_cyclegan_models(
    *,
    input_nc: int = 3,
    output_nc: int = 3,
    ngf: int = 64,
    ndf: int = 64,
    n_blocks: int = 9,
    device: torch.device | str | None = None,
    init_type: str = "normal",
    init_gain: float = 0.02,
) -> tuple[ResnetGenerator, ResnetGenerator, NLayerDiscriminator, NLayerDiscriminator]:
    """Construct bidirectional ResNet-9 generators and unconditional PatchGANs."""

    norm_layer: NormFactory = partial(
        nn.InstanceNorm2d, affine=False, track_running_stats=False
    )
    generator_a_to_b = ResnetGenerator(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        n_blocks=n_blocks,
        norm_layer=norm_layer,
    )
    generator_b_to_a = ResnetGenerator(
        input_nc=output_nc,
        output_nc=input_nc,
        ngf=ngf,
        n_blocks=n_blocks,
        norm_layer=norm_layer,
    )
    discriminator_a = NLayerDiscriminator(
        input_nc=input_nc,
        ndf=ndf,
        n_layers=3,
        norm_layer=norm_layer,
    )
    discriminator_b = NLayerDiscriminator(
        input_nc=output_nc,
        ndf=ndf,
        n_layers=3,
        norm_layer=norm_layer,
    )
    networks = (
        generator_a_to_b,
        generator_b_to_a,
        discriminator_a,
        discriminator_b,
    )
    for network in networks:
        init_net(network, device=device, init_type=init_type, init_gain=init_gain)
    return networks


# Explicit descriptive aliases for the canonical configurations.
UNet256Generator = UnetGenerator
PatchGAN70Discriminator = NLayerDiscriminator


__all__ = [
    "GANLoss",
    "NLayerDiscriminator",
    "PatchGAN70Discriminator",
    "ResnetBlock",
    "ResnetGenerator",
    "UNet256Generator",
    "UnetGenerator",
    "UnetSkipConnectionBlock",
    "ImagePool",
    "build_cyclegan_models",
    "build_pix2pix_models",
    "count_parameters",
    "init_net",
    "init_weights",
]
