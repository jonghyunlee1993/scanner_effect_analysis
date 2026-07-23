"""Generator G(content, z): AdaIN residual blocks -> AdaIN upsampling -> RGB in [-1,1].

z is mapped to per-block AdaIN (gamma, beta) and injected at BOTH the residual blocks
and every upsampling stage (multi-scale style control -- a more expressive style path
than modulating the res blocks alone). Upsampling restores resolution so output H, W
match the content encoder's input.
"""
import torch.nn as nn

from .adain import AdaIN, StyleMLP


class AdaINResBlock(nn.Module):
    """Residual block with two AdaIN-normalized conv layers."""

    def __init__(self, dim):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, 3)
        self.adain1 = AdaIN()
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, 3)
        self.adain2 = AdaIN()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g1, b1, g2, b2):
        h = self.relu(self.adain1(self.conv1(self.pad1(x)), g1, b1))
        h = self.adain2(self.conv2(self.pad2(h)), g2, b2)
        return x + h


class AdaINUpBlock(nn.Module):
    """Nearest upsample -> conv -> AdaIN(style) -> ReLU (style-modulated upsampling)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_dim, out_dim, 3)
        self.adain = AdaIN()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, gamma, beta):
        x = self.conv(self.pad(self.up(x)))
        return self.relu(self.adain(x, gamma, beta))


class Generator(nn.Module):
    def __init__(
        self,
        content_ch=256,
        style_dim=32,
        ngf=64,
        n_res=4,
        n_up=2,
        nuclei_head=False,
    ):
        super().__init__()
        self.n_res = n_res
        self.n_up = n_up
        self.res_blocks = nn.ModuleList([AdaINResBlock(content_ch) for _ in range(n_res)])
        # Two AdaIN layers per residual block, all at content_ch channels.
        self.res_mlp = StyleMLP(style_dim, n_layers=2 * n_res, channels=content_ch)

        dims = [content_ch // (2 ** i) for i in range(n_up + 1)]
        dims[-1] = ngf  # final upsample stage lands on ngf
        self.up_blocks = nn.ModuleList([AdaINUpBlock(dims[i], dims[i + 1]) for i in range(n_up)])
        # Per-upsample-stage (gamma, beta) predictor from z.
        self.up_mlp = nn.ModuleList([nn.Linear(style_dim, dims[i + 1] * 2) for i in range(n_up)])

        self.out = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7), nn.Tanh())
        self.nuclei_head = None
        if nuclei_head:
            # Occupancy and instance-boundary logits share the final full-resolution
            # renderer features. The head is supervised by offline StarDist labels
            # and is not required as an input at inference time.
            self.nuclei_head = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(ngf, ngf // 2, 3),
                nn.ReLU(inplace=True),
                nn.Conv2d(ngf // 2, 2, 1),
            )

    def _decode_features(self, content, z):
        """Decode content and style into the full-resolution renderer feature map."""
        params = self.res_mlp(z)  # (N, 2*n_res, 2, content_ch)
        x = content
        for i, block in enumerate(self.res_blocks):
            g1, b1 = params[:, 2 * i, 0], params[:, 2 * i, 1]
            g2, b2 = params[:, 2 * i + 1, 0], params[:, 2 * i + 1, 1]
            x = block(x, g1, b1, g2, b2)
        for i, block in enumerate(self.up_blocks):
            gb = self.up_mlp[i](z)             # (N, 2*out_dim)
            c = gb.shape[1] // 2
            x = block(x, gb[:, :c], gb[:, c:])
        return x

    def forward(self, content, z):
        """content (N, C, h, w), z (N, D) -> RGB in ``[-1, 1]``."""
        return self.out(self._decode_features(content, z))

    def forward_with_nuclei(self, content, z):
        """Return RGB plus full-resolution nuclei occupancy/boundary logits."""
        if self.nuclei_head is None:
            raise RuntimeError("nuclei head is not enabled for this generator")
        features = self._decode_features(content, z)
        return self.out(features), self.nuclei_head(features)
