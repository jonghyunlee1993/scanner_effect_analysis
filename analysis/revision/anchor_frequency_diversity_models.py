#!/usr/bin/env python3
"""RV13 model loaders and embedding paths (library, no sbatch).

Nine embedding streams from seven encoders, each with its released preprocessing applied to
the same 256 x 256 field (0.5052 um/px). Every stream takes float RGB images in [0, 1]
(HWC, float32); uint8 patches enter as ``uint8 / 255`` exactly as in RV02/RV03, and the
unquantized band-manipulated renders of RV02 enter unchanged.

* ``conch_pre`` / ``conch_projected`` and ``seal_conch_pre`` / ``seal_conch_projected``:
  ``load_encoder`` / ``embed`` of ``scripts/features/review_multiencoder_scanner.py`` (the
  producer of ``outputs/encoder_review_2026-09-25``): bicubic 256 -> 448, clamp, OpenAI CLIP
  normalization, fp16 autocast, ``visual.forward_no_head`` (pre-projection) and its product
  with ``proj_contrast`` (projected), both from one forward pass. SEAL-CONCH merges the
  released LoRA (``seal_conch_vision.pth``) into CONCH's last visual block.
* ``seal_uni2``: UNI2-h (``load_encoder("uni2")``) with the released SEAL LoRA
  (``seal_univ2_vision.pth``) merged by ``merge_seal_lora``, embedded through the review
  ``embed`` with ``name="uni2"`` (bicubic 224, bf16) -- the same path that produced the RV03
  UNI2-h embeddings, so SEAL-UNI2 - UNI2-h differs only in the weights.
* ``plip``: ``vinid/plip`` ``CLIPModel.get_image_features`` (CLIP image projection). The
  released ``CLIPImageProcessor`` (resize shortest side 224 bicubic, centre crop 224, CLIP
  normalization) is reproduced on float tensors: antialiased bicubic (a = -0.5, PIL's
  kernel) 256 -> 224, centre crop, normalization. fp32.
* ``dinov2``: ``facebook/dinov2-large`` ``Dinov2Model`` ``pooler_output`` (layer-normed CLS).
  The released ``BitImageProcessor`` (resize shortest side 256 bicubic -- the identity for a
  256 field -- centre crop 224, ImageNet normalization) is reproduced on float tensors. fp32.
* ``exaonepath`` (as released) and ``exaonepath_raw`` (off-label): EXAONEPath ViT-B/16
  (``LGAI-EXAONE/EXAONEPath`` ``model.safetensors``) loaded into timm's VisionTransformer with
  the Hugging Face ``config.json`` architecture (qkv bias, CLS token after the final
  LayerNorm, LayerNorm eps 1e-5 as ``VisionTransformer(**config)`` builds it). Model-card
  transform: ``Resize(256, bicubic)``, ``CenterCrop(224)``, ImageNet ``Normalize`` (torchvision,
  applied to the batch tensor). "As released" first applies the released ``macenko.py``
  normalizer (:class:`ReleasedMacenko`, a line-by-line Python 3.9 port -- the released file
  uses a ``match`` statement -- fitted to the released TCGA target image, using the vendored
  torchstain ``cov``/``percentile``); failures (exception or NaN) fall back to the input image
  as the released code does, but are always returned as a flag. fp32.

The parity of each path with its official code is checked by
``anchor_frequency_diversity_parity.py``.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
REVISION = Path(__file__).resolve().parent
for _path in (PROJECT, PROJECT / "src", PROJECT / "scripts", REVISION, REVISION / "third_party/torchstain"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

REVIEW_HF_CACHE = PROJECT / "outputs/encoder_review_2026-09-25/hf_cache"  # CONCH + SEAL weights (read-only)
EXAONEPATH_REPO = "LGAI-EXAONE/EXAONEPath"
EXAONEPATH_REVISION = "383a1247a2c543d2a04787eb9a71292f74c9ce9f"
PLIP_REPO = "vinid/plip"
PLIP_REVISION = "67ade53ddd32195868f422585f72698ef5d15094"
DINOV2_REPO = "facebook/dinov2-large"
DINOV2_REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
MACENKO_TARGET = "macenko_target/target_TCGA-55-A48X_coords_[19440  9824]_[4096 4096].png"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# encoder -> embedding streams it produces
ENCODERS = {
    "conch": ("conch_pre", "conch_projected"),
    "seal_conch": ("seal_conch_pre", "seal_conch_projected"),
    "seal_uni2": ("seal_uni2",),
    "plip": ("plip",),
    "dinov2": ("dinov2",),
    "exaonepath": ("exaonepath", "exaonepath_raw"),
}
NEW_MODELS = tuple(stream for streams in ENCODERS.values() for stream in streams)
DIMS = {"conch_pre": 512, "conch_projected": 512, "seal_conch_pre": 512, "seal_conch_projected": 512,
        "seal_uni2": 1536, "plip": 512, "dinov2": 1024, "exaonepath": 768, "exaonepath_raw": 768}
BATCH_SIZE = {"conch": 32, "seal_conch": 32, "seal_uni2": 32, "plip": 64, "dinov2": 64, "exaonepath": 64}


def snapshot(repo: str, revision: str) -> Path:
    """Local snapshot directory of a pinned Hugging Face revision (offline)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, local_files_only=True))


# ----------------------------------------------------------------------------- Macenko port

class ReleasedMacenko:
    """EXAONEPath ``macenko.py`` (``macenko_normalizer`` + ``TorchMacenkoNormalizer``), ported.

    Differences from the released file are syntactic only: the ``match`` on
    ``(form, dtype)`` is the ``('chw', 'float')`` branch that ``macenko_normalizer.__call__``
    uses, and ``__call__`` also returns a status string instead of printing. ``cov`` and
    ``percentile`` are torchstain's (vendored 1.4.1, the functions the released file imports).
    """

    def __init__(self, target_path: Path | None = None):
        import torch
        from PIL import Image

        if target_path is None:
            target_path = snapshot(EXAONEPATH_REPO, EXAONEPATH_REVISION) / MACENKO_TARGET
        self.HERef = torch.tensor([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
        self.maxCRef = torch.tensor([1.9705, 1.0308])
        self.class_default_HERef = self.HERef.clone()
        self.class_default_maxCRef = self.maxCRef.clone()
        target = Image.open(target_path)
        self.target_mode = target.mode
        self.target_size = target.size
        # transform_before_macenko = ToTensor() then x * 255; fit() replaces the class defaults.
        self.fit(self._to_tensor_255(target))

    @staticmethod
    def _to_tensor_255(image):
        import torchvision.transforms as transforms

        return transforms.ToTensor()(image) * 255

    @staticmethod
    def _convert_rgb2od(I, Io, beta):
        import torch

        I = I.permute(1, 2, 0)
        OD = -torch.log((I.reshape((-1, I.shape[-1])).float() + 1) / Io)
        ODhat = OD[~torch.any(OD < beta, dim=1)]
        return OD, ODhat

    @staticmethod
    def _find_HE(ODhat, eigvecs, alpha):
        import torch
        from torchstain.torch.utils import percentile

        That = torch.matmul(ODhat, eigvecs)
        phi = torch.atan2(That[:, 1], That[:, 0])
        minPhi = percentile(phi, alpha)
        maxPhi = percentile(phi, 100 - alpha)
        vMin = torch.matmul(eigvecs, torch.stack((torch.cos(minPhi), torch.sin(minPhi)))).unsqueeze(1)
        vMax = torch.matmul(eigvecs, torch.stack((torch.cos(maxPhi), torch.sin(maxPhi)))).unsqueeze(1)
        return torch.where(vMin[0] > vMax[0], torch.cat((vMin, vMax), dim=1), torch.cat((vMax, vMin), dim=1))

    @staticmethod
    def _find_concentration(OD, HE):
        import torch

        return torch.linalg.lstsq(HE, OD.T)[0]

    def _compute_matrices(self, I, Io, alpha, beta):
        import torch
        from torchstain.torch.utils import cov, percentile

        OD, ODhat = self._convert_rgb2od(I, Io=Io, beta=beta)
        _, eigvecs = torch.linalg.eigh(cov(ODhat.T))
        eigvecs = eigvecs[:, [1, 2]]
        HE = self._find_HE(ODhat, eigvecs, alpha)
        C = self._find_concentration(OD, HE)
        maxC = torch.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])
        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15):
        HE, _, maxC = self._compute_matrices(I, Io, alpha, beta)
        self.HERef = HE
        self.maxCRef = maxC

    def normalize(self, I, Io=240, alpha=1, beta=0.15):
        import torch

        c, h, w = I.shape
        HE, C, maxC = self._compute_matrices(I, Io, alpha, beta)
        C *= (self.maxCRef / maxC).unsqueeze(-1)
        Inorm = Io * torch.exp(-torch.matmul(self.HERef, C))
        Inorm[Inorm > 255] = 255
        return Inorm.reshape(c, h, w).float() / 255.

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        """``image``: HWC float32 in [0, 1] (uint8 / 255 or an unquantized render).

        Returns the normalized HWC float32 image in [0, 1] and ``"ok"``, ``"nan"`` or
        ``"error: ..."``. On failure the image is the input, as ``macenko_normalizer``
        returns ``ToTensor()(image)``.
        """
        import torch

        # ToTensor(PIL uint8) = uint8 / 255 in float32, then x * 255 (released order of operations).
        tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).permute(2, 0, 1).contiguous() * 255
        try:
            normalized = self.normalize(tensor)
        except Exception as error:  # the released code catches every exception
            return np.asarray(image, dtype=np.float32), f"error: {error}"
        if torch.any(torch.isnan(normalized)):
            return np.asarray(image, dtype=np.float32), "nan"
        return normalized.permute(1, 2, 0).contiguous().numpy(), "ok"


# ----------------------------------------------------------------------------- loaders

def load_exaonepath_timm(device: str = "cuda"):
    import json

    import timm.models.vision_transformer as vit
    import torch
    from safetensors.torch import load_file

    root = snapshot(EXAONEPATH_REPO, EXAONEPATH_REVISION)
    config = json.loads((root / "config.json").read_text())
    if (config["embed_dim"], config["depth"], config["num_heads"], config["patch_size"], config["qkv_bias"],
            config["mlp_ratio"], config["num_classes"]) != (768, 12, 12, 16, True, 4, 0):
        raise ValueError(f"unexpected EXAONEPath config {config}")
    model = vit.VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        qkv_bias=True, class_token=True, global_pool="token", num_classes=0,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-5),  # VisionTransformer(**config): nn.LayerNorm default eps
    )
    model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
    return model.eval().to(device)


class Encoder:
    """One loaded encoder with a ``__call__(images) -> {stream: [n, d] unit vectors}``."""

    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self.batch_size = BATCH_SIZE[name]
        if name in ("conch", "seal_conch"):
            from scripts.features.review_multiencoder_scanner import load_encoder

            self.model, self.size, self.mean, self.std, variants = load_encoder(name, REVIEW_HF_CACHE)
            if tuple(variants) != ENCODERS[name]:
                raise ValueError(f"{name}: unexpected variants {variants}")
        elif name == "seal_uni2":
            from scripts.features.review_multiencoder_scanner import load_encoder, merge_seal_lora

            self.model, self.size, self.mean, self.std, variants = load_encoder("uni2", REVIEW_HF_CACHE)
            if tuple(variants) != ("uni2",):
                raise ValueError(f"seal_uni2: unexpected variants {variants}")
            merge_seal_lora(self.model, REVIEW_HF_CACHE.parent)
            self.model.eval()
        elif name == "plip":
            # vinid/plip ships only pytorch_model.bin, which transformers >= 4.5x refuses to torch.load
            # with torch < 2.6; the same weights are loaded here with torch.load(weights_only=True).
            import torch
            from transformers import CLIPConfig, CLIPModel

            root = snapshot(PLIP_REPO, PLIP_REVISION)
            model = CLIPModel(CLIPConfig.from_pretrained(root))
            state = torch.load(root / "pytorch_model.bin", map_location="cpu", weights_only=True)
            result = model.load_state_dict(state, strict=False)
            if result.missing_keys or any(not key.endswith("position_ids") for key in result.unexpected_keys):
                raise ValueError(f"PLIP weights do not match CLIPModel: {result}")
            self.plip_ignored_keys = list(result.unexpected_keys)
            self.model = model.eval().to(device)
        elif name == "dinov2":
            from transformers import Dinov2Model

            self.model = Dinov2Model.from_pretrained(DINOV2_REPO, revision=DINOV2_REVISION).eval().to(device)
        elif name == "exaonepath":
            self.model = load_exaonepath_timm(device)
            import torchvision.transforms as transforms

            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            raise ValueError(name)

    # -- float-tensor versions of the released image processors
    def _plip_pixels(self, x):
        import torch.nn.functional as F

        h, w = x.shape[-2:]
        scale = 224 / min(h, w)
        size = (max(224, round(h * scale)), max(224, round(w * scale)))
        if (h, w) != size:
            x = F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True)
        top, left = (x.shape[-2] - 224) // 2, (x.shape[-1] - 224) // 2
        x = x[..., top:top + 224, left:left + 224]
        return self._normalize(x, CLIP_MEAN, CLIP_STD)

    def _dinov2_pixels(self, x):
        import torch.nn.functional as F

        h, w = x.shape[-2:]
        if min(h, w) != 256:
            scale = 256 / min(h, w)
            x = F.interpolate(x, size=(round(h * scale), round(w * scale)), mode="bicubic",
                              align_corners=False, antialias=True)
        top, left = (x.shape[-2] - 224) // 2, (x.shape[-1] - 224) // 2
        x = x[..., top:top + 224, left:left + 224]
        return self._normalize(x, IMAGENET_MEAN, IMAGENET_STD)

    def _normalize(self, x, mean, std):
        import torch

        mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean_t) / std_t

    def _batches(self, images):
        import torch

        for start in range(0, len(images), self.batch_size):
            block = np.stack(images[start:start + self.batch_size]).astype(np.float32, copy=False)
            yield torch.from_numpy(block).permute(0, 3, 1, 2).contiguous().to(self.device)

    def __call__(self, images: list[np.ndarray], macenko: list[np.ndarray] | None = None) -> dict[str, np.ndarray]:
        import torch
        import torch.nn.functional as F

        if self.name in ("conch", "seal_conch", "seal_uni2"):
            from scripts.features.review_multiencoder_scanner import embed

            out = embed(self.model, "uni2" if self.name == "seal_uni2" else self.name, images,
                        self.size, self.mean, self.std, self.batch_size)
            if self.name == "seal_uni2":
                out = {"seal_uni2": out["uni2"]}
            return {key: np.asarray(value, dtype=np.float32) for key, value in out.items()}
        collected: dict[str, list[np.ndarray]] = {}
        with torch.inference_mode():
            if self.name == "exaonepath":
                if macenko is None or len(macenko) != len(images):
                    raise ValueError("exaonepath needs the Macenko-normalized images")
                for stream, source in (("exaonepath", macenko), ("exaonepath_raw", images)):
                    for x in self._batches(source):
                        value = self.model(self.transform(x))
                        collected.setdefault(stream, []).append(F.normalize(value.float(), dim=-1).cpu().numpy())
            else:
                for x in self._batches(images):
                    if self.name == "plip":
                        value = self.model.get_image_features(pixel_values=self._plip_pixels(x))
                    else:
                        value = self.model(pixel_values=self._dinov2_pixels(x)).pooler_output
                    collected.setdefault(self.name, []).append(F.normalize(value.float(), dim=-1).cpu().numpy())
        return {key: np.concatenate(value).astype(np.float32) for key, value in collected.items()}
