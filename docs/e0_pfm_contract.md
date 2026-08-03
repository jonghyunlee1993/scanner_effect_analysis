# E0/E4 core PFM execution contract

Status: frozen before PFM outcome extraction on 2026-08-02.

TRIDENT is pinned to commit `a6305acfef68d4c6da65e837dde1e0d1870d60e1` and is
used only for encoder construction, checkpoint loading and official evaluation transforms.
The prenorm manifest/DataLoader owns paired center identity, route geometry, native FOV and
image-condition construction.

The initial runtime is also frozen to PyTorch 2.5.1, torchvision 0.20.1, timm 0.9.8,
huggingface_hub 0.29.1 and CONCH 0.1.0 from source commit
`02d6ac59cc20874bff0f581de258c2b257f69a84`. The smoke gate fails on runtime drift.

| Encoder | Hugging Face revision | Native FOV | Feature contract |
|---|---|---:|---|
| ResNet50 | `timm/resnet50.tv_in1k@78f3ecfdb38e06d9b8397f662e7ab8fee96026fa` | 256 px | stage 3, adaptive pooled, 1,024-D |
| UNI v1 | `MahmoodLab/uni@b55a5ec6cade1a39edfe6534189a9b8ca7a022f0` | 256 px | ViT-L/16 class token, 1,024-D |
| CONCH v1 | `MahmoodLab/conch@f9ca9f877171a28ade80228fb195ac5d79003357` | 512 px | `with_proj=False`, `normalize=False`, 512-D |
| Virchow2 | `paige-ai/Virchow2@3158645804b69e3f3bc4439d4116edddf0840a72` | 224 px | class + mean non-register patch tokens, 2,560-D |

`fetch_e0_pfm_checkpoints.py` downloads only these exact revisions into a project-owned
cache and writes byte size and SHA-256 for every checkpoint. A later GPU smoke test must
verify output shape, dtype, finite values and deterministic repeated inference before cohort
feature extraction. No TRIDENT sampling or slide segmentation is run.

`smoke_e0_pfm_encoders.py` additionally refuses to run if TRIDENT HEAD or any tracked
encoder/transform source file differs from the pinned commit. It loads each local checkpoint,
applies the exposed official eval transform to a deterministic model-sized RGB image and
requires the expected feature dimension, finite output and bit-identical repeated eval.
