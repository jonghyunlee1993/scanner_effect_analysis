#!/usr/bin/env python3
"""RV17 stage 1 (GPU): ViT-Tiny/14 DINOv2 from scratch on AT2/GT450 mixtures, Gaussian blur off.

Protocol: ``scanner_mixture_protocol.md``. One array task per arm (AT2 share p):
``p100`` (1.0), ``p080`` (0.8), ``p020`` (0.2), ``p000`` (0.0). Each pretraining location
contributes one image -- AT2 if u < p else GT450, u ~ U(0, 1) drawn once per location (seed 17) --
so the sets are nested and contain the same tissue. Init seed 0 and data-order seed 0 in every
arm; sample keys, augmentation draws, masks and drop-path draws are shared across arms (the RV16
machinery, imported from ``scanner_composition_pretrain``). The Gaussian blur of
``DataAugmentationDINO`` is replaced by an identity that consumes the same random draws.

Outputs (``results/scanner_mixture/train/<arm>/``): ``teacher_e{000,010,025}.pt``, ``log.csv``,
``locations.csv.gz`` (with the scanner of every location), ``config.yaml``, ``run.json``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import math
import os
import tempfile
import time
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REVISION = Path(__file__).resolve().parent
if str(REVISION) not in sys.path:
    sys.path.insert(0, str(REVISION))

import scanner_composition_pretrain as P  # noqa: E402  (adds the vendored DINOv2 to sys.path)
from scanner_composition_pretrain import C  # noqa: E402

OUTPUT = REVISION / "results/scanner_mixture"
ARMS = {"p100": 1.0, "p080": 0.8, "p020": 0.2, "p000": 0.0}
ARM_ORDER = ("p100", "p080", "p020", "p000")
ASSIGNMENT_SEED = 17
ORDER_SEED = 0
SAVE_EPOCHS = (0, 10, 25)
OVERRIDES = {"student": {"arch": "vit_tiny"},
             "dino": {"head_n_prototypes": 16384}, "ibot": {"head_n_prototypes": 16384},
             "optim": {"epochs": 25, "warmup_epochs": 3, "freeze_last_layer_epochs": 1},
             "teacher": {"warmup_teacher_temp_epochs": 8}}


def vit_tiny(patch_size=16, num_register_tokens=0, in_chans=3, channel_adaptive=False, **kwargs):
    """``vit_small`` of the vendored DINOv2 code with embedding 192 and 3 heads."""
    from dinov2.layers import MemEffAttention, NestedTensorBlock as Block
    from dinov2.models.vision_transformer import DinoVisionTransformer

    return DinoVisionTransformer(patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
                                 block_fn=partial(Block, attn_class=MemEffAttention),
                                 num_register_tokens=num_register_tokens, in_chans=in_chans,
                                 channel_adaptive=channel_adaptive, **kwargs)


def register_vit_tiny() -> None:
    from dinov2.models import vision_transformer as vits

    vits.vit_tiny = vit_tiny  # dinov2.models.build_model looks the arch up in this module


def build_config():
    from omegaconf import OmegaConf

    cfg = OmegaConf.merge(P.build_config(), OmegaConf.create(OVERRIDES))
    return cfg


def load_mixed_images(slides: pd.DataFrame, p: float) -> tuple[np.ndarray, pd.DataFrame]:
    """Every cached location, AT2 or GT450 by the fixed per-location draw."""
    at2, gt450 = C.SCANNERS.index("at2"), C.SCANNERS.index("gt450")
    total = int(slides.pair_count.sum())
    u = np.random.default_rng(ASSIGNMENT_SEED).random(total)
    images = np.empty((total, 256, 256, 3), dtype=np.uint8)
    rows, offset = [], 0
    for slide in slides.itertuples(index=False):
        with h5py.File(slide.cache_path, "r") as cache:
            if tuple(x.decode().lower() for x in cache["scanner_names"][:]) != C.SCANNERS:
                raise ValueError(f"{slide.slide_id}: scanner order")
            n = cache["images"].shape[0]
            if n != int(slide.pair_count):
                raise ValueError(f"{slide.slide_id}: pair count")
            use_at2 = u[offset:offset + n] < p
            # whole-column reads (chunked per image) are far faster than h5py point selections
            if use_at2.any():
                images[offset:offset + n][use_at2] = cache["images"][:, at2][use_at2]
            if (~use_at2).any():
                images[offset:offset + n][~use_at2] = cache["images"][:, gt450][~use_at2]
            source_index = np.asarray(cache["source_index"][:], dtype=np.int64)
        rows.append(pd.DataFrame({"slide_id": slide.slide_id, "location_index": np.arange(n),
                                  "source_index": source_index, "u": u[offset:offset + n],
                                  "scanner": np.where(use_at2, "at2", "gt450")}))
        offset += n
    return images, pd.concat(rows, ignore_index=True)


def disable_blur(transform) -> int:
    """Replace every DINOv2 GaussianBlur's inner blur by an identity drawing the same sigma."""
    from torchvision import transforms as T
    from dinov2.data.transforms import GaussianBlur

    class DrawOnlyBlur(T.GaussianBlur):
        def forward(self, img):
            self.get_params(self.sigma[0], self.sigma[1])  # same RNG consumption as the real blur
            return img

    count = 0
    seen = set()

    def visit(node):
        nonlocal count
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, GaussianBlur):
            inner = node.transforms[0]
            node.transforms[0] = DrawOnlyBlur(kernel_size=inner.kernel_size, sigma=inner.sigma)
            count += 1
        for child in getattr(node, "transforms", []) or []:
            visit(child)
        for name in ("global_transfo1", "global_transfo2", "local_transfo"):
            if hasattr(node, name):
                visit(getattr(node, name))

    visit(transform)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True, help="0..3 = p100, p080, p020, p000")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--max-slides", type=int, default=0, help="smoke test only")
    parser.add_argument("--max-iters", type=int, default=0, help="smoke test only")
    args = parser.parse_args()
    arm = ARM_ORDER[args.task_index]
    p = ARMS[arm]

    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from dinov2.data.augmentations import DataAugmentationDINO
    from dinov2.data.masking import MaskingGenerator
    from dinov2.loss import DINOLoss, KoLeoLoss, iBOTPatchLoss

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = True
    register_vit_tiny()
    out = args.output_root / "train" / arm
    out.mkdir(parents=True, exist_ok=True)
    dist.init_process_group("nccl", init_method=f"file://{Path(tempfile.mkdtemp()) / 'pg'}", rank=0, world_size=1)

    cfg = build_config()
    OmegaConf.save(cfg, out / "config.yaml")
    started = time.time()
    slides = P.pretraining_slides(args.max_slides)
    images, index = load_mixed_images(slides, p)
    C.write_frame(out / "locations.csv.gz", index)
    n = len(images)
    batch_size = int(cfg.train.batch_size_per_gpu)
    epoch_length = n // batch_size
    total_iters = int(cfg.optim.epochs) * epoch_length
    end_iter = min(total_iters, args.max_iters) if args.max_iters else total_iters
    share = float((index.scanner == "at2").mean())
    print(f"arm {arm}: AT2 share target {p}, realized {share:.4f}; {len(slides)} slides, {n} images loaded in "
          f"{time.time() - started:.0f}s; epoch {epoch_length} iters, total {total_iters}", flush=True)
    save_at = {round(e * total_iters / int(cfg.optim.epochs)): e for e in SAVE_EPOCHS}

    student, teacher = P.build_networks(cfg)
    init_hashes = {"student": P.state_hash(student), "teacher": P.state_hash(teacher)}
    student.cuda().train()
    teacher.cuda().eval()
    losses = (DINOLoss(cfg.dino.head_n_prototypes).cuda(), iBOTPatchLoss(cfg.ibot.head_n_prototypes).cuda(), KoLeoLoss().cuda())
    optimizer = torch.optim.AdamW(P.param_groups(cfg, student), betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))
    lr_s, wd_s, mom_s, temp_s, last_s = P.build_schedulers(cfg, epoch_length)
    P.save_teacher(out / "teacher_e000.pt", teacher, cfg, {"arm": arm, "iteration": 0, "epoch": 0})
    torch.cuda.manual_seed_all(P.seed_of(P.INIT_SEED, 4))

    patch, size = cfg.student.patch_size, cfg.crops.global_crops_size
    mask_generator = MaskingGenerator(input_size=(size // patch, size // patch),
                                      max_num_patches=0.5 * size // patch * size // patch)
    transform = DataAugmentationDINO(cfg.crops.global_crops_scale, cfg.crops.local_crops_scale,
                                     cfg.crops.local_crops_number, global_crops_size=size,
                                     local_crops_size=cfg.crops.local_crops_size)
    replaced = disable_blur(transform)
    if replaced != 3:
        raise RuntimeError(f"expected 3 Gaussian blurs in DataAugmentationDINO, replaced {replaced}")
    collate = partial(P.keyed_collate, order_seed=ORDER_SEED, mask_ratio_tuple=tuple(cfg.ibot.mask_ratio_min_max),
                      mask_probability=cfg.ibot.mask_sample_probability, n_tokens=(size // patch) ** 2,
                      mask_generator=mask_generator, dtype=torch.bfloat16)
    loader = torch.utils.data.DataLoader(P.KeyedImages(images, ORDER_SEED, transform), batch_size=batch_size,
                                         sampler=P.KeySampler(n, end_iter * batch_size, ORDER_SEED),
                                         num_workers=args.workers, collate_fn=collate, pin_memory=True,
                                         drop_last=True, prefetch_factor=4 if args.workers else None)
    run = {"arm": arm, "at2_share_target": p, "at2_share_realized": share, "order_seed": ORDER_SEED,
           "init_seed": P.INIT_SEED, "assignment_seed": ASSIGNMENT_SEED, "blur": "off (draw-only identity)",
           "slides": len(slides), "images": n, "batch_size": batch_size, "epoch_length": epoch_length,
           "total_iters": total_iters, "end_iter": end_iter, "lr": float(cfg.optim.lr), "init_hashes": init_hashes,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "smoke": bool(args.max_iters or args.max_slides)}
    C.write_json(out / "run.json", run)
    log_path = out / "log.csv"
    if log_path.exists():
        log_path.unlink()
    first_keys, iteration, tick = [], 0, time.time()
    for batch in loader:
        lr, wd, mom = lr_s[iteration], wd_s[iteration], mom_s[iteration]
        teacher_temp, last_lr = temp_s[iteration], last_s[iteration]
        for group in optimizer.param_groups:
            group["weight_decay"] = wd * group["wd_multiplier"]
            group["lr"] = (last_lr if group["is_last_layer"] else lr) * group["lr_multiplier"]
        optimizer.zero_grad(set_to_none=True)
        loss_dict = P.train_step(batch, student, teacher, losses, cfg, teacher_temp)
        for module in student.values():
            torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.optim.clip_grad)
        optimizer.step()
        with torch.no_grad():
            for key in student:
                t_params = list(teacher[key].parameters())
                torch._foreach_mul_(t_params, mom)
                torch._foreach_add_(t_params, list(student[key].parameters()), alpha=1 - mom)
        values = {k: float(v.item()) for k, v in loss_dict.items()}
        if not all(math.isfinite(v) for v in values.values()):
            raise FloatingPointError(f"non-finite loss at iteration {iteration}: {values}")
        if iteration < 3:
            first_keys.append(batch["keys"][:4])
        iteration += 1
        if iteration % 50 == 0 or iteration == end_iter:
            elapsed = time.time() - tick
            row = {"iteration": iteration, "epoch": iteration / epoch_length, "lr": lr, "wd": wd, "momentum": mom,
                   "teacher_temp": teacher_temp, "total_loss": sum(values.values()), **values,
                   "seconds": elapsed, "images_per_s": 50 * batch_size / max(elapsed, 1e-9)}
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
            tick = time.time()
        if iteration in save_at and save_at[iteration] > 0:
            P.save_teacher(out / f"teacher_e{save_at[iteration]:03d}.pt", teacher, cfg,
                           {"arm": arm, "iteration": iteration, "epoch": save_at[iteration]})
    run.update({"finished_iteration": iteration, "first_batch_keys": first_keys, "wall_s": round(time.time() - started, 1)})
    C.write_json(out / "run.json", run)
    print(json.dumps({"status": "complete", "arm": arm, "iteration": iteration}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
