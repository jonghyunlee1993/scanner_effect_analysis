#!/usr/bin/env python3
"""RV16 stage 1 (GPU): DINOv2 ViT-S/14 from scratch on one scanner's images of the pretraining slides.

Protocol: ``scanner_composition_pilot_protocol.md``. One array task per arm:

* ``A``  -- AT2 image of every cached location of the fold-1..4 slides, data-order seed 0
* ``B``  -- GT450 image of the same locations, data-order seed 0
* ``A1`` -- as A with data-order seed 1 (noise floor)
* ``B1`` -- as B with data-order seed 1 (noise floor)

Everything except the scanner column is shared: init seed 0, the location list and its order,
and -- within one data-order seed -- every random draw. Each sample is addressed by a key
``epoch * N + location``; its augmentation RNG is seeded from (data-order seed, key) and the
iBOT masks of a batch from (data-order seed, keys of the batch), so arms A and B see the same
location with the same crops, jitter, blur, solarization and masks at the same step.

The objective, augmentation, heads, schedules and parameter groups are those of the vendored
official DINOv2 code (``third_party/dinov2_7764ea0f``). Implementation deviations, none of which
changes the objective: one GPU without FSDP, bf16 autocast instead of fp16 + grad scaler, the
student run separately on global and local crops instead of xFormers nested tensors, heads
applied per input instead of one block-diagonal call, and a world-size-1 process group so the
Sinkhorn-Knopp all-reduces are identities.

Outputs (``results/scanner_composition_pilot/train/<arm>/``): ``teacher_e{000,010,025,050,100}.pt``
(teacher backbone state dict), ``state_last.pt`` (resumable), ``log.csv``, ``config.yaml``,
``run.json``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
REVISION = Path(__file__).resolve().parent
DINOV2 = REVISION / "third_party/dinov2_7764ea0f"
for _path in (PROJECT, PROJECT / "src", REVISION, DINOV2):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import corrected_embeddings_common as C  # noqa: E402

OUTPUT = REVISION / "results/scanner_composition_pilot"
HELD_OUT_FOLD = 0
ARMS = {"A": ("at2", 0), "B": ("gt450", 0), "A1": ("at2", 1), "B1": ("gt450", 1)}
INIT_SEED = 0
SAVE_EPOCHS = (0, 10, 25, 50, 100)
OVERRIDES = {
    # method settings of the released ViT-L/14 config (train/vitl14.yaml)
    "train": {"centering": "sinkhorn_knopp", "batch_size_per_gpu": 256, "seed": INIT_SEED},
    "ibot": {"separate_head": True},
    "crops": {"local_crops_size": 98},
    # ViT-S/14 scale
    "student": {"arch": "vit_small", "patch_size": 14, "drop_path_rate": 0.1},
    "optim": {"epochs": 100},
}


def seed_of(*parts: int) -> int:
    """Stable 32-bit seed from integers (independent of PYTHONHASHSEED)."""
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


# ----------------------------------------------------------------------------- data
def pretraining_slides(max_slides: int = 0) -> pd.DataFrame:
    cohort = C.load_cohort()
    slides = cohort[cohort.fold.ne(HELD_OUT_FOLD)].sort_values("slide_id", kind="stable").reset_index(drop=True)
    return slides.iloc[:max_slides] if max_slides else slides


def load_images(slides: pd.DataFrame, scanner: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Every cached location of the slides, one scanner column, in (slide_id, location) order."""
    column = C.SCANNERS.index(scanner)
    total = int(slides.pair_count.sum())
    images = np.empty((total, 256, 256, 3), dtype=np.uint8)
    rows, offset = [], 0
    for slide in slides.itertuples(index=False):
        with h5py.File(slide.cache_path, "r") as cache:
            names = tuple(x.decode().lower() for x in cache["scanner_names"][:])
            if names != C.SCANNERS:
                raise ValueError(f"{slide.slide_id}: scanner order {names}")
            n = cache["images"].shape[0]
            if n != int(slide.pair_count):
                raise ValueError(f"{slide.slide_id}: {n} cached pairs, cohort says {slide.pair_count}")
            images[offset:offset + n] = cache["images"][:, column]
            source_index = np.asarray(cache["source_index"][:], dtype=np.int64)
        rows.append(pd.DataFrame({"slide_id": slide.slide_id, "location_index": np.arange(n),
                                  "source_index": source_index}))
        offset += n
    return images, pd.concat(rows, ignore_index=True)


class KeyedImages:
    """Map-style dataset addressed by key = epoch * N + location (official transform, PIL input)."""

    def __init__(self, images: np.ndarray, order_seed: int, transform):
        self.images = images
        self.order_seed = order_seed
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, key: int):
        import torch
        from PIL import Image

        seed = seed_of(self.order_seed, 1, key)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return self.transform(Image.fromarray(self.images[key % len(self.images)])), key


class KeySampler:
    """Epoch-wise permutations seeded by (data-order seed, epoch); yields sample keys."""

    def __init__(self, n: int, total: int, order_seed: int, start: int = 0):
        self.n, self.total, self.order_seed, self.start = n, total, order_seed, start

    def __iter__(self):
        import torch

        produced = 0
        epoch = 0
        while produced < self.total:
            generator = torch.Generator()
            generator.manual_seed(seed_of(self.order_seed, 2, epoch))
            for location in torch.randperm(self.n, generator=generator).tolist():
                if produced >= self.total:
                    return
                if produced >= self.start:
                    yield epoch * self.n + location
                produced += 1
            epoch += 1

    def __len__(self) -> int:
        return self.total - self.start


def keyed_collate(samples, order_seed: int, **kwargs):
    from dinov2.data.collate import collate_data_and_cast

    keys = [key for _, key in samples]
    random.seed(seed_of(order_seed, 3, keys[0], keys[-1], len(keys)))
    batch = collate_data_and_cast([(item, ()) for item, _ in samples], **kwargs)
    batch["keys"] = keys
    return batch


# ----------------------------------------------------------------------------- model
def build_config():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(DINOV2 / "dinov2/configs/ssl_default_config.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.create(OVERRIDES))
    # sqrt_wrt_1024 scaling rule of dinov2.utils.config.apply_scaling_rules_to_cfg, world size 1
    cfg.optim.lr = cfg.optim.base_lr * math.sqrt(cfg.train.batch_size_per_gpu / 1024.0)
    return cfg


def vit_kwargs(cfg) -> dict:
    """Keyword arguments of dinov2.models.build_model for the teacher (used again for evaluation)."""
    s = cfg.student
    return dict(img_size=cfg.crops.global_crops_size, patch_size=s.patch_size, init_values=s.layerscale,
                ffn_layer=s.ffn_layer, block_chunks=s.block_chunks, qkv_bias=s.qkv_bias, proj_bias=s.proj_bias,
                ffn_bias=s.ffn_bias, num_register_tokens=s.num_register_tokens,
                interpolate_offset=s.interpolate_offset, interpolate_antialias=s.interpolate_antialias,
                in_chans=s.in_chans, channel_adaptive=s.channel_adaptive)


def build_networks(cfg):
    """Student and teacher as in SSLMetaArch.__init__ + prepare_for_distributed_training."""
    import torch
    from torch import nn
    from dinov2.layers import DINOHead
    from dinov2.models import build_model_from_cfg

    torch.manual_seed(INIT_SEED)
    np.random.seed(INIT_SEED)
    random.seed(INIT_SEED)
    student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
    head = partial(DINOHead, in_dim=embed_dim, out_dim=cfg.dino.head_n_prototypes, hidden_dim=cfg.dino.head_hidden_dim,
                   bottleneck_dim=cfg.dino.head_bottleneck_dim, nlayers=cfg.dino.head_nlayers)
    ibot_head = partial(DINOHead, in_dim=embed_dim, out_dim=cfg.ibot.head_n_prototypes, hidden_dim=cfg.ibot.head_hidden_dim,
                        bottleneck_dim=cfg.ibot.head_bottleneck_dim, nlayers=cfg.ibot.head_nlayers)
    student = nn.ModuleDict({"backbone": student_backbone, "dino_head": head(), "ibot_head": ibot_head()})
    teacher = nn.ModuleDict({"backbone": teacher_backbone, "dino_head": head(), "ibot_head": ibot_head()})
    for key in student:
        teacher[key].load_state_dict(student[key].state_dict())
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return student, teacher


def state_hash(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_schedulers(cfg, epoch_length: int):
    """dinov2.train.train.build_schedulers with OFFICIAL_EPOCH_LENGTH = one pass over the data."""
    from dinov2.utils.utils import CosineScheduler

    total = cfg.optim.epochs * epoch_length
    lr = dict(base_value=cfg.optim.lr, final_value=cfg.optim.min_lr, total_iters=total,
              warmup_iters=cfg.optim.warmup_epochs * epoch_length, start_warmup_value=0)
    wd = dict(base_value=cfg.optim.weight_decay, final_value=cfg.optim.weight_decay_end, total_iters=total)
    momentum = dict(base_value=cfg.teacher.momentum_teacher, final_value=cfg.teacher.final_momentum_teacher,
                    total_iters=total)
    temp_iters = cfg.teacher.warmup_teacher_temp_epochs * epoch_length
    teacher_temp = dict(base_value=cfg.teacher.teacher_temp, final_value=cfg.teacher.teacher_temp,
                        total_iters=temp_iters, warmup_iters=temp_iters,
                        start_warmup_value=cfg.teacher.warmup_teacher_temp)
    last_layer = CosineScheduler(**lr)
    last_layer.schedule[: cfg.optim.freeze_last_layer_epochs * epoch_length] = 0
    return CosineScheduler(**lr), CosineScheduler(**wd), CosineScheduler(**momentum), CosineScheduler(**teacher_temp), last_layer


def param_groups(cfg, student):
    from dinov2.utils.param_groups import fuse_params_groups, get_params_groups_with_decay

    groups = []
    for module in student.values():
        fused = fuse_params_groups(get_params_groups_with_decay(
            model=module, lr_decay_rate=cfg.optim.layerwise_decay, patch_embed_lr_mult=cfg.optim.patch_embed_lr_mult))
        for group in fused:
            group["foreach"] = True
        groups += fused
    return groups


# ----------------------------------------------------------------------------- step
def train_step(batch, student, teacher, losses, cfg, teacher_temp):
    """SSLMetaArch.forward_backward (Sinkhorn-Knopp centering, separate iBOT head)."""
    import torch

    dino_loss, ibot_loss, koleo_loss = losses
    n_global, n_local = 2, cfg.crops.local_crops_number
    global_crops = batch["collated_global_crops"].cuda(non_blocking=True)
    local_crops = batch["collated_local_crops"].cuda(non_blocking=True)
    masks = batch["collated_masks"].cuda(non_blocking=True)
    mask_indices = batch["mask_indices_list"].cuda(non_blocking=True)
    n_masked_tensor = batch["n_masked_patches"].cuda(non_blocking=True)
    n_masked = mask_indices.shape[0]
    upperbound = batch["upperbound"]
    masks_weight = batch["masks_weight"].cuda(non_blocking=True)
    local_terms = max(n_local * n_global, 1)
    global_terms = (n_global - 1) * n_global

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            out = teacher["backbone"](global_crops, is_training=True)
            cls = out["x_norm_clstoken"].chunk(n_global)
            cls = torch.cat((cls[1], cls[0]))  # A matched to B
            patches = out["x_norm_patchtokens"]
            buffer = patches.new_zeros(upperbound, patches.shape[-1])
            buffer[:n_masked].copy_(torch.index_select(patches.flatten(0, 1), dim=0, index=mask_indices))
            teacher_cls_head = teacher["dino_head"](cls)
            teacher_masked_head = teacher["ibot_head"](buffer)[:n_masked]
            teacher_dino = dino_loss.sinkhorn_knopp_teacher(teacher_cls_head, teacher_temp=teacher_temp).view(
                n_global, -1, teacher_cls_head.shape[-1])
            teacher_ibot = ibot_loss.sinkhorn_knopp_teacher(teacher_masked_head, teacher_temp=teacher_temp,
                                                            n_masked_patches_tensor=n_masked_tensor)
        student_global = student["backbone"](global_crops, masks=masks, is_training=True)
        student_local = student["backbone"](local_crops, masks=None, is_training=True)
        local_cls_head = student["dino_head"](student_local["x_norm_clstoken"]).float()
        global_cls_head = student["dino_head"](student_global["x_norm_clstoken"]).float()
        student_patches = student_global["x_norm_patchtokens"]
        student_buffer = student_patches.new_zeros(upperbound, student_patches.shape[-1])
        student_buffer[:n_masked].copy_(torch.index_select(student_patches.flatten(0, 1), dim=0, index=mask_indices))
        student_masked_head = student["ibot_head"](student_buffer)[:n_masked].float()

    loss_dict = {}
    total = 0.0
    dino_local = dino_loss(student_output_list=local_cls_head.chunk(n_local),
                           teacher_out_softmaxed_centered_list=teacher_dino) / (global_terms + local_terms)
    loss_dict["dino_local_crops_loss"] = dino_local
    total = total + cfg.dino.loss_weight * dino_local
    scales = 2  # global crops are processed together
    dino_global = dino_loss(student_output_list=[global_cls_head],
                            teacher_out_softmaxed_centered_list=[teacher_dino.flatten(0, 1)]) * scales / (global_terms + local_terms)
    loss_dict["dino_global_crops_loss"] = dino_global
    total = total + cfg.dino.loss_weight * dino_global
    koleo = cfg.dino.koleo_loss_weight * sum(koleo_loss(p.float()) for p in student_global["x_norm_clstoken"].chunk(2))
    loss_dict["koleo_loss"] = koleo / scales
    total = total + koleo
    ibot = ibot_loss.forward_masked(student_masked_head, teacher_ibot, student_masks_flat=masks,
                                    n_masked_patches=n_masked, masks_weight=masks_weight) * scales * (1.0 / n_global)
    loss_dict["ibot_loss"] = ibot / 2
    total = total + cfg.ibot.loss_weight * ibot
    total.backward()
    return loss_dict


def save_teacher(path: Path, teacher, cfg, meta: dict) -> None:
    import torch
    from omegaconf import OmegaConf

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"teacher_backbone": {k: v.detach().float().cpu() for k, v in teacher["backbone"].state_dict().items()},
                "arch": cfg.student.arch, "vit_kwargs": vit_kwargs(cfg),
                "config": OmegaConf.to_container(cfg), **meta}, temporary)
    temporary.replace(path)


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), default=None)
    parser.add_argument("--task-index", type=int, default=None, help="array index into A, B, A1, B1")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--max-slides", type=int, default=0, help="smoke test only")
    parser.add_argument("--max-iters", type=int, default=0, help="smoke test only")
    parser.add_argument("--epochs", type=int, default=0, help="smoke test only (overrides optim.epochs)")
    parser.add_argument("--stop-epoch", type=int, default=0,
                        help="stop after this epoch's checkpoint; schedules keep the full optim.epochs length")
    args = parser.parse_args()
    arm = args.arm or ("A", "B", "A1", "B1")[args.task_index]
    scanner, order_seed = ARMS[arm]

    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from dinov2.data.augmentations import DataAugmentationDINO
    from dinov2.data.masking import MaskingGenerator
    from dinov2.loss import DINOLoss, KoLeoLoss, iBOTPatchLoss

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = True  # as dinov2/train/train.py
    out = args.output_root / "train" / arm
    out.mkdir(parents=True, exist_ok=True)
    rendezvous = Path(tempfile.mkdtemp()) / "pg"
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1)

    cfg = build_config()
    if args.epochs:
        cfg.optim.epochs = args.epochs
    OmegaConf.save(cfg, out / "config.yaml")
    started = time.time()
    slides = pretraining_slides(args.max_slides)
    images, index = load_images(slides, scanner)
    C.write_frame(out / "locations.csv.gz", index)
    n = len(images)
    batch_size = int(cfg.train.batch_size_per_gpu)
    epoch_length = n // batch_size
    total_iters = int(cfg.optim.epochs) * epoch_length
    print(f"arm {arm}: scanner {scanner}, order seed {order_seed}, {len(slides)} slides, {n} images "
          f"({images.nbytes / 1e9:.1f} GB) loaded in {time.time() - started:.0f}s; epoch {epoch_length} iters, "
          f"total {total_iters}", flush=True)
    save_at = {round(e * total_iters / int(cfg.optim.epochs)): e for e in SAVE_EPOCHS}

    student, teacher = build_networks(cfg)
    init_hashes = {"student": state_hash(student), "teacher": state_hash(teacher)}
    student.cuda().train()
    teacher.cuda().eval()
    losses = (DINOLoss(cfg.dino.head_n_prototypes).cuda(), iBOTPatchLoss(cfg.ibot.head_n_prototypes).cuda(), KoLeoLoss().cuda())
    optimizer = torch.optim.AdamW(param_groups(cfg, student), betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))
    lr_s, wd_s, mom_s, temp_s, last_s = build_schedulers(cfg, epoch_length)

    start_iter = 0
    resume = out / "state_last.pt"
    if resume.exists() and not args.max_iters:
        state = torch.load(resume, map_location="cuda")
        student.load_state_dict(state["student"])
        teacher.load_state_dict(state["teacher"])
        optimizer.load_state_dict(state["optimizer"])
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        start_iter = int(state["iteration"])
        print(f"resumed at iteration {start_iter}", flush=True)
    else:
        save_teacher(out / "teacher_e000.pt", teacher, cfg, {"arm": arm, "iteration": 0, "epoch": 0})
    # Drop path draws come from the CUDA generator; same seed and shapes in every arm.
    if start_iter == 0:
        torch.cuda.manual_seed_all(seed_of(INIT_SEED, 4))

    patch = cfg.student.patch_size
    size = cfg.crops.global_crops_size
    mask_generator = MaskingGenerator(input_size=(size // patch, size // patch),
                                      max_num_patches=0.5 * size // patch * size // patch)
    transform = DataAugmentationDINO(cfg.crops.global_crops_scale, cfg.crops.local_crops_scale,
                                     cfg.crops.local_crops_number, global_crops_size=size,
                                     local_crops_size=cfg.crops.local_crops_size)
    collate = partial(keyed_collate, order_seed=order_seed, mask_ratio_tuple=tuple(cfg.ibot.mask_ratio_min_max),
                      mask_probability=cfg.ibot.mask_sample_probability, n_tokens=(size // patch) ** 2,
                      mask_generator=mask_generator, dtype=torch.bfloat16)
    end_iter = min(total_iters, args.max_iters) if args.max_iters else total_iters
    if args.stop_epoch:
        end_iter = min(end_iter, round(args.stop_epoch * total_iters / int(cfg.optim.epochs)))
    sampler = KeySampler(n, end_iter * batch_size, order_seed, start=start_iter * batch_size)
    loader = torch.utils.data.DataLoader(KeyedImages(images, order_seed, transform), batch_size=batch_size,
                                         sampler=sampler, num_workers=args.workers, collate_fn=collate,
                                         pin_memory=True, drop_last=True, persistent_workers=False,
                                         prefetch_factor=4 if args.workers else None)

    run = {"arm": arm, "scanner": scanner, "order_seed": order_seed, "init_seed": INIT_SEED, "slides": len(slides),
           "images": n, "batch_size": batch_size, "epoch_length": epoch_length, "total_iters": total_iters,
           "end_iter": end_iter, "lr": float(cfg.optim.lr), "init_hashes": init_hashes,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "smoke": bool(args.max_iters or args.max_slides or args.epochs)}
    run_json = out / ("run.json" if start_iter == 0 else f"run_resume_{start_iter}.json")
    C.write_json(run_json, run)
    log_path = out / "log.csv"
    if start_iter == 0 and log_path.exists():
        log_path.unlink()
    elif log_path.exists():  # drop rows of the interrupted run beyond the resumed state
        previous = pd.read_csv(log_path)
        previous[previous.iteration <= start_iter].to_csv(log_path, index=False)
    log_rows: list[dict] = []
    first_keys = []
    iteration = start_iter
    tick = time.time()
    for batch in loader:
        lr, wd, mom = lr_s[iteration], wd_s[iteration], mom_s[iteration]
        teacher_temp, last_lr = temp_s[iteration], last_s[iteration]
        for group in optimizer.param_groups:
            group["weight_decay"] = wd * group["wd_multiplier"]
            group["lr"] = (last_lr if group["is_last_layer"] else lr) * group["lr_multiplier"]
        optimizer.zero_grad(set_to_none=True)
        loss_dict = train_step(batch, student, teacher, losses, cfg, teacher_temp)
        if cfg.optim.clip_grad:
            for module in student.values():
                torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.optim.clip_grad)
        optimizer.step()
        with torch.no_grad():
            for key in student:
                s_params = [p for p in student[key].parameters()]
                t_params = [p for p in teacher[key].parameters()]
                torch._foreach_mul_(t_params, mom)
                torch._foreach_add_(t_params, s_params, alpha=1 - mom)
        values = {k: float(v.item()) for k, v in loss_dict.items()}
        if not all(math.isfinite(v) for v in values.values()):
            raise FloatingPointError(f"non-finite loss at iteration {iteration}: {values}")
        if iteration < start_iter + 3:
            first_keys.append(batch["keys"][:4])
        iteration += 1
        if iteration % 50 == 0 or iteration == end_iter:
            elapsed = time.time() - tick
            row = {"iteration": iteration, "epoch": iteration / epoch_length, "lr": lr, "wd": wd, "momentum": mom,
                   "teacher_temp": teacher_temp, "total_loss": sum(values.values()), **values,
                   "seconds": elapsed, "images_per_s": 50 * batch_size / max(elapsed, 1e-9)}
            log_rows.append(row)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            pd.DataFrame(log_rows).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
            log_rows = []
            tick = time.time()
        if iteration in save_at and save_at[iteration] > 0:
            save_teacher(out / f"teacher_e{save_at[iteration]:03d}.pt", teacher, cfg,
                         {"arm": arm, "iteration": iteration, "epoch": save_at[iteration]})
        if (iteration % (10 * epoch_length) == 0 or iteration == end_iter) and not args.max_iters:
            temporary = resume.with_suffix(".tmp")
            torch.save({"student": student.state_dict(), "teacher": teacher.state_dict(),
                        "optimizer": optimizer.state_dict(), "iteration": iteration,
                        "cuda_rng": torch.cuda.get_rng_state()}, temporary)
            temporary.replace(resume)
    run.update({"finished_iteration": iteration, "first_batch_keys": first_keys,
                "wall_s": round(time.time() - started, 1)})
    C.write_json(run_json, run)
    if args.max_iters:
        save_teacher(out / "teacher_smoke.pt", teacher, cfg, {"arm": arm, "iteration": iteration, "epoch": iteration / epoch_length})
    print(json.dumps({"status": "complete", "arm": arm, "iteration": iteration}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
