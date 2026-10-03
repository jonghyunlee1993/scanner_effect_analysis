#!/usr/bin/env python3
"""RV18 stage 1 (GPU): continue UNI's self-supervised training on AT2/GT450 mixtures (blur off).

Protocol: ``uni_continued_protocol.md``. UNI v1 (ViT-L/16) is loaded into the vendored DINOv2
``vit_large``; its state-dict keys match one to one and ``mask_token`` starts at zero. A parity
check against the paper's timm UNI runs first (cosine >= 0.999 on real patches, fatal). Data,
composition arms (p = AT2 share; RV17 assignment, seed 17), pairing and the blur-off
augmentation are those of RV17; heads are new (16,384 prototypes). The backbone is frozen for the
first half epoch, ramps up over a quarter epoch and then follows the cosine schedule (peak 1e-4
at the top block, layer-wise decay 0.9). Student blocks use activation checkpointing.

Outputs (``results/uni_continued/train/<arm>/``): ``teacher_e{000,001,002,003}.pt``, ``log.csv``,
``locations.csv.gz``, ``config.yaml``, ``run.json``, ``parity.json``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import glob
import json
import math
import os
import tempfile
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

REVISION = Path(__file__).resolve().parent
if str(REVISION) not in sys.path:
    sys.path.insert(0, str(REVISION))

import scanner_composition_pretrain as P  # noqa: E402  (adds the vendored DINOv2 to sys.path)
import scanner_mixture_pretrain as X  # noqa: E402
from scanner_composition_pretrain import C  # noqa: E402

OUTPUT = REVISION / "results/uni_continued"
OUTPUTS = {"uni": OUTPUT, "virchow2": REVISION / "results/virchow2_continued"}
VIRCHOW2_CHECKPOINT = Path("/mnt/isilon/oldridge_lab/leej/PFM_alignment/datasets/image_only_pan_v1/hf_cache/"
                           "models--paige-ai--Virchow2/snapshots/3158645804b69e3f3bc4439d4116edddf0840a72/"
                           "pytorch_model.bin")  # the manuscript's Virchow2 checkpoint (scripts/features/review_multiencoder_scanner.py)
ARMS = {**X.ARMS, "p050": 0.5}  # p050 available for round 2
ARM_ORDER = X.ARM_ORDER  # round-1 array order
ORDER_SEED = 0
EPOCHS = 3
BACKBONE_PEAK_LR = 1e-4
HEAD_PEAK_LR = 2e-3
OVERRIDES = {"student": {"arch": "vit_large", "patch_size": 16, "drop_path_rate": 0.1},
             "crops": {"local_crops_size": 96},
             "dino": {"head_n_prototypes": 16384}, "ibot": {"head_n_prototypes": 16384},
             "optim": {"epochs": EPOCHS}}


BASE_OVERRIDES = {
    "uni": {},
    # Virchow2: ViT-H/14, 4 registers, SwiGLU (hidden 3416 per branch = dinov2 swiglufused with mlp_ratio 4), RV19
    "virchow2": {"student": {"arch": "vit_huge2", "patch_size": 14, "ffn_layer": "swiglufused",
                             "num_register_tokens": 4}, "crops": {"local_crops_size": 98}},
}


def vit_huge2(patch_size=14, num_register_tokens=0, in_chans=3, channel_adaptive=False, **kwargs):
    """Virchow2's ViT-H/14 in the vendored DINOv2 code (embedding 1280, depth 32, 16 heads)."""
    from dinov2.layers import MemEffAttention, NestedTensorBlock as Block
    from dinov2.models.vision_transformer import DinoVisionTransformer

    return DinoVisionTransformer(patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4.0,
                                 block_fn=partial(Block, attn_class=MemEffAttention),
                                 num_register_tokens=num_register_tokens, in_chans=in_chans,
                                 channel_adaptive=channel_adaptive, **kwargs)


def register_vit_huge2() -> None:
    from dinov2.models import vision_transformer as vits

    vits.vit_huge2 = vit_huge2


def load_virchow2_into(backbone) -> dict:
    import torch

    state = torch.load(VIRCHOW2_CHECKPOINT, map_location="cpu", weights_only=True)
    renamed = {}
    for key, value in state.items():
        key = key.replace(".mlp.fc1.", ".mlp.w12.").replace(".mlp.fc2.", ".mlp.w3.")
        renamed[key] = value
    # timm adds positions to the register tokens (pos_embed covers cls + 4 registers + patches);
    # DINOv2 inserts registers after the position embedding, so fold their positions in.
    pos = renamed.pop("pos_embed")
    n_reg = renamed["reg_token"].shape[1]
    renamed["register_tokens"] = renamed.pop("reg_token") + pos[:, 1:1 + n_reg]
    renamed["pos_embed"] = torch.cat([pos[:, :1], pos[:, 1 + n_reg:]], dim=1)
    result = backbone.load_state_dict(renamed, strict=False)
    if list(result.missing_keys) != ["mask_token"] or result.unexpected_keys:
        raise ValueError(f"Virchow2 -> vit_huge2 mismatch: missing {result.missing_keys}, unexpected {result.unexpected_keys}")
    with torch.no_grad():
        backbone.mask_token.zero_()
    return {"missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}


def parity_check_virchow2(backbone, images: np.ndarray) -> dict:
    """Converted backbone vs the manuscript's timm Virchow2 (CLS + mean patch token), fp32."""
    import timm
    import torch
    import torch.nn.functional as F

    reference_model = timm.create_model(
        "vit_huge_patch14_224", pretrained=False, img_size=224, init_values=1e-5, num_classes=0, reg_tokens=4,
        mlp_ratio=5.3375, global_pool="", dynamic_img_size=True, mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU)
    reference_model.load_state_dict(torch.load(VIRCHOW2_CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    reference_model = reference_model.eval().cuda()
    mean = torch.tensor((0.485, 0.456, 0.406), device="cuda").view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device="cuda").view(1, 3, 1, 1)
    x = torch.from_numpy(images).cuda().permute(0, 3, 1, 2).float() / 255.0
    x = (F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1) - mean) / std
    backbone = backbone.cuda().eval()
    with torch.no_grad():
        tokens = reference_model.forward_features(x)
        prefix = int(reference_model.num_prefix_tokens)
        reference = F.normalize(torch.cat([tokens[:, 0], tokens[:, prefix:].mean(1)], -1).float(), dim=-1)
        out = backbone.forward_features(x)
        ours = F.normalize(torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(1)], -1).float(), dim=-1)
    cosine = (reference * ours).sum(-1).cpu().numpy()
    del reference_model
    torch.cuda.empty_cache()
    return {"n": int(len(cosine)), "min_cosine": float(cosine.min()), "mean_cosine": float(cosine.mean())}


def uni_weights() -> Path:
    paths = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--MahmoodLab--uni/snapshots/*/pytorch_model.bin"))
    if len(paths) != 1:
        raise FileNotFoundError(f"UNI weights: {paths}")
    return Path(paths[0])


def load_uni_into(backbone) -> dict:
    import torch

    state = torch.load(uni_weights(), map_location="cpu", weights_only=True)
    result = backbone.load_state_dict(state, strict=False)
    if list(result.missing_keys) != ["mask_token"] or result.unexpected_keys:
        raise ValueError(f"UNI -> vit_large mismatch: missing {result.missing_keys}, unexpected {result.unexpected_keys}")
    with torch.no_grad():
        backbone.mask_token.zero_()
    return {"missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}


def parity_check(backbone, images: np.ndarray) -> dict:
    """Converted backbone vs the paper's timm UNI (prenorm.embedding.load_uni path) on real patches."""
    import torch
    import torch.nn.functional as F
    from prenorm.embedding import load_uni

    device = torch.device("cuda")
    timm_model, size, mean, std = load_uni(device)
    x = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    x = (x - mean) / std
    backbone = backbone.to(device).eval()
    with torch.no_grad():
        reference = F.normalize(timm_model(x).float(), dim=-1)
        ours = F.normalize(backbone.forward_features(x)["x_norm_clstoken"].float(), dim=-1)
    cosine = (reference * ours).sum(-1).cpu().numpy()
    del timm_model
    torch.cuda.empty_cache()
    return {"n": int(len(cosine)), "min_cosine": float(cosine.min()), "mean_cosine": float(cosine.mean())}


class CheckpointedBlock:
    """Wraps a transformer block so its activations are recomputed in backward (student only)."""

    def __new__(cls, block):
        import torch
        from torch.utils.checkpoint import checkpoint

        class _Wrapped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.block = inner

            def forward(self, x):
                if self.training and torch.is_grad_enabled() and any(p.requires_grad for p in self.block.parameters()):
                    return checkpoint(self.block, x, use_reentrant=False)
                return self.block(x)

        return _Wrapped(block)


def build_config(base: str = "uni"):
    from omegaconf import OmegaConf

    return OmegaConf.merge(P.build_config(), OmegaConf.create(OVERRIDES), OmegaConf.create(BASE_OVERRIDES[base]))


def grouped_params(cfg, student) -> list[dict]:
    from dinov2.utils.param_groups import fuse_params_groups, get_params_groups_with_decay

    groups = []
    for key, module in student.items():
        fused = fuse_params_groups(get_params_groups_with_decay(
            model=module, lr_decay_rate=cfg.optim.layerwise_decay, patch_embed_lr_mult=cfg.optim.patch_embed_lr_mult))
        for group in fused:
            group["foreach"] = True
            group["module"] = key
        groups += fused
    return groups


def schedules(epoch_length: int):
    from dinov2.utils.utils import CosineScheduler

    total = EPOCHS * epoch_length
    freeze, ramp = epoch_length // 2, epoch_length // 4
    head = CosineScheduler(HEAD_PEAK_LR, 1e-6, total, warmup_iters=freeze, start_warmup_value=0)
    backbone = CosineScheduler(BACKBONE_PEAK_LR, 1e-6, total)
    last = CosineScheduler(HEAD_PEAK_LR, 1e-6, total, warmup_iters=freeze, start_warmup_value=0)
    last.schedule[:freeze] = 0
    wd = CosineScheduler(0.04, 0.4, total)
    momentum = CosineScheduler(0.992, 1.0, total)
    temp = CosineScheduler(0.07, 0.07, epoch_length, warmup_iters=epoch_length, start_warmup_value=0.04)

    def backbone_lr(it: int) -> float:
        if it < freeze:
            return 0.0
        return backbone[it] * min(1.0, (it - freeze + 1) / ramp)

    return {"head": head, "backbone": backbone_lr, "last": last, "wd": wd, "momentum": momentum, "temp": temp,
            "freeze": freeze, "ramp": ramp, "total": total}


def main() -> None:
    global EPOCHS, BACKBONE_PEAK_LR, ORDER_SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True, help="index into --arms")
    parser.add_argument("--arms", default=",".join(ARM_ORDER), help="comma-separated arm names (p100, p080, p050, p020, p000)")
    parser.add_argument("--suffix", default="", help="run-name suffix for round-2 variants, e.g. _s1")
    parser.add_argument("--order-seed", type=int, default=ORDER_SEED)
    parser.add_argument("--backbone-lr", type=float, default=BACKBONE_PEAK_LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--blur", choices=("off", "on"), default="off", help="on = standard DataAugmentationDINO blur")
    parser.add_argument("--stop-epoch", type=int, default=0,
                        help="stop after this epoch's checkpoint; schedules keep the full --epochs length")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--base", choices=sorted(BASE_OVERRIDES), default="uni", help="starting PFM (RV18 uni, RV19 virchow2)")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-slides", type=int, default=0, help="smoke test only")
    parser.add_argument("--max-iters", type=int, default=0, help="smoke test only")
    args = parser.parse_args()
    args.output_root = args.output_root or OUTPUTS[args.base]
    arm = args.arms.split(",")[args.task_index]
    p = ARMS[arm] if arm in ARMS else int(arm[1:4]) / 100  # pXXX = AT2 share XXX%
    if not (arm.startswith("p") and len(arm) == 4 and 0.0 <= p <= 1.0):
        raise ValueError(f"arm name must be pXXX: {arm}")
    EPOCHS, BACKBONE_PEAK_LR, ORDER_SEED = args.epochs, args.backbone_lr, args.order_seed
    run_name = arm + args.suffix

    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from dinov2.data.augmentations import DataAugmentationDINO
    from dinov2.data.masking import MaskingGenerator
    from dinov2.loss import DINOLoss, KoLeoLoss, iBOTPatchLoss

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = True
    out = args.output_root / "train" / run_name
    out.mkdir(parents=True, exist_ok=True)
    dist.init_process_group("nccl", init_method=f"file://{Path(tempfile.mkdtemp()) / 'pg'}", rank=0, world_size=1)

    register_vit_huge2()
    cfg = build_config(args.base)
    OmegaConf.save(cfg, out / "config.yaml")
    started = time.time()
    slides = P.pretraining_slides(args.max_slides)
    images, index = X.load_mixed_images(slides, p)
    C.write_frame(out / "locations.csv.gz", index)
    n = len(images)
    batch_size = int(cfg.train.batch_size_per_gpu)
    epoch_length = n // batch_size
    sched = schedules(epoch_length)
    end_iter = min(sched["total"], args.max_iters) if args.max_iters else sched["total"]
    if args.stop_epoch:
        end_iter = min(end_iter, args.stop_epoch * epoch_length)
    share = float((index.scanner == "at2").mean())
    print(f"arm {arm}: AT2 share {share:.4f}; {n} images in {time.time() - started:.0f}s; epoch {epoch_length} iters, "
          f"total {sched['total']}, backbone frozen for {sched['freeze']}", flush=True)

    student, teacher = P.build_networks(cfg)  # random backbones + new heads (init seed 0)
    loader_fn = load_uni_into if args.base == "uni" else load_virchow2_into
    load_report = {"student": loader_fn(student["backbone"]), "teacher": loader_fn(teacher["backbone"])}
    rng = np.random.default_rng(0)
    probe = images[np.sort(rng.choice(n, size=min(32, n), replace=False))]
    parity = (parity_check if args.base == "uni" else parity_check_virchow2)(teacher["backbone"], probe)
    C.write_json(out / "parity.json", {**parity, "load": load_report})
    print(f"parity with the manuscript's timm {args.base}: {parity}", flush=True)
    if parity["min_cosine"] < 0.999:
        raise ValueError(f"UNI conversion parity failed: {parity}")
    init_hashes = {"student": P.state_hash(student), "teacher": P.state_hash(teacher)}
    for i, block in enumerate(student["backbone"].blocks):
        student["backbone"].blocks[i] = CheckpointedBlock(block)
    student.cuda().train()
    teacher.cuda().eval()
    losses = (DINOLoss(cfg.dino.head_n_prototypes).cuda(), iBOTPatchLoss(cfg.ibot.head_n_prototypes).cuda(), KoLeoLoss().cuda())
    # Build the optimizer while every parameter requires grad: dinov2's param-group helper skips
    # frozen parameters, so freezing first would leave the backbone out of the optimizer.
    optimizer = torch.optim.AdamW(grouped_params(cfg, student), betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))
    backbone_ids = {id(q) for q in student["backbone"].parameters()}
    in_optimizer = sum(id(q) in backbone_ids for g in optimizer.param_groups for q in g["params"])
    if in_optimizer != len(backbone_ids):
        raise RuntimeError(f"backbone parameters in optimizer: {in_optimizer} of {len(backbone_ids)}")
    student["backbone"].requires_grad_(False)  # head warm-up
    initial = [q.detach().clone() for q in student["backbone"].parameters()]
    initial_norm = torch.sqrt(sum((q.float() ** 2).sum() for q in initial)).item()

    def relative_change(module) -> float:
        with torch.no_grad():
            diff = sum(((q.float() - q0.float()) ** 2).sum() for q, q0 in zip(module.parameters(), initial))
        return float(torch.sqrt(diff).item() / initial_norm)
    P.save_teacher(out / "teacher_e000.pt", teacher, cfg, {"arm": run_name, "iteration": 0, "epoch": 0})
    torch.cuda.manual_seed_all(P.seed_of(P.INIT_SEED, 4))
    save_at = {e * epoch_length: e for e in range(1, EPOCHS + 1)}

    patch, size = cfg.student.patch_size, cfg.crops.global_crops_size
    mask_generator = MaskingGenerator(input_size=(size // patch, size // patch),
                                      max_num_patches=0.5 * size // patch * size // patch)
    transform = DataAugmentationDINO(cfg.crops.global_crops_scale, cfg.crops.local_crops_scale,
                                     cfg.crops.local_crops_number, global_crops_size=size,
                                     local_crops_size=cfg.crops.local_crops_size)
    if args.blur == "off" and X.disable_blur(transform) != 3:
        raise RuntimeError("expected 3 Gaussian blurs in DataAugmentationDINO")
    collate = partial(P.keyed_collate, order_seed=ORDER_SEED, mask_ratio_tuple=tuple(cfg.ibot.mask_ratio_min_max),
                      mask_probability=cfg.ibot.mask_sample_probability, n_tokens=(size // patch) ** 2,
                      mask_generator=mask_generator, dtype=torch.bfloat16)
    loader = torch.utils.data.DataLoader(P.KeyedImages(images, ORDER_SEED, transform), batch_size=batch_size,
                                         sampler=P.KeySampler(n, end_iter * batch_size, ORDER_SEED),
                                         num_workers=args.workers, collate_fn=collate, pin_memory=True,
                                         drop_last=True, prefetch_factor=2 if args.workers else None)
    run = {"arm": arm, "run": run_name, "epochs": EPOCHS, "at2_share_target": p, "at2_share_realized": share, "order_seed": ORDER_SEED,
           "init": ("UNI v1 (MahmoodLab/uni) -> dinov2 vit_large" if args.base == "uni"
                    else "Virchow2 (paige-ai/Virchow2) -> dinov2 vit_huge2"), "base": args.base, "blur": "off (draw-only identity)" if args.blur == "off" else "on (standard)",
           "images": n, "batch_size": batch_size, "epoch_length": epoch_length, "total_iters": sched["total"],
           "end_iter": end_iter, "backbone_peak_lr": BACKBONE_PEAK_LR, "head_peak_lr": HEAD_PEAK_LR,
           "freeze_iters": sched["freeze"], "ramp_iters": sched["ramp"], "init_hashes": init_hashes,
           "parity": parity, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "smoke": bool(args.max_iters or args.max_slides)}
    C.write_json(out / "run.json", run)
    log_path = out / "log.csv"
    if log_path.exists():
        log_path.unlink()
    first_keys, iteration, tick, last_logged = [], 0, time.time(), 0
    for batch in loader:
        if iteration == sched["freeze"]:
            student["backbone"].requires_grad_(True)
            print(f"iteration {iteration}: backbone unfrozen", flush=True)
        head_lr, backbone_lr, last_lr = sched["head"][iteration], sched["backbone"](iteration), sched["last"][iteration]
        wd, mom, teacher_temp = sched["wd"][iteration], sched["momentum"][iteration], sched["temp"][iteration]
        for group in optimizer.param_groups:
            group["weight_decay"] = wd * group["wd_multiplier"]
            base = last_lr if group["is_last_layer"] else (backbone_lr if group["module"] == "backbone" else head_lr)
            group["lr"] = base * group["lr_multiplier"]
        optimizer.zero_grad(set_to_none=True)
        loss_dict = P.train_step(batch, student, teacher, losses, cfg, teacher_temp)
        for module in student.values():
            params = [q for q in module.parameters() if q.grad is not None]
            if params:
                torch.nn.utils.clip_grad_norm_(params, cfg.optim.clip_grad)
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
        if iteration % 25 == 0 or iteration == end_iter:
            elapsed = time.time() - tick
            row = {"iteration": iteration, "epoch": iteration / epoch_length, "head_lr": head_lr,
                   "backbone_lr": backbone_lr, "wd": wd, "momentum": mom, "teacher_temp": teacher_temp,
                   "total_loss": sum(values.values()), **values, "seconds": elapsed,
                   "images_per_s": (iteration - last_logged) * batch_size / max(elapsed, 1e-9),
                   "max_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                   "student_backbone_rel_change": relative_change(student["backbone"]),
                   "teacher_backbone_rel_change": relative_change(teacher["backbone"])}
            if iteration >= sched["freeze"] + 25 and row["student_backbone_rel_change"] == 0.0:
                raise RuntimeError(f"student backbone unchanged at iteration {iteration} after unfreezing")
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
            tick, last_logged = time.time(), iteration
        if iteration in save_at:
            # teacher blocks are plain (only student blocks are wrapped), so keys match vit_large
            P.save_teacher(out / f"teacher_e{save_at[iteration]:03d}.pt", teacher, cfg,
                           {"arm": run_name, "iteration": iteration, "epoch": save_at[iteration]})
    run.update({"finished_iteration": iteration, "first_batch_keys": first_keys, "wall_s": round(time.time() - started, 1)})
    C.write_json(out / "run.json", run)
    print(json.dumps({"status": "complete", "arm": arm, "iteration": iteration}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
