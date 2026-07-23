"""Train the Exp-06 low-frequency harmonizer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get(
        "PRENORM_CONFIG", "configs/experiments/exp06a_lf16_10slide.yaml"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    from prenorm.callbacks import MilestoneCheckpoint
    from prenorm.exp06.data import Exp06DataModule
    from prenorm.exp06.module import Exp06Module
    from utils.config import load_config, to_dict

    cfg = load_config(args.config)
    if int(cfg.design_version) != 6:
        raise ValueError("train_exp06.py requires design_version: 6")
    if args.fast_dev_run:
        cfg.trainer.fast_dev_run = True
        cfg.loader.num_workers = 0
        cfg.loader.batches_per_epoch = 3
        cfg.loader.val_batches = 3
        cfg.wandb.mode = "offline"
    output_dir = Path(cfg.paths.output_dir)
    Path(cfg.paths.ckpt_dir).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(to_dict(cfg), indent=2, sort_keys=True) + "\n"
    )

    pl.seed_everything(int(cfg.runtime.seed), workers=True)
    data = Exp06DataModule(cfg)
    module = Exp06Module(cfg)
    logger = WandbLogger(
        project=cfg.wandb.project, name=cfg.wandb.name,
        mode=cfg.wandb.mode, save_dir=cfg.paths.wandb_dir,
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.paths.ckpt_dir, monitor="val/total", mode="min",
            save_top_k=1, save_last=True, filename="best",
        ),
        MilestoneCheckpoint(cfg.paths.ckpt_dir, list(cfg.trainer.milestone_epochs)),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = pl.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=cfg.trainer.accelerator,
        devices=int(cfg.trainer.devices),
        precision=cfg.optim.precision,
        accumulate_grad_batches=int(cfg.trainer.accumulate_grad_batches),
        gradient_clip_val=float(cfg.optim.grad_clip),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        check_val_every_n_epoch=1,
        fast_dev_run=bool(cfg.trainer.fast_dev_run),
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(module, datamodule=data, ckpt_path=args.resume)
    if not bool(cfg.trainer.fast_dev_run):
        trainer.save_checkpoint(str(Path(cfg.paths.ckpt_dir) / "last.ckpt"))


if __name__ == "__main__":
    main()
