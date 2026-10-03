"""Panel A execution entrypoints for missing stain and GAN conditions.

This module intentionally leaves the historical ``scanner_gan`` modules and
their outputs untouched.  It reuses their data/model/metric primitives while
generalizing the already frozen full-training contracts from AT2/GT450 to the
four missing AT2/non-reference scanner pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import h5py
import pandas as pd
import torch
import torch.nn.functional as F

from scanner_gan.data import (
    INDEX_COLUMNS,
    PairedScannerDataset,
    SlideBalancedSampler,
    UnpairedScannerDataset,
    UnpairedSlideBalancedSampler,
    WorkerLocalH5Handles,
    assign_split_roles,
    load_sample_index,
    scanner_lookup,
)
from scanner_gan.models import GANLoss, ImagePool, build_cyclegan_models, build_pix2pix_models, count_parameters
from scanner_gan.train import (
    Pix2PixConfig,
    _amp_dtype,
    _loader,
    central_crop,
    pass_learning_rate as pix2pix_pass_learning_rate,
    save_checkpoint as save_pix2pix_checkpoint,
    seed_everything,
    set_learning_rate,
    set_requires_grad,
    sha256,
    utc_now,
    validate as validate_pix2pix,
    write_json,
)
from scanner_gan.train import TRAINING_VERSION as PIX2PIX_CHECKPOINT_VERSION
from scanner_gan.train_cyclegan import (
    CycleGANConfig,
    amp_dtype as cyclegan_amp_dtype,
    make_loader as make_cyclegan_loader,
    pass_learning_rate as cyclegan_pass_learning_rate,
    save_checkpoint as save_cyclegan_checkpoint,
    validate as validate_cyclegan,
)
from scanner_gan.train_cyclegan import TRAINING_VERSION as CYCLEGAN_CHECKPOINT_VERSION
from scanner_batch_extensions import (
    ENDPOINTS,
    absolute_measurements,
    contrasts_from_absolute,
    rowwise_correlation,
)
from manuscript_completion.stain import (
    fit_macenko_od,
    fit_vahadane_od,
    normalize,
    parameters_from_json,
    parameters_to_json,
    tissue_od,
)


PANEL_A_TARGETS = ("versa", "akoya", "gt450", "s360", "s60")
MISSING_GAN_TARGETS = ("versa", "akoya", "s360", "s60")
PANEL_A_TRAINING_VERSION = "manuscript_panel_a_full_training_v1"
FINAL_STAIN_SUBDIR = "01_stain_v4_mu_convergence"
SUPERSEDED_STAIN_SUBDIRS = {
    "", ".", "01_stain", "01_stain_v2", "01_stain_v2_macenko",
    "01_stain_v3", "01_stain_v3_mu_vahadane",
}
RUNTIME_DISTRIBUTIONS = (
    "numpy", "pandas", "scipy", "scikit-learn", "scikit-image", "h5py",
    "torch", "torchvision", "timm",
)


def runtime_environment_provenance() -> dict[str, Any]:
    """Return the version-sensitive runtime identity used by the formal jobs."""

    torchvision = importlib.import_module("torchvision")
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            distribution: importlib_metadata.version(distribution)
            for distribution in RUNTIME_DISTRIBUTIONS
        },
        "torch_cuda_build": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "torch_module_version": str(torch.__version__),
        "torchvision_module_version": str(torchvision.__version__),
    }


def verify_runtime_contract(manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Fail closed when a queued job sees code or inputs outside its contract."""

    selected = manifest_path or os.environ.get("PANEL_A_CONTRACT_MANIFEST")
    if not selected:
        raise RuntimeError("PANEL_A_CONTRACT_MANIFEST is required for Panel A entrypoints")
    manifest_file = Path(selected).resolve()
    expected_manifest_sha = os.environ.get("PANEL_A_CONTRACT_SHA256")
    if not expected_manifest_sha:
        raise RuntimeError("PANEL_A_CONTRACT_SHA256 is required for Panel A entrypoints")
    observed_manifest_sha = sha256(manifest_file)
    if observed_manifest_sha != expected_manifest_sha:
        raise RuntimeError(f"Panel A contract manifest hash mismatch: expected {expected_manifest_sha}, observed {observed_manifest_sha}")
    payload = json.loads(manifest_file.read_text())
    if payload.get("status") != "frozen" or not payload.get("files"):
        raise RuntimeError(f"Panel A contract is not frozen: {manifest_file}")
    repository = Path(__file__).resolve().parents[2]
    mismatches: list[str] = []
    for relative, expected in payload["files"].items():
        path = repository / relative
        observed = sha256(path) if path.is_file() else "missing"
        if observed != expected:
            mismatches.append(f"{relative}: expected {expected}, observed {observed}")
    for name, item in payload.get("inputs", {}).items():
        path = Path(item["path"])
        observed = sha256(path) if path.is_file() else "missing"
        if observed != item["sha256"]:
            mismatches.append(f"input {name}: expected {item['sha256']}, observed {observed}")
    if payload.get("formal_panel_a") is True:
        expected_runtime = payload.get("environment_provenance", {}).get("runtime")
        observed_runtime = runtime_environment_provenance()
        if expected_runtime != observed_runtime:
            mismatches.append(
                "runtime environment provenance mismatch: "
                f"expected {expected_runtime}, observed {observed_runtime}"
            )
        stage = os.environ.get("PANEL_A_STAGE", "")
        stage_contracts = payload.get("runtime_environments", {})
        if stage not in stage_contracts:
            mismatches.append(f"unbound PANEL_A_STAGE: {stage!r}")
        else:
            for variable, specification in stage_contracts[stage].items():
                if not isinstance(specification, Mapping):
                    mismatches.append(f"invalid runtime binding for {stage}/{variable}")
                    continue
                actual = os.environ.get(variable)
                expected_value = str(specification.get("value", ""))
                if actual is None:
                    mismatches.append(f"missing runtime environment {variable} for {stage}")
                    continue
                if specification.get("kind") == "path":
                    actual = str(Path(actual).resolve())
                    expected_value = str(Path(expected_value).resolve())
                if actual != expected_value:
                    mismatches.append(
                        f"runtime environment {variable} for {stage}: "
                        f"expected {expected_value}, observed {actual}"
                    )
        if stage in {"gan_predict", "gan_image_eval", "gan_uni_eval"}:
            method = os.environ.get("JOB_METHOD", "").strip().lower()
            subdir = os.environ.get("JOB_METHOD_SUBDIR", "")
            expected_subdir = {"pix2pix": "03_pix2pix", "cyclegan": "04_cyclegan"}.get(method)
            if expected_subdir is None or subdir != expected_subdir:
                mismatches.append(
                    f"invalid GAN method/subdir binding for {stage}: "
                    f"{method!r}/{subdir!r}"
                )
    if mismatches:
        raise RuntimeError("Panel A runtime contract mismatch:\n" + "\n".join(mismatches))
    return payload


def write_frame(path: str | Path, frame: pd.DataFrame) -> None:
    """Atomically write CSV with compression determined from the final name."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def ensure_gzip_csv(path: str | Path) -> bool:
    """Repair historical plain-text ``*.csv.gz`` output and return True if changed."""

    path = Path(path)
    if not path.name.endswith(".gz"):
        return False
    with path.open("rb") as stream:
        magic = stream.read(2)
    if magic == b"\x1f\x8b":
        return False
    frame = pd.read_csv(path, compression=None, dtype={"slide_id": str})
    write_frame(path, frame)
    summary_path = path.with_name(path.name.removesuffix(".csv.gz") + ".summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["output_sha256"] = sha256(path)
        summary["gzip_repaired"] = True
        write_json(summary_path, summary)
    return True


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


class GenericPairedScannerDataset(PairedScannerDataset):
    """The existing registered-pair loader without its legacy direction gate."""

    def __init__(
        self,
        sample_index,
        source_scanner: str,
        *,
        target_scanner: str,
        split_role: str | None = None,
        augment: bool = False,
        seed: int = 0,
        crop_size: int = 252,
        max_open_files: int = 8,
        allow_bidirectional_reference_pair: bool = False,
    ) -> None:
        source_scanner = str(source_scanner).strip().lower()
        target_scanner = str(target_scanner).strip().lower()
        if source_scanner == target_scanner:
            raise ValueError("source_scanner and target_scanner must differ")
        if source_scanner != "at2" or target_scanner not in PANEL_A_TARGETS:
            raise ValueError(f"Panel A mapping must be AT2->non-reference, got {source_scanner}->{target_scanner}")
        frame = sample_index.copy()
        if split_role is not None:
            if "split_role" not in frame:
                raise ValueError("split_role filtering requires assigned roles")
            frame = frame[frame["split_role"] == str(split_role)].copy()
        if frame.empty:
            raise ValueError("paired dataset contains no samples")
        missing = set(INDEX_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"sample index is missing columns: {sorted(missing)}")
        orders = frame["scanner_order"].drop_duplicates().tolist()
        if len(orders) != 1:
            raise ValueError(f"dataset has multiple scanner orders: {orders}")
        self.scanner_names = tuple(name.strip().lower() for name in str(orders[0]).split(","))
        lookup = scanner_lookup(self.scanner_names)
        if source_scanner not in lookup or target_scanner not in lookup:
            raise ValueError(f"mapping unavailable in {self.scanner_names}")
        self.source_scanner = source_scanner
        self.target_scanner = target_scanner
        self.source_scanner_index = lookup[source_scanner]
        self.target_scanner_index = lookup[target_scanner]
        self.sample_index = frame.reset_index(drop=True)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.crop_size = int(crop_size)
        self.handles = WorkerLocalH5Handles(self.scanner_names, max_open_files=max_open_files)


def survey_vahadane_source_convergence(
    locked_index_path: Path,
    output_path: Path,
    fold: int,
    *,
    samples_per_fold: int = 10,
    max_iterations: tuple[int, ...] = (1000, 2000, 4000),
    tolerances: tuple[float, ...] = (1e-3,),
    alphas: tuple[float, ...] = (1e-3,),
    solvers: tuple[str, ...] = ("cd",),
) -> dict[str, Any]:
    """Survey source-only AT2 fits before freezing the Vahadane iteration cap."""

    locked = pd.read_csv(locked_index_path, dtype={"slide_id": str})
    rows = locked[locked.fold.astype(int) == int(fold)].sort_values(["slide_id", "location_index"])
    if rows.empty:
        raise ValueError(f"no locked source rows for fold {fold}")
    positions = np.unique(np.linspace(0, len(rows) - 1, min(samples_per_fold, len(rows)), dtype=int))
    selected = rows.iloc[positions]
    records: list[dict[str, Any]] = []
    for item in selected.itertuples(index=False):
        names = tuple(str(item.scanner_order).split(",")); source_index = scanner_lookup(names)["at2"]
        with h5py.File(str(item.cache_path), "r") as store:
            source = np.asarray(store["images"][int(item.location_index), source_index], dtype=np.uint8)
        od = tissue_od(source)
        seed_specs = [(f"evaluation:{scanner}", stable_seed(str(item.slide_id), int(item.location_index), scanner, "vahadane")) for scanner in PANEL_A_TARGETS]
        for context, seed in seed_specs:
            for solver in solvers:
                for alpha in alphas:
                    for tolerance in tolerances:
                        for max_iter in max_iterations:
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                fit = fit_vahadane_od(od, seed=seed, max_pixels=4096, max_iter=max_iter, tolerance=tolerance,alpha=alpha,solver=solver)
                            record={"fold":int(fold),"slide_id":str(item.slide_id),"location_index":int(item.location_index),"seed_context":context,"seed":int(seed),"solver":solver,"alpha_W":float(alpha),"max_iter":int(max_iter),"tolerance":float(tolerance),"fit_valid":fit is not None,"converged":bool(fit is not None and fit.converged),"iterations":int(fit.iterations) if fit is not None else 0,"basis_condition":float(np.linalg.cond(fit.basis)) if fit is not None else float("nan"),"reconstruction_error":float(fit.reconstruction_error) if fit is not None else float("nan"),"warning_count":len(caught)}
                            for row_index in range(3):
                                for column_index in range(2): record[f"basis_{row_index}{column_index}"]=float(fit.basis[row_index,column_index]) if fit is not None else float("nan")
                            records.append(record)
    # Exact reference sentinel seeds are also surveyed on the first held-out
    # source input; no target scanner images or outcome metrics are accessed.
    first = rows.iloc[0]
    names=tuple(str(first.scanner_order).split(",")); source_index=scanner_lookup(names)["at2"]
    with h5py.File(str(first.cache_path),"r") as store: source=np.asarray(store["images"][int(first.location_index),source_index],dtype=np.uint8)
    od=tissue_od(source)
    for scanner in PANEL_A_TARGETS:
        seed=stable_seed("sentinel","vahadane",scanner,fold)
        for solver in solvers:
            for alpha in alphas:
                for tolerance in tolerances:
                    for max_iter in max_iterations:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always"); fit=fit_vahadane_od(od,seed=seed,max_pixels=4096,max_iter=max_iter,tolerance=tolerance,alpha=alpha,solver=solver)
                        record={"fold":int(fold),"slide_id":str(first.slide_id),"location_index":int(first.location_index),"seed_context":f"reference_sentinel:{scanner}","seed":int(seed),"solver":solver,"alpha_W":float(alpha),"max_iter":int(max_iter),"tolerance":float(tolerance),"fit_valid":fit is not None,"converged":bool(fit is not None and fit.converged),"iterations":int(fit.iterations) if fit is not None else 0,"basis_condition":float(np.linalg.cond(fit.basis)) if fit is not None else float("nan"),"reconstruction_error":float(fit.reconstruction_error) if fit is not None else float("nan"),"warning_count":len(caught)}
                        for row_index in range(3):
                            for column_index in range(2): record[f"basis_{row_index}{column_index}"]=float(fit.basis[row_index,column_index]) if fit is not None else float("nan")
                        records.append(record)
    frame=pd.DataFrame(records); summary=[]
    basis_columns=[f"basis_{row}{column}" for row in range(3) for column in range(2)]
    keys=["slide_id","location_index","seed_context","seed"]
    baseline=frame[(frame.solver=="cd")&np.isclose(frame.alpha_W,.001)&np.isclose(frame.tolerance,.001)&(frame.max_iter==1000)].set_index(keys)
    frame["basis_cosine_to_baseline"]=np.nan; frame["reconstruction_error_ratio_to_baseline"]=np.nan
    for index,row in frame.iterrows():
        key=tuple(row[name] for name in keys)
        if key not in baseline.index or not bool(row.fit_valid): continue
        reference=baseline.loc[key]
        if isinstance(reference,pd.DataFrame): reference=reference.iloc[0]
        if not bool(reference.fit_valid): continue
        basis=row[basis_columns].to_numpy(float).reshape(3,2); base=reference[basis_columns].to_numpy(float).reshape(3,2)
        frame.at[index,"basis_cosine_to_baseline"]=float(np.mean(np.sum(basis*base,axis=0)/(np.linalg.norm(basis,axis=0)*np.linalg.norm(base,axis=0))))
        frame.at[index,"reconstruction_error_ratio_to_baseline"]=float(row.reconstruction_error/max(float(reference.reconstruction_error),1e-12))
    for (solver,alpha,tolerance,max_iter),group in frame.groupby(["solver","alpha_W","tolerance","max_iter"]):
        summary.append({"solver":solver,"alpha_W":float(alpha),"tolerance":float(tolerance),"max_iter":int(max_iter),"fits":len(group),"valid_count":int(group.fit_valid.sum()),"converged_count":int(group.converged.sum()),"nonconverged_count":int((~group.converged).sum()),"basis_condition_max":float(group.basis_condition.max()),"basis_cosine_to_baseline_median":float(group.basis_cosine_to_baseline.median()),"basis_cosine_to_baseline_p05":float(group.basis_cosine_to_baseline.quantile(.05)),"reconstruction_error_median":float(group.reconstruction_error.median()),"reconstruction_error_ratio_to_baseline_median":float(group.reconstruction_error_ratio_to_baseline.median())})
    output_path.parent.mkdir(parents=True,exist_ok=True); write_frame(output_path.with_suffix(".csv.gz"),frame)
    payload={"status":"pass","analysis":"input_only_vahadane_convergence_survey","fold":int(fold),"target_outcomes_accessed":False,"locked_index_sha256":sha256(locked_index_path),"samples_per_fold":len(selected),"summary":summary,"rows_sha256":sha256(output_path.with_suffix(".csv.gz"))}; write_json(output_path,payload); return payload


def _validate_panel_mapping(source: str, target: str) -> tuple[str, str]:
    source = source.strip().lower()
    target = target.strip().lower()
    if source != "at2" or target not in MISSING_GAN_TARGETS:
        raise ValueError(f"new Panel A jobs are restricted to AT2->{MISSING_GAN_TARGETS}, got {source}->{target}")
    return source, target


def _validate_formal_training_settings(max_passes: int, max_steps: int) -> None:
    if int(max_passes) != 200 or int(max_steps) != 0:
        raise ValueError("formal Panel A training requires max_passes=200 and max_steps=0; smoke runs are never formal outputs")


def train_pix2pix(config: Pix2PixConfig) -> dict[str, Any]:
    """Run the frozen 200-pass Pix2Pix contract for one missing direction/fold."""

    _validate_formal_training_settings(config.max_passes, config.max_steps)
    if not torch.cuda.is_available():
        raise RuntimeError("formal Pix2Pix training requires CUDA")
    source, target = _validate_panel_mapping(config.source_scanner, config.target_scanner)
    if config.validation_fold != (config.test_fold + 1) % 5:
        raise ValueError("validation fold must equal (test_fold + 1) mod 5")
    if config.selection_policy != "target_l1":
        raise ValueError("Panel A full training locks image-only target_l1 selection")
    if int(config.selection_every)!=5: raise ValueError("Panel A checkpoint selection grid is every 5 passes")
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path=output_dir/"metrics.jsonl"
    if metrics_path.exists(): raise FileExistsError(f"formal metrics history already exists: {metrics_path}")
    seed_everything(config.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = _amp_dtype(config.amp)

    full = load_sample_index(config.sample_index, verify_caches=False)
    split = assign_split_roles(full, config.test_fold, config.validation_fold)
    train_data = GenericPairedScannerDataset(split, source, target_scanner=target, split_role="train", augment=True, seed=config.seed, crop_size=config.crop_size)
    val_data = GenericPairedScannerDataset(split, source, target_scanner=target, split_role="validation", augment=False, seed=config.seed, crop_size=config.crop_size)
    train_sampler = SlideBalancedSampler(train_data.sample_index, locations_per_slide=config.locations_per_train_slide, seed=config.seed)
    val_sampler = SlideBalancedSampler(val_data.sample_index, locations_per_slide=config.locations_per_validation_slide, seed=config.seed + 1)
    train_loader = _loader(train_data, train_sampler, batch_size=config.batch_size, workers=config.num_workers, device=device)
    val_loader = _loader(val_data, val_sampler, batch_size=config.validation_batch_size, workers=config.num_workers, device=device)
    generator, discriminator = build_pix2pix_models(ngf=config.ngf, ndf=config.ndf, device=device)
    gan_loss = GANLoss("vanilla").to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    manifest: dict[str, Any] = {
        "training_version": PANEL_A_TRAINING_VERSION,
        "architecture_contract": "scanner_gan.train pix2pix_scanner_reference_v2",
        "status": "running",
        "started_utc": utc_now(),
        "direction": f"{source}->{target}",
        "selection_boundary": "image-only inner-validation target L1; UNI forbidden until lock",
        "sample_index": str(Path(config.sample_index).resolve()),
        "sample_index_sha256": sha256(config.sample_index),
        "train_slides": int(train_data.sample_index.slide_id.nunique()),
        "validation_slides": int(val_data.sample_index.slide_id.nunique()),
        "test_slides": int(split.loc[split.split_role == "test", "slide_id"].nunique()),
        "train_samples_per_pass": len(train_sampler),
        "validation_samples_per_pass": len(val_sampler),
        "generator_parameters": count_parameters(generator),
        "discriminator_parameters": count_parameters(discriminator),
        "config": asdict(config),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    global_step, best_l1, best_checkpoint, selected_pass = 0, float("inf"), "", -1
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    reached_max = False
    for pass_index in range(config.max_passes):
        train_sampler.set_epoch(pass_index)
        val_sampler.set_epoch(0)
        lr = pix2pix_pass_learning_rate(config, pass_index)
        set_learning_rate((optimizer_g, optimizer_d), lr)
        generator.train(); discriminator.train()
        sums = {"d": 0.0, "g": 0.0, "g_gan": 0.0, "g_l1": 0.0}
        seen = 0
        pass_started = time.monotonic()
        for batch in train_loader:
            source_image = batch["source_image"].to(device, non_blocking=True)
            source_valid = batch["source_valid"].to(device, non_blocking=True)
            target_valid = batch["target_valid"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
                generated = generator(source_image)
                generated_valid = central_crop(generated, config.crop_size)
            set_requires_grad(discriminator, True)
            optimizer_d.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
                real_logits = discriminator(torch.cat((source_valid, target_valid), dim=1))
                fake_logits = discriminator(torch.cat((source_valid, generated_valid.detach()), dim=1))
                loss_d = 0.5 * (gan_loss(real_logits, True) + gan_loss(fake_logits, False))
            loss_d.backward(); optimizer_d.step()
            set_requires_grad(discriminator, False)
            optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
                fake_logits = discriminator(torch.cat((source_valid, generated_valid), dim=1))
                loss_g_gan = gan_loss(fake_logits, True)
                loss_g_l1 = F.l1_loss(generated_valid, target_valid)
                loss_g = loss_g_gan + config.lambda_l1 * loss_g_l1
            loss_g.backward(); optimizer_g.step()
            count = int(source_image.shape[0]); seen += count; global_step += 1
            for key, value in (("d", loss_d), ("g", loss_g), ("g_gan", loss_g_gan), ("g_l1", loss_g_l1)):
                sums[key] += float(value.detach().float()) * count
            if config.max_steps > 0 and global_step >= config.max_steps:
                reached_max = True; break
        validation = validate_pix2pix(generator, val_loader, device, dtype)
        completed = pass_index + 1
        row = {"completed_passes": completed, "global_step": global_step, "learning_rate": lr, "train_samples": seen, **{f"train_{k}": v/max(seen,1) for k,v in sums.items()}, "pass_seconds": time.monotonic()-pass_started, "elapsed_seconds": time.monotonic()-started, "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device)/2**30, **{f"validation_{k}": v for k,v in validation.items()}}
        history.append(row)
        with metrics_path.open("a") as stream: stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        selection = completed % config.selection_every == 0 or completed == config.max_passes or reached_max
        if selection and validation["l1_normalized"] < best_l1:
            best_l1 = float(validation["l1_normalized"])
            path = output_dir / "checkpoints/best_image_only.pt"
            save_pix2pix_checkpoint(path, generator=generator, discriminator=discriminator, optimizer_g=None, optimizer_d=None, config=config, completed_passes=completed, global_step=global_step, validation=validation)
            best_checkpoint = str(path); selected_pass=completed
        if reached_max: break
    final_validation = {k.removeprefix("validation_"): v for k,v in history[-1].items() if k.startswith("validation_")}
    final_path = output_dir / "checkpoints/final.pt"
    save_pix2pix_checkpoint(final_path, generator=generator, discriminator=discriminator, optimizer_g=optimizer_g, optimizer_d=optimizer_d, config=config, completed_passes=len(history), global_step=global_step, validation=final_validation)
    if len(history)!=200 or not best_checkpoint: raise RuntimeError("formal Pix2Pix training did not complete all 200 passes")
    manifest.update(status="complete", completed_utc=utc_now(), completed_passes=len(history), global_step=global_step, elapsed_seconds=time.monotonic()-started, selection_grid_passes=list(range(5,201,5)), selected_checkpoint_pass=selected_pass, best_image_only_checkpoint=best_checkpoint, best_image_only_checkpoint_sha256=sha256(best_checkpoint), best_validation_l1=best_l1 if math.isfinite(best_l1) else None, metrics_history_path=str(metrics_path), metrics_history_sha256=sha256(metrics_path), metrics_history_rows=len(history), final_checkpoint=str(final_path), final_checkpoint_sha256=sha256(final_path))
    write_json(output_dir / "run_manifest.json", manifest)
    train_data.close(); val_data.close()
    return manifest


def train_cyclegan(config: CycleGANConfig) -> dict[str, Any]:
    """Run the frozen pair-blind CycleGAN contract for one missing pair/fold."""

    _validate_formal_training_settings(config.max_passes, config.max_steps)
    if not torch.cuda.is_available():
        raise RuntimeError("formal CycleGAN training requires CUDA")
    if config.domain_a.strip().lower() not in MISSING_GAN_TARGETS or config.domain_b.strip().lower() != "at2":
        raise ValueError("Panel A CycleGAN domain_a must be a missing target and domain_b AT2")
    domain_a, domain_b = config.domain_a.strip().lower(), "at2"
    if config.validation_fold != (config.test_fold + 1) % 5:
        raise ValueError("validation fold must equal (test_fold + 1) mod 5")
    if int(config.selection_every)!=5: raise ValueError("Panel A checkpoint selection grid is every 5 passes")
    output_dir = Path(config.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path=output_dir/"metrics.jsonl"
    if metrics_path.exists(): raise FileExistsError(f"formal metrics history already exists: {metrics_path}")
    seed_everything(config.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    dtype = cyclegan_amp_dtype(config.amp)
    full = load_sample_index(config.sample_index, verify_caches=False)
    split = assign_split_roles(full, config.test_fold, config.validation_fold)
    train_data = UnpairedScannerDataset(split, domain_a, domain_b, split_role="train", augment=True, seed=config.seed)
    val_data = UnpairedScannerDataset(split, domain_a, domain_b, split_role="validation", augment=False, seed=config.seed + 1)
    train_sampler = UnpairedSlideBalancedSampler(train_data.sample_index, locations_per_slide=config.locations_per_train_slide, seed=config.seed)
    val_sampler = UnpairedSlideBalancedSampler(val_data.sample_index, locations_per_slide=config.locations_per_validation_slide, seed=config.seed + 1)
    train_loader = make_cyclegan_loader(train_data, train_sampler, batch_size=config.batch_size, workers=config.num_workers, device=device)
    val_loader = make_cyclegan_loader(val_data, val_sampler, batch_size=config.validation_batch_size, workers=config.num_workers, device=device)
    networks = build_cyclegan_models(ngf=config.ngf, ndf=config.ndf, n_blocks=config.n_blocks, device=device)
    g_ab, g_ba, d_a, d_b = networks
    gan_loss = GANLoss("lsgan").to(device)
    optimizer_g = torch.optim.Adam(list(g_ab.parameters()) + list(g_ba.parameters()), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    optimizer_d = torch.optim.Adam(list(d_a.parameters()) + list(d_b.parameters()), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    pool_a, pool_b = ImagePool(config.pool_size, seed=config.seed+101), ImagePool(config.pool_size, seed=config.seed+202)
    manifest: dict[str, Any] = {"training_version": PANEL_A_TRAINING_VERSION, "architecture_contract": "scanner_gan.train_cyclegan cyclegan_unpaired_bidirectional_v1", "status": "running", "started_utc": utc_now(), "direction": f"{domain_a}<->{domain_b}", "selection_boundary": "pair-blind marginal image endpoints on inner validation; same-location keys and UNI forbidden until lock", "sample_index": str(Path(config.sample_index).resolve()), "sample_index_sha256": sha256(config.sample_index), "train_slides": int(train_data.sample_index.slide_id.nunique()), "validation_slides": int(val_data.sample_index.slide_id.nunique()), "test_slides": int(split.loc[split.split_role == "test", "slide_id"].nunique()), "train_samples_per_pass_per_domain": len(train_sampler), "validation_samples_per_check_per_domain": len(val_sampler), "generator_parameters_each": count_parameters(g_ab), "discriminator_parameters_each": count_parameters(d_a), "config": asdict(config), "torch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name(device)}
    write_json(output_dir / "run_manifest.json", manifest)
    global_step, best_score, best_checkpoint, selected_pass = 0, float("inf"), "", -1
    history: list[dict[str, Any]] = []; started = time.monotonic(); reached_max = False
    for pass_index in range(config.max_passes):
        train_sampler.set_epoch(pass_index)
        lr = cyclegan_pass_learning_rate(config, pass_index); set_learning_rate((optimizer_g, optimizer_d), lr)
        for network in networks: network.train()
        sums = {key: 0.0 for key in ("d_a","d_b","g","g_a_to_b","g_b_to_a","cycle_a","cycle_b","identity_a","identity_b")}
        seen = same_slide = same_location = 0; pass_started = time.monotonic()
        for batch in train_loader:
            real_a = batch["source_image"].to(device, non_blocking=True); real_b = batch["target_image"].to(device, non_blocking=True)
            same_slide += int(np.asarray(batch["accidental_same_slide"]).sum()); same_location += int(np.asarray(batch["accidental_same_location"]).sum())
            set_requires_grad(d_a, False); set_requires_grad(d_b, False); optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
                fake_b=g_ab(real_a); rec_a=g_ba(fake_b); fake_a=g_ba(real_b); rec_b=g_ab(fake_a); id_a=g_ba(real_a); id_b=g_ab(real_b)
                lgab=gan_loss(d_b(fake_b),True); lgba=gan_loss(d_a(fake_a),True); lca=F.l1_loss(rec_a,real_a); lcb=F.l1_loss(rec_b,real_b); lia=F.l1_loss(id_a,real_a); lib=F.l1_loss(id_b,real_b)
                lg=lgab+lgba+config.lambda_cycle*(lca+lcb)+config.lambda_identity*(lia+lib)
            lg.backward(); optimizer_g.step()
            set_requires_grad(d_a, True); set_requires_grad(d_b, True); optimizer_d.zero_grad(set_to_none=True)
            hist_a, hist_b = pool_a.query(fake_a), pool_b.query(fake_b)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
                lda=.5*(gan_loss(d_a(real_a),True)+gan_loss(d_a(hist_a),False)); ldb=.5*(gan_loss(d_b(real_b),True)+gan_loss(d_b(hist_b),False)); ld=lda+ldb
            ld.backward(); optimizer_d.step()
            count=int(real_a.shape[0]); seen+=count; global_step+=1
            for key,value in (("d_a",lda),("d_b",ldb),("g",lg),("g_a_to_b",lgab),("g_b_to_a",lgba),("cycle_a",lca),("cycle_b",lcb),("identity_a",lia),("identity_b",lib)): sums[key]+=float(value.detach().float())*count
            if config.max_steps > 0 and global_step >= config.max_steps: reached_max=True; break
        completed=pass_index+1
        row: dict[str,Any]={"completed_passes":completed,"global_step":global_step,"learning_rate":lr,"train_samples_per_domain":seen,"accidental_same_slide_pairs":same_slide,"accidental_same_location_pairs":same_location,**{f"train_{k}":v/max(seen,1) for k,v in sums.items()},"pass_seconds":time.monotonic()-pass_started,"elapsed_seconds":time.monotonic()-started,"max_memory_allocated_gib":torch.cuda.max_memory_allocated(device)/2**30}
        selection=completed%config.selection_every==0 or completed==config.max_passes or reached_max
        validation: dict[str,Any]={}
        if selection:
            val_sampler.set_epoch(0); validation=validate_cyclegan(g_ab,g_ba,val_loader,device,dtype); row["validation"]=validation
            score=float(validation["marginal_score"])
            if score<best_score:
                best_score=score; path=output_dir/"checkpoints/best_image_only.pt"
                save_cyclegan_checkpoint(path,networks=networks,optimizers=None,config=config,completed_passes=completed,global_step=global_step,validation=validation); best_checkpoint=str(path); selected_pass=completed
        history.append(row)
        with metrics_path.open("a") as stream: stream.write(json.dumps(row,sort_keys=True)+"\n")
        print(json.dumps(row,sort_keys=True),flush=True)
        if reached_max: break
    final_validation=history[-1].get("validation") or validate_cyclegan(g_ab,g_ba,val_loader,device,dtype)
    final_path=output_dir/"checkpoints/final.pt"
    save_cyclegan_checkpoint(final_path,networks=networks,optimizers=(optimizer_g,optimizer_d),config=config,completed_passes=len(history),global_step=global_step,validation=final_validation)
    if len(history)!=200 or not best_checkpoint: raise RuntimeError("formal CycleGAN training did not complete all 200 passes")
    manifest.update(status="complete",completed_utc=utc_now(),completed_passes=len(history),global_step=global_step,elapsed_seconds=time.monotonic()-started,selection_grid_passes=list(range(5,201,5)),selected_checkpoint_pass=selected_pass,best_image_only_checkpoint=best_checkpoint,best_image_only_checkpoint_sha256=sha256(best_checkpoint),best_validation_marginal_score=best_score if math.isfinite(best_score) else None,metrics_history_path=str(metrics_path),metrics_history_sha256=sha256(metrics_path),metrics_history_rows=len(history),final_checkpoint=str(final_path),final_checkpoint_sha256=sha256(final_path))
    write_json(output_dir/"run_manifest.json",manifest); train_data.close(); val_data.close(); return manifest


def _generic_checkpoint_validator(
    payload: Mapping[str, Any], *, source_scanner: str, target_scanner: str, test_fold: int
) -> dict[str, Any]:
    source, target = _validate_panel_mapping(source_scanner, target_scanner)
    required = {"source_scanner", "target_scanner", "mapping", "test_fold", "validation_fold", "completed_passes", "global_step", "config", "generator"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"checkpoint missing fields: {sorted(missing)}")
    if str(payload["source_scanner"]).lower() != source or str(payload["target_scanner"]).lower() != target or str(payload["mapping"]).lower() != f"{source}->{target}":
        raise ValueError("checkpoint direction mismatch")
    if int(payload["test_fold"]) != int(test_fold) or int(payload["validation_fold"]) != (int(test_fold)+1)%5:
        raise ValueError("checkpoint fold mismatch")
    if int(payload["completed_passes"]) < 1 or int(payload["global_step"]) < 1:
        raise ValueError("checkpoint is incomplete")
    config = dict(payload["config"])
    for key in ("source_scanner", "target_scanner", "test_fold", "validation_fold", "ngf", "crop_size"):
        if key not in config: raise ValueError(f"checkpoint config missing {key}")
    return config


def _generic_run_validator(
    manifest: Mapping[str, Any], *, config: Mapping[str, Any], source_scanner: str,
    target_scanner: str, test_fold: int, sample_index_sha256: str
) -> None:
    source, target = _validate_panel_mapping(source_scanner, target_scanner)
    if str(manifest.get("status", "")).lower() != "complete": raise ValueError("training run is incomplete")
    if str(manifest.get("direction", "")).lower() != f"{source}->{target}": raise ValueError("run direction mismatch")
    if str(manifest.get("sample_index_sha256", "")).lower() != sample_index_sha256.lower(): raise ValueError("sample index hash mismatch")
    if int(config["test_fold"]) != int(test_fold): raise ValueError("run fold mismatch")
    if "image-only" not in str(manifest.get("selection_boundary", "")).lower(): raise ValueError("checkpoint selection boundary mismatch")


def _generic_cyclegan_run_validator(
    manifest: Mapping[str, Any], *, config: Mapping[str, Any], source_scanner: str,
    target_scanner: str, test_fold: int, sample_index_sha256: str
) -> None:
    source, target = _validate_panel_mapping(source_scanner, target_scanner)
    domains = {str(config["domain_a"]).lower(), str(config["domain_b"]).lower()}
    if domains != {source, target}: raise ValueError("CycleGAN domains mismatch")
    if str(manifest.get("status", "")).lower() != "complete": raise ValueError("CycleGAN run incomplete")
    if str(manifest.get("sample_index_sha256", "")).lower() != sample_index_sha256.lower(): raise ValueError("sample index hash mismatch")
    if int(config["test_fold"]) != int(test_fold): raise ValueError("CycleGAN fold mismatch")
    if "pair-blind" not in str(manifest.get("selection_boundary", "")).lower(): raise ValueError("CycleGAN selection was not pair-blind")


def validate_formal_training_artifacts(
    *, method: str, checkpoint_path: Path, run_manifest_path: Path,
    sample_index_path: Path, target_scanner: str, test_fold: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind a formal prediction to its exact 200-pass checkpoint and run."""

    method=method.strip().lower(); source="at2"; target=_validate_panel_mapping(source,target_scanner)[1]
    checkpoint_path=checkpoint_path.resolve(); run_manifest_path=run_manifest_path.resolve()
    payload=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    if not isinstance(payload,Mapping): raise TypeError("checkpoint root must be a mapping")
    config=payload.get("config")
    if not isinstance(config,Mapping): raise ValueError("checkpoint config missing")
    config=dict(config); manifest=json.loads(run_manifest_path.read_text())
    expected_validation=(int(test_fold)+1)%5
    _validate_formal_training_settings(int(config.get("max_passes",-1)),int(config.get("max_steps",-1)))
    selected_pass=int(payload.get("completed_passes",-1))
    if selected_pass not in range(5,201,5): raise ValueError("selected checkpoint pass is outside the frozen five-pass selection grid")
    if int(payload.get("test_fold",-1))!=int(test_fold) or int(payload.get("validation_fold",-1))!=expected_validation: raise ValueError("checkpoint fold contract mismatch")
    if method=="pix2pix":
        expected_version=PIX2PIX_CHECKPOINT_VERSION
        expected={"source_scanner":source,"target_scanner":target,"test_fold":int(test_fold),"validation_fold":expected_validation}
        observed={"source_scanner":str(payload.get("source_scanner","")).lower(),"target_scanner":str(payload.get("target_scanner","")).lower(),"test_fold":int(payload.get("test_fold",-1)),"validation_fold":int(payload.get("validation_fold",-1))}
        if str(payload.get("mapping","")).lower()!=f"{source}->{target}": raise ValueError("Pix2Pix checkpoint mapping mismatch")
    elif method=="cyclegan":
        expected_version=CYCLEGAN_CHECKPOINT_VERSION
        expected={"domain_a":target,"domain_b":source,"test_fold":int(test_fold),"validation_fold":expected_validation}
        observed={"domain_a":str(payload.get("domain_a","")).lower(),"domain_b":str(payload.get("domain_b","")).lower(),"test_fold":int(payload.get("test_fold",-1)),"validation_fold":int(payload.get("validation_fold",-1))}
        if str(payload.get("mapping","")).lower()!=f"{target}<->{source}": raise ValueError("CycleGAN checkpoint mapping mismatch")
    else: raise ValueError(f"unknown GAN method {method}")
    if str(payload.get("training_version",""))!=expected_version: raise ValueError("checkpoint training_version mismatch")
    for field,value in expected.items():
        config_value=str(config.get(field,"")).lower() if isinstance(value,str) else int(config.get(field,-1))
        if observed[field]!=value or config_value!=value: raise ValueError(f"checkpoint top-level/config mismatch for {field}")
    if str(manifest.get("training_version",""))!=PANEL_A_TRAINING_VERSION: raise ValueError("run manifest is not the frozen Panel A training version")
    if str(manifest.get("status",""))!="complete" or int(manifest.get("completed_passes",-1))!=200: raise ValueError("run manifest is not a complete formal 200-pass run")
    if str(manifest.get("sample_index_sha256",""))!=sha256(sample_index_path): raise ValueError("run manifest sample-index hash mismatch")
    manifest_config=manifest.get("config")
    if not isinstance(manifest_config,Mapping) or dict(manifest_config)!=config: raise ValueError("run manifest/checkpoint config mismatch")
    recorded_path=Path(str(manifest.get("best_image_only_checkpoint",""))).resolve()
    if recorded_path!=checkpoint_path: raise ValueError("prediction checkpoint is not the run manifest's exact selected checkpoint")
    checkpoint_hash=sha256(checkpoint_path)
    if str(manifest.get("best_image_only_checkpoint_sha256",""))!=checkpoint_hash: raise ValueError("selected checkpoint SHA256 mismatch")
    metrics_path=Path(str(manifest.get("metrics_history_path",""))).resolve()
    expected_metrics_path=run_manifest_path.parent/"metrics.jsonl"
    if metrics_path!=expected_metrics_path or not metrics_path.is_file() or sha256(metrics_path)!=str(manifest.get("metrics_history_sha256","")): raise ValueError("formal metrics history path/SHA mismatch")
    with metrics_path.open() as stream: history=[json.loads(line) for line in stream if line.strip()]
    if len(history)!=200 or int(manifest.get("metrics_history_rows",-1))!=200 or [int(row.get("completed_passes",-1)) for row in history]!=list(range(1,201)): raise ValueError("formal metrics history is not the complete ordered 200-pass run")
    selection_grid=list(range(5,201,5))
    if [int(x) for x in manifest.get("selection_grid_passes",[])]!=selection_grid: raise ValueError("formal selection grid is incomplete")
    candidates=[]
    for row in history:
        completed=int(row["completed_passes"])
        if completed not in selection_grid: continue
        if method=="pix2pix": score=float(row["validation_l1_normalized"])
        else:
            validation=row.get("validation")
            if not isinstance(validation,Mapping) or "marginal_score" not in validation: raise ValueError("CycleGAN selection-grid validation is missing")
            score=float(validation["marginal_score"])
        candidates.append((score,completed))
    selected_score,argmin_pass=min(candidates,key=lambda value:(value[0],value[1]))
    if selected_pass!=argmin_pass or int(manifest.get("selected_checkpoint_pass",-1))!=argmin_pass: raise ValueError("selected checkpoint is not the image-only history argmin")
    manifest_score=float(manifest["best_validation_l1"] if method=="pix2pix" else manifest["best_validation_marginal_score"])
    if not np.isclose(manifest_score,selected_score,rtol=0,atol=1e-12): raise ValueError("run manifest best score does not match metrics history argmin")
    payload_validation=payload.get("validation")
    if not isinstance(payload_validation,Mapping): raise ValueError("selected checkpoint validation payload missing")
    payload_score=float(payload_validation["l1_normalized"] if method=="pix2pix" else payload_validation["marginal_score"])
    if not np.isclose(payload_score,selected_score,rtol=0,atol=1e-12): raise ValueError("selected checkpoint validation score does not match history argmin")
    selected_row=history[argmin_pass-1]
    if int(payload.get("global_step",-1))!=int(selected_row.get("global_step",-2)): raise ValueError("selected checkpoint global step does not match metrics history")
    final_path=Path(str(manifest.get("final_checkpoint",""))).resolve(); expected_final=run_manifest_path.parent/"checkpoints/final.pt"
    if final_path!=expected_final or not final_path.is_file() or sha256(final_path)!=str(manifest.get("final_checkpoint_sha256","")): raise ValueError("final checkpoint path/SHA mismatch")
    final_payload=torch.load(final_path,map_location="cpu",weights_only=True)
    if int(final_payload.get("completed_passes",-1))!=200 or final_payload.get("config")!=config or str(final_payload.get("training_version",""))!=expected_version: raise ValueError("final checkpoint is not the exact formal pass-200 artifact")
    return dict(payload),manifest


def predict_gan(
    *, method: str, checkpoint: Path, run_manifest: Path, sample_index: Path,
    locked_index: Path, target_scanner: str, test_fold: int, output_dir: Path,
    batch_size: int = 32, num_workers: int = 4,
) -> dict[str, Any]:
    """Use the audited historical writer with Panel A-only direction gates."""

    validate_formal_training_artifacts(method=method,checkpoint_path=checkpoint,run_manifest_path=run_manifest,sample_index_path=sample_index,target_scanner=target_scanner,test_fold=test_fold)
    import scanner_gan.predict as prediction
    prediction.PairedScannerDataset = GenericPairedScannerDataset
    prediction.validate_checkpoint_payload = _generic_checkpoint_validator
    prediction.validate_run_manifest = _generic_run_validator
    prediction.validate_cyclegan_run_manifest = _generic_cyclegan_run_validator
    result=prediction.predict(
        checkpoint_path=checkpoint, run_manifest_path=run_manifest,
        sample_index_path=sample_index, locked_index_path=locked_index,
        output_dir=output_dir, source_scanner="at2", target_scanner=target_scanner,
        test_fold=test_fold, batch_size=batch_size, num_workers=num_workers,
        device="cuda", amp="bfloat16", overwrite=True, method=method,
    )
    if str(result.get("status"))!="complete" or str(result.get("direction","")).lower()!=f"at2->{target_scanner}" or int(result.get("test_fold",-1))!=int(test_fold) or str(result.get("method","")).lower()!=method:
        raise RuntimeError("prediction manifest failed formal direction/fold/method validation")
    if int(result.get("samples",-1))!=int(result.get("slides",-1))*40: raise RuntimeError("prediction output is not exactly 40 locked locations per slide")
    return result


def stable_seed(*values: object) -> int:
    token = "|".join(str(value) for value in values).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def load_vahadane_selection(path: Path) -> dict[str, Any]:
    payload=json.loads(path.read_text())
    expected={"solver":"mu","init":"nndsvdar","alpha_W":.001,"alpha_H":0.0,"l1_ratio":1.0,"tolerance":.001,"max_iter":2000,"max_pixels_source":4096,"max_pixels_reference":65536,"shuffle":False}
    if payload.get("status")!="selected" or payload.get("selected_configuration")!=expected: raise ValueError("Vahadane selection manifest does not encode the frozen global MU configuration")
    if payload.get("analysis")!="global_source_only_vahadane_contract_selection_v4" or payload.get("target_images_accessed") is not False or payload.get("target_outcomes_accessed") is not False: raise ValueError("Vahadane v4 selection was not source-input-only")
    prior=payload.get("prior_source_only_solver_selection",{})
    prior_path=path.parent/str(prior.get("manifest",""))
    if prior.get("sha256")!="1d8f9553dccd23e259991514e602bf672b31185ea88bdcef90fc62e0e1fa71dd" or not prior_path.is_file() or sha256(prior_path)!=prior.get("sha256"): raise ValueError("Vahadane prior source-only selection hash mismatch")
    audit=payload.get("locked_source_convergence_audit",{})
    audit_contract=path.parent/str(audit.get("contract",""))
    if audit.get("contract_sha256")!="2a7f59d6c468e11e39ccf9e9583a383d6574a20039b3915098ec65f49273f6be" or not audit_contract.is_file() or sha256(audit_contract)!=audit.get("contract_sha256"): raise ValueError("Vahadane locked-source audit contract hash mismatch")
    audit_root=path.parent/"vahadane_locked_source_full_audit_v4"
    audit_files=audit.get("files",{})
    if set(audit_files)!={f"fold_{fold}.{suffix}" for fold in range(5) for suffix in ("csv.gz","json")}:
        raise ValueError("Vahadane locked-source audit does not bind all five fold outputs")
    for name,file_hash in audit_files.items():
        audit_path=audit_root/name
        if not audit_path.is_file() or sha256(audit_path)!=file_hash: raise ValueError(f"Vahadane locked-source audit hash mismatch: {name}")
    results=audit.get("results",{})
    if set(results)!={"1000","2000","4000"}: raise ValueError("Vahadane audit candidate grid mismatch")
    for cap in (1000,2000,4000):
        value=results[str(cap)]
        valid=(int(value.get("evaluation_valid",-1))==20600 and int(value.get("sentinel_valid",-1))==25 and float(value.get("basis_condition_max",float("inf")))<50)
        converged=(int(value.get("evaluation_converged",-1))==20600 and int(value.get("evaluation_nonconverged",-1))==0 and int(value.get("sentinel_converged",-1))==25)
        if bool(value.get("eligible"))!=(valid and converged): raise ValueError(f"Vahadane audit eligibility mismatch at max_iter={cap}")
    if results["1000"].get("eligible") is not False or results["2000"].get("eligible") is not True or results["4000"].get("eligible") is not True: raise ValueError("Vahadane minimal eligible iteration cap is not 2000")
    return payload


def selected_vahadane_kwargs(selection: Mapping[str,Any], *, reference: bool) -> dict[str,Any]:
    config=dict(selection["selected_configuration"])
    return {"solver":config["solver"],"alpha":float(config["alpha_W"]),"tolerance":float(config["tolerance"]),"max_iter":int(config["max_iter"]),"max_pixels":int(config["max_pixels_reference" if reference else "max_pixels_source"])}


def fit_stain_reference(
    sample_index_path: Path,
    locked_index_path: Path,
    target_scanner: str,
    test_fold: int,
    output_path: Path,
    vahadane_selection_path: Path,
    *,
    images_per_slide: int = 5,
    pixels_per_image: int = 512,
) -> dict[str, Any]:
    """Fit fold-safe target stain bases and run an actual-cohort sentinel."""

    target_scanner = target_scanner.strip().lower()
    if target_scanner not in PANEL_A_TARGETS:
        raise ValueError(f"invalid Panel A target {target_scanner}")
    index = pd.read_csv(sample_index_path, dtype={"slide_id": str})
    locked = pd.read_csv(locked_index_path, dtype={"slide_id": str})
    train = index[index.fold.astype(int) != int(test_fold)]
    arrays: list[np.ndarray] = []
    image_count = 0
    scanner_names = tuple(str(train.scanner_order.iloc[0]).split(","))
    scanner_index = scanner_lookup(scanner_names)[target_scanner]
    for slide_id, group in train.groupby("slide_id", sort=True):
        ordered = group.sort_values("location_index")
        positions = np.linspace(0, len(ordered) - 1, min(images_per_slide, len(ordered)), dtype=int)
        selected = ordered.iloc[np.unique(positions)]
        with h5py.File(str(selected.cache_path.iloc[0]), "r") as store:
            for location in selected.location_index.astype(int):
                image = np.asarray(store["images"][location, scanner_index], dtype=np.uint8)
                od = tissue_od(image)
                if len(od) > pixels_per_image:
                    offset = stable_seed(slide_id, location, target_scanner, test_fold) % len(od)
                    pos = (offset + np.linspace(0, len(od)-1, pixels_per_image, dtype=int)) % len(od)
                    od = od[pos]
                if len(od): arrays.append(od)
                image_count += 1
    vahadane_selection=load_vahadane_selection(vahadane_selection_path)
    vahadane_reference_kwargs=selected_vahadane_kwargs(vahadane_selection,reference=True)
    vahadane_source_kwargs=selected_vahadane_kwargs(vahadane_selection,reference=False)
    pixels = np.concatenate(arrays) if arrays else np.empty((0, 3))
    macenko = fit_macenko_od(pixels)
    vahadane=fit_vahadane_od(pixels,seed=stable_seed("reference",target_scanner,test_fold),**vahadane_reference_kwargs)
    if macenko is None or vahadane is None: raise RuntimeError("stain reference fitting failed")
    if not vahadane.converged: raise RuntimeError(f"selected Vahadane reference did not converge in {vahadane.iterations} iterations")

    # Implementation sentinel: parameters are already frozen above. Applying
    # them twice to the first held-out AT2 image must be bit deterministic.
    heldout = locked[locked.fold.astype(int) == int(test_fold)].sort_values(["slide_id", "location_index"]).iloc[0]
    source_index = scanner_lookup(tuple(str(heldout.scanner_order).split(",")))["at2"]
    with h5py.File(str(heldout.cache_path), "r") as store:
        source = np.asarray(store["images"][int(heldout.location_index), source_index], dtype=np.uint8)
    sentinel: dict[str, Any] = {}
    for method, reference in (("macenko", macenko),("vahadane",vahadane)):
        seed = stable_seed("sentinel", method, target_scanner, test_fold)
        config=vahadane_source_kwargs if method=="vahadane" else None
        first = normalize(source, reference, method=method, seed=seed,vahadane_config=config)
        second = normalize(source, reference, method=method, seed=seed,vahadane_config=config)
        if not np.array_equal(first.image, second.image):
            raise RuntimeError(f"{method} sentinel is not deterministic")
        sentinel[method] = {
            "fallback": bool(first.fallback),
            "clipped_fraction": float(first.clipped_fraction),
            "finite": bool(np.isfinite(first.image).all()),
            "repeat_sha256_equal": True,
            "source_converged": bool(first.source_parameters.converged) if first.source_parameters is not None else False,
            "source_iterations": int(first.source_parameters.iterations) if first.source_parameters is not None else 0,
            "source_reconstruction_error": float(first.source_parameters.reconstruction_error) if first.source_parameters is not None else float("nan"),
        }
        if method=="vahadane" and not sentinel[method]["source_converged"]: raise RuntimeError("selected Vahadane held-out source sentinel did not converge")
    payload = {
        "status": "pass",
        "analysis": "panel_a_stain_reference",
        "target_scanner": target_scanner,
        "test_fold": int(test_fold),
        "training_slides": int(train.slide_id.nunique()),
        "sampled_images": int(image_count),
        "sampled_tissue_pixels": int(len(pixels)),
        "sample_index": {"path": str(sample_index_path.resolve()), "sha256": sha256(sample_index_path)},
        "locked_index": {"path": str(locked_index_path.resolve()), "sha256": sha256(locked_index_path)},
        "parameters": {"macenko": parameters_to_json(macenko),"vahadane":parameters_to_json(vahadane)},
        "sentinel": sentinel,
        "method_status": {"macenko":{"status":"evaluated"},"vahadane":{"status":"evaluated","configuration":vahadane_selection["selected_configuration"]}},
        "vahadane_selection_manifest": {"path":str(vahadane_selection_path.resolve()),"sha256":sha256(vahadane_selection_path)},
    }
    write_json(output_path, payload)
    return payload


def evaluate_stain_slide(
    locked_index_path: Path,
    reference_root: Path,
    target_reference_root: Path,
    slide_index: int,
    output_root: Path,
    vahadane_selection_path: Path,
) -> dict[str, Any]:
    """Evaluate valid stain methods at all 40 locked locations on one slide."""

    vahadane_selection=load_vahadane_selection(vahadane_selection_path)
    selection_sha=sha256(vahadane_selection_path)
    vahadane_source_kwargs=selected_vahadane_kwargs(vahadane_selection,reference=False)

    locked = pd.read_csv(locked_index_path, dtype={"slide_id": str})
    slide_ids = sorted(locked.slide_id.unique())
    if slide_index < 0 or slide_index >= len(slide_ids):
        raise IndexError(slide_index)
    slide_id = slide_ids[slide_index]
    rows_for_slide = locked[locked.slide_id == slide_id].sort_values("location_index")
    if len(rows_for_slide) != 40:
        raise ValueError(f"{slide_id}: expected 40 locked locations, got {len(rows_for_slide)}")
    fold = int(rows_for_slide.fold.iloc[0]); tissue = str(rows_for_slide.tissue_type.iloc[0])
    scanner_names = tuple(str(rows_for_slide.scanner_order.iloc[0]).split(","))
    lookup = scanner_lookup(scanner_names)
    references = {
        scanner: json.loads((reference_root / f"fold_{fold}" / f"{scanner}.json").read_text())
        for scanner in PANEL_A_TARGETS
    }
    for value in references.values():
        if value.get("method_status",{}).get("vahadane",{}).get("status")!="evaluated" or value.get("vahadane_selection_manifest",{}).get("sha256")!=selection_sha: raise ValueError("reference method-status/selection mismatch")
    reference_hashes = {
        scanner: sha256(reference_root / f"fold_{fold}" / f"{scanner}.json")
        for scanner in PANEL_A_TARGETS
    }
    reference_bundle_sha256 = hashlib.sha256(
        "|".join(f"{key}:{reference_hashes[key]}" for key in sorted(reference_hashes)).encode()
    ).hexdigest()
    stain_code_sha256 = sha256(Path(__file__).with_name("stain.py"))
    endpoint_reference_path=target_reference_root/f"{slide_id}.csv"
    endpoint_reference = pd.read_csv(endpoint_reference_path).set_index(["location_index", "scanner"])
    output = output_root / "shards" / f"{slide_id}.csv.gz"
    summary_path = output_root / "shards" / f"{slide_id}.summary.json"
    if output.exists() and summary_path.exists():
        old = json.loads(summary_path.read_text())
        if old.get("status") == "pass" and old.get("output_sha256") == sha256(output) and old.get("reference_bundle_sha256") == reference_bundle_sha256 and old.get("stain_code_sha256") == stain_code_sha256:
            return {"status": "already_complete", "slide_id": slide_id}
    result_rows: list[dict[str, Any]] = []
    cache_path = str(rows_for_slide.cache_path.iloc[0])
    with h5py.File(cache_path, "r") as store:
        source_indices = np.asarray(store["source_index"], dtype=np.int64)
        for location_offset, item in enumerate(rows_for_slide.itertuples(index=False)):
            location = int(item.location_index)
            source = np.asarray(store["images"][location, lookup["at2"]], dtype=np.uint8)
            source_scalar, source_radial, source_gradient, _ = absolute_measurements(source[None], 1)
            for scanner in PANEL_A_TARGETS:
                target = np.asarray(store["images"][location, lookup[scanner]], dtype=np.uint8)
                normalized = []
                reports = []
                for method in ("macenko","vahadane"):
                    reference = parameters_from_json(references[scanner]["parameters"][method])
                    report = normalize(source, reference, method=method, seed=stable_seed(slide_id, location, scanner, method),vahadane_config=vahadane_source_kwargs if method=="vahadane" else None)
                    if method=="vahadane" and (report.source_parameters is None or not report.source_parameters.converged): raise RuntimeError(f"selected Vahadane did not converge for {slide_id}/{location}/{scanner}")
                    normalized.append(report.image); reports.append(report)
                generated_images = np.stack(normalized)
                scalar, radial, gradient, _ = absolute_measurements(generated_images, 1)
                generated = contrasts_from_absolute(scalar, radial, source_scalar, source_radial)
                target_scalar, target_radial, target_gradient, _ = absolute_measurements(target[None], 1)
                target_vector = contrasts_from_absolute(target_scalar, target_radial, source_scalar, source_radial)[0]
                ref = endpoint_reference.loc[(location, scanner)]
                scale = np.maximum(np.asarray([float(ref[f"scale_{name}"]) for name in ENDPOINTS]), 1e-6)
                expected_target = np.asarray([float(ref[f"target_{name}"]) for name in ENDPOINTS])
                if not np.allclose(target_vector, expected_target, atol=3e-3, rtol=3e-3):
                    raise ValueError(f"target endpoint mismatch {slide_id}/{location}/{scanner}")
                fidelity = rowwise_correlation(gradient, np.repeat(source_gradient, len(normalized), axis=0))
                target_similarity = rowwise_correlation(gradient, np.repeat(target_gradient, len(normalized), axis=0))
                for method_index, method in enumerate(("macenko","vahadane")):
                    residual = (generated[method_index] - target_vector) / scale
                    row: dict[str, Any] = {
                        "slide_id": slide_id, "tissue_type": tissue, "fold": fold,
                        "location_index": location, "source_index": int(source_indices[location]),
                        "scanner": scanner, "arm": method,
                        "distance": float(np.sqrt(np.mean(residual**2))),
                        "endpoint_coverage": float(np.mean(np.abs(residual) <= 1.0)),
                        "all_endpoints_covered": bool(np.all(np.abs(residual) <= 1.0)),
                        "fidelity_ncc": float(fidelity[method_index]),
                        "target_gradient_ncc": float(target_similarity[method_index]),
                        "saturation_fraction": float(scalar[method_index, 8]),
                        "stain_fallback": bool(reports[method_index].fallback),
                        "stain_preclip_fraction": float(reports[method_index].clipped_fraction),
                        "stain_converged": bool(reports[method_index].source_parameters.converged) if reports[method_index].source_parameters is not None else False,
                        "stain_iterations": int(reports[method_index].source_parameters.iterations) if reports[method_index].source_parameters is not None else 0,
                        "stain_reconstruction_error": float(reports[method_index].source_parameters.reconstruction_error) if reports[method_index].source_parameters is not None else 0.0,
                    }
                    for endpoint_index, endpoint in enumerate(ENDPOINTS):
                        row[f"generated_{endpoint}"] = float(generated[method_index, endpoint_index])
                        row[f"target_{endpoint}"] = float(target_vector[endpoint_index])
                        row[f"scaled_residual_{endpoint}"] = float(residual[endpoint_index])
                    result_rows.append(row)
            print(f"[{slide_id}] {location_offset+1}/40", flush=True)
    frame = pd.DataFrame(result_rows)
    expected = 40 * len(PANEL_A_TARGETS)*2
    if len(frame) != expected or not np.isfinite(frame.select_dtypes("number")).all().all():
        raise ValueError(f"invalid stain shard {slide_id}: {len(frame)}")
    write_frame(output, frame)
    summary = {"status": "pass", "slide_id": slide_id, "rows": len(frame), "fallbacks": int(frame.stain_fallback.sum()), "nonconverged": int((~frame.stain_converged.astype(bool)).sum()), "method_status":{"macenko":"evaluated","vahadane":"evaluated"},"vahadane_selection_sha256":selection_sha,"reference_hashes": reference_hashes, "reference_bundle_sha256": reference_bundle_sha256,"target_reference":{"path":str(endpoint_reference_path.resolve()),"sha256":sha256(endpoint_reference_path)}, "stain_code_sha256": stain_code_sha256, "output_sha256": sha256(output)}
    write_json(summary_path, summary)
    return summary


def aggregate_stain_results(output_root: Path, vahadane_selection_path: Path) -> dict[str, Any]:
    vahadane_selection=load_vahadane_selection(vahadane_selection_path)
    paths = sorted((output_root / "shards").glob("*.csv.gz"))
    if len(paths) != 103:
        raise ValueError(f"expected 103 stain shards, got {len(paths)}")
    repaired=0; frames=[]; observed_slides=[]; selection_sha=sha256(vahadane_selection_path); code_sha=sha256(Path(__file__).with_name("stain.py"))
    for path in paths:
        repaired+=ensure_gzip_csv(path)
        summary_path=path.with_name(path.name.removesuffix(".csv.gz")+".summary.json")
        if not summary_path.is_file(): raise FileNotFoundError(summary_path)
        shard_summary=json.loads(summary_path.read_text())
        if shard_summary.get("status")!="pass" or shard_summary.get("output_sha256")!=sha256(path) or shard_summary.get("vahadane_selection_sha256")!=selection_sha or shard_summary.get("stain_code_sha256")!=code_sha: raise ValueError(f"stain shard provenance mismatch: {path}")
        target_reference_item=shard_summary.get("target_reference",{}); target_reference_path=Path(str(target_reference_item.get("path","")))
        if not target_reference_path.is_file() or sha256(target_reference_path)!=str(target_reference_item.get("sha256","")): raise ValueError("stain shard target-reference hash mismatch")
        shard=pd.read_csv(path,dtype={"slide_id":str})
        slide_values=shard.slide_id.unique()
        if len(shard)!=400 or len(slide_values)!=1 or str(slide_values[0])!=str(shard_summary.get("slide_id")) or shard.location_index.nunique()!=40 or set(shard.scanner)!=set(PANEL_A_TARGETS) or set(shard.arm)!={"macenko","vahadane"}: raise ValueError(f"stain shard cardinality/schema mismatch: {path}")
        if len(shard_summary.get("reference_hashes",{}))!=5: raise ValueError("stain shard does not bind five fold references")
        fold=int(shard.fold.iloc[0])
        for scanner,expected in shard_summary["reference_hashes"].items():
            reference_path=output_root/"references"/f"fold_{fold}"/f"{scanner}.json"
            if not reference_path.is_file() or sha256(reference_path)!=expected: raise ValueError("stain shard reference hash mismatch")
        observed_slides.append(str(slide_values[0])); frames.append(shard)
    if len(set(observed_slides))!=103: raise ValueError("stain shards do not contain 103 unique slides")
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != 103 * 40 * 5*2 or set(frame.arm)!={"macenko","vahadane"}:
        raise ValueError(f"unexpected stain row count {len(frame)}")
    summary_rows = []
    metrics = ("distance", "endpoint_coverage", "all_endpoints_covered", "fidelity_ncc", "target_gradient_ncc", "saturation_fraction", "stain_fallback", "stain_preclip_fraction", "stain_converged", "stain_iterations", "stain_reconstruction_error")
    for (scanner, arm), group in frame.groupby(["scanner", "arm"], sort=True):
        slide = group.groupby("slide_id")[list(metrics)].mean()
        rng = np.random.default_rng(stable_seed("bootstrap", scanner, arm))
        draw_index = rng.integers(0, len(slide), size=(20000, len(slide)))
        row: dict[str, Any] = {"scanner": scanner, "arm": arm, "slides": len(slide), "locations": len(group)}
        for metric in metrics:
            values = slide[metric].to_numpy(float)
            draws = values[draw_index].mean(axis=1)
            row[metric] = float(values.mean())
            row[f"{metric}_ci_low"] = float(np.quantile(draws, .025))
            row[f"{metric}_ci_high"] = float(np.quantile(draws, .975))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    write_frame(output_root / "stain_location_metrics.csv.gz", frame)
    write_frame(output_root / "stain_scanner_summary.csv", summary)
    reference_paths = sorted((output_root / "references").glob("fold_*/*.json"))
    expected_reference_relatives={f"fold_{fold}/{scanner}.json" for fold in range(5) for scanner in PANEL_A_TARGETS}
    observed_reference_relatives={str(path.relative_to(output_root/"references")) for path in reference_paths}
    if observed_reference_relatives!=expected_reference_relatives: raise ValueError("stain reference bundle is not exactly five folds x five scanners")
    sentinel_failures = 0; reference_condition=[]; reference_nonconverged=0
    for path in reference_paths:
        value = json.loads(path.read_text())
        if value.get("vahadane_selection_manifest",{}).get("sha256")!=sha256(vahadane_selection_path): raise ValueError("reference Vahadane selection hash mismatch")
        vahadane=value.get("parameters",{}).get("vahadane",{}); basis=np.asarray(vahadane.get("basis",[]),dtype=float)
        reference_condition.append(float(np.linalg.cond(basis)) if basis.shape==(3,2) else float("inf")); reference_nonconverged+=int(not bool(vahadane.get("converged",False)))
        sentinel_failures += sum(
            not bool(item.get("finite")) or not bool(item.get("repeat_sha256_equal"))
            for item in value.get("sentinel", {}).values()
        )
    source_nonconverged=int((~frame.loc[frame.arm=="vahadane","stain_converged"].astype(bool)).sum()); condition_max=max(reference_condition,default=float("inf")); warning_count=reference_nonconverged+source_nonconverged
    fit_quality_valid = bool(
        sentinel_failures == 0
        and reference_nonconverged==0
        and np.isfinite(condition_max) and condition_max<50
        and source_nonconverged==0
        and np.isfinite(frame.select_dtypes("number")).all().all()
    )
    payload = {
        "status": "pass" if fit_quality_valid else "fail",
        "shards": len(paths), "rows": len(frame), "summary_rows": len(summary),
        "gzip_shards_repaired": int(repaired),
        "fallback_count": int(frame.stain_fallback.sum()),
        "fallback_fraction": float(frame.stain_fallback.mean()),
        "method_status": {"macenko":{"status":"evaluated","fit_quality_valid":fit_quality_valid},"vahadane":{"status":"evaluated","fit_quality_valid":fit_quality_valid,"configuration":vahadane_selection["selected_configuration"]}},
        "vahadane_selection_manifest_sha256":sha256(vahadane_selection_path),
        "vahadane_convergence_warning_count": int(warning_count),
        "vahadane_reference_nonconverged_count": int(reference_nonconverged),
        "vahadane_source_nonconverged_count": int(source_nonconverged),
        "reference_vahadane_basis_condition_max": float(condition_max),
        "sentinel_failures": int(sentinel_failures),
        "fit_quality_valid": fit_quality_valid,
        "interpretation": "Macenko and the globally selected MU Vahadane configuration passed deterministic finite reference/source sentinel and conditioning gates",
        "location_sha256":sha256(output_root/"stain_location_metrics.csv.gz"),"summary_sha256":sha256(output_root/"stain_scanner_summary.csv"),
        "outputs":{"location_metrics":{"path":str((output_root/"stain_location_metrics.csv.gz").resolve()),"sha256":sha256(output_root/"stain_location_metrics.csv.gz")},"scanner_summary":{"path":str((output_root/"stain_scanner_summary.csv").resolve()),"sha256":sha256(output_root/"stain_scanner_summary.csv")}},
    }
    write_json(output_root / "aggregate_manifest.json", payload)
    if not fit_quality_valid:
        raise RuntimeError(f"stain fit-quality gate failed: {payload}")
    return payload


def stain_uni_extract(
    *, task_index: int, task_count: int, existing_uni_metrics: Path,
    locked_index: Path, reference_root: Path, output_root: Path,
    vahadane_selection_path: Path, batch_size: int = 64,
) -> dict[str, Any]:
    """Embed raw, real target, and evaluable stains on the locked 20 locations."""

    import torch.nn.functional as torch_functional
    from prenorm.embedding import load_uni
    if not torch.cuda.is_available(): raise RuntimeError("UNI extraction requires CUDA")
    selection=load_vahadane_selection(vahadane_selection_path); selection_sha=sha256(vahadane_selection_path); vahadane_source_kwargs=selected_vahadane_kwargs(selection,reference=False)
    old = pd.read_csv(existing_uni_metrics, dtype={"slide_id": str}, usecols=["slide_id", "location_index"])
    chosen = old.drop_duplicates().sort_values(["slide_id", "location_index"])
    counts = chosen.groupby("slide_id").size()
    if len(counts) != 103 or not (counts == 20).all(): raise ValueError("existing UNI panel is not 103 x 20")
    locked = pd.read_csv(locked_index, dtype={"slide_id": str})
    locked = locked.merge(chosen, on=["slide_id", "location_index"], how="inner", validate="one_to_one")
    slide_ids = sorted(chosen.slide_id.unique())[int(task_index)::int(task_count)]
    device=torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32=True
    model,size,mean,std=load_uni(device)
    completed=[]
    locked_sha=sha256(locked_index)
    for slide_id in slide_ids:
        rows=locked[locked.slide_id==slide_id].sort_values("location_index")
        if len(rows)!=20: raise ValueError(f"{slide_id}: expected 20 UNI locations")
        fold=int(rows.fold.iloc[0]); names=tuple(str(rows.scanner_order.iloc[0]).split(",")); lookup=scanner_lookup(names)
        conditions=["source:at2"]+[f"target:{s}" for s in PANEL_A_TARGETS]+[f"{method}:{s}" for method in ("macenko","vahadane") for s in PANEL_A_TARGETS]
        image_array=np.empty((20,len(conditions),256,256,3),dtype=np.uint8)
        reference_paths={s:reference_root/f"fold_{fold}"/f"{s}.json" for s in PANEL_A_TARGETS}
        refs={s:json.loads(path.read_text()) for s,path in reference_paths.items()}
        if any(value.get("vahadane_selection_manifest",{}).get("sha256")!=selection_sha for value in refs.values()): raise ValueError("stain UNI reference selection mismatch")
        reference_hashes={scanner:sha256(path) for scanner,path in reference_paths.items()}
        reference_bundle_sha=hashlib.sha256("|".join(f"{key}:{reference_hashes[key]}" for key in sorted(reference_hashes)).encode()).hexdigest()
        with h5py.File(str(rows.cache_path.iloc[0]),"r") as store:
            for i,item in enumerate(rows.itertuples(index=False)):
                location=int(item.location_index); source=np.asarray(store["images"][location,lookup["at2"]],dtype=np.uint8); image_array[i,0]=source
                for sidx,scanner in enumerate(PANEL_A_TARGETS, start=1):
                    image_array[i,sidx]=np.asarray(store["images"][location,lookup[scanner]],dtype=np.uint8)
                    for method_index,method in enumerate(("macenko","vahadane")):
                        ref=parameters_from_json(refs[scanner]["parameters"][method])
                        report=normalize(source,ref,method=method,seed=stable_seed(slide_id,location,scanner,method),vahadane_config=vahadane_source_kwargs if method=="vahadane" else None)
                        if method=="vahadane" and (report.source_parameters is None or not report.source_parameters.converged): raise RuntimeError(f"selected Vahadane did not converge for {slide_id}/{location}/{scanner}")
                        image_array[i,1+len(PANEL_A_TARGETS)+method_index*len(PANEL_A_TARGETS)+(sidx-1)]=report.image
        flat=image_array.reshape(-1,256,256,3); embeddings=[]
        with torch.inference_mode():
            for start in range(0,len(flat),batch_size):
                tensor=torch.from_numpy(flat[start:start+batch_size].astype(np.float32)).permute(0,3,1,2).div_(255).to(device,non_blocking=True)
                tensor=torch_functional.interpolate(tensor,size=(size,size),mode="bicubic",align_corners=False)
                with torch.autocast(device_type="cuda",dtype=torch.float16): feature=model((tensor-mean)/std)
                embeddings.append(torch_functional.normalize(feature.float(),dim=1).cpu().numpy())
        feature=np.concatenate(embeddings).astype(np.float32).reshape(20,len(conditions),-1)
        output=output_root/"shards"/f"{slide_id}.h5"; output.parent.mkdir(parents=True,exist_ok=True); temporary=output.with_suffix(f".h5.{os.getpid()}.tmp")
        with h5py.File(temporary,"w") as store:
            store.attrs["analysis"]="panel_a_stain_uni_v4"; store.attrs["slide_id"]=slide_id; store.attrs["fold"]=fold; store.attrs["locked_index_sha256"]=locked_sha; store.attrs["vahadane_selection_sha256"]=selection_sha; store.attrs["reference_bundle_sha256"]=reference_bundle_sha
            store.create_dataset("features",data=feature,compression="lzf"); store.create_dataset("condition_names",data=np.asarray(conditions,dtype="S32")); store.create_dataset("location_index",data=rows.location_index.to_numpy(np.int64))
        temporary.replace(output); completed.append({"slide_id":slide_id,"fold":fold,"sha256":sha256(output),"reference_hashes":reference_hashes,"reference_bundle_sha256":reference_bundle_sha})
        print(json.dumps(completed[-1]),flush=True)
    manifest={"status":"pass","task_index":task_index,"task_count":task_count,"slides":len(completed),"outputs":completed,"existing_uni_metrics_sha256":sha256(existing_uni_metrics),"locked_index_sha256":locked_sha,"method_status":{"macenko":"evaluated","vahadane":"evaluated"},"vahadane_selection_sha256":selection_sha}
    write_json(output_root/"task_manifests"/f"task_{task_index:03d}.json",manifest); return manifest


def aggregate_stain_uni(output_root: Path) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    task_paths=sorted((output_root/"task_manifests").glob("task_*.json"))
    if len(task_paths)!=13: raise ValueError(f"expected 13 UNI task manifests, got {len(task_paths)}")
    expected_outputs={}; locked_hashes=set(); selection_hashes=set(); existing_hashes=set(); task_indices=[]
    for task_path in task_paths:
        task=json.loads(task_path.read_text()); task_indices.append(int(task.get("task_index",-1)))
        if task.get("status")!="pass" or int(task.get("task_count",-1))!=13 or int(task.get("slides",-1))!=len(task.get("outputs",[])): raise ValueError(f"invalid stain UNI task manifest: {task_path}")
        locked_hashes.add(str(task.get("locked_index_sha256",""))); selection_hashes.add(str(task.get("vahadane_selection_sha256",""))); existing_hashes.add(str(task.get("existing_uni_metrics_sha256","")))
        for item in task["outputs"]:
            slide_id=str(item["slide_id"])
            if slide_id in expected_outputs: raise ValueError(f"duplicate stain UNI slide across task manifests: {slide_id}")
            expected_outputs[slide_id]=item
    if sorted(task_indices)!=list(range(13)) or len(expected_outputs)!=103 or len(locked_hashes)!=1 or len(selection_hashes)!=1 or len(existing_hashes)!=1: raise ValueError("stain UNI task-manifest cohort/hash coverage mismatch")
    paths=sorted((output_root/"shards").glob("*.h5"))
    if len(paths)!=103: raise ValueError(f"expected 103 UNI shards, got {len(paths)}")
    metadata=[]; generated_values=[]; target_values=[]
    observed_slides=set(); expected_conditions=["source:at2"]+[f"target:{s}" for s in PANEL_A_TARGETS]+[f"{method}:{s}" for method in ("macenko","vahadane") for s in PANEL_A_TARGETS]
    for path in paths:
        with h5py.File(path,"r") as store:
            feat=np.asarray(store["features"],dtype=np.float32); names=[x.decode() for x in store["condition_names"][:]]; locations=np.asarray(store["location_index"],dtype=int); slide_id=str(store.attrs["slide_id"]); fold=int(store.attrs["fold"])
            if str(store.attrs.get("analysis",""))!="panel_a_stain_uni_v4" or str(store.attrs.get("locked_index_sha256",""))!=next(iter(locked_hashes)) or str(store.attrs.get("vahadane_selection_sha256",""))!=next(iter(selection_hashes)): raise ValueError(f"stain UNI shard contract mismatch: {path}")
            reference_bundle_sha=str(store.attrs.get("reference_bundle_sha256",""))
        if slide_id in observed_slides or slide_id not in expected_outputs: raise ValueError(f"unexpected/duplicate stain UNI slide: {slide_id}")
        expected=expected_outputs[slide_id]
        if sha256(path)!=str(expected.get("sha256","")) or fold!=int(expected.get("fold",-1)) or reference_bundle_sha!=str(expected.get("reference_bundle_sha256","")): raise ValueError(f"stain UNI task/shard hash mismatch: {slide_id}")
        reference_hashes=expected.get("reference_hashes",{})
        if len(reference_hashes)!=5: raise ValueError("stain UNI shard does not bind five references")
        for scanner,expected_hash in reference_hashes.items():
            ref=output_root.parent/"references"/f"fold_{fold}"/f"{scanner}.json"
            if not ref.is_file() or sha256(ref)!=expected_hash: raise ValueError("stain UNI reference hash mismatch")
        if names!=expected_conditions or feat.shape!=(20,len(expected_conditions),1024) or len(np.unique(locations))!=20 or not np.isfinite(feat).all(): raise ValueError(f"stain UNI shard shape/schema/finite mismatch: {slide_id}")
        norms=np.linalg.norm(feat,axis=2)
        if not np.allclose(norms,1.0,atol=2e-3,rtol=0): raise ValueError(f"stain UNI embedding norm mismatch: {slide_id}")
        observed_slides.add(slide_id)
        index={name:i for i,name in enumerate(names)}; source=feat[:,index["source:at2"]]
        for scanner in PANEL_A_TARGETS:
            target=feat[:,index[f"target:{scanner}"]]; raw_cos=np.sum(source*target,axis=1)
            for arm in ("raw","macenko","vahadane"):
                generated=source if arm=="raw" else feat[:,index[f"{arm}:{scanner}"]]
                gen_cos=np.sum(generated*target,axis=1); src_cos=np.sum(generated*source,axis=1)
                for i,location in enumerate(locations):
                    metadata.append({"slide_id":slide_id,"fold":fold,"location_index":int(location),"scanner":scanner,"arm":arm,"source_target_cosine":float(raw_cos[i]),"generated_target_cosine":float(gen_cos[i]),"generated_source_cosine":float(src_cos[i]),"target_gain":float(gen_cos[i]-raw_cos[i])}); generated_values.append(generated[i]); target_values.append(target[i])
    if len(observed_slides)!=103: raise ValueError("stain UNI does not cover 103 unique slides")
    frame=pd.DataFrame(metadata); gen=np.asarray(generated_values,np.float32); real=np.asarray(target_values,np.float32)
    if len(frame)!=103*20*5*3 or sorted(frame.fold.unique())!=list(range(5)): raise ValueError("stain UNI aggregate cardinality/fold mismatch")
    predictions=[]
    for (scanner,arm), group_index in frame.groupby(["scanner","arm"]).groups.items():
        mask=np.zeros(len(frame),dtype=bool); mask[np.asarray(list(group_index),dtype=int)]=True
        for fold in range(5):
            train=mask&(frame.fold.to_numpy()!=fold); test=mask&(frame.fold.to_numpy()==fold)
            xtrain=np.concatenate([gen[train],real[train]]); ytrain=np.concatenate([np.zeros(train.sum()),np.ones(train.sum())]); xtest=np.concatenate([gen[test],real[test]]); ytest=np.concatenate([np.zeros(test.sum()),np.ones(test.sum())])
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000,solver="liblinear",random_state=20260917)); clf.fit(xtrain,ytrain); prob=clf.predict_proba(xtest)[:,1]; pred=(prob>=.5).astype(int)
            test_meta=pd.concat([frame.loc[test,["slide_id","location_index"]],frame.loc[test,["slide_id","location_index"]]],ignore_index=True)
            for item,truth,guess,score in zip(test_meta.itertuples(index=False),ytest,pred,prob): predictions.append({"scanner":scanner,"arm":arm,"fold":fold,"slide_id":item.slide_id,"location_index":int(item.location_index),"truth":int(truth),"predicted":int(guess),"probability":float(score),"correct":int(truth==guess)})
    pred_frame=pd.DataFrame(predictions); summaries=[]
    for (scanner,arm),group in frame.groupby(["scanner","arm"]):
        slide=group.groupby("slide_id")[["generated_target_cosine","generated_source_cosine","target_gain"]].mean(); p=pred_frame[(pred_frame.scanner==scanner)&(pred_frame.arm==arm)]; slide_acc=p.groupby("slide_id").correct.mean()
        row={"scanner":scanner,"arm":arm,"slides":len(slide),"locations":len(group),"balanced_accuracy":float(balanced_accuracy_score(p.truth,p.predicted)),"macro_auroc":float(roc_auc_score(p.truth,p.probability)),"ci_cluster_unit":"slide","ci_level":.95,"classifier_estimand":"five-fold out-of-fold real-target versus generated discrimination; 0.5 indicates no separability and higher values indicate worse scanner harmonization"}
        rng=np.random.default_rng(stable_seed("uni-bootstrap",scanner,arm)); idx=rng.integers(0,len(slide),size=(20000,len(slide)))
        for metric in ("generated_target_cosine","generated_source_cosine","target_gain"):
            values=slide[metric].to_numpy(); draws=values[idx].mean(1); row[metric]=float(values.mean()); row[f"{metric}_ci_low"]=float(np.quantile(draws,.025)); row[f"{metric}_ci_high"]=float(np.quantile(draws,.975))
        values=slide_acc.reindex(slide.index).to_numpy(); draws=values[idx].mean(1); row["domain_accuracy_slide_mean"]=float(values.mean()); row["domain_accuracy_ci_low"]=float(np.quantile(draws,.025)); row["domain_accuracy_ci_high"]=float(np.quantile(draws,.975))
        # With exactly 20 generated and 20 target observations per slide,
        # balanced accuracy is the mean of the two slide/class sensitivities.
        # This makes the cluster bootstrap exact without treating locations as
        # independent replicates.
        class_accuracy=p.groupby(["slide_id","truth"]).correct.mean().unstack("truth").reindex(slide.index)
        if class_accuracy.isna().any().any() or set(class_accuracy.columns)!={0,1}: raise ValueError("domain predictions do not contain both classes for every slide")
        balanced_by_slide=class_accuracy[[0,1]].mean(axis=1).to_numpy(); balanced_draws=balanced_by_slide[idx].mean(1)
        row["balanced_accuracy_ci_low"]=float(np.quantile(balanced_draws,.025)); row["balanced_accuracy_ci_high"]=float(np.quantile(balanced_draws,.975))
        # AUROC is pooled across held-out locations.  Re-sample whole slides
        # and use their multiplicities as sample weights (2,000 deterministic
        # replicates keeps this nonlinear interval tractable).
        slide_codes=pd.Categorical(p.slide_id,categories=slide.index).codes
        auc_draws=np.empty(2000,dtype=float)
        for draw_number in range(len(auc_draws)):
            multiplicity=np.bincount(idx[draw_number],minlength=len(slide))
            auc_draws[draw_number]=roc_auc_score(p.truth,p.probability,sample_weight=multiplicity[slide_codes])
        row["macro_auroc_ci_low"]=float(np.quantile(auc_draws,.025)); row["macro_auroc_ci_high"]=float(np.quantile(auc_draws,.975)); row["macro_auroc_bootstrap_replicates"]=len(auc_draws); summaries.append(row)
    write_frame(output_root/"location_metrics.csv.gz",frame); write_frame(output_root/"classifier_predictions.csv.gz",pred_frame); write_frame(output_root/"summary.csv",pd.DataFrame(summaries))
    payload={"status":"pass","slides":len(observed_slides),"locations":int(frame[["slide_id","location_index"]].drop_duplicates().shape[0]),"rows":len(frame),"summary_rows":len(summaries),"task_manifests":len(task_paths),"locked_index_sha256":next(iter(locked_hashes)),"vahadane_selection_sha256":next(iter(selection_hashes)),"method_status":{"raw":"evaluated","macenko":"evaluated","vahadane":"evaluated"},"outputs":{"location_metrics":{"path":str((output_root/"location_metrics.csv.gz").resolve()),"sha256":sha256(output_root/"location_metrics.csv.gz")},"classifier_predictions":{"path":str((output_root/"classifier_predictions.csv.gz").resolve()),"sha256":sha256(output_root/"classifier_predictions.csv.gz")},"summary":{"path":str((output_root/"summary.csv").resolve()),"sha256":sha256(output_root/"summary.csv")}}}; write_json(output_root/"aggregate_manifest.json",payload); return payload


def validate_gan_evaluation(
    *, evaluation_root: Path, space: str, method: str, target_scanner: str,
    allow_reused: bool = False,
) -> dict[str, Any]:
    """Validate the exact five-fold/103-slide locked evaluation and hashes."""

    from scanner_gan.evaluate_images import IMAGE_EVALUATION_VERSION
    from scanner_gan.evaluate_uni import UNI_EVALUATION_VERSION
    space=space.strip().lower(); method=method.strip().lower(); target=target_scanner.strip().lower()
    if space not in {"image","uni"} or method not in {"pix2pix","cyclegan"}: raise ValueError("invalid evaluation kind")
    manifest_path=evaluation_root/"evaluation_manifest.json"; manifest=json.loads(manifest_path.read_text())
    current_version=IMAGE_EVALUATION_VERSION if space=="image" else UNI_EVALUATION_VERSION
    allowed_versions={current_version}
    if allow_reused and method=="pix2pix":
        allowed_versions.add("pix2pix_locked_image_safety_v2" if space=="image" else "pix2pix_frozen_uni_reference_v2")
    if str(manifest.get("evaluation_version","")) not in allowed_versions: raise ValueError(f"unexpected {space} evaluation version")
    if str(manifest.get("status",""))!="complete": raise ValueError(f"{space} evaluation incomplete")
    if str(manifest.get("source_scanner","")).lower()!="at2" or str(manifest.get("target_scanner","")).lower()!=target: raise ValueError(f"{space} direction mismatch")
    observed_method=str(manifest.get("method",method if allow_reused else "")).lower()
    if observed_method!=method: raise ValueError(f"{space} method mismatch")
    if str(manifest.get("analysis_role",""))!="primary_translation" or str(manifest.get("inference_input_scanner","")).lower()!="at2": raise ValueError(f"{space} analysis role mismatch")
    if sorted(int(x) for x in manifest.get("outer_folds",[]))!=list(range(5)): raise ValueError(f"{space} folds are not exactly 0..4")
    if int(manifest.get("locations",-1))!=4120 or int(manifest.get("physical_slides",-1))!=103: raise ValueError(f"{space} evaluation is not exactly 103 slides/4120 locations")
    outputs=manifest.get("outputs",{})
    names={"location_metrics":"location_metrics.csv.gz","slide_metrics":"slide_metrics.csv","summary":"summary.csv"}
    for key,name in names.items():
        item=outputs.get(key,{}) if isinstance(outputs,Mapping) else {}
        path=Path(str(item.get("path",""))).resolve(); expected=(evaluation_root/name).resolve()
        if path!=expected or not path.is_file() or sha256(path)!=str(item.get("sha256","")): raise ValueError(f"{space} output hash/path mismatch for {key}")
    location=pd.read_csv(evaluation_root/"location_metrics.csv.gz",dtype={"slide_id":str})
    slide=pd.read_csv(evaluation_root/"slide_metrics.csv",dtype={"slide_id":str})
    if len(location)!=4120 or location.slide_id.nunique()!=103 or len(slide)!=103 or slide.slide_id.nunique()!=103: raise ValueError(f"{space} CSV cardinality mismatch")
    fold_column="fold" if "fold" in location else "test_fold"
    if fold_column not in location or sorted(location[fold_column].astype(int).unique())!=list(range(5)): raise ValueError(f"{space} location folds mismatch")
    prediction_items=manifest.get("prediction_manifests",[])
    if len(prediction_items)!=5: raise ValueError(f"{space} does not bind exactly five prediction manifests")
    seen_folds=[]; total_samples=0; total_slides=0
    for item in prediction_items:
        path=Path(str(item.get("path",""))).resolve()
        if not path.is_file() or sha256(path)!=str(item.get("sha256","")): raise ValueError("prediction manifest hash mismatch")
        prediction=json.loads(path.read_text()); seen_folds.append(int(prediction.get("test_fold",-1))); total_samples+=int(prediction.get("samples",-1)); total_slides+=int(prediction.get("slides",-1))
        if str(prediction.get("status",""))!="complete" or str(prediction.get("direction","")).lower()!=f"at2->{target}": raise ValueError("prediction manifest direction/status mismatch")
        prediction_method=str(prediction.get("method",method if allow_reused else "")).lower()
        if prediction_method!=method or int(prediction.get("locations_per_slide",-1))!=40: raise ValueError("prediction method/location contract mismatch")
    if sorted(seen_folds)!=list(range(5)) or total_samples!=4120 or total_slides!=103: raise ValueError("prediction manifests do not total five folds/103 slides/4120 locations")
    if space=="image":
        gates=manifest.get("safety_gates",{})
        if not gates or any("pass" not in value for value in gates.values()): raise ValueError("image safety/fidelity gates missing")
        gate_pass=all(bool(value["pass"]) for value in gates.values())
        if bool(manifest.get("all_available_safety_gates_pass"))!=gate_pass: raise ValueError("image safety gate summary mismatch")
    else:
        boundary=str(manifest.get("interpretation_boundary",""))
        if "cannot override" not in boundary: raise ValueError("UNI interpretation boundary is missing")
    return manifest


def assemble_panel_a(*, analysis_root: Path, panel_root: Path, stain_subdir: str) -> dict[str, Any]:
    """Assemble reused and new AT2-to-scanner results into one long table."""

    summary_rows: list[dict[str, Any]]=[]; slide_frames: list[pd.DataFrame]=[]
    if stain_subdir != FINAL_STAIN_SUBDIR:
        raise ValueError(
            f"final assembly requires the frozen stain root {FINAL_STAIN_SUBDIR!r}, "
            f"got {stain_subdir!r}"
        )
    stain_root = panel_root / stain_subdir
    stain_manifest_path = stain_root / "aggregate_manifest.json"
    stain_manifest = json.loads(stain_manifest_path.read_text())
    if stain_manifest.get("status") != "pass" or stain_manifest.get("fit_quality_valid") is not True:
        raise RuntimeError("selected stain result has not passed the fit-quality gate")
    for key,name in {"location_metrics":"stain_location_metrics.csv.gz","scanner_summary":"stain_scanner_summary.csv"}.items():
        item=stain_manifest.get("outputs",{}).get(key,{})
        path=Path(str(item.get("path",""))).resolve(); expected=(stain_root/name).resolve()
        if path!=expected or not path.is_file() or sha256(path)!=str(item.get("sha256","")): raise RuntimeError(f"stain aggregate output hash mismatch: {key}")
    stain_uni_manifest_path=stain_root/"uni/aggregate_manifest.json"
    stain_uni_manifest=json.loads(stain_uni_manifest_path.read_text())
    if stain_uni_manifest.get("status")!="pass": raise RuntimeError("selected stain UNI result is incomplete")
    for key,name in {"location_metrics":"location_metrics.csv.gz","classifier_predictions":"classifier_predictions.csv.gz","summary":"summary.csv"}.items():
        item=stain_uni_manifest.get("outputs",{}).get(key,{})
        path=Path(str(item.get("path",""))).resolve(); expected=(stain_root/"uni"/name).resolve()
        if path!=expected or not path.is_file() or sha256(path)!=str(item.get("sha256","")): raise RuntimeError(f"stain UNI aggregate output hash mismatch: {key}")
    vahadane_status=stain_manifest.get("method_status",{}).get("vahadane",{})
    if vahadane_status.get("status")!="evaluated" or vahadane_status.get("fit_quality_valid") is not True: raise RuntimeError("final assembly requires valid globally selected Vahadane results")
    classical_image_location=pd.read_csv(analysis_root/"02_correction/location_metrics.csv",dtype={"slide_id":str})
    classical_image_location=classical_image_location[classical_image_location.arm.isin(["raw","reinhard","frequency","combined"])]
    classical_image_path=analysis_root/"02_correction/scanner_summary.csv"
    classical_image=pd.read_csv(classical_image_path)
    classical_image=classical_image[classical_image.arm.isin(["raw","reinhard","frequency","combined"])]
    for item in classical_image.itertuples(index=False):
        for metric in ("distance","endpoint_coverage","all_endpoints_covered","fidelity_ncc"):
            summary_rows.append({"space":"image","scanner":item.scanner,"method":item.arm,"metric":metric,"estimate":float(getattr(item,metric)),"ci_low":float(getattr(item,f"{metric}_ci_low")),"ci_high":float(getattr(item,f"{metric}_ci_high")),"provenance":"reused_02_correction"})
    # The historical scanner summary predates these two common Panel A image
    # endpoints, so reproduce its slide-cluster bootstrap from location rows.
    for (scanner,arm),group in classical_image_location.groupby(["scanner","arm"]):
        slide=group.groupby("slide_id")[["target_gradient_ncc","saturation_fraction"]].mean()
        rng=np.random.default_rng(stable_seed("assembly-image",scanner,arm)); idx=rng.integers(0,len(slide),size=(20000,len(slide)))
        for metric in slide.columns:
            values=slide[metric].to_numpy(); draws=values[idx].mean(1)
            summary_rows.append({"space":"image","scanner":scanner,"method":arm,"metric":metric,"estimate":float(values.mean()),"ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),"provenance":"reused_02_correction_reaggregated_by_scanner"})
    stain_image=pd.read_csv(stain_root/"stain_scanner_summary.csv")
    if len(stain_image)!=10 or set(stain_image.scanner)!=set(PANEL_A_TARGETS) or set(stain_image.arm)!={"macenko","vahadane"}: raise RuntimeError("stain image summary cardinality mismatch")
    for item in stain_image.itertuples(index=False):
        for metric in ("distance","endpoint_coverage","all_endpoints_covered","fidelity_ncc","target_gradient_ncc","saturation_fraction","stain_fallback","stain_preclip_fraction"):
            summary_rows.append({"space":"image","scanner":item.scanner,"method":item.arm,"metric":metric,"estimate":float(getattr(item,metric)),"ci_low":float(getattr(item,f"{metric}_ci_low")),"ci_high":float(getattr(item,f"{metric}_ci_high")),"provenance":"new_panel_a_stain"})

    # Scanner-specific classical UNI summaries (the historical summary pooled scanners).
    classical_uni=pd.read_csv(analysis_root/"03_uni/location_metrics.csv",dtype={"slide_id":str})
    classical_uni=classical_uni[classical_uni.arm.isin(["raw","reinhard","frequency","combined"])]
    for (scanner,arm),group in classical_uni.groupby(["scanner","arm"]):
        slide=group.groupby("slide_id")[["generated_target_cosine","generated_source_cosine","target_gain"]].mean(); rng=np.random.default_rng(stable_seed("assembly",scanner,arm)); idx=rng.integers(0,len(slide),size=(20000,len(slide)))
        for metric in slide.columns:
            values=slide[metric].to_numpy(); draws=values[idx].mean(1); summary_rows.append({"space":"uni","scanner":scanner,"method":arm,"metric":metric,"estimate":float(values.mean()),"ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),"provenance":"reused_03_uni_reaggregated_by_scanner"})
    stain_uni=pd.read_csv(stain_root/"uni/summary.csv")
    if len(stain_uni)!=15 or set(stain_uni.scanner)!=set(PANEL_A_TARGETS) or set(stain_uni.arm)!={"raw","macenko","vahadane"}: raise RuntimeError("stain UNI summary cardinality mismatch")
    for item in stain_uni.itertuples(index=False):
        if item.arm=="raw": continue
        for metric in ("generated_target_cosine","generated_source_cosine","target_gain"):
            summary_rows.append({"space":"uni","scanner":item.scanner,"method":item.arm,"metric":metric,"estimate":float(getattr(item,metric)),"ci_low":float(getattr(item,f"{metric}_ci_low")),"ci_high":float(getattr(item,f"{metric}_ci_high")),"provenance":"new_panel_a_stain_uni"})
        for metric in ("balanced_accuracy","macro_auroc","domain_accuracy_slide_mean"):
            ci_stem="domain_accuracy" if metric=="domain_accuracy_slide_mean" else metric
            summary_rows.append({"space":"uni","scanner":item.scanner,"method":item.arm,"metric":metric,"estimate":float(getattr(item,metric)),"ci_low":float(getattr(item,f"{ci_stem}_ci_low")),"ci_high":float(getattr(item,f"{ci_stem}_ci_high")),"estimand":str(item.classifier_estimand),"ci_cluster_unit":str(item.ci_cluster_unit),"provenance":"new_panel_a_stain_uni"})

    learned_specs=[
        ("pix2pix","gt450",analysis_root/"06_learned_baselines/09_bidirectional_full_training","reused_full_training"),
        ("cyclegan","gt450",analysis_root/"06_learned_baselines/11_cyclegan_full_training","reused_inflight_22759523"),
    ]
    for method in ("pix2pix","cyclegan"):
        subdir="03_pix2pix" if method=="pix2pix" else "04_cyclegan"
        for scanner in MISSING_GAN_TARGETS: learned_specs.append((method,scanner,panel_root/subdir,"new_panel_a_full_training"))
    for method,scanner,root,provenance in learned_specs:
        image_root=root/"03_image_evaluation"/f"at2_to_{scanner}"
        image_manifest=validate_gan_evaluation(evaluation_root=image_root,space="image",method=method,target_scanner=scanner,allow_reused=provenance.startswith("reused"))
        fidelity_gate_pass=bool(image_manifest["all_available_safety_gates_pass"])
        summary_rows.append({"space":"image","scanner":scanner,"method":method,"metric":"fidelity_gate_pass","estimate":float(fidelity_gate_pass),"ci_low":float("nan"),"ci_high":float("nan"),"provenance":provenance})
        for space,folder in (("image","03_image_evaluation"),("uni","04_uni")):
            path=root/folder/f"at2_to_{scanner}"/"summary.csv"
            validate_gan_evaluation(evaluation_root=path.parent,space=space,method=method,target_scanner=scanner,allow_reused=provenance.startswith("reused"))
            table=pd.read_csv(path)
            for item in table.itertuples(index=False):
                summary_rows.append({"space":space,"scanner":scanner,"method":method,"metric":item.metric,"estimate":float(item.estimate),"ci_low":float(item.ci_low),"ci_high":float(item.ci_high),"fidelity_gate_pass":fidelity_gate_pass,"representation_success_eligible":bool(fidelity_gate_pass) if space=="uni" else float("nan"),"provenance":provenance})
            slide_path=root/folder/f"at2_to_{scanner}"/"slide_metrics.csv"
            slide=pd.read_csv(slide_path,dtype={"slide_id":str}); slide["space"]=space; slide["scanner"]=scanner; slide["method"]=method; slide_frames.append(slide)

    # Compact slide-level audit file spanning the conventional arms.
    image_location=classical_image_location
    image_slide=image_location.groupby(["slide_id","scanner","arm"],as_index=False)[["distance","endpoint_coverage","all_endpoints_covered","fidelity_ncc","target_gradient_ncc","saturation_fraction"]].mean().rename(columns={"arm":"method"}); image_slide["space"]="image"; slide_frames.append(image_slide)
    stain_image_loc=pd.read_csv(stain_root/"stain_location_metrics.csv.gz",dtype={"slide_id":str}); stain_image_slide=stain_image_loc.groupby(["slide_id","scanner","arm"],as_index=False)[["distance","endpoint_coverage","all_endpoints_covered","fidelity_ncc","target_gradient_ncc","saturation_fraction","stain_fallback","stain_preclip_fraction"]].mean().rename(columns={"arm":"method"}); stain_image_slide["space"]="image"; slide_frames.append(stain_image_slide)
    uni_slide=classical_uni.groupby(["slide_id","scanner","arm"],as_index=False)[["generated_target_cosine","generated_source_cosine","target_gain"]].mean().rename(columns={"arm":"method"}); uni_slide["space"]="uni"; slide_frames.append(uni_slide)
    stain_uni_loc=pd.read_csv(stain_root/"uni/location_metrics.csv.gz",dtype={"slide_id":str}); stain_uni_slide=stain_uni_loc[stain_uni_loc.arm.isin(["macenko","vahadane"])].groupby(["slide_id","scanner","arm"],as_index=False)[["generated_target_cosine","generated_source_cosine","target_gain"]].mean().rename(columns={"arm":"method"}); stain_uni_slide["space"]="uni"; slide_frames.append(stain_uni_slide)
    summary=pd.DataFrame(summary_rows).sort_values(["space","scanner","method","metric"])
    details=pd.concat(slide_frames,ignore_index=True,sort=False)
    write_frame(panel_root/"baseline_panel_a_all_scanners.csv",summary)
    write_frame(panel_root/"baseline_location_or_slide_metrics.csv.gz",details)
    lines=["# Panel A baseline values","", "Direction: AT2 → each non-reference scanner.", "", markdown_table(summary)]
    (panel_root/"baseline_table_values.md").write_text("\n".join(lines)+"\n")
    payload={"status":"pass","summary_rows":len(summary),"detail_rows":len(details),"methods":sorted(summary.method.unique()),"scanners":sorted(summary.scanner.unique()),"method_status":{"vahadane":{"status":"evaluated","configuration":vahadane_status.get("configuration")}},"stain_subdir":stain_subdir,"stain_manifest_sha256":sha256(stain_manifest_path),"stain_uni_manifest_sha256":sha256(stain_uni_manifest_path),"stain_fallback_count":stain_manifest["fallback_count"],"vahadane_convergence_warning_count":stain_manifest["vahadane_convergence_warning_count"],"image_estimand":"mean of slide-level means with 95% slide-cluster bootstrap CI","stain_uni_separability_estimand":"five-fold out-of-fold real-target versus generated classifier; 0.5 indicates no separability, higher indicates worse harmonization; 95% CI clusters by slide","slurm_dependency_job_ids":os.environ.get("JOB_PANEL_A_DEPENDENCIES","").split(":"),"summary_sha256":sha256(panel_root/"baseline_panel_a_all_scanners.csv"),"details_sha256":sha256(panel_root/"baseline_location_or_slide_metrics.csv.gz")}
    write_json(panel_root/"method_fit_manifest.json",payload); return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train-pix2pix", "train-cyclegan"):
        item = sub.add_parser(name)
        item.add_argument("--sample-index", type=Path, required=True)
        item.add_argument("--target-scanner", choices=MISSING_GAN_TARGETS, required=True)
        item.add_argument("--test-fold", type=int, choices=range(5), required=True)
        item.add_argument("--output-dir", type=Path, required=True)
        item.add_argument("--seed", type=int, required=True)
        item.add_argument("--max-passes", type=int, default=200)
        item.add_argument("--max-steps", type=int, default=0)
        item.add_argument("--batch-size", type=int, default=16)
        item.add_argument("--validation-batch-size", type=int, default=32)
        item.add_argument("--num-workers", type=int, default=4)
        item.add_argument("--selection-every", type=int, default=5)
    fit = sub.add_parser("fit-stain-reference")
    fit.add_argument("--sample-index", type=Path, required=True)
    fit.add_argument("--locked-index", type=Path, required=True)
    fit.add_argument("--target-scanner", choices=PANEL_A_TARGETS, required=True)
    fit.add_argument("--test-fold", type=int, choices=range(5), required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--vahadane-selection-manifest", type=Path, required=True)
    evaluate = sub.add_parser("evaluate-stain-slide")
    evaluate.add_argument("--locked-index", type=Path, required=True)
    evaluate.add_argument("--reference-root", type=Path, required=True)
    evaluate.add_argument("--target-reference-root", type=Path, required=True)
    evaluate.add_argument("--slide-index", type=int, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--vahadane-selection-manifest", type=Path, required=True)
    aggregate = sub.add_parser("aggregate-stain")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--vahadane-selection-manifest", type=Path, required=True)
    predict_parser = sub.add_parser("predict-gan")
    predict_parser.add_argument("--method", choices=("pix2pix", "cyclegan"), required=True)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--run-manifest", type=Path, required=True)
    predict_parser.add_argument("--sample-index", type=Path, required=True)
    predict_parser.add_argument("--locked-index", type=Path, required=True)
    predict_parser.add_argument("--target-scanner", choices=MISSING_GAN_TARGETS, required=True)
    predict_parser.add_argument("--test-fold", type=int, choices=range(5), required=True)
    predict_parser.add_argument("--output-dir", type=Path, required=True)
    predict_parser.add_argument("--batch-size", type=int, default=32)
    predict_parser.add_argument("--num-workers", type=int, default=4)
    uni = sub.add_parser("stain-uni-extract")
    uni.add_argument("--task-index", type=int, required=True)
    uni.add_argument("--task-count", type=int, required=True)
    uni.add_argument("--existing-uni-metrics", type=Path, required=True)
    uni.add_argument("--locked-index", type=Path, required=True)
    uni.add_argument("--reference-root", type=Path, required=True)
    uni.add_argument("--output-root", type=Path, required=True)
    uni.add_argument("--vahadane-selection-manifest", type=Path, required=True)
    uni.add_argument("--batch-size", type=int, default=64)
    uni_agg = sub.add_parser("stain-uni-aggregate")
    uni_agg.add_argument("--output-root", type=Path, required=True)
    survey = sub.add_parser("survey-vahadane")
    survey.add_argument("--locked-index", type=Path, required=True)
    survey.add_argument("--output", type=Path, required=True)
    survey.add_argument("--fold", type=int, choices=range(5), required=True)
    survey.add_argument("--samples-per-fold", type=int, default=10)
    survey.add_argument("--max-iterations", type=lambda value: tuple(int(x) for x in value.split(",")), default=(1000,2000,4000))
    survey.add_argument("--tolerances", type=lambda value: tuple(float(x) for x in value.split(",")), default=(1e-3,))
    survey.add_argument("--alphas", type=lambda value: tuple(float(x) for x in value.split(",")), default=(1e-3,))
    survey.add_argument("--solvers", type=lambda value: tuple(x.strip() for x in value.split(",")), default=("cd",))
    validate_eval = sub.add_parser("validate-evaluation")
    validate_eval.add_argument("--evaluation-root",type=Path,required=True)
    validate_eval.add_argument("--space",choices=("image","uni"),required=True)
    validate_eval.add_argument("--method",choices=("pix2pix","cyclegan"),required=True)
    validate_eval.add_argument("--target-scanner",choices=MISSING_GAN_TARGETS,required=True)
    assemble = sub.add_parser("assemble-panel-a")
    assemble.add_argument("--analysis-root", type=Path, required=True)
    assemble.add_argument("--panel-root", type=Path, required=True)
    assemble.add_argument("--stain-subdir", default=FINAL_STAIN_SUBDIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_runtime_contract()
    if args.command == "fit-stain-reference":
        fit_stain_reference(args.sample_index, args.locked_index, args.target_scanner, args.test_fold, args.output, args.vahadane_selection_manifest)
        return
    if args.command == "evaluate-stain-slide":
        evaluate_stain_slide(args.locked_index, args.reference_root, args.target_reference_root, args.slide_index, args.output_root, args.vahadane_selection_manifest)
        return
    if args.command == "aggregate-stain":
        aggregate_stain_results(args.output_root,args.vahadane_selection_manifest)
        return
    if args.command == "predict-gan":
        predict_gan(method=args.method, checkpoint=args.checkpoint, run_manifest=args.run_manifest, sample_index=args.sample_index, locked_index=args.locked_index, target_scanner=args.target_scanner, test_fold=args.test_fold, output_dir=args.output_dir, batch_size=args.batch_size, num_workers=args.num_workers)
        return
    if args.command == "stain-uni-extract":
        stain_uni_extract(task_index=args.task_index, task_count=args.task_count, existing_uni_metrics=args.existing_uni_metrics, locked_index=args.locked_index, reference_root=args.reference_root, output_root=args.output_root, vahadane_selection_path=args.vahadane_selection_manifest, batch_size=args.batch_size)
        return
    if args.command == "stain-uni-aggregate":
        aggregate_stain_uni(args.output_root)
        return
    if args.command == "survey-vahadane":
        survey_vahadane_source_convergence(args.locked_index,args.output,args.fold,samples_per_fold=args.samples_per_fold,max_iterations=args.max_iterations,tolerances=args.tolerances,alphas=args.alphas,solvers=args.solvers)
        return
    if args.command == "validate-evaluation":
        validate_gan_evaluation(evaluation_root=args.evaluation_root,space=args.space,method=args.method,target_scanner=args.target_scanner)
        return
    if args.command == "assemble-panel-a":
        assemble_panel_a(analysis_root=args.analysis_root, panel_root=args.panel_root, stain_subdir=args.stain_subdir)
        return
    validation_fold=(args.test_fold+1)%5
    if args.command == "train-pix2pix":
        config=Pix2PixConfig(sample_index=str(args.sample_index),source_scanner="at2",target_scanner=args.target_scanner,test_fold=args.test_fold,validation_fold=validation_fold,output_dir=str(args.output_dir),seed=args.seed,max_passes=args.max_passes,max_steps=args.max_steps,locations_per_train_slide=100,locations_per_validation_slide=40,batch_size=args.batch_size,validation_batch_size=args.validation_batch_size,num_workers=args.num_workers,learning_rate=2e-4,lambda_l1=100,amp="bfloat16",constant_passes=100,selection_policy="target_l1",selection_every=args.selection_every,save_every=0)
        train_pix2pix(config)
    elif args.command == "train-cyclegan":
        config=CycleGANConfig(sample_index=str(args.sample_index),domain_a=args.target_scanner,domain_b="at2",test_fold=args.test_fold,validation_fold=validation_fold,output_dir=str(args.output_dir),seed=args.seed,max_passes=args.max_passes,max_steps=args.max_steps,locations_per_train_slide=100,locations_per_validation_slide=40,batch_size=args.batch_size,validation_batch_size=args.validation_batch_size,num_workers=args.num_workers,learning_rate=2e-4,lambda_cycle=10,lambda_identity=5,pool_size=50,amp="bfloat16",constant_passes=100,selection_every=args.selection_every,save_every=0)
        train_cyclegan(config)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
