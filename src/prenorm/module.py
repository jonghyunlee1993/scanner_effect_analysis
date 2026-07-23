"""Lightning training module for prototype-only registration supervision."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytorch_lightning as pl
import torch

from utils.config import to_dict

from .augmentation import ContentNuisanceAugmenter
from .losses import (
    apply_scanner_pair_weights,
    build_supervision_weights,
    coordinate_variance_loss,
    frequency_asymmetric_standard_loss,
    high_frequency_reliability,
    nuclei_masked_source_detail_loss,
    nuclei_masked_source_orientation_loss,
    prototype_regression_loss,
    reference_lowpass_pair_distances,
    reference_neighborhood_consistency_loss,
    nuclei_reconstruction_loss,
    sample_translation_pairs,
    spatial_pair_distances,
    standard_loss,
    structural_detail_loss,
    target_structural_detail_loss,
    translation_loss,
    weighted_mean_with_coverage,
)
from .metrics import focus_score, ssim, standard_convergence
from .models import Canonicalizer


CHECKPOINT_FORMAT_VERSION = 3


class Phase1Module(pl.LightningModule):
    """Train the AT2-anchored canonicalizer and observed scanner renderers."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model
        holdout = getattr(cfg.experiment, "holdout_scanner", None)
        model_scanners = [scanner for scanner in cfg.scanners if scanner != holdout]
        self.model = Canonicalizer(
            scanners=model_scanners,
            reference_scanner=cfg.reference_scanner,
            stain_reference_path=cfg.paths.stain_reference,
            input_mode=cfg.input.mode,
            content_ch=model_cfg.content_ch,
            style_dim=model_cfg.style_dim,
            ngf=model_cfg.ngf,
            n_downsample=model_cfg.n_downsample,
            n_res=model_cfg.n_res,
            canonical_skip=getattr(model_cfg, "canonical_skip", "none"),
            canonical_skip_kernel=getattr(model_cfg, "canonical_skip_kernel", 9),
            nuclei_head=bool(getattr(model_cfg, "nuclei_head", False)),
        )
        self.model.input_builder.group_dropout_p = float(cfg.input.group_dropout_p)
        augmentation = cfg.augmentation
        self.nuisance = ContentNuisanceAugmenter(
            brightness=augmentation.brightness,
            contrast=augmentation.contrast,
            gamma=augmentation.gamma,
            saturation=augmentation.saturation,
            illumination=augmentation.illumination,
            blur_probability=augmentation.blur_probability,
            blur_sigma_max=augmentation.blur_sigma_max,
            unsharp_probability=augmentation.unsharp_probability,
            noise=augmentation.noise,
            jpeg_quality=tuple(augmentation.jpeg_quality),
            resample_min_scale=augmentation.resample_min_scale,
        )
        self.reference_index = self.model.reference_index
        self.loss_weights = cfg.loss
        configured_pair_weights = getattr(self.loss_weights, "pair_scanner_weights", None)
        if configured_pair_weights is None:
            self.pair_scanner_weights = {}
        elif isinstance(configured_pair_weights, dict):
            self.pair_scanner_weights = dict(configured_pair_weights)
        else:
            self.pair_scanner_weights = dict(vars(configured_pair_weights))
        unknown_pair_scanners = sorted(
            set(self.pair_scanner_weights).difference(self.model.scanners)
        )
        if unknown_pair_scanners:
            raise ValueError(
                "pair weights configured for unknown scanners: "
                + ", ".join(unknown_pair_scanners)
            )
        self.q_min = float(cfg.registration.q_min)
        self.stain_reference_hash = hashlib.sha256(
            Path(cfg.paths.stain_reference).read_bytes()
        ).hexdigest()
        self._reference_batch = None
        self.save_hyperparameters({"resolved_config": to_dict(cfg)})

    @property
    def C(self):
        """Expose the content encoder for diagnostics."""
        return self.model.C

    @property
    def G(self):
        """Expose the renderer for diagnostics."""
        return self.model.G

    @property
    def B(self):
        """Expose scanner prototypes for diagnostics."""
        return self.model.B

    def _training_progress(self) -> float:
        total = max(int(self.trainer.estimated_stepping_batches), 1)
        return min(float(self.global_step) / total, 1.0)

    def _schedule(self, train: bool) -> dict[str, float]:
        if not train:
            return {
                "pair": float(self.loss_weights.pair_end),
                "cross": 1.0,
                "nuisance": 0.0,
                "gradient": float(self.loss_weights.gradient_end),
            }

        progress = self._training_progress()
        bootstrap = float(self.cfg.schedule.bootstrap_end)
        identification = float(self.cfg.schedule.identification_end)
        hardening = float(self.cfg.schedule.hardening_end)
        if progress < bootstrap:
            return {
                "pair": float(self.loss_weights.pair_start),
                "cross": 0.0,
                "nuisance": 0.0,
                "gradient": 0.0,
            }
        if progress < identification:
            ratio = (progress - bootstrap) / (identification - bootstrap)
            pair = self.loss_weights.pair_start + ratio * (
                self.loss_weights.pair_end - self.loss_weights.pair_start
            )
            return {"pair": float(pair), "cross": ratio, "nuisance": 0.0, "gradient": 0.0}
        if progress < hardening:
            ratio = (progress - identification) / (hardening - identification)
            return {
                "pair": float(self.loss_weights.pair_end),
                "cross": 1.0,
                "nuisance": ratio,
                "gradient": ratio * float(self.loss_weights.gradient_end),
            }
        return {
            "pair": float(self.loss_weights.pair_end),
            "cross": 1.0,
            "nuisance": 1.0,
            "gradient": float(self.loss_weights.gradient_end),
        }

    def _style_regression(
        self,
        rgb: torch.Tensor,
        present: torch.Tensor,
        slide_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        estimates = []
        targets = []
        group_count = int(torch.unique(slide_group).numel())
        for group in torch.unique(slide_group):
            coordinate = slide_group == group
            for scanner in range(rgb.shape[1]):
                scanner_present = present[coordinate, scanner]
                if not bool(scanner_present.any()):
                    continue
                context = rgb[coordinate, scanner][None]
                estimate = self.model.E_set(context, scanner_present[None])
                estimates.append(estimate[0])
                targets.append(self.model.B[scanner])
        if not estimates:
            return rgb.sum() * 0.0, rgb.new_zeros(())
        estimate_tensor = torch.stack(estimates)
        target_tensor = torch.stack(targets)
        loss, _ = prototype_regression_loss(
            estimate_tensor,
            target_tensor,
            torch.ones(len(estimates), dtype=torch.bool, device=rgb.device),
        )
        coverage = rgb.new_tensor(len(estimates) / max(group_count * rgb.shape[1], 1))
        return loss, coverage

    @staticmethod
    def _translation_term(
        rendered: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        selected: torch.Tensor,
        gradient_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(selected.any()):
            return rendered.sum() * 0.0, rendered.new_zeros(())
        return translation_loss(
            rendered[selected],
            targets[selected],
            weights[selected],
            gradient_weight=gradient_weight,
        )

    def _shared_step(self, batch, train: bool, batch_idx: int) -> dict[str, torch.Tensor]:
        clean_rgb = batch["rgb"]
        present = batch["present"]
        geom_ok = batch["geom_ok"]
        q_reg = batch["q_reg"]
        schedule = self._schedule(train)
        supervision = build_supervision_weights(present, geom_ok, q_reg, self.q_min)

        content_rgb = self.nuisance(clean_rgb, schedule["nuisance"]) if train else clean_rgb
        group_mask = (
            self.model.M.sample_group_mask(clean_rgb.shape[0], clean_rgb.device)
            if train
            else None
        )
        content = self.model.encode(content_rgb, group_mask=group_mask)
        flat_content = content.reshape(-1, *content.shape[2:])
        flat_source = clean_rgb.reshape(-1, *clean_rgb.shape[2:])
        nuclei_weight = float(getattr(self.loss_weights, "nuclei_reconstruction", 0.0))
        if nuclei_weight > 0:
            flat_canonical, flat_nuclei_logits = self.model.decode_canonical_with_nuclei(
                flat_content, flat_source
            )
            nuclei_logits = flat_nuclei_logits.view(
                *content.shape[:2], 2, *clean_rgb.shape[-2:]
            )
        else:
            flat_canonical = self.model.decode_canonical(flat_content, flat_source)
            nuclei_logits = None
        canonical = flat_canonical.view(*content.shape[:2], 3, *clean_rgb.shape[-2:])

        standard_mode = getattr(self.loss_weights, "standard_mode", "full")
        if standard_mode == "full":
            standard_value, standard_coverage = standard_loss(
                canonical,
                clean_rgb[:, self.reference_index],
                supervision["standard"],
                gradient_weight=schedule["gradient"],
            )
            standard_low_value = standard_value.detach()
            standard_high_value = canonical.sum() * 0.0
            standard_high_coverage = canonical.new_zeros(())
            detail_reliability = torch.ones_like(supervision["standard"])
            detail_coherence = torch.ones_like(supervision["standard"])
            detail_energy_ratio = torch.ones_like(supervision["standard"])
            standard_high_weights = torch.zeros_like(supervision["standard"])
        elif standard_mode == "frequency_asymmetric":
            (
                detail_reliability,
                detail_coherence,
                detail_energy_ratio,
            ) = high_frequency_reliability(
                clean_rgb,
                clean_rgb[:, self.reference_index],
                q_reg,
                self.reference_index,
                coherence_low=float(self.loss_weights.standard_coherence_low),
                coherence_high=float(self.loss_weights.standard_coherence_high),
                energy_low=float(self.loss_weights.standard_detail_energy_low),
                energy_high=float(self.loss_weights.standard_detail_energy_high),
                q_low=float(self.loss_weights.standard_detail_q_low),
                q_high=float(self.loss_weights.standard_detail_q_high),
                kernel_size=int(self.loss_weights.standard_lowpass_kernel),
                sigma=float(self.loss_weights.standard_lowpass_sigma),
            )
            standard_high_weights = supervision["standard"] * detail_reliability
            (
                standard_value,
                standard_low_value,
                standard_high_value,
                standard_coverage,
                standard_high_coverage,
            ) = frequency_asymmetric_standard_loss(
                canonical,
                clean_rgb[:, self.reference_index],
                supervision["standard"],
                standard_high_weights,
                lowpass_kernel=int(self.loss_weights.standard_lowpass_kernel),
                lowpass_sigma=float(self.loss_weights.standard_lowpass_sigma),
                high_frequency_weight=float(
                    self.loss_weights.standard_high_frequency_weight
                ),
                gradient_weight=schedule["gradient"],
            )
        else:
            raise ValueError(f"unsupported standard loss mode: {standard_mode}")

        random = None if train else torch.Generator().manual_seed(
            int(self.cfg.runtime.seed) + batch_idx
        )
        batch_index, source_index, target_index, pair_weight, is_self = sample_translation_pairs(
            present,
            supervision["cross"],
            generator=random,
        )
        if batch_index.numel():
            sampled_content = content[batch_index, source_index]
            rendered = self.model.decode(sampled_content, target_index)
            targets = clean_rgb[batch_index, target_index]
        else:
            rendered = clean_rgb.new_empty((0, 3, *clean_rgb.shape[-2:]))
            targets = rendered
        self_value, self_coverage = self._translation_term(
            rendered, targets, pair_weight, is_self, schedule["gradient"]
        )
        cross_value, cross_coverage = self._translation_term(
            rendered, targets, pair_weight, ~is_self, schedule["gradient"]
        )
        translate_value = self_value + schedule["cross"] * cross_value

        pair_weights = apply_scanner_pair_weights(
            supervision["cross"], self.model.scanners, self.pair_scanner_weights
        )
        pair_mode = getattr(self.loss_weights, "pair_mode", "symmetric")
        if pair_mode == "symmetric":
            pair_distances = spatial_pair_distances(content)
            used_pair_weights = pair_weights
        elif pair_mode == "reference_lowpass":
            pair_distances = reference_lowpass_pair_distances(
                content,
                self.reference_index,
                kernel_size=int(getattr(self.loss_weights, "pair_lowpass_kernel", 5)),
            )
            used_pair_weights = pair_weights[:, :, self.reference_index]
        else:
            raise ValueError(f"unsupported pair loss mode: {pair_mode}")
        pair_value, pair_coverage = weighted_mean_with_coverage(
            pair_distances, used_pair_weights
        )

        neighborhood_weight = float(
            getattr(self.loss_weights, "neighborhood_consistency", 0.0)
        )
        if neighborhood_weight > 0:
            neighborhood_value, neighborhood_coverage = (
                reference_neighborhood_consistency_loss(
                    content,
                    present,
                    geom_ok,
                    q_reg,
                    batch["slide_group"],
                    self.reference_index,
                    temperature=float(
                        getattr(self.loss_weights, "neighborhood_temperature", 0.1)
                    ),
                    pool_size=int(
                        getattr(self.loss_weights, "neighborhood_pool_size", 4)
                    ),
                    min_q=float(
                        getattr(self.loss_weights, "neighborhood_min_q", 0.8)
                    ),
                )
            )
        else:
            neighborhood_value = content.sum() * 0.0
            neighborhood_coverage = content.new_zeros(())
        neighborhood_ramp = schedule["nuisance"] if train else 1.0

        pair_versa_value = pair_value.detach() * 0.0
        pair_nonversa_value = pair_value.detach()
        pair_versa_mass_fraction = pair_value.detach() * 0.0
        if "versa" in self.model.scanners:
            versa_index = self.model.scanners.index("versa")
            if pair_mode == "symmetric":
                versa_pairs = torch.zeros(
                    len(self.model.scanners), len(self.model.scanners),
                    dtype=torch.bool, device=clean_rgb.device,
                )
                versa_pairs[versa_index, :] = True
                versa_pairs[:, versa_index] = True
            else:
                versa_pairs = torch.zeros(
                    len(self.model.scanners), dtype=torch.bool, device=clean_rgb.device
                )
                versa_pairs[versa_index] = True
            versa_weights = used_pair_weights * versa_pairs.to(used_pair_weights.dtype)
            nonversa_weights = used_pair_weights * (~versa_pairs).to(used_pair_weights.dtype)
            pair_versa_value, _ = weighted_mean_with_coverage(
                pair_distances.detach(), versa_weights
            )
            pair_nonversa_value, _ = weighted_mean_with_coverage(
                pair_distances.detach(), nonversa_weights
            )
            total_mass = used_pair_weights.detach().sum()
            if bool(total_mass > 0):
                pair_versa_mass_fraction = versa_weights.detach().sum() / total_mass
        detail_weight = float(getattr(self.loss_weights, "detail", 0.0))
        if detail_weight > 0:
            detail_value, detail_coverage = structural_detail_loss(
                canonical,
                clean_rgb,
                clean_rgb[:, self.reference_index],
                supervision["standard"],
                q_reg,
                high_q=float(getattr(self.loss_weights, "detail_high_q", 0.8)),
            )
        else:
            detail_value = canonical.sum() * 0.0
            detail_coverage = canonical.new_zeros(())

        reference = clean_rgb[:, self.reference_index]
        expanded_reference = reference.unsqueeze(1).expand_as(canonical)
        identity_weights = torch.zeros_like(supervision["standard"])
        identity_weights[:, self.reference_index] = supervision["self"][:, self.reference_index]
        at2_identity_detail_value, at2_identity_detail_coverage = target_structural_detail_loss(
            canonical, expanded_reference, identity_weights
        )

        paired_detail_weights = supervision["standard"] * (
            q_reg >= float(getattr(self.loss_weights, "paired_target_detail_high_q", 0.9))
        ).to(supervision["standard"].dtype)
        paired_detail_weights[:, self.reference_index] = 0.0
        paired_target_detail_value, paired_target_detail_coverage = target_structural_detail_loss(
            canonical, expanded_reference, paired_detail_weights
        )
        nuclei_rgb_weight = float(getattr(self.loss_weights, "nuclei_rgb_detail", 0.0))
        if nuclei_rgb_weight > 0:
            if "nuclei_labels" not in batch:
                raise ValueError("nuclei_rgb_detail requires batch nuclei_labels")
            nuclei_rgb_value, nuclei_rgb_coverage = nuclei_masked_source_detail_loss(
                canonical,
                clean_rgb,
                batch["nuclei_labels"],
                supervision["self"],
                dilation=int(getattr(self.loss_weights, "nuclei_rgb_dilation", 3)),
                laplacian_weight=float(
                    getattr(self.loss_weights, "nuclei_rgb_laplacian", 0.25)
                ),
            )
        else:
            nuclei_rgb_value = canonical.sum() * 0.0
            nuclei_rgb_coverage = canonical.new_zeros(())
        nuclei_orientation_weight = float(
            getattr(self.loss_weights, "nuclei_orientation", 0.0)
        )
        if nuclei_orientation_weight > 0:
            if "nuclei_labels" not in batch:
                raise ValueError("nuclei_orientation requires batch nuclei_labels")
            nuclei_orientation_value, nuclei_orientation_coverage = (
                nuclei_masked_source_orientation_loss(
                    canonical,
                    clean_rgb,
                    batch["nuclei_labels"],
                    supervision["self"],
                    dilation=int(
                        getattr(self.loss_weights, "nuclei_orientation_dilation", 3)
                    ),
                    smoothing_kernel=int(
                        getattr(
                            self.loss_weights,
                            "nuclei_orientation_smoothing_kernel",
                            7,
                        )
                    ),
                    smoothing_sigma=float(
                        getattr(
                            self.loss_weights,
                            "nuclei_orientation_smoothing_sigma",
                            1.25,
                        )
                    ),
                    magnitude_floor=float(
                        getattr(
                            self.loss_weights,
                            "nuclei_orientation_magnitude_floor",
                            0.1,
                        )
                    ),
                )
            )
        else:
            nuclei_orientation_value = canonical.sum() * 0.0
            nuclei_orientation_coverage = canonical.new_zeros(())
        if nuclei_weight > 0:
            if "nuclei_labels" not in batch:
                raise ValueError("nuclei_reconstruction requires batch nuclei_labels")
            nuclei_labels = batch["nuclei_labels"]
            nuclei_source_weights = supervision["self"]
            nuclei_reference_weights = supervision["standard"] * (
                q_reg >= float(getattr(self.loss_weights, "nuclei_reference_high_q", 0.9))
            ).to(supervision["standard"].dtype)
            nuclei_reference_weights[:, self.reference_index] = 0.0
            (
                nuclei_value,
                nuclei_source_value,
                nuclei_reference_value,
                nuclei_coverage,
            ) = nuclei_reconstruction_loss(
                nuclei_logits,
                nuclei_labels,
                nuclei_labels[:, self.reference_index],
                nuclei_source_weights,
                nuclei_reference_weights,
                boundary_weight=float(getattr(self.loss_weights, "nuclei_boundary", 0.5)),
            )
        else:
            nuclei_value = canonical.sum() * 0.0
            nuclei_source_value = nuclei_value.detach()
            nuclei_reference_value = nuclei_value.detach()
            nuclei_coverage = canonical.new_zeros(())
        gradient_end = float(getattr(self.loss_weights, "gradient_end", 0.0))
        paired_detail_ramp = 1.0 if gradient_end <= 0 else min(
            max(float(schedule["gradient"]) / gradient_end, 0.0), 1.0
        )
        variance_value, variance_coverage = coordinate_variance_loss(
            content,
            supervision["standard"],
            batch["slide_group"],
            gamma=float(self.loss_weights.variance_gamma),
        )
        style_value, style_coverage = self._style_regression(
            clean_rgb, present, batch["slide_group"]
        )
        total = (
            float(self.loss_weights.standard) * standard_value
            + float(self.loss_weights.translate) * translate_value
            + schedule["pair"] * pair_value
            + neighborhood_weight * neighborhood_ramp * neighborhood_value
            + detail_weight * detail_value
            + float(getattr(self.loss_weights, "at2_identity_detail", 0.0))
            * at2_identity_detail_value
            + float(getattr(self.loss_weights, "paired_target_detail", 0.0))
            * paired_detail_ramp * paired_target_detail_value
            + nuclei_rgb_weight * schedule["cross"] * nuclei_rgb_value
            + nuclei_orientation_weight * schedule["cross"] * nuclei_orientation_value
            + nuclei_weight * schedule["cross"] * nuclei_value
            + float(self.loss_weights.variance) * variance_value
            + float(self.loss_weights.style) * style_value
        )

        anchored = (supervision["standard"] > 0).to(clean_rgb.dtype).mean()
        convergence = standard_convergence(canonical, present & geom_ok)
        reference_target = clean_rgb[:, self.reference_index]
        reference_prediction = canonical[:, self.reference_index]
        reference_ssim = ssim(reference_prediction, reference_target)
        focus_delta = (
            focus_score(canonical.flatten(0, 1))
            - focus_score(clean_rgb.flatten(0, 1))
        ).abs().mean()
        output = {
            "loss": total,
            "standard": standard_value,
            "standard_low": standard_low_value,
            "standard_high": standard_high_value,
            "translate": translate_value,
            "translate_self": self_value,
            "translate_cross": cross_value,
            "pair": pair_value,
            "pair_versa": pair_versa_value,
            "pair_nonversa": pair_nonversa_value,
            "pair_versa_weight_mass_fraction": pair_versa_mass_fraction,
            "neighborhood_consistency": neighborhood_value,
            "detail": detail_value,
            "at2_identity_detail": at2_identity_detail_value,
            "paired_target_detail": paired_target_detail_value,
            "nuclei_rgb_detail": nuclei_rgb_value,
            "nuclei_orientation": nuclei_orientation_value,
            "nuclei_reconstruction": nuclei_value,
            "nuclei_source": nuclei_source_value,
            "nuclei_reference": nuclei_reference_value,
            "variance": variance_value,
            "style": style_value,
            "coverage_standard": standard_coverage,
            "coverage_standard_high": standard_high_coverage,
            "coverage_self": self_coverage,
            "coverage_cross": cross_coverage,
            "coverage_pair": pair_coverage,
            "coverage_neighborhood_consistency": neighborhood_coverage,
            "coverage_detail": detail_coverage,
            "coverage_at2_identity_detail": at2_identity_detail_coverage,
            "coverage_paired_target_detail": paired_target_detail_coverage,
            "coverage_nuclei_rgb_detail": nuclei_rgb_coverage,
            "coverage_nuclei_orientation": nuclei_orientation_coverage,
            "coverage_nuclei": nuclei_coverage,
            "coverage_variance": variance_coverage,
            "coverage_style": style_coverage,
            "zero_coverage_standard": (standard_coverage == 0).to(clean_rgb.dtype),
            "zero_coverage_standard_high": (
                standard_high_coverage == 0
            ).to(clean_rgb.dtype),
            "zero_coverage_self": (self_coverage == 0).to(clean_rgb.dtype),
            "zero_coverage_cross": (cross_coverage == 0).to(clean_rgb.dtype),
            "zero_coverage_pair": (pair_coverage == 0).to(clean_rgb.dtype),
            "zero_coverage_neighborhood_consistency": (
                neighborhood_coverage == 0
            ).to(clean_rgb.dtype),
            "zero_coverage_detail": (detail_coverage == 0).to(clean_rgb.dtype),
            "zero_coverage_at2_identity_detail": (
                at2_identity_detail_coverage == 0
            ).to(clean_rgb.dtype),
            "zero_coverage_paired_target_detail": (
                paired_target_detail_coverage == 0
            ).to(clean_rgb.dtype),
            "zero_coverage_nuclei_rgb_detail": (
                nuclei_rgb_coverage == 0
            ).to(clean_rgb.dtype),
            "zero_coverage_nuclei_orientation": (
                nuclei_orientation_coverage == 0
            ).to(clean_rgb.dtype),
            "zero_coverage_nuclei": (nuclei_coverage == 0).to(clean_rgb.dtype),
            "zero_coverage_variance": (variance_coverage == 0).to(clean_rgb.dtype),
            "zero_coverage_style": (style_coverage == 0).to(clean_rgb.dtype),
            "frac_anchored": anchored,
            "canonical_ssim": convergence,
            "reference_ssim": reference_ssim,
            "focus_delta": focus_delta,
            "detail_reliability": (
                detail_reliability * (supervision["standard"] > 0)
            ).sum() / (supervision["standard"] > 0).sum().clamp_min(1),
            "detail_coherence": (
                detail_coherence * (supervision["standard"] > 0)
            ).sum() / (supervision["standard"] > 0).sum().clamp_min(1),
            "detail_energy_ratio": (
                detail_energy_ratio * (supervision["standard"] > 0)
            ).sum() / (supervision["standard"] > 0).sum().clamp_min(1),
            "pair_weight": clean_rgb.new_tensor(schedule["pair"]),
            "neighborhood_ramp": clean_rgb.new_tensor(neighborhood_ramp),
            "nuisance_strength": clean_rgb.new_tensor(schedule["nuisance"]),
            "canonical": canonical,
            "rendered": rendered,
            "render_targets": targets,
            "render_target_index": target_index,
        }
        for scanner_index, scanner in enumerate(self.model.scanners):
            usable = supervision["standard"][:, scanner_index] > 0
            denominator = usable.sum().clamp_min(1)
            output[f"detail_reliability_{scanner}"] = (
                detail_reliability[:, scanner_index] * usable
            ).sum() / denominator
            output[f"detail_coherence_{scanner}"] = (
                detail_coherence[:, scanner_index] * usable
            ).sum() / denominator
            output[f"detail_energy_ratio_{scanner}"] = (
                detail_energy_ratio[:, scanner_index] * usable
            ).sum() / denominator
            output[f"coverage_standard_high_{scanner}"] = (
                standard_high_weights[:, scanner_index] > 0
            ).to(clean_rgb.dtype).mean()
        return output

    def training_step(self, batch, batch_idx):
        output = self._shared_step(batch, train=True, batch_idx=batch_idx)
        batch_size = batch["rgb"].shape[0]
        metrics = {f"train/{key}": value for key, value in output.items()
                   if value.ndim == 0 and key != "loss"}
        self.log("train/loss", output["loss"], prog_bar=True, batch_size=batch_size)
        self.log_dict(metrics, batch_size=batch_size)
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        output = self._shared_step(batch, train=False, batch_idx=batch_idx)
        batch_size = batch["rgb"].shape[0]
        metrics = {f"val/{key}": value for key, value in output.items()
                   if value.ndim == 0 and key != "loss"}
        self.log("val/loss", output["loss"], prog_bar=True, batch_size=batch_size)
        self.log("val/l_standard", output["standard"], prog_bar=True, batch_size=batch_size)
        self.log_dict(metrics, batch_size=batch_size)

        if not getattr(self.trainer, "sanity_checking", False):
            target_count = int(self.cfg.image_log.n_ref_tuples)
            current = 0 if self._reference_batch is None else self._reference_batch["rgb"].shape[0]
            take = min(target_count - current, batch_size)
            if take > 0:
                addition = {
                    key: batch[key][:take].detach().cpu()
                    for key in ("rgb", "present", "geom_ok")
                }
                if self._reference_batch is None:
                    self._reference_batch = addition
                else:
                    self._reference_batch = {
                        key: torch.cat((self._reference_batch[key], addition[key]))
                        for key in addition
                    }

    def canonicalize(self, rgb: torch.Tensor) -> torch.Tensor:
        """Run the label-free inference path through M, C, G, and the AT2 prototype."""
        return self.model.canonicalize(rgb)

    def render(self, rgb: torch.Tensor, target_idx) -> torch.Tensor:
        """Render RGB with an observed scanner prototype."""
        return self.model.render(rgb, target_idx)

    def checkpoint_metadata(self) -> dict:
        """Return the strict semantic contract stored next to model weights."""
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "design_version": int(self.cfg.design_version),
            "scanners": list(self.model.scanners),
            "reference_scanner": self.cfg.reference_scanner,
            "input_mode": self.cfg.input.mode,
            "channel_order": list(self.model.M.channel_names),
            "stain_reference_hash": self.stain_reference_hash,
            "style_dim": int(self.cfg.model.style_dim),
            "alpha": 0.0,
            "phase": 1,
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["prenorm_metadata"] = self.checkpoint_metadata()

    def configure_optimizers(self):
        optimizer = self.cfg.optim
        return torch.optim.Adam(
            self.model.parameters(),
            lr=optimizer.lr,
            betas=tuple(optimizer.betas),
            weight_decay=optimizer.weight_decay,
        )
