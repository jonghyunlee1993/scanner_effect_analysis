"""Recover primary-route geometry and extract native anti-aliased pilot patches.

The historical primary registered TIFFs do not retain their VALIS transform
objects.  Because each registered scanner image is a resampled copy of its own
native WSI, this pilot recovers a scanner-native -> primary-output similarity
transform from same-scanner thumbnails.  A low-pass integer residual is then
estimated at each frozen manifest location.  Pixel values used for the AA patch
come only from the native WSI; the historical output supplies geometry/QC only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCANNERS = ("gt450", "versa")
BANDS = {"low_mid": (0.10, 0.30), "mid": (0.30, 0.60), "high": (0.60, 0.90)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-id", default="12.5_11")
    parser.add_argument("--scanners", nargs="+", default=list(SCANNERS))
    parser.add_argument(
        "--raw-root",
        default="/mnt/isilon/oldridge_lab/batch_effects/pan_normal/raw/raw_images",
    )
    parser.add_argument(
        "--registry",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/registered_ref_at2_all/"
            "registered_images/qc_registered_eval/registered_pair_summary.csv"
        ),
    )
    parser.add_argument(
        "--manifest", default="outputs/e0_feature_manifest_109/feature_manifest.csv"
    )
    parser.add_argument("--output", default="outputs/e0_primary_transform_recovery_pilot")
    parser.add_argument("--thumbnail-size", type=int, default=4096)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--search-margin", type=int, default=64)
    parser.add_argument("--max-patches", type=int, default=40)
    parser.add_argument("--minimum-sift-inliers", type=int, default=80)
    return parser.parse_args()


def find_raw_path(root: Path, scanner: str, slide_id: str) -> Path:
    matches = sorted(
        path
        for path in (root / scanner).glob(f"{slide_id}.*")
        if path.suffix.lower() in {".svs", ".ndpi", ".qptiff", ".tif", ".tiff"}
    )
    if len(matches) != 1:
        raise ValueError(f"expected one native slide for {scanner}/{slide_id}: {matches}")
    return matches[0].resolve()


def ensure_rgb(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    if value.shape[2] > 3:
        value = value[..., :3]
    return np.ascontiguousarray(value.astype(np.uint8))


def thumbnail(image, maximum: int):
    scale = min(1.0, maximum / max(image.width, image.height))
    if scale < 1.0:
        thumb = image.resize(scale, kernel="lanczos3")
    else:
        thumb = image
    return ensure_rgb(np.asarray(thumb)), scale


def gray_features(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(gray)


def scale_thumbnail_affine_to_full(
    thumbnail_affine: np.ndarray,
    native_full_wh,
    native_thumbnail_wh,
    registered_full_wh,
    registered_thumbnail_wh,
):
    native_full_wh = np.asarray(native_full_wh, dtype=float)
    native_thumbnail_wh = np.asarray(native_thumbnail_wh, dtype=float)
    registered_full_wh = np.asarray(registered_full_wh, dtype=float)
    registered_thumbnail_wh = np.asarray(registered_thumbnail_wh, dtype=float)
    source_to_thumb = np.diag(
        [
            native_thumbnail_wh[0] / native_full_wh[0],
            native_thumbnail_wh[1] / native_full_wh[1],
            1.0,
        ]
    )
    registered_to_thumb = np.diag(
        [
            registered_thumbnail_wh[0] / registered_full_wh[0],
            registered_thumbnail_wh[1] / registered_full_wh[1],
            1.0,
        ]
    )
    return np.linalg.inv(registered_to_thumb) @ thumbnail_affine @ source_to_thumb


def recover_similarity(native_rgb, registered_rgb, minimum_inliers):
    cv2.setRNGSeed(20260802)
    sift = cv2.SIFT_create(nfeatures=12_000, contrastThreshold=0.01, edgeThreshold=15)
    key_native, descriptor_native = sift.detectAndCompute(gray_features(native_rgb), None)
    key_registered, descriptor_registered = sift.detectAndCompute(
        gray_features(registered_rgb), None
    )
    if descriptor_native is None or descriptor_registered is None:
        raise RuntimeError("SIFT found no descriptors")
    # Exact 20k×20k brute-force matching dominates this otherwise lightweight
    # recovery.  KD-tree FLANN is deterministic enough for candidate generation;
    # the final geometry still has to pass the same strict RANSAC inlier gate.
    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=8),
        dict(checks=96),
    )
    pairs = matcher.knnMatch(descriptor_native, descriptor_registered, k=2)
    accepted = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    if len(accepted) < minimum_inliers:
        raise RuntimeError(f"only {len(accepted)} ratio-test SIFT matches")
    source = np.float32([key_native[match.queryIdx].pt for match in accepted])
    destination = np.float32([key_registered[match.trainIdx].pt for match in accepted])
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        source,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=20_000,
        confidence=0.999,
        refineIters=50,
    )
    if affine is None:
        raise RuntimeError("RANSAC similarity estimation failed")
    inlier = inlier_mask.ravel().astype(bool)
    if int(inlier.sum()) < minimum_inliers:
        raise RuntimeError(f"only {int(inlier.sum())} RANSAC inliers")
    homogeneous = np.eye(3, dtype=float)
    homogeneous[:2] = affine
    predicted = cv2.transform(source[inlier, None, :], affine)[:, 0, :]
    error = np.linalg.norm(predicted - destination[inlier], axis=1)
    metrics = {
        "sift_keypoints_native": len(key_native),
        "sift_keypoints_registered": len(key_registered),
        "sift_ratio_matches": len(accepted),
        "sift_inliers": int(inlier.sum()),
        "sift_inlier_fraction": float(inlier.mean()),
        "thumbnail_reprojection_median_px": float(np.median(error)),
        "thumbnail_reprojection_q95_px": float(np.quantile(error, 0.95)),
    }
    return homogeneous, metrics, source[inlier], destination[inlier]


def extract_rgb(image, x: int, y: int, size: int):
    return ensure_rgb(np.asarray(image.extract_area(int(x), int(y), int(size), int(size))))


def lowpass_gray(rgb, sigma=3.0):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.GaussianBlur(gray, (0, 0), sigma)


def residual_integer_offset(reconstructed_ext, registered_patch, margin: int):
    score = cv2.matchTemplate(
        lowpass_gray(reconstructed_ext),
        lowpass_gray(registered_patch),
        cv2.TM_CCOEFF_NORMED,
    )
    score = np.nan_to_num(score, nan=-1.0, posinf=-1.0, neginf=-1.0)
    row, column = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(row - margin), int(column - margin), float(score[row, column])


def frequency_geometry(size: int, mpp: float, bins=72):
    one = np.fft.fftfreq(size, d=mpp)
    radius = np.sqrt(one[:, None] ** 2 + one[None, :] ** 2)
    nyquist = 1.0 / (2.0 * mpp)
    edges = np.linspace(0.0, nyquist, bins + 1)
    index = np.digitize(radius.ravel(), edges) - 1
    valid = (index >= 0) & (index < bins)
    counts = np.bincount(index[valid], minlength=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return index, valid, counts, centers


def radial_power(rgb: np.ndarray, geometry, window):
    index, valid, counts, _ = geometry
    optical_density = -np.log((rgb.astype(np.float64) + 1.0) / 256.0).mean(axis=2)
    optical_density -= optical_density.mean()
    power = np.abs(np.fft.fft2(optical_density * window)) ** 2
    return np.bincount(
        index[valid], weights=power.ravel()[valid], minlength=len(counts)
    ) / np.maximum(counts, 1)


def geometric_band(curve, frequency, bounds):
    selected = (frequency >= bounds[0]) & (frequency < bounds[1])
    return float(np.exp(np.mean(np.log(np.maximum(curve[selected], 1e-20)))))


def make_warped_views(native_image, native_to_registered, registered_shape_rc):
    from valis import warp_tools

    inverse = np.linalg.inv(native_to_registered)
    native_shape_rc = np.array([native_image.height, native_image.width])
    original = warp_tools.warp_img(
        native_image,
        M=inverse,
        transformation_src_shape_rc=native_shape_rc,
        transformation_dst_shape_rc=registered_shape_rc,
        out_shape_rc=registered_shape_rc,
        interp_method="bicubic",
        bg_color=[0] * native_image.bands,
    )
    singular = np.linalg.svd(native_to_registered[:2, :2], compute_uv=False)
    pre_scale = float(singular.min())
    antialiased_native = native_image.resize(pre_scale, kernel="lanczos3")
    antialiased = warp_tools.warp_img(
        antialiased_native,
        M=inverse,
        transformation_src_shape_rc=native_shape_rc,
        transformation_dst_shape_rc=registered_shape_rc,
        out_shape_rc=registered_shape_rc,
        interp_method="bicubic",
        bg_color=[0] * antialiased_native.bands,
    )
    return original, antialiased, pre_scale


def render_thumbnail_overlay(native_rgb, registered_rgb, affine, output: Path, scanner: str):
    warped = cv2.warpAffine(
        native_rgb,
        affine[:2].astype(np.float32),
        (registered_rgb.shape[1], registered_rgb.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )
    overlay = np.zeros_like(registered_rgb)
    overlay[..., 0] = registered_rgb[..., 0]
    overlay[..., 1] = warped[..., 1]
    overlay[..., 2] = registered_rgb[..., 2]
    plt.imsave(output / f"thumbnail_overlay_{scanner}.png", overlay)


def main():
    import pyvips

    args = parse_args()
    output = Path(args.output) / args.slide_id
    output.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(args.registry, dtype={"slide_id": str}).set_index("slide_id")
    if args.slide_id not in registry.index:
        raise KeyError(args.slide_id)
    manifest = pd.read_csv(args.manifest, dtype={"slide_id": str})
    locations = manifest[manifest["slide_id"].eq(args.slide_id)].sort_values("location_id")
    if len(locations) != 100:
        raise ValueError(f"expected 100 frozen locations, got {len(locations)}")

    patch_rows = []
    spectral_rows = []
    scanner_summaries = []
    target_mpp = 0.5052
    size = args.patch_size
    geometry = frequency_geometry(size, target_mpp)
    frequency = geometry[3]
    one = np.hanning(size)
    window = np.outer(one, one)

    for scanner in args.scanners:
        native_path = find_raw_path(Path(args.raw_root), scanner, args.slide_id)
        registered_path = Path(str(registry.loc[args.slide_id, f"{scanner}_path"]))
        native_image = pyvips.Image.new_from_file(str(native_path), access="random")
        registered_image = pyvips.Image.new_from_file(str(registered_path), access="random")
        native_thumb, _ = thumbnail(native_image, args.thumbnail_size)
        registered_thumb, _ = thumbnail(registered_image, args.thumbnail_size)
        thumbnail_affine, sift_metrics, source_inliers, destination_inliers = recover_similarity(
            native_thumb, registered_thumb, args.minimum_sift_inliers
        )
        full_affine = scale_thumbnail_affine_to_full(
            thumbnail_affine,
            (native_image.width, native_image.height),
            (native_thumb.shape[1], native_thumb.shape[0]),
            (registered_image.width, registered_image.height),
            (registered_thumb.shape[1], registered_thumb.shape[0]),
        )
        original_view, aa_view, pre_scale = make_warped_views(
            native_image,
            full_affine,
            np.array([registered_image.height, registered_image.width]),
        )
        render_thumbnail_overlay(
            native_thumb, registered_thumb, thumbnail_affine, output, scanner
        )

        accepted = 0
        original_power = []
        aa_power = []
        for row in locations.itertuples(index=False):
            x = int(row.x + getattr(row, f"{scanner}_dx"))
            y = int(row.y + getattr(row, f"{scanner}_dy"))
            margin = args.search_margin
            if (
                x < 0
                or y < 0
                or x + size > registered_image.width
                or y + size > registered_image.height
                or x - margin < 0
                or y - margin < 0
                or x + size + margin > original_view.width
                or y + size + margin > original_view.height
            ):
                continue
            registered_patch = extract_rgb(registered_image, x, y, size)
            reconstructed_ext = extract_rgb(
                original_view, x - margin, y - margin, size + 2 * margin
            )
            dy, dx, ncc = residual_integer_offset(
                reconstructed_ext, registered_patch, margin
            )
            reconstruction = reconstructed_ext[
                margin + dy : margin + dy + size,
                margin + dx : margin + dx + size,
            ]
            aa_patch = extract_rgb(aa_view, x + dx, y + dy, size)
            mae = float(np.mean(np.abs(reconstruction.astype(float) - registered_patch)))
            patch_rows.append(
                {
                    "slide_id": args.slide_id,
                    "scanner": scanner,
                    "location_id": int(row.location_id),
                    "registered_x": x,
                    "registered_y": y,
                    "native_recovery_dy": dy,
                    "native_recovery_dx": dx,
                    "native_recovery_ncc": ncc,
                    "reconstruction_rgb_mae": mae,
                    "search_boundary": bool(abs(dy) >= margin or abs(dx) >= margin),
                }
            )
            if ncc < 0.75 or abs(dy) >= margin or abs(dx) >= margin:
                continue
            original_power.append(radial_power(reconstruction, geometry, window))
            aa_power.append(radial_power(aa_patch, geometry, window))
            accepted += 1
            if accepted >= args.max_patches:
                break
        if accepted < min(20, args.max_patches):
            raise RuntimeError(f"{scanner}: only {accepted} recovered tissue patches")
        mean_original = np.mean(original_power, axis=0)
        mean_aa = np.mean(aa_power, axis=0)
        absolute_amplitude = np.sqrt(
            np.maximum(mean_aa, 1e-20) / np.maximum(mean_original, 1e-20)
        )
        anchor = (frequency >= 0.03) & (frequency <= 0.10)
        anchor_scale = float(
            np.exp(np.mean(np.log(np.maximum(absolute_amplitude[anchor], 1e-20))))
        )
        normalized = absolute_amplitude / anchor_scale
        for index, value in enumerate(frequency):
            spectral_rows.append(
                {
                    "slide_id": args.slide_id,
                    "scanner": scanner,
                    "frequency_cyc_per_um": float(value),
                    "original_power": float(mean_original[index]),
                    "antialiased_power": float(mean_aa[index]),
                    "aa_to_original_amplitude": float(absolute_amplitude[index]),
                    "anchor_normalized_aa_to_original": float(normalized[index]),
                }
            )
        scanner_patch_rows = pd.DataFrame(patch_rows)
        scanner_patch_rows = scanner_patch_rows[scanner_patch_rows["scanner"].eq(scanner)]
        singular = np.linalg.svd(full_affine[:2, :2], compute_uv=False)
        scanner_summaries.append(
            {
                "slide_id": args.slide_id,
                "scanner": scanner,
                **sift_metrics,
                "native_width": native_image.width,
                "native_height": native_image.height,
                "registered_width": registered_image.width,
                "registered_height": registered_image.height,
                "native_px_per_registered_px": float(1.0 / np.sqrt(np.prod(singular))),
                "affine_anisotropy_ratio": float(singular.max() / singular.min()),
                "explicit_aa_pre_scale": pre_scale,
                "patches_measured": int(len(scanner_patch_rows)),
                "patches_spectral": accepted,
                "residual_abs_shift_median": float(
                    np.median(
                        np.hypot(
                            scanner_patch_rows["native_recovery_dy"],
                            scanner_patch_rows["native_recovery_dx"],
                        )
                    )
                ),
                "residual_abs_shift_q95": float(
                    np.quantile(
                        np.hypot(
                            scanner_patch_rows["native_recovery_dy"],
                            scanner_patch_rows["native_recovery_dx"],
                        ),
                        0.95,
                    )
                ),
                "residual_ncc_median": float(scanner_patch_rows["native_recovery_ncc"].median()),
                "residual_ncc_min": float(scanner_patch_rows["native_recovery_ncc"].min()),
                "reconstruction_rgb_mae_median": float(
                    scanner_patch_rows["reconstruction_rgb_mae"].median()
                ),
                **{
                    f"absolute_aa_to_original_{name}": geometric_band(
                        absolute_amplitude, frequency, bounds
                    )
                    for name, bounds in BANDS.items()
                },
                **{
                    f"normalized_aa_to_original_{name}": geometric_band(
                        normalized, frequency, bounds
                    )
                    for name, bounds in BANDS.items()
                },
            }
        )

    patch_frame = pd.DataFrame(patch_rows)
    spectral_frame = pd.DataFrame(spectral_rows)
    summary_frame = pd.DataFrame(scanner_summaries)
    patch_frame.to_csv(output / "patch_recovery_metrics.csv", index=False)
    spectral_frame.to_csv(output / "spectral_sensitivity.csv", index=False)
    summary_frame.to_csv(output / "scanner_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for scanner in args.scanners:
        selected = spectral_frame[spectral_frame["scanner"].eq(scanner)]
        axes[0].plot(
            selected["frequency_cyc_per_um"],
            selected["aa_to_original_amplitude"],
            label=scanner.upper(),
        )
        axes[1].plot(
            selected["frequency_cyc_per_um"],
            selected["anchor_normalized_aa_to_original"],
            label=scanner.upper(),
        )
    for axis in axes:
        axis.axvspan(0.60, 0.90, color="#0F766E", alpha=0.10)
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_xlabel("Frequency (cycles/µm)")
        axis.set_xlim(0, 0.99)
        axis.legend()
    axes[0].set_ylabel("AA / historical amplitude")
    axes[0].set_title("A  Absolute real-tissue sensitivity")
    axes[1].set_ylabel("Anchor-normalized AA / historical")
    axes[1].set_title("B  ERT-shape sensitivity")
    figure.tight_layout()
    figure.savefig(output / "figure_native_aa_patch_pilot.png", dpi=220)
    figure.savefig(output / "figure_native_aa_patch_pilot.pdf")
    plt.close(figure)

    summary = {
        "analysis": "e0_primary_same_scanner_transform_recovery_pilot",
        "slide_id": args.slide_id,
        "scanners": list(args.scanners),
        "geometry_source": "same-scanner native/current SIFT similarity + per-location low-pass integer residual",
        "pixel_source": "native WSI only",
        "historical_registered_pixels_used_for": "geometry recovery and reconstruction QC only",
        "explicit_aa": "Lanczos3 reduction at min affine singular value then residual bicubic affine",
        "scanner_summary": scanner_summaries,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
