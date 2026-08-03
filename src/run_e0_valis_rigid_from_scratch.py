"""Run targeted pairwise VALIS rigid registration from native WSIs.

Only scanner–slide cells selected by the frozen E0 failure hierarchy are eligible.
The exported registered image is used for geometry and QC; downstream RGB pixels
remain sourced from the scanner-native WSI.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from build_e0_rigid_native_fallback import MOVING_SCANNERS, rigid_branch
from run_e0_native_geometry_cohort import file_fingerprint, find_raw_path


ALGORITHM_VERSION = "valis_1.2.0_pairwise_native_rigid_reference_crop_v2"
EXPECTED_VALIS_VERSION = "1.2.0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-id", required=True)
    parser.add_argument("--scanner", required=True, choices=MOVING_SCANNERS)
    parser.add_argument(
        "--raw-root",
        default="/mnt/isilon/oldridge_lab/batch_effects/pan_normal/raw/raw_images",
    )
    parser.add_argument("--output", default="outputs/e0_valis_from_scratch")
    return parser.parse_args()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def stage_link(source: Path, destination: Path):
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"staging link points elsewhere: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"staging path already exists: {destination}")
    destination.symlink_to(source.resolve())


def main():
    import pyvips
    import torch
    from valis import registration, slide_io

    args = parse_args()
    valis_version = importlib.metadata.version("valis-wsi")
    if valis_version != EXPECTED_VALIS_VERSION:
        raise RuntimeError(
            f"expected VALIS {EXPECTED_VALIS_VERSION}, found {valis_version}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("targeted VALIS rigid rerun requires a visible CUDA device")
    slide_id = str(args.slide_id)
    scanner = str(args.scanner)
    raw_root = Path(args.raw_root)
    reference_path = find_raw_path(raw_root, "at2", slide_id)
    moving_path = find_raw_path(raw_root, scanner, slide_id)

    output = Path(args.output) / scanner / slide_id
    summary_path = output / "summary.json"
    registered_root = Path(args.output) / "registered_images"
    registered_path = (
        registered_root
        / rigid_branch(scanner)
        / scanner
        / f"{slide_id}.ome.tiff"
    )
    if summary_path.exists() and registered_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("complete") is True:
            print(json.dumps(summary, indent=2))
            return

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"incomplete output already exists; preserve it and choose a new --output: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    stage = output / "staging"
    stage.mkdir()
    reference_link = stage / f"{slide_id}_at2{reference_path.suffix.lower()}"
    moving_link = stage / f"{slide_id}_{scanner}{moving_path.suffix.lower()}"
    stage_link(reference_path, reference_link)
    stage_link(moving_path, moving_link)

    valis_output = output / "valis"
    registered_path.parent.mkdir(parents=True, exist_ok=True)
    registrar = None
    try:
        registrar = registration.Valis(
            str(stage),
            str(valis_output),
            img_list=[str(reference_link), str(moving_link)],
            series=0,
            image_type="brightfield",
            reference_img_f=str(reference_link),
            align_to_reference=True,
            non_rigid_registrar_cls=None,
            crop="reference",
        )
        # VALIS 1.2.0 only creates this dictionary when a non-rigid registrar is
        # enabled, but cleanup() dereferences it unconditionally. Keep the
        # rigid-only public configuration while supplying the missing cleanup key.
        if not hasattr(registrar, "non_rigid_reg_kwargs"):
            registrar.non_rigid_reg_kwargs = {
                registration.NON_RIGID_REG_CLASS_KEY: None
            }
        reader_dict = {
            str(reference_link): [slide_io.VipsSlideReader],
            str(moving_link): [slide_io.VipsSlideReader],
        }
        rigid_registrar, non_rigid_registrar, error_frame = registrar.register(
            reader_dict=reader_dict
        )
        if rigid_registrar is None or non_rigid_registrar is not None:
            raise RuntimeError("VALIS did not return the requested rigid-only registration")
        moving_slide = registrar.get_slide(str(moving_link))
        reference_slide = registrar.get_slide(str(reference_link))
        error_frame.to_csv(output / "valis_summary.csv", index=False)
        moving_slide.warp_and_save_slide(
            str(registered_path),
            level=0,
            non_rigid=False,
            crop="reference",
            interp_method="bicubic",
            pyramid=True,
        )
        registered = pyvips.Image.new_from_file(str(registered_path), access="sequential")
        reference = pyvips.Image.new_from_file(str(reference_path), access="sequential")
        dimensions_match = bool(
            registered.width == reference.width
            and registered.height == reference.height
        )
        if not dimensions_match:
            raise ValueError(
                "reference-crop export dimensions differ: "
                f"registered={(registered.width, registered.height)} "
                f"reference={(reference.width, reference.height)}"
            )
        summary = {
            "analysis": "e0_valis_rigid_from_scratch",
            "algorithm_version": ALGORITHM_VERSION,
            "slide_id": slide_id,
            "scanner": scanner,
            "complete": True,
            "pixel_source": "native_wsi",
            "registered_pixels_used_for": "geometry and QC only",
            "transform_family": "VALIS default SimilarityTransform; rigid only",
            "reference_crop": True,
            "non_rigid": False,
            "valis_1_2_rigid_only_cleanup_workaround": True,
            "slide_reader": "valis.slide_io.VipsSlideReader",
            "reference_native_path": str(reference_path.resolve()),
            "moving_native_path": str(moving_path.resolve()),
            "registered_path": str(registered_path.resolve()),
            "reference_fingerprint": file_fingerprint(reference_path),
            "moving_fingerprint": file_fingerprint(moving_path),
            "registered_fingerprint": file_fingerprint(registered_path),
            "reference_dimensions_wh": [int(reference.width), int(reference.height)],
            "registered_dimensions_wh": [int(registered.width), int(registered.height)],
            "moving_processed_shape_rc": moving_slide.processed_img_shape_rc,
            "moving_registered_shape_rc": moving_slide.reg_img_shape_rc,
            "moving_rigid_inverse_matrix": moving_slide.M,
            "reference_processed_shape_rc": reference_slide.processed_img_shape_rc,
            "reference_registered_shape_rc": reference_slide.reg_img_shape_rc,
            "reference_rigid_inverse_matrix": reference_slide.M,
            "valis_version": valis_version,
            "max_image_dim_px": int(registrar.max_image_dim_px),
            "max_processed_image_dim_px": int(
                registrar.max_processed_image_dim_px
            ),
            "thumbnail_size": int(registrar.thumbnail_size),
            "torch_version": str(torch.__version__),
            "cuda_available": True,
            "cuda_device": torch.cuda.get_device_name(0),
            "pyvips_version": str(pyvips.__version__),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, default=json_value) + "\n"
        )
        print(json.dumps(summary, indent=2, default=json_value))
    finally:
        registration.kill_jvm()


if __name__ == "__main__":
    main()
