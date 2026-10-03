#!/usr/bin/env python3
"""Build a paired Pan Normal patch cache from phase-corrected coordinates.

The phase-corrected coordinate is the default.  A wide low-pass NCC correction
is used only when an independent optical-density gradient matcher verifies a
residual displacement and agrees with the proposed correction.  Every output
row is a six-scanner tuple with a common source coordinate index.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import h5py
import numpy as np
import openslide
from PIL import Image, ImageDraw, ImageFont


SCANNERS = ("at2", "versa", "akoya", "gt450", "s360", "s60")
NON_REFERENCE_SCANNERS = SCANNERS[1:]
REFERENCE_SCANNER = "at2"
PATCH_SIZE = 256
SEARCH_MARGIN = 96
SPATIAL_BINS = 10

PHASE_ROOT = Path(
    "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/"
    "registered_ref_at2_all/registered_curated_features_phase_corrected_filtered"
)
IMAGE_ROOT = Path(
    "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/"
    "registered_ref_at2_all/registered_images"
)
DEFAULT_OUTPUT_ROOT = Path("data/preprocessing_patch_extraction_matching_and_QC")

TISSUE_OD_THRESHOLD = 0.08
TISSUE_FRACTION_MIN = 0.90
TISSUE_CLOSE_KERNEL = 5
GRAD_SCORE_MIN = 0.50
GRAD_GAP_MIN = 0.02
PHASE_KEEP_TOLERANCE_PX = 2.0
MATCHER_AGREEMENT_TOLERANCE_PX = 2.0
GLOBAL_NCC_MIN = 0.50

MODE_REFERENCE = 0
MODE_PHASE_KEPT = 1
MODE_GLOBAL_RESCUE = 2
MODE_NAMES = {
    MODE_REFERENCE: "reference",
    MODE_PHASE_KEPT: "phase_kept",
    MODE_GLOBAL_RESCUE: "global_rescue",
}
EXAMPLE_CATEGORIES = (
    "phase_verified",
    "slight_rescue_2to5px",
    "large_rescue_over5px",
    "rejected",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def feature_dir(scanner: str, ignored: bool = False) -> Path:
    base = PHASE_ROOT / scanner
    if ignored:
        base = base / "ignore"
    return base / "20x_256px_0px_overlap" / "features_uni_v2"


def h5_path(scanner: str, slide_id: str) -> Path:
    return feature_dir(scanner) / f"{slide_id}.h5"


def image_path(scanner: str, slide_id: str) -> Path:
    return IMAGE_ROOT / scanner / f"{slide_id}.tiff"


def source_ids(scanner: str, ignored: bool = False) -> set[str]:
    directory = feature_dir(scanner, ignored=ignored)
    if not directory.is_dir():
        return set()
    return {path.stem for path in directory.glob("*.h5")}


def source_hash(slide_ids: Iterable[str]) -> str:
    content = "\n".join(slide_ids) + "\n"
    return hashlib.sha256(content.encode()).hexdigest()


def build_config(slide_ids: list[str], ignored_ids: list[str]) -> dict[str, Any]:
    return {
        "created_utc": utc_now(),
        "pipeline": "preprocessing_patch_extraction_matching_and_QC",
        "pipeline_version": 1,
        "phase_root": str(PHASE_ROOT),
        "image_root": str(IMAGE_ROOT),
        "scanners": list(SCANNERS),
        "reference_scanner": REFERENCE_SCANNER,
        "cohort_rule": "intersection of six scanner direct feature directories",
        "cohort_slide_count": len(slide_ids),
        "cohort_sha256": source_hash(slide_ids),
        "excluded_rule": "slides under scanner ignore directories",
        "excluded_slide_count": len(ignored_ids),
        "excluded_slide_ids": ignored_ids,
        "selection": {
            "maximum_pairs_per_slide": 1000,
            "sampling": "deterministic 10x10 spatial-bin round robin",
            "duplicates": False,
        },
        "patch": {"level": 0, "size_px": PATCH_SIZE},
        "tissue_gate": {
            "selection_scanner": REFERENCE_SCANNER,
            "reason": "avoid scanner-dependent cohort selection",
            "mean_optical_density_threshold": TISSUE_OD_THRESHOLD,
            "morphological_close_kernel_px": TISSUE_CLOSE_KERNEL,
            "minimum_fraction": TISSUE_FRACTION_MIN,
        },
        "matching_gate": {
            "search_margin_px": SEARCH_MARGIN,
            "phase_keep_residual_tolerance_px": PHASE_KEEP_TOLERANCE_PX,
            "od_gradient_score_min": GRAD_SCORE_MIN,
            "od_gradient_nonprimary_peak_gap_min": GRAD_GAP_MIN,
            "boundary_peaks_rejected": True,
            "global_lowpass_ncc_min": GLOBAL_NCC_MIN,
            "global_vs_gradient_agreement_tolerance_px": MATCHER_AGREEMENT_TOLERANCE_PX,
            "historical_slide_prior": False,
        },
    }


def method_document(
    config: dict[str, Any], summary: dict[str, Any] | None = None
) -> str:
    ignored = ", ".join(f"`{item}`" for item in config["excluded_slide_ids"])
    lines = [
        "# preprocessing_patch_extraction_matching_and_QC",
        "",
        "## 목적",
        "",
        "Pan Normal의 phase-corrected 좌표를 기본값으로 사용해, 분석에 바로 사용할 수 있는 "
        "6-scanner paired RGB patch cache를 만든다. Feature 값은 입력으로 사용하지 않는다.",
        "",
        "## Cohort",
        "",
        f"- 포함: 여섯 scanner의 일반 feature 경로에 공통으로 존재하는 {config['cohort_slide_count']}개 slide.",
        f"- 제외: 각 scanner의 `ignore` 경로에 있는 {config['excluded_slide_count']}개 slide: {ignored}.",
        f"- Cohort SHA-256: `{config['cohort_sha256']}`.",
        "- `12.5_9`를 포함한다. 슬라이드당 1,000개는 최소치가 아니라 상한이다.",
        "",
        "## Patch 선택",
        "",
        "1. AT2 phase-coordinate를 10x10 공간 bin으로 나누고, slide ID로 고정된 seed를 사용해 "
        "각 bin 내부를 섞은 뒤 round-robin으로 순회한다.",
        "2. AT2 256x256 level-0 patch의 mean optical-density가 0.08보다 큰 pixel mask를 만들고 "
        "5x5 closing 후 tissue fraction이 0.90 이상인 후보만 matching으로 보낸다.",
        "3. Scanner별 색/밝기 차이가 표본 선택을 좌우하지 않도록 tissue gate는 AT2 reference에만 적용한다. "
        "모든 scanner의 최종 tissue fraction은 별도 지표로 저장한다.",
        "4. 중복 없이 QC를 통과한 pair를 slide별 최대 1,000개 저장한다. 후보 또는 통과 pair가 "
        "1,000개보다 적으면 가능한 pair를 모두 저장한다.",
        "",
        "## Matching과 QC",
        "",
        "- Phase 좌표가 기본값이다.",
        "- 각 non-AT2 scanner에서 phase 좌표 주위 +/-96 px를 검색한다.",
        "- 독립 verifier는 optical-density gradient magnitude NCC이다. score >=0.50, 비주 peak와의 "
        "gap >=0.02, search-boundary가 아닌 peak만 신뢰한다.",
        "- verifier residual이 2 px 이내면 phase 좌표를 그대로 유지한다.",
        "- residual이 2 px보다 크면 low-pass luminance global NCC를 계산한다. NCC >=0.50이고, "
        "boundary가 아니며, gradient shift와 2 px 이내로 일치할 때만 보정 좌표를 채택한다.",
        "- confidence 부족, peak ambiguity, boundary, 두 matcher 불일치, padding 후보는 tuple 전체를 버린다.",
        "- 과거 E0의 slide-prior distance rejection은 사용하지 않는다.",
        "- `total_rejection_counts`는 scanner 순서대로 검사하다가 만난 첫 탈락 원인을 센 값이다. "
        "따라서 scanner별 잠재 실패율의 완전한 분해로 해석하지 않는다.",
        "",
        "## Cache schema",
        "",
        "각 `cache/<slide_id>.h5`는 다음을 포함한다.",
        "",
        "- `images`: `(N, 6, 256, 256, 3)` RGB uint8, scanner 순서 " + ", ".join(SCANNERS) + ".",
        "- `source_index`, `phase_coords_xy`, `final_coords_xy`, `applied_shift_xy`.",
        "- `alignment/mode`, gradient/global matcher score, peak gap, shift와 agreement error.",
        "- `quality/tissue_fraction`, `quality/laplacian_variance`.",
        "- `qc_examples`: phase success, slight rescue, large rescue, rejection 대표 예시.",
        "",
        "## Output 구조",
        "",
        "```text",
        "data/preprocessing_patch_extraction_matching_and_QC/",
        "  preprocessing_patch_extraction_matching_and_QC.md",
        "  config/config.json",
        "  manifests/cohort.csv, excluded_slides.csv, submissions.json",
        "  cache/<slide_id>.h5",
        "  qc/slides/<slide_id>.png",
        "  qc/results/<slide_id>.json",
        "  qc/condition_galleries/<condition>.png",
        "  qc/slide_summary.csv, run_summary.json",
        "  logs/",
        "```",
        "",
        "## 재현",
        "",
        "```bash",
        "python src/preprocessing_patch_extraction_matching_and_QC.py prepare",
        "sbatch --array=0-102%103 --export=ALL,JOB_CONDA_PREFIX=/home/leej70/miniconda3/envs/cpath,JOB_WORKDIR=$PWD,JOB_OUTPUT_ROOT=$PWD/data/preprocessing_patch_extraction_matching_and_QC scripts/preprocessing/preprocessing_patch_extraction_matching_and_QC.sbatch",
        "python src/preprocessing_patch_extraction_matching_and_QC.py aggregate",
        "```",
    ]
    if summary is None:
        lines.extend(
            [
                "",
                "## 실행 상태",
                "",
                "아직 최종 aggregation 전이다. 완료 후 실제 slide 수, pair 수, rescue 및 rejection 통계로 갱신된다.",
            ]
        )
    else:
        stats = summary.get("pair_count", {})
        below_cap = ", ".join(
            f"`{row['slide_id']}` ({row['accepted_pair_count']})"
            for row in summary.get("below_cap_slides", [])
        )
        submissions = ", ".join(
            f"{row['kind']} `{row['job_id']}`"
            for row in summary.get("submissions", [])
        )
        lines.extend(
            [
                "",
                "## 최종 실행 결과",
                "",
                f"- 결과가 기록된 slide: {summary['result_slide_count']} / {summary['expected_slide_count']}.",
                f"- cache가 생성된 slide: {summary['cached_slide_count']}.",
                f"- 총 paired patches: {summary['total_pair_count']:,}.",
                f"- slide별 pair 수: min {stats.get('min', 0)}, median {stats.get('median', 0)}, "
                f"max {stats.get('max', 0)}.",
                f"- 1,000-pair 상한 도달 slide: {summary['cap_reached_slide_count']}.",
                f"- Source 후보 {summary['source_candidate_count']:,}개 중 {summary['processed_candidate_count']:,}개를 "
                f"실제 검사했으며, 검사 후보 대비 tuple 통과율은 {summary['acceptance_fraction_among_processed']:.1%}이다. "
                "상한 도달 시 조기 종료하므로 이는 전체 source 후보에 대한 비율이 아니다.",
                f"- 통과한 non-AT2 scanner-pair 중 phase 좌표 유지 {summary['phase_kept_scanner_pair_count']:,} "
                f"({summary['phase_kept_scanner_pair_fraction']:.1%}), global rescue "
                f"{summary['global_rescue_scanner_pair_count']:,} ({summary['global_rescue_scanner_pair_fraction']:.1%}).",
                f"- 1,000개 미만인 {len(summary.get('below_cap_slides', []))}개 slide: {below_cap}.",
                f"- Cache apparent size: {summary['cache_apparent_bytes'] / (1024 ** 3):.1f} GiB.",
                f"- SLURM 기록: {submissions}.",
                f"- 최종 aggregation UTC: {summary['aggregated_utc']}.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def prepare(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    direct_sets = {scanner: source_ids(scanner) for scanner in SCANNERS}
    ignored_sets = {scanner: source_ids(scanner, ignored=True) for scanner in SCANNERS}

    first_direct = direct_sets[SCANNERS[0]]
    first_ignored = ignored_sets[SCANNERS[0]]
    direct_mismatch = {
        scanner: sorted(values.symmetric_difference(first_direct))
        for scanner, values in direct_sets.items()
        if values != first_direct
    }
    ignored_mismatch = {
        scanner: sorted(values.symmetric_difference(first_ignored))
        for scanner, values in ignored_sets.items()
        if values != first_ignored
    }
    if direct_mismatch:
        raise RuntimeError(f"direct cohort differs across scanners: {direct_mismatch}")
    if ignored_mismatch:
        raise RuntimeError(f"ignore cohort differs across scanners: {ignored_mismatch}")

    slide_ids = sorted(first_direct)
    ignored_ids = sorted(first_ignored)
    if set(slide_ids) & set(ignored_ids):
        raise RuntimeError("direct and ignore cohorts overlap")
    if args.expected_slides is not None and len(slide_ids) != args.expected_slides:
        raise RuntimeError(
            f"expected {args.expected_slides} direct slides, found {len(slide_ids)}"
        )

    for relative in (
        "config",
        "manifests",
        "cache",
        "qc/slides",
        "qc/results",
        "qc/condition_galleries",
        "logs",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)

    cohort_rows: list[dict[str, Any]] = []
    for array_index, slide_id in enumerate(slide_ids):
        counts = []
        for scanner in SCANNERS:
            source = h5_path(scanner, slide_id)
            image = image_path(scanner, slide_id)
            if not source.is_file() or not image.is_file():
                raise FileNotFoundError(f"missing input for {scanner}/{slide_id}")
            with h5py.File(source, "r") as handle:
                if "coords" not in handle:
                    raise KeyError(f"{source}: coords dataset missing")
                counts.append(int(len(handle["coords"])))
        if len(set(counts)) != 1:
            raise RuntimeError(f"coordinate count mismatch for {slide_id}: {counts}")
        cohort_rows.append(
            {
                "array_index": array_index,
                "slide_id": slide_id,
                "source_candidate_count": counts[0],
            }
        )
    excluded_rows = [
        {"slide_id": slide_id, "reason": "source_ignore_all_scanners"}
        for slide_id in ignored_ids
    ]
    atomic_write_csv(
        output_root / "manifests/cohort.csv",
        cohort_rows,
        ["array_index", "slide_id", "source_candidate_count"],
    )
    atomic_write_csv(
        output_root / "manifests/excluded_slides.csv",
        excluded_rows,
        ["slide_id", "reason"],
    )
    submissions = output_root / "manifests/submissions.json"
    if not submissions.exists():
        atomic_write_json(submissions, {"submissions": []})

    config = build_config(slide_ids, ignored_ids)
    config["output_root"] = str(output_root)
    config["environment_at_prepare"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "opencv": cv2.__version__,
        "h5py": h5py.__version__,
        "openslide": openslide.__version__,
        "numpy": np.__version__,
    }
    atomic_write_json(output_root / "config/config.json", config)
    atomic_write_text(
        output_root / "preprocessing_patch_extraction_matching_and_QC.md",
        method_document(config),
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "cohort_slides": len(slide_ids),
                "ignored_slides": len(ignored_ids),
                "cohort_sha256": config["cohort_sha256"],
            },
            sort_keys=True,
        )
    )


def load_cohort(output_root: Path) -> list[dict[str, str]]:
    path = output_root / "manifests/cohort.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["array_index"]))
    return rows


def read_rgb(handle: openslide.OpenSlide, x: int, y: int, size: int) -> np.ndarray:
    return np.asarray(
        handle.read_region((int(x), int(y)), 0, (size, size)).convert("RGB"),
        dtype=np.uint8,
    )


def in_bounds(handle: openslide.OpenSlide, x: int, y: int, size: int) -> bool:
    width, height = handle.dimensions
    return x >= 0 and y >= 0 and x + size <= width and y + size <= height


def od_mean(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float32)
    return -np.log(np.maximum((value + 1.0) / 256.0, 1.0 / 256.0)).mean(axis=-1)


def tissue_fraction(rgb: np.ndarray) -> float:
    mask = (od_mean(rgb) > TISSUE_OD_THRESHOLD).astype(np.uint8)
    kernel = np.ones((TISSUE_CLOSE_KERNEL, TISSUE_CLOSE_KERNEL), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return float(mask.mean())


def laplacian_variance(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def gray_blur(rgb: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.GaussianBlur(gray, (0, 0), sigma)


def od_gradient(rgb: np.ndarray) -> np.ndarray:
    value = cv2.GaussianBlur(od_mean(rgb), (0, 0), 1.0)
    gx = cv2.Scharr(value, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(value, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    return cv2.GaussianBlur(magnitude, (0, 0), 0.8)


def match_map(ext: np.ndarray, ref: np.ndarray, representation) -> np.ndarray:
    result = cv2.matchTemplate(
        representation(ext), representation(ref), cv2.TM_CCOEFF_NORMED
    )
    return np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)


def peak(ncc_map: np.ndarray) -> tuple[int, int, float]:
    row, col = np.unravel_index(int(np.argmax(ncc_map)), ncc_map.shape)
    return int(row - SEARCH_MARGIN), int(col - SEARCH_MARGIN), float(ncc_map[row, col])


def peak_gap(ncc_map: np.ndarray, dy: int, dx: int, exclusion: int = 8) -> float:
    masked = ncc_map.copy()
    row, col = SEARCH_MARGIN + dy, SEARCH_MARGIN + dx
    y0 = max(row - exclusion, 0)
    y1 = min(row + exclusion + 1, masked.shape[0])
    x0 = max(col - exclusion, 0)
    x1 = min(col + exclusion + 1, masked.shape[1])
    masked[y0:y1, x0:x1] = -1.0
    return float(ncc_map[row, col] - np.max(masked))


def spatial_order(coords: np.ndarray, slide_id: str) -> list[int]:
    if len(coords) == 0:
        return []
    x = coords[:, 0].astype(np.float64)
    y = coords[:, 1].astype(np.float64)
    xspan = max(float(x.max() - x.min() + 1.0), 1.0)
    yspan = max(float(y.max() - y.min() + 1.0), 1.0)
    bx = np.minimum(((x - x.min()) / xspan * SPATIAL_BINS).astype(int), SPATIAL_BINS - 1)
    by = np.minimum(((y - y.min()) / yspan * SPATIAL_BINS).astype(int), SPATIAL_BINS - 1)
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(by.tolist(), bx.tolist())):
        grouped[key].append(index)
    seed = int(hashlib.sha256(slide_id.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    for indices in grouped.values():
        rng.shuffle(indices)
    ordered: list[int] = []
    keys = sorted(grouped)
    for round_index in range(max(len(grouped[key]) for key in keys)):
        for key in keys:
            indices = grouped[key]
            if round_index < len(indices):
                ordered.append(indices[round_index])
    return ordered


def match_scanner(
    handle: openslide.OpenSlide,
    source_x: int,
    source_y: int,
    reference_patch: np.ndarray,
) -> dict[str, Any]:
    search_x = source_x - SEARCH_MARGIN
    search_y = source_y - SEARCH_MARGIN
    search_size = PATCH_SIZE + 2 * SEARCH_MARGIN
    if not in_bounds(handle, search_x, search_y, search_size):
        return {"accepted": False, "reason": "search_out_of_bounds"}

    ext = read_rgb(handle, search_x, search_y, search_size)
    grad_map = match_map(ext, reference_patch, od_gradient)
    grad_dy, grad_dx, grad_score = peak(grad_map)
    gap = peak_gap(grad_map, grad_dy, grad_dx)
    grad_boundary = abs(grad_dy) >= SEARCH_MARGIN or abs(grad_dx) >= SEARCH_MARGIN
    base = {
        "grad_dx": grad_dx,
        "grad_dy": grad_dy,
        "grad_score": grad_score,
        "grad_gap": gap,
        "grad_boundary": grad_boundary,
        "global_dx": 0,
        "global_dy": 0,
        "global_ncc": math.nan,
        "global_boundary": False,
        "agreement_error": math.nan,
        "ext": ext,
    }
    if grad_boundary:
        return {**base, "accepted": False, "reason": "gradient_boundary"}
    if grad_score < GRAD_SCORE_MIN:
        return {**base, "accepted": False, "reason": "gradient_low_score"}
    if gap < GRAD_GAP_MIN:
        return {**base, "accepted": False, "reason": "gradient_ambiguous"}

    phase_residual = float(math.hypot(grad_dx, grad_dy))
    base["phase_residual"] = phase_residual
    if phase_residual <= PHASE_KEEP_TOLERANCE_PX:
        crop = ext[
            SEARCH_MARGIN : SEARCH_MARGIN + PATCH_SIZE,
            SEARCH_MARGIN : SEARCH_MARGIN + PATCH_SIZE,
        ]
        return {
            **base,
            "accepted": True,
            "mode": MODE_PHASE_KEPT,
            "shift_dx": 0,
            "shift_dy": 0,
            "patch": crop,
        }

    global_map = match_map(ext, reference_patch, gray_blur)
    global_dy, global_dx, global_ncc = peak(global_map)
    global_boundary = (
        abs(global_dy) >= SEARCH_MARGIN or abs(global_dx) >= SEARCH_MARGIN
    )
    agreement = float(math.hypot(global_dx - grad_dx, global_dy - grad_dy))
    base.update(
        {
            "global_dx": global_dx,
            "global_dy": global_dy,
            "global_ncc": global_ncc,
            "global_boundary": global_boundary,
            "agreement_error": agreement,
        }
    )
    if global_boundary:
        return {**base, "accepted": False, "reason": "global_boundary"}
    if global_ncc < GLOBAL_NCC_MIN:
        return {**base, "accepted": False, "reason": "global_low_ncc"}
    if agreement > MATCHER_AGREEMENT_TOLERANCE_PX:
        return {**base, "accepted": False, "reason": "matcher_disagreement"}

    row = SEARCH_MARGIN + global_dy
    col = SEARCH_MARGIN + global_dx
    crop = ext[row : row + PATCH_SIZE, col : col + PATCH_SIZE]
    if crop.shape != (PATCH_SIZE, PATCH_SIZE, 3):
        return {**base, "accepted": False, "reason": "final_crop_invalid"}
    return {
        **base,
        "accepted": True,
        "mode": MODE_GLOBAL_RESCUE,
        "shift_dx": global_dx,
        "shift_dy": global_dy,
        "patch": crop,
    }


def cache_datasets(handle: h5py.File, capacity: int) -> dict[str, h5py.Dataset]:
    first = (capacity,)
    maximum = (capacity,)
    datasets = {
        "images": handle.create_dataset(
            "images",
            shape=(capacity, len(SCANNERS), PATCH_SIZE, PATCH_SIZE, 3),
            maxshape=(capacity, len(SCANNERS), PATCH_SIZE, PATCH_SIZE, 3),
            dtype=np.uint8,
            chunks=(1, 1, PATCH_SIZE, PATCH_SIZE, 3),
            compression="lzf",
            shuffle=True,
        ),
        "source_index": handle.create_dataset(
            "source_index", shape=first, maxshape=maximum, dtype=np.int64
        ),
        "phase_coords_xy": handle.create_dataset(
            "phase_coords_xy",
            shape=(capacity, len(SCANNERS), 2),
            maxshape=(capacity, len(SCANNERS), 2),
            dtype=np.int32,
        ),
        "final_coords_xy": handle.create_dataset(
            "final_coords_xy",
            shape=(capacity, len(SCANNERS), 2),
            maxshape=(capacity, len(SCANNERS), 2),
            dtype=np.int32,
        ),
        "applied_shift_xy": handle.create_dataset(
            "applied_shift_xy",
            shape=(capacity, len(SCANNERS), 2),
            maxshape=(capacity, len(SCANNERS), 2),
            dtype=np.int16,
        ),
    }
    alignment = handle.create_group("alignment")
    for name, dtype in (
        ("mode", np.uint8),
        ("gradient_shift_xy", np.int16),
        ("global_shift_xy", np.int16),
        ("gradient_score", np.float32),
        ("gradient_peak_gap", np.float32),
        ("global_ncc", np.float32),
        ("phase_residual_px", np.float32),
        ("matcher_agreement_error_px", np.float32),
    ):
        if name.endswith("_xy"):
            shape = (capacity, len(SCANNERS), 2)
            maxshape = (capacity, len(SCANNERS), 2)
        else:
            shape = (capacity, len(SCANNERS))
            maxshape = (capacity, len(SCANNERS))
        datasets[f"alignment/{name}"] = alignment.create_dataset(
            name, shape=shape, maxshape=maxshape, dtype=dtype
        )
    quality = handle.create_group("quality")
    for name in ("tissue_fraction", "laplacian_variance"):
        datasets[f"quality/{name}"] = quality.create_dataset(
            name,
            shape=(capacity, len(SCANNERS)),
            maxshape=(capacity, len(SCANNERS)),
            dtype=np.float32,
        )
    handle.create_dataset("scanner_names", data=np.asarray(SCANNERS, dtype="S8"))
    return datasets


def resize_cache(datasets: dict[str, h5py.Dataset], length: int) -> None:
    for dataset in datasets.values():
        shape = list(dataset.shape)
        shape[0] = length
        dataset.resize(tuple(shape))


def write_pair(
    datasets: dict[str, h5py.Dataset],
    row: int,
    source_index: int,
    phase_coords: np.ndarray,
    patches: np.ndarray,
    scanner_results: list[dict[str, Any]],
) -> None:
    final_coords = phase_coords.astype(np.int32).copy()
    shifts = np.zeros((len(SCANNERS), 2), dtype=np.int16)
    modes = np.full(len(SCANNERS), MODE_REFERENCE, dtype=np.uint8)
    grad_shift = np.zeros((len(SCANNERS), 2), dtype=np.int16)
    global_shift = np.zeros((len(SCANNERS), 2), dtype=np.int16)
    grad_score = np.full(len(SCANNERS), np.nan, dtype=np.float32)
    grad_gap = np.full(len(SCANNERS), np.nan, dtype=np.float32)
    global_ncc = np.full(len(SCANNERS), np.nan, dtype=np.float32)
    phase_residual = np.zeros(len(SCANNERS), dtype=np.float32)
    agreement = np.full(len(SCANNERS), np.nan, dtype=np.float32)

    for scanner_index, result in enumerate(scanner_results, start=1):
        shift = np.asarray([result["shift_dx"], result["shift_dy"]], dtype=np.int16)
        shifts[scanner_index] = shift
        final_coords[scanner_index] += shift.astype(np.int32)
        modes[scanner_index] = result["mode"]
        grad_shift[scanner_index] = [result["grad_dx"], result["grad_dy"]]
        global_shift[scanner_index] = [result["global_dx"], result["global_dy"]]
        grad_score[scanner_index] = result["grad_score"]
        grad_gap[scanner_index] = result["grad_gap"]
        global_ncc[scanner_index] = result["global_ncc"]
        phase_residual[scanner_index] = result["phase_residual"]
        agreement[scanner_index] = result["agreement_error"]

    tissue = np.asarray([tissue_fraction(patch) for patch in patches], dtype=np.float32)
    focus = np.asarray([laplacian_variance(patch) for patch in patches], dtype=np.float32)
    datasets["images"][row] = patches
    datasets["source_index"][row] = source_index
    datasets["phase_coords_xy"][row] = phase_coords
    datasets["final_coords_xy"][row] = final_coords
    datasets["applied_shift_xy"][row] = shifts
    datasets["alignment/mode"][row] = modes
    datasets["alignment/gradient_shift_xy"][row] = grad_shift
    datasets["alignment/global_shift_xy"][row] = global_shift
    datasets["alignment/gradient_score"][row] = grad_score
    datasets["alignment/gradient_peak_gap"][row] = grad_gap
    datasets["alignment/global_ncc"][row] = global_ncc
    datasets["alignment/phase_residual_px"][row] = phase_residual
    datasets["alignment/matcher_agreement_error_px"][row] = agreement
    datasets["quality/tissue_fraction"][row] = tissue
    datasets["quality/laplacian_variance"][row] = focus


def phase_tuple(
    handles: dict[str, openslide.OpenSlide], coords: dict[str, np.ndarray], index: int
) -> np.ndarray:
    patches = []
    for scanner in SCANNERS:
        x, y = map(int, coords[scanner][index])
        if in_bounds(handles[scanner], x, y, PATCH_SIZE):
            patches.append(read_rgb(handles[scanner], x, y, PATCH_SIZE))
        else:
            patches.append(np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8))
    return np.stack(patches)


def edge_overlay(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    mov_gray = cv2.cvtColor(moving, cv2.COLOR_RGB2GRAY)
    ref_edge = cv2.Canny(ref_gray, 60, 140) > 0
    mov_edge = cv2.Canny(mov_gray, 60, 140) > 0
    base = ((ref_gray.astype(np.float32) + mov_gray.astype(np.float32)) / 2).astype(np.uint8)
    result = np.repeat(base[:, :, None], 3, axis=2)
    result[ref_edge] = [0, 220, 80]
    result[mov_edge] = [230, 40, 210]
    result[ref_edge & mov_edge] = [255, 255, 255]
    return result


def store_qc_examples(handle: h5py.File, examples: dict[str, dict[str, Any]]) -> None:
    group = handle.create_group("qc_examples")
    images = np.full(
        (len(EXAMPLE_CATEGORIES), len(SCANNERS), PATCH_SIZE, PATCH_SIZE, 3),
        255,
        dtype=np.uint8,
    )
    present = np.zeros(len(EXAMPLE_CATEGORIES), dtype=np.uint8)
    labels: list[str] = []
    worst_scanners: list[str] = []
    for index, category in enumerate(EXAMPLE_CATEGORIES):
        example = examples.get(category)
        if example is None:
            labels.append("not observed")
            worst_scanners.append(REFERENCE_SCANNER)
            continue
        images[index] = example["images"]
        present[index] = 1
        labels.append(str(example["label"]))
        worst_scanners.append(str(example["worst_scanner"]))
    group.create_dataset(
        "images",
        data=images,
        chunks=(1, 1, PATCH_SIZE, PATCH_SIZE, 3),
        compression="lzf",
        shuffle=True,
    )
    string_type = h5py.string_dtype("utf-8")
    group.create_dataset(
        "categories", data=np.asarray(EXAMPLE_CATEGORIES, dtype=object), dtype=string_type
    )
    group.create_dataset("present", data=present)
    group.create_dataset(
        "labels", data=np.asarray(labels, dtype=object), dtype=string_type
    )
    group.create_dataset(
        "worst_scanner", data=np.asarray(worst_scanners, dtype=object), dtype=string_type
    )


def font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def render_slide_qc(
    slide_id: str,
    examples: dict[str, dict[str, Any]],
    destination: Path,
    accepted: int,
    processed: int,
) -> None:
    panel = 176
    label_height = 46
    header = 52
    columns = len(SCANNERS) + 1
    canvas = Image.new(
        "RGB", (columns * panel, header + len(EXAMPLE_CATEGORIES) * (panel + label_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 6),
        f"{slide_id} | accepted {accepted} / processed {processed} | reference tissue >= 0.90",
        fill="black",
        font=font(),
    )
    for column, name in enumerate((*SCANNERS, "edge overlay")):
        draw.text((column * panel + 6, 29), name, fill="black", font=font())
    for row, category in enumerate(EXAMPLE_CATEGORIES):
        top = header + row * (panel + label_height)
        example = examples.get(category)
        if example is None:
            draw.rectangle((0, top, columns * panel, top + panel), fill=(238, 238, 238))
            draw.text((8, top + 8), f"{category}: not observed", fill=(80, 80, 80), font=font())
            continue
        patches = example["images"]
        for column, patch in enumerate(patches):
            image = Image.fromarray(patch).resize((panel, panel), Image.Resampling.BILINEAR)
            canvas.paste(image, (column * panel, top))
        worst_index = SCANNERS.index(example["worst_scanner"])
        overlay = edge_overlay(patches[0], patches[worst_index])
        overlay_image = Image.fromarray(overlay).resize((panel, panel), Image.Resampling.BILINEAR)
        canvas.paste(overlay_image, ((columns - 1) * panel, top))
        draw.text(
            (8, top + panel + 3),
            f"{category} | {example['label']}",
            fill="black",
            font=font(),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.png")
    canvas.save(temporary)
    os.replace(temporary, destination)


def slide(args: argparse.Namespace) -> None:
    cv2.setNumThreads(1)
    output_root = Path(args.output_root).resolve()
    cohort = load_cohort(output_root)
    if not 0 <= args.task_index < len(cohort):
        raise SystemExit(f"task index must be in 0..{len(cohort)-1}")
    slide_id = cohort[args.task_index]["slide_id"]
    target = int(args.maximum_pairs)
    if target <= 0:
        raise ValueError("maximum-pairs must be positive")

    cache_path = output_root / "cache" / f"{slide_id}.h5"
    result_path = output_root / "qc/results" / f"{slide_id}.json"
    qc_path = output_root / "qc/slides" / f"{slide_id}.png"
    if (cache_path.exists() or result_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"output already exists for {slide_id}; pass --overwrite for an explicit rerun"
        )

    coords: dict[str, np.ndarray] = {}
    for scanner in SCANNERS:
        with h5py.File(h5_path(scanner, slide_id), "r") as handle:
            coords[scanner] = np.asarray(handle["coords"][:], dtype=np.int64)
    counts = {scanner: len(values) for scanner, values in coords.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"coordinate counts differ for {slide_id}: {counts}")
    order = spatial_order(coords[REFERENCE_SCANNER], slide_id)
    if args.max_candidates is not None:
        order = order[: args.max_candidates]
    capacity = min(target, len(order))
    if capacity == 0:
        raise RuntimeError(f"{slide_id}: no source coordinates")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(
        f".{cache_path.name}.{os.environ.get('SLURM_JOB_ID', os.getpid())}.tmp"
    )
    if temporary.exists():
        temporary.unlink()

    handles = {
        scanner: openslide.OpenSlide(str(image_path(scanner, slide_id)))
        for scanner in SCANNERS
    }
    accepted = 0
    processed = 0
    rejections: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}
    started = utc_now()
    output_handle: h5py.File | None = None
    datasets: dict[str, h5py.Dataset] = {}
    try:
        output_handle = h5py.File(temporary, "w")
        datasets = cache_datasets(output_handle, capacity)
        for source_index in order:
            if accepted >= target:
                break
            processed += 1
            phase_coords = np.stack([coords[scanner][source_index] for scanner in SCANNERS]).astype(
                np.int32
            )
            ref_x, ref_y = map(int, phase_coords[0])
            if not in_bounds(handles[REFERENCE_SCANNER], ref_x, ref_y, PATCH_SIZE):
                rejections["reference_out_of_bounds"] += 1
                continue
            reference_patch = read_rgb(
                handles[REFERENCE_SCANNER], ref_x, ref_y, PATCH_SIZE
            )
            reference_tissue = tissue_fraction(reference_patch)
            if reference_tissue < TISSUE_FRACTION_MIN:
                rejections["reference_tissue_below_0.90"] += 1
                continue

            scanner_results: list[dict[str, Any]] = []
            rejected: tuple[str, str] | None = None
            for scanner_index, scanner in enumerate(NON_REFERENCE_SCANNERS, start=1):
                source_x, source_y = map(int, phase_coords[scanner_index])
                result = match_scanner(
                    handles[scanner], source_x, source_y, reference_patch
                )
                if not result["accepted"]:
                    rejected = (scanner, result["reason"])
                    rejections[f"{scanner}:{result['reason']}"] += 1
                    break
                scanner_results.append(result)
            if rejected is not None:
                if "rejected" not in examples:
                    examples["rejected"] = {
                        "images": phase_tuple(handles, coords, source_index),
                        "worst_scanner": rejected[0],
                        "label": f"{rejected[0]} {rejected[1]} | source index {source_index}",
                    }
                continue

            patches = np.stack(
                [reference_patch, *(result["patch"] for result in scanner_results)]
            )
            write_pair(
                datasets,
                accepted,
                source_index,
                phase_coords,
                patches,
                scanner_results,
            )
            accepted += 1
            for result in scanner_results:
                modes[MODE_NAMES[result["mode"]]] += 1

            moves = np.asarray(
                [math.hypot(result["shift_dx"], result["shift_dy"]) for result in scanner_results]
            )
            max_move = float(moves.max(initial=0.0))
            worst_nonref = int(np.argmax(moves)) if len(moves) else 0
            worst_scanner = NON_REFERENCE_SCANNERS[worst_nonref]
            if max_move <= PHASE_KEEP_TOLERANCE_PX:
                category = "phase_verified"
            elif max_move <= 5.0:
                category = "slight_rescue_2to5px"
            else:
                category = "large_rescue_over5px"
            if category not in examples:
                examples[category] = {
                    "images": patches.copy(),
                    "worst_scanner": worst_scanner,
                    "label": f"max applied shift {max_move:.1f}px | source index {source_index}",
                }

        resize_cache(datasets, accepted)
        store_qc_examples(output_handle, examples)
        output_handle.attrs.update(
            {
                "slide_id": slide_id,
                "pair_count": accepted,
                "maximum_pairs": target,
                "cap_reached": accepted >= target,
                "source_candidate_count": len(coords[REFERENCE_SCANNER]),
                "processed_candidate_count": processed,
                "patch_size": PATCH_SIZE,
                "scanner_order": ",".join(SCANNERS),
                "reference_scanner": REFERENCE_SCANNER,
                "tissue_fraction_min": TISSUE_FRACTION_MIN,
                "tissue_od_threshold": TISSUE_OD_THRESHOLD,
                "search_margin_px": SEARCH_MARGIN,
                "phase_keep_tolerance_px": PHASE_KEEP_TOLERANCE_PX,
                "matcher_agreement_tolerance_px": MATCHER_AGREEMENT_TOLERANCE_PX,
                "gradient_score_min": GRAD_SCORE_MIN,
                "gradient_peak_gap_min": GRAD_GAP_MIN,
                "global_ncc_min": GLOBAL_NCC_MIN,
                "created_utc": utc_now(),
            }
        )
        output_handle.flush()
        output_handle.close()
        output_handle = None
    finally:
        if output_handle is not None:
            output_handle.close()
        for handle in handles.values():
            handle.close()

    if accepted == 0:
        temporary.unlink(missing_ok=True)
        status = "no_passing_pairs"
    else:
        if cache_path.exists() and args.overwrite:
            cache_path.unlink()
        os.replace(temporary, cache_path)
        status = "pass"

    render_slide_qc(slide_id, examples, qc_path, accepted, processed)
    result = {
        "status": status,
        "slide_id": slide_id,
        "array_index": args.task_index,
        "started_utc": started,
        "completed_utc": utc_now(),
        "source_candidate_count": len(coords[REFERENCE_SCANNER]),
        "processed_candidate_count": processed,
        "accepted_pair_count": accepted,
        "maximum_pairs": target,
        "cap_reached": accepted >= target,
        "acceptance_fraction_among_processed": accepted / processed if processed else 0.0,
        "rejection_counts": dict(sorted(rejections.items())),
        "accepted_scanner_pair_modes": dict(sorted(modes.items())),
        "example_categories_present": sorted(examples),
        "cache_path": str(cache_path) if accepted else None,
        "qc_image_path": str(qc_path),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
        },
    }
    atomic_write_json(result_path, result)
    print(json.dumps(result, sort_keys=True))


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def render_condition_gallery(
    category: str,
    entries: list[dict[str, Any]],
    destination: Path,
) -> None:
    panel = 132
    label_height = 42
    header = 38
    canvas = Image.new(
        "RGB",
        (len(SCANNERS) * panel, header + len(entries) * (panel + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((7, 5), category, fill="black", font=font())
    for column, scanner in enumerate(SCANNERS):
        draw.text((column * panel + 5, 21), scanner, fill="black", font=font())
    for row, entry in enumerate(entries):
        top = header + row * (panel + label_height)
        for column, patch in enumerate(entry["images"]):
            image = Image.fromarray(patch).resize((panel, panel), Image.Resampling.BILINEAR)
            canvas.paste(image, (column * panel, top))
        draw.text(
            (6, top + panel + 2),
            f"{entry['slide_id']} | {entry['label']}",
            fill="black",
            font=font(),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.png")
    canvas.save(temporary)
    os.replace(temporary, destination)


def validate_cache(path: Path, expected_slide: str, maximum_pairs: int) -> int:
    with h5py.File(path, "r") as handle:
        count = int(handle.attrs["pair_count"])
        if str(handle.attrs["slide_id"]) != expected_slide:
            raise RuntimeError(f"{path}: slide_id mismatch")
        if not 0 < count <= maximum_pairs:
            raise RuntimeError(f"{path}: invalid pair count {count}")
        expected = (count, len(SCANNERS), PATCH_SIZE, PATCH_SIZE, 3)
        if handle["images"].shape != expected:
            raise RuntimeError(f"{path}: images shape {handle['images'].shape}, expected {expected}")
        source_index = np.asarray(handle["source_index"][:], dtype=np.int64)
        if len(np.unique(source_index)) != count:
            raise RuntimeError(f"{path}: duplicate source indices")
        phase = np.asarray(handle["phase_coords_xy"][:], dtype=np.int32)
        final = np.asarray(handle["final_coords_xy"][:], dtype=np.int32)
        shifts = np.asarray(handle["applied_shift_xy"][:], dtype=np.int32)
        if not np.array_equal(final, phase + shifts):
            raise RuntimeError(f"{path}: coordinate/shift invariant failed")
        if not np.all(np.asarray(handle["quality/tissue_fraction"][:, 0]) >= TISSUE_FRACTION_MIN):
            raise RuntimeError(f"{path}: reference tissue gate invariant failed")
        return count


def aggregate(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    cohort = load_cohort(output_root)
    result_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    gallery_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_rejections: Counter[str] = Counter()
    total_modes: Counter[str] = Counter()
    maximum_pairs = int(args.maximum_pairs)

    for cohort_row in cohort:
        slide_id = cohort_row["slide_id"]
        result_path = output_root / "qc/results" / f"{slide_id}.json"
        if not result_path.is_file():
            missing.append(slide_id)
            continue
        result = json.loads(result_path.read_text())
        cache_path = output_root / "cache" / f"{slide_id}.h5"
        cache_count = 0
        if result["status"] == "pass":
            if not cache_path.is_file():
                raise FileNotFoundError(f"{slide_id}: result says pass but cache is missing")
            cache_count = validate_cache(cache_path, slide_id, maximum_pairs)
            if cache_count != int(result["accepted_pair_count"]):
                raise RuntimeError(f"{slide_id}: result/cache pair count mismatch")
            with h5py.File(cache_path, "r") as handle:
                categories = decode_strings(handle["qc_examples/categories"][:])
                labels = decode_strings(handle["qc_examples/labels"][:])
                present = np.asarray(handle["qc_examples/present"][:], dtype=bool)
                for index, category in enumerate(categories):
                    if present[index] and len(gallery_entries[category]) < args.gallery_examples:
                        gallery_entries[category].append(
                            {
                                "slide_id": slide_id,
                                "label": labels[index],
                                "images": np.asarray(handle["qc_examples/images"][index]),
                            }
                        )
        total_rejections.update(result.get("rejection_counts", {}))
        total_modes.update(result.get("accepted_scanner_pair_modes", {}))
        result_rows.append(
            {
                "array_index": cohort_row["array_index"],
                "slide_id": slide_id,
                "status": result["status"],
                "source_candidate_count": result["source_candidate_count"],
                "processed_candidate_count": result["processed_candidate_count"],
                "accepted_pair_count": result["accepted_pair_count"],
                "cap_reached": result["cap_reached"],
                "acceptance_fraction_among_processed": result[
                    "acceptance_fraction_among_processed"
                ],
                "cache_bytes": cache_path.stat().st_size if cache_path.is_file() else 0,
            }
        )

    fields = [
        "array_index",
        "slide_id",
        "status",
        "source_candidate_count",
        "processed_candidate_count",
        "accepted_pair_count",
        "cap_reached",
        "acceptance_fraction_among_processed",
        "cache_bytes",
    ]
    atomic_write_csv(output_root / "qc/slide_summary.csv", result_rows, fields)

    for category in EXAMPLE_CATEGORIES:
        entries = gallery_entries.get(category, [])
        if entries:
            render_condition_gallery(
                category,
                entries,
                output_root / "qc/condition_galleries" / f"{category}.png",
            )

    counts = [int(row["accepted_pair_count"]) for row in result_rows if row["status"] == "pass"]
    processed_count = sum(int(row["processed_candidate_count"]) for row in result_rows)
    source_count = sum(int(row["source_candidate_count"]) for row in result_rows)
    cache_apparent_bytes = sum(int(row["cache_bytes"]) for row in result_rows)
    below_cap_slides = [
        {
            "slide_id": row["slide_id"],
            "accepted_pair_count": int(row["accepted_pair_count"]),
            "source_candidate_count": int(row["source_candidate_count"]),
        }
        for row in sorted(
            result_rows,
            key=lambda item: (int(item["accepted_pair_count"]), item["slide_id"]),
        )
        if row["status"] == "pass" and not bool(row["cap_reached"])
    ]
    scanner_pair_count = int(sum(total_modes.values()))
    phase_kept_count = int(total_modes.get("phase_kept", 0))
    global_rescue_count = int(total_modes.get("global_rescue", 0))
    submissions_path = output_root / "manifests/submissions.json"
    submissions = (
        json.loads(submissions_path.read_text()).get("submissions", [])
        if submissions_path.exists()
        else []
    )
    summary = {
        "pipeline": "preprocessing_patch_extraction_matching_and_QC",
        "aggregated_utc": utc_now(),
        "expected_slide_count": len(cohort),
        "result_slide_count": len(result_rows),
        "missing_result_slides": missing,
        "cached_slide_count": len(counts),
        "no_passing_pair_slide_count": sum(row["status"] != "pass" for row in result_rows),
        "maximum_pairs_per_slide": maximum_pairs,
        "cap_reached_slide_count": sum(bool(row["cap_reached"]) for row in result_rows),
        "total_pair_count": int(sum(counts)),
        "source_candidate_count": source_count,
        "processed_candidate_count": processed_count,
        "acceptance_fraction_among_processed": (
            float(sum(counts) / processed_count) if processed_count else 0.0
        ),
        "cache_apparent_bytes": cache_apparent_bytes,
        "below_cap_slides": below_cap_slides,
        "pair_count": {
            "min": int(min(counts)) if counts else 0,
            "median": float(np.median(counts)) if counts else 0,
            "max": int(max(counts)) if counts else 0,
        },
        "accepted_nonreference_scanner_pair_count": scanner_pair_count,
        "phase_kept_scanner_pair_count": phase_kept_count,
        "phase_kept_scanner_pair_fraction": (
            float(phase_kept_count / scanner_pair_count) if scanner_pair_count else 0.0
        ),
        "global_rescue_scanner_pair_count": global_rescue_count,
        "global_rescue_scanner_pair_fraction": (
            float(global_rescue_count / scanner_pair_count) if scanner_pair_count else 0.0
        ),
        "total_rejection_counts": dict(sorted(total_rejections.items())),
        "total_accepted_scanner_pair_modes": dict(sorted(total_modes.items())),
        "condition_gallery_example_counts": {
            category: len(gallery_entries.get(category, [])) for category in EXAMPLE_CATEGORIES
        },
        "submissions": submissions,
    }
    atomic_write_json(output_root / "qc/run_summary.json", summary)
    config = json.loads((output_root / "config/config.json").read_text())
    atomic_write_text(
        output_root / "preprocessing_patch_extraction_matching_and_QC.md",
        method_document(config, summary),
    )
    print(json.dumps(summary, sort_keys=True))
    if missing:
        raise SystemExit(f"missing {len(missing)} result slides")


def record_submission(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    path = output_root / "manifests/submissions.json"
    payload = json.loads(path.read_text()) if path.exists() else {"submissions": []}
    payload.setdefault("submissions", []).append(
        {
            "recorded_utc": utc_now(),
            "kind": args.kind,
            "job_id": str(args.job_id),
            "command": args.command,
        }
    )
    atomic_write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    prepare_parser.add_argument("--expected-slides", type=int, default=103)
    prepare_parser.set_defaults(function=prepare)

    slide_parser = subparsers.add_parser("slide")
    slide_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    slide_parser.add_argument("--task-index", type=int, required=True)
    slide_parser.add_argument("--maximum-pairs", type=int, default=1000)
    slide_parser.add_argument("--max-candidates", type=int)
    slide_parser.add_argument("--overwrite", action="store_true")
    slide_parser.set_defaults(function=slide)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    aggregate_parser.add_argument("--maximum-pairs", type=int, default=1000)
    aggregate_parser.add_argument("--gallery-examples", type=int, default=8)
    aggregate_parser.set_defaults(function=aggregate)

    record_parser = subparsers.add_parser("record-submission")
    record_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    record_parser.add_argument("--kind", required=True)
    record_parser.add_argument("--job-id", required=True)
    record_parser.add_argument("--command", required=True)
    record_parser.set_defaults(function=record_submission)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
