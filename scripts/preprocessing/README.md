# Phase 0 v3 launchers

현재 기준 데이터셋은 full 50-slide v3이며, slide-level split은 40/5/5
(8:1:1)다. GPU 학습은 이 단계에서 실행하지 않는다.

| 순서 | launcher | 출력 |
|---|---|---|
| 1 | `s01s02_50slide_v3.sbatch` | spatial grid와 candidate parquet |
| 2 | `s03_50slide_v3.sbatch` | slide별 RGB HDF5와 sidecar |
| 3 | `finalize_50slide_v3.sbatch` | flat index와 train-only AT2 stain artifact |
| 4 | `alignment_samples_50slide_v3.sbatch` | slide별 10-tuple 정렬 패널 및 CSV |

```bash
sbatch --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD" scripts/preprocessing/s01s02_50slide_v3.sbatch
sbatch --dependency=afterok:<s01s02_jobid> --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD" --array=0-49%24 scripts/preprocessing/s03_50slide_v3.sbatch
sbatch --dependency=afterok:<s03_array_jobid> --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD" scripts/preprocessing/finalize_50slide_v3.sbatch
sbatch --dependency=afterok:<s03_array_jobid> --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD" --array=0-49%24 scripts/viz/alignment_samples_50slide_v3.sbatch
```

의존성을 포함한 검증용 전체 흐름은 repository root에서 `bash scripts/submit_10slide_dry_run.sh`로 제출한다.
