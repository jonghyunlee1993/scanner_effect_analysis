# Scanner Spectrum

`scanner-spectrum`은 registered multi-scanner H&E 영상에서 scanner effect를
주파수 대역과 feature representation 수준으로 분해하는 분석 프로젝트다.

현재 연구 질문은 하나의 RGB canonicalizer를 학습하는 것이 아니라 다음 경계를 정량화하는 것이다.

1. 저주파 scanner difference 중 단순 affine transform으로 설명되는 부분은 얼마인가?
2. 저주파 교정이 병리 foundation model embedding alignment에도 전달되는가?
3. scanner별 고주파 정보량과 고주파 손상이 embedding drift에 미치는 영향은 무엇인가?
4. 이 현상이 pathology FM과 natural-image FM에서 공통적으로 나타나는가?
5. content-preserving image normalization과 feature-level correction의 현실적인 한계는 어디인가?

## 현재 근거

- Scanner-specific paired affine은 held-out 저주파 MAE를 크게 줄인다.
- Train-slide RGB affine과 명시적 clipping만으로 held-out scanner 저주파 차이의 상당 부분을 교정한다.
- 저주파 feature 수준 scanner separability는 감소하지만, UNI embedding scanner separability는 거의 변하지 않는다.
- 고주파를 억제하면 morphology가 손상되고, source 고주파를 보존하면 scanner signature가 함께 남는다.

이 결과는 “저주파 시각 정규화”와 “PFM representation invariance”가 서로 다른 문제임을 시사한다.
현재 수치는 10-slide pilot에 기반하므로 109-slide 분석과 복수 FM 검증 전에는 최종 결론으로 사용하지 않는다.

## Active 코드

```text
src/prenorm/embedding.py      UNI embedding과 scanner/alignment 지표
src/prenorm/metrics.py        image-level 평가 지표
src/prenorm/exp01/            registered data split과 frequency decomposition
src/prenorm/exp02/            affine projection, pairwise, HF trajectory 분석
src/eval_exp01_baselines.py   image normalization baseline 평가
src/eval_exp02_*.py           저주파, UNI, band-oracle 분석
src/run_exp02_pairwise_uni.py pairwise UNI/blur 측정
src/render_exp02_pairwise_uni.py publication figure 렌더링
src/run_exp02_hf_trajectory.py scanner×HF perturbation UNI trajectory
src/render_exp02_hf_trajectory.py 조건별 fixed-UMAP, reference centroid, pair figures
src/run_exp02_lf_aligned_hf_trajectory.py rotating LF-affine 후 HF trajectory
src/render_exp02_lf_aligned_hf_trajectory.py raw/LF-aligned 방향 비교와 Akoya audit
src/preprocess/               RGB store와 index 전처리
configs/experiments/exp01*.yaml
scripts/exp01/
scripts/exp02/
```

현재 결과와 해석은 다음 문서에 기록되어 있다.

- [Experiment index](docs/experiment_index.md)
- [Research synthesis](docs/research_synthesis.md)
- [Exp-01 execution log](docs/exp01_execution_log.md)
- [Exp-02 execution log](docs/exp02_execution_log.md)
- [Exp-03 common-space feasibility plan](docs/exp03_common_space_feasibility_plan.md)
- [Retired attempts](docs/retired_attempts.md)

## 데이터 계약

대용량 HDF5, parquet index, embedding, checkpoint와 결과물은 Git으로 추적하지 않는다.
로컬 `data/`, `outputs/`, `logs/`에서 관리한다.

Internal AT2/GT450/VERSA/Akoya lattice와 독립적으로 구축된 AT2/S60 lattice는
서로 다른 공간 grid다. 현재 분석에서 S60 affine도 train slide로 fit하므로,
S60 결과는 unseen-scanner external validation이 아니라 별도 `S60 lattice`
분석이다.
샘플 또는 evaluation group을 만들 때 반드시 `lattice_id`를 포함해야 하며,
서로 다른 lattice의 `(slide_id, tuple_id)`를 같은 registered location으로 취급하면 안 된다.

## 검증

가벼운 핵심 검사는 다음과 같다.

```bash
conda activate cpath
python -m pytest -q tests/test_exp01.py tests/test_exp02.py
```

전체 suite는 로그인 노드가 아니라 `defq`에서 실행한다.

```bash
sbatch --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD" \
  scripts/test_full_suite.sbatch
```

Exp-02의 pairwise UNI 측정은 `gpuq`, figure 렌더링은 `defq`에서 실행한다.

```bash
sbatch --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD" \
  scripts/exp02/pairwise_uni_gpu.sbatch
sbatch --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD" \
  scripts/exp02/render_pairwise_cpu.sbatch
```

## Archive

종료된 v4 canonicalizer modeling 코드, config, launcher와 결과는
`archived/2026-07-23_modeling_v4/`로 이동했다. `archived/`는 Git에서 제외되며
이전 실험의 provenance를 위한 로컬 보관소로만 사용한다.

이번 분석 전환에서 제거한 tracked source와 launcher는 Git history에서 복구할
수 있고, 시도·결과·폐기 이유는 `docs/retired_attempts.md`에 남긴다.
