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
- Range-safe local projection은 source high-frequency coefficient를 보존하면서 affine 교정력을 대부분 유지한다.
- 저주파 feature 수준 scanner separability는 감소하지만, UNI embedding scanner separability는 거의 변하지 않는다.
- 고주파를 억제하면 morphology가 손상되고, source 고주파를 보존하면 scanner signature가 함께 남는다.

이 결과는 “저주파 시각 정규화”와 “PFM representation invariance”가 서로 다른 문제임을 시사한다.
현재 수치는 10-slide pilot에 기반하므로 109-slide 분석과 복수 FM 검증 전에는 최종 결론으로 사용하지 않는다.

## Active 코드

```text
src/prenorm/exp06/            fixed Laplacian pyramid와 저주파 pilot
src/prenorm/exp07/            range-safe affine projection
src/eval_exp07_*.py           저주파, UNI, band-oracle 분석
src/preprocess/               RGB store와 index 전처리
configs/experiments/exp06*.yaml
scripts/exp06/
scripts/exp07/
```

현재 결과와 해석은 다음 문서에 기록되어 있다.

- [Exp-06 execution log](docs/exp06_execution_log.md)
- [Exp-07 execution log](docs/exp07_execution_log.md)
- [Exp-07 analysis plan](docs/exp07_amortized_affine_residual_plan.md)

## 데이터 계약

대용량 HDF5, parquet index, embedding, checkpoint와 결과물은 Git으로 추적하지 않는다.
로컬 `data/`, `outputs/`, `logs/`에서 관리한다.

Internal AT2/GT450/VERSA/Akoya lattice와 external AT2/S60 lattice는 서로 다른 공간 grid다.
샘플 또는 evaluation group을 만들 때 반드시 `lattice_id`를 포함해야 하며,
서로 다른 lattice의 `(slide_id, tuple_id)`를 같은 registered location으로 취급하면 안 된다.

## 검증

가벼운 핵심 검사는 다음과 같다.

```bash
conda activate cpath
python -m pytest -q tests/test_exp06.py tests/test_exp07.py
```

전체 suite는 로그인 노드가 아니라 `defq`에서 실행한다.

```bash
sbatch --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD" \
  scripts/test_full_suite.sbatch
```

## Archive

종료된 v4 canonicalizer modeling 코드, config, launcher와 결과는
`archived/2026-07-23_modeling_v4/`로 이동했다. `archived/`는 Git에서 제외되며
이전 실험의 provenance를 위한 로컬 보관소로만 사용한다.
