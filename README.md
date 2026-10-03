# Scanner Spectrum

PanNormal scanner batch effect 논문의 분석 코드와 고정된 근거다. 현재 영문 원고는
별도 Git 저장소인 `00_manuscript/`에 있으며, 이 저장소와 나란히 있어야 한다.

## 구조

| 경로 | 내용 |
| --- | --- |
| `analysis/revision/` | 현재 원고의 분석(RV-P0–RV14, RV18–RV19)과 그림·표 생성기. 분석별 프로토콜, 결과 색인, 원고 자산 지도는 [`analysis/revision/README.md`](analysis/revision/README.md) |
| `analysis/paper/` | revision 이전에 고정되어 현재 원고에도 쓰이는 분석(혼합효과 모형, 색·주파수 위치, 증강 오라클, 영상 교정 벤치마크 등). [`analysis/paper/README.md`](analysis/paper/README.md) |
| `src/` | 재사용 함수(`prenorm/`), 영상 변환 모델(`scanner_gan/`), 업스트림 파이프라인 프로그램 |
| `scripts/` | 업스트림 파이프라인의 SLURM 래퍼. [`scripts/README.md`](scripts/README.md) |
| `data/` | 매칭된 패치와 출처 정보 (Git 제외) |
| `outputs/` | 고정된 업스트림 결과와 모델 (Git 제외) |
| `docs/records/` | 정리하며 지운 분석과 파일의 기록. [`docs/records/README.md`](docs/records/README.md) |

`analysis/*/results/`도 Git에서 제외된다.

## 원고 재생성

`cpath` 환경에서 저장소 루트를 기준으로 실행한다.

```bash
# 그림과 표 다시 만들기 (00_manuscript/figures, 00_manuscript/tables)
sbatch --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD" analysis/revision/manuscript_assets_rebuild.sbatch
# PDF 빌드 (analysis/paper/results/manuscript_build/)
sbatch --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD" analysis/paper/build_scanner_tissue_manuscript.sbatch
```

그림 1(raw paired gallery)은 `00_manuscript/figures/make_raw_paired_gallery.py`,
보충 그림 S2(coherence)는 `analysis/revision/frequency_transfer_figures.py`가 만든다.

## 정리 기록

2026-10-02에 원고에 없는 분석(RV15–RV17, Virchow2 epoch 2–3), 쓰지 않는 체크포인트,
smoke test, 캐시, 로그, 이전 원고용 검증 도구, 테스트, `.Trash/`를 지웠다. 지운 경로와
이유, 지운 분석의 요약은 `docs/records/`에 있다.
