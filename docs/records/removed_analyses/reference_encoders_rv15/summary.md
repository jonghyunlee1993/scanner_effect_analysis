# RV15 natural-image reference encoders: CLIP (PLIP's parent) and ResNet50

Status: computed; not yet discussed. Protocol: `analysis/revision/README.md`, RV15 (written before outcomes).

## QC

- PLIP re-embedded in the RV15 job vs RV13's stored PLIP embeddings: 166860 images on 103 slides, min cosine 1.000000 (gate 0.999), median 1.0000000.
- Band renders vs RV02 shards: 103/103 slides identical.
- Loader parity (`parity/parity.json`): pass = True.
  - clip / features_float_path_vs_released_processor: min cosine 0.9991099
  - resnet50 / float_preprocessing_vs_torchvision_resize_normalize: min cosine 0.9999999
  - resnet50 / full_field_224_vs_released_transform_resize256_crop224 (information): min cosine 0.9494985
  - resnet50 / pooled_stream_vs_torchvision_forward: min cosine 0.9999999
  - resnet50_layer3 / full_field_224_vs_released_transform_resize256_crop224 (information): min cosine 0.9857500
  - CLIP vs PLIP: vision architecture equal True, preprocessor equal True, same state-dict keys/shapes True; PLIP changed 200/200 vision tensors (relative L2 change 0.0142) and 197/197 text tensors (0.0171).
- RV14 reproduced from the same code: 319 model-table values, max abs diff 4.4e-16; 21 correlations, max abs diff 1.1e-16; pass = True.
- Chance levels (random other-slide neighbours, from label counts): RI 0.089, SO 0.016, OS 0.163. Detectability chance 0.5; six-way probe 0.167.

## Endpoints (estimate [95% CI]; band sensitivity = shift / between-tissue distance at dose 0.25)

| Model | Group | Low–mid | Mid | High | Normalized distance | Best probe | RI (k*) | Retrieval | SO | OS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNI | WSI-pretrained | 0.0050 [0.0045, 0.0056] | 0.0049 [0.0045, 0.0053] | 0.0064 [0.0059, 0.0069] | 0.243 [0.235, 0.252] | 0.992 [0.987, 0.996] | 0.343 [0.286, 0.406] (15) | 0.615 [0.549, 0.690] | 0.186 [0.158, 0.212] | 0.355 [0.308, 0.402] |
| UNI2-h | WSI-pretrained | 0.0048 [0.0043, 0.0053] | 0.0076 [0.0067, 0.0085] | 0.0083 [0.0075, 0.0092] | 0.313 [0.297, 0.328] | 0.995 [0.989, 0.999] | 0.302 [0.244, 0.369] (10) | 0.655 [0.586, 0.731] | 0.151 [0.127, 0.175] | 0.348 [0.292, 0.404] |
| Virchow2 | WSI-pretrained | 0.0021 [0.0019, 0.0024] | 0.0019 [0.0017, 0.0020] | 0.0014 [0.0013, 0.0015] | 0.122 [0.114, 0.131] | 0.996 [0.992, 0.998] | 0.529 [0.464, 0.594] (5) | 0.640 [0.564, 0.727] | 0.275 [0.242, 0.308] | 0.245 [0.210, 0.281] |
| H-optimus-1 | WSI-pretrained | 0.0040 [0.0035, 0.0046] | 0.0032 [0.0029, 0.0036] | 0.0019 [0.0018, 0.0021] | 0.160 [0.147, 0.173] | 0.997 [0.995, 0.999] | 0.506 [0.435, 0.576] (10) | 0.669 [0.600, 0.749] | 0.267 [0.231, 0.299] | 0.260 [0.218, 0.303] |
| EXAONEPath | WSI-pretrained | 0.0079 [0.0069, 0.0091] | 0.0109 [0.0097, 0.0123] | 0.0100 [0.0088, 0.0112] | 0.476 [0.453, 0.500] | 0.991 [0.982, 0.997] | 0.179 [0.142, 0.220] (5) | 0.506 [0.424, 0.591] | 0.105 [0.087, 0.124] | 0.481 [0.432, 0.530] |
| SEAL-UNI2 | WSI-pretrained | 0.0042 [0.0038, 0.0047] | 0.0075 [0.0066, 0.0086] | 0.0083 [0.0075, 0.0093] | 0.300 [0.284, 0.316] | 0.996 [0.992, 0.999] | 0.394 [0.328, 0.462] (20) | 0.664 [0.596, 0.739] | 0.205 [0.177, 0.234] | 0.316 [0.267, 0.365] |
| CONCH | WSI-pretrained | 0.0017 [0.0014, 0.0020] | 0.0027 [0.0025, 0.0030] | 0.0040 [0.0037, 0.0044] | 0.139 [0.132, 0.147] | 0.973 [0.964, 0.980] | 0.410 [0.351, 0.469] (5) | 0.587 [0.514, 0.672] | 0.205 [0.178, 0.231] | 0.296 [0.261, 0.332] |
| SEAL-CONCH | WSI-pretrained | 0.0014 [0.0012, 0.0017] | 0.0026 [0.0023, 0.0028] | 0.0037 [0.0034, 0.0041] | 0.166 [0.155, 0.178] | 0.978 [0.971, 0.984] | 0.407 [0.350, 0.465] (10) | 0.603 [0.535, 0.683] | 0.208 [0.179, 0.235] | 0.302 [0.268, 0.336] |
| PLIP | reference (RV13) | 0.0026 [0.0023, 0.0029] | 0.0041 [0.0037, 0.0045] | 0.0055 [0.0050, 0.0060] | 0.460 [0.430, 0.493] | 0.990 [0.985, 0.994] | 0.059 [0.046, 0.074] (5) | 0.338 [0.262, 0.422] | 0.039 [0.032, 0.047] | 0.630 [0.589, 0.668] |
| DINOv2 | reference (RV13) | 0.0051 [0.0038, 0.0071] | 0.0038 [0.0031, 0.0049] | 0.0046 [0.0039, 0.0056] | 0.178 [0.167, 0.190] | 0.957 [0.945, 0.969] | 0.273 [0.227, 0.322] (5) | 0.363 [0.295, 0.440] | 0.133 [0.111, 0.159] | 0.356 [0.333, 0.376] |
| EXAONEPath (no Macenko) | reference (RV13, off-label) | 0.0038 [0.0034, 0.0043] | 0.0085 [0.0078, 0.0093] | 0.0085 [0.0077, 0.0093] | 0.378 [0.360, 0.396] | 0.995 [0.992, 0.998] | 0.114 [0.086, 0.146] (5) | 0.479 [0.395, 0.564] | 0.066 [0.052, 0.079] | 0.511 [0.455, 0.566] |
| CLIP ViT-B/32 | reference (RV15) | 0.0047 [0.0039, 0.0056] | 0.0076 [0.0067, 0.0087] | 0.0052 [0.0046, 0.0057] | 0.357 [0.329, 0.384] | 0.988 [0.984, 0.992] | 0.090 [0.069, 0.115] (10) | 0.288 [0.221, 0.361] | 0.058 [0.045, 0.072] | 0.585 [0.555, 0.612] |
| ResNet50 (avgpool) | reference (RV15, primary) | 0.0035 [0.0031, 0.0041] | 0.0039 [0.0035, 0.0043] | 0.0039 [0.0035, 0.0044] | 0.234 [0.220, 0.249] | 0.848 [0.830, 0.867] | 0.167 [0.131, 0.205] (5) | 0.232 [0.170, 0.298] | 0.091 [0.071, 0.110] | 0.454 [0.427, 0.479] |
| ResNet50 (layer3, CLAM) | reference (RV15, secondary) | 0.0027 [0.0023, 0.0031] | 0.0033 [0.0029, 0.0036] | 0.0037 [0.0033, 0.0041] | 0.210 [0.195, 0.225] | 0.944 [0.931, 0.957] | 0.121 [0.092, 0.153] (5) | 0.283 [0.209, 0.359] | 0.074 [0.058, 0.092] | 0.539 [0.503, 0.574] |

| Model | Between-tissue distance | Six-way probe | Colour share | Frequency-alone share | Frequency share | High (dose 0.50) | High / low–mid (0.25) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UNI | 0.878 [0.870, 0.885] | 0.994 [0.990, 0.998] | 0.158 [0.150, 0.168] | 0.041 [0.036, 0.045] | 0.002 [-0.004, 0.007] | 0.0239 [0.0222, 0.0257] | 1.27 [1.17, 1.38] |
| UNI2-h | 0.893 [0.885, 0.900] | 0.995 [0.989, 0.998] | 0.114 [0.107, 0.122] | 0.069 [0.064, 0.075] | 0.053 [0.049, 0.057] | 0.0274 [0.0252, 0.0299] | 1.73 [1.57, 1.91] |
| Virchow2 | 0.760 [0.738, 0.780] | 0.995 [0.993, 0.997] | 0.164 [0.151, 0.177] | 0.114 [0.106, 0.122] | 0.049 [0.036, 0.063] | 0.0049 [0.0045, 0.0053] | 0.66 [0.60, 0.72] |
| H-optimus-1 | 0.898 [0.890, 0.906] | 0.988 [0.979, 0.994] | 0.065 [0.059, 0.071] | 0.026 [0.021, 0.031] | -0.006 [-0.013, 0.000] | 0.0077 [0.0071, 0.0083] | 0.48 [0.43, 0.53] |
| EXAONEPath | 0.786 [0.775, 0.798] | 0.967 [0.951, 0.977] | 0.246 [0.227, 0.266] | 0.035 [0.032, 0.038] | 0.036 [0.031, 0.040] | 0.0347 [0.0316, 0.0382] | 1.26 [1.16, 1.36] |
| SEAL-UNI2 | 0.870 [0.862, 0.879] | 0.995 [0.989, 0.998] | 0.117 [0.109, 0.125] | 0.086 [0.081, 0.092] | 0.062 [0.057, 0.066] | 0.0272 [0.0248, 0.0298] | 1.98 [1.79, 2.20] |
| CONCH | 0.566 [0.548, 0.583] | 0.958 [0.946, 0.969] | 0.209 [0.198, 0.220] | 0.086 [0.079, 0.091] | 0.018 [0.009, 0.026] | 0.0142 [0.0130, 0.0155] | 2.42 [2.12, 2.74] |
| SEAL-CONCH | 0.247 [0.233, 0.260] | 0.963 [0.950, 0.974] | 0.274 [0.260, 0.288] | 0.144 [0.136, 0.153] | 0.050 [0.041, 0.060] | 0.0132 [0.0121, 0.0144] | 2.59 [2.29, 2.89] |
| PLIP | 0.205 [0.196, 0.214] | 0.989 [0.984, 0.993] | 0.376 [0.367, 0.384] | 0.096 [0.092, 0.100] | 0.042 [0.038, 0.045] | 0.0189 [0.0172, 0.0206] | 2.11 [1.94, 2.31] |
| DINOv2 | 0.121 [0.106, 0.138] | 0.948 [0.934, 0.961] | 0.171 [0.155, 0.187] | 0.085 [0.070, 0.098] | 0.002 [-0.014, 0.017] | 0.0151 [0.0133, 0.0171] | 0.90 [0.75, 1.11] |
| EXAONEPath (no Macenko) | 0.744 [0.731, 0.757] | 0.996 [0.994, 0.998] | 0.186 [0.178, 0.193] | 0.045 [0.042, 0.048] | 0.019 [0.015, 0.022] | 0.0314 [0.0288, 0.0341] | 2.23 [2.03, 2.46] |
| CLIP ViT-B/32 | 0.077 [0.070, 0.086] | 0.982 [0.975, 0.986] | 0.222 [0.209, 0.236] | 0.076 [0.070, 0.082] | 0.052 [0.044, 0.059] | 0.0183 [0.0164, 0.0201] | 1.10 [0.94, 1.27] |
| ResNet50 (avgpool) | 0.290 [0.274, 0.306] | 0.865 [0.847, 0.882] | 0.413 [0.396, 0.428] | 0.050 [0.041, 0.060] | 0.050 [0.043, 0.056] | 0.0140 [0.0125, 0.0154] | 1.11 [1.01, 1.22] |
| ResNet50 (layer3, CLAM) | 0.179 [0.169, 0.191] | 0.960 [0.948, 0.970] | 0.451 [0.430, 0.472] | 0.084 [0.077, 0.092] | 0.072 [0.065, 0.079] | 0.0133 [0.0119, 0.0148] | 1.37 [1.24, 1.49] |

## CLIP − PLIP (paired; pre-registered rule)

Δ = CLIP − PLIP, 95% CI from the shared slide resamples; m = ¼ |WSI median − PLIP| (fixed in the protocol); inherited share f = (W − CLIP) / (W − PLIP), W = median of the eight WSI models per resample.

| Endpoint | Role | CLIP | PLIP | Δ [95% CI] | m | Category | f [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| normalized_distance | primary | 0.357 | 0.460 | -0.103 [-0.127, -0.081] | 0.064 | PLIP worse | 0.60 [0.51, 0.68] |
| robustness_index | primary | 0.090 | 0.059 | +0.031 [+0.014, +0.051] | 0.085 | CLIP ≈ PLIP | 0.91 [0.85, 0.96] |
| os_rate | primary | 0.585 | 0.630 | -0.045 [-0.070, -0.019] | 0.080 | CLIP ≈ PLIP | 0.86 [0.79, 0.94] |
| so_rate | secondary | 0.058 | 0.039 | +0.018 [+0.007, +0.031] | 0.041 | CLIP ≈ PLIP | 0.89 [0.81, 0.96] |
| tissue_retrieval | secondary | 0.288 | 0.338 | -0.050 [-0.126, +0.027] | 0.072 | partial / inconclusive | 1.17 [0.92, 1.54] |

**Overall reading (pre-registered rule):** mixed: normalized_distance -> pathology fine-tuning added scanner organization; robustness_index -> inherited from natural-image initialization; os_rate -> inherited from natural-image initialization.

Band sensitivity, CLIP / PLIP ratio (similar if the CI lies within [0.8, 1.25]):

| Band, dose | CLIP | PLIP | Δ [95% CI] | Ratio [95% CI] | Category |
| --- | --- | --- | --- | --- | --- |
| low_mid_d0.25 | 0.0047 | 0.0026 | +0.0021 [+0.0015, +0.0028] | 1.82 [1.62, 2.05] | CLIP higher |
| mid_d0.25 | 0.0076 | 0.0041 | +0.0035 [+0.0028, +0.0043] | 1.85 [1.69, 2.02] | CLIP higher |
| high_d0.25 | 0.0052 | 0.0055 | -0.0003 [-0.0007, +0.0001] | 0.95 [0.88, 1.02] | similar |
| low_mid_d0.5 | 0.0174 | 0.0103 | +0.0071 [+0.0053, +0.0091] | 1.69 [1.54, 1.85] | CLIP higher |
| mid_d0.5 | 0.0286 | 0.0148 | +0.0139 [+0.0115, +0.0162] | 1.94 [1.79, 2.08] | CLIP higher |
| high_d0.5 | 0.0183 | 0.0189 | -0.0006 [-0.0018, +0.0006] | 0.97 [0.91, 1.03] | similar |

Other CLIP − PLIP differences (information):

- detectability_best: CLIP 0.988, PLIP 0.990, Δ -0.002 [-0.005, +0.002].
- detectability_linear: CLIP 0.987, PLIP 0.989, Δ -0.002 [-0.007, +0.003].
- scanner_probe_6way: CLIP 0.982, PLIP 0.989, Δ -0.007 [-0.012, -0.003].
- between_tissue_distance: CLIP 0.077, PLIP 0.205, Δ -0.127 [-0.134, -0.120].
- colour_share: CLIP 0.222, PLIP 0.376, Δ -0.154 [-0.165, -0.142].
- frequency_share: CLIP 0.052, PLIP 0.042, Δ +0.010 [+0.002, +0.018].
- high_over_low_mid_d0.25: CLIP 1.100, PLIP 2.112, Δ -1.011 [-1.138, -0.883].

## RI confound (point estimates vs the range of the eight WSI models)

| Model | RI | OS | Normalized distance | SO | Retrieval | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| PLIP | 0.059 | 0.630 | 0.460 | 0.039 | 0.338 | low RI read as scanner organization |
| DINOv2 | 0.273 | 0.356 | 0.178 | 0.133 | 0.363 | RI within or above the WSI range |
| CLIP ViT-B/32 | 0.090 | 0.585 | 0.357 | 0.058 | 0.288 | low RI read as scanner organization |
| ResNet50 (avgpool) | 0.167 | 0.454 | 0.234 | 0.091 | 0.232 | low RI read as weak tissue information |
| ResNet50 (layer3, CLAM) | 0.121 | 0.539 | 0.210 | 0.074 | 0.283 | low RI read as scanner organization |

## Position on the RV14 trend (OLS across the eight WSI models, high band, dose 0.25)

| Model | Outcome | Sensitivity | Observed | Trend prediction | Residual [95% CI] | Residual / SD | On trend | Extrapolated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PLIP | robustness_index | 0.0055 | 0.059 | 0.386 | -0.327 [-0.377, -0.279] | -6.3 | False | False |
| DINOv2 | robustness_index | 0.0046 | 0.273 | 0.412 | -0.139 [-0.188, -0.091] | -2.7 | False | False |
| EXAONEPath (no Macenko) | robustness_index | 0.0085 | 0.114 | 0.290 | -0.176 [-0.213, -0.143] | -3.4 | False | False |
| CLIP ViT-B/32 | robustness_index | 0.0052 | 0.090 | 0.395 | -0.305 [-0.353, -0.257] | -5.9 | False | False |
| ResNet50 (avgpool) | robustness_index | 0.0039 | 0.167 | 0.433 | -0.267 [-0.312, -0.221] | -5.2 | False | False |
| ResNet50 (layer3, CLAM) | robustness_index | 0.0037 | 0.121 | 0.442 | -0.322 [-0.367, -0.276] | -6.2 | False | False |
| PLIP | normalized_distance | 0.0055 | 0.460 | 0.238 | +0.222 [+0.202, +0.243] | +4.5 | False | False |
| DINOv2 | normalized_distance | 0.0046 | 0.178 | 0.209 | -0.031 [-0.055, -0.015] | -0.6 | True | False |
| EXAONEPath (no Macenko) | normalized_distance | 0.0085 | 0.378 | 0.343 | +0.035 [+0.016, +0.052] | +0.7 | True | False |
| CLIP ViT-B/32 | normalized_distance | 0.0052 | 0.357 | 0.228 | +0.129 [+0.110, +0.147] | +2.6 | False | False |
| ResNet50 (avgpool) | normalized_distance | 0.0039 | 0.234 | 0.186 | +0.048 [+0.040, +0.056] | +1.0 | True | False |
| ResNet50 (layer3, CLAM) | normalized_distance | 0.0037 | 0.210 | 0.176 | +0.035 [+0.026, +0.043] | +0.7 | True | False |

Mid and low–mid bands: `trend_residuals.csv`.

Exploratory (not a test): Spearman ρ across 12 models (eight WSI + PLIP, DINOv2, CLIP, ResNet50 avgpool); permutation p from 100,000 random permutations.

| Predictor | Outcome | ρ [95% CI] | p |
| --- | --- | --- | --- |
| normalized_shift_high_d0.25 | robustness_index | -0.51 [-0.62, -0.41] | 0.0939 |
| normalized_shift_high_d0.25 | normalized_distance | +0.83 [+0.77, +0.87] | 0.0014 |
| normalized_shift_mid_d0.25 | robustness_index | -0.66 [-0.73, -0.45] | 0.0215 |
| normalized_shift_mid_d0.25 | normalized_distance | +0.91 [+0.82, +0.92] | 0.0001 |
| normalized_shift_low_mid_d0.25 | robustness_index | -0.41 [-0.52, -0.29] | 0.1930 |
| normalized_shift_low_mid_d0.25 | normalized_distance | +0.57 [+0.51, +0.68] | 0.0554 |

Eight WSI models (RV14, unchanged): high vs RI ρ = -0.90; high vs normalized distance ρ = +0.90.

## ResNet50: layer3 (CLAM) − avgpool (paired, descriptive)

| Statistic | layer3 | avgpool | Δ [95% CI] |
| --- | --- | --- | --- |
| normalized_shift_low_mid_d0.25 | 0.0027 | 0.0035 | -0.0009 [-0.0011, -0.0007] |
| normalized_shift_mid_d0.25 | 0.0033 | 0.0039 | -0.0006 [-0.0007, -0.0005] |
| normalized_shift_high_d0.25 | 0.0037 | 0.0039 | -0.0003 [-0.0004, -0.0002] |
| normalized_distance | 0.210 | 0.234 | -0.024 [-0.031, -0.017] |
| robustness_index | 0.121 | 0.167 | -0.046 [-0.062, -0.030] |
| os_rate | 0.539 | 0.454 | +0.086 [+0.071, +0.100] |
| so_rate | 0.074 | 0.091 | -0.017 [-0.026, -0.007] |
| tissue_retrieval | 0.283 | 0.232 | +0.051 [-0.006, +0.104] |
| detectability_best | 0.944 | 0.848 | +0.096 [+0.082, +0.110] |
| colour_share | 0.451 | 0.413 | +0.038 [+0.029, +0.047] |
| frequency_share | 0.072 | 0.050 | +0.022 [+0.017, +0.027] |

## Files

- `endpoints.csv` (wide; estimate, `_lo`, `_hi`), `model_statistics.csv` (every RV13 statistic + SO/OS), `summary.json`.
- `contrasts.csv` (CLIP − PLIP with margins, categories, inherited share, ratios; ResNet50 layer3 − avgpool).
- `ri_confound.csv`, `trend_residuals.csv`, `trend_fits.csv`, `correlations_wsi8.csv`, `correlations_exploratory_12.csv`.
- `fig_sensitivity_robustness_rv15.{png,pdf}`: RV14 main figure with CLIP and ResNet50 (avgpool) as open reference markers; dashed line = OLS fit across the eight WSI models (descriptive).
- `embeddings/<stream>/<slide>.h5`, `render/`, `tasks/`, `qc/` (PLIP and band parity, RV14 reproduction), `parity/`.
