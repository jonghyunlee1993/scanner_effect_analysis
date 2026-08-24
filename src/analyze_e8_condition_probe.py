"""Scanner probe over any set of feature conditions, not just the RF1U ones.

The locked RF1U probe (`analyze_rf1u_scanner_probe.py`) hardcodes its conditions:
raw, the three RF1U targets and the two E5 feature-space comparators. Every new
arm -- the E4 controls, a lambda-swept destination, Vahadane, a learned residual
model trained in a separate session -- would otherwise need its own copy.

Every feature shard in this project already shares one layout:

    <root>/<encoder_id>/shards/<slide_id>.h5
        condition  (C,)            utf-8 names
        features   (C, 6, 100, D)  scanner-major, then location, then dimension

so a probe only needs the list of roots. That is the whole contract, and it is
what `docs/e8_condition_registry.md` asks a new arm to satisfy.

Three estimators are reported per condition because the linear probe alone
cannot separate "the scanner is gone" from "the scanner is no longer linearly
decodable". A feature-space affine that whitens per-scanner second moments
removes exactly what a linear probe reads, so a linear-only answer would be
close to circular. The MLP and the cosine k-NN see directions and neighbourhood
structure the linear probe cannot.

All three share the frozen splits, location stride and seed of the locked probe
so the numbers sit in the same table as the ones already reported. Comparability
is the only goal here, so none of the three is tuned.

An invariance-only endpoint is degenerate by itself: a constant representation
reaches chance while destroying every patch. Read this beside the frozen content
and collapse verdicts, never alone.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from e4_primary_metrics import l2_normalize
from e5_comparator_population import SCANNERS
from e5_reinhard_residual_frequency import RF1_FOLDS, fold_assignments
from fetch_e0_pfm_checkpoints import sha256


LOCATION_STRIDE = 4
PROBE_SEED = 20260803

# Frozen before any condition was scored (F3): the MLP exists to test whether a
# non-linear direction survives, not to maximise accuracy, so its width, depth,
# schedule and stopping rule are fixed for every condition and every model.
MLP_HIDDEN = 256
MLP_DROPOUT = 0.1
MLP_EPOCHS = 200
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_BATCH = 512
KNN_K = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--encoder-index", type=int, required=True)
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="LABEL=ROOT",
        help="feature root holding <encoder_id>/shards/<slide>.h5; repeatable",
    )
    parser.add_argument(
        "--probe",
        default="linear,mlp,knn",
        help="comma-separated subset of linear, mlp, knn",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="label-permutation replicates for the linear probe (0 disables)",
    )
    parser.add_argument("--location-stride", type=int, default=LOCATION_STRIDE)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_sources(values: list[str]) -> dict[str, Path]:
    sources = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--source expects LABEL=ROOT, got {value!r}")
        label, root = value.split("=", 1)
        if label in sources:
            raise ValueError(f"duplicate source label {label!r}")
        sources[label] = Path(root)
    return sources


def fold_scores(fit_predict, features, groups, folds):
    """Balanced accuracy per frozen slide fold, plus a pooled confusion matrix.

    `fit_predict` takes (train_x, train_y, test_x) and returns test predictions,
    so the three estimators differ only in that callable.
    """
    scores, matrix = [], np.zeros((len(SCANNERS), len(SCANNERS)), dtype=np.int64)
    for fold in range(RF1_FOLDS):
        test = folds == fold
        train = ~test
        if not test.any() or not train.any():
            raise ValueError(f"fold {fold} is empty")
        predicted = fit_predict(features[train], groups[train], features[test])
        scores.append(float(balanced_accuracy_score(groups[test], predicted)))
        matrix += confusion_matrix(
            groups[test], predicted, labels=np.arange(len(SCANNERS))
        )
    return scores, matrix


def linear_fit_predict(train_x, train_y, test_x):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=1.0,
            random_state=PROBE_SEED,
        ),
    )
    model.fit(train_x, train_y)
    return model.predict(test_x)


def mlp_fit_predict(train_x, train_y, test_x):
    """One hidden layer, class-balanced, fixed schedule, no early stopping.

    Standardisation uses training-fold moments only, matching the linear probe.
    """
    torch.manual_seed(PROBE_SEED)
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    x = torch.from_numpy(((train_x - mean) / scale).astype(np.float32))
    y = torch.from_numpy(train_y.astype(np.int64))
    counts = np.bincount(train_y, minlength=len(SCANNERS)).astype(np.float64)
    weight = torch.from_numpy((counts.sum() / np.maximum(counts, 1)).astype(np.float32))

    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], MLP_HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Dropout(MLP_DROPOUT),
        torch.nn.Linear(MLP_HIDDEN, len(SCANNERS)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    generator = torch.Generator().manual_seed(PROBE_SEED)
    model.train()
    for _ in range(MLP_EPOCHS):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(order), MLP_BATCH):
            batch = order[start : start + MLP_BATCH]
            optimizer.zero_grad()
            loss_fn(model(x[batch]), y[batch]).backward()
            optimizer.step()

    model.eval()
    test = torch.from_numpy((((test_x - mean) / scale)).astype(np.float32))
    with torch.no_grad():
        return model(test).argmax(dim=1).numpy()


def knn_fit_predict(train_x, train_y, test_x):
    """Cosine k-NN vote. Features arrive L2-normalised, so cosine is a dot product."""
    train = torch.from_numpy(train_x.astype(np.float32))
    train = train / train.norm(dim=1, keepdim=True).clamp_min(1e-12)
    labels = torch.from_numpy(train_y.astype(np.int64))
    predictions = []
    query_all = torch.from_numpy(test_x.astype(np.float32))
    query_all = query_all / query_all.norm(dim=1, keepdim=True).clamp_min(1e-12)
    for start in range(0, len(query_all), 1024):
        query = query_all[start : start + 1024]
        neighbours = (query @ train.T).topk(KNN_K, dim=1).indices
        votes = labels[neighbours]
        counts = torch.zeros(len(query), len(SCANNERS), dtype=torch.int64)
        counts.scatter_add_(1, votes, torch.ones_like(votes))
        predictions.append(counts.argmax(dim=1).numpy())
    return np.concatenate(predictions)


PROBES = {"linear": linear_fit_predict, "mlp": mlp_fit_predict, "knn": knn_fit_predict}


def read_shard(path: Path, positions, feature_dim: int):
    with h5py.File(path, "r") as source:
        names = [value.decode() for value in source["condition"][:]]
        values = np.asarray(source["features"][:], dtype=np.float32)
    if values.ndim != 4 or values.shape[1:3] != (len(SCANNERS), 100):
        raise ValueError(f"{path}: expected (C, 6, 100, D), got {values.shape}")
    if values.shape[3] != feature_dim:
        raise ValueError(f"{path}: feature dim {values.shape[3]} != {feature_dim}")
    return names, [
        l2_normalize(values[index][:, positions]).reshape(-1, feature_dim)
        for index in range(len(names))
    ]


def main():
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    model = contract["models"][args.encoder_index]
    model_id = model["encoder_id"]
    feature_dim = int(model["feature_dim"])
    positions = list(range(0, 100, args.location_stride))
    probes = [name.strip() for name in args.probe.split(",") if name.strip()]
    for name in probes:
        if name not in PROBES:
            raise ValueError(f"unknown probe {name!r}")
    sources = parse_sources(args.source)

    raw_paths = sorted((Path(args.raw) / model_id / "shards").glob("*.h5"))
    slide_ids = [path.stem for path in raw_paths]
    if len(slide_ids) != 109:
        raise ValueError(f"{model_id}: expected 109 raw shards, got {len(slide_ids)}")
    assignments = fold_assignments(slide_ids)

    stacks: dict[str, list] = {"raw": []}
    origin = {"raw": "raw"}
    groups, folds = [], []
    for path in raw_paths:
        slide_id = path.stem
        with h5py.File(path, "r") as source:
            raw = np.asarray(source["features"][:], dtype=np.float32)
        if raw.shape != (len(SCANNERS), 100, feature_dim):
            raise ValueError(f"{model_id}/{slide_id}: unexpected raw feature shape")
        stacks["raw"].append(l2_normalize(raw[:, positions]).reshape(-1, feature_dim))

        for label, root in sources.items():
            names, blocks = read_shard(
                root / model_id / "shards" / f"{slide_id}.h5", positions, feature_dim
            )
            for name, block in zip(names, blocks):
                key = f"{label}:{name}"
                stacks.setdefault(key, []).append(block)
                origin[key] = label

        groups.append(np.repeat(np.arange(len(SCANNERS)), len(positions)).astype(np.int64))
        folds.append(np.full(len(SCANNERS) * len(positions), assignments[slide_id]))

    groups = np.concatenate(groups)
    folds = np.concatenate(folds)
    expected = len(slide_ids)
    for key, parts in stacks.items():
        if len(parts) != expected:
            raise RuntimeError(f"{key}: {len(parts)} slides, expected {expected}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, matrices = [], []
    rng = np.random.default_rng(PROBE_SEED)

    for condition, parts in stacks.items():
        features = np.concatenate(parts).astype(np.float64)
        for probe in probes:
            scores, matrix = fold_scores(PROBES[probe], features, groups, folds)
            rows.append(
                {
                    "encoder_id": model_id,
                    "source": origin[condition],
                    "condition": condition,
                    "probe": probe,
                    "samples": int(len(features)),
                    "balanced_accuracy": float(np.mean(scores)),
                    "fold_min": float(np.min(scores)),
                    "fold_max": float(np.max(scores)),
                    **{f"fold_{index}": value for index, value in enumerate(scores)},
                }
            )
            normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
            for true_index, true_name in enumerate(SCANNERS):
                for pred_index, pred_name in enumerate(SCANNERS):
                    matrices.append(
                        {
                            "encoder_id": model_id,
                            "condition": condition,
                            "probe": probe,
                            "true_scanner": true_name,
                            "predicted_scanner": pred_name,
                            "count": int(matrix[true_index, pred_index]),
                            "row_fraction": float(normalized[true_index, pred_index]),
                        }
                    )
            print(
                f"{model_id} {probe:6s} {condition:44s} "
                f"BACC {np.mean(scores):.4f} [{np.min(scores):.4f}, {np.max(scores):.4f}]",
                flush=True,
            )

        if args.permutations and "linear" in probes:
            # A systematically sub-chance probe is a measurement warning, not a
            # result; the permutation null says what "no signal" actually scores
            # under these folds and this class balance.
            null = []
            for _ in range(args.permutations):
                shuffled = groups.copy()
                for fold in range(RF1_FOLDS):
                    mask = folds == fold
                    shuffled[mask] = rng.permutation(shuffled[mask])
                scores, _ = fold_scores(linear_fit_predict, features, shuffled, folds)
                null.append(float(np.mean(scores)))
            rows.append(
                {
                    "encoder_id": model_id,
                    "source": origin[condition],
                    "condition": condition,
                    "probe": "linear_permutation_null",
                    "samples": int(len(features)),
                    "balanced_accuracy": float(np.mean(null)),
                    "fold_min": float(np.min(null)),
                    "fold_max": float(np.max(null)),
                }
            )
            print(
                f"{model_id} null   {condition:44s} "
                f"BACC {np.mean(null):.4f} over {args.permutations} permutations",
                flush=True,
            )

    table = output / f"{model_id}.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted({key for row in rows for key in row})
        )
        writer.writeheader()
        writer.writerows(rows)
    matrix_table = output / f"{model_id}.confusion.csv"
    with matrix_table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrices[0]))
        writer.writeheader()
        writer.writerows(matrices)

    summary = {
        "analysis": "e8_condition_probe",
        "status": "SECONDARY_INVARIANCE_ONLY_ENDPOINT",
        "caveat": (
            "An invariance-only endpoint is degenerate on its own: a constant "
            "representation reaches chance while destroying all content. Read "
            "this next to the frozen content and collapse verdicts, never alone."
        ),
        "encoder_id": model_id,
        "sources": {label: str(root) for label, root in sources.items()},
        "conditions": list(stacks),
        "probes": probes,
        "estimators": {
            "linear": "StandardScaler + multinomial logistic regression, balanced, C=1",
            "mlp": (
                f"train-fold standardisation + {MLP_HIDDEN}-unit ReLU MLP, dropout "
                f"{MLP_DROPOUT}, AdamW lr {MLP_LR} wd {MLP_WEIGHT_DECAY}, "
                f"{MLP_EPOCHS} epochs, class-weighted cross-entropy"
            ),
            "knn": f"cosine k-NN vote, k={KNN_K}, on L2-normalised features",
        },
        "splits": "frozen RF1 physical-slide folds",
        "chance_balanced_accuracy": 1.0 / len(SCANNERS),
        "locations_per_slide": len(positions),
        "location_stride": args.location_stride,
        "permutations": args.permutations,
        "seed": PROBE_SEED,
        "artifacts": {table.name: sha256(table), matrix_table.name: sha256(matrix_table)},
    }
    (output / f"{model_id}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
