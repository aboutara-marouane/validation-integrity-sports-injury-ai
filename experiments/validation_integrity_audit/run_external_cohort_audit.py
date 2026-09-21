#!/usr/bin/env python3
"""Validation-integrity audit on the independent Wu et al. (2026) cohort.

This analysis does not claim to reproduce every tuning decision in the source
publication.  Instead, it holds three representative classifiers fixed and
changes only two validation-design choices:

1. whether weekly rows or inferred participants define the resampling unit;
2. whether supervised feature selection is performed globally or within each
   training fold.

The official supplementary workbooks do not contain a participant-ID column.
Participant trajectories are therefore reconstructed from five invariant
baseline fields.  The program refuses to run unless this fingerprint yields
exactly 142 unique, contiguous trajectories in both workbooks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

try:
    import xgboost
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    raise RuntimeError("xgboost is required") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.leakage_safe_validation.run_leakage_safe_validation import shuffled_group_folds


FINGERPRINT_COLUMNS = [
    "sex", "Age", "lower_limb_days_total", "average_run_hours",
    "average_interval_training_frequency",
]


@dataclass(frozen=True)
class Config:
    data_dir: str = "data/external/wu_2026"
    output_dir: str = "experiments/validation_integrity_audit/results/external_wu_2026"
    repeats: int = 3
    folds: int = 5
    selected_features: int = 20
    bootstrap_iterations: int = 2000
    seed: int = 20260830


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def infer_participants(frame: pd.DataFrame) -> pd.Series:
    missing = sorted(set(FINGERPRINT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Participant fingerprint fields missing: {missing}")
    groups = frame.groupby(FINGERPRINT_COLUMNS, sort=False, dropna=False).ngroup()
    runs = int(groups.ne(groups.shift()).sum())
    if groups.nunique() != 142 or runs != 142:
        raise ValueError(
            "Released-row participant reconstruction failed: "
            f"unique={groups.nunique()}, contiguous_runs={runs}; expected 142/142"
        )
    return groups.astype(int).rename("inferred_participant_id")


def model(name: str, prevalence: float, seed: int):
    if name == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(10, 90))),
            ("classifier", LogisticRegression(
                C=0.25, class_weight="balanced", solver="liblinear",
                max_iter=4000, random_state=seed,
            )),
        ])
    if name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=350, max_depth=10, min_samples_leaf=5,
                max_features="sqrt", class_weight="balanced_subsample",
                n_jobs=-1, random_state=seed,
            )),
        ])
    if name == "xgboost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", XGBClassifier(
                n_estimators=350, max_depth=3, learning_rate=0.03,
                min_child_weight=5.0, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=5.0, reg_alpha=0.1,
                scale_pos_weight=(1.0 - prevalence) / prevalence,
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", n_jobs=-1, random_state=seed,
            )),
        ])
    raise ValueError(name)


def calibration_statistics(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    z = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=4000).fit(z, y)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    membership = np.digitize(p, np.unique(edges)[1:-1], right=True)
    value = 0.0
    for group in np.unique(membership):
        mask = membership == group
        value += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(value)


def metric_row(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration_statistics(y, p)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece_10": expected_calibration_error(y, p),
    }


def select_columns(X_train: pd.DataFrame, y_train: pd.Series, k: int) -> list[str]:
    selector = SelectKBest(f_classif, k=min(k, X_train.shape[1])).fit(X_train, y_train)
    return X_train.columns[selector.get_support()].tolist()


def split_iterator(protocol: str, y: pd.Series, groups: pd.Series, folds: int, seed: int):
    if protocol.startswith("row_"):
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    return shuffled_group_folds(groups, folds, seed)


def run_protocol(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, dataset: str,
    protocol: str, model_name: str, config: Config,
    globally_selected: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_parts, fold_rows = [], []
    for repeat in range(config.repeats):
        seed = config.seed + repeat * 1009
        for fold, (train_idx, test_idx) in enumerate(
            split_iterator(protocol, y, groups, config.folds, seed)
        ):
            if protocol.endswith("global_selection"):
                columns = globally_selected
            elif protocol.endswith("local_selection"):
                columns = select_columns(X.iloc[train_idx], y.iloc[train_idx], config.selected_features)
            elif protocol.endswith("all_features"):
                columns = list(X.columns)
            else:  # pragma: no cover
                raise ValueError(protocol)
            fitted = clone(model(model_name, float(y.iloc[train_idx].mean()), seed + fold)).fit(
                X.iloc[train_idx][columns], y.iloc[train_idx]
            )
            probability = fitted.predict_proba(X.iloc[test_idx][columns])[:, 1]
            train_groups = set(groups.iloc[train_idx]); test_groups = set(groups.iloc[test_idx])
            fold_rows.append({
                "dataset": dataset, "protocol": protocol, "model": model_name,
                "repeat": repeat, "fold": fold, "n_train": len(train_idx),
                "n_test": len(test_idx), "n_features": len(columns),
                "train_prevalence": float(y.iloc[train_idx].mean()),
                "test_prevalence": float(y.iloc[test_idx].mean()),
                "participant_overlap": len(train_groups & test_groups),
                "selected_columns": json.dumps(columns),
                **metric_row(y.iloc[test_idx].to_numpy(), probability),
            })
            prediction_parts.append(pd.DataFrame({
                "dataset": dataset, "protocol": protocol, "model": model_name,
                "repeat": repeat, "fold": fold, "row_index": test_idx,
                "inferred_participant_id": groups.iloc[test_idx].to_numpy(),
                "y_true": y.iloc[test_idx].to_numpy(), "probability": probability,
            }))
        print(f"{dataset} {protocol} {model_name} repeat {repeat + 1}/{config.repeats}", flush=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    coverage = predictions.groupby(["repeat", "row_index"]).size()
    if not coverage.eq(1).all() or len(coverage) != config.repeats * len(y):
        raise RuntimeError(f"Incomplete or duplicated OOF coverage for {protocol}/{model_name}")
    if protocol.startswith("group_") and not folds.participant_overlap.eq(0).all():
        raise RuntimeError(f"Participant leakage detected in {protocol}/{model_name}")
    return predictions, folds


def participant_bootstrap(
    predictions: pd.DataFrame, iterations: int, seed: int,
) -> pd.DataFrame:
    averaged = predictions.groupby(
        ["row_index", "inferred_participant_id", "y_true"], as_index=False
    ).probability.mean()
    participants = averaged.inferred_participant_id.unique()
    rng = np.random.default_rng(seed)
    rows = []
    point = metric_row(averaged.y_true.to_numpy(), averaged.probability.to_numpy())
    for iteration in range(iterations):
        sampled = rng.choice(participants, len(participants), replace=True)
        pieces = [averaged[averaged.inferred_participant_id.eq(g)] for g in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        if boot.y_true.nunique() < 2:
            continue
        result = metric_row(boot.y_true.to_numpy(), boot.probability.to_numpy())
        rows.append({"iteration": iteration, **result})
    boot = pd.DataFrame(rows)
    summary = []
    for metric, estimate in point.items():
        summary.append({
            "metric": metric, "estimate": estimate,
            "ci_low": float(boot[metric].quantile(0.025)),
            "ci_high": float(boot[metric].quantile(0.975)),
            "bootstrap_valid": len(boot),
        })
    return pd.DataFrame(summary)


def paired_protocol_contrasts(predictions: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    """Paired participant-bootstrap contrasts after repeat-averaging by row."""
    comparisons = [
        ("row_local_selection", "group_local_selection", "row_minus_group_local"),
        ("group_global_selection", "group_local_selection", "global_minus_local_with_grouping"),
        ("row_global_selection", "group_local_selection", "combined_optimism"),
        ("group_local_selection", "group_all_features", "local_selection_minus_all_features"),
    ]
    averaged = predictions.groupby(
        ["dataset", "model", "protocol", "row_index", "inferred_participant_id", "y_true"],
        as_index=False,
    ).probability.mean()
    rows = []
    for condition_index, ((dataset, model_name), condition) in enumerate(averaged.groupby(["dataset", "model"])):
        participants = condition.inferred_participant_id.unique()
        rng = np.random.default_rng(seed + condition_index * 1009)
        indexed = {
            protocol: group.set_index("row_index").sort_index()
            for protocol, group in condition.groupby("protocol")
        }
        for first, second, contrast_name in comparisons:
            left, right = indexed[first], indexed[second]
            if not left.index.equals(right.index) or not np.array_equal(left.y_true, right.y_true):
                raise RuntimeError(f"Unpaired protocol predictions for {dataset}/{model_name}")
            y_array = left.y_true.to_numpy()
            left_probability = left.probability.to_numpy()
            right_probability = right.probability.to_numpy()
            group_array = left.inferred_participant_id.to_numpy()
            group_indices = {g: np.flatnonzero(group_array == g) for g in participants}
            def contrast_metrics(y, p):
                return {
                    "roc_auc": float(roc_auc_score(y, p)),
                    "average_precision": float(average_precision_score(y, p)),
                    "brier": float(brier_score_loss(y, p)),
                }
            point_left = contrast_metrics(y_array, left_probability)
            point_right = contrast_metrics(y_array, right_probability)
            boot = {metric: [] for metric in point_left}
            for _ in range(iterations):
                sampled = rng.choice(participants, len(participants), replace=True)
                sample_indices = np.concatenate([group_indices[g] for g in sampled])
                sample_y = y_array[sample_indices]
                if np.unique(sample_y).size < 2:
                    continue
                left_metrics = contrast_metrics(sample_y, left_probability[sample_indices])
                right_metrics = contrast_metrics(sample_y, right_probability[sample_indices])
                for metric in boot:
                    boot[metric].append(left_metrics[metric] - right_metrics[metric])
            for metric, values in boot.items():
                values = np.asarray(values)
                rows.append({
                    "dataset": dataset, "model": model_name, "contrast": contrast_name,
                    "first_protocol": first, "second_protocol": second, "metric": metric,
                    "difference": point_left[metric] - point_right[metric],
                    "ci_low": float(np.quantile(values, .025)),
                    "ci_high": float(np.quantile(values, .975)),
                    "bootstrap_valid": len(values),
                })
    return pd.DataFrame(rows)


def plot_results(summary: pd.DataFrame, out: Path):
    labels = {
        "row_global_selection": "Row CV + global selection",
        "row_local_selection": "Row CV + fold-local selection",
        "group_global_selection": "Grouped CV + global selection",
        "group_local_selection": "Grouped CV + fold-local selection",
        "group_all_features": "Grouped CV + all features",
    }
    metrics = [("roc_auc", "ROC AUC"), ("average_precision", "Average precision"), ("brier", "Brier score")]
    for dataset in summary.dataset.unique():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        subset = summary[summary.dataset.eq(dataset)]
        for ax, (metric, title) in zip(axes, metrics):
            part = subset[subset.metric.eq(metric)].copy()
            part["label"] = part.protocol.map(labels)
            for idx, model_name in enumerate(["logistic", "random_forest", "xgboost"]):
                m = part[part.model.eq(model_name)]
                x = np.arange(len(labels)) + (idx - 1) * 0.22
                ordered = m.set_index("protocol").reindex(labels).reset_index()
                ax.errorbar(
                    x, ordered.estimate,
                    yerr=[ordered.estimate - ordered.ci_low, ordered.ci_high - ordered.estimate],
                    marker="o", capsize=3, linewidth=1.4, label=model_name.replace("_", " "),
                )
            ax.set_xticks(np.arange(len(labels)), [labels[p] for p in labels], rotation=35, ha="right")
            ax.set_title(title); ax.grid(alpha=0.25)
        axes[0].legend(frameon=False)
        fig.suptitle(f"Validation-design audit: {dataset.replace('_', ' ')}")
        fig.tight_layout()
        fig.savefig(out / f"validation_audit_{dataset}.pdf", bbox_inches="tight")
        fig.savefig(out / f"validation_audit_{dataset}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=Config.data_dir)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--repeats", type=int, default=Config.repeats)
    parser.add_argument("--folds", type=int, default=Config.folds)
    parser.add_argument("--selected-features", type=int, default=Config.selected_features)
    parser.add_argument("--bootstrap-iterations", type=int, default=Config.bootstrap_iterations)
    parser.add_argument("--seed", type=int, default=Config.seed)
    args = parser.parse_args()
    config = Config(**vars(args))
    data_dir = PROJECT_ROOT / config.data_dir
    out = PROJECT_ROOT / config.output_dir
    out.mkdir(parents=True, exist_ok=True)

    all_predictions, all_folds, all_summaries, provenance = [], [], [], []
    fingerprints = None
    for dataset, filename in [
        ("class1_features", "class1_features.xlsx"),
        ("all_features", "all_features.xlsx"),
    ]:
        path = data_dir / filename
        frame = pd.read_excel(path)
        groups = infer_participants(frame)
        if fingerprints is None:
            fingerprints = groups.to_numpy()
        elif not np.array_equal(fingerprints, groups.to_numpy()):
            raise ValueError("Participant reconstruction differs between official workbooks")
        y = frame["RRI"].astype(int)
        X = frame.drop(columns=["RRI"]).astype(float)
        global_columns = select_columns(X, y, config.selected_features)
        provenance.append({
            "dataset": dataset, "file": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256(path), "rows": len(frame), "features": X.shape[1],
            "positives": int(y.sum()), "participants": int(groups.nunique()),
            "contiguous_participant_runs": int(groups.ne(groups.shift()).sum()),
            "global_selected_columns": json.dumps(global_columns),
        })
        protocols = [
            "row_global_selection", "row_local_selection",
            "group_global_selection", "group_local_selection", "group_all_features",
        ]
        for protocol in protocols:
            for model_name in ["logistic", "random_forest", "xgboost"]:
                predictions, folds = run_protocol(
                    X, y, groups, dataset, protocol, model_name, config, global_columns
                )
                all_predictions.append(predictions); all_folds.append(folds)
                ci = participant_bootstrap(
                    predictions, config.bootstrap_iterations,
                    config.seed + len(all_summaries) * 131,
                )
                ci.insert(0, "model", model_name); ci.insert(0, "protocol", protocol)
                ci.insert(0, "dataset", dataset); all_summaries.append(ci)

    predictions = pd.concat(all_predictions, ignore_index=True)
    folds = pd.concat(all_folds, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)
    predictions.to_csv(out / "external_oof_predictions.csv.gz", index=False, compression="gzip")
    folds.to_csv(out / "external_fold_metrics.csv", index=False)
    summary.to_csv(out / "external_cluster_bootstrap_summary.csv", index=False)
    paired_protocol_contrasts(
        predictions, config.bootstrap_iterations, config.seed + 70000
    ).to_csv(out / "external_paired_protocol_contrasts.csv", index=False)
    pd.DataFrame(provenance).to_csv(out / "external_data_provenance.csv", index=False)
    plot_results(summary, out)
    run = {
        "config": asdict(config), "python": platform.python_version(),
        "pandas": pd.__version__, "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "participant_id_status": "inferred from five invariant baseline fields",
    }
    (out / "run_manifest.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"Wrote audit artifacts to {out}")


if __name__ == "__main__":
    main()
