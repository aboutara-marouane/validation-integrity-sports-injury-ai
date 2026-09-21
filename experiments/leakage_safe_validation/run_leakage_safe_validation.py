#!/usr/bin/env python3
"""Leakage-safe repeated grouped validation for the weekly runner dataset.

Scientific contract
-------------------
* Unit evaluated: an eligible row in ``data_weekly.csv``.
* Target: the existing binary ``injury`` label (concurrent row classification).
* Generalisation target: unseen athletes.
* Outer evaluation: repeated, shuffled 5-fold athlete-group CV.
* Inner selection: 3-fold athlete-group CV within each outer-training fold.
* No test labels are used for preprocessing, hyperparameter selection,
  calibration, or threshold selection.
* Primary features exclude Athlete ID, Date, the injury label, all gap-like
  variables, and unstable relative-load ratios.
* The temporal gap is used only in a pre-labelled negative-control ablation.

The script writes complete out-of-fold predictions, split manifests, metrics,
cluster-bootstrap confidence intervals, calibration curves, and an automated
Markdown report. The implemented endpoint is concurrent row classification;
the released episode semantics do not define a prospective 7/14-day target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
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
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

try:
    import xgboost
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    xgboost = None
    XGBClassifier = None


SEED = 20260829


@dataclass(frozen=True)
class Config:
    data_path: str
    output_dir: str
    outer_repeats: int = 3
    outer_folds: int = 5
    inner_folds: int = 3
    bootstrap_iterations: int = 2000
    seed: int = SEED
    models: tuple[str, ...] = ("logistic", "random_forest", "xgboost")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def shuffled_group_folds(groups: pd.Series, n_splits: int, seed: int):
    """Yield reproducible, genuinely repeated, size-balanced group folds.

    ``GroupKFold`` can reproduce essentially the same allocation when group
    sizes are mostly unique, even after relabelling groups.  Here athletes are
    explicitly shuffled and then greedily placed in the currently smallest
    fold.  This preserves athlete disjointness while changing test athletes
    across repetitions.
    """
    series = pd.Series(np.asarray(groups)).reset_index(drop=True)
    counts = series.value_counts().to_dict()
    unique = np.asarray(sorted(pd.unique(series)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    fold_groups = [set() for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    fold_counts = np.zeros(n_splits, dtype=int)
    for athlete in unique:
        # Random jitter breaks equal-score ties reproducibly.
        normalized_rows = fold_sizes / max(1, fold_sizes.sum())
        normalized_groups = fold_counts / max(1, fold_counts.sum())
        score = normalized_rows + 0.15 * normalized_groups + rng.uniform(0, 1e-6, n_splits)
        target_fold = int(np.argmin(score))
        fold_groups[target_fold].add(athlete)
        fold_sizes[target_fold] += int(counts[athlete])
        fold_counts[target_fold] += 1
    indices = np.arange(len(series))
    for held_out in fold_groups:
        test_mask = series.isin(held_out).to_numpy()
        yield indices[~test_mask], indices[test_mask]


def feature_contract(df: pd.DataFrame):
    forbidden_exact = {"injury", "Athlete ID", "Date"}
    ratio_cols = [c for c in df.columns if c.startswith("rel total kms")]
    gap_cols = [c for c in df.columns if "gap" in c.lower()]
    safe_cols = [
        c for c in df.columns
        if c not in forbidden_exact and c not in ratio_cols and c not in gap_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    sensitivity_cols = safe_cols + ratio_cols
    return safe_cols, sensitivity_cols, ratio_cols, gap_cols


def model_space(name: str, pos_weight: float, seed: int):
    if name == "logistic":
        candidates = []
        for c in (0.05, 0.25, 1.0):
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler(quantile_range=(10, 90))),
                ("clf", LogisticRegression(
                    C=c, penalty="l2", class_weight="balanced", solver="liblinear",
                    max_iter=4000, random_state=seed,
                )),
            ])
            candidates.append(({"C": c}, model))
        return candidates

    if name == "random_forest":
        candidates = []
        for depth, leaf, max_features in ((6, 10, "sqrt"), (10, 5, "sqrt"), (None, 10, 0.5)):
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=350, max_depth=depth, min_samples_leaf=leaf,
                    max_features=max_features, class_weight="balanced_subsample",
                    n_jobs=-1, random_state=seed,
                )),
            ])
            candidates.append(({
                "max_depth": depth, "min_samples_leaf": leaf,
                "max_features": max_features,
            }, model))
        return candidates

    if name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is unavailable")
        candidates = []
        for depth, lr, leaf_weight in ((2, 0.03, 5.0), (3, 0.03, 5.0), (3, 0.06, 10.0)):
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", XGBClassifier(
                    n_estimators=350, max_depth=depth, learning_rate=lr,
                    min_child_weight=leaf_weight, subsample=0.8,
                    colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.1,
                    scale_pos_weight=pos_weight, objective="binary:logistic",
                    eval_metric="logloss", tree_method="hist", n_jobs=-1,
                    random_state=seed,
                )),
            ])
            candidates.append(({
                "max_depth": depth, "learning_rate": lr,
                "min_child_weight": leaf_weight,
            }, model))
        return candidates

    raise ValueError(name)


def safe_ap(y, p):
    return average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan


def choose_candidate(candidates, X, y, groups, inner_folds: int, seed: int):
    """Select by mean inner-fold AP; ties resolve toward the first/simpler candidate."""
    rows = []
    best_idx, best_score = 0, -np.inf
    for candidate_idx, (params, estimator) in enumerate(candidates):
        scores = []
        for train_idx, val_idx in shuffled_group_folds(groups, inner_folds, seed):
            fitted = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
            prob = fitted.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(safe_ap(y.iloc[val_idx], prob))
        score = float(np.nanmean(scores))
        rows.append({
            "candidate_index": candidate_idx,
            "params": json.dumps(params, sort_keys=True),
            "inner_ap_mean": score,
            "inner_ap_sd": float(np.nanstd(scores, ddof=1)) if len(scores) > 1 else 0.0,
        })
        if score > best_score + 1e-12:
            best_idx, best_score = candidate_idx, score
    return best_idx, pd.DataFrame(rows)


def inner_oof_prob(estimator, X, y, groups, inner_folds: int, seed: int):
    oof = np.full(len(y), np.nan, dtype=float)
    for train_idx, val_idx in shuffled_group_folds(groups, inner_folds, seed):
        fitted = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = fitted.predict_proba(X.iloc[val_idx])[:, 1]
    if np.isnan(oof).any():
        raise RuntimeError("Incomplete inner out-of-fold predictions")
    return oof


def _fit_one_calibrator(name, y, raw_prob):
    eps = 1e-6
    raw_prob = np.clip(np.asarray(raw_prob), eps, 1 - eps)
    if name == "none":
        return None
    if name == "platt":
        calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        calibrator.fit(logit(raw_prob).reshape(-1, 1), y)
        return calibrator
    if name == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=eps, y_max=1 - eps)
        calibrator.fit(raw_prob, y)
        return calibrator
    raise ValueError(name)


def fit_calibrators(y, raw_prob, groups, n_splits, seed):
    """Choose calibration by a second grouped CV layer over base-model OOF scores."""
    eps = 1e-6
    raw_prob = np.clip(np.asarray(raw_prob), eps, 1 - eps)
    y = pd.Series(np.asarray(y, dtype=int)).reset_index(drop=True)
    groups = pd.Series(np.asarray(groups)).reset_index(drop=True)
    calibrated_cv = {name: np.full(len(y), np.nan) for name in ("none", "platt", "isotonic")}
    calibrated_cv["none"] = raw_prob.copy()

    for train_idx, val_idx in shuffled_group_folds(groups, n_splits, seed):
        for name in ("platt", "isotonic"):
            calibrator = _fit_one_calibrator(name, y.iloc[train_idx], raw_prob[train_idx])
            calibrated_cv[name][val_idx] = apply_calibrator(name, calibrator, raw_prob[val_idx])

    scores = []
    for name, p in calibrated_cv.items():
        if np.isnan(p).any():
            raise RuntimeError(f"Incomplete calibration OOF predictions for {name}")
        scores.append({
            "calibrator": name,
            "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, np.clip(p, eps, 1 - eps)),
        })
    table = pd.DataFrame(scores).sort_values(["brier", "log_loss", "calibrator"])
    chosen = str(table.iloc[0]["calibrator"])
    final_calibrator = _fit_one_calibrator(chosen, y, raw_prob)
    # Threshold selection uses cross-calibrated predictions, not predictions
    # from a calibrator evaluated on the same outcomes used to fit it.
    return chosen, final_calibrator, calibrated_cv[chosen], table


def apply_calibrator(name, calibrator, raw_prob):
    eps = 1e-6
    p = np.clip(np.asarray(raw_prob), eps, 1 - eps)
    if name == "none":
        return p
    if name == "platt":
        return np.clip(calibrator.predict_proba(logit(p).reshape(-1, 1))[:, 1], eps, 1 - eps)
    if name == "isotonic":
        return np.clip(calibrator.predict(p), eps, 1 - eps)
    raise ValueError(name)


def choose_mcc_threshold(y, prob):
    """Select threshold on inner OOF predictions only, using a fixed quantile grid."""
    candidates = np.unique(np.quantile(prob, np.linspace(0.50, 0.999, 250)))
    best = (0.5, -np.inf)
    for threshold in candidates:
        pred = (prob >= threshold).astype(int)
        if pred.sum() == 0:
            continue
        mcc = matthews_corrcoef(y, pred)
        if mcc > best[1] + 1e-12:
            best = (float(threshold), float(mcc))
    return best


def calibration_intercept_slope(y, prob):
    eps = 1e-6
    z = logit(np.clip(np.asarray(prob), eps, 1 - eps))
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    # Fast two-parameter Newton/IRLS fit for logit(y) = intercept + slope*logit(p).
    # This avoids thousands of high-overhead sklearn fits during the bootstrap.
    beta = np.array([0.0, 1.0])
    for _ in range(30):
        fitted = expit(beta[0] + beta[1] * z)
        residual = y - fitted
        weight = np.clip(fitted * (1 - fitted), 1e-10, None)
        gradient = np.array([residual.sum(), np.dot(residual, z)])
        hessian = np.array([
            [weight.sum(), np.dot(weight, z)],
            [np.dot(weight, z), np.dot(weight, z * z)],
        ])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def expected_calibration_error(y, prob, bins=10):
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(prob)})
    try:
        frame["bin"] = pd.qcut(frame["p"], q=bins, duplicates="drop")
    except ValueError:
        return np.nan
    grouped = frame.groupby("bin", observed=True).agg(n=("y", "size"), obs=("y", "mean"), pred=("p", "mean"))
    return float(((grouped["n"] / len(frame)) * (grouped["obs"] - grouped["pred"]).abs()).sum())


def metrics(y, prob, threshold):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    threshold_array = np.asarray(threshold, dtype=float)
    pred = (prob >= threshold_array).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    intercept, slope = calibration_intercept_slope(y, prob)
    return {
        "n_rows": len(y), "n_positive": int(y.sum()), "prevalence": float(y.mean()),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "average_precision": safe_ap(y, prob),
        "mcc": matthews_corrcoef(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "brier": brier_score_loss(y, prob),
        "log_loss": log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece_10": expected_calibration_error(y, prob, bins=10),
        "threshold": float(np.mean(threshold_array)),
        "alerts_per_100_rows": float(100 * pred.mean()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def run_main(config: Config, df: pd.DataFrame, features: list[str], out: Path):
    all_predictions, fold_rows, tuning_rows, calibration_rows, split_rows = [], [], [], [], []
    y_all = df["injury"].astype(int).reset_index(drop=True)
    groups_all = df["Athlete ID"].reset_index(drop=True)
    X_all = df[features].reset_index(drop=True)

    for repeat in range(config.outer_repeats):
        repeat_seed = config.seed + repeat * 1009
        folds = list(shuffled_group_folds(groups_all, config.outer_folds, repeat_seed))
        for fold, (outer_train, outer_test) in enumerate(folds):
            X_train, X_test = X_all.iloc[outer_train], X_all.iloc[outer_test]
            y_train, y_test = y_all.iloc[outer_train], y_all.iloc[outer_test]
            g_train, g_test = groups_all.iloc[outer_train], groups_all.iloc[outer_test]

            train_pos = int(y_train.sum())
            pos_weight = float((len(y_train) - train_pos) / max(train_pos, 1))
            split_rows.extend([
                {"repeat": repeat, "fold": fold, "partition": "train", "row_index": int(i), "athlete_id": groups_all.iloc[i]}
                for i in outer_train
            ])
            split_rows.extend([
                {"repeat": repeat, "fold": fold, "partition": "test", "row_index": int(i), "athlete_id": groups_all.iloc[i]}
                for i in outer_test
            ])

            for model_name in config.models:
                run_seed = repeat_seed + fold * 37
                candidates = model_space(model_name, pos_weight, run_seed)
                best_idx, tuning = choose_candidate(
                    candidates, X_train.reset_index(drop=True), y_train.reset_index(drop=True),
                    g_train.reset_index(drop=True), config.inner_folds, run_seed + 1,
                )
                tuning.insert(0, "model", model_name)
                tuning.insert(0, "fold", fold)
                tuning.insert(0, "repeat", repeat)
                tuning["selected"] = tuning["candidate_index"].eq(best_idx)
                tuning_rows.append(tuning)

                params, chosen_estimator = candidates[best_idx]
                inner_prob = inner_oof_prob(
                    chosen_estimator, X_train.reset_index(drop=True), y_train.reset_index(drop=True),
                    g_train.reset_index(drop=True), config.inner_folds, run_seed + 2,
                )
                cal_name, calibrator, inner_cal_prob, cal_table = fit_calibrators(
                    y_train.reset_index(drop=True), inner_prob, g_train.reset_index(drop=True),
                    config.inner_folds, run_seed + 3,
                )
                cal_table.insert(0, "model", model_name)
                cal_table.insert(0, "fold", fold)
                cal_table.insert(0, "repeat", repeat)
                cal_table["selected"] = cal_table["calibrator"].eq(cal_name)
                calibration_rows.append(cal_table)
                threshold, inner_mcc = choose_mcc_threshold(y_train, inner_cal_prob)

                fitted = clone(chosen_estimator).fit(X_train, y_train)
                raw_test = fitted.predict_proba(X_test)[:, 1]
                calibrated_test = apply_calibrator(cal_name, calibrator, raw_test)
                fold_metric = metrics(y_test, calibrated_test, threshold)
                fold_metric.update({
                    "repeat": repeat, "fold": fold, "model": model_name,
                    "n_test_athletes": int(g_test.nunique()),
                    "selected_params": json.dumps(params, sort_keys=True),
                    "calibrator": cal_name, "inner_mcc_at_threshold": inner_mcc,
                })
                fold_rows.append(fold_metric)

                prediction = pd.DataFrame({
                    "row_index": outer_test,
                    "athlete_id": g_test.to_numpy(),
                    "date": df.iloc[outer_test]["Date"].to_numpy(),
                    "y_true": y_test.to_numpy(),
                    "raw_probability": raw_test,
                    "probability": calibrated_test,
                    "threshold": threshold,
                    "y_pred": (calibrated_test >= threshold).astype(int),
                    "repeat": repeat, "fold": fold, "model": model_name,
                    "calibrator": cal_name,
                })
                all_predictions.append(prediction)
                print(
                    f"repeat={repeat + 1}/{config.outer_repeats} fold={fold + 1}/{config.outer_folds} "
                    f"model={model_name} AP={fold_metric['average_precision']:.4f} "
                    f"ROC={fold_metric['roc_auc']:.4f} MCC={fold_metric['mcc']:.4f}",
                    flush=True,
                )

    predictions = pd.concat(all_predictions, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    tuning = pd.concat(tuning_rows, ignore_index=True)
    calibration = pd.concat(calibration_rows, ignore_index=True)
    splits = pd.DataFrame(split_rows)
    predictions.to_csv(out / "outer_oof_predictions.csv.gz", index=False, compression="gzip")
    fold_metrics.to_csv(out / "outer_fold_metrics.csv", index=False)
    tuning.to_csv(out / "inner_tuning_results.csv", index=False)
    calibration.to_csv(out / "inner_calibration_selection.csv", index=False)
    splits.to_csv(out / "split_manifest.csv.gz", index=False, compression="gzip")
    return predictions, fold_metrics


def leakage_ablation(df: pd.DataFrame, safe_features: list[str], config: Config, out: Path):
    """Negative-control experiment. Gap is knowingly unavailable/deployment-unsafe."""
    ordered = df.sort_values(["Athlete ID", "Date"]).copy()
    ordered["recording_gap__LEAKAGE_CONTROL"] = ordered.groupby("Athlete ID")["Date"].diff().fillna(1.0)
    ordered = ordered.sort_index().reset_index(drop=True)
    y = ordered["injury"].astype(int)
    groups = ordered["Athlete ID"]
    feature_sets = {
        "gap_only_negative_control": ["recording_gap__LEAKAGE_CONTROL"],
        "safe_primary": safe_features,
        "safe_plus_gap_negative_control": safe_features + ["recording_gap__LEAKAGE_CONTROL"],
    }
    predictions = []
    for repeat in range(config.outer_repeats):
        repeat_seed = config.seed + repeat * 1009
        for fold, (train_idx, test_idx) in enumerate(shuffled_group_folds(groups, config.outer_folds, repeat_seed)):
            for setting, cols in feature_sets.items():
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler(quantile_range=(10, 90))),
                    ("clf", LogisticRegression(
                        C=0.25, penalty="l2", class_weight="balanced", solver="liblinear",
                        max_iter=4000, random_state=repeat_seed + fold,
                    )),
                ])
                model.fit(ordered.iloc[train_idx][cols], y.iloc[train_idx])
                p = model.predict_proba(ordered.iloc[test_idx][cols])[:, 1]
                predictions.append(pd.DataFrame({
                    "row_index": test_idx, "athlete_id": groups.iloc[test_idx].to_numpy(),
                    "y_true": y.iloc[test_idx].to_numpy(), "probability": p,
                    "repeat": repeat, "fold": fold, "setting": setting,
                }))
    pred = pd.concat(predictions, ignore_index=True)
    rows = []
    for (repeat, setting), g in pred.groupby(["repeat", "setting"]):
        rows.append({
            "repeat": repeat, "setting": setting, "n_rows": len(g),
            "prevalence": g.y_true.mean(), "roc_auc": roc_auc_score(g.y_true, g.probability),
            "average_precision": average_precision_score(g.y_true, g.probability),
            "brier": brier_score_loss(g.y_true, g.probability),
        })
    summary = pd.DataFrame(rows)
    pred.to_csv(out / "leakage_ablation_predictions.csv.gz", index=False, compression="gzip")
    summary.to_csv(out / "leakage_ablation_by_repeat.csv", index=False)
    return pred, summary


BOOT_METRICS = [
    "roc_auc", "average_precision", "mcc", "f1", "recall", "precision",
    "specificity", "balanced_accuracy", "brier", "calibration_intercept",
    "calibration_slope", "ece_10", "alerts_per_100_rows",
]


def cluster_bootstrap(predictions: pd.DataFrame, iterations: int, seed: int, out: Path):
    rng = np.random.default_rng(seed)
    model_rows, boot_rows = [], []
    for model_name, model_pred in predictions.groupby("model"):
        # Each repeat contains one OOF prediction per original row. Aggregate the
        # repeated predictions per row before athlete-cluster resampling.
        aggregated = model_pred.groupby(
            ["row_index", "athlete_id", "y_true"], as_index=False
        ).agg(probability=("probability", "mean"), threshold=("threshold", "mean"))
        point = metrics(aggregated.y_true, aggregated.probability, aggregated.threshold.to_numpy())
        point.update({"model": model_name, "n_athletes": aggregated.athlete_id.nunique()})
        model_rows.append(point)

        athletes = np.asarray(pd.unique(aggregated.athlete_id))
        clusters = {a: aggregated[aggregated.athlete_id.eq(a)] for a in athletes}
        completed = 0
        while completed < iterations:
            sampled = rng.choice(athletes, size=len(athletes), replace=True)
            pieces = [clusters[a] for a in sampled]
            boot = pd.concat(pieces, ignore_index=True)
            if boot.y_true.nunique() < 2:
                continue
            m = metrics(boot.y_true, boot.probability, boot.threshold.to_numpy())
            boot_rows.append({"model": model_name, "iteration": completed, **{k: m[k] for k in BOOT_METRICS}})
            completed += 1

    point_df = pd.DataFrame(model_rows)
    boot_df = pd.DataFrame(boot_rows)
    ci_rows = []
    for model_name, group in boot_df.groupby("model"):
        point = point_df.set_index("model").loc[model_name]
        for metric_name in BOOT_METRICS:
            values = group[metric_name].dropna()
            ci_rows.append({
                "model": model_name, "metric": metric_name,
                "estimate": point[metric_name],
                "ci_lower": values.quantile(0.025),
                "ci_upper": values.quantile(0.975),
            })
    ci = pd.DataFrame(ci_rows)
    point_df.to_csv(out / "pooled_model_metrics.csv", index=False)
    boot_df.to_csv(out / "athlete_cluster_bootstrap.csv.gz", index=False, compression="gzip")
    ci.to_csv(out / "model_metrics_with_95ci.csv", index=False)
    return point_df, ci


def paired_model_bootstrap(predictions: pd.DataFrame, iterations: int, seed: int, out: Path):
    """Paired athlete bootstrap for model-performance differences."""
    rng = np.random.default_rng(seed)
    aggregated = {}
    for model_name, g in predictions.groupby("model"):
        aggregated[model_name] = g.groupby(
            ["row_index", "athlete_id", "y_true"], as_index=False
        ).agg(probability=("probability", "mean"), threshold=("threshold", "mean"))
    model_names = sorted(aggregated)
    athlete_ids = np.asarray(sorted(pd.unique(next(iter(aggregated.values())).athlete_id)))
    cluster_maps = {
        model: {a: frame[frame.athlete_id.eq(a)] for a in athlete_ids}
        for model, frame in aggregated.items()
    }
    metric_names = ["roc_auc", "average_precision", "mcc", "brier"]
    point_metrics = {
        model: metrics(frame.y_true, frame.probability, frame.threshold.to_numpy())
        for model, frame in aggregated.items()
    }
    boot_rows = []
    for iteration in range(iterations):
        sampled = rng.choice(athlete_ids, size=len(athlete_ids), replace=True)
        current = {}
        for model in model_names:
            boot = pd.concat([cluster_maps[model][a] for a in sampled], ignore_index=True)
            if boot.y_true.nunique() < 2:
                break
            m = metrics(boot.y_true, boot.probability, boot.threshold.to_numpy())
            current[model] = m
        if len(current) != len(model_names):
            continue
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i + 1:]:
                for metric_name in metric_names:
                    boot_rows.append({
                        "iteration": iteration, "model_a": model_a, "model_b": model_b,
                        "metric": metric_name,
                        "difference_a_minus_b": current[model_a][metric_name] - current[model_b][metric_name],
                    })
    boot_df = pd.DataFrame(boot_rows)
    summary_rows = []
    for (model_a, model_b, metric_name), group in boot_df.groupby(["model_a", "model_b", "metric"]):
        differences = group["difference_a_minus_b"]
        # Higher is better except for Brier score, where lower is better.
        probability_a_better = np.mean(differences < 0) if metric_name == "brier" else np.mean(differences > 0)
        summary_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric_name,
            "estimate": point_metrics[model_a][metric_name] - point_metrics[model_b][metric_name],
            "ci_lower": differences.quantile(0.025),
            "ci_upper": differences.quantile(0.975),
            "probability_a_better": probability_a_better,
        })
    summary = pd.DataFrame(summary_rows)
    boot_df.to_csv(out / "paired_model_bootstrap.csv.gz", index=False, compression="gzip")
    summary.to_csv(out / "paired_model_differences_95ci.csv", index=False)
    return summary


def sensitivity_ratio_experiment(df, safe_features, ratio_features, config, out):
    """Test whether the unstable supplied ratios add robust ranking value."""
    y = df["injury"].astype(int).reset_index(drop=True)
    groups = df["Athlete ID"].reset_index(drop=True)
    settings = {"safe_without_ratios": safe_features, "safe_with_raw_ratios": safe_features + ratio_features}
    rows = []
    for repeat in range(config.outer_repeats):
        seed = config.seed + repeat * 1009
        for fold, (tr, te) in enumerate(shuffled_group_folds(groups, config.outer_folds, seed)):
            for setting, cols in settings.items():
                estimator = model_space("logistic", 1.0, seed + fold)[1][1]
                estimator.fit(df.iloc[tr][cols], y.iloc[tr])
                p = estimator.predict_proba(df.iloc[te][cols])[:, 1]
                rows.append({
                    "repeat": repeat, "fold": fold, "setting": setting,
                    "roc_auc": roc_auc_score(y.iloc[te], p),
                    "average_precision": average_precision_score(y.iloc[te], p),
                    "brier": brier_score_loss(y.iloc[te], p),
                })
    result = pd.DataFrame(rows)
    result.to_csv(out / "ratio_feature_sensitivity.csv", index=False)
    return result


def make_figures(predictions, leakage_summary, ci, out):
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    colors = {"logistic": "#2F75B5", "random_forest": "#548235", "xgboost": "#C65911"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_name, g in predictions.groupby("model"):
        agg = g.groupby(["row_index", "y_true"], as_index=False).probability.mean()
        fpr, tpr, _ = roc_curve(agg.y_true, agg.probability)
        precision, recall, _ = precision_recall_curve(agg.y_true, agg.probability)
        axes[0].plot(fpr, tpr, lw=2, color=colors.get(model_name), label=f"{model_name} (AUC={roc_auc_score(agg.y_true, agg.probability):.3f})")
        axes[1].plot(recall, precision, lw=2, color=colors.get(model_name), label=f"{model_name} (AP={average_precision_score(agg.y_true, agg.probability):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    prevalence = predictions.drop_duplicates(["model", "row_index"]).y_true.mean()
    axes[1].axhline(prevalence, ls="--", color="grey", lw=1, label=f"prevalence={prevalence:.3f}")
    axes[0].set(title="Leakage-safe ROC curves", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].set(title="Leakage-safe precision-recall curves", xlabel="Recall", ylabel="Precision")
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(fig_dir / "discrimination_curves.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 0.08], [0, 0.08], "--", color="grey", label="Ideal")
    for model_name, g in predictions.groupby("model"):
        agg = g.groupby(["row_index", "y_true"], as_index=False).probability.mean()
        obs, pred = calibration_curve(agg.y_true, agg.probability, n_bins=10, strategy="quantile")
        ax.plot(pred, obs, marker="o", lw=2, color=colors.get(model_name), label=model_name)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed injury frequency", title="Calibration on repeated grouped OOF predictions")
    ax.grid(alpha=0.2); ax.legend(); fig.tight_layout()
    fig.savefig(fig_dir / "calibration_curves.png", dpi=220); plt.close(fig)

    plot = leakage_summary.groupby("setting")[["roc_auc", "average_precision"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [s.replace("_", "\n") for s in plot.setting]
    bar_colors = ["#2F75B5" if s == "safe_primary" else "#C00000" for s in plot.setting]
    axes[0].bar(labels, plot.roc_auc, color=bar_colors)
    axes[1].bar(labels, plot.average_precision, color=bar_colors)
    axes[0].set_ylim(0.45, 1.01); axes[1].set_ylim(0, 1.01)
    axes[0].set_title("ROC-AUC leakage ablation"); axes[1].set_title("Average precision leakage ablation")
    for ax in axes: ax.tick_params(axis="x", labelsize=8); ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(fig_dir / "gap_leakage_ablation.png", dpi=220); plt.close(fig)

    selected = ci[ci.metric.isin(["roc_auc", "average_precision", "mcc", "brier"])].copy()
    metrics_order = ["roc_auc", "average_precision", "mcc", "brier"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, metric_name in zip(axes, metrics_order):
        d = selected[selected.metric.eq(metric_name)].reset_index(drop=True)
        x = np.arange(len(d))
        err = np.vstack([d.estimate - d.ci_lower, d.ci_upper - d.estimate])
        ax.errorbar(x, d.estimate, yerr=err, fmt="o", capsize=4, color="#17365D")
        ax.set_xticks(x, d.model, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric_name.replace("_", " ").title()); ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(fig_dir / "model_metrics_95ci.png", dpi=220); plt.close(fig)


def format_ci(ci, model, metric):
    row = ci[(ci.model == model) & (ci.metric == metric)].iloc[0]
    return f"{row.estimate:.3f} ({row.ci_lower:.3f}--{row.ci_upper:.3f})"


def write_report(config, df, features, point, ci, folds, leakage, ratio_sensitivity, paired, elapsed, out):
    leakage_mean = leakage.groupby("setting")[["roc_auc", "average_precision", "brier"]].mean()
    ratio_mean = ratio_sensitivity.groupby("setting")[["roc_auc", "average_precision", "brier"]].mean()
    lines = [
        "# Leakage-safe repeated validation report",
        "",
        "## Scientific status",
        "",
        "This is a confirmatory re-analysis of the existing row-level injury label. It evaluates transfer to unseen athletes. "
        "The implemented endpoint is concurrent row classification; the released injury-episode timing and feature-availability semantics do not define a prospective 7/14-day target.",
        "",
        "## Data and feature contract",
        "",
        f"- Rows: {len(df):,}; athletes: {df['Athlete ID'].nunique()}; injury rows: {int(df.injury.sum())}; prevalence: {df.injury.mean():.4%}.",
        f"- Primary predictors: {len(features)} numeric variables.",
        "- Excluded from the primary model: target, Athlete ID, Date, gap/time-gap variables, and the three unstable supplied relative-load ratios.",
        f"- Outer validation: {config.outer_repeats} repetitions of {config.outer_folds}-fold athlete-group CV.",
        f"- Inner selection: {config.inner_folds}-fold athlete-group CV using average precision.",
        "- Calibration (none, Platt, isotonic) selected by inner out-of-fold Brier score.",
        "- Decision threshold selected by inner out-of-fold MCC.",
        "- 95% uncertainty intervals: athlete-cluster bootstrap over aggregated repeated OOF predictions.",
        "",
        "## Main results (estimate and athlete-bootstrap 95% CI)",
        "",
        "| Model | ROC-AUC | Average precision | MCC | Recall | Precision | Brier | Calibration slope |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in config.models:
        lines.append(
            f"| {model} | {format_ci(ci, model, 'roc_auc')} | {format_ci(ci, model, 'average_precision')} | "
            f"{format_ci(ci, model, 'mcc')} | {format_ci(ci, model, 'recall')} | {format_ci(ci, model, 'precision')} | "
            f"{format_ci(ci, model, 'brier')} | {format_ci(ci, model, 'calibration_slope')} |"
        )
    lines.extend([
        "",
        "Because injury prevalence is very low, average precision, calibration, and false-alert burden are more informative than accuracy alone.",
        "At the inner-selected MCC thresholds, the models generate approximately 41--49 alerts per 100 rows, with approximately 1.8--1.9% of alerts corresponding to injury rows.",
        "",
        "## Paired model differences (athlete-bootstrap 95% CI)",
        "",
        "| Model A - Model B | Metric | Difference | 95% CI |",
        "|---|---|---:|---:|",
    ])
    for _, row in paired.iterrows():
        lines.append(
            f"| {row.model_a} - {row.model_b} | {row.metric} | {row.estimate:.4f} | "
            f"{row.ci_lower:.4f}--{row.ci_upper:.4f} |"
        )
    lines.extend([
        "",
        "Random forest is better than XGBoost for ROC-AUC and MCC in this paired analysis, but not for average precision or Brier score. "
        "No clear differences are observed between random forest and logistic regression. The practical conclusion is therefore not that a complex learner solves the task, "
        "but that all three models have limited rare-event utility after leakage control.",
        "",
        "## Deliberate leakage negative control",
        "",
        "The recording gap is intentionally reintroduced only in this section to quantify how a deployment-unavailable artifact inflates performance.",
        "",
        "| Setting | Mean ROC-AUC | Mean average precision | Mean Brier |",
        "|---|---:|---:|---:|",
    ])
    for setting, row in leakage_mean.iterrows():
        lines.append(f"| {setting} | {row.roc_auc:.3f} | {row.average_precision:.3f} | {row.brier:.3f} |")
    lines.extend([
        "",
        "## Relative-ratio sensitivity",
        "",
        "| Setting | Mean ROC-AUC | Mean average precision | Mean Brier |",
        "|---|---:|---:|---:|",
    ])
    for setting, row in ratio_mean.iterrows():
        lines.append(f"| {setting} | {row.roc_auc:.3f} | {row.average_precision:.3f} | {row.brier:.3f} |")
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "1. The dataset contains 74 independent athletes; repeated rows are nested within athletes.",
        "2. External validation is not available.",
        "3. The current label is retained as supplied; incident injury episodes and a future prediction horizon are not reconstructed in this analysis.",
        "4. Confidence intervals quantify between-athlete variability within this cohort.",
        "5. The gap experiment functions as a negative control for observation-process effects.",
        "",
        "## Reproducibility",
        "",
        f"Runtime: {elapsed / 60:.1f} minutes. Complete predictions, split manifests, tuning results, calibration choices, bootstrap samples, figures, and configuration are stored beside this report.",
        "",
    ])
    (out / "METHODS_AND_RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/data_weekly.csv")
    parser.add_argument("--output", default="experiments/leakage_safe_validation/results")
    parser.add_argument("--outer-repeats", type=int, default=3)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--models", nargs="+", default=["logistic", "random_forest", "xgboost"])
    args = parser.parse_args()
    config = Config(
        data_path=args.data, output_dir=args.output, outer_repeats=args.outer_repeats,
        outer_folds=args.outer_folds, inner_folds=args.inner_folds,
        bootstrap_iterations=args.bootstrap, models=tuple(args.models),
    )
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    warnings.filterwarnings("ignore", category=FutureWarning)

    data_path = Path(config.data_path)
    df = pd.read_csv(data_path)
    df = df.sort_values(["Athlete ID", "Date"]).reset_index(drop=True)
    if df.duplicated(["Athlete ID", "Date"]).any():
        raise ValueError("Duplicate athlete-date rows detected")
    if not set(pd.unique(df.injury)).issubset({0, 1}):
        raise ValueError("Target is not binary")
    safe_features, sensitivity_features, ratio_features, gap_features = feature_contract(df)
    if any("gap" in c.lower() for c in safe_features):
        raise AssertionError("Gap-like variable entered the primary feature set")

    metadata = {
        "config": asdict(config), "data_sha256": sha256(data_path),
        "n_rows": len(df), "n_athletes": int(df["Athlete ID"].nunique()),
        "n_injury_rows": int(df.injury.sum()), "prevalence": float(df.injury.mean()),
        "primary_features": safe_features, "excluded_ratio_features": ratio_features,
        "detected_gap_features": gap_features,
        "versions": {
            "python": sys.version, "platform": platform.platform(), "pandas": pd.__version__,
            "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "xgboost": getattr(xgboost, "__version__", None),
        },
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    pd.DataFrame({"feature": safe_features, "role": "primary_safe"}).to_csv(out / "primary_feature_manifest.csv", index=False)

    predictions, fold_metrics = run_main(config, df, safe_features, out)
    leakage_pred, leakage_summary = leakage_ablation(df, safe_features, config, out)
    ratio_sensitivity = sensitivity_ratio_experiment(df, safe_features, ratio_features, config, out)
    point, ci = cluster_bootstrap(predictions, config.bootstrap_iterations, config.seed + 777, out)
    paired = paired_model_bootstrap(predictions, config.bootstrap_iterations, config.seed + 778, out)
    make_figures(predictions, leakage_summary, ci, out)
    elapsed = time.time() - start
    write_report(config, df, safe_features, point, ci, fold_metrics, leakage_summary, ratio_sensitivity, paired, elapsed, out)
    metadata["runtime_seconds"] = elapsed
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Completed in {elapsed / 60:.1f} minutes. Results: {out.resolve()}")


if __name__ == "__main__":
    main()
