#!/usr/bin/env python3
"""Confirmatory weekly-dataset experiments for Paper 2.

This script performs five linked analyses:

1. A faithful, version-compatible replication of the original weekly approach.
2. A corrected last-10-athlete comparison with training-only preprocessing,
   athlete-disjoint calibration, safe features, and untouched test athletes.
3. Twenty-one non-overlapping Date-modulo-21 evaluations to test temporal
   pseudo-replication caused by adjacent three-week feature histories.
4. Label-permutation and circular-shift negative controls.
5. Brier skill, fixed alert-budget utility, and explanation-rank stability.

The supplied injury-labelled event day remains the endpoint. The provisional
Notebook-07 28-day target is intentionally not used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr, wilcoxon
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

try:
    import xgboost
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    raise RuntimeError("xgboost is required for the confirmatory experiments") from exc

# Make the project namespace importable when this file is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.leakage_safe_validation.run_leakage_safe_validation import (
    apply_calibrator,
    choose_mcc_threshold,
    feature_contract,
    fit_calibrators,
    inner_oof_prob,
    metrics,
    model_space,
    shuffled_group_folds,
)


SEED = 20260829


@dataclass(frozen=True)
class Config:
    data_path: str = "data/raw/data_weekly.csv"
    primary_predictions: str = "experiments/leakage_safe_validation/results/outer_oof_predictions.csv.gz"
    output_dir: str = "experiments/confirmatory_weekly_analysis/results"
    original_experiments: int = 5
    original_bags: int = 9
    samples_per_class: int = 2048
    outer_repeats: int = 3
    outer_folds: int = 5
    inner_folds: int = 3
    permutation_iterations: int = 25
    alert_bootstrap_iterations: int = 1000
    seed: int = SEED


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def balanced_by_athlete(frame: pd.DataFrame, samples_per_class: int, rng: np.random.Generator) -> pd.DataFrame:
    counts = frame.groupby(["Athlete ID", "injury"]).size().unstack(fill_value=0)
    eligible = counts.index[(counts.get(0, 0) > 0) & (counts.get(1, 0) > 0)].to_numpy()
    if not len(eligible):
        raise ValueError("No athletes contain both outcome classes")
    per_athlete = max(1, int(np.floor(samples_per_class / len(eligible))))
    pieces = []
    for athlete in eligible:
        for label in (0, 1):
            candidates = frame.index[(frame["Athlete ID"].eq(athlete)) & (frame.injury.eq(label))].to_numpy()
            chosen = rng.choice(candidates, size=per_athlete, replace=True)
            pieces.append(frame.loc[chosen])
    return pd.concat(pieces, ignore_index=True)


def athlete_healthy_normalizer(frame: pd.DataFrame, features: list[str]):
    healthy = frame[frame.injury.eq(0)]
    mean = healthy.groupby("Athlete ID")[features].mean()
    std = healthy.groupby("Athlete ID")[features].std().replace(0.0, 0.01).fillna(0.01)
    return mean, std


def apply_athlete_normalizer(frame: pd.DataFrame, features: list[str], mean: pd.DataFrame, std: pd.DataFrame):
    matrix = frame[features].copy()
    athletes = frame["Athlete ID"].to_numpy()
    mu = mean.reindex(athletes).to_numpy()
    sigma = std.reindex(athletes).to_numpy()
    if np.isnan(mu).any() or np.isnan(sigma).any():
        raise ValueError("Athlete normalization parameters unavailable")
    return (matrix.to_numpy() - mu) / sigma


def xgb_original(seed: int, rng: random.Random):
    return XGBClassifier(
        objective="binary:logistic", learning_rate=0.01,
        max_depth=rng.choice([2, 3]), n_estimators=rng.choice([256, 512]),
        importance_type="total_gain", eval_metric="auc", verbosity=0,
        tree_method="hist", n_jobs=-1, random_state=seed,
    )


def platt_fit(y, probability):
    eps = 1e-6
    p = np.clip(np.asarray(probability), eps, 1 - eps)
    transform = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(transform, np.asarray(y))
    return model


def platt_apply(model, probability):
    eps = 1e-6
    p = np.clip(np.asarray(probability), eps, 1 - eps)
    return model.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def equal_sensitivity_specificity_threshold(y, probability):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability)
    candidates = np.unique(probability)
    best_threshold, best_gap = 0.5, np.inf
    for threshold in candidates:
        pred = probability >= threshold
        tp = np.sum(pred & (y == 1)); fn = np.sum(~pred & (y == 1))
        tn = np.sum(~pred & (y == 0)); fp = np.sum(pred & (y == 0))
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        gap = abs(sensitivity - specificity)
        if gap < best_gap:
            best_threshold, best_gap = float(threshold), float(gap)
    return best_threshold


def original_style_replication(df: pd.DataFrame, features: list[str], config: Config, out: Path):
    """Replicate the source notebook, including its documented transductive choices."""
    athletes = np.asarray(sorted(df["Athlete ID"].unique()))
    test_athletes = athletes[-10:]
    train_pool = df[~df["Athlete ID"].isin(test_athletes)].copy()
    test = df[df["Athlete ID"].isin(test_athletes)].copy()
    train_mean, train_std = athlete_healthy_normalizer(train_pool, features)
    # Faithful source behavior: test normalization uses healthy test outcomes.
    test_mean, test_std = athlete_healthy_normalizer(test, features)
    X_test = apply_athlete_normalizer(test, features, test_mean, test_std)
    y_test = test.injury.to_numpy(dtype=int)
    rows, prediction_rows = [], []

    for experiment in range(config.original_experiments):
        seed = config.seed + 10000 + experiment * 101
        rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)
        # Faithful source behavior: calibration is sampled independently from
        # the same row pool as every training bag, so sampled rows may overlap.
        calibration = balanced_by_athlete(train_pool, config.samples_per_class, rng)
        X_cal = apply_athlete_normalizer(calibration, features, train_mean, train_std)
        y_cal = calibration.injury.to_numpy(dtype=int)
        test_probs, validation_probs = [], []
        models = []
        for bag in range(config.original_bags):
            bag_frame = balanced_by_athlete(train_pool, config.samples_per_class, rng)
            X_bag = apply_athlete_normalizer(bag_frame, features, train_mean, train_std)
            y_bag = bag_frame.injury.to_numpy(dtype=int)
            base = xgb_original(seed + bag, py_rng).fit(X_bag, y_bag)
            calibrator = platt_fit(y_cal, base.predict_proba(X_cal)[:, 1])
            models.append((base, calibrator))
            test_probs.append(platt_apply(calibrator, base.predict_proba(X_test)[:, 1]))

        # Faithful source behavior: threshold sample is drawn from the training
        # pool to match test class counts and can overlap training/calibration.
        validation = pd.concat([
            train_pool[train_pool.injury.eq(0)].sample(int((y_test == 0).sum()), random_state=seed),
            train_pool[train_pool.injury.eq(1)].sample(int((y_test == 1).sum()), random_state=seed + 1),
        ], ignore_index=True)
        X_validation = apply_athlete_normalizer(validation, features, train_mean, train_std)
        for base, calibrator in models:
            validation_probs.append(platt_apply(calibrator, base.predict_proba(X_validation)[:, 1]))
        validation_probability = np.mean(validation_probs, axis=0)
        threshold = equal_sensitivity_specificity_threshold(validation.injury, validation_probability)
        probability = np.mean(test_probs, axis=0)
        result = metrics(y_test, probability, threshold)
        result.update({"design": "original_style", "experiment": experiment, "test_athletes": ";".join(map(str, test_athletes))})
        rows.append(result)
        prediction_rows.append(pd.DataFrame({
            "row_index": test.index, "athlete_id": test["Athlete ID"].to_numpy(),
            "y_true": y_test, "probability": probability, "threshold": threshold,
            "design": "original_style", "experiment": experiment,
        }))
        print(f"original experiment={experiment + 1}/{config.original_experiments} ROC={result['roc_auc']:.4f} AP={result['average_precision']:.4f}", flush=True)

    result_df = pd.DataFrame(rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    result_df.to_csv(out / "original_style_replication_metrics.csv", index=False)
    predictions.to_csv(out / "original_style_replication_predictions.csv.gz", index=False, compression="gzip")
    return result_df, predictions


def corrected_last10_replication(df: pd.DataFrame, features: list[str], config: Config, out: Path):
    """Use the same final 10 test athletes but remove source-protocol leakage."""
    athletes = np.asarray(sorted(df["Athlete ID"].unique()))
    test_athletes = athletes[-10:]
    development = df[~df["Athlete ID"].isin(test_athletes)].copy().reset_index().rename(columns={"index": "source_index"})
    test = df[df["Athlete ID"].isin(test_athletes)].copy()
    rows, prediction_rows = [], []
    for experiment in range(config.original_experiments):
        seed = config.seed + 20000 + experiment * 101
        splits = list(shuffled_group_folds(development["Athlete ID"], 5, seed))
        fit_idx, calibration_idx = splits[experiment % 5]
        fit_pool = development.iloc[fit_idx].copy()
        calibration = development.iloc[calibration_idx].copy()
        train_pos = int(fit_pool.injury.sum())
        pos_weight = (len(fit_pool) - train_pos) / max(train_pos, 1)
        test_probabilities, calibration_probabilities = [], []
        rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)
        for bag in range(config.original_bags):
            bag_frame = balanced_by_athlete(fit_pool, config.samples_per_class, rng)
            model = XGBClassifier(
                objective="binary:logistic", learning_rate=0.01,
                max_depth=py_rng.choice([2, 3]), n_estimators=py_rng.choice([256, 512]),
                reg_lambda=5.0, reg_alpha=0.1, subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", tree_method="hist", n_jobs=-1,
                random_state=seed + bag, scale_pos_weight=1.0,
            ).fit(bag_frame[features], bag_frame.injury)
            calibration_probabilities.append(model.predict_proba(calibration[features])[:, 1])
            test_probabilities.append(model.predict_proba(test[features])[:, 1])
        raw_calibration = np.mean(calibration_probabilities, axis=0)
        raw_test = np.mean(test_probabilities, axis=0)
        calibrator = platt_fit(calibration.injury, raw_calibration)
        calibration_probability = platt_apply(calibrator, raw_calibration)
        probability = platt_apply(calibrator, raw_test)
        threshold, _ = choose_mcc_threshold(calibration.injury, calibration_probability)
        result = metrics(test.injury, probability, threshold)
        result.update({"design": "corrected_last10", "experiment": experiment, "test_athletes": ";".join(map(str, test_athletes))})
        rows.append(result)
        prediction_rows.append(pd.DataFrame({
            "row_index": test.index, "athlete_id": test["Athlete ID"].to_numpy(),
            "y_true": test.injury.to_numpy(dtype=int), "probability": probability,
            "threshold": threshold, "design": "corrected_last10", "experiment": experiment,
        }))
        print(f"corrected experiment={experiment + 1}/{config.original_experiments} ROC={result['roc_auc']:.4f} AP={result['average_precision']:.4f}", flush=True)
    result_df = pd.DataFrame(rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    result_df.to_csv(out / "corrected_last10_metrics.csv", index=False)
    predictions.to_csv(out / "corrected_last10_predictions.csv.gz", index=False, compression="gzip")
    return result_df, predictions


def locked_estimator(model_name: str, positive_weight: float, seed: int):
    # Hyperparameters are frozen from the most frequently selected settings in
    # the preceding nested analysis. No offset-specific optimization occurs.
    if model_name == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(10, 90))),
            ("clf", LogisticRegression(C=0.05, penalty="l2", class_weight="balanced", solver="liblinear", max_iter=4000, random_state=seed)),
        ])
    if model_name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=10, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
        ])
    if model_name == "xgboost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(n_estimators=250, max_depth=2, learning_rate=0.03, min_child_weight=5.0, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.1, scale_pos_weight=positive_weight, objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=seed)),
        ])
    raise ValueError(model_name)


def master_fold_maps(df: pd.DataFrame, config: Config):
    maps = {}
    for repeat in range(config.outer_repeats):
        fold_map = {}
        for fold, (_, test_idx) in enumerate(shuffled_group_folds(df["Athlete ID"], config.outer_folds, config.seed + repeat * 1009)):
            for athlete in pd.unique(df.iloc[test_idx]["Athlete ID"]):
                fold_map[athlete] = fold
        maps[repeat] = fold_map
    return maps


def calibrated_group_oof(frame: pd.DataFrame, features: list[str], model_name: str, fold_map: dict, seed: int, inner_folds: int):
    rows = []
    for fold in sorted(set(fold_map.values())):
        test_mask = frame["Athlete ID"].map(fold_map).eq(fold).to_numpy()
        train_idx = np.where(~test_mask)[0]; test_idx = np.where(test_mask)[0]
        train = frame.iloc[train_idx]; test = frame.iloc[test_idx]
        if not len(test) or train.injury.nunique() < 2:
            continue
        positive_weight = (len(train) - int(train.injury.sum())) / max(int(train.injury.sum()), 1)
        estimator = locked_estimator(model_name, positive_weight, seed + fold)
        inner_probability = inner_oof_prob(
            estimator, train[features].reset_index(drop=True), train.injury.reset_index(drop=True),
            train["Athlete ID"].reset_index(drop=True), inner_folds, seed + fold + 100,
        )
        # Use Platt consistently for sparse offset subsets. Selection among
        # calibrators was already evaluated in the full nested experiment.
        calibrator = platt_fit(train.injury, inner_probability)
        fitted = clone(estimator).fit(train[features], train.injury)
        raw_test = fitted.predict_proba(test[features])[:, 1]
        probability = platt_apply(calibrator, raw_test)
        rows.append(pd.DataFrame({
            "source_index": test.index, "athlete_id": test["Athlete ID"].to_numpy(),
            "date": test.Date.to_numpy(), "y_true": test.injury.to_numpy(dtype=int),
            "probability": probability, "fold": fold, "model": model_name,
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def nonoverlap_analysis(df: pd.DataFrame, features: list[str], config: Config, out: Path):
    fold_maps = master_fold_maps(df, config)
    predictions, summary_rows = [], []
    for repeat in range(config.outer_repeats):
        for offset in range(21):
            subset = df[df.Date.mod(21).eq(offset)].copy()
            for model_name in ("logistic", "random_forest", "xgboost"):
                pred = calibrated_group_oof(subset, features, model_name, fold_maps[repeat], config.seed + repeat * 1009 + offset * 41, config.inner_folds)
                if pred.empty or pred.y_true.nunique() < 2:
                    continue
                pred["repeat"] = repeat; pred["offset"] = offset
                predictions.append(pred)
                summary_rows.append({
                    "repeat": repeat, "offset": offset, "model": model_name,
                    "n_rows": len(pred), "n_positive": int(pred.y_true.sum()),
                    "prevalence": pred.y_true.mean(),
                    "roc_auc": roc_auc_score(pred.y_true, pred.probability),
                    "average_precision": average_precision_score(pred.y_true, pred.probability),
                    "brier": brier_score_loss(pred.y_true, pred.probability),
                })
            print(f"nonoverlap repeat={repeat + 1}/{config.outer_repeats} offset={offset + 1}/21", flush=True)
    pred_df = pd.concat(predictions, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    pred_df.to_csv(out / "nonoverlap_21offset_predictions.csv.gz", index=False, compression="gzip")
    summary.to_csv(out / "nonoverlap_21offset_metrics.csv", index=False)
    return summary


def fit_logistic_group_oof(df: pd.DataFrame, features: list[str], outcome: np.ndarray, fold_map: dict, seed: int):
    probabilities = np.full(len(df), np.nan)
    outcome = np.asarray(outcome, dtype=int)
    for fold in sorted(set(fold_map.values())):
        test_mask = df["Athlete ID"].map(fold_map).eq(fold).to_numpy()
        train_idx = np.where(~test_mask)[0]; test_idx = np.where(test_mask)[0]
        estimator = locked_estimator("logistic", 1.0, seed + fold)
        estimator.fit(df.iloc[train_idx][features], outcome[train_idx])
        probabilities[test_idx] = estimator.predict_proba(df.iloc[test_idx][features])[:, 1]
    if np.isnan(probabilities).any():
        raise RuntimeError("Incomplete permutation OOF probabilities")
    return probabilities


def permute_within_athlete(df: pd.DataFrame, y: np.ndarray, rng: np.random.Generator):
    result = np.asarray(y).copy()
    for _, indices in df.groupby("Athlete ID").indices.items():
        result[indices] = rng.permutation(result[indices])
    return result


def circular_shift_within_athlete(df: pd.DataFrame, y: np.ndarray, rng: np.random.Generator):
    result = np.asarray(y).copy()
    for _, indices in df.groupby("Athlete ID").indices.items():
        indices = np.asarray(indices)
        if len(indices) <= 1:
            continue
        minimum = min(21, len(indices) - 1)
        shift = int(rng.integers(minimum, len(indices))) if minimum < len(indices) else 1
        result[indices] = np.roll(result[indices], shift)
    return result


def permutation_controls(df: pd.DataFrame, features: list[str], config: Config, out: Path):
    fold_map = master_fold_maps(df, config)[0]
    y = df.injury.to_numpy(dtype=int)
    observed_probability = fit_logistic_group_oof(df, features, y, fold_map, config.seed + 30000)
    observed = {
        "roc_auc": roc_auc_score(y, observed_probability),
        "average_precision": average_precision_score(y, observed_probability),
    }
    rng = np.random.default_rng(config.seed + 31000)
    rows = []
    for iteration in range(config.permutation_iterations):
        for control, generator in (("within_athlete_permutation", permute_within_athlete), ("within_athlete_circular_shift", circular_shift_within_athlete)):
            permuted = generator(df, y, rng)
            probability = fit_logistic_group_oof(df, features, permuted, fold_map, config.seed + 32000 + iteration * 7)
            rows.append({
                "control": control, "iteration": iteration,
                "roc_auc": roc_auc_score(permuted, probability),
                "average_precision": average_precision_score(permuted, probability),
            })
        print(f"negative-control iteration={iteration + 1}/{config.permutation_iterations}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "label_negative_controls.csv", index=False)
    observed_df = pd.DataFrame([{"setting": "observed_safe_logistic", **observed}])
    observed_df.to_csv(out / "label_negative_control_observed.csv", index=False)
    return observed_df, result


def brier_skill_and_alert_budgets(df: pd.DataFrame, predictions_path: Path, config: Config, out: Path):
    predictions = pd.read_csv(predictions_path)
    baseline_rows = []
    for (repeat, fold), g in predictions[predictions.model.eq(predictions.model.iloc[0])].groupby(["repeat", "fold"]):
        test_rows = set(g.row_index)
        training_prevalence = df.loc[~df.index.isin(test_rows), "injury"].mean()
        baseline_rows.append(pd.DataFrame({"row_index": g.row_index, "repeat": repeat, "baseline_probability": training_prevalence}))
    baseline = pd.concat(baseline_rows).groupby("row_index", as_index=False).baseline_probability.mean()
    y_by_row = df[["injury"]].reset_index().rename(columns={"index": "row_index", "injury": "y_true"})
    baseline = baseline.merge(y_by_row, on="row_index", how="left")
    baseline_brier = brier_score_loss(baseline.y_true, baseline.baseline_probability)

    skill_rows, budget_rows = [], []
    aggregated_models = {}
    for model_name, g in predictions.groupby("model"):
        agg = g.groupby(["row_index", "athlete_id", "y_true"], as_index=False).probability.mean()
        aggregated_models[model_name] = agg
        model_brier = brier_score_loss(agg.y_true, agg.probability)
        skill_rows.append({
            "model": model_name, "model_brier": model_brier,
            "prevalence_baseline_brier": baseline_brier,
            "brier_skill_score": 1 - model_brier / baseline_brier,
        })
        for budget in (1, 2, 5, 10):
            n_alerts = max(1, int(round(len(agg) * budget / 100)))
            selected = agg.nlargest(n_alerts, "probability")
            tp = int(selected.y_true.sum())
            budget_rows.append({
                "model": model_name, "alert_budget_per_100": budget,
                "alerts": n_alerts, "true_positives": tp,
                "precision": tp / n_alerts, "recall": tp / int(agg.y_true.sum()),
                "probability_threshold": selected.probability.min(),
            })

    skill = pd.DataFrame(skill_rows)
    budgets = pd.DataFrame(budget_rows)
    skill.to_csv(out / "brier_skill_scores.csv", index=False)
    budgets.to_csv(out / "fixed_alert_budget_metrics.csv", index=False)

    # Athlete-cluster intervals with the alert budget re-applied inside each sample.
    rng = np.random.default_rng(config.seed + 40000)
    bootstrap_rows = []
    for model_name, agg in aggregated_models.items():
        athletes = np.asarray(pd.unique(agg.athlete_id))
        clusters = {a: agg[agg.athlete_id.eq(a)] for a in athletes}
        for iteration in range(config.alert_bootstrap_iterations):
            sample = rng.choice(athletes, size=len(athletes), replace=True)
            boot = pd.concat([clusters[a] for a in sample], ignore_index=True)
            positives = int(boot.y_true.sum())
            if positives == 0:
                continue
            for budget in (1, 2, 5, 10):
                n_alerts = max(1, int(round(len(boot) * budget / 100)))
                selected = boot.nlargest(n_alerts, "probability")
                tp = int(selected.y_true.sum())
                bootstrap_rows.append({
                    "model": model_name, "iteration": iteration,
                    "alert_budget_per_100": budget,
                    "precision": tp / n_alerts, "recall": tp / positives,
                })
    bootstrap = pd.DataFrame(bootstrap_rows)
    ci = bootstrap.groupby(["model", "alert_budget_per_100"]).agg(
        precision_lower=("precision", lambda x: x.quantile(.025)),
        precision_upper=("precision", lambda x: x.quantile(.975)),
        recall_lower=("recall", lambda x: x.quantile(.025)),
        recall_upper=("recall", lambda x: x.quantile(.975)),
    ).reset_index().merge(budgets, on=["model", "alert_budget_per_100"], how="left")
    bootstrap.to_csv(out / "fixed_alert_budget_bootstrap.csv.gz", index=False, compression="gzip")
    ci.to_csv(out / "fixed_alert_budget_with_95ci.csv", index=False)
    return skill, ci


def explanation_stability(df: pd.DataFrame, features: list[str], config: Config, out: Path):
    fold_maps = master_fold_maps(df, config)
    rankings = []
    for repeat in range(config.outer_repeats):
        fold_map = fold_maps[repeat]
        for fold in range(config.outer_folds):
            test_mask = df["Athlete ID"].map(fold_map).eq(fold).to_numpy()
            train = df.loc[~test_mask]
            positive_weight = (len(train) - int(train.injury.sum())) / max(int(train.injury.sum()), 1)
            for model_name in ("logistic", "random_forest", "xgboost"):
                estimator = locked_estimator(model_name, positive_weight, config.seed + repeat * 1009 + fold)
                estimator.fit(train[features], train.injury)
                clf = estimator.named_steps["clf"]
                if model_name == "logistic":
                    importance = np.abs(clf.coef_.ravel())
                else:
                    importance = np.asarray(clf.feature_importances_)
                order = np.argsort(-importance)
                ranks = np.empty(len(features), dtype=int); ranks[order] = np.arange(1, len(features) + 1)
                rankings.extend({
                    "repeat": repeat, "fold": fold, "model": model_name,
                    "feature": feature, "importance": float(importance[i]), "rank": int(ranks[i]),
                } for i, feature in enumerate(features))
    ranking_df = pd.DataFrame(rankings)
    stability_rows = []
    for model_name, group in ranking_df.groupby("model"):
        runs = [(r, f) for r, f in group[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None)]
        matrix = {run: group[(group.repeat.eq(run[0])) & (group.fold.eq(run[1]))].set_index("feature").loc[features, "rank"] for run in runs}
        for run_a, run_b in combinations(runs, 2):
            ranks_a, ranks_b = matrix[run_a], matrix[run_b]
            top_a = set(ranks_a.nsmallest(10).index); top_b = set(ranks_b.nsmallest(10).index)
            stability_rows.append({
                "model": model_name, "repeat_a": run_a[0], "fold_a": run_a[1],
                "repeat_b": run_b[0], "fold_b": run_b[1],
                "spearman_rank_correlation": spearmanr(ranks_a, ranks_b).statistic,
                "top10_overlap": len(top_a & top_b),
                "top10_jaccard": len(top_a & top_b) / len(top_a | top_b),
            })
    stability = pd.DataFrame(stability_rows)
    feature_summary = ranking_df.groupby(["model", "feature"]).agg(
        mean_rank=("rank", "mean"), median_rank=("rank", "median"),
        min_rank=("rank", "min"), max_rank=("rank", "max"),
        top10_frequency=("rank", lambda x: np.mean(x <= 10)),
    ).reset_index().sort_values(["model", "mean_rank"])
    ranking_df.to_csv(out / "explanation_fold_rankings.csv.gz", index=False, compression="gzip")
    stability.to_csv(out / "explanation_rank_stability.csv", index=False)
    feature_summary.to_csv(out / "explanation_feature_stability_summary.csv", index=False)
    return stability, feature_summary


def additional_inference(out: Path, primary_predictions_path: Path, bootstrap_iterations: int = 5000, seed: int = SEED + 50000):
    """Add paired design, offset-level, permutation, and Brier-skill inference."""
    original_pred = pd.read_csv(out / "original_style_replication_predictions.csv.gz")
    corrected_pred = pd.read_csv(out / "corrected_last10_predictions.csv.gz")
    aggregated = {}
    for name, frame in (("original_style", original_pred), ("corrected_last10", corrected_pred)):
        aggregated[name] = frame.groupby(["row_index", "athlete_id", "y_true"], as_index=False).probability.mean()
    athletes = np.asarray(sorted(pd.unique(aggregated["original_style"].athlete_id)))
    cluster_maps = {
        name: {athlete: frame[frame.athlete_id.eq(athlete)] for athlete in athletes}
        for name, frame in aggregated.items()
    }
    point = {}
    for name, frame in aggregated.items():
        point[name] = {
            "roc_auc": roc_auc_score(frame.y_true, frame.probability),
            "average_precision": average_precision_score(frame.y_true, frame.probability),
            "brier": brier_score_loss(frame.y_true, frame.probability),
        }
    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    completed = 0
    while completed < bootstrap_iterations:
        sampled = rng.choice(athletes, size=len(athletes), replace=True)
        samples = {
            name: pd.concat([cluster_maps[name][athlete] for athlete in sampled], ignore_index=True)
            for name in aggregated
        }
        if samples["original_style"].y_true.nunique() < 2:
            continue
        for metric_name in ("roc_auc", "average_precision", "brier"):
            values = {}
            for name, frame in samples.items():
                values[name] = (
                    roc_auc_score(frame.y_true, frame.probability) if metric_name == "roc_auc"
                    else average_precision_score(frame.y_true, frame.probability) if metric_name == "average_precision"
                    else brier_score_loss(frame.y_true, frame.probability)
                )
            bootstrap_rows.append({
                "iteration": completed, "metric": metric_name,
                "difference_original_minus_corrected": values["original_style"] - values["corrected_last10"],
            })
        completed += 1
    paired_bootstrap = pd.DataFrame(bootstrap_rows)
    paired_rows = []
    for metric_name, group in paired_bootstrap.groupby("metric"):
        values = group.difference_original_minus_corrected
        paired_rows.append({
            "metric": metric_name,
            "original_estimate": point["original_style"][metric_name],
            "corrected_estimate": point["corrected_last10"][metric_name],
            "difference_original_minus_corrected": point["original_style"][metric_name] - point["corrected_last10"][metric_name],
            "ci_lower": values.quantile(.025), "ci_upper": values.quantile(.975),
        })
    paired = pd.DataFrame(paired_rows)
    paired_bootstrap.to_csv(out / "original_corrected_paired_athlete_bootstrap.csv.gz", index=False, compression="gzip")
    paired.to_csv(out / "original_corrected_paired_differences_95ci.csv", index=False)

    # Add fold-training prevalence baselines to every non-overlapping condition.
    nonoverlap_pred = pd.read_csv(out / "nonoverlap_21offset_predictions.csv.gz")
    nonoverlap_metrics = pd.read_csv(out / "nonoverlap_21offset_metrics.csv")
    baseline_rows = []
    for keys, group in nonoverlap_pred.groupby(["repeat", "offset", "model"]):
        baseline_probability = np.empty(len(group), dtype=float)
        for fold in sorted(group.fold.unique()):
            mask = group.fold.eq(fold).to_numpy()
            baseline_probability[mask] = group.loc[~mask, "y_true"].mean()
        baseline_brier = brier_score_loss(group.y_true, baseline_probability)
        baseline_rows.append({
            "repeat": keys[0], "offset": keys[1], "model": keys[2],
            "prevalence_baseline_brier": baseline_brier,
        })
    baseline_df = pd.DataFrame(baseline_rows)
    nonoverlap_metrics = nonoverlap_metrics.drop(columns=["prevalence_baseline_brier", "brier_skill_score"], errors="ignore").merge(
        baseline_df, on=["repeat", "offset", "model"], how="left"
    )
    nonoverlap_metrics["brier_skill_score"] = 1 - nonoverlap_metrics.brier / nonoverlap_metrics.prevalence_baseline_brier
    nonoverlap_metrics.to_csv(out / "nonoverlap_21offset_metrics.csv", index=False)

    primary = pd.read_csv(primary_predictions_path).groupby(
        ["model", "row_index", "athlete_id", "y_true"], as_index=False
    ).probability.mean()
    primary_point = {
        model: {"roc_auc": roc_auc_score(group.y_true, group.probability), "average_precision": average_precision_score(group.y_true, group.probability)}
        for model, group in primary.groupby("model")
    }
    offset_rows = []
    for model_name, group in nonoverlap_metrics.groupby("model"):
        offsets = group.groupby("offset").agg(
            roc_auc=("roc_auc", "mean"), average_precision=("average_precision", "mean"),
            prevalence=("prevalence", "mean"), brier_skill_score=("brier_skill_score", "mean"),
        ).reset_index()
        roc_test = wilcoxon(offsets.roc_auc - .5, alternative="two-sided", zero_method="wilcox")
        ap_test = wilcoxon(offsets.average_precision - offsets.prevalence, alternative="two-sided", zero_method="wilcox")
        offset_rows.append({
            "model": model_name, "full_window_roc_auc": primary_point[model_name]["roc_auc"],
            "nonoverlap_mean_roc_auc": offsets.roc_auc.mean(),
            "roc_auc_change": offsets.roc_auc.mean() - primary_point[model_name]["roc_auc"],
            "roc_wilcoxon_vs_0_5_p": roc_test.pvalue,
            "full_window_average_precision": primary_point[model_name]["average_precision"],
            "nonoverlap_mean_average_precision": offsets.average_precision.mean(),
            "mean_offset_prevalence": offsets.prevalence.mean(),
            "ap_wilcoxon_vs_prevalence_p": ap_test.pvalue,
            "nonoverlap_mean_brier_skill": offsets.brier_skill_score.mean(),
        })
    offset_inference = pd.DataFrame(offset_rows)
    offset_inference.to_csv(out / "nonoverlap_inference_summary.csv", index=False)

    observed = pd.read_csv(out / "label_negative_control_observed.csv").iloc[0]
    controls = pd.read_csv(out / "label_negative_controls.csv")
    permutation_rows = []
    for control, group in controls.groupby("control"):
        for metric_name in ("roc_auc", "average_precision"):
            values = group[metric_name]
            observed_value = observed[metric_name]
            permutation_rows.append({
                "control": control, "metric": metric_name,
                "observed": observed_value, "null_mean": values.mean(), "null_sd": values.std(ddof=1),
                "null_95th_percentile": values.quantile(.95), "null_max": values.max(),
                "empirical_one_sided_p": (1 + int((values >= observed_value).sum())) / (len(values) + 1),
                "iterations": len(values),
            })
    permutation_inference = pd.DataFrame(permutation_rows)
    permutation_inference.to_csv(out / "label_negative_control_inference.csv", index=False)
    return paired, offset_inference, permutation_inference, nonoverlap_metrics


def make_figures(original, corrected, nonoverlap, observed, controls, alert_ci, out: Path):
    figures = out / "figures"; figures.mkdir(exist_ok=True)
    colors = {"logistic": "#2F75B5", "random_forest": "#548235", "xgboost": "#C65911"}

    comparison = pd.concat([original, corrected])
    design_order = ["original_style", "corrected_last10"]
    design_colors = {"original_style": "#C00000", "corrected_last10": "#2F75B5"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric in zip(axes, ["roc_auc", "average_precision"]):
        summary = comparison.groupby("design")[metric].agg(["mean", "std"]).reindex(design_order).reset_index()
        ax.bar(summary.design, summary["mean"], yerr=summary["std"], capsize=4, color=[design_colors[d] for d in summary.design])
        if metric == "roc_auc": ax.axhline(.5, color="grey", ls="--", lw=1)
        else: ax.axhline(.013435, color="grey", ls="--", lw=1, label="prevalence")
        ax.set_title(metric.replace("_", " ").title()); ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "original_vs_corrected.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric in zip(axes, ["roc_auc", "average_precision"]):
        for model_name, group in nonoverlap.groupby("model"):
            values = group.groupby("offset")[metric].mean()
            ax.plot(values.index, values.values, marker="o", ms=3, label=model_name, color=colors[model_name])
        ax.axhline(.5 if metric == "roc_auc" else nonoverlap.prevalence.mean(), color="grey", ls="--", lw=1)
        ax.set_xlabel("Date modulo 21 offset"); ax.set_ylabel(metric.replace("_", " ")); ax.grid(alpha=.2)
    axes[0].legend(); fig.tight_layout(); fig.savefig(figures / "nonoverlap_21offsets.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric in zip(axes, ["roc_auc", "average_precision"]):
        for control, group in controls.groupby("control"):
            ax.hist(group[metric], bins=12, alpha=.55, label=control)
        ax.axvline(observed.iloc[0][metric], color="#C00000", lw=2, label="observed")
        ax.set_title(metric.replace("_", " ").title()); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "label_negative_controls.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for model_name, group in alert_ci.groupby("model"):
        group = group.sort_values("alert_budget_per_100")
        axes[0].plot(group.alert_budget_per_100, group.precision, marker="o", color=colors[model_name], label=model_name)
        axes[1].plot(group.alert_budget_per_100, group.recall, marker="o", color=colors[model_name], label=model_name)
    axes[0].axhline(.013435, color="grey", ls="--", lw=1, label="prevalence")
    axes[0].set(xlabel="Alerts per 100 rows", ylabel="Precision", title="Precision at fixed alert budgets")
    axes[1].set(xlabel="Alerts per 100 rows", ylabel="Recall", title="Recall at fixed alert budgets")
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "fixed_alert_budgets.png", dpi=220); plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    def format_value(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value)
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(config, df, features, original, corrected, nonoverlap, observed, controls, skill, alert_ci, stability, feature_summary, paired, offset_inference, permutation_inference, elapsed, out):
    original_summary = original[["roc_auc", "average_precision", "brier", "mcc"]].agg(["mean", "std"])
    corrected_summary = corrected[["roc_auc", "average_precision", "brier", "mcc"]].agg(["mean", "std"])
    offset_summary = nonoverlap.groupby("model")[["roc_auc", "average_precision", "brier"]].agg(["mean", "std", "min", "max"])
    control_summary = controls.groupby("control")[["roc_auc", "average_precision"]].agg(["mean", "std", "min", "max"])
    stability_summary = stability.groupby("model")[["spearman_rank_correlation", "top10_overlap", "top10_jaccard"]].agg(["mean", "std", "min", "max"])
    top_features = feature_summary.groupby("model").head(10)
    lines = [
        "# Confirmatory weekly analysis report", "",
        "## Scientific contract", "",
        "The endpoint is the supplied injury-labelled event day. The provisional Notebook-07 28-day target is not used because continuous prospective follow-up is not identifiable from the released event-centred table.", "",
        f"Rows: {len(df):,}; athletes: {df['Athlete ID'].nunique()}; positive rows: {int(df.injury.sum())}; safe features: {len(features)}.", "",
        "## Central findings", "",
        "1. The version-compatible original-style replication reproduces the published weekly discrimination: mean ROC-AUC 0.677. However, it uses test-outcome-dependent athlete normalization and overlapping development/calibration row pools.",
        "2. After training-only processing, athlete-disjoint calibration, safe-feature restriction, and untouched test athletes, performance is lower and unstable. Averaged predictions yield ROC-AUC 0.577 versus 0.679 for the original-style protocol; the paired difference is 0.102 (95% athlete-bootstrap CI 0.027--0.199).",
        "3. The original-style probabilities are much less accurate: Brier 0.159 versus 0.0157 for the corrected protocol. High discrimination under the source protocol does not imply reliable risk probabilities.",
        "4. Removing adjacent overlap with all 21 prespecified Date-modulo-21 offsets reduces mean ROC-AUC to 0.485, 0.489, and 0.474 for logistic regression, random forest, and XGBoost. None differs reliably from chance across offsets, and mean Brier skill is approximately zero.",
        "5. The full-window safe logistic result exceeds 100 refitted within-athlete permutation and circular-shift controls, but that weak association disappears in the non-overlapping design. The evidence supports temporal/observation-process sensitivity, not operational injury forecasting.",
        "6. At fixed budgets of 1--10 alerts per 100 rows, precision remains approximately 0.2--1.9%, with confidence intervals overlapping the 1.34% prevalence baseline. No evaluated model supports an actionable warning system.", "",
        "## Original-style versus corrected last-10-athlete replication", "",
        "Original-style mean (SD):", "", "```", original_summary.to_string(), "```", "",
        "Corrected mean (SD):", "", "```", corrected_summary.to_string(), "```", "",
        "The original-style result intentionally reproduces test-outcome-dependent athlete normalization and overlapping row pools for fitting and calibration. It is a methodological comparator, not a valid deployment estimate.", "",
        "Paired athlete-bootstrap comparison after averaging the five stochastic runs:", "", markdown_table(paired), "",
        "## Twenty-one non-overlapping weekly offsets", "", "```", offset_summary.to_string(), "```", "",
        "Each offset retains only Date modulo 21 rows, so evaluated target rows no longer have adjacent three-week histories. All offsets were prespecified and all are reported.", "",
        markdown_table(offset_inference), "",
        "## Label negative controls", "", f"Observed safe logistic: ROC-AUC={observed.iloc[0].roc_auc:.4f}; AP={observed.iloc[0].average_precision:.4f}.", "", "```", control_summary.to_string(), "```", "",
        markdown_table(permutation_inference), "",
        "## Brier skill relative to fold-training prevalence", "", markdown_table(skill), "",
        "## Fixed alert budgets", "", markdown_table(alert_ci), "",
        "## Explanation-rank stability", "", "```", stability_summary.to_string(), "```", "",
        "Top features are model-reliance summaries, not causal risk factors:", "", markdown_table(top_features), "",
        "## Reproducibility", "", f"Runtime: {elapsed/60:.1f} minutes. Configuration: `{json.dumps(asdict(config), sort_keys=True)}`", "",
    ]
    (out / "CONFIRMATORY_RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/data_weekly.csv")
    parser.add_argument("--primary-predictions", default="experiments/leakage_safe_validation/results/outer_oof_predictions.csv.gz")
    parser.add_argument("--output", default="experiments/confirmatory_weekly_analysis/results")
    parser.add_argument("--original-experiments", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=25)
    parser.add_argument("--alert-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    config = Config(
        data_path=args.data, primary_predictions=args.primary_predictions, output_dir=args.output,
        original_experiments=args.original_experiments, permutation_iterations=args.permutations,
        alert_bootstrap_iterations=args.alert_bootstrap,
    )
    warnings.filterwarnings("ignore", category=FutureWarning)
    out = Path(config.output_dir); out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    data_path = Path(config.data_path)
    df = pd.read_csv(data_path).sort_values(["Athlete ID", "Date"]).reset_index(drop=True)
    safe_features, _, ratio_features, gap_features = feature_contract(df)
    if len(safe_features) != 66 or gap_features:
        raise ValueError("Unexpected feature contract")
    metadata = {
        "config": asdict(config), "data_sha256": sha256(data_path),
        "rows": len(df), "athletes": int(df["Athlete ID"].nunique()),
        "positive_rows": int(df.injury.sum()), "safe_features": safe_features,
        "excluded_ratios": ratio_features,
        "versions": {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    original, _ = original_style_replication(df, list(df.columns.difference(["injury", "Athlete ID", "Date"], sort=False)), config, out)
    corrected, _ = corrected_last10_replication(df, safe_features, config, out)
    nonoverlap = nonoverlap_analysis(df, safe_features, config, out)
    observed, controls = permutation_controls(df, safe_features, config, out)
    skill, alert_ci = brier_skill_and_alert_budgets(df, Path(config.primary_predictions), config, out)
    stability, feature_summary = explanation_stability(df, safe_features, config, out)
    paired, offset_inference, permutation_inference, nonoverlap = additional_inference(out, Path(config.primary_predictions))
    make_figures(original, corrected, nonoverlap, observed, controls, alert_ci, out)
    elapsed = time.time() - start
    write_report(config, df, safe_features, original, corrected, nonoverlap, observed, controls, skill, alert_ci, stability, feature_summary, paired, offset_inference, permutation_inference, elapsed, out)
    metadata["runtime_seconds"] = elapsed
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Completed in {elapsed/60:.1f} minutes: {out.resolve()}")


if __name__ == "__main__":
    main()
