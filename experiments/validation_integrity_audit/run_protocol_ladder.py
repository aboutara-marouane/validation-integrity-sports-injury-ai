#!/usr/bin/env python3
"""Sequential validation-protocol audit for the original weekly cohort.

Each rung changes one design component while retaining the same final ten
athletes as an untouched comparison set.  The differences are descriptive
protocol-step changes, not causal effects, because validation choices interact.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.confirmatory_weekly_analysis.run_confirmatory_weekly_analysis import (
    balanced_by_athlete, equal_sensitivity_specificity_threshold, platt_apply,
    platt_fit, xgb_original,
)
from experiments.leakage_safe_validation.run_leakage_safe_validation import (
    feature_contract, shuffled_group_folds,
)


@dataclass(frozen=True)
class Config:
    data_path: str = "data/raw/data_weekly.csv"
    output_dir: str = "experiments/validation_integrity_audit/results/protocol_ladder"
    experiments: int = 5
    bags: int = 9
    samples_per_class: int = 2048
    bootstrap_iterations: int = 5000
    seed: int = 20260830


STAGES = [
    {"stage": "S0_source_style", "normalization": "healthy", "disjoint_calibration": False, "safe_features": False, "natural_calibration": False, "natural_fit": False},
    {"stage": "S1_outcome_blind_normalization", "normalization": "all", "disjoint_calibration": False, "safe_features": False, "natural_calibration": False, "natural_fit": False},
    {"stage": "S2_training_only_preprocessing", "normalization": "none", "disjoint_calibration": False, "safe_features": False, "natural_calibration": False, "natural_fit": False},
    {"stage": "S3_athlete_disjoint_calibration", "normalization": "none", "disjoint_calibration": True, "safe_features": False, "natural_calibration": False, "natural_fit": False},
    {"stage": "S4_safe_feature_contract", "normalization": "none", "disjoint_calibration": True, "safe_features": True, "natural_calibration": False, "natural_fit": False},
    {"stage": "S5_natural_prevalence_calibration", "normalization": "none", "disjoint_calibration": True, "safe_features": True, "natural_calibration": True, "natural_fit": False},
    {"stage": "S6_natural_rows_weighted_fitting", "normalization": "none", "disjoint_calibration": True, "safe_features": True, "natural_calibration": True, "natural_fit": True},
]


def athlete_moments(frame: pd.DataFrame, features: list[str], healthy_only: bool):
    source = frame[frame.injury.eq(0)] if healthy_only else frame
    mean = source.groupby("Athlete ID")[features].mean()
    std = source.groupby("Athlete ID")[features].std().replace(0.0, 0.01).fillna(0.01)
    return mean, std


def transform(frame: pd.DataFrame, features: list[str], moments):
    if moments is None:
        return frame[features].to_numpy()
    mean, std = moments
    athletes = frame["Athlete ID"].to_numpy()
    mu, sigma = mean.reindex(athletes).to_numpy(), std.reindex(athletes).to_numpy()
    if np.isnan(mu).any() or np.isnan(sigma).any():
        raise ValueError("Unavailable athlete moments")
    return (frame[features].to_numpy() - mu) / sigma


def safe_metrics(y, p):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def natural_xgb(seed: int, py_rng: random.Random, prevalence: float):
    return XGBClassifier(
        objective="binary:logistic", learning_rate=0.01,
        max_depth=py_rng.choice([2, 3]), n_estimators=py_rng.choice([256, 512]),
        min_child_weight=5.0, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5.0, reg_alpha=0.1,
        scale_pos_weight=(1 - prevalence) / max(prevalence, 1e-9),
        eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=seed,
    )


def run_stage(df, all_features, safe_features, specification, config):
    athletes = np.asarray(sorted(df["Athlete ID"].unique()))
    test_athletes = athletes[-10:]
    development = df[~df["Athlete ID"].isin(test_athletes)].copy()
    test = df[df["Athlete ID"].isin(test_athletes)].copy()
    development["_source_index"] = development.index
    features = safe_features if specification["safe_features"] else all_features
    rows, predictions = [], []

    for experiment in range(config.experiments):
        seed = config.seed + experiment * 101
        rng, py_rng = np.random.default_rng(seed), random.Random(seed)
        if specification["disjoint_calibration"]:
            splits = list(shuffled_group_folds(development["Athlete ID"], 5, seed))
            fit_idx, cal_idx = splits[experiment % 5]
            fit_pool, cal_pool = development.iloc[fit_idx].copy(), development.iloc[cal_idx].copy()
        else:
            fit_pool = cal_pool = development

        if specification["normalization"] == "healthy":
            development_moments = athlete_moments(development, features, healthy_only=True)
            test_moments = athlete_moments(test, features, healthy_only=True)
        elif specification["normalization"] == "all":
            development_moments = athlete_moments(development, features, healthy_only=False)
            test_moments = athlete_moments(test, features, healthy_only=False)
        else:
            development_moments = test_moments = None

        calibration = cal_pool if specification["natural_calibration"] else balanced_by_athlete(
            cal_pool, config.samples_per_class, rng
        )
        X_cal = transform(calibration, features, development_moments)
        y_cal = calibration.injury.to_numpy(dtype=int)
        X_test = transform(test, features, test_moments)
        raw_calibration, raw_test = [], []
        overlap_counts = []

        for bag in range(config.bags):
            if specification["natural_fit"]:
                bag_frame = fit_pool
                prevalence = float(bag_frame.injury.mean())
                fitted = natural_xgb(seed + bag, py_rng, prevalence)
            else:
                bag_frame = balanced_by_athlete(fit_pool, config.samples_per_class, rng)
                fitted = xgb_original(seed + bag, py_rng)
            fitted.fit(transform(bag_frame, features, development_moments), bag_frame.injury)
            raw_calibration.append(fitted.predict_proba(X_cal)[:, 1])
            raw_test.append(fitted.predict_proba(X_test)[:, 1])
            if "_source_index" in bag_frame and "_source_index" in calibration:
                overlap_counts.append(len(set(bag_frame._source_index) & set(calibration._source_index)))

        mean_cal, mean_test = np.mean(raw_calibration, axis=0), np.mean(raw_test, axis=0)
        calibrator = platt_fit(y_cal, mean_cal)
        cal_probability, probability = platt_apply(calibrator, mean_cal), platt_apply(calibrator, mean_test)
        threshold = equal_sensitivity_specificity_threshold(y_cal, cal_probability)
        result = safe_metrics(test.injury.to_numpy(), probability)
        result.update({
            "stage": specification["stage"], "experiment": experiment,
            "features": len(features), "fit_rows": len(fit_pool),
            "calibration_rows": len(calibration),
            "fit_calibration_athlete_overlap": len(set(fit_pool["Athlete ID"]) & set(cal_pool["Athlete ID"])),
            "sampled_row_overlap_mean": float(np.mean(overlap_counts)) if overlap_counts else 0.0,
            "threshold": threshold, "test_prevalence": float(test.injury.mean()),
        })
        rows.append(result)
        predictions.append(pd.DataFrame({
            "stage": specification["stage"], "experiment": experiment,
            "row_index": test.index, "athlete_id": test["Athlete ID"].to_numpy(),
            "y_true": test.injury.to_numpy(dtype=int), "probability": probability,
        }))
        print(f"{specification['stage']} experiment {experiment + 1}/{config.experiments} ROC={result['roc_auc']:.3f}", flush=True)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def paired_bootstrap(predictions: pd.DataFrame, iterations: int, seed: int):
    averaged = predictions.groupby(["stage", "row_index", "athlete_id", "y_true"], as_index=False).probability.mean()
    athletes = averaged.athlete_id.unique()
    rng = np.random.default_rng(seed)
    point = {stage: safe_metrics(g.y_true, g.probability) for stage, g in averaged.groupby("stage")}
    boot_rows = []
    for iteration in range(iterations):
        sampled = rng.choice(athletes, len(athletes), replace=True)
        for stage, group in averaged.groupby("stage"):
            sample = pd.concat([group[group.athlete_id.eq(a)] for a in sampled], ignore_index=True)
            if sample.y_true.nunique() == 2:
                boot_rows.append({"iteration": iteration, "stage": stage, **safe_metrics(sample.y_true, sample.probability)})
    boot = pd.DataFrame(boot_rows)
    summary = []
    for stage in [s["stage"] for s in STAGES]:
        previous = None if stage == STAGES[0]["stage"] else STAGES[[s["stage"] for s in STAGES].index(stage) - 1]["stage"]
        for metric, estimate in point[stage].items():
            values = boot[boot.stage.eq(stage)].set_index("iteration")[metric]
            row = {
                "stage": stage, "metric": metric, "estimate": estimate,
                "ci_low": values.quantile(.025), "ci_high": values.quantile(.975),
                "previous_stage": previous,
            }
            if previous:
                prior = boot[boot.stage.eq(previous)].set_index("iteration")[metric]
                delta = values - prior
                row.update({
                    "delta_from_previous": estimate - point[previous][metric],
                    "delta_ci_low": delta.quantile(.025), "delta_ci_high": delta.quantile(.975),
                })
            summary.append(row)
    return pd.DataFrame(summary)


def plot(summary, out):
    labels = [s["stage"].replace("_", " ") for s in STAGES]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, metric, title in zip(axes, ["roc_auc", "average_precision", "brier"], ["ROC AUC", "Average precision", "Brier score"]):
        data = summary[summary.metric.eq(metric)].set_index("stage").reindex([s["stage"] for s in STAGES])
        x = np.arange(len(data))
        ax.errorbar(x, data.estimate, yerr=[data.estimate-data.ci_low, data.ci_high-data.estimate], marker="o", capsize=4)
        ax.set_xticks(x, labels, rotation=35, ha="right"); ax.set_title(title); ax.grid(alpha=.25)
    fig.suptitle("Sequential validation-protocol audit (same ten test athletes)")
    fig.tight_layout(); fig.savefig(out / "protocol_ladder.pdf", bbox_inches="tight")
    fig.savefig(out / "protocol_ladder.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=Config.data_path)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--experiments", type=int, default=Config.experiments)
    parser.add_argument("--bags", type=int, default=Config.bags)
    parser.add_argument("--samples-per-class", type=int, default=Config.samples_per_class)
    parser.add_argument("--bootstrap-iterations", type=int, default=Config.bootstrap_iterations)
    parser.add_argument("--seed", type=int, default=Config.seed)
    config = Config(**vars(parser.parse_args()))
    out = PROJECT_ROOT / config.output_dir; out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROJECT_ROOT / config.data_path)
    safe, _, _, _ = feature_contract(df)
    all_features = [c for c in df.columns if c not in {"injury", "Athlete ID", "Date"} and pd.api.types.is_numeric_dtype(df[c])]
    metrics, predictions = [], []
    for specification in STAGES:
        stage_metrics, stage_predictions = run_stage(df, all_features, safe, specification, config)
        metrics.append(stage_metrics); predictions.append(stage_predictions)
    metrics = pd.concat(metrics, ignore_index=True); predictions = pd.concat(predictions, ignore_index=True)
    summary = paired_bootstrap(predictions, config.bootstrap_iterations, config.seed + 50000)
    metrics.to_csv(out / "protocol_ladder_experiment_metrics.csv", index=False)
    predictions.to_csv(out / "protocol_ladder_predictions.csv.gz", index=False, compression="gzip")
    summary.to_csv(out / "protocol_ladder_cluster_bootstrap.csv", index=False)
    (out / "stage_contract.json").write_text(json.dumps(STAGES, indent=2) + "\n")
    plot(summary, out)
    print(f"Wrote protocol ladder to {out}")


if __name__ == "__main__":
    main()
