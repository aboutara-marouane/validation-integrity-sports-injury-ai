#!/usr/bin/env python3
"""Controlled stress tests for identity and global-selection leakage.

The simulated outcomes contain no population-generalizable predictor signal.
Any above-chance grouped/fold-local performance is therefore sampling error;
systematic optimism under invalid protocols directly measures the tested
validation failure mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.leakage_safe_validation.run_leakage_safe_validation import shuffled_group_folds


@dataclass(frozen=True)
class Config:
    output_dir: str = "experiments/validation_integrity_audit/results/synthetic_stress_tests"
    replications: int = 50
    participants: int = 100
    rows_per_participant: int = 30
    null_features: int = 300
    selected_features: int = 20
    folds: int = 5
    seed: int = 20260830


def select(X, y, k):
    fitted = SelectKBest(f_classif, k=min(k, X.shape[1])).fit(X, y)
    return np.flatnonzero(fitted.get_support())


def evaluate(X, y, groups, protocol, model_name, k, folds, seed):
    global_columns = select(X, y, k) if "global" in protocol else None
    if protocol.startswith("row"):
        splits = StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y)
    else:
        splits = shuffled_group_folds(pd.Series(groups), folds, seed)
    probability = np.full(len(y), np.nan)
    overlap = []
    for fold, (train, test) in enumerate(splits):
        columns = global_columns if global_columns is not None else (
            select(X[train], y[train], k) if "local" in protocol else np.arange(X.shape[1])
        )
        if model_name == "random_forest":
            fitted = RandomForestClassifier(
                n_estimators=80, max_depth=None, min_samples_leaf=3,
                max_features="sqrt", class_weight="balanced_subsample",
                n_jobs=-1, random_state=seed + fold,
            )
        else:
            fitted = make_pipeline(
                StandardScaler(), LogisticRegression(
                    C=.25, class_weight="balanced", solver="liblinear",
                    max_iter=2000, random_state=seed + fold,
                ),
            )
        fitted.fit(X[train][:, columns], y[train])
        probability[test] = fitted.predict_proba(X[test][:, columns])[:, 1]
        overlap.append(len(set(groups[train]) & set(groups[test])))
    if np.isnan(probability).any():
        raise RuntimeError("Incomplete synthetic OOF predictions")
    return {
        "roc_auc": roc_auc_score(y, probability),
        "average_precision": average_precision_score(y, probability),
        "participant_overlap_mean": float(np.mean(overlap)),
    }


def identity_data(config, rng):
    n, t = config.participants, config.rows_per_participant
    groups = np.repeat(np.arange(n), t)
    fingerprint = rng.normal(size=(n, 8))
    X = np.column_stack([
        np.repeat(fingerprint, t, axis=0),
        rng.normal(size=(n * t, 8)),
    ])
    # Risk varies by participant but is independent of every fingerprint.
    risk = expit(-2.4 + rng.normal(0, 1.7, n))
    y = rng.binomial(1, np.repeat(risk, t))
    return X, y, groups


def selection_data(config, rng):
    rows = config.participants * config.rows_per_participant
    X = rng.normal(size=(rows, config.null_features))
    y = rng.binomial(1, .10, rows)
    # Unique groups make identity separation irrelevant in this null scenario.
    return X, y, np.arange(rows)


def summarize(results):
    rows = []
    for (scenario, protocol, model_name), group in results.groupby(["scenario", "protocol", "model"]):
        for metric in ["roc_auc", "average_precision"]:
            rows.append({
                "scenario": scenario, "protocol": protocol, "model": model_name,
                "metric": metric, "mean": group[metric].mean(), "sd": group[metric].std(ddof=1),
                "q025": group[metric].quantile(.025), "q975": group[metric].quantile(.975),
                "replications": len(group),
            })
    return pd.DataFrame(rows)


def plot(results, out):
    order = ["row_all", "group_all", "row_global", "row_local"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (scenario, group) in zip(axes, results.groupby("scenario", sort=False)):
        available = [p for p in order if p in set(group.protocol)]
        data = [group[group.protocol.eq(p)].roc_auc for p in available]
        ax.boxplot(data, tick_labels=[p.replace("_", " ") for p in available], showfliers=False)
        ax.axhline(.5, color="grey", linestyle="--", linewidth=1)
        ax.set_title(scenario.replace("_", " ")); ax.set_ylabel("ROC AUC")
        ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Controlled validation-integrity stress tests (true generalizable AUC = 0.5)")
    fig.tight_layout(); fig.savefig(out / "synthetic_stress_tests.pdf", bbox_inches="tight")
    fig.savefig(out / "synthetic_stress_tests.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    for field, info in Config.__dataclass_fields__.items():
        default = info.default
        parser.add_argument("--" + field.replace("_", "-"), type=type(default), default=default)
    config = Config(**vars(parser.parse_args()))
    out = PROJECT_ROOT / config.output_dir; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for replication in range(config.replications):
        seed = config.seed + replication * 1009
        rng = np.random.default_rng(seed)
        X, y, groups = identity_data(config, rng)
        for protocol in ["row_all", "group_all"]:
            result = evaluate(X, y, groups, protocol, "random_forest", X.shape[1], config.folds, seed)
            rows.append({"scenario": "identity_only", "protocol": protocol, "model": "random_forest", "replication": replication, "prevalence": y.mean(), **result})
        X, y, groups = selection_data(config, rng)
        for protocol in ["row_global", "row_local"]:
            result = evaluate(X, y, groups, protocol, "logistic", config.selected_features, config.folds, seed)
            rows.append({"scenario": "selection_only", "protocol": protocol, "model": "logistic", "replication": replication, "prevalence": y.mean(), **result})
        print(f"simulation {replication + 1}/{config.replications}", flush=True)
    results = pd.DataFrame(rows); summary = summarize(results)
    results.to_csv(out / "synthetic_replication_metrics.csv", index=False)
    summary.to_csv(out / "synthetic_summary.csv", index=False)
    (out / "simulation_contract.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    plot(results, out)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
