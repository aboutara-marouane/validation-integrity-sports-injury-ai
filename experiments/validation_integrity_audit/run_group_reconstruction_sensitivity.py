#!/usr/bin/env python3
"""Sensitivity analysis for inferred participant grouping in Wu et al. data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.validation_integrity_audit.run_external_cohort_audit import (
    Config, FINGERPRINT_COLUMNS, participant_bootstrap, run_protocol,
    select_columns,
)


def equivalent_partition(first: pd.Series, second: pd.Series) -> bool:
    mapping_a = pd.DataFrame({"a": first, "b": second}).groupby("a").b.nunique().max()
    mapping_b = pd.DataFrame({"a": first, "b": second}).groupby("b").a.nunique().max()
    return bool(mapping_a == 1 and mapping_b == 1)


def grouping_definitions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    first3 = FINGERPRINT_COLUMNS[:3]
    first4 = FINGERPRINT_COLUMNS[:4]
    full = frame.groupby(FINGERPRINT_COLUMNS, sort=False, dropna=False).ngroup().astype(int)
    four = frame.groupby(first4, sort=False, dropna=False).ngroup().astype(int)
    coarse = frame.groupby(first3, sort=False, dropna=False).ngroup().astype(int)
    changes = frame[first3].ne(frame[first3].shift()).any(axis=1)
    changes.iloc[0] = True
    contiguous = (changes.cumsum() - 1).astype(int)
    return {
        "five_field_exact": pd.Series(full),
        "four_field_exact": pd.Series(four),
        "three_field_coarsened": pd.Series(coarse),
        "three_field_contiguous_runs": pd.Series(contiguous),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/external/wu_2026")
    parser.add_argument("--output-dir", default="experiments/validation_integrity_audit/results/group_reconstruction_sensitivity")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    out = PROJECT_ROOT / args.output_dir; out.mkdir(parents=True, exist_ok=True)
    all_predictions, all_summaries, audit_rows = [], [], []

    for dataset, filename in [("class1_features", "class1_features.xlsx"), ("all_features", "all_features.xlsx")]:
        frame = pd.read_excel(PROJECT_ROOT / args.data_dir / filename)
        definitions = grouping_definitions(frame)
        primary = definitions["five_field_exact"]
        for name, groups in definitions.items():
            audit_rows.append({
                "dataset": dataset, "grouping_rule": name,
                "unique_groups": int(groups.nunique()),
                "contiguous_runs": int(groups.ne(groups.shift()).sum()),
                "minimum_rows": int(groups.value_counts().min()),
                "maximum_rows": int(groups.value_counts().max()),
                "partition_equivalent_to_five_field": equivalent_partition(primary, groups),
            })

        # The three-field exact grouping changes the partition by merging
        # eight full-fingerprint trajectories.
        groups = definitions["three_field_coarsened"]
        X, y = frame.drop(columns="RRI").astype(float), frame.RRI.astype(int)
        global_columns = select_columns(X, y, 20)
        for model_name in ["logistic", "random_forest", "xgboost"]:
            config = Config(
                data_dir=args.data_dir, output_dir=args.output_dir,
                repeats=3, folds=5, selected_features=20,
                bootstrap_iterations=args.bootstrap_iterations, seed=args.seed,
            )
            predictions, _ = run_protocol(
                X, y, groups, dataset, "group_local_selection", model_name,
                config, global_columns,
            )
            predictions["grouping_rule"] = "three_field_coarsened"
            all_predictions.append(predictions)
            summary = participant_bootstrap(predictions, args.bootstrap_iterations, args.seed + len(all_summaries) * 1009)
            summary.insert(0, "grouping_rule", "three_field_coarsened")
            summary.insert(0, "model", model_name); summary.insert(0, "dataset", dataset)
            all_summaries.append(summary)

    audit = pd.DataFrame(audit_rows)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)
    audit.to_csv(out / "group_reconstruction_audit.csv", index=False)
    predictions.to_csv(out / "coarsened_group_oof_predictions.csv.gz", index=False, compression="gzip")
    summary.to_csv(out / "coarsened_group_bootstrap_summary.csv", index=False)
    (out / "sensitivity_contract.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(audit.to_string(index=False))
    print(summary[summary.metric.isin(["roc_auc", "average_precision"])].to_string(index=False))


if __name__ == "__main__":
    main()
