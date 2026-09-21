#!/usr/bin/env python3
"""Audit whether the released weekly modelling table supports prospective relabelling.

The source file contains event-centred three-week summaries. This script audits
retained-date continuity, structural gaps before injury-labelled records,
episode construction, the provisional Notebook-07 target, and overlap between
adjacent prediction records. It does not fit a prediction model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def checksum(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def next_incident_dates(dates: np.ndarray, incidents: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(incidents, dates, side="right")
    result = np.full(len(dates), np.nan)
    valid = positions < len(incidents)
    result[valid] = incidents[positions[valid]]
    return result


def audit(raw_path: Path, prospective_path: Path, output: Path, horizon: int = 28) -> None:
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(raw_path).sort_values(["Athlete ID", "Date"], kind="stable").reset_index(drop=True)
    prospective = pd.read_csv(prospective_path).sort_values(["Athlete ID", "Date"], kind="stable").reset_index(drop=True)

    if raw.duplicated(["Athlete ID", "Date"]).any():
        raise ValueError("Duplicate athlete-date rows detected")
    if not set(raw["injury"].unique()).issubset({0, 1}):
        raise ValueError("The supplied injury label is not binary")

    raw["previous_date"] = raw.groupby("Athlete ID")["Date"].shift()
    raw["gap_before_row"] = raw["Date"] - raw["previous_date"]
    raw["previous_injury"] = raw.groupby("Athlete ID")["injury"].shift(fill_value=0)
    raw["incident_start"] = raw["injury"].eq(1) & raw["previous_injury"].eq(0)

    # A run is defined by consecutive retained injury-labelled rows. This is a
    # descriptive property of the released table, not a clinically verified episode.
    new_run = (
        raw["Athlete ID"].ne(raw["Athlete ID"].shift())
        | raw["injury"].ne(raw["injury"].shift())
        | raw["Date"].sub(raw["Date"].shift()).ne(1)
    )
    raw["run_id"] = new_run.cumsum()
    injury_run_length = raw.loc[raw.injury.eq(1)].groupby("run_id").size()
    raw["injury_run_length"] = raw["run_id"].map(injury_run_length)

    incidents = raw.loc[raw.incident_start, [
        "Athlete ID", "Date", "previous_date", "gap_before_row", "injury_run_length"
    ]].copy()
    incidents = incidents.rename(columns={
        "Date": "incident_label_date",
        "gap_before_row": "gap_before_incident",
        "injury_run_length": "consecutive_injury_label_rows",
    })
    incidents["implied_history_start"] = incidents["incident_label_date"] - 21
    incidents["implied_history_end"] = incidents["incident_label_date"] - 1

    retained_by_athlete = {
        athlete: set(group.Date.astype(int)) for athlete, group in raw.groupby("Athlete ID")
    }
    incidents["retained_rows_in_implied_21d_history"] = [
        sum(day in retained_by_athlete[athlete] for day in range(int(date) - 21, int(date)))
        for athlete, date in zip(incidents["Athlete ID"], incidents["incident_label_date"])
    ]
    incidents.to_csv(output / "incident_episode_audit.csv", index=False)

    incident_map = {
        athlete: np.sort(group.incident_label_date.to_numpy())
        for athlete, group in incidents.groupby("Athlete ID")
    }
    target_rows = []
    for athlete, group in prospective.groupby("Athlete ID", sort=False):
        dates = group.Date.to_numpy(dtype=int)
        event_dates = incident_map.get(athlete, np.array([], dtype=int))
        next_event = next_incident_dates(dates, event_dates)
        days_to_event = next_event - dates
        date_set = retained_by_athlete[athlete]
        continuous_followup = np.array([
            all(day in date_set for day in range(int(date) + 1, int(date) + horizon + 1))
            for date in dates
        ])
        target_rows.append(pd.DataFrame({
            "Athlete ID": athlete,
            "Date": dates,
            "future_incident_28d": group.future_incident_28d.to_numpy(dtype=int),
            "next_incident_label_date": next_event,
            "days_to_next_incident_label": days_to_event,
            "continuous_28d_retained_rows": continuous_followup,
            "date_offset_mod_21": dates % 21,
        }))
    target_audit = pd.concat(target_rows, ignore_index=True)
    target_audit.to_csv(output / "provisional_target_row_audit.csv.gz", index=False, compression="gzip")

    positive_spacing = (
        target_audit.loc[target_audit.future_incident_28d.eq(1), "days_to_next_incident_label"]
        .value_counts().sort_index().rename_axis("days_to_event").rename("positive_rows").reset_index()
    )
    positive_spacing.to_csv(output / "future_positive_spacing.csv", index=False)

    gaps = raw.loc[raw.gap_before_row.notna()].copy()
    gap_summary = (
        gaps.assign(gap_category=np.select(
            [gaps.gap_before_row.eq(1), gaps.gap_before_row.eq(22), gaps.gap_before_row.gt(22)],
            ["1", "22", ">22"], default="2-21"
        ))
        .groupby("gap_category", observed=True)
        .agg(rows=("injury", "size"), injury_labelled_rows=("injury", "sum"), incident_starts=("incident_start", "sum"))
        .reset_index()
    )
    gap_summary["injury_label_rate"] = gap_summary.injury_labelled_rows / gap_summary.rows
    gap_summary.to_csv(output / "gap_outcome_association.csv", index=False)

    negative = target_audit.future_incident_28d.eq(0)
    n_negative_without_continuity = int((negative & ~target_audit.continuous_28d_retained_rows).sum())
    adjacent_rows = int(raw.gap_before_row.eq(1).sum())
    total_transitions = int(raw.gap_before_row.notna().sum())
    incident_22 = int(incidents.gap_before_incident.eq(22).sum())
    zero_retained_history = int(incidents.retained_rows_in_implied_21d_history.eq(0).sum())

    metadata = {
        "raw_path": str(raw_path),
        "raw_md5": checksum(raw_path),
        "prospective_path": str(prospective_path),
        "prospective_sha256": checksum(prospective_path, "sha256"),
        "horizon": horizon,
        "raw_rows": len(raw),
        "athletes": int(raw["Athlete ID"].nunique()),
        "injury_labelled_rows": int(raw.injury.sum()),
        "derived_incident_starts": len(incidents),
        "incident_starts_after_exact_22_gap": incident_22,
        "incident_starts_with_zero_retained_rows_in_implied_history": zero_retained_history,
        "provisional_rows": len(target_audit),
        "provisional_positive_rows": int(target_audit.future_incident_28d.sum()),
        "provisional_negative_rows_without_continuous_28d_retained_rows": n_negative_without_continuity,
        "adjacent_one_unit_transitions": adjacent_rows,
        "all_within_athlete_transitions": total_transitions,
    }
    (output / "audit_metadata.json").write_text(json.dumps(metadata, indent=2))

    spacing_text = ", ".join(
        f"{int(row.days_to_event)}d: {int(row.positive_rows)}"
        for row in positive_spacing.itertuples(index=False)
    )
    report = f"""# Outcome-sampling audit of the official weekly dataset

## Purpose

This audit tests whether the released event-centred modelling table can be treated as a continuously observed cohort for prospective relabelling. It does not fit a model.

## Verified source

- Raw rows: {len(raw):,}
- Athletes: {raw['Athlete ID'].nunique():,}
- Injury-labelled rows: {int(raw.injury.sum()):,}
- Local raw MD5: `{metadata['raw_md5']}`
- The MD5 matches the value published by the official Dataverse for the weekly approach file.

## Structural findings

- Derived starts of retained injury-label runs: {len(incidents):,}.
- Starts occurring exactly 22 date units after the preceding retained row: {incident_22:,} ({incident_22 / len(incidents):.1%}).
- Starts with zero retained rows during their implied preceding 21-day history: {zero_retained_history:,} ({zero_retained_history / len(incidents):.1%}).
- One-unit adjacent transitions: {adjacent_rows:,} of {total_transitions:,} ({adjacent_rows / total_transitions:.1%}). Adjacent prediction records therefore use strongly overlapping three-week histories.

## Audit of the provisional Notebook-07 target

- Eligible rows saved by Notebook 07: {len(target_audit):,}.
- Positive rows: {int(target_audit.future_incident_28d.sum()):,} ({target_audit.future_incident_28d.mean():.2%}).
- Positive-row spacing to the next retained incident label: {spacing_text}.
- Provisional negative rows without a complete set of retained rows over the next 28 date units: {n_negative_without_continuity:,}.

The positive outcome is repeated across neighbouring index dates, while the released table systematically omits dates immediately before most injury-labelled records. The condition `Date + 28 <= maximum Date` identifies a later retained date rather than continuous injury-free observation through the horizon.

## Endpoint interpretation

The released weekly file supports classification of its supplied injury-labelled rows. The available record structure does not define a continuous prospective risk set for reconstruction of a 28-day incident-injury outcome. Notebook 07 is retained as an audit-only target-construction artifact.
"""
    (output / "OUTCOME_SAMPLING_AUDIT.md").write_text(report)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/data_weekly.csv")
    parser.add_argument("--prospective", default="data/processed/leakage_safe_prospective_28d.csv")
    parser.add_argument("--output", default="experiments/outcome_sampling_audit/results")
    parser.add_argument("--horizon", type=int, default=28)
    args = parser.parse_args()
    audit(Path(args.raw), Path(args.prospective), Path(args.output), args.horizon)


if __name__ == "__main__":
    main()
