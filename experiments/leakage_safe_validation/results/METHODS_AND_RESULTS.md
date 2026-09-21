# Leakage-safe repeated validation report

## Scientific status

This is a confirmatory re-analysis of the existing row-level injury label. It evaluates transfer to unseen athletes. It does **not** claim prospective 7/14-day forecasting because injury-episode timing and feature-availability semantics require source confirmation.

## Data and feature contract

- Rows: 42,798; athletes: 74; injury rows: 575; prevalence: 1.3435%.
- Primary predictors: 66 numeric variables.
- Excluded from the primary model: target, Athlete ID, Date, gap/time-gap variables, and the three unstable supplied relative-load ratios.
- Outer validation: 3 repetitions of 5-fold athlete-group CV.
- Inner selection: 3-fold athlete-group CV using average precision.
- Calibration (none, Platt, isotonic) selected by inner out-of-fold Brier score.
- Decision threshold selected by inner out-of-fold MCC only.
- 95% uncertainty intervals: athlete-cluster bootstrap over aggregated repeated OOF predictions.

## Main results (estimate and athlete-bootstrap 95% CI)

| Model | ROC-AUC | Average precision | MCC | Recall | Precision | Brier | Calibration slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.608 (0.580--0.639) | 0.018 (0.015--0.022) | 0.042 (0.027--0.058) | 0.609 (0.537--0.673) | 0.019 (0.015--0.024) | 0.013 (0.011--0.016) | 0.816 (0.576--1.265) |
| random_forest | 0.615 (0.581--0.654) | 0.018 (0.015--0.023) | 0.045 (0.029--0.062) | 0.687 (0.606--0.754) | 0.019 (0.015--0.023) | 0.013 (0.011--0.016) | 0.965 (0.690--1.310) |
| xgboost | 0.592 (0.558--0.625) | 0.017 (0.014--0.023) | 0.035 (0.020--0.052) | 0.563 (0.477--0.643) | 0.018 (0.014--0.023) | 0.013 (0.011--0.016) | 0.972 (0.615--1.379) |

Because injury prevalence is very low, average precision, calibration, and false-alert burden are more informative than accuracy alone.

At the inner-selected MCC thresholds, the models generate approximately 41--49 alerts per 100 rows, with approximately 1.8--1.9% of alerts corresponding to injury rows.

## Paired model differences (athlete-bootstrap 95% CI)

| Model A - Model B | Metric | Difference | 95% CI |
|---|---|---:|---:|
| logistic - random_forest | Average precision | -0.0005 | -0.0033--0.0017 |
| logistic - random_forest | Brier | 0.00004 | -0.00001--0.00013 |
| logistic - random_forest | MCC | -0.0038 | -0.0157--0.0083 |
| logistic - random_forest | ROC-AUC | -0.0077 | -0.0349--0.0192 |
| logistic - xgboost | Average precision | 0.0005 | -0.0052--0.0027 |
| logistic - xgboost | Brier | 0.00003 | -0.00002--0.00012 |
| logistic - xgboost | MCC | 0.0064 | -0.0064--0.0179 |
| logistic - xgboost | ROC-AUC | 0.0160 | -0.0104--0.0414 |
| random_forest - xgboost | Average precision | 0.0010 | -0.0038--0.0031 |
| random_forest - xgboost | Brier | -0.00001 | -0.00002--0.00001 |
| random_forest - xgboost | MCC | 0.0101 | 0.0013--0.0178 |
| random_forest - xgboost | ROC-AUC | 0.0236 | 0.0010--0.0432 |

Random forest is better than XGBoost for ROC-AUC and MCC in this paired analysis, but not for average precision or Brier score. No clear differences are observed between random forest and logistic regression. The practical conclusion is therefore not that a complex learner solves the task, but that all three models have limited rare-event utility after leakage control.

## Deliberate leakage negative control

The recording gap is intentionally reintroduced only in this section to quantify how a deployment-unavailable artifact inflates performance.

| Setting | Mean ROC-AUC | Mean average precision | Mean Brier |
|---|---:|---:|---:|
| gap_only_negative_control | 0.983 | 0.519 | 0.010 |
| safe_plus_gap_negative_control | 0.985 | 0.527 | 0.011 |
| safe_primary | 0.614 | 0.018 | 0.236 |

## Relative-ratio sensitivity

| Setting | Mean ROC-AUC | Mean average precision | Mean Brier |
|---|---:|---:|---:|
| safe_with_raw_ratios | 0.624 | 0.021 | 0.235 |
| safe_without_ratios | 0.612 | 0.019 | 0.235 |

## Interpretation limits

1. The dataset contains 74 independent athletes; repeated rows are nested within athletes.
2. External validation is not available.
3. The current label is retained as supplied; incident injury episodes and a future prediction horizon are not reconstructed in this analysis.
4. Confidence intervals quantify between-athlete variability within this cohort.
5. The gap experiment functions as a negative control for observation-process effects.

## Reproducibility

Runtime: 12.9 minutes. Complete predictions, split manifests, tuning results, calibration choices, bootstrap samples, figures, and configuration are stored beside this report.
