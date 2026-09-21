# Confirmatory weekly analysis report

## Scientific contract

The endpoint is the supplied injury-labelled event day. The provisional Notebook-07 28-day target is not used because continuous prospective follow-up is not identifiable from the released event-centred table.

Rows: 42,798; athletes: 74; positive rows: 575; safe features: 66.

## Central findings

1. The version-compatible original-style replication reproduces the published weekly discrimination: mean ROC-AUC 0.677. However, it uses test-outcome-dependent athlete normalization and overlapping development/calibration row pools.
2. After training-only processing, athlete-disjoint calibration, safe-feature restriction, and untouched test athletes, performance is lower and unstable. Averaged predictions yield ROC-AUC 0.577 versus 0.679 for the original-style protocol; the paired difference is 0.102 (95% athlete-bootstrap CI 0.027--0.199).
3. The original-style probabilities are much less accurate: Brier 0.159 versus 0.0157 for the corrected protocol. High discrimination under the source protocol does not imply reliable risk probabilities.
4. Removing adjacent overlap with all 21 prespecified Date-modulo-21 offsets reduces mean ROC-AUC to 0.485, 0.489, and 0.474 for logistic regression, random forest, and XGBoost. None differs reliably from chance across offsets, and mean Brier skill is approximately zero.
5. The full-window safe logistic result exceeds 100 refitted within-athlete permutation and circular-shift controls, but that weak association disappears in the non-overlapping design. The evidence supports temporal/observation-process sensitivity, not operational injury forecasting.
6. At fixed budgets of 1--10 alerts per 100 rows, precision remains approximately 0.2--1.9%, with confidence intervals overlapping the 1.34% prevalence baseline. No evaluated model supports an actionable warning system.

## Original-style versus corrected last-10-athlete replication

Original-style mean (SD):

```
       roc_auc  average_precision     brier       mcc
mean  0.677099           0.090942  0.159687  0.062527
std   0.002167           0.018552  0.006789  0.002981
```

Corrected mean (SD):

```
       roc_auc  average_precision     brier      mcc
mean  0.539869           0.022170  0.015664  0.01346
std   0.081549           0.005417  0.000008  0.01508
```

The original-style result intentionally reproduces test-outcome-dependent athlete normalization and overlapping row pools for fitting and calibration. It is a methodological comparator, not a valid deployment estimate.

Paired athlete-bootstrap comparison after averaging the five stochastic runs:

| metric | original_estimate | corrected_estimate | difference_original_minus_corrected | ci_lower | ci_upper |
|---|---|---|---|---|---|
| average_precision | 0.086387 | 0.0254742 | 0.0609128 | 0.00785903 | 0.11686 |
| brier | 0.158896 | 0.0156519 | 0.143244 | 0.126766 | 0.160158 |
| roc_auc | 0.6788 | 0.57707 | 0.10173 | 0.0274162 | 0.198716 |

## Twenty-one non-overlapping weekly offsets

```
                roc_auc                               average_precision                                   brier                              
                   mean       std       min       max              mean       std       min       max      mean       std       min       max
model                                                                                                                                        
logistic       0.485404  0.096985  0.291111  0.672577          0.015833  0.007600  0.005169  0.039619  0.013245  0.004535  0.006850  0.022803
random_forest  0.488792  0.104417  0.286875  0.716898          0.018511  0.012885  0.005745  0.072962  0.013236  0.004546  0.006831  0.022809
xgboost        0.473574  0.100454  0.247329  0.731273          0.018092  0.010440  0.004496  0.060447  0.013247  0.004548  0.006846  0.022811
```

Each offset retains only Date modulo 21 rows, so evaluated target rows no longer have adjacent three-week histories. All offsets were prespecified and all are reported.

| model | full_window_roc_auc | nonoverlap_mean_roc_auc | roc_auc_change | roc_wilcoxon_vs_0_5_p | full_window_average_precision | nonoverlap_mean_average_precision | mean_offset_prevalence | ap_wilcoxon_vs_prevalence_p | nonoverlap_mean_brier_skill |
|---|---|---|---|---|---|---|---|---|---|
| logistic | 0.607667 | 0.485404 | -0.122263 | 0.516761 | 0.0176719 | 0.0158331 | 0.0134226 | 0.190687 | -0.000674601 |
| random_forest | 0.61532 | 0.488792 | -0.126529 | 0.494802 | 0.0181552 | 0.0185111 | 0.0134226 | 0.103214 | 0.000357884 |
| xgboost | 0.591689 | 0.473574 | -0.118115 | 0.0957994 | 0.0171921 | 0.0180918 | 0.0134226 | 0.0501919 | -0.000516328 |

## Label negative controls

Observed safe logistic: ROC-AUC=0.6207; AP=0.0187.

```
                                roc_auc                               average_precision                              
                                   mean       std       min       max              mean       std       min       max
control                                                                                                              
within_athlete_circular_shift  0.533454  0.023937  0.480672  0.588738          0.015214  0.001352  0.012618  0.019035
within_athlete_permutation     0.541793  0.016024  0.479273  0.576253          0.015559  0.000944  0.012834  0.017616
```

| control | metric | observed | null_mean | null_sd | null_95th_percentile | null_max | empirical_one_sided_p | iterations |
|---|---|---|---|---|---|---|---|---|
| within_athlete_circular_shift | roc_auc | 0.620725 | 0.533454 | 0.0239375 | 0.571985 | 0.588738 | 0.00990099 | 100 |
| within_athlete_circular_shift | average_precision | 0.0187358 | 0.0152143 | 0.001352 | 0.0174918 | 0.0190351 | 0.019802 | 100 |
| within_athlete_permutation | roc_auc | 0.620725 | 0.541793 | 0.0160236 | 0.570732 | 0.576253 | 0.00990099 | 100 |
| within_athlete_permutation | average_precision | 0.0187358 | 0.0155595 | 0.0009437 | 0.0172459 | 0.0176162 | 0.00990099 | 100 |

## Brier skill relative to fold-training prevalence

| model | model_brier | prevalence_baseline_brier | brier_skill_score |
|---|---|---|---|
| logistic | 0.0132685 | 0.0132588 | -0.000732519 |
| random_forest | 0.013232 | 0.0132588 | 0.00202109 |
| xgboost | 0.0132384 | 0.0132588 | 0.00153699 |

## Fixed alert budgets

| model | alert_budget_per_100 | precision_lower | precision_upper | recall_lower | recall_upper | alerts | true_positives | precision | recall | probability_threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| logistic | 1 | 0.00451243 | 0.0286448 | 0.00332117 | 0.0212783 | 428 | 6 | 0.0140187 | 0.0104348 | 0.0236208 |
| logistic | 2 | 0.0094451 | 0.0287457 | 0.013654 | 0.0432191 | 856 | 15 | 0.0175234 | 0.026087 | 0.0220755 |
| logistic | 5 | 0.0126304 | 0.0239788 | 0.0471204 | 0.0897377 | 2140 | 39 | 0.0182243 | 0.0678261 | 0.0204699 |
| logistic | 10 | 0.012172 | 0.0214141 | 0.0905756 | 0.163829 | 4280 | 69 | 0.0161215 | 0.12 | 0.0198662 |
| random_forest | 1 | 0 | 0.0356166 | 0 | 0.0254175 | 428 | 1 | 0.00233645 | 0.00173913 | 0.0279985 |
| random_forest | 2 | 0 | 0.0304673 | 0 | 0.0449414 | 856 | 14 | 0.0163551 | 0.0243478 | 0.0265669 |
| random_forest | 5 | 0.00792511 | 0.0268008 | 0.029133 | 0.101405 | 2140 | 35 | 0.0163551 | 0.0608696 | 0.0243731 |
| random_forest | 10 | 0.0124453 | 0.0282585 | 0.093479 | 0.204766 | 4280 | 81 | 0.0189252 | 0.14087 | 0.0222757 |
| xgboost | 1 | 0.00215042 | 0.0303646 | 0.00163112 | 0.0233923 | 428 | 6 | 0.0140187 | 0.0104348 | 0.022662 |
| xgboost | 2 | 0.002319 | 0.022415 | 0.00368654 | 0.0338353 | 856 | 9 | 0.010514 | 0.0156522 | 0.0218267 |
| xgboost | 5 | 0.00720712 | 0.0242324 | 0.026982 | 0.0903631 | 2140 | 29 | 0.0135514 | 0.0504348 | 0.0206327 |
| xgboost | 10 | 0.0117531 | 0.022842 | 0.0867036 | 0.172626 | 4280 | 70 | 0.0163551 | 0.121739 | 0.0195728 |

## Explanation-rank stability

```
              spearman_rank_correlation                               top10_overlap                   top10_jaccard                              
                                   mean       std       min       max          mean       std min max          mean       std       min       max
model                                                                                                                                            
logistic                       0.480282  0.103300  0.232063  0.726292      5.495238  1.217847   3   9      0.389066  0.123712  0.176471  0.818182
random_forest                  0.885612  0.033821  0.783321  0.943263      7.714286  0.755929   6  10      0.634131  0.102559  0.428571  1.000000
xgboost                        0.428308  0.088200  0.224966  0.661497      5.542857  1.315796   3   9      0.395070  0.130329  0.176471  0.818182
```

Top features are model-reliance summaries, not causal risk factors:

| model | feature | mean_rank | median_rank | min_rank | max_rank | top10_frequency |
|---|---|---|---|---|---|---|
| logistic | total kms | 1.53333 | 1 | 1 | 5 | 1 |
| logistic | total km Z3-Z4-Z5-T1-T2 | 2.46667 | 2 | 1 | 7 | 1 |
| logistic | avg training success | 5 | 5 | 2 | 10 | 1 |
| logistic | avg exertion | 9.06667 | 6 | 2 | 34 | 0.733333 |
| logistic | total kms.2 | 9.33333 | 8 | 2 | 24 | 0.533333 |
| logistic | nr. rest days | 10 | 7 | 1 | 25 | 0.666667 |
| logistic | max training success.1 | 11.4 | 6 | 2 | 35 | 0.733333 |
| logistic | min recovery.2 | 12.8667 | 11 | 6 | 27 | 0.4 |
| logistic | max km one day.1 | 17.5333 | 13 | 6 | 57 | 0.2 |
| logistic | max exertion | 19.2667 | 16 | 4 | 65 | 0.4 |
| random_forest | max exertion | 1.2 | 1 | 1 | 3 | 1 |
| random_forest | total km Z3-Z4-Z5-T1-T2 | 2.46667 | 2 | 1 | 5 | 1 |
| random_forest | avg exertion | 4.53333 | 4 | 2 | 8 | 1 |
| random_forest | max exertion.2 | 5.93333 | 6 | 1 | 12 | 0.933333 |
| random_forest | max exertion.1 | 6.2 | 5 | 2 | 17 | 0.8 |
| random_forest | total kms | 6.46667 | 7 | 3 | 11 | 0.933333 |
| random_forest | max recovery | 8.33333 | 8 | 3 | 13 | 0.6 |
| random_forest | total kms.2 | 8.66667 | 9 | 5 | 13 | 0.733333 |
| random_forest | total kms.1 | 8.93333 | 8 | 5 | 24 | 0.933333 |
| random_forest | max km one day | 9.46667 | 11 | 3 | 13 | 0.466667 |
| xgboost | max exertion | 1 | 1 | 1 | 1 | 1 |
| xgboost | total km Z3-Z4-Z5-T1-T2 | 4.26667 | 3 | 2 | 8 | 1 |
| xgboost | nr. strength trainings | 8.46667 | 4 | 4 | 33 | 0.866667 |
| xgboost | nr. rest days | 10.8667 | 9 | 2 | 35 | 0.666667 |
| xgboost | avg exertion | 11.5333 | 9 | 2 | 32 | 0.666667 |
| xgboost | max exertion.1 | 12.1333 | 7 | 2 | 57 | 0.666667 |
| xgboost | max exertion.2 | 12.8667 | 9 | 2 | 34 | 0.666667 |
| xgboost | total kms | 14.6 | 13 | 5 | 38 | 0.266667 |
| xgboost | max training success | 15.4 | 11 | 2 | 46 | 0.466667 |
| xgboost | total kms.2 | 17.1333 | 18 | 9 | 28 | 0.2 |

## Reproducibility

Runtime: 22.0 minutes. Configuration: `{"alert_bootstrap_iterations": 1000, "data_path": "data/raw/data_weekly.csv", "inner_folds": 3, "original_bags": 9, "original_experiments": 5, "outer_folds": 5, "outer_repeats": 3, "output_dir": "experiments/confirmatory_weekly_analysis/results", "permutation_iterations": 100, "primary_predictions": "experiments/leakage_safe_validation/results/outer_oof_predictions.csv.gz", "samples_per_class": 2048, "seed": 20260829}`
