# Validation-integrity audit: confirmatory results

Run date: 2026-08-30. Both analyses completed with their full prespecified
settings. These results supersede the smoke-test directories.

## Main finding

Validation design changes the scientific conclusion in two independently
collected runner cohorts and across logistic regression, random forest, and
XGBoost. This supports a methodological paper about trustworthy AI validation;
it does not support a deployment-ready injury-warning model.

## Controlled ground-truth stress tests

Fifty regenerated null datasets were evaluated per scenario. In the identity-
only scenario, participant risk was independent of all predictor fingerprints:
random row validation nevertheless yielded mean random-forest ROC AUC 0.799
(SD 0.029), whereas participant-grouped validation yielded 0.491 (SD 0.043).
In the selection-only scenario, selecting 20 of 300 null predictors globally
yielded mean logistic ROC AUC 0.631 (SD 0.012), while fold-local selection
yielded 0.502 (SD 0.029). These tests demonstrate the audit mechanisms when
true population-generalizable ROC AUC is exactly 0.5 by construction.

## Cohort 1: sequential protocol ladder

The same final ten athletes were evaluated at every rung after averaging five
stochastic nine-model ensembles per row. Confidence intervals use 5,000 paired
athlete-cluster bootstrap samples.

| Protocol rung | ROC AUC (95% CI) | AP (95% CI) | Brier (95% CI) |
|---|---:|---:|---:|
| S0 source style | 0.675 (0.578–0.789) | 0.0848 (0.0401–0.1722) | 0.1580 (0.1357–0.1798) |
| S1 outcome-blind transductive normalization | 0.681 (0.582–0.798) | 0.1034 (0.0393–0.2324) | 0.1582 (0.1376–0.1780) |
| S2 training-only preprocessing | 0.597 (0.500–0.703) | 0.0255 (0.0166–0.0842) | 0.2340 (0.1395–0.3289) |
| S3 athlete-disjoint calibration | 0.591 (0.495–0.700) | 0.0259 (0.0160–0.1034) | 0.2459 (0.2027–0.2893) |
| S4 safe feature contract | 0.593 (0.497–0.704) | 0.0266 (0.0165–0.1077) | 0.2460 (0.2022–0.2898) |
| S5 natural-prevalence calibration | 0.591 (0.495–0.694) | 0.0259 (0.0164–0.0879) | 0.0157 (0.0097–0.0214) |
| S6 natural-row, class-weighted fitting | 0.620 (0.501–0.743) | 0.0250 (0.0143–0.0668) | 0.0156 (0.0097–0.0213) |

Key paired protocol-step changes are:

- S0 to S1 ROC AUC: +0.0065 (95% CI −0.0014 to +0.0120). Removing
  outcome-conditioned normalization alone did not remove optimism.
- S1 to S2 ROC AUC: −0.0837 (−0.1772 to −0.0080); AP: −0.0779
  (−0.1707 to −0.0134). Removing access to the test-athlete feature
  distributions was the decisive ranking change.
- S4 to S5 Brier: −0.2303 (−0.2756 to −0.1851). Calibration on a balanced
  sample severely distorted probabilities at natural prevalence.
- The fitting/calibration athlete overlap is 64 and sampled row overlap is
  approximately 561 at S0–S2; both are exactly zero at S3–S6.

These adjacent changes are not causal component effects because design choices
interact.

## Cohort 2: independent validation-integrity audit

The official Wu et al. processed release contains 6,181 weekly rows, 564
positive labels, and 142 participants. Both workbooks omit participant ID and
week. Five invariant fields reconstruct exactly 142 unique and 142 contiguous
trajectories in both matrices. All grouped folds have zero inferred-participant
overlap; random-row folds share 128–134 participants.

Using fold-local selection, row-wise versus grouped ROC-AUC estimates were:

| Matrix | Model | Row CV | Grouped CV | Paired row-minus-group difference (95% CI) |
|---|---|---:|---:|---:|
| 39 predictors | Logistic | 0.656 | 0.613 | 0.0436 (0.0226–0.0659) |
| 39 predictors | Random forest | 0.730 | 0.523 | 0.2066 (0.1477–0.2700) |
| 39 predictors | XGBoost | 0.706 | 0.558 | 0.1481 (0.1026–0.1942) |
| 257 predictors | Logistic | 0.697 | 0.644 | 0.0526 (0.0272–0.0747) |
| 257 predictors | Random forest | 0.718 | 0.586 | 0.1316 (0.0883–0.1730) |
| 257 predictors | XGBoost | 0.704 | 0.574 | 0.1299 (0.0898–0.1725) |

With participant grouping fixed in the 257-predictor matrix, global rather
than fold-local feature selection inflated ROC AUC by 0.0520 (0.0292–0.0745),
0.0532 (0.0207–0.0837), and 0.0654 (0.0297–0.1021) for logistic regression,
random forest, and XGBoost. The two mechanisms are therefore separable:
identity mixing is not repaired by fold-local selection, and participant
grouping is not sufficient when supervised selection remains global.

The inferred identity partition is internally robust: exact four- and
five-field fingerprints and contiguous three-field runs yield the identical
142 groups in both matrices. An exact three-field rule conservatively merges
them into 134 groups. Under this coarser rule, grouped fold-local ROC AUC is
0.596/0.518/0.556 for logistic/random-forest/XGBoost with 39 predictors and
0.630/0.550/0.545 with 257 predictors. 

