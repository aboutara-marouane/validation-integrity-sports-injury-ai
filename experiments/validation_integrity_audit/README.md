# Validation-integrity audit

This directory turns the paper from a model-comparison exercise into a
reusable methodological audit of sports-injury AI. It asks whether a reported
result survives changes in the statistical unit, transformation locality,
feature-selection locality, calibration independence, prevalence handling,
and temporal-history overlap.

The framework reports an integrity profile rather than an arbitrary composite
score. A high score could conceal one fatal violation; the profile keeps every
validation dimension inspectable.

## Two complementary experiments

`run_protocol_ladder.py` uses the original Løvdal weekly cohort and the same
last ten test athletes at every rung. It changes one protocol component at a
time: label-conditioned normalization, outcome-blind transductive
normalization, training-only processing, athlete-disjoint calibration, the
safe feature contract, natural-prevalence calibration, and non-resampled
class-weighted fitting. Since these choices interact, adjacent differences are described as
protocol-step changes rather than causal effects.

`run_external_cohort_audit.py` applies the audit to the independent processed
weekly cohort released with Wu et al. (2026). It compares random row-wise and
participant-grouped validation, and global versus fold-local supervised
feature selection, across logistic regression, random forest, and XGBoost.
The deliberately invalid global-selection conditions are comparators, not
candidate deployment models.

`run_synthetic_stress_tests.py` supplies ground-truth controls. One null
scenario isolates participant memorization, and another isolates global
supervised-selection leakage. Neither contains population-generalizable
predictor signal, so a valid protocol should return ROC-AUC near 0.5.

`run_group_reconstruction_sensitivity.py` proves which alternative fingerprint
rules produce the same released-row partition and repeats the grouped,
fold-local audit after conservatively merging the 142 inferred trajectories
into 134 groups.


## Analysis commands

From the project root:

```bash
python experiments/validation_integrity_audit/run_protocol_ladder.py
python experiments/validation_integrity_audit/run_external_cohort_audit.py
python experiments/validation_integrity_audit/run_synthetic_stress_tests.py
python experiments/validation_integrity_audit/run_group_reconstruction_sensitivity.py
```

Both programs save row-level out-of-fold predictions, fold metrics,
participant-cluster bootstrap intervals, data hashes, run manifests, and
publication-resolution PDF/PNG figures under `results/`.
