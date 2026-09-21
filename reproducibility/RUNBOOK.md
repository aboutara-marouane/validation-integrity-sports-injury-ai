# Reproducibility runbook

The commands below use the repository root as the working directory. The full
analysis is computationally intensive; the release `results/` directories
contain the outputs used in the manuscript, while the listed commands
regenerate them.

## 1. Environment and source-file verification

```bash
conda env create -f environment.yml
conda activate sports-injury-validation-integrity
python data/download_and_verify_data.py
```

The source datasets are not redistributed. See `data/README.md` and
`DATA_GOVERNANCE.md` for access and licence information.

## 2. Analysis sequence

The analysis sequence is:

```bash
jupyter lab notebooks/build_provisional_outcome_audit_data.ipynb
python experiments/outcome_sampling_audit/run_outcome_sampling_audit.py
python experiments/leakage_safe_validation/run_leakage_safe_validation.py
python experiments/confirmatory_weekly_analysis/run_confirmatory_weekly_analysis.py
python experiments/validation_integrity_audit/run_synthetic_stress_tests.py
python experiments/validation_integrity_audit/run_protocol_ladder.py
python experiments/validation_integrity_audit/run_external_cohort_audit.py
python experiments/validation_integrity_audit/run_group_reconstruction_sensitivity.py
python reproducibility/build_release_manifest.py
```

The first notebook creates the audit-only derived input required by the
outcome-sampling script.

## 3. Manuscript compilation

```bash
cd manuscript
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
