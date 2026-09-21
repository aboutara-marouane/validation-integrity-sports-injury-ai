# Validation Integrity in Longitudinal Sports-Injury AI

This repository reproduces the analyses, figures, tables, and manuscript for
the accompanying two-cohort methodological audit.

## Environment and data verification

```bash
conda env create -f environment.yml
conda activate sports-injury-validation-integrity
python data/download_and_verify_data.py
```

The source data are not redistributed. See `data/README.md` for the two
official source records, expected filenames, and checksum verification.

## Reproduction sequence

The outcome-sampling component uses
`notebooks/build_provisional_outcome_audit_data.ipynb` followed by
`experiments/outcome_sampling_audit/run_outcome_sampling_audit.py`. The
confirmatory, leakage-safe, and validation-integrity analyses are implemented
in the corresponding `experiments/` directories. Manuscript compilation is
described in `manuscript/README.md`.

All reported analyses use relative paths from the repository root. Results
already included in `experiments/*/results/` are the release outputs used by
the manuscript.
