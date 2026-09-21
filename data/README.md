# Data access and verification

The source data are **not included** in this repository. They remain subject
to the access and reuse terms of their original providers.

## Source files and expected locations

The Løvdal et al. DataverseNL file
`week_approach_maskedID_timeseries.csv` is available at
<https://doi.org/10.34894/UWU9PV>. The analysis path is
`data/raw/data_weekly.csv`.

The Wu et al. (2026) supplementary workbooks are available at
<https://doi.org/10.1038/s41746-026-02413-y>. The 39-predictor workbook
(`MOESM2`) corresponds to `data/external/wu_2026/class1_features.xlsx`; the
257-predictor workbook (`MOESM3`) corresponds to
`data/external/wu_2026/all_features.xlsx`.

The verification command is:

```bash
python data/download_and_verify_data.py
```

The program does not download or redistribute data. It verifies that the
files obtained by the user match the checksums recorded in
`source_manifest.csv`, and exits with a non-zero status if a file is missing
or differs. The expected paths and checksums are also used by the analysis
scripts.

The main cohort-1 and cohort-2 analyses use only these three source files.
For the outcome-sampling audit, first run
`notebooks/build_provisional_outcome_audit_data.ipynb`. It deterministically
derives the audit-only file `data/processed/leakage_safe_prospective_28d.csv`
from `data/raw/data_weekly.csv`; it is an audit-only derived file rather than
an additional source dataset. No Paper 3 data are used.
