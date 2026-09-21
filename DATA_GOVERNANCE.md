# Data-governance basis for public release

The repository does not include the three source datasets. Users obtain them
from the original providers and verify them using `data/source_manifest.csv`.

## Licence assessment (checked 2026-09-21)

- **Cohort 1 (Løvdal et al., DataverseNL, DOI 10.34894/UWU9PV):** the
  repository record specifies CC0 1.0. CC0 permits unrestricted reuse and
  redistribution, including derived row-level outputs.
- **Cohort 2 (Wu et al., npj Digital Medicine, DOI
  10.1038/s41746-026-02413-y):** the article and its linked supplementary
  data downloads are published under CC BY 4.0. The article page lists the
  processed workbooks as supplementary data and contains no separate credit
  line restricting them. CC BY 4.0 permits sharing and adaptation, including
  derived row-level outputs, provided that the source is credited, the CC BY
  4.0 licence is linked, and the analysis-derived nature of the outputs is
  indicated.

Accordingly, the following source-derived files may be retained in the public
release:

- `experiments/leakage_safe_validation/results/outer_oof_predictions.csv.gz`
- `experiments/leakage_safe_validation/results/split_manifest.csv.gz`
- `experiments/confirmatory_weekly_analysis/results/*predictions.csv.gz`
- `experiments/validation_integrity_audit/results/external_wu_2026/external_oof_predictions.csv.gz`
- `experiments/validation_integrity_audit/results/protocol_ladder/protocol_ladder_predictions.csv.gz`
- `experiments/validation_integrity_audit/results/group_reconstruction_sensitivity/coarsened_group_oof_predictions.csv.gz`

For cohort 2, retain this attribution in repository documentation or the
release notes: “Derived from the processed supplementary data of Wu et al.,
*npj Digital Medicine* (2026), DOI: 10.1038/s41746-026-02413-y, licensed
CC BY 4.0; modified through the validation-integrity analyses.” Cite the
source article in the manuscript and do not imply that its authors endorse
this reanalysis.

Do not upload any source data not already publicly released by its provider,
or introduce new direct identifiers. This assessment addresses the published
licence terms and is not legal advice.

This checklist concerns data rights only; the repository code is released
under the MIT License.
