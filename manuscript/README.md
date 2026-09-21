# Research manuscript

`main.tex` is a journal-neutral two-cohort methodological manuscript. It uses
the canonical evidence from `notebooks/confirmatory_weekly_validation.ipynb`,
the original confirmatory experiments, and the validation-integrity audit in
`experiments/validation_integrity_audit`.

Compile from this directory so that the relative figure paths resolve:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The draft includes the formal seven-domain Validation-Integrity Profile,
controlled ground-truth tests, the seven-rung protocol decomposition, an
independent 142-participant cohort audit, conservative grouping sensitivity,
paired participant-bootstrap contrasts, and publication-resolution figures.
`supplement.tex` provides the reconstruction proof and applied checklist.
Cohort 1 is presented as retrospective classification of an injury-labelled
event day.
