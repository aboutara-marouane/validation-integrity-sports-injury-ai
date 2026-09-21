# Outcome-sampling audit of the official weekly dataset

## Purpose

This audit tests whether the released event-centred modelling table can be treated as a continuously observed cohort for prospective relabelling. It does not fit a model.

## Verified source

- Raw rows: 42,798
- Athletes: 74
- Injury-labelled rows: 575
- Local raw MD5: `5fe8c80e2a2fa267789be363aacad43a`
- The MD5 matches the value published by the official Dataverse for the weekly approach file.

## Structural findings

- Derived starts of retained injury-label runs: 389.
- Starts occurring exactly 22 date units after the preceding retained row: 347 (89.2%).
- Starts with zero retained rows during their implied preceding 21-day history: 389 (100.0%).
- One-unit adjacent transitions: 41,751 of 42,724 (97.7%). Adjacent prediction records therefore use strongly overlapping three-week histories.


