# Stage 3R reproducibility reconciliation

## Decision

Stage 3R remains the authoritative historical Logistic Regression baseline. Its
artifacts under `reports/stage3_remediated/` are preserved unchanged. For the
paired Stage 4 comparison, Logistic Regression is recomputed on the same locked
development rows and folds in the current numerical environment.

Exact equality is required for raw-data and split hashes, row and class counts,
stored fold membership, confusion counts, integer cost totals, feature boundaries,
and fixed model configuration values. Reproduced floating-point metrics use
`rtol=0` and `atol=1e-10`, without pre-comparison rounding. This tolerance does
not permit changed classifications, folds, preprocessing, hyperparameters,
datasets, or feature sets. It must not be increased without explicit authorization.

The `1e-10` tolerance was selected because the largest accepted, investigated
Stage 3R fold-level numerical drift was approximately `2.10e-11`.

## Reconciliation evidence

The aggregate Stage 3R comparison was exact for ROC-AUC, Average Precision,
precision, F1, balanced accuracy, accuracy, all four confusion counts, and total
and average 5:1 cost. Recall differed by approximately `5.55e-17`. Brier score
differed by approximately `4.08e-12`; the maximum fold-level Brier difference was
approximately `2.10e-11`. The locked split hash was unchanged, and fold membership
was reconstructed deterministically from the same ordered development rows with
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

Historical Stage 3R row-level OOF probabilities were not saved, so bitwise
probability-array identity cannot be established retrospectively. Stage 4 closes
that evidence gap by persisting development-only fold membership and row-level OOF
predictions with SHA-256 hashes.

## Environments

The original Stage 3R run is known to have used scikit-learn 1.4.2. Its exact
Python, NumPy, and SciPy versions were not recorded, which is a reproducibility
limitation.

The reconciled Stage 4 environment is:

- Python 3.12.0
- scikit-learn 1.4.2
- NumPy 2.5.3
- SciPy 1.18.1
- XGBoost 3.4.1

The matching discrete outputs, unchanged data/split/methodology, and very small
floating differences support the documented conclusion that the discrepancy is
numerical stack drift rather than methodological drift.
