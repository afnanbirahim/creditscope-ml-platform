# CreditScope model card

## Purpose and intended use

CreditScope is an educational/research decision-support prototype demonstrating reproducible credit-risk modelling, explainability, and governance. It may support learning and methodological review. It is **not** a production lending approval system and must not autonomously approve, reject, price, or otherwise determine credit access.

## Data and target

The modelling data are UCI Statlog German Credit Data (ID 144), with South German Credit (ID 573) used for corrected semantic guidance. The target is good versus bad credit risk, not observed loan default. The data describe 1,000 historical West German credits from 1973–1975 and have severe temporal/geographic transportability limits.

## Governed policy

Seventeen predictors use Stage 3R preprocessing: duration and transformed credit amount remain numeric; fifteen categorical/discretized variables are one-hot encoded. `personal_status_sex`, `age_years`, and `foreign_worker` are audit-only. The model is fixed untuned `XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method="hist")` in five-fold sigmoid `CalibratedClassifierCV(ensemble=True, n_jobs=None)`. Member calibrated probabilities are averaged and threshold 0.16 is applied. The assumed prototype cost is `5 × FN + FP`.

## Final locked evaluation

The one-time 200-row Stage 7 holdout results were ROC-AUC 0.804643, Average Precision 0.679212, precision 0.398496, recall 0.883333, F1 0.549223, balanced accuracy 0.655952, specificity 0.428571, Brier 0.152338, log loss 0.475878, and accuracy 0.565000. Confusion counts were TN 60, FP 80, FN 7, TP 53; 5:1 cost was 115. No Stage 8 result changes these values.

## Explainability

Tree SHAP is calculated on each underlying XGBoost member's raw-margin output and aggregated descriptively. It does not exactly decompose the calibrated ensemble probability. Importance and local contributions are associative model explanations, not causal effects.

## Responsible-AI findings and limitations

Direct demographic exclusions do not remove proxy risk. Checking/savings status, employment, housing, property, telephone, and job may encode socioeconomic position. Development OOF subgroup rates differ, but are descriptive, sometimes unstable, and cannot establish fairness, absence of discrimination, or legal compliance. The non-foreign-worker source group has only 31 development observations; one documented personal-status/sex category has no observations. The compound field cannot isolate sex or represent modern gender concepts. Credit amount has an undocumented monotonic transformation. Measurement timing is incomplete.

## Human oversight

Any research demonstration should expose uncertainty, explanations, policy assumptions, and avenues for human review. Real deployment would require representative contemporary data, jurisdiction-specific review, impact assessment, monitoring, recourse design, security controls, and independent validation.
