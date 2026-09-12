"""Development-only cross-validation metrics for CreditScope baselines."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from creditscope.modeling import PREDICTIVE_FEATURES

DEFAULT_THRESHOLD = 0.50
FALSE_NEGATIVE_COST = 5
FALSE_POSITIVE_COST = 1
CV_SPLITS = 5
RANDOM_STATE = 42


@dataclass(frozen=True)
class CrossValidationResult:
    """Fold metrics and aligned out-of-fold development predictions."""

    model: str
    fold_metrics: pd.DataFrame
    oof_probabilities: pd.Series
    oof_predictions: pd.Series


CVSplit = tuple[np.ndarray, np.ndarray]


def make_cv_splits(
    X_development: pd.DataFrame, y_development: pd.Series
) -> tuple[CVSplit, ...]:
    """Precompute the one paired five deterministic CV partition."""
    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    splits = tuple(
        (train.copy(), validation.copy())
        for train, validation in cv.split(X_development, y_development)
    )
    assignment_counts = np.zeros(len(X_development), dtype=np.int8)
    for _, validation in splits:
        assignment_counts[validation] += 1
    if not np.all(assignment_counts == 1):
        raise RuntimeError("Each development row must occur in one validation fold.")
    return splits


def threshold_predictions(
    probabilities: pd.Series, threshold: float = DEFAULT_THRESHOLD
) -> pd.Series:
    """Apply the fixed inclusive probability threshold to the positive class."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    return (probabilities >= threshold).astype("int8")


def asymmetric_cost(
    y_true: pd.Series, y_pred: pd.Series
) -> dict[str, float | int]:
    """Calculate UCI cost: false negative=5 and false positive=1."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    total = int(
        FALSE_NEGATIVE_COST * false_negative
        + FALSE_POSITIVE_COST * false_positive
    )
    return {
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "false_positive_cost": FALSE_POSITIVE_COST,
        "total_cost": total,
        "average_cost": total / len(y_true),
    }


def classification_metrics(
    y_true: pd.Series, probabilities: pd.Series, predictions: pd.Series
) -> dict[str, float]:
    """Compute metrics with bad credit risk (1) as the positive class."""
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "precision_bad_credit_risk": precision_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "recall_bad_credit_risk": recall_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "f1_bad_credit_risk": f1_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "brier_score": brier_score_loss(y_true, probabilities, pos_label=1),
        "accuracy": accuracy_score(y_true, predictions),
    }


def evaluate_cross_validation(
    model_name: str,
    pipeline: Pipeline,
    X_development: pd.DataFrame,
    y_development: pd.Series,
    *,
    cv_splits: Sequence[CVSplit] | None = None,
    include_training_metrics: bool = False,
) -> CrossValidationResult:
    """Generate five-fold metrics and OOF predictions on development data only."""
    if list(X_development.columns) != list(PREDICTIVE_FEATURES):
        raise ValueError("Cross-validation received an unexpected predictive feature set.")
    if set(y_development.unique()) != {0, 1}:
        raise ValueError("Development target must contain exactly 0 and 1.")

    paired_splits = tuple(
        make_cv_splits(X_development, y_development)
        if cv_splits is None
        else cv_splits
    )
    if len(paired_splits) != CV_SPLITS:
        raise ValueError(f"Expected exactly {CV_SPLITS} cross-validation folds.")
    oof_probabilities = pd.Series(index=y_development.index, dtype="float64")
    assignment_counts = pd.Series(0, index=y_development.index, dtype="int8")
    fold_rows: list[dict[str, float | int | str]] = []

    for fold_number, (train_positions, validation_positions) in enumerate(
        paired_splits, start=1
    ):
        X_train = X_development.iloc[train_positions]
        y_train = y_development.iloc[train_positions]
        X_validation = X_development.iloc[validation_positions]
        y_validation = y_development.iloc[validation_positions]

        fitted = clone(pipeline).fit(X_train, y_train)
        probabilities = pd.Series(
            fitted.predict_proba(X_validation)[:, 1],
            index=X_validation.index,
            dtype="float64",
        )
        predictions = threshold_predictions(probabilities)
        oof_probabilities.loc[X_validation.index] = probabilities
        assignment_counts.loc[X_validation.index] += 1
        metrics = classification_metrics(y_validation, probabilities, predictions)
        cost = asymmetric_cost(y_validation, predictions)
        row: dict[str, float | int | str] = {
                "model": model_name,
                "fold": fold_number,
                "training_rows": len(X_train),
                "validation_rows": len(X_validation),
                "validation_good_credit_risk": int((y_validation == 0).sum()),
                "validation_bad_credit_risk": int((y_validation == 1).sum()),
                **metrics,
                "default_threshold_total_cost": int(cost["total_cost"]),
                "default_threshold_average_cost": float(cost["average_cost"]),
            }
        if include_training_metrics:
            training_probabilities = pd.Series(
                fitted.predict_proba(X_train)[:, 1],
                index=X_train.index,
                dtype="float64",
            )
            training_predictions = threshold_predictions(training_probabilities)
            training_metrics = classification_metrics(
                y_train, training_probabilities, training_predictions
            )
            row.update(
                {f"training_{metric}": value for metric, value in training_metrics.items()}
            )
        fold_rows.append(row)

    if oof_probabilities.isna().any() or not (assignment_counts == 1).all():
        raise RuntimeError("Each development row must receive exactly one OOF probability.")
    oof_predictions = threshold_predictions(oof_probabilities)
    return CrossValidationResult(
        model=model_name,
        fold_metrics=pd.DataFrame(fold_rows),
        oof_probabilities=oof_probabilities,
        oof_predictions=oof_predictions,
    )


def oof_summary(
    result: CrossValidationResult, y_development: pd.Series
) -> dict[str, float | int | str]:
    """Summarize aggregate OOF development metrics at threshold 0.50."""
    metrics = classification_metrics(
        y_development, result.oof_probabilities, result.oof_predictions
    )
    cost = asymmetric_cost(y_development, result.oof_predictions)
    return {
        "model": result.model,
        "evaluation_population": "development out-of-fold predictions",
        "threshold": DEFAULT_THRESHOLD,
        **metrics,
        **cost,
    }


def fold_metric_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return mean and sample standard deviation across CV folds."""
    metric_columns = [
        "roc_auc",
        "average_precision",
        "precision_bad_credit_risk",
        "recall_bad_credit_risk",
        "f1_bad_credit_risk",
        "balanced_accuracy",
        "brier_score",
        "accuracy",
        "default_threshold_total_cost",
        "default_threshold_average_cost",
    ]
    rows = []
    for model_name, group in fold_metrics.groupby("model", sort=False):
        for metric in metric_columns:
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "fold_mean": float(group[metric].mean()),
                    "fold_std": float(group[metric].std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def coefficient_table(fitted_logistic_pipeline: Pipeline) -> pd.DataFrame:
    """Extract coefficients from a baseline fitted on development data only."""
    preprocessor = fitted_logistic_pipeline.named_steps["preprocessor"]
    classifier = fitted_logistic_pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    rows = []
    for transformed_name, coefficient in zip(names, coefficients, strict=True):
        encoded_name = transformed_name.split("__", maxsplit=1)[1]
        source_feature = next(
            feature
            for feature in sorted(PREDICTIVE_FEATURES, key=len, reverse=True)
            if encoded_name == feature or encoded_name.startswith(f"{feature}_")
        )
        rows.append(
            {
                "transformed_feature": transformed_name,
                "source_feature": source_feature,
                "coefficient": float(coefficient),
                "odds_ratio_exp_coefficient": math.exp(float(coefficient)),
                "absolute_coefficient": abs(float(coefficient)),
                "direction_for_bad_credit_risk_score": (
                    "higher"
                    if coefficient > 0
                    else "lower"
                    if coefficient < 0
                    else "neutral"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "absolute_coefficient", ascending=False, ignore_index=True
    )
