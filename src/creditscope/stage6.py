"""Stage 6 development-only calibration and threshold policy governance."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from creditscope.analysis import build_analysis_dataset
from creditscope.data import load_verified_raw_snapshot
from creditscope.evaluation import asymmetric_cost, threshold_predictions
from creditscope.modeling import (
    AUDIT_ATTRIBUTES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PREDICTIVE_FEATURES,
    XGBOOST_PARAMETERS,
    build_xgboost_pipeline,
)
from creditscope.schema import TARGET_BINARY_NAME
from creditscope.splits import DEVELOPMENT_SPLIT, ROW_ID_NAME, source_row_ids
from creditscope.stage5 import (
    STAGE4_FOLD_SHA256,
    STAGE4_OOF_SHA256,
    load_outer_folds,
)

LOCKED_SPLIT_SHA256 = (
    "32fb4c9ac2cdb358e3bffd145cd97e1a393c14946888d46212d0b395523e99af"
)
STAGE5_OOF_SHA256 = (
    "eecccdb9ae24d747562b5914c5f54bc5838688df4019b013552d1b87217460f5"
)
RAW_SNAPSHOT_SHA256 = (
    "4ce6007c9eb3dbcd67e7858773566b0eb3ff140033cf67b2d8a42b3a89edfb9d"
)
CALIBRATION_METHODS = ("none", "sigmoid", "isotonic")
CALIBRATION_PARSIMONY_TOLERANCE = 1e-4
INNER_SPLITS = 4
RANDOM_STATE = 42
THRESHOLD_GRID = tuple(float(value) for value in np.round(np.arange(0.05, 0.51, 0.01), 2))
THEORETICAL_THRESHOLD = 1.0 / 6.0
STAGE7_METRICS = (
    "roc_auc",
    "average_precision",
    "precision_bad_credit_risk",
    "recall_bad_credit_risk",
    "f1_bad_credit_risk",
    "balanced_accuracy",
    "specificity",
    "brier_score",
    "log_loss",
    "accuracy",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
    "total_cost",
    "average_cost",
)


@dataclass(frozen=True)
class NestedPolicyResult:
    """Honest outer-fold evidence for calibration and threshold selection."""

    oof_predictions: pd.DataFrame
    outer_metrics: pd.DataFrame
    outer_choices: pd.DataFrame
    inner_calibration_comparison: pd.DataFrame
    inner_threshold_search: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_inner_splits(X: pd.DataFrame, y: pd.Series) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return the fixed four-fold inner split for a training partition."""
    splitter = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return tuple(
        (train.copy(), validation.copy())
        for train, validation in splitter.split(X, y)
    )


def build_probability_estimator(
    method: str,
    *,
    calibration_cv: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> Any:
    """Build the fixed Stage 4 XGBoost with an optional calibration wrapper."""
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration method: {method}")
    base = build_xgboost_pipeline()
    if method == "none":
        return base
    cv = (
        StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        if calibration_cv is None
        else calibration_cv
    )
    return CalibratedClassifierCV(
        estimator=base,
        method=method,
        cv=cv,
        ensemble=True,
    )


def validate_training_validation_isolation(
    training_ids: pd.Index, validation_ids: pd.Index
) -> None:
    """Fail if any policy-fitting row is also an evaluation row."""
    if not set(training_ids).isdisjoint(validation_ids):
        raise ValueError("Training and validation identifiers overlap.")


def calibration_oof_probabilities(
    X: pd.DataFrame,
    y: pd.Series,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Generate honest OOF probabilities for exactly three calibration choices."""
    output = pd.DataFrame(index=X.index, columns=CALIBRATION_METHODS, dtype="float64")
    assignment = pd.Series(0, index=X.index, dtype="int8")
    for train_positions, validation_positions in splits:
        X_train = X.iloc[train_positions]
        y_train = y.iloc[train_positions]
        X_validation = X.iloc[validation_positions]
        validate_training_validation_isolation(X_train.index, X_validation.index)
        assignment.loc[X_validation.index] += 1
        for method in CALIBRATION_METHODS:
            fitted = build_probability_estimator(method).fit(X_train, y_train)
            output.loc[X_validation.index, method] = fitted.predict_proba(
                X_validation
            )[:, 1]
    if output.isna().any().any() or not assignment.eq(1).all():
        raise RuntimeError("Calibration OOF evidence is incomplete.")
    if not output.apply(lambda column: column.between(0.0, 1.0).all()).all():
        raise RuntimeError("Calibration probabilities must be within [0, 1].")
    return output


def probability_metrics(y: pd.Series, probabilities: pd.Series) -> dict[str, float]:
    """Return threshold-independent probability diagnostics."""
    return {
        "brier_score": float(brier_score_loss(y, probabilities, pos_label=1)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
    }


def calibration_comparison_table(
    y: pd.Series,
    probabilities: pd.DataFrame,
    *,
    outer_fold: int | str,
) -> pd.DataFrame:
    """Compare the fixed calibration candidates on aligned OOF rows."""
    rows = []
    for method in CALIBRATION_METHODS:
        rows.append(
            {
                "outer_fold": outer_fold,
                "calibration_method": method,
                **probability_metrics(y, probabilities[method]),
            }
        )
    return pd.DataFrame(rows)


def select_calibration_method(comparison: pd.DataFrame) -> str:
    """Minimize Brier score with the predefined 1e-4 parsimony rule."""
    if set(comparison["calibration_method"]) != set(CALIBRATION_METHODS):
        raise ValueError("Calibration candidate set changed.")
    minimum = float(comparison["brier_score"].min())
    eligible = set(
        comparison.loc[
            comparison["brier_score"] <= minimum + CALIBRATION_PARSIMONY_TOLERANCE,
            "calibration_method",
        ]
    )
    return next(method for method in CALIBRATION_METHODS if method in eligible)


def decision_metrics(
    y: pd.Series, probabilities: pd.Series, threshold: float
) -> dict[str, float | int]:
    """Evaluate one fixed threshold with bad credit risk as positive."""
    predictions = threshold_predictions(probabilities, threshold)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    specificity = tn / (tn + fp)
    cost = asymmetric_cost(y, predictions)
    return {
        "threshold": float(threshold),
        **probability_metrics(y, probabilities),
        "precision_bad_credit_risk": float(
            precision_score(y, predictions, pos_label=1, zero_division=0)
        ),
        "recall_bad_credit_risk": float(
            recall_score(y, predictions, pos_label=1, zero_division=0)
        ),
        "f1_bad_credit_risk": float(
            f1_score(y, predictions, pos_label=1, zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "specificity": float(specificity),
        "accuracy": float(accuracy_score(y, predictions)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "total_cost": int(cost["total_cost"]),
        "average_cost": float(cost["average_cost"]),
    }


def threshold_search_table(y: pd.Series, probabilities: pd.Series) -> pd.DataFrame:
    """Evaluate the exact predefined 0.05--0.50 threshold grid."""
    return pd.DataFrame(
        [decision_metrics(y, probabilities, threshold) for threshold in THRESHOLD_GRID]
    )


def select_threshold(search: pd.DataFrame) -> float:
    """Apply the frozen cost/FN/balanced-accuracy/reference tie-break order."""
    if tuple(search["threshold"].astype(float)) != THRESHOLD_GRID:
        raise ValueError("Threshold grid changed.")
    ranked = search.assign(
        negative_balanced_accuracy=-search["balanced_accuracy"],
        reference_distance=(search["threshold"] - THEORETICAL_THRESHOLD).abs(),
    ).sort_values(
        [
            "total_cost",
            "false_negative",
            "negative_balanced_accuracy",
            "reference_distance",
            "threshold",
        ],
        kind="stable",
    )
    return float(ranked.iloc[0]["threshold"])


def run_nested_policy_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> NestedPolicyResult:
    """Evaluate the combined calibration-choice and threshold procedure honestly."""
    oof_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    choice_rows: list[dict[str, Any]] = []
    calibration_rows: list[pd.DataFrame] = []
    threshold_rows: list[pd.DataFrame] = []
    assigned = pd.Series(0, index=X.index, dtype="int8")

    for outer_fold, (train_positions, validation_positions) in enumerate(
        outer_folds, start=1
    ):
        X_train = X.iloc[train_positions]
        y_train = y.iloc[train_positions]
        X_validation = X.iloc[validation_positions]
        y_validation = y.iloc[validation_positions]
        validate_training_validation_isolation(X_train.index, X_validation.index)

        inner_splits = make_inner_splits(X_train, y_train)
        inner_probabilities = calibration_oof_probabilities(
            X_train, y_train, inner_splits
        )
        inner_comparison = calibration_comparison_table(
            y_train, inner_probabilities, outer_fold=outer_fold
        )
        selected_method = select_calibration_method(inner_comparison)
        calibration_rows.append(inner_comparison)

        inner_thresholds = threshold_search_table(
            y_train, inner_probabilities[selected_method]
        )
        selected_threshold = select_threshold(inner_thresholds)
        inner_thresholds.insert(0, "selected_calibration_method", selected_method)
        inner_thresholds.insert(0, "outer_fold", outer_fold)
        inner_thresholds["selected_threshold"] = inner_thresholds[
            "threshold"
        ].eq(selected_threshold)
        threshold_rows.append(inner_thresholds)

        fitted = build_probability_estimator(
            selected_method,
            calibration_cv=inner_splits if selected_method != "none" else None,
        ).fit(X_train, y_train)
        outer_probabilities = pd.Series(
            fitted.predict_proba(X_validation)[:, 1],
            index=X_validation.index,
            dtype="float64",
        )
        outer_predictions = threshold_predictions(
            outer_probabilities, selected_threshold
        )
        assigned.loc[X_validation.index] += 1
        fold_metrics = decision_metrics(
            y_validation, outer_probabilities, selected_threshold
        )
        metric_rows.append({"outer_fold": outer_fold, **fold_metrics})
        selected_inner = inner_comparison.set_index("calibration_method").loc[
            selected_method
        ]
        choice_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_calibration_method": selected_method,
                "selected_threshold": selected_threshold,
                "inner_selected_brier_score": float(selected_inner["brier_score"]),
                "inner_selected_log_loss": float(selected_inner["log_loss"]),
                "inner_selected_roc_auc": float(selected_inner["roc_auc"]),
                "inner_selected_average_precision": float(
                    selected_inner["average_precision"]
                ),
                "outer_brier_score": float(fold_metrics["brier_score"]),
            }
        )
        oof_rows.append(
            pd.DataFrame(
                {
                    ROW_ID_NAME: X_validation.index,
                    "outer_fold": outer_fold,
                    TARGET_BINARY_NAME: y_validation.to_numpy(),
                    "selected_calibration_method": selected_method,
                    "selected_threshold": selected_threshold,
                    "probability": outer_probabilities.to_numpy(),
                    "prediction": outer_predictions.to_numpy(),
                }
            )
        )

    oof = pd.concat(oof_rows, ignore_index=True).set_index(ROW_ID_NAME).loc[X.index]
    oof = oof.reset_index()
    if not assigned.eq(1).all() or oof[ROW_ID_NAME].duplicated().any():
        raise RuntimeError("Every development row must receive one outer prediction.")
    return NestedPolicyResult(
        oof_predictions=oof,
        outer_metrics=pd.DataFrame(metric_rows),
        outer_choices=pd.DataFrame(choice_rows),
        inner_calibration_comparison=pd.concat(calibration_rows, ignore_index=True),
        inner_threshold_search=pd.concat(threshold_rows, ignore_index=True),
    )


def aggregate_nested_policy(oof: pd.DataFrame) -> dict[str, Any]:
    """Aggregate the honest outer-validation policy predictions."""
    y = oof[TARGET_BINARY_NAME].astype("int8")
    probabilities = oof["probability"].astype("float64")
    predictions = oof["prediction"].astype("int8")
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    cost = asymmetric_cost(y, predictions)
    return {
        "policy": "Nested calibration-and-threshold selection (Stage 6)",
        "evaluation_population": "800-row development outer-fold OOF",
        **probability_metrics(y, probabilities),
        "precision_bad_credit_risk": float(
            precision_score(y, predictions, pos_label=1, zero_division=0)
        ),
        "recall_bad_credit_risk": float(
            recall_score(y, predictions, pos_label=1, zero_division=0)
        ),
        "f1_bad_credit_risk": float(
            f1_score(y, predictions, pos_label=1, zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "specificity": float(tn / (tn + fp)),
        "accuracy": float(accuracy_score(y, predictions)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "total_cost": int(cost["total_cost"]),
        "average_cost": float(cost["average_cost"]),
    }


def full_development_calibration_oof(
    X: pd.DataFrame,
    y: pd.Series,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    stage4_uncalibrated: pd.Series,
) -> pd.DataFrame:
    """Generate full-development OOF evidence after honest nested evaluation."""
    output = pd.DataFrame(index=X.index, dtype="float64")
    output["none"] = stage4_uncalibrated.loc[X.index].astype("float64")
    for method in ("sigmoid", "isotonic"):
        values = pd.Series(index=X.index, dtype="float64")
        assigned = pd.Series(0, index=X.index, dtype="int8")
        for train_positions, validation_positions in outer_folds:
            X_train = X.iloc[train_positions]
            y_train = y.iloc[train_positions]
            X_validation = X.iloc[validation_positions]
            validate_training_validation_isolation(X_train.index, X_validation.index)
            calibration_cv = make_inner_splits(X_train, y_train)
            fitted = build_probability_estimator(
                method, calibration_cv=calibration_cv
            ).fit(X_train, y_train)
            values.loc[X_validation.index] = fitted.predict_proba(X_validation)[:, 1]
            assigned.loc[X_validation.index] += 1
        if values.isna().any() or not assigned.eq(1).all():
            raise RuntimeError(f"Full-development {method} OOF evidence is incomplete.")
        output[method] = values
    return output


def calibration_stability_table(choices: pd.DataFrame) -> pd.DataFrame:
    """Summarize calibration selections across fixed outer folds."""
    rows = []
    for method in CALIBRATION_METHODS:
        selected = choices[choices["selected_calibration_method"] == method]
        rows.append(
            {
                "calibration_method": method,
                "selection_count": len(selected),
                "selection_percentage": float(len(selected) / len(choices)),
                "mean_selected_inner_brier": (
                    float(selected["inner_selected_brier_score"].mean())
                    if len(selected)
                    else np.nan
                ),
                "mean_outer_brier_when_selected": (
                    float(selected["outer_brier_score"].mean())
                    if len(selected)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def threshold_stability_table(choices: pd.DataFrame) -> pd.DataFrame:
    """Return the predefined stability statistics for outer selected thresholds."""
    thresholds = choices["selected_threshold"].astype(float)
    return pd.DataFrame(
        [
            {
                "mean": float(thresholds.mean()),
                "median": float(thresholds.median()),
                "minimum": float(thresholds.min()),
                "maximum": float(thresholds.max()),
                "sample_sd": float(thresholds.std(ddof=1)),
                "default_threshold_reference": 0.50,
                "theoretical_cost_reference": THEORETICAL_THRESHOLD,
            }
        ]
    )


def reliability_table(
    y: pd.Series, probabilities: pd.DataFrame, bins: int = 10
) -> pd.DataFrame:
    """Create quantile-bin development-only reliability points."""
    rows = []
    for method in CALIBRATION_METHODS:
        frame = pd.DataFrame({"target": y, "probability": probabilities[method]})
        frame["bin"] = pd.qcut(
            frame["probability"], q=bins, labels=False, duplicates="drop"
        )
        grouped = frame.groupby("bin", observed=True)
        for bin_id, group in grouped:
            rows.append(
                {
                    "calibration_method": method,
                    "bin": int(bin_id) + 1,
                    "count": len(group),
                    "mean_predicted_probability": float(group["probability"].mean()),
                    "observed_bad_credit_risk_rate": float(group["target"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_figures(
    nested: NestedPolicyResult,
    full_comparison: pd.DataFrame,
    reliability: pd.DataFrame,
    threshold_search: pd.DataFrame,
    key_thresholds: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Create focused development-only Stage 6 diagnostics."""
    plt.style.use("seaborn-v0_8-whitegrid")
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(full_comparison["calibration_method"], full_comparison["brier_score"])
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Calibration candidates — DEVELOPMENT ONLY")
    path = output_dir / "calibration_comparison.png"
    _save_figure(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")
    for method in CALIBRATION_METHODS:
        subset = reliability[reliability["calibration_method"] == method]
        ax.plot(
            subset["mean_predicted_probability"],
            subset["observed_bad_credit_risk_rate"],
            marker="o",
            label=method,
        )
    ax.set_xlabel("Mean predicted bad-credit-risk probability")
    ax.set_ylabel("Observed bad-credit-risk rate")
    ax.set_title("Reliability curves — DEVELOPMENT ONLY")
    ax.legend()
    path = output_dir / "reliability_curves.png"
    _save_figure(fig, path)
    paths.append(path)

    selected_threshold = float(
        threshold_search.loc[threshold_search["selected_threshold"], "threshold"].iloc[0]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_search["threshold"], threshold_search["total_cost"])
    ax.axvline(selected_threshold, color="purple", linestyle="--", label="Selected")
    ax.axvline(THEORETICAL_THRESHOLD, color="black", linestyle=":", label="1/6 reference")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("5:1 total cost")
    ax.set_title("Cost versus threshold — DEVELOPMENT ONLY")
    ax.legend()
    path = output_dir / "cost_vs_threshold.png"
    _save_figure(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_search["threshold"], threshold_search["precision_bad_credit_risk"], label="Precision")
    ax.plot(threshold_search["threshold"], threshold_search["recall_bad_credit_risk"], label="Recall")
    ax.axvline(selected_threshold, color="purple", linestyle="--", label="Selected")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Precision and recall versus threshold — DEVELOPMENT ONLY")
    ax.legend()
    path = output_dir / "precision_recall_vs_threshold.png"
    _save_figure(fig, path)
    paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (_, row) in zip(axes, key_thresholds.iterrows(), strict=True):
        matrix = np.array(
            [[row["true_negative"], row["false_positive"]], [row["false_negative"], row["true_positive"]]]
        )
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, int(value), ha="center", va="center")
        ax.set_title(str(row["threshold_label"]))
        ax.set_xticks([0, 1], ["Good (0)", "Bad (1)"])
        ax.set_yticks([0, 1], ["Good (0)", "Bad (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Key-threshold confusion matrices — DEVELOPMENT ONLY")
    path = output_dir / "key_threshold_confusion_matrices.png"
    _save_figure(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        nested.outer_choices["outer_fold"].astype(str),
        nested.outer_choices["selected_threshold"],
    )
    ax.axhline(THEORETICAL_THRESHOLD, color="black", linestyle=":", label="1/6 reference")
    ax.axhline(0.50, color="gray", linestyle="--", label="Default 0.50")
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Selected threshold")
    ax.set_title("Outer-fold threshold selections — DEVELOPMENT ONLY")
    ax.legend()
    path = output_dir / "outer_fold_thresholds.png"
    _save_figure(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    method_codes = {method: index for index, method in enumerate(CALIBRATION_METHODS)}
    ax.scatter(
        nested.outer_choices["outer_fold"],
        nested.outer_choices["selected_calibration_method"].map(method_codes),
        s=90,
    )
    ax.set_yticks(list(method_codes.values()), list(method_codes.keys()))
    ax.set_xticks(nested.outer_choices["outer_fold"])
    ax.set_xlabel("Outer fold")
    ax.set_title("Calibration selections — DEVELOPMENT ONLY")
    path = output_dir / "outer_fold_calibration_selections.png"
    _save_figure(fig, path)
    paths.append(path)
    return paths


def _write_holdout_protocol(path: Path, policy: dict[str, Any]) -> None:
    method = policy["calibration"]["method"]
    threshold = policy["decision_threshold"]
    lines = [
        "# Frozen one-time final-holdout protocol",
        "",
        "> This document defines Stage 7 but does not execute it. No holdout values were inspected in Stage 6.",
        "",
        "1. Verify the raw snapshot, split manifest, Stage 4 fold-membership, and frozen-policy SHA-256 values.",
        "2. Load exactly the 17 frozen predictive features; keep the three audit-only attributes outside prediction.",
        "3. Construct the unchanged Stage 3R tree preprocessing and fixed untuned Stage 4 XGBoost.",
        f"4. On the 800 development observations, fit `CalibratedClassifierCV(estimator=<fixed Stage 4 XGBoost pipeline>, method=\"{method}\", cv=<exact persisted Stage 4 five-fold splits>, ensemble=True)`. `n_jobs` is not passed and therefore resolves to `None` under scikit-learn 1.4.2. This creates five fold-specific classifier/calibrator pairs; it does not train one XGBoost on all 800 rows.",
        "5. Generate probabilities for the sealed 200-row holdout once. The final predictor averages the five calibrated probabilities produced by those fold-specific pairs.",
        f"6. Apply the already frozen threshold `{threshold:.2f}` without modification.",
        "7. Report only the metrics predefined below. Do not alter the model, features, preprocessing, calibration, or threshold after seeing results.",
        "",
        "## Predefined metrics",
        "",
        *[f"- `{metric}`" for metric in STAGE7_METRICS],
        "",
        "## Confidence intervals",
        "",
        "No confidence-interval procedure is authorized or predefined. Stage 7 must not add one without explicit authorization before holdout access.",
        "",
        "The holdout is a one-time evaluation, not a new model-selection dataset.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(
    path: Path,
    aggregate: dict[str, Any],
    stage4: dict[str, Any],
    choices: pd.DataFrame,
    threshold_stability: pd.DataFrame,
    full_method: str,
    final_threshold: float,
    key_thresholds: pd.DataFrame,
) -> None:
    comparison_columns = [
        "roc_auc",
        "average_precision",
        "precision_bad_credit_risk",
        "recall_bad_credit_risk",
        "f1_bad_credit_risk",
        "balanced_accuracy",
        "brier_score",
        "accuracy",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "total_cost",
        "average_cost",
    ]
    comparison = pd.DataFrame(
        [
            {"policy": "Stage 4 XGBoost, threshold 0.50", **stage4},
            {"policy": aggregate["policy"], **aggregate},
        ]
    )[["policy", *comparison_columns]]
    stability = threshold_stability.iloc[0]
    lines = [
        "# Stage 6 calibration and threshold policy",
        "",
        "Stage 5 tuning remains preserved negative evidence. Stage 6 uses only the fixed untuned Stage 4 XGBoost and the locked 800-row development population.",
        "",
        "## Honest nested policy estimate",
        "",
        comparison.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Outer-fold policy choices",
        "",
        choices.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Selected thresholds ranged from {stability['minimum']:.2f} to {stability['maximum']:.2f} (mean {stability['mean']:.3f}, sample SD {stability['sample_sd']:.3f}). The theoretical 1/6 threshold is a reference, not a forced choice.",
        "",
        "## Frozen development-wide policy",
        "",
        f"- Calibration: `{full_method}`",
        f"- Threshold: `{final_threshold:.2f}`",
        "- Model: fixed untuned Stage 4 XGBoost",
        "- Cost: `5 × FN + FP`",
        "",
        "## Key development thresholds",
        "",
        key_thresholds.to_markdown(index=False, floatfmt=".6f"),
        "",
        "No holdout prediction, metric, calibration fit, threshold selection, SHAP analysis, feature selection, or additional hyperparameter search occurred.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_stage6_artifacts(project_root: Path) -> dict[str, Any]:
    """Execute Stage 6 using development rows only and freeze the Stage 7 policy."""
    split_path = project_root / "reports" / "split_manifest.csv"
    raw_path = project_root / "data" / "raw" / "german_credit.csv"
    stage4_oof_path = project_root / "reports" / "stage4" / "development_oof_predictions.csv"
    stage5_oof_path = project_root / "reports" / "stage5" / "nested_oof_predictions.csv"
    hashes_before = {
        "raw": _sha256(raw_path),
        "split": _sha256(split_path),
        "stage4_fold": _sha256(project_root / "reports" / "stage4" / "fold_membership.csv"),
        "stage4_oof": _sha256(stage4_oof_path),
        "stage5_oof": _sha256(stage5_oof_path),
    }
    expected = {
        "raw": RAW_SNAPSHOT_SHA256,
        "split": LOCKED_SPLIT_SHA256,
        "stage4_fold": STAGE4_FOLD_SHA256,
        "stage4_oof": STAGE4_OOF_SHA256,
        "stage5_oof": STAGE5_OOF_SHA256,
    }
    if hashes_before != expected:
        raise ValueError(f"Locked evidence changed: {hashes_before}")

    data = build_analysis_dataset(load_verified_raw_snapshot(project_root))
    data.index = source_row_ids(data)
    manifest = pd.read_csv(split_path).set_index(ROW_ID_NAME)
    development_ids = manifest.index[manifest["split"] == DEVELOPMENT_SPLIT]
    X = data.loc[development_ids, list(PREDICTIVE_FEATURES)].copy()
    y = data.loc[development_ids, TARGET_BINARY_NAME].astype("int8").copy()
    if len(X) != 800 or y.value_counts().to_dict() != {0: 560, 1: 240}:
        raise ValueError("Locked development population changed.")
    if set(X.columns) & set(AUDIT_ATTRIBUTES) or len(X.columns) != 17:
        raise ValueError("Predictive feature governance changed.")

    _fold_membership, outer_folds = load_outer_folds(project_root, X.index)
    stage4_oof = pd.read_csv(stage4_oof_path).set_index(ROW_ID_NAME)
    if not stage4_oof.index.equals(X.index):
        raise ValueError("Stage 4 OOF evidence no longer aligns with development rows.")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nested = run_nested_policy_evaluation(X, y, outer_folds)
        # Full-development selection happens only after honest outer evaluation.
        full_probabilities = full_development_calibration_oof(
            X,
            y,
            outer_folds,
            stage4_oof["xgboost_probability"],
        )

    aggregate = aggregate_nested_policy(nested.oof_predictions)
    stage4_comparison = pd.read_csv(
        project_root / "reports" / "stage4" / "model_comparison.csv"
    ).set_index("model")
    stage4_reference = stage4_comparison.loc["XGBoost"].to_dict()
    full_comparison = calibration_comparison_table(
        y, full_probabilities, outer_fold="full_development"
    ).drop(columns="outer_fold")
    full_method = select_calibration_method(full_comparison)
    threshold_search = threshold_search_table(y, full_probabilities[full_method])
    final_threshold = select_threshold(threshold_search)
    threshold_search["selected_threshold"] = threshold_search["threshold"].eq(
        final_threshold
    )

    key_specs = [
        ("Default 0.50", 0.50),
        ("Theoretical 1/6 reference", THEORETICAL_THRESHOLD),
        ("Empirically selected", final_threshold),
    ]
    key_thresholds = pd.DataFrame(
        [
            {
                "threshold_label": label,
                **decision_metrics(y, full_probabilities[full_method], threshold),
            }
            for label, threshold in key_specs
        ]
    )
    calibration_stability = calibration_stability_table(nested.outer_choices)
    threshold_stability = threshold_stability_table(nested.outer_choices)
    reliability = reliability_table(y, full_probabilities)

    output_dir = project_root / "reports" / "stage6"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    nested_oof_path = output_dir / "nested_policy_oof_predictions.csv"
    threshold_path = output_dir / "threshold_search.csv"
    nested.oof_predictions.to_csv(
        nested_oof_path, index=False, lineterminator="\n"
    )
    nested.outer_metrics.to_csv(output_dir / "outer_policy_metrics.csv", index=False)
    nested.outer_choices.to_csv(
        output_dir / "outer_fold_policy_choices.csv", index=False
    )
    nested.inner_calibration_comparison.to_csv(
        output_dir / "calibration_comparison.csv", index=False
    )
    nested.inner_threshold_search.to_csv(
        output_dir / "outer_inner_threshold_search.csv", index=False
    )
    calibration_stability.to_csv(
        output_dir / "calibration_stability.csv", index=False
    )
    threshold_stability.to_csv(output_dir / "threshold_stability.csv", index=False)
    full_comparison.to_csv(
        output_dir / "full_development_calibration_comparison.csv", index=False
    )
    reliability.to_csv(output_dir / "reliability_comparison.csv", index=False)
    threshold_search.to_csv(threshold_path, index=False, lineterminator="\n")
    key_thresholds.to_csv(output_dir / "key_threshold_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"policy": "Stage 4 XGBoost, threshold 0.50", **stage4_reference},
            aggregate,
        ]
    ).to_csv(output_dir / "policy_comparison.csv", index=False)

    frozen_policy: dict[str, Any] = {
        "policy_status": "frozen_before_final_holdout_access",
        "stage": 6,
        "model": {
            "class": "xgboost.XGBClassifier",
            "role": "fixed untuned Stage 4 finalist",
            "hyperparameters": XGBOOST_PARAMETERS,
        },
        "feature_governance": {
            "predictive_feature_count": 17,
            "predictive_features": list(PREDICTIVE_FEATURES),
            "audit_only_excluded": list(AUDIT_ATTRIBUTES),
        },
        "preprocessing": {
            "numeric_passthrough_after_median_imputation": list(NUMERICAL_FEATURES),
            "one_hot_after_most_frequent_imputation": list(CATEGORICAL_FEATURES),
            "one_hot_handle_unknown": "ignore",
        },
        "calibration": {
            "method": full_method,
            "selection_metric": "brier_score",
            "parsimony_tolerance": CALIBRATION_PARSIMONY_TOLERANCE,
            "parsimony_order": list(CALIBRATION_METHODS),
            "implementation_class": "sklearn.calibration.CalibratedClassifierCV",
            "estimator": "fixed Stage 4 XGBoost preprocessing/model pipeline",
            "cv": "exact persisted Stage 4 five-fold development splits",
            "ensemble": True,
            "n_jobs": None,
            "n_jobs_explicitly_set": False,
            "prediction_aggregation": (
                "arithmetic mean of calibrated probabilities from five "
                "fold-specific classifier/calibrator pairs"
            ),
            "stage7_fit_protocol": (
                "Fit CalibratedClassifierCV(estimator=fixed Stage 4 XGBoost "
                "pipeline, method='sigmoid', cv=exact persisted Stage 4 five-fold "
                "development splits, ensemble=True) on the 800 development "
                "observations; n_jobs is omitted and resolves to None."
                if full_method != "none"
                else "fit fixed preprocessing and XGBoost on all development rows"
            ),
        },
        "decision_threshold": final_threshold,
        "threshold_grid": list(THRESHOLD_GRID),
        "theoretical_cost_reference": THEORETICAL_THRESHOLD,
        "cost_function": {
            "false_negative_cost": 5,
            "false_positive_cost": 1,
            "formula": "5 * false_negative + false_positive",
        },
        "locked_evidence": {
            "raw_snapshot_sha256": RAW_SNAPSHOT_SHA256,
            "split_manifest_sha256": LOCKED_SPLIT_SHA256,
            "stage4_fold_membership_sha256": STAGE4_FOLD_SHA256,
            "development_rows": 800,
            "final_holdout_rows_sealed": 200,
        },
        "stage7_predefined_metrics": list(STAGE7_METRICS),
        "confidence_intervals": {
            "authorized": False,
            "method": None,
            "reason": "No confidence-interval procedure was predefined in Stage 6.",
        },
        "change_control": (
            "Any model-development change after this freeze invalidates the planned "
            "one-time holdout evaluation."
        ),
    }
    policy_path = output_dir / "frozen_model_policy.json"
    policy_path.write_text(
        json.dumps(frozen_policy, indent=2) + "\n", encoding="utf-8"
    )
    _write_holdout_protocol(
        output_dir / "final_holdout_protocol.md", frozen_policy
    )

    figure_paths = create_figures(
        nested,
        full_comparison,
        reliability,
        threshold_search,
        key_thresholds,
        figures_dir,
    )
    _write_summary(
        output_dir / "stage6_decision_policy_summary.md",
        aggregate,
        stage4_reference,
        nested.outer_choices,
        threshold_stability,
        full_method,
        final_threshold,
        key_thresholds,
    )

    hashes_after = {
        "raw": _sha256(raw_path),
        "split": _sha256(split_path),
        "stage4_fold": _sha256(project_root / "reports" / "stage4" / "fold_membership.csv"),
        "stage4_oof": _sha256(stage4_oof_path),
        "stage5_oof": _sha256(stage5_oof_path),
    }
    if hashes_after != hashes_before:
        raise RuntimeError("Stage 6 changed locked historical evidence.")

    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": version("xgboost"),
        "joblib": version("joblib"),
        "threadpoolctl": version("threadpoolctl"),
    }
    run_summary: dict[str, Any] = {
        "stage": "Stage 6: development-only calibration and threshold governance",
        "environment": environment,
        "development_rows": 800,
        "final_holdout_rows_sealed": 200,
        "model_parameters": XGBOOST_PARAMETERS,
        "calibration_candidates": list(CALIBRATION_METHODS),
        "calibration_selection": {
            "metric": "brier_score",
            "parsimony_tolerance": CALIBRATION_PARSIMONY_TOLERANCE,
            "parsimony_order": list(CALIBRATION_METHODS),
        },
        "threshold_grid": list(THRESHOLD_GRID),
        "threshold_tie_break": [
            "lower total cost",
            "fewer false negatives",
            "higher balanced accuracy",
            "closest to 1/6 theoretical reference",
        ],
        "honest_nested_policy_metrics": aggregate,
        "full_development_selected_calibration": full_method,
        "full_development_selected_threshold": final_threshold,
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "nested_policy_oof_sha256": _sha256(nested_oof_path),
        "frozen_model_policy_sha256": _sha256(policy_path),
        "threshold_search_sha256": _sha256(threshold_path),
        "warnings": sorted({str(item.message) for item in caught}),
        "final_holdout_predictions_generated": False,
        "final_holdout_metrics_generated": False,
        "final_holdout_inspected": False,
        "additional_hyperparameter_tuning_performed": False,
        "feature_selection_performed": False,
        "shap_performed": False,
        "stage7_metrics_generated": False,
        "figures": [str(path.relative_to(project_root)) for path in figure_paths],
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(generate_stage6_artifacts(args.project_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
