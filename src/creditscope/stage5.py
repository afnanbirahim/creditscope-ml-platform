"""Run controlled nested-CV XGBoost tuning on development data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from creditscope.analysis import build_analysis_dataset
from creditscope.data import load_verified_raw_snapshot
from creditscope.evaluation import (
    DEFAULT_THRESHOLD,
    asymmetric_cost,
    classification_metrics,
    threshold_predictions,
)
from creditscope.modeling import (
    AUDIT_ATTRIBUTES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PREDICTIVE_FEATURES,
    build_tree_preprocessor,
    partition_model_data,
)
from creditscope.splits import write_locked_split
from creditscope.stage3 import LOCKED_MANIFEST_SHA256

plt.switch_backend("Agg")

RANDOM_STATE = 42
INNER_SPLITS = 4
SEARCH_ITERATIONS = 40
PRIMARY_SCORER = "average_precision"
STAGE4_FOLD_SHA256 = (
    "513fc9249a4c00be3ba67fa99ea20a7ac76b653694130cf66733ea08f97d0b92"
)
STAGE4_OOF_SHA256 = (
    "c1e96b552ca7f5510f8aa9eda88db128e9cfbdd8ef30ca8535fac69646c486f9"
)

SEARCH_SPACE: dict[str, list[float | int]] = {
    "classifier__n_estimators": [150, 250, 350, 500, 700],
    "classifier__max_depth": [2, 3, 4, 5],
    "classifier__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.10],
    "classifier__min_child_weight": [1, 3, 5, 8],
    "classifier__subsample": [0.70, 0.80, 0.90, 1.00],
    "classifier__colsample_bytree": [0.70, 0.80, 0.90, 1.00],
    "classifier__gamma": [0.0, 0.25, 0.50, 1.0],
    "classifier__reg_alpha": [0.0, 0.01, 0.10, 0.50],
    "classifier__reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
}

SCORING = {
    PRIMARY_SCORER: "average_precision",
    "roc_auc": "roc_auc",
    "neg_brier_score": "neg_brier_score",
    "recall_bad_credit_risk": make_scorer(recall_score, pos_label=1),
}


@dataclass(frozen=True)
class NestedTuningResult:
    """Development-only evidence from the complete nested procedure."""

    probabilities: pd.Series
    predictions: pd.Series
    outer_fold_metrics: pd.DataFrame
    outer_best_params: pd.DataFrame
    search_results: pd.DataFrame
    train_validation_gaps: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_classifier_prefix(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        name.removeprefix("classifier__"): value
        for name, value in parameters.items()
    }


def build_tuning_pipeline() -> Pipeline:
    """Build the fixed Stage 5 XGBoost family with fold-local preprocessing."""
    from xgboost import XGBClassifier

    return Pipeline(
        [
            ("preprocessor", build_tree_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )


def make_inner_cv() -> StratifiedKFold:
    """Return the one fixed inner search splitter."""
    return StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def make_randomized_search(
    *, cv: Any, refit: str | bool = PRIMARY_SCORER, n_jobs: int = -1
) -> RandomizedSearchCV:
    """Construct the frozen 40-draw Average-Precision search."""
    return RandomizedSearchCV(
        estimator=build_tuning_pipeline(),
        param_distributions=SEARCH_SPACE,
        n_iter=SEARCH_ITERATIONS,
        scoring=SCORING,
        refit=refit,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
    )


def load_outer_folds(
    project_root: Path, development_index: pd.Index
) -> tuple[pd.DataFrame, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    """Load, hash, and align the persisted Stage 4 outer-fold membership."""
    path = project_root / "reports" / "stage4" / "fold_membership.csv"
    if _sha256(path) != STAGE4_FOLD_SHA256:
        raise ValueError("Stage 4 fold-membership SHA-256 does not match the lock.")
    membership = pd.read_csv(path)
    if list(membership.columns) != ["source_row_id", "fold"]:
        raise ValueError("Stage 4 fold-membership schema changed.")
    if membership["source_row_id"].duplicated().any() or len(membership) != 800:
        raise ValueError("Outer-fold membership must contain 800 unique rows.")
    aligned = membership.set_index("source_row_id").loc[development_index]
    if not aligned.index.equals(development_index):
        raise ValueError("Outer folds do not align with the locked development order.")
    if set(aligned["fold"]) != {1, 2, 3, 4, 5}:
        raise ValueError("Outer-fold identifiers must be 1 through 5.")
    folds = tuple(
        (
            np.flatnonzero(aligned["fold"].to_numpy() != fold),
            np.flatnonzero(aligned["fold"].to_numpy() == fold),
        )
        for fold in range(1, 6)
    )
    if any(len(train) != 640 or len(validation) != 160 for train, validation in folds):
        raise ValueError("Each outer fold must contain 640 train and 160 validation rows.")
    return membership, folds


def validate_inner_outer_isolation(
    outer_train_positions: np.ndarray,
    outer_validation_positions: np.ndarray,
    y_development: pd.Series,
) -> None:
    """Prove inner indices refer only to the current outer-training subset."""
    if set(outer_train_positions) & set(outer_validation_positions):
        raise ValueError("Outer training and validation positions overlap.")
    inner_seen: set[int] = set()
    y_outer_train = y_development.iloc[outer_train_positions]
    placeholder = np.zeros((len(outer_train_positions), 1))
    for inner_train, inner_validation in make_inner_cv().split(
        placeholder, y_outer_train
    ):
        mapped_train = set(outer_train_positions[inner_train])
        mapped_validation = set(outer_train_positions[inner_validation])
        if mapped_train & set(outer_validation_positions):
            raise RuntimeError("Outer validation entered an inner training fold.")
        if mapped_validation & set(outer_validation_positions):
            raise RuntimeError("Outer validation entered an inner validation fold.")
        inner_seen.update(mapped_validation)
    if inner_seen != set(outer_train_positions):
        raise RuntimeError("Inner validation folds do not cover outer training exactly.")


def _compact_search_results(
    search: RandomizedSearchCV, outer_fold: int | str
) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)
    rows = pd.DataFrame(
        {
            "outer_fold": outer_fold,
            "candidate": np.arange(1, len(results) + 1),
            "rank_inner_average_precision": results[
                "rank_test_average_precision"
            ].astype(int),
            "mean_inner_average_precision": results[
                "mean_test_average_precision"
            ],
            "std_inner_average_precision": results[
                "std_test_average_precision"
            ],
            "mean_inner_roc_auc": results["mean_test_roc_auc"],
            "std_inner_roc_auc": results["std_test_roc_auc"],
            "mean_inner_brier_score": -results["mean_test_neg_brier_score"],
            "std_inner_brier_score": results["std_test_neg_brier_score"],
            "mean_inner_recall_at_0_50": results[
                "mean_test_recall_bad_credit_risk"
            ],
            "std_inner_recall_at_0_50": results[
                "std_test_recall_bad_credit_risk"
            ],
        }
    )
    for parameter in SEARCH_SPACE:
        clean_name = parameter.removeprefix("classifier__")
        rows[clean_name] = [params[parameter] for params in results["params"]]
    return rows.sort_values(
        ["outer_fold", "rank_inner_average_precision"], ignore_index=True
    )


def run_nested_tuning(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    outer_folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> NestedTuningResult:
    """Run five unbiased outer evaluations with search confined to outer training."""
    if list(X_development.columns) != list(PREDICTIVE_FEATURES):
        raise ValueError("Stage 5 received an unexpected feature boundary.")
    probabilities = pd.Series(index=y_development.index, dtype="float64")
    assignment_count = pd.Series(0, index=y_development.index, dtype="int8")
    fold_metrics: list[dict[str, Any]] = []
    best_params_rows: list[dict[str, Any]] = []
    search_tables: list[pd.DataFrame] = []
    gap_rows: list[dict[str, Any]] = []

    for outer_fold, (train_positions, validation_positions) in enumerate(
        outer_folds, start=1
    ):
        validate_inner_outer_isolation(
            train_positions, validation_positions, y_development
        )
        X_train = X_development.iloc[train_positions]
        y_train = y_development.iloc[train_positions]
        X_validation = X_development.iloc[validation_positions]
        y_validation = y_development.iloc[validation_positions]
        search = make_randomized_search(cv=make_inner_cv())
        search.fit(X_train, y_train)
        search_tables.append(_compact_search_results(search, outer_fold))

        best = search.best_estimator_
        validation_probability = pd.Series(
            best.predict_proba(X_validation)[:, 1],
            index=X_validation.index,
            dtype="float64",
        )
        validation_prediction = threshold_predictions(validation_probability)
        probabilities.loc[X_validation.index] = validation_probability
        assignment_count.loc[X_validation.index] += 1
        metrics = classification_metrics(
            y_validation, validation_probability, validation_prediction
        )
        cost = asymmetric_cost(y_validation, validation_prediction)
        fold_metrics.append(
            {
                "outer_fold": outer_fold,
                "training_rows": len(X_train),
                "validation_rows": len(X_validation),
                **metrics,
                **cost,
            }
        )
        clean_params = _strip_classifier_prefix(search.best_params_)
        best_params_rows.append(
            {
                "outer_fold": outer_fold,
                "best_inner_average_precision": float(search.best_score_),
                **clean_params,
            }
        )

        training_probability = pd.Series(
            best.predict_proba(X_train)[:, 1], index=X_train.index, dtype="float64"
        )
        training_prediction = threshold_predictions(training_probability)
        training_metrics = classification_metrics(
            y_train, training_probability, training_prediction
        )
        for metric in ("roc_auc", "average_precision", "recall_bad_credit_risk"):
            gap_rows.append(
                {
                    "model": "Nested-tuned XGBoost",
                    "outer_fold": outer_fold,
                    "metric": metric,
                    "training_value": training_metrics[metric],
                    "validation_value": metrics[metric],
                    "training_minus_validation": (
                        training_metrics[metric] - metrics[metric]
                    ),
                }
            )

    if probabilities.isna().any() or not (assignment_count == 1).all():
        raise RuntimeError("Every development row must receive one nested OOF score.")
    return NestedTuningResult(
        probabilities=probabilities,
        predictions=threshold_predictions(probabilities),
        outer_fold_metrics=pd.DataFrame(fold_metrics),
        outer_best_params=pd.DataFrame(best_params_rows),
        search_results=pd.concat(search_tables, ignore_index=True),
        train_validation_gaps=pd.DataFrame(gap_rows),
    )


def aggregate_summary(
    model: str,
    y_true: pd.Series,
    probabilities: pd.Series,
    predictions: pd.Series,
) -> dict[str, Any]:
    return {
        "model": model,
        "evaluation_population": "800-row development nested OOF predictions",
        "threshold": DEFAULT_THRESHOLD,
        **classification_metrics(y_true, probabilities, predictions),
        **asymmetric_cost(y_true, predictions),
    }


def tuned_vs_untuned_table(comparison: pd.DataFrame) -> pd.DataFrame:
    untuned = comparison.set_index("model").loc["Untuned XGBoost (Stage 4)"]
    tuned = comparison.set_index("model").loc["Nested-tuned XGBoost (Stage 5)"]
    metrics = (
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
    )
    return pd.DataFrame(
        {
            "metric": metrics,
            "untuned_stage4_xgboost": [float(untuned[item]) for item in metrics],
            "nested_tuned_stage5_xgboost": [float(tuned[item]) for item in metrics],
            "absolute_difference": [
                abs(float(tuned[item]) - float(untuned[item])) for item in metrics
            ],
        }
    )


def prediction_change_summary(oof: pd.DataFrame) -> pd.DataFrame:
    untuned_correct = oof["untuned_xgboost_prediction"] == oof["bad_credit_risk"]
    tuned_correct = oof["tuned_xgboost_prediction"] == oof["bad_credit_risk"]
    probability_change = (
        oof["tuned_xgboost_probability"] - oof["untuned_xgboost_probability"]
    )
    absolute_change = probability_change.abs()
    values: list[tuple[str, float | int]] = [
        ("classification_agreement_count", int((oof["untuned_xgboost_prediction"] == oof["tuned_xgboost_prediction"]).sum())),
        ("classification_agreement_proportion", float((oof["untuned_xgboost_prediction"] == oof["tuned_xgboost_prediction"]).mean())),
        ("classifications_changed", int((oof["untuned_xgboost_prediction"] != oof["tuned_xgboost_prediction"]).sum())),
        ("changed_wrong_to_correct", int((~untuned_correct & tuned_correct).sum())),
        ("changed_correct_to_wrong", int((untuned_correct & ~tuned_correct).sum())),
        ("probability_change_mean", float(probability_change.mean())),
        ("probability_change_std", float(probability_change.std(ddof=1))),
        ("absolute_probability_change_mean", float(absolute_change.mean())),
        ("absolute_probability_change_median", float(absolute_change.median())),
        ("absolute_probability_change_p90", float(absolute_change.quantile(0.90))),
        ("absolute_probability_change_max", float(absolute_change.max())),
        ("absolute_probability_change_ge_0_10", int((absolute_change >= 0.10).sum())),
        ("absolute_probability_change_ge_0_20", int((absolute_change >= 0.20).sum())),
    ]
    return pd.DataFrame(values, columns=["measure", "value"])


def hyperparameter_stability(best_params: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    parameter_columns = [
        name
        for name in best_params.columns
        if name not in {"outer_fold", "best_inner_average_precision"}
    ]
    for parameter in parameter_columns:
        for value, count in best_params[parameter].value_counts(dropna=False).items():
            rows.append(
                {
                    "hyperparameter": parameter,
                    "selected_value": value,
                    "selection_count": int(count),
                    "selection_proportion": float(count / len(best_params)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["hyperparameter", "selection_count", "selected_value"],
        ascending=[True, False, True],
        ignore_index=True,
    )


def fold_metric_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "roc_auc",
        "average_precision",
        "precision_bad_credit_risk",
        "recall_bad_credit_risk",
        "f1_bad_credit_risk",
        "balanced_accuracy",
        "brier_score",
        "accuracy",
        "total_cost",
        "average_cost",
    )
    return pd.DataFrame(
        {
            "metric": columns,
            "outer_fold_mean": [float(fold_metrics[c].mean()) for c in columns],
            "outer_fold_sample_sd": [
                float(fold_metrics[c].std(ddof=1)) for c in columns
            ],
        }
    )


def calibration_points(
    y_true: pd.Series, comparison_oof: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, probability_column in (
        ("Untuned XGBoost (Stage 4)", "untuned_xgboost_probability"),
        ("Nested-tuned XGBoost (Stage 5)", "tuned_xgboost_probability"),
    ):
        observed, predicted = calibration_curve(
            y_true,
            comparison_oof[probability_column],
            n_bins=10,
            strategy="quantile",
        )
        for bin_number, (mean_probability, observed_rate) in enumerate(
            zip(predicted, observed, strict=True), start=1
        ):
            rows.append(
                {
                    "model": model,
                    "bin": bin_number,
                    "mean_predicted_bad_credit_risk_probability": mean_probability,
                    "observed_bad_credit_risk_rate": observed_rate,
                }
            )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figures(
    y_true: pd.Series,
    oof: pd.DataFrame,
    calibration: pd.DataFrame,
    gaps: pd.DataFrame,
    stability: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    created: list[Path] = []
    models = (
        ("Untuned XGBoost", "untuned_xgboost_probability", "#4B8B5A"),
        ("Nested-tuned XGBoost", "tuned_xgboost_probability", "#8B4A7D"),
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for label, column, color in models:
        RocCurveDisplay.from_predictions(y_true, oof[column], name=label, ax=ax, color=color)
    ax.set_title("XGBoost ROC: development OOF predictions")
    path = figures_dir / "tuned_vs_untuned_roc.png"
    _save_figure(fig, path)
    created.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for label, column, color in models:
        PrecisionRecallDisplay.from_predictions(
            y_true, oof[column], name=label, ax=ax, color=color
        )
    ax.set_title("XGBoost precision-recall: development OOF predictions")
    path = figures_dir / "tuned_vs_untuned_precision_recall.png"
    _save_figure(fig, path)
    created.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (label, prefix) in zip(
        axes,
        (("Untuned XGBoost", "untuned"), ("Nested-tuned XGBoost", "tuned")),
        strict=True,
    ):
        matrix = confusion_matrix(y_true, oof[f"{prefix}_xgboost_prediction"], labels=[0, 1])
        ConfusionMatrixDisplay(matrix, display_labels=["Good risk (0)", "Bad risk (1)"]).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format="d"
        )
        ax.set_title(label)
    fig.suptitle("Development OOF confusion matrices at threshold 0.50")
    fig.tight_layout()
    path = figures_dir / "confusion_matrices.png"
    _save_figure(fig, path)
    created.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.plot([0, 1], [0, 1], "--", color="#555555", label="Perfect calibration")
    for model, group in calibration.groupby("model", sort=False):
        ax.plot(
            group["mean_predicted_bad_credit_risk_probability"],
            group["observed_bad_credit_risk_rate"],
            marker="o",
            label=model,
        )
    ax.set(
        title="Calibration diagnostic: development nested OOF",
        xlabel="Mean predicted bad-credit-risk probability",
        ylabel="Observed bad-credit-risk rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(frameon=False)
    path = figures_dir / "calibration_comparison.png"
    _save_figure(fig, path)
    created.append(path)

    gap_mean = gaps.groupby(["model", "metric"], sort=False)[
        "training_minus_validation"
    ].mean().unstack("metric")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    gap_mean[["roc_auc", "average_precision", "recall_bad_credit_risk"]].plot.bar(ax=ax)
    ax.set_title("Mean train-minus-validation gaps")
    ax.set_ylabel("Metric gap")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=10)
    ax.legend(title="Metric", frameon=False)
    fig.tight_layout()
    path = figures_dir / "train_validation_gap.png"
    _save_figure(fig, path)
    created.append(path)

    probability_difference = (
        oof["tuned_xgboost_probability"] - oof["untuned_xgboost_probability"]
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.hist(probability_difference, bins=30, color="#8B4A7D", edgecolor="white")
    ax.axvline(0, color="#333333", linestyle="--")
    ax.set(
        title="Tuned minus untuned OOF probability differences",
        xlabel="Probability difference",
        ylabel="Development observations",
    )
    path = figures_dir / "probability_change_distribution.png"
    _save_figure(fig, path)
    created.append(path)

    max_counts = stability.groupby("hyperparameter")["selection_count"].max().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(max_counts.index, max_counts.values, color="#6C7A89")
    ax.set(
        title="Most frequent selected value across five outer folds",
        xlabel="Selection count",
        ylabel="Hyperparameter",
        xlim=(0, 5.2),
    )
    path = figures_dir / "hyperparameter_stability.png"
    _save_figure(fig, path)
    created.append(path)
    return created


def _write_summary(
    path: Path,
    comparison: pd.DataFrame,
    fold_summary: pd.DataFrame,
    best_params: pd.DataFrame,
    changes: pd.DataFrame,
    gaps: pd.DataFrame,
    final_params: dict[str, Any],
) -> None:
    indexed = comparison.set_index("model")
    untuned = indexed.loc["Untuned XGBoost (Stage 4)"]
    tuned = indexed.loc["Nested-tuned XGBoost (Stage 5)"]
    change = changes.set_index("measure")["value"]
    gap_mean = gaps.groupby(["model", "metric"])["training_minus_validation"].mean()
    unique_configurations = len(
        best_params.drop(columns=["outer_fold", "best_inner_average_precision"])
        .astype(str)
        .drop_duplicates()
    )
    lines = [
        "# Stage 5 controlled XGBoost tuning",
        "",
        "Nested cross-validation was performed only on the locked 800-row development partition. The final 200-row holdout was not scored, predicted, or used in preprocessing or search.",
        "",
        "## Design",
        "",
        "- The persisted Stage 4 folds are the five outer evaluation folds.",
        "- Each outer-training partition used a four-fold stratified inner search.",
        "- Every inner search used the frozen 40-draw search space and selected by Average Precision.",
        "- XGBoost used one thread per fit while the search controlled parallelism.",
        "- Threshold remained 0.50; 5:1 cost was monitored but not optimized.",
        "- No weighting, resampling, calibration, SHAP, feature selection, or final-holdout evaluation occurred.",
        "",
        "## Honest nested OOF comparison",
        "",
        comparison[["model", "roc_auc", "average_precision", "precision_bad_credit_risk", "recall_bad_credit_risk", "f1_bad_credit_risk", "balanced_accuracy", "brier_score", "accuracy", "false_positive", "false_negative", "total_cost"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Nested tuning changed {int(change['classifications_changed'])} classifications; {int(change['changed_wrong_to_correct'])} changed from wrong to correct and {int(change['changed_correct_to_wrong'])} changed from correct to wrong.",
        "",
        "## Outer-fold variability",
        "",
        fold_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Hyperparameter stability",
        "",
        f"The five outer folds selected {unique_configurations} distinct configurations. This is evidence about tuning stability, not a reason to force one retrospective configuration.",
        "",
        best_params.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Train-versus-validation diagnostic",
        "",
        f"Nested-tuned mean ROC-AUC gap: {gap_mean.loc[('Nested-tuned XGBoost', 'roc_auc')]:.4f}; Stage 4 untuned mean ROC-AUC gap: {gap_mean.loc[('Untuned XGBoost (Stage 4)', 'roc_auc')]:.4f}.",
        "",
        "## Final development-wide configuration search",
        "",
        "The following configuration was selected by a separate 40-draw search on all 800 development rows:",
        "",
        "```json",
        json.dumps(final_params, indent=2),
        "```",
        "",
        "Its internal CV score is selection evidence, not an unbiased performance estimate. The nested OOF result above is the honest development estimate of the tuning procedure.",
        "",
        "## Interpretation",
        "",
        f"Average Precision changed from {float(untuned['average_precision']):.4f} to {float(tuned['average_precision']):.4f}; total 5:1 cost at the unchanged threshold changed from {int(untuned['total_cost'])} to {int(tuned['total_cost'])}. Small changes must be interpreted alongside fold variability and configuration instability.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_stage5_artifacts(project_root: Path) -> dict[str, Any]:
    """Execute controlled Stage 5 tuning without touching final-test performance."""
    split_path = project_root / "reports" / "split_manifest.csv"
    raw_path = project_root / "data" / "raw" / "german_credit.csv"
    split_hash_before = _sha256(split_path)
    raw_hash_before = _sha256(raw_path)
    if split_hash_before != LOCKED_MANIFEST_SHA256:
        raise ValueError("Locked split SHA-256 changed before Stage 5.")

    data = build_analysis_dataset(load_verified_raw_snapshot(project_root))
    manifest, split_summary = write_locked_split(data, project_root / "reports")
    if split_summary["manifest_sha256"] != LOCKED_MANIFEST_SHA256:
        raise ValueError("Canonical split content changed before Stage 5.")
    partitions = partition_model_data(data, manifest)
    if len(partitions.X_development) != 800:
        raise ValueError("Stage 5 requires exactly 800 development observations.")
    fold_membership, outer_folds = load_outer_folds(
        project_root, partitions.X_development.index
    )

    stage4_oof_path = project_root / "reports" / "stage4" / "development_oof_predictions.csv"
    if _sha256(stage4_oof_path) != STAGE4_OOF_SHA256:
        raise ValueError("Stage 4 OOF artifact SHA-256 changed.")
    stage4_oof = pd.read_csv(stage4_oof_path).set_index("source_row_id")
    if not stage4_oof.index.equals(partitions.X_development.index):
        raise ValueError("Stage 4 OOF evidence does not align with development rows.")
    if not stage4_oof["bad_credit_risk"].astype("int8").equals(
        partitions.y_development
    ):
        raise ValueError("Stage 4 OOF target does not match development target.")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        nested = run_nested_tuning(
            partitions.X_development, partitions.y_development, outer_folds
        )

        # Configuration selection only: no estimator is refit on all 800 rows.
        final_search = make_randomized_search(cv=outer_folds, refit=False)
        final_search.fit(partitions.X_development, partitions.y_development)

    final_results = pd.DataFrame(final_search.cv_results_)
    final_best_index = int(final_results["rank_test_average_precision"].argmin())
    final_best_params = _strip_classifier_prefix(
        final_results.loc[final_best_index, "params"]
    )
    final_summary = _compact_search_results(final_search, "full_development")

    tuned_row = aggregate_summary(
        "Nested-tuned XGBoost (Stage 5)",
        partitions.y_development,
        nested.probabilities,
        nested.predictions,
    )
    stage4_comparison = pd.read_csv(
        project_root / "reports" / "stage4" / "model_comparison.csv"
    ).set_index("model")
    historical = stage4_comparison.loc["XGBoost"].to_dict()
    historical["model"] = "Untuned XGBoost (Stage 4)"
    comparison = pd.DataFrame([historical, tuned_row])
    tuned_vs_untuned = tuned_vs_untuned_table(comparison)

    nested_oof = pd.DataFrame(
        {
            "source_row_id": partitions.X_development.index,
            "outer_fold": fold_membership.set_index("source_row_id").loc[
                partitions.X_development.index, "fold"
            ].to_numpy(),
            "bad_credit_risk": partitions.y_development.to_numpy(),
            "tuned_xgboost_probability": nested.probabilities.to_numpy(),
            "tuned_xgboost_prediction": nested.predictions.to_numpy(),
            "untuned_xgboost_probability": stage4_oof[
                "xgboost_probability"
            ].to_numpy(),
            "untuned_xgboost_prediction": stage4_oof[
                "xgboost_prediction"
            ].to_numpy(),
        }
    )
    if nested_oof.isna().any().any() or nested_oof["source_row_id"].duplicated().any():
        raise RuntimeError("Nested OOF evidence is incomplete.")
    changes = prediction_change_summary(nested_oof)
    stability = hyperparameter_stability(nested.outer_best_params)
    fold_summary = fold_metric_summary(nested.outer_fold_metrics)
    calibration = calibration_points(partitions.y_development, nested_oof)

    untuned_gaps = pd.read_csv(
        project_root / "reports" / "stage4" / "train_validation_gap.csv"
    )
    untuned_gaps = untuned_gaps.loc[
        (untuned_gaps["model"] == "XGBoost")
        & untuned_gaps["metric"].isin(
            ["roc_auc", "average_precision", "recall_bad_credit_risk"]
        )
    ].rename(columns={"fold": "outer_fold"})
    untuned_gaps["model"] = "Untuned XGBoost (Stage 4)"
    gaps = pd.concat(
        [
            untuned_gaps[
                ["model", "outer_fold", "metric", "training_value", "validation_value", "training_minus_validation"]
            ],
            nested.train_validation_gaps,
        ],
        ignore_index=True,
    )

    output_dir = project_root / "reports" / "stage5"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    nested_oof_path = output_dir / "nested_oof_predictions.csv"
    fold_path = output_dir / "outer_fold_membership.csv"
    nested_oof.to_csv(nested_oof_path, index=False, lineterminator="\n")
    fold_membership.to_csv(fold_path, index=False, lineterminator="\n")
    comparison.to_csv(output_dir / "nested_model_comparison.csv", index=False)
    nested.outer_fold_metrics.to_csv(output_dir / "outer_fold_metrics.csv", index=False)
    fold_summary.to_csv(output_dir / "outer_fold_metric_summary.csv", index=False)
    nested.outer_best_params.to_csv(output_dir / "outer_fold_best_params.csv", index=False)
    stability.to_csv(output_dir / "hyperparameter_stability.csv", index=False)
    nested.search_results.to_csv(output_dir / "search_results_compact.csv", index=False)
    tuned_vs_untuned.to_csv(output_dir / "tuned_vs_untuned.csv", index=False)
    changes.to_csv(output_dir / "prediction_change_summary.csv", index=False)
    gaps.to_csv(output_dir / "train_validation_gap.csv", index=False)
    final_summary.to_csv(output_dir / "final_development_search_summary.csv", index=False)
    (output_dir / "final_development_search_best_params.json").write_text(
        json.dumps(
            {
                "selection_metric": PRIMARY_SCORER,
                "best_mean_inner_average_precision": float(
                    final_results.loc[final_best_index, "mean_test_average_precision"]
                ),
                "best_parameters": final_best_params,
                "unbiased_performance_estimate": False,
                "estimator_refit_on_all_development_rows": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    calibration.to_csv(output_dir / "calibration_comparison.csv", index=False)

    figure_paths = create_figures(
        partitions.y_development,
        nested_oof,
        calibration,
        gaps,
        stability,
        figures_dir,
    )
    _write_summary(
        output_dir / "stage5_tuning_summary.md",
        comparison,
        fold_summary,
        nested.outer_best_params,
        changes,
        gaps,
        final_best_params,
    )

    split_hash_after = _sha256(split_path)
    raw_hash_after = _sha256(raw_path)
    stage4_fold_hash_after = _sha256(
        project_root / "reports" / "stage4" / "fold_membership.csv"
    )
    if split_hash_after != split_hash_before or raw_hash_after != raw_hash_before:
        raise RuntimeError("Stage 5 changed locked source/split evidence.")
    if stage4_fold_hash_after != STAGE4_FOLD_SHA256:
        raise RuntimeError("Stage 5 changed the Stage 4 outer folds.")

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
        "stage": "Stage 5: controlled XGBoost hyperparameter tuning",
        "environment": environment,
        "development_rows": 800,
        "final_test_rows_locked_and_not_scored": 200,
        "predictive_feature_count": len(PREDICTIVE_FEATURES),
        "numeric_features": list(NUMERICAL_FEATURES),
        "categorical_one_hot_features": list(CATEGORICAL_FEATURES),
        "audit_only_attributes": list(AUDIT_ATTRIBUTES),
        "outer_cv": {
            "source": "persisted Stage 4 fold membership",
            "folds": 5,
            "sha256": stage4_fold_hash_after,
        },
        "inner_cv": {
            "type": "StratifiedKFold",
            "folds": INNER_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "search": {
            "type": "RandomizedSearchCV",
            "n_iter": SEARCH_ITERATIONS,
            "random_state": RANDOM_STATE,
            "primary_metric": PRIMARY_SCORER,
            "search_n_jobs": -1,
            "xgboost_n_jobs": 1,
            "space": SEARCH_SPACE,
        },
        "threshold": DEFAULT_THRESHOLD,
        "cost_definition": "5 * false_negative + 1 * false_positive",
        "split_sha256_before": split_hash_before,
        "split_sha256_after": split_hash_after,
        "raw_sha256_before": raw_hash_before,
        "raw_sha256_after": raw_hash_after,
        "stage4_oof_sha256": _sha256(stage4_oof_path),
        "outer_fold_membership_sha256": _sha256(fold_path),
        "nested_oof_predictions_sha256": _sha256(nested_oof_path),
        "final_selected_parameters": final_best_params,
        "warnings": sorted({str(item.message) for item in caught_warnings}),
        "final_test_predictions_generated": False,
        "final_test_metrics_generated": False,
        "threshold_optimization_performed": False,
        "class_or_sample_weighting_performed": False,
        "resampling_performed": False,
        "probability_calibration_performed": False,
        "shap_performed": False,
        "feature_selection_performed": False,
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
    summary = generate_stage5_artifacts(args.project_root.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
