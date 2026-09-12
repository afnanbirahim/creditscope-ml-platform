"""Run the untuned Stage 4 tree-model comparison on development data only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import warnings
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path

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
)
from sklearn.pipeline import Pipeline

from creditscope.analysis import build_analysis_dataset
from creditscope.data import load_verified_raw_snapshot
from creditscope.evaluation import (
    CrossValidationResult,
    evaluate_cross_validation,
    fold_metric_summary,
    make_cv_splits,
    oof_summary,
)
from creditscope.modeling import (
    AUDIT_ATTRIBUTES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PREDICTIVE_FEATURES,
    RANDOM_FOREST_PARAMETERS,
    XGBOOST_PARAMETERS,
    build_logistic_pipeline,
    build_random_forest_pipeline,
    build_xgboost_pipeline,
    partition_model_data,
)
from creditscope.splits import write_locked_split
from creditscope.stage3 import LOCKED_MANIFEST_SHA256

plt.switch_backend("Agg")

MODEL_NAMES = ("Logistic Regression", "Random Forest", "XGBoost")
COLORS = {
    "Logistic Regression": "#2B6F8A",
    "Random Forest": "#C26D2A",
    "XGBoost": "#4B8B5A",
}
GAP_METRICS = (
    "roc_auc",
    "average_precision",
    "recall_bad_credit_risk",
    "balanced_accuracy",
    "brier_score",
    "accuracy",
)
FLOAT_METRIC_ATOL = 1e-10
EXACT_REPRODUCIBILITY_COLUMNS = (
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
    "total_cost",
)
FLOAT_REPRODUCIBILITY_COLUMNS = (
    "roc_auc",
    "average_precision",
    "precision_bad_credit_risk",
    "recall_bad_credit_risk",
    "f1_bad_credit_risk",
    "balanced_accuracy",
    "brier_score",
    "accuracy",
    "average_cost",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_table(
    y_development: pd.Series,
    results: list[CrossValidationResult],
) -> pd.DataFrame:
    """Return comparable quantile-bin OOF calibration points."""
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        observed, predicted = calibration_curve(
            y_development,
            result.oof_probabilities,
            n_bins=10,
            strategy="quantile",
        )
        for bin_number, (mean_probability, observed_rate) in enumerate(
            zip(predicted, observed, strict=True), start=1
        ):
            rows.append(
                {
                    "model": result.model,
                    "bin": bin_number,
                    "mean_predicted_bad_credit_risk_probability": mean_probability,
                    "observed_bad_credit_risk_rate": observed_rate,
                }
            )
    return pd.DataFrame(rows)


def train_validation_gap_table(
    results: list[CrossValidationResult],
) -> pd.DataFrame:
    """Create a long-form diagnostic without changing model settings."""
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        for row in result.fold_metrics.to_dict(orient="records"):
            for metric in GAP_METRICS:
                training_column = f"training_{metric}"
                if training_column not in row:
                    continue
                training_value = float(row[training_column])
                validation_value = float(row[metric])
                rows.append(
                    {
                        "model": result.model,
                        "fold": int(row["fold"]),
                        "metric": metric,
                        "training_value": training_value,
                        "validation_value": validation_value,
                        "training_minus_validation": training_value
                        - validation_value,
                    }
                )
    return pd.DataFrame(rows)


def feature_importance_table(fitted_pipeline: Pipeline) -> pd.DataFrame:
    """Extract model-specific encoded feature importances."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]
    transformed_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    rows = []
    for transformed_name, importance in zip(
        transformed_names, importances, strict=True
    ):
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
                "importance": float(importance),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        "importance", ascending=False, ignore_index=True
    )
    frame["importance_rank"] = np.arange(1, len(frame) + 1)
    return frame


def model_agreement_table(results: list[CrossValidationResult]) -> pd.DataFrame:
    """Summarize agreement among aligned OOF development predictions."""
    result_by_name = {result.model: result for result in results}
    if tuple(result_by_name) != MODEL_NAMES:
        raise ValueError("Agreement analysis requires LR, RF, and XGBoost in order.")
    prediction_frame = pd.DataFrame(
        {name: result_by_name[name].oof_predictions for name in MODEL_NAMES}
    )
    probability_frame = pd.DataFrame(
        {name: result_by_name[name].oof_probabilities for name in MODEL_NAMES}
    )
    comparisons = (
        ("Logistic Regression vs Random Forest", MODEL_NAMES[0], MODEL_NAMES[1]),
        ("Logistic Regression vs XGBoost", MODEL_NAMES[0], MODEL_NAMES[2]),
        ("Random Forest vs XGBoost", MODEL_NAMES[1], MODEL_NAMES[2]),
    )
    rows = []
    for label, left, right in comparisons:
        agreement = prediction_frame[left] == prediction_frame[right]
        rows.append(
            {
                "comparison": label,
                "criterion": "same class at threshold 0.50",
                "observation_count": int(agreement.sum()),
                "proportion_of_development": float(agreement.mean()),
            }
        )
    all_agree = prediction_frame.nunique(axis="columns") == 1
    rows.append(
        {
            "comparison": "All three models",
            "criterion": "same class at threshold 0.50",
            "observation_count": int(all_agree.sum()),
            "proportion_of_development": float(all_agree.mean()),
        }
    )
    probability_spread = (
        probability_frame.max(axis="columns")
        - probability_frame.min(axis="columns")
        >= 0.20
    )
    rows.append(
        {
            "comparison": "All three models",
            "criterion": "maximum OOF probability spread at least 0.20",
            "observation_count": int(probability_spread.sum()),
            "proportion_of_development": float(probability_spread.mean()),
        }
    )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_stage4_figures(
    y_development: pd.Series,
    results: list[CrossValidationResult],
    calibration: pd.DataFrame,
    gaps: pd.DataFrame,
    importance_tables: dict[str, pd.DataFrame],
    figures_dir: Path,
) -> list[Path]:
    """Create focused Stage 4 development-only comparison figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    created: list[Path] = []

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for result in results:
        RocCurveDisplay.from_predictions(
            y_development,
            result.oof_probabilities,
            name=result.model,
            color=COLORS[result.model],
            ax=ax,
        )
    ax.set_title("ROC comparison: development-set OOF predictions")
    path = figures_dir / "roc_comparison.png"
    _save_figure(fig, path)
    created.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for result in results:
        PrecisionRecallDisplay.from_predictions(
            y_development,
            result.oof_probabilities,
            name=result.model,
            color=COLORS[result.model],
            ax=ax,
        )
    ax.set_title("Precision-recall comparison: development-set OOF predictions")
    path = figures_dir / "precision_recall_comparison.png"
    _save_figure(fig, path)
    created.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, result in zip(axes, results, strict=True):
        matrix = confusion_matrix(
            y_development, result.oof_predictions, labels=[0, 1]
        )
        ConfusionMatrixDisplay(
            matrix,
            display_labels=["Good risk (0)", "Bad risk (1)"],
        ).plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(result.model)
    fig.suptitle("Development OOF confusion matrices at threshold 0.50")
    fig.tight_layout()
    path = figures_dir / "confusion_matrices.png"
    _save_figure(fig, path)
    created.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.plot([0, 1], [0, 1], "--", color="#555555", label="Perfect calibration")
    for model_name, group in calibration.groupby("model", sort=False):
        ax.plot(
            group["mean_predicted_bad_credit_risk_probability"],
            group["observed_bad_credit_risk_rate"],
            marker="o",
            color=COLORS[model_name],
            label=model_name,
        )
    ax.set(
        title="Calibration comparison: development-set OOF probabilities",
        xlabel="Mean predicted bad-credit-risk probability",
        ylabel="Observed bad-credit-risk rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(frameon=False)
    path = figures_dir / "calibration_comparison.png"
    _save_figure(fig, path)
    created.append(path)

    gap_summary = (
        gaps.groupby(["model", "metric"], sort=False)[
            ["training_value", "validation_value"]
        ]
        .mean()
        .reset_index()
    )
    selected = gap_summary[gap_summary["metric"].isin(["roc_auc", "average_precision"])]
    labels = selected["model"] + "\n" + selected["metric"].str.replace("_", " ")
    positions = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(9, 5.4))
    width = 0.36
    ax.bar(
        positions - width / 2,
        selected["training_value"],
        width,
        label="Fold training",
        color="#93A8AC",
    )
    ax.bar(
        positions + width / 2,
        selected["validation_value"],
        width,
        label="Fold validation",
        color="#D08B5B",
    )
    ax.set_xticks(positions, labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_title("Untuned tree-model train-versus-validation diagnostic")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figures_dir / "train_validation_diagnostic.png"
    _save_figure(fig, path)
    created.append(path)

    for model_name, table in importance_tables.items():
        top = table.head(15).sort_values("importance")
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        ax.barh(top["transformed_feature"], top["importance"], color=COLORS[model_name])
        ax.set(
            title=f"{model_name}: top model-intrinsic feature importances",
            xlabel="Model-specific importance",
            ylabel="Encoded feature",
        )
        fig.tight_layout()
        slug = "random_forest" if model_name == "Random Forest" else "xgboost"
        path = figures_dir / f"feature_importance_{slug}.png"
        _save_figure(fig, path)
        created.append(path)
    return created


def verify_logistic_reference(
    project_root: Path,
    comparison: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> dict[str, object]:
    """Apply the authorized exact/count and floating-point reconciliation policy."""
    reference_path = (
        project_root / "reports" / "stage3_remediated" / "baseline_comparison.csv"
    )
    reference = pd.read_csv(reference_path).set_index("model").loc[
        "Logistic Regression"
    ]
    observed = comparison.set_index("model").loc["Logistic Regression"]
    if not observed[list(EXACT_REPRODUCIBILITY_COLUMNS)].equals(
        reference[list(EXACT_REPRODUCIBILITY_COLUMNS)]
    ):
        raise ValueError("Recomputed Logistic Regression exact counts/cost do not match Stage 3R.")
    float_differences = (
        observed[list(FLOAT_REPRODUCIBILITY_COLUMNS)].astype(float)
        - reference[list(FLOAT_REPRODUCIBILITY_COLUMNS)].astype(float)
    ).abs()
    if not np.allclose(
        observed[list(FLOAT_REPRODUCIBILITY_COLUMNS)].astype(float),
        reference[list(FLOAT_REPRODUCIBILITY_COLUMNS)].astype(float),
        rtol=0,
        atol=FLOAT_METRIC_ATOL,
    ):
        raise ValueError("Recomputed Logistic Regression does not match Stage 3R.")

    stored_folds = pd.read_csv(
        project_root / "reports" / "stage3_remediated" / "fold_metrics.csv"
    )
    stored_folds = stored_folds.loc[
        stored_folds["model"] == "Logistic Regression"
    ].reset_index(drop=True)
    observed_folds = fold_metrics.reset_index(drop=True)
    exact_fold_columns = (
        "fold",
        "training_rows",
        "validation_rows",
        "validation_good_credit_risk",
        "validation_bad_credit_risk",
        "default_threshold_total_cost",
    )
    if not observed_folds[list(exact_fold_columns)].equals(
        stored_folds[list(exact_fold_columns)]
    ):
        raise ValueError("Recomputed Logistic Regression fold identities/counts differ.")
    float_fold_columns = tuple(
        column
        for column in observed_folds.columns
        if column not in {"model", *exact_fold_columns}
    )
    fold_differences = (
        observed_folds[list(float_fold_columns)].astype(float)
        - stored_folds[list(float_fold_columns)].astype(float)
    ).abs()
    if not np.allclose(
        observed_folds[list(float_fold_columns)].astype(float),
        stored_folds[list(float_fold_columns)].astype(float),
        rtol=0,
        atol=FLOAT_METRIC_ATOL,
    ):
        raise ValueError("Recomputed Logistic Regression fold metrics do not match Stage 3R.")
    return {
        "floating_point_atol": FLOAT_METRIC_ATOL,
        "maximum_aggregate_absolute_difference": float(float_differences.max()),
        "maximum_fold_absolute_difference": float(fold_differences.max().max()),
        "exact_counts_and_cost_match": True,
        "floating_metrics_match": True,
    }


def fold_membership_table(
    row_index: pd.Index,
    cv_splits: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> pd.DataFrame:
    """Persist the paired development-fold assignment using stable row identifiers."""
    assignments = pd.Series(index=row_index, dtype="int8", name="fold")
    for fold_number, (_, validation_positions) in enumerate(cv_splits, start=1):
        assignments.iloc[validation_positions] = fold_number
    if assignments.isna().any() or not assignments.between(1, 5).all():
        raise RuntimeError("Every development row must have exactly one fold identifier.")
    return assignments.astype("int8").rename_axis("source_row_id").reset_index()


def oof_prediction_table(
    y_development: pd.Series,
    fold_membership: pd.DataFrame,
    results: list[CrossValidationResult],
) -> pd.DataFrame:
    """Create row-level development OOF evidence without final-holdout data."""
    frame = fold_membership.set_index("source_row_id")
    frame["bad_credit_risk"] = y_development.astype("int8")
    slugs = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
    }
    for result in results:
        slug = slugs[result.model]
        frame[f"{slug}_probability"] = result.oof_probabilities
        frame[f"{slug}_prediction"] = result.oof_predictions.astype("int8")
    if frame.isna().any().any() or len(frame) != len(y_development):
        raise RuntimeError("OOF evidence is incomplete or misaligned.")
    return frame.reset_index()


def numerical_environment() -> tuple[dict[str, str], str]:
    """Return concise versions plus scikit-learn's detailed environment report."""
    packages = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": version("xgboost"),
        "joblib": version("joblib"),
        "threadpoolctl": version("threadpoolctl"),
    }
    detail = io.StringIO()
    with redirect_stdout(detail):
        sklearn.show_versions()
    return packages, detail.getvalue()


def _write_summary(
    output_dir: Path,
    comparison: pd.DataFrame,
    fold_summary: pd.DataFrame,
    calibration: pd.DataFrame,
    gaps: pd.DataFrame,
    agreement: pd.DataFrame,
    importance_tables: dict[str, pd.DataFrame],
) -> None:
    indexed = comparison.set_index("model")
    strongest_auc = indexed["roc_auc"].idxmax()
    strongest_average_precision = indexed["average_precision"].idxmax()
    strongest_f1 = indexed["f1_bad_credit_risk"].idxmax()
    strongest_balanced_accuracy = indexed["balanced_accuracy"].idxmax()
    strongest_recall = indexed["recall_bad_credit_risk"].idxmax()
    lowest_cost = indexed["total_cost"].idxmin()
    lowest_brier = indexed["brier_score"].idxmin()
    gap_means = gaps.groupby(["model", "metric"])[
        "training_minus_validation"
    ].mean()
    all_agree = agreement.loc[
        (agreement["comparison"] == "All three models")
        & (agreement["criterion"] == "same class at threshold 0.50")
    ].iloc[0]
    calibration_diagnostic = calibration.assign(
        absolute_bin_gap=lambda frame: (
            frame["observed_bad_credit_risk_rate"]
            - frame["mean_predicted_bad_credit_risk_probability"]
        ).abs()
    ).groupby("model", sort=False)["absolute_bin_gap"].mean()
    rf_top = importance_tables["Random Forest"].iloc[0]
    xgb_top = importance_tables["XGBoost"].iloc[0]
    lines = [
        "# Stage 4 untuned tree-model comparison",
        "",
        (
            "All results below use out-of-fold predictions on the locked 800-row "
            "development partition. The final 200-row holdout was not scored or "
            "inspected."
        ),
        "",
        "## Locked methodology",
        "",
        "- Logistic Regression is the unchanged Stage 3R reference.",
        "- Random Forest and XGBoost are single, untuned candidates.",
        "- All models use the same five precomputed stratified folds.",
        "- The class threshold is fixed at 0.50. No class weighting was used.",
        "- Bad credit risk (1) is positive; cost is `5 * FN + 1 * FP`.",
        "- Direct audit attributes are excluded; 2 numeric and 15 one-hot fields remain.",
        "",
        "## Aggregate development OOF comparison",
        "",
        comparison[
            [
                "model",
                "roc_auc",
                "average_precision",
                "recall_bad_credit_risk",
                "balanced_accuracy",
                "brier_score",
                "accuracy",
                "false_positive",
                "false_negative",
                "total_cost",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        (
            "Accuracy is secondary: predicting every observation as good risk reaches "
            "70% accuracy on this development population while missing every bad-risk "
            "case."
        ),
        "",
        "## Fold variability",
        "",
        fold_summary[fold_summary["metric"].isin(["roc_auc", "average_precision", "brier_score"])]
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Complexity diagnostic",
        "",
        (
            f"Random Forest mean train-minus-validation ROC-AUC gap: "
            f"{gap_means.loc[('Random Forest', 'roc_auc')]:.4f}."
        ),
        (
            f"XGBoost mean train-minus-validation ROC-AUC gap: "
            f"{gap_means.loc[('XGBoost', 'roc_auc')]:.4f}."
        ),
        "These gaps are diagnostics, not tuning criteria in this stage.",
        "",
        "## Model agreement",
        "",
        (
            f"All three candidates assign the same 0.50-threshold class to "
            f"{int(all_agree['observation_count'])} of 800 development observations "
            f"({float(all_agree['proportion_of_development']):.1%})."
        ),
        "",
        "## Candidate tradeoffs",
        "",
        f"- Strongest ROC-AUC: {strongest_auc}.",
        f"- Strongest Average Precision: {strongest_average_precision}.",
        f"- Strongest bad-risk recall at 0.50: {strongest_recall}.",
        f"- Strongest bad-risk F1 at 0.50: {strongest_f1}.",
        f"- Strongest balanced accuracy at 0.50: {strongest_balanced_accuracy}.",
        f"- Lowest 5:1 cost at 0.50: {lowest_cost}.",
        f"- Lowest Brier score: {lowest_brier}.",
        "",
        (
            "These development-only results do not select a final or production model. "
            "Stage 5 must decide whether controlled tuning is justified and how to "
            "govern it."
        ),
        "",
        "## Importance limitations",
        "",
        (
            "Random Forest impurity importance and XGBoost gain-derived importance are "
            "model-specific diagnostics. Correlation and one-hot expansion can split or "
            "distort importance, and impurity measures may favor variables with more "
            "split opportunities. Importance is not causality and was not used for "
            "feature selection."
        ),
        "",
        (
            f"The leading encoded Random Forest importance was "
            f"`{rf_top['transformed_feature']}` ({float(rf_top['importance']):.4f}); "
            f"the leading XGBoost importance was "
            f"`{xgb_top['transformed_feature']}` "
            f"({float(xgb_top['importance']):.4f})."
        ),
        "",
        "## Probability-quality diagnostic",
        "",
        (
            "Mean absolute gaps across ten quantile reliability bins were "
            + ", ".join(
                f"{model}: {value:.4f}"
                for model, value in calibration_diagnostic.items()
            )
            + ". These descriptive bin gaps and Brier scores do not constitute "
            "probability calibration, and the small dataset limits interpretation."
        ),
    ]
    (output_dir / "stage4_model_comparison_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def generate_stage4_artifacts(project_root: Path) -> dict[str, object]:
    """Execute Stage 4 without producing any final-holdout predictions."""
    manifest_path = project_root / "reports" / "split_manifest.csv"
    manifest_hash_before = _sha256(manifest_path)
    if manifest_hash_before != LOCKED_MANIFEST_SHA256:
        raise ValueError("Split manifest hash does not match the Stage 3 lock.")

    raw_path = project_root / "data" / "raw" / "german_credit.csv"
    raw_hash_before = _sha256(raw_path)
    data = build_analysis_dataset(load_verified_raw_snapshot(project_root))
    manifest, split_summary = write_locked_split(data, project_root / "reports")
    if split_summary["manifest_sha256"] != LOCKED_MANIFEST_SHA256:
        raise ValueError("Canonical split content does not match the Stage 3 lock.")
    partitions = partition_model_data(data, manifest)

    paired_splits = make_cv_splits(
        partitions.X_development, partitions.y_development
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        logistic = evaluate_cross_validation(
            "Logistic Regression",
            build_logistic_pipeline(),
            partitions.X_development,
            partitions.y_development,
            cv_splits=paired_splits,
        )
        logistic_comparison = pd.DataFrame(
            [oof_summary(logistic, partitions.y_development)]
        )
        reconciliation = verify_logistic_reference(
            project_root, logistic_comparison, logistic.fold_metrics
        )

        # The tree candidates are constructed and trained only after the gate passes.
        random_forest = evaluate_cross_validation(
            "Random Forest",
            build_random_forest_pipeline(),
            partitions.X_development,
            partitions.y_development,
            cv_splits=paired_splits,
            include_training_metrics=True,
        )
        xgboost = evaluate_cross_validation(
            "XGBoost",
            build_xgboost_pipeline(),
            partitions.X_development,
            partitions.y_development,
            cv_splits=paired_splits,
            include_training_metrics=True,
        )
    results = [logistic, random_forest, xgboost]

    output_dir = project_root / "reports" / "stage4"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics = pd.concat(
        [result.fold_metrics for result in results], ignore_index=True
    )
    fold_summary = fold_metric_summary(fold_metrics)
    comparison = pd.DataFrame(
        [oof_summary(result, partitions.y_development) for result in results]
    )
    cost = comparison[
        [
            "model",
            "threshold",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
            "total_cost",
            "average_cost",
        ]
    ].copy()
    calibration = calibration_table(partitions.y_development, results)
    gaps = train_validation_gap_table([random_forest, xgboost])
    agreement = model_agreement_table(results)

    fitted_rf = build_random_forest_pipeline().fit(
        partitions.X_development, partitions.y_development
    )
    fitted_xgb = build_xgboost_pipeline().fit(
        partitions.X_development, partitions.y_development
    )
    importance_tables = {
        "Random Forest": feature_importance_table(fitted_rf),
        "XGBoost": feature_importance_table(fitted_xgb),
    }

    fold_membership = fold_membership_table(
        partitions.X_development.index, paired_splits
    )
    oof_predictions = oof_prediction_table(
        partitions.y_development, fold_membership, results
    )

    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    fold_summary.to_csv(output_dir / "fold_metric_summary.csv", index=False)
    cost.to_csv(output_dir / "cost_comparison.csv", index=False)
    calibration.to_csv(output_dir / "calibration_comparison.csv", index=False)
    gaps.to_csv(output_dir / "train_validation_gap.csv", index=False)
    agreement.to_csv(output_dir / "model_agreement.csv", index=False)
    importance_tables["Random Forest"].to_csv(
        output_dir / "feature_importance_random_forest.csv", index=False
    )
    importance_tables["XGBoost"].to_csv(
        output_dir / "feature_importance_xgboost.csv", index=False
    )
    fold_membership_path = output_dir / "fold_membership.csv"
    oof_predictions_path = output_dir / "development_oof_predictions.csv"
    fold_membership.to_csv(fold_membership_path, index=False, lineterminator="\n")
    oof_predictions.to_csv(oof_predictions_path, index=False, lineterminator="\n")
    fold_membership_sha256 = _sha256(fold_membership_path)
    oof_predictions_sha256 = _sha256(oof_predictions_path)
    figure_paths = create_stage4_figures(
        partitions.y_development,
        results,
        calibration,
        gaps,
        importance_tables,
        figures_dir,
    )
    _write_summary(
        output_dir,
        comparison,
        fold_summary,
        calibration,
        gaps,
        agreement,
        importance_tables,
    )

    manifest_hash_after = _sha256(manifest_path)
    raw_hash_after = _sha256(raw_path)
    if manifest_hash_after != manifest_hash_before:
        raise RuntimeError("Stage 4 changed the locked split manifest.")
    if raw_hash_after != raw_hash_before:
        raise RuntimeError("Stage 4 changed the Stage 1 raw snapshot.")

    environment, show_versions = numerical_environment()
    (output_dir / "sklearn_show_versions.txt").write_text(
        show_versions, encoding="utf-8"
    )
    warning_messages = sorted({str(item.message) for item in caught_warnings})
    run_summary: dict[str, object] = {
        "stage": "Stage 4: untuned tree-model comparison",
        "environment": environment,
        "development_rows": len(partitions.X_development),
        "final_test_rows_locked_and_not_scored": 200,
        "predictive_feature_count": len(PREDICTIVE_FEATURES),
        "numeric_features": list(NUMERICAL_FEATURES),
        "categorical_one_hot_features": list(CATEGORICAL_FEATURES),
        "audit_only_attributes": list(AUDIT_ATTRIBUTES),
        "random_forest_parameters": RANDOM_FOREST_PARAMETERS,
        "xgboost_parameters": XGBOOST_PARAMETERS,
        "cv": {
            "type": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "same_precomputed_folds_for_all_models": True,
        },
        "threshold": 0.50,
        "cost_definition": "5 * false_negative + 1 * false_positive",
        "manifest_sha256_before": manifest_hash_before,
        "manifest_sha256_after": manifest_hash_after,
        "raw_snapshot_sha256_before": raw_hash_before,
        "raw_snapshot_sha256_after": raw_hash_after,
        "logistic_matches_stage3r_under_reconciliation_policy": True,
        "reproducibility_reconciliation": reconciliation,
        "fold_membership_sha256": fold_membership_sha256,
        "development_oof_predictions_sha256": oof_predictions_sha256,
        "compatibility_warnings": warning_messages,
        "final_test_predictions_generated": False,
        "final_test_metrics_generated": False,
        "hyperparameter_search_performed": False,
        "threshold_optimization_performed": False,
        "class_weighting_performed": False,
        "resampling_performed": False,
        "probability_calibration_performed": False,
        "shap_performed": False,
        "figures": [str(path.relative_to(project_root)) for path in figure_paths],
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    summary = generate_stage4_artifacts(args.project_root.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
