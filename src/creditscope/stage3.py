"""Execute the semantically remediated development-only baseline workflow."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from creditscope.analysis import build_analysis_dataset
from creditscope.data import load_verified_raw_snapshot
from creditscope.evaluation import (
    CrossValidationResult,
    coefficient_table,
    evaluate_cross_validation,
    fold_metric_summary,
    oof_summary,
)
from creditscope.modeling import (
    AUDIT_ATTRIBUTES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_dummy_pipeline,
    build_logistic_pipeline,
    partition_model_data,
)
from creditscope.splits import write_locked_split

plt.switch_backend("Agg")

LOCKED_MANIFEST_SHA256 = (
    "32fb4c9ac2cdb358e3bffd145cd97e1a393c14946888d46212d0b395523e99af"
)


def feature_set_frame() -> pd.DataFrame:
    """Document predictive feature types and audit-only exclusions."""
    rows = []
    groups = (
        (
            NUMERICAL_FEATURES,
            "predictive",
            "continuous/quantitative",
            "median imputation then standard scaling",
        ),
        (
            CATEGORICAL_FEATURES,
            "predictive",
            "categorical or discretized/ordinal",
            "most-frequent imputation then one-hot encoding",
        ),
        (
            AUDIT_ATTRIBUTES,
            "audit only",
            "direct sensitive/demographic attribute",
            "excluded from preprocessing and prediction",
        ),
    )
    for features, role, feature_type, encoding in groups:
        for feature in features:
            rows.append(
                {
                    "feature": feature,
                    "role": role,
                    "feature_type": feature_type,
                    "baseline_encoding": encoding,
                    "included_in_baseline_prediction": role == "predictive",
                }
            )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_baseline_figures(
    y_development: pd.Series,
    results: list[CrossValidationResult],
    figures_dir: Path,
) -> list[Path]:
    """Create OOF development-only discrimination and calibration figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"Dummy prior": "#7F8C8D", "Logistic Regression": "#2B6F8A"}
    created = []

    fig, axes = plt.subplots(1, len(results), figsize=(11, 4.5))
    for ax, result in zip(axes, results, strict=True):
        matrix = confusion_matrix(
            y_development, result.oof_predictions, labels=[0, 1]
        )
        display = ConfusionMatrixDisplay(
            matrix,
            display_labels=["Good risk (0)", "Bad risk (1)"],
        )
        display.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(f"{result.model}\nOOF development predictions")
    fig.suptitle("Confusion matrices at the fixed 0.50 threshold")
    fig.tight_layout()
    path = figures_dir / "confusion_matrices.png"
    _save_figure(fig, path)
    created.append(path)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for result in results:
        RocCurveDisplay.from_predictions(
            y_development,
            result.oof_probabilities,
            name=result.model,
            color=colors[result.model],
            ax=ax,
        )
    ax.set_title("ROC curves from OOF development probabilities")
    path = figures_dir / "roc_curves.png"
    _save_figure(fig, path)
    created.append(path)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for result in results:
        PrecisionRecallDisplay.from_predictions(
            y_development,
            result.oof_probabilities,
            name=result.model,
            color=colors[result.model],
            ax=ax,
        )
    ax.set_title("Precision-recall curves from OOF development probabilities")
    path = figures_dir / "precision_recall_curves.png"
    _save_figure(fig, path)
    created.append(path)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#555555",
        label="Perfect calibration",
    )
    for result in results:
        observed_rate, mean_probability = calibration_curve(
            y_development,
            result.oof_probabilities,
            n_bins=10,
            strategy="quantile",
        )
        ax.plot(
            mean_probability,
            observed_rate,
            marker="o",
            label=result.model,
            color=colors[result.model],
        )
    ax.set(
        title="Calibration from OOF development probabilities",
        xlabel="Mean predicted bad-credit-risk probability",
        ylabel="Observed bad-credit-risk rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(frameon=False)
    path = figures_dir / "calibration.png"
    _save_figure(fig, path)
    created.append(path)
    return created


def create_before_after_comparison(
    legacy_results: pd.DataFrame,
    remediated_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare aggregate Logistic Regression OOF results without ranking them."""
    original = legacy_results.set_index("model").loc["Logistic Regression"]
    remediated = remediated_results.set_index("model").loc[
        "Logistic Regression"
    ]
    metrics = (
        ("ROC-AUC", "roc_auc"),
        ("Average Precision", "average_precision"),
        ("Precision for bad credit risk", "precision_bad_credit_risk"),
        ("Recall for bad credit risk", "recall_bad_credit_risk"),
        ("F1 for bad credit risk", "f1_bad_credit_risk"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("Brier score", "brier_score"),
        ("Accuracy", "accuracy"),
        ("False positives", "false_positive"),
        ("False negatives", "false_negative"),
        ("True positives", "true_positive"),
        ("True negatives", "true_negative"),
        ("Total 5:1 cost", "total_cost"),
        ("Average 5:1 cost", "average_cost"),
    )
    rows = []
    for label, column in metrics:
        original_value = float(original[column])
        remediated_value = float(remediated[column])
        rows.append(
            {
                "metric": label,
                "original_stage3": original_value,
                "remediated_stage3r": remediated_value,
                "absolute_difference": abs(remediated_value - original_value),
            }
        )
    return pd.DataFrame(rows)


def generate_stage3_artifacts(project_root: Path) -> dict[str, object]:
    """Execute Stage 3R without evaluating final-test predictive performance."""
    data = build_analysis_dataset(load_verified_raw_snapshot(project_root))
    manifest, split_summary = write_locked_split(data, project_root / "reports")
    if split_summary["manifest_sha256"] != LOCKED_MANIFEST_SHA256:
        raise ValueError("The split manifest does not match the locked Stage 3 hash.")
    partitions = partition_model_data(data, manifest)

    dummy_result = evaluate_cross_validation(
        "Dummy prior",
        build_dummy_pipeline(),
        partitions.X_development,
        partitions.y_development,
    )
    logistic_result = evaluate_cross_validation(
        "Logistic Regression",
        build_logistic_pipeline(),
        partitions.X_development,
        partitions.y_development,
    )
    results = [dummy_result, logistic_result]

    output_dir = project_root / "reports" / "stage3_remediated"
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics = pd.concat(
        [result.fold_metrics for result in results], ignore_index=True
    )
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    fold_metric_summary(fold_metrics).to_csv(
        output_dir / "cv_metric_summary.csv", index=False
    )

    comparison = pd.DataFrame(
        [oof_summary(result, partitions.y_development) for result in results]
    )
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)
    legacy_results_path = (
        project_root
        / "reports"
        / "stage3_legacy_pre_semantic_audit"
        / "baseline_comparison.csv"
    )
    if not legacy_results_path.is_file():
        raise FileNotFoundError("The preserved Stage 3 comparison table is required.")
    create_before_after_comparison(
        pd.read_csv(legacy_results_path), comparison
    ).to_csv(output_dir / "original_vs_remediated.csv", index=False)
    comparison[
        [
            "model",
            "threshold",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
            "false_negative_cost",
            "false_positive_cost",
            "total_cost",
            "average_cost",
        ]
    ].to_csv(output_dir / "cost_baseline.csv", index=False)
    feature_set_frame().to_csv(output_dir / "feature_set_summary.csv", index=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fitted_logistic = build_logistic_pipeline().fit(
            partitions.X_development, partitions.y_development
        )
    convergence_warnings = sum(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    coefficients = coefficient_table(fitted_logistic)
    coefficients.to_csv(output_dir / "logistic_coefficients.csv", index=False)

    figures = create_baseline_figures(
        partitions.y_development,
        results,
        output_dir / "figures",
    )
    run_summary = {
        "stage": "3R semantic remediation",
        "authoritative_baseline": True,
        "legacy_results_preserved": True,
        "development_rows_used_for_cv": len(partitions.X_development),
        "final_test_rows_isolated": len(partitions.X_test),
        "predictive_feature_count": len(partitions.X_development.columns),
        "numeric_feature_count": len(NUMERICAL_FEATURES),
        "one_hot_feature_count": len(CATEGORICAL_FEATURES),
        "audit_attribute_count": len(partitions.audit_development.columns),
        "cv_folds": 5,
        "random_state": 42,
        "default_threshold": 0.5,
        "convergence_warnings": convergence_warnings,
        "figures_created": len(figures),
        "final_test_performance_evaluated": False,
        "test_rows_used_for_fitting": False,
        "hyperparameter_tuning_performed": False,
        "threshold_optimization_performed": False,
        "resampling_performed": False,
        "class_weight": None,
        "models_trained": ["DummyClassifier", "LogisticRegression"],
        "split_manifest_sha256": split_summary["manifest_sha256"],
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
    result = generate_stage3_artifacts(args.project_root.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
