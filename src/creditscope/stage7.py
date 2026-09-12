"""Stage 7 one-time evaluation of the frozen policy on the final holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

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

from creditscope.analysis import build_analysis_dataset
from creditscope.data import load_verified_raw_snapshot
from creditscope.evaluation import asymmetric_cost, threshold_predictions
from creditscope.modeling import (
    AUDIT_ATTRIBUTES,
    PREDICTIVE_FEATURES,
    XGBOOST_PARAMETERS,
)
from creditscope.schema import TARGET_BINARY_NAME
from creditscope.splits import (
    DEVELOPMENT_SPLIT,
    ROW_ID_NAME,
    TEST_SPLIT,
    source_row_ids,
)
from creditscope.stage5 import load_outer_folds
from creditscope.stage6 import build_probability_estimator

EXPECTED_ENVIRONMENT = {
    "python": "3.12.0",
    "numpy": "2.5.3",
    "scipy": "1.18.1",
    "scikit_learn": "1.4.2",
    "xgboost": "3.4.1",
}
LOCKED_ARTIFACTS = {
    "raw_snapshot": (
        "data/raw/german_credit.csv",
        "4ce6007c9eb3dbcd67e7858773566b0eb3ff140033cf67b2d8a42b3a89edfb9d",
    ),
    "split_manifest": (
        "reports/split_manifest.csv",
        "32fb4c9ac2cdb358e3bffd145cd97e1a393c14946888d46212d0b395523e99af",
    ),
    "stage4_fold_membership": (
        "reports/stage4/fold_membership.csv",
        "513fc9249a4c00be3ba67fa99ea20a7ac76b653694130cf66733ea08f97d0b92",
    ),
    "stage4_oof": (
        "reports/stage4/development_oof_predictions.csv",
        "c1e96b552ca7f5510f8aa9eda88db128e9cfbdd8ef30ca8535fac69646c486f9",
    ),
    "stage5_nested_oof": (
        "reports/stage5/nested_oof_predictions.csv",
        "eecccdb9ae24d747562b5914c5f54bc5838688df4019b013552d1b87217460f5",
    ),
    "stage6_nested_oof": (
        "reports/stage6/nested_policy_oof_predictions.csv",
        "a11703e95c1fd17c05de3c2636aba4bab74bdd3d93a992140cefd0b58bbda621",
    ),
    "stage6_threshold_search": (
        "reports/stage6/threshold_search.csv",
        "4f738292925c51242b34fad89d8a9c763c19014a3de3afea31f3968c3e7ac1a9",
    ),
    "frozen_policy": (
        "reports/stage6/frozen_model_policy.json",
        "afee2e4ea82d438fa0ba638a8b7bdadaecbce9a4ada7da1f72d66866e4313afc",
    ),
}
FROZEN_THRESHOLD = 0.16
FROZEN_CALIBRATION_METHOD = "sigmoid"
FROZEN_ENSEMBLE = True
FINAL_METRIC_NAMES = (
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
PREDICTION_COLUMNS = (
    ROW_ID_NAME,
    TARGET_BINARY_NAME,
    "calibrated_bad_credit_risk_probability",
    "frozen_threshold",
    "final_prediction",
    "correct_prediction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def environment_versions() -> dict[str, str]:
    """Return the frozen scientific environment versions."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": version("xgboost"),
    }


def verify_environment() -> dict[str, str]:
    """Fail before access if the numerical environment is not exact."""
    observed = environment_versions()
    if observed != EXPECTED_ENVIRONMENT:
        raise RuntimeError(
            f"Stage 7 environment mismatch: expected {EXPECTED_ENVIRONMENT}, "
            f"observed {observed}."
        )
    from xgboost import XGBClassifier  # noqa: F401

    if CalibratedClassifierCV is None:  # pragma: no cover
        raise RuntimeError("CalibratedClassifierCV is unavailable.")
    return observed


def verify_locked_artifacts(project_root: Path) -> dict[str, str]:
    """Require exact hashes for every preregistered artifact."""
    observed = {
        name: _sha256(project_root / relative_path)
        for name, (relative_path, _) in LOCKED_ARTIFACTS.items()
    }
    expected = {name: digest for name, (_, digest) in LOCKED_ARTIFACTS.items()}
    if observed != expected:
        raise RuntimeError(
            f"Locked evidence mismatch before final evaluation: {observed}"
        )
    return observed


def membership_summary(project_root: Path) -> dict[str, Any]:
    """Validate locked membership metadata without loading model features."""
    manifest = pd.read_csv(project_root / "reports" / "split_manifest.csv")
    development = manifest[manifest["split"] == DEVELOPMENT_SPLIT]
    holdout = manifest[manifest["split"] == TEST_SPLIT]
    if len(development) != 800 or len(holdout) != 200:
        raise RuntimeError("Locked development/holdout row counts changed.")
    if development[ROW_ID_NAME].nunique() != 800 or holdout[ROW_ID_NAME].nunique() != 200:
        raise RuntimeError("Locked split contains duplicate identifiers.")
    if not set(development[ROW_ID_NAME]).isdisjoint(holdout[ROW_ID_NAME]):
        raise RuntimeError("Development and holdout identifiers overlap.")
    if development[TARGET_BINARY_NAME].value_counts().to_dict() != {0: 560, 1: 240}:
        raise RuntimeError("Development class counts changed.")
    if holdout[TARGET_BINARY_NAME].value_counts().to_dict() != {0: 140, 1: 60}:
        raise RuntimeError("Holdout class counts changed from locked metadata.")
    return {
        "development_rows": 800,
        "holdout_rows": 200,
        "development_class_counts": {"good_credit_risk": 560, "bad_credit_risk": 240},
        "holdout_class_counts_from_locked_manifest": {"good_credit_risk": 140, "bad_credit_risk": 60},
        "unique_development_ids": 800,
        "unique_holdout_ids": 200,
        "overlap": 0,
    }


def validate_frozen_policy(project_root: Path) -> dict[str, Any]:
    """Validate the exact Stage 6P policy before any holdout access."""
    path = project_root / LOCKED_ARTIFACTS["frozen_policy"][0]
    policy = json.loads(path.read_text(encoding="utf-8"))
    calibration = policy["calibration"]
    if policy["model"]["hyperparameters"] != XGBOOST_PARAMETERS:
        raise RuntimeError("Frozen model configuration changed.")
    if policy["feature_governance"]["predictive_features"] != list(PREDICTIVE_FEATURES):
        raise RuntimeError("Frozen predictive feature set changed.")
    if set(policy["feature_governance"]["audit_only_excluded"]) != set(AUDIT_ATTRIBUTES):
        raise RuntimeError("Frozen audit-only exclusions changed.")
    if calibration["method"] != FROZEN_CALIBRATION_METHOD:
        raise RuntimeError("Frozen calibration method changed.")
    if calibration["ensemble"] is not FROZEN_ENSEMBLE:
        raise RuntimeError("Frozen calibration ensemble behavior changed.")
    if calibration["n_jobs"] is not None or calibration["n_jobs_explicitly_set"] is not False:
        raise RuntimeError("Frozen calibration n_jobs behavior changed.")
    if policy["decision_threshold"] != FROZEN_THRESHOLD:
        raise RuntimeError("Frozen threshold changed.")
    if tuple(policy["stage7_predefined_metrics"]) != FINAL_METRIC_NAMES:
        raise RuntimeError("Frozen final metric set changed.")
    return policy


def build_final_estimator(
    persisted_five_folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> CalibratedClassifierCV:
    """Build the exact five-member frozen sigmoid-calibrated ensemble."""
    estimator = build_probability_estimator(
        FROZEN_CALIBRATION_METHOD,
        calibration_cv=persisted_five_folds,
    )
    if not isinstance(estimator, CalibratedClassifierCV):
        raise TypeError("Frozen final estimator is not calibrated.")
    if estimator.method != "sigmoid" or estimator.ensemble is not True:
        raise RuntimeError("Frozen calibration configuration changed.")
    if estimator.n_jobs is not None:
        raise RuntimeError("CalibratedClassifierCV n_jobs must resolve to None.")
    return estimator


def final_metric_record(
    y_true: pd.Series,
    probabilities: pd.Series,
    predictions: pd.Series,
) -> dict[str, float | int]:
    """Calculate only the metrics frozen before holdout access."""
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    cost = asymmetric_cost(y_true, predictions)
    record: dict[str, float | int] = {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "precision_bad_credit_risk": float(
            precision_score(y_true, predictions, pos_label=1, zero_division=0)
        ),
        "recall_bad_credit_risk": float(
            recall_score(y_true, predictions, pos_label=1, zero_division=0)
        ),
        "f1_bad_credit_risk": float(
            f1_score(y_true, predictions, pos_label=1, zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "specificity": float(tn / (tn + fp)),
        "brier_score": float(brier_score_loss(y_true, probabilities, pos_label=1)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "total_cost": int(cost["total_cost"]),
        "average_cost": float(cost["average_cost"]),
    }
    if tuple(record) != FINAL_METRIC_NAMES:
        raise RuntimeError("Final metric implementation changed.")
    return record


def make_prediction_artifact(
    row_ids: pd.Index,
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Build the preregistered holdout prediction schema at threshold 0.16."""
    probability_series = pd.Series(probabilities, index=row_ids, dtype="float64")
    predictions = threshold_predictions(probability_series, FROZEN_THRESHOLD)
    artifact = pd.DataFrame(
        {
            ROW_ID_NAME: row_ids,
            TARGET_BINARY_NAME: y_true.loc[row_ids].astype("int8").to_numpy(),
            "calibrated_bad_credit_risk_probability": probability_series.to_numpy(),
            "frozen_threshold": FROZEN_THRESHOLD,
            "final_prediction": predictions.to_numpy(),
        }
    )
    artifact["correct_prediction"] = (
        artifact["final_prediction"] == artifact[TARGET_BINARY_NAME]
    ).astype("int8")
    return artifact.loc[:, list(PREDICTION_COLUMNS)]


def validate_prediction_artifact(
    artifact: pd.DataFrame,
    development_ids: pd.Index,
    holdout_ids: pd.Index,
) -> None:
    """Validate one and only one prediction for every locked holdout row."""
    if tuple(artifact.columns) != PREDICTION_COLUMNS:
        raise RuntimeError("Final prediction schema changed.")
    if len(artifact) != 200 or artifact[ROW_ID_NAME].nunique() != 200:
        raise RuntimeError("Final prediction artifact must contain 200 unique rows.")
    if set(artifact[ROW_ID_NAME]) != set(holdout_ids):
        raise RuntimeError("Final predictions do not match locked holdout membership.")
    if not set(artifact[ROW_ID_NAME]).isdisjoint(development_ids):
        raise RuntimeError("Development identifiers entered holdout predictions.")
    if not artifact["calibrated_bad_credit_risk_probability"].between(0.0, 1.0).all():
        raise RuntimeError("Final probabilities are outside [0, 1].")
    if set(artifact["final_prediction"]) - {0, 1}:
        raise RuntimeError("Final predictions are outside {0, 1}.")
    if not artifact["frozen_threshold"].eq(FROZEN_THRESHOLD).all():
        raise RuntimeError("Final threshold is not uniformly 0.16.")


def create_pre_access_manifest(project_root: Path) -> tuple[Path, str]:
    """Write the immutable pre-access record after all software checks pass."""
    output_dir = project_root / "reports" / "stage7"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "final_holdout_predictions.csv"
    if prediction_path.exists():
        raise RuntimeError("Final holdout predictions already exist; rerun refused.")
    manifest_path = output_dir / "pre_access_manifest.json"
    if manifest_path.exists():
        raise RuntimeError("Pre-access manifest already exists; replacement refused.")

    environment = verify_environment()
    hashes = verify_locked_artifacts(project_root)
    membership = membership_summary(project_root)
    policy = validate_frozen_policy(project_root)
    source_paths = {
        "stage7_implementation": project_root / "src" / "creditscope" / "stage7.py",
        "stage7_tests": project_root / "tests" / "test_stage7.py",
    }
    manifest: dict[str, Any] = {
        "utc_timestamp": _utc_timestamp(),
        "stage": "Stage 7 pre-access gate",
        "environment": environment,
        "locked_hashes": hashes,
        "frozen_policy_sha256": LOCKED_ARTIFACTS["frozen_policy"][1],
        "fixed_threshold": FROZEN_THRESHOLD,
        "fixed_calibration_method": FROZEN_CALIBRATION_METHOD,
        "ensemble": FROZEN_ENSEMBLE,
        "calibrated_classifier_n_jobs": None,
        "predefined_metrics": list(FINAL_METRIC_NAMES),
        **membership,
        "source_code_hashes": {
            name: _sha256(path) for name, path in source_paths.items()
        },
        "frozen_policy_validated": policy["policy_status"],
        "holdout_predictions_generated": False,
        "holdout_features_or_targets_loaded_for_evaluation": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    digest = _sha256(manifest_path)
    (output_dir / "pre_access_manifest.sha256").write_text(
        f"{digest}  pre_access_manifest.json\n", encoding="utf-8"
    )
    return manifest_path, digest


def _write_access_log(path: Path, **updates: Any) -> dict[str, Any]:
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing.update(updates)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return existing


def _write_final_summary(
    path: Path,
    metrics: dict[str, float | int],
    development: dict[str, Any],
) -> None:
    comparison = pd.DataFrame(
        [
            {"evidence": "Stage 6 nested development OOF", **development},
            {"evidence": "Stage 7 final locked holdout", **metrics},
        ]
    )
    lines = [
        "# Stage 7 one-time final holdout evaluation",
        "",
        "The sealed 200-row holdout was accessed once after the Stage 6P policy freeze. No model, feature, preprocessing, calibration, or threshold change followed access.",
        "",
        "## Frozen policy",
        "",
        "- Fixed untuned Stage 4 XGBoost pipeline",
        "- Sigmoid `CalibratedClassifierCV`, exact persisted five-fold development splits, `ensemble=True`, `n_jobs=None`",
        "- Five fold-specific classifier/calibrator pairs; probabilities are averaged",
        "- Threshold `0.16`",
        "- Cost `5 * FN + FP`",
        "",
        "## Final metrics and development comparison",
        "",
        comparison.to_markdown(index=False, floatfmt=".6f"),
        "",
        "ROC-AUC and Average Precision describe discrimination; Brier score and log loss describe probability quality; threshold metrics and 5:1 cost describe the frozen decision policy.",
        "",
        "The 5:1 error cost is an educational/research modelling assumption, not a claim about real-world lending economics. Predictions are good-credit-risk or bad-credit-risk classes, not lending approvals or rejections.",
        "",
        "No confidence intervals, alternative thresholds, alternative models, SHAP analysis, or post-hoc model development were performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute_one_time_holdout_evaluation(
    project_root: Path,
    pre_access_manifest_sha256: str,
) -> dict[str, Any]:
    """Fit the frozen ensemble and score the sealed holdout exactly once."""
    output_dir = project_root / "reports" / "stage7"
    manifest_path = output_dir / "pre_access_manifest.json"
    prediction_path = output_dir / "final_holdout_predictions.csv"
    if prediction_path.exists() or (output_dir / "final_holdout_metrics.csv").exists():
        raise RuntimeError("Final holdout evaluation artifacts already exist; rerun refused.")
    if not manifest_path.exists() or _sha256(manifest_path) != pre_access_manifest_sha256:
        raise RuntimeError("Pre-access manifest is missing or its SHA-256 does not match.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["holdout_predictions_generated"] is not False:
        raise RuntimeError("Pre-access manifest does not certify a sealed holdout.")
    verify_environment()
    locked_hashes = verify_locked_artifacts(project_root)
    validate_frozen_policy(project_root)
    source_checks = {
        "stage7_implementation": _sha256(project_root / "src" / "creditscope" / "stage7.py"),
        "stage7_tests": _sha256(project_root / "tests" / "test_stage7.py"),
    }
    if source_checks != manifest["source_code_hashes"]:
        raise RuntimeError("Stage 7 source changed after the pre-access manifest.")

    access_log_path = output_dir / "holdout_access_log.json"
    _write_access_log(
        access_log_path,
        access_started_utc=_utc_timestamp(),
        holdout_source_opened=False,
        holdout_features_selected=False,
        holdout_targets_viewed=False,
        predictions_generated=False,
        status="starting irreversible evaluation",
    )
    try:
        # First source access occurs only after the immutable pre-access gate.
        data = build_analysis_dataset(load_verified_raw_snapshot(project_root))
        data.index = source_row_ids(data)
        manifest_data = pd.read_csv(
            project_root / "reports" / "split_manifest.csv"
        ).set_index(ROW_ID_NAME)
        _write_access_log(
            access_log_path,
            holdout_source_opened=True,
            source_opened_utc=_utc_timestamp(),
            status="source opened; holdout rows not selected",
        )

        development_ids = manifest_data.index[
            manifest_data["split"] == DEVELOPMENT_SPLIT
        ]
        X_development = data.loc[development_ids, list(PREDICTIVE_FEATURES)].copy()
        y_development = data.loc[development_ids, TARGET_BINARY_NAME].astype("int8").copy()
        _, persisted_folds = load_outer_folds(project_root, X_development.index)
        final_estimator = build_final_estimator(persisted_folds)
        final_estimator.fit(X_development, y_development)
        if len(final_estimator.calibrated_classifiers_) != 5:
            raise RuntimeError("Frozen calibrated ensemble did not create five members.")

        holdout_ids = manifest_data.index[manifest_data["split"] == TEST_SPLIT]
        X_holdout = data.loc[holdout_ids, list(PREDICTIVE_FEATURES)].copy()
        y_holdout = data.loc[holdout_ids, TARGET_BINARY_NAME].astype("int8").copy()
        _write_access_log(
            access_log_path,
            holdout_features_selected=True,
            holdout_targets_viewed=True,
            holdout_rows=200,
            holdout_selected_utc=_utc_timestamp(),
            status="holdout opened for one-time prediction and predefined metrics",
        )
        probabilities = final_estimator.predict_proba(X_holdout)[:, 1]
        artifact = make_prediction_artifact(holdout_ids, y_holdout, probabilities)
        validate_prediction_artifact(artifact, development_ids, holdout_ids)
        artifact.to_csv(prediction_path, index=False, lineterminator="\n")
        prediction_hash = _sha256(prediction_path)
        _write_access_log(
            access_log_path,
            predictions_generated=True,
            predictions_generated_utc=_utc_timestamp(),
            final_holdout_predictions_sha256=prediction_hash,
            status="predictions persisted; calculating predefined metrics only",
        )

        metric_record = final_metric_record(
            artifact[TARGET_BINARY_NAME],
            artifact["calibrated_bad_credit_risk_probability"],
            artifact["final_prediction"],
        )
        pd.DataFrame([metric_record]).to_csv(
            output_dir / "final_holdout_metrics.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame(
            [
                {
                    "actual_class": "good_credit_risk_0",
                    "predicted_good_credit_risk_0": metric_record["true_negative"],
                    "predicted_bad_credit_risk_1": metric_record["false_positive"],
                },
                {
                    "actual_class": "bad_credit_risk_1",
                    "predicted_good_credit_risk_0": metric_record["false_negative"],
                    "predicted_bad_credit_risk_1": metric_record["true_positive"],
                },
            ]
        ).to_csv(output_dir / "final_confusion_matrix.csv", index=False, lineterminator="\n")

        stage6 = json.loads(
            (project_root / "reports" / "stage6" / "run_summary.json").read_text(encoding="utf-8")
        )["honest_nested_policy_metrics"]
        comparison_metrics = list(FINAL_METRIC_NAMES)
        development_comparison = {key: stage6[key] for key in comparison_metrics}
        comparison = pd.DataFrame(
            [
                {"evidence": "Stage 6 nested development OOF", **development_comparison},
                {"evidence": "Stage 7 final locked holdout", **metric_record},
            ]
        )
        comparison.to_csv(
            output_dir / "development_holdout_comparison.csv",
            index=False,
            lineterminator="\n",
        )
        _write_final_summary(
            output_dir / "final_evaluation_summary.md",
            metric_record,
            development_comparison,
        )

        run_summary: dict[str, Any] = {
            "stage": "Stage 7 one-time final locked holdout evaluation",
            "completed_utc": _utc_timestamp(),
            "environment": environment_versions(),
            "locked_hashes": locked_hashes,
            "frozen_policy_sha256": LOCKED_ARTIFACTS["frozen_policy"][1],
            "pre_access_manifest_sha256": pre_access_manifest_sha256,
            "final_holdout_predictions_sha256": prediction_hash,
            "model_configuration": XGBOOST_PARAMETERS,
            "calibration_configuration": {
                "class": "sklearn.calibration.CalibratedClassifierCV",
                "method": "sigmoid",
                "cv": "exact persisted Stage 4 five-fold development splits",
                "ensemble": True,
                "n_jobs": None,
                "ensemble_members": 5,
                "aggregation": "arithmetic mean of calibrated probabilities",
            },
            "threshold": FROZEN_THRESHOLD,
            "predefined_metrics": metric_record,
            "confidence_intervals_calculated": False,
            "alternative_thresholds_evaluated": False,
            "alternative_models_evaluated": False,
            "post_holdout_model_changes": False,
            "shap_performed": False,
            "holdout_access_count": 1,
            "holdout_rows": 200,
        }
        (output_dir / "run_summary.json").write_text(
            json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
        )
        _write_access_log(
            access_log_path,
            evaluation_completed_utc=_utc_timestamp(),
            status="completed once; rerun prohibited",
        )
        return run_summary
    except Exception as error:
        _write_access_log(
            access_log_path,
            failure_utc=_utc_timestamp(),
            failure_type=type(error).__name__,
            failure_message=str(error),
            status="failed after access; automatic restart prohibited",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    parser.add_argument("--pre-access-manifest-sha256")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    if args.prepare:
        path, digest = create_pre_access_manifest(project_root)
        print(json.dumps({"path": str(path), "sha256": digest}, indent=2))
        return
    if not args.pre_access_manifest_sha256:
        parser.error("--evaluate requires --pre-access-manifest-sha256")
    print(
        json.dumps(
            execute_one_time_holdout_evaluation(
                project_root, args.pre_access_manifest_sha256
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
