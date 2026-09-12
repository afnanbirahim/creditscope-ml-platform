"""Reproducibly package the already-frozen CreditScope inference policy."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from creditscope.analysis import validate_analysis_dataset
from creditscope.inference import (
    CATEGORY_DOMAINS,
    FROZEN_THRESHOLD,
    MODEL_VERSION,
    file_sha256,
    load_frozen_model,
    predict_batch,
)
from creditscope.modeling import (
    AUDIT_ATTRIBUTES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PREDICTIVE_FEATURES,
    XGBOOST_PARAMETERS,
)
from creditscope.schema import (
    CATEGORY_LABELS,
    FEATURE_NAME_MAP,
    FEATURE_SPECS,
    TARGET_BINARY_MAP,
    TARGET_BINARY_NAME,
    TARGET_ORIGINAL_NAME,
)
from creditscope.splits import (
    DEVELOPMENT_SPLIT,
    ROW_ID_NAME,
    TEST_SPLIT,
)
from creditscope.stage5 import load_outer_folds
from creditscope.stage7 import build_final_estimator

LOCKED_BUILD_EVIDENCE = {
    "raw_data_sha256": (
        "data/raw/german_credit.csv",
        "4ce6007c9eb3dbcd67e7858773566b0eb3ff140033cf67b2d8a42b3a89edfb9d",
    ),
    "split_sha256": (
        "reports/split_manifest.csv",
        "32fb4c9ac2cdb358e3bffd145cd97e1a393c14946888d46212d0b395523e99af",
    ),
    "fold_membership_sha256": (
        "reports/stage4/fold_membership.csv",
        "513fc9249a4c00be3ba67fa99ea20a7ac76b653694130cf66733ea08f97d0b92",
    ),
    "frozen_policy_sha256": (
        "reports/stage6/frozen_model_policy.json",
        "afee2e4ea82d438fa0ba638a8b7bdadaecbce9a4ada7da1f72d66866e4313afc",
    ),
    "stage7_predictions_sha256": (
        "reports/stage7/final_holdout_predictions.csv",
        "dc43c8f81bfd993ec6735253be3fdc572ba44deda2743d4c7b9f8dbff41dce18",
    ),
    "stage7_metrics_sha256": (
        "reports/stage7/final_holdout_metrics.csv",
        "c676b0f0ea0a02a4bfe8b681cc29903831aef83631def31c38eeb599879b124d",
    ),
}


def verify_build_evidence(project_root: Path) -> dict[str, str]:
    """Check immutable evidence by hash without reading holdout predictions."""
    observed = {
        name: file_sha256(project_root / relative)
        for name, (relative, _) in LOCKED_BUILD_EVIDENCE.items()
    }
    expected = {name: expected for name, (_, expected) in LOCKED_BUILD_EVIDENCE.items()}
    if observed != expected:
        raise RuntimeError(f"Frozen build evidence mismatch: {observed}")
    return observed


def load_development_training_data(
    project_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Index,
    tuple[tuple[np.ndarray, np.ndarray], ...],
]:
    """Return exactly the locked 800 development rows and persisted folds."""
    split = pd.read_csv(project_root / "reports" / "split_manifest.csv").set_index(
        ROW_ID_NAME
    )
    development_ids = split.index[split["split"] == DEVELOPMENT_SPLIT]
    holdout_ids = split.index[split["split"] == TEST_SPLIT]
    if len(development_ids) != 800 or len(holdout_ids) != 200:
        raise RuntimeError("Locked split population changed.")
    if not set(development_ids).isdisjoint(holdout_ids):
        raise RuntimeError("Development and holdout memberships overlap.")
    development_positions = {
        int(str(row_id).rsplit("-", maxsplit=1)[1]) for row_id in development_ids
    }
    raw = pd.read_csv(
        project_root / "data" / "raw" / "german_credit.csv",
        skiprows=lambda line_number: (
            line_number != 0 and line_number not in development_positions
        ),
    )
    kept_ids = pd.Index(
        [f"uci144-row-{position:04d}" for position in sorted(development_positions)],
        name=ROW_ID_NAME,
    )
    if len(raw) != 800 or len(kept_ids) != 800:
        raise RuntimeError("Selective source load did not return 800 development rows.")
    data = raw[list(FEATURE_NAME_MAP)].rename(columns=FEATURE_NAME_MAP).copy()
    data[TARGET_ORIGINAL_NAME] = raw["class"].astype("int64").to_numpy()
    data[TARGET_BINARY_NAME] = (
        raw["class"].astype("int64").map(TARGET_BINARY_MAP).astype("int8").to_numpy()
    )
    validate_analysis_dataset(data, expected_rows=800)
    data.index = kept_ids
    data = data.loc[development_ids]
    X = data.loc[development_ids, list(PREDICTIVE_FEATURES)].copy()
    y = data.loc[development_ids, TARGET_BINARY_NAME].astype("int8").copy()
    if len(X) != 800 or X.index.tolist() != development_ids.tolist():
        raise RuntimeError("Build population is not the exact development membership.")
    if set(X.columns) & set(AUDIT_ATTRIBUTES):
        raise RuntimeError("Audit-only attributes entered the build feature matrix.")
    _, folds = load_outer_folds(project_root, X.index)
    return X, y, development_ids, folds


def validate_frozen_architecture(estimator: CalibratedClassifierCV) -> None:
    """Require the exact calibrated five-member Stage 6P architecture."""
    if estimator.method != "sigmoid" or estimator.ensemble is not True:
        raise RuntimeError("Calibration method or ensemble setting changed.")
    if estimator.n_jobs is not None:
        raise RuntimeError("CalibratedClassifierCV n_jobs must remain None.")
    if len(estimator.calibrated_classifiers_) != 5:
        raise RuntimeError("Production estimator must contain five calibrated members.")
    for calibrated in estimator.calibrated_classifiers_:
        classifier = calibrated.estimator.named_steps["classifier"]
        observed = {key: classifier.get_params()[key] for key in XGBOOST_PARAMETERS}
        if observed != XGBOOST_PARAMETERS:
            raise RuntimeError(
                "A fitted member differs from frozen XGBoost parameters."
            )


def _semantic_caveat(name: str) -> str:
    caveats = {
        "credit_amount": "Historical DM-denominated source quantity subjected to an undocumented monotonic transformation; not a literal contemporary monetary amount.",
        "employment_duration": "Discretized duration category; categorical encoding does not imply equal spacing.",
        "installment_rate_percent": "Discretized installment-rate category, not a literal percentage measurement.",
        "residence_duration": "Discretized residence-duration category; not a continuous duration.",
        "property": "Ordered source concept represented categorically; unknown/no property is substantive.",
        "existing_credits_count": "Discretized credit-count information, not an unrestricted count.",
        "job": "Ordinal source concept represented categorically; codes are not numerical magnitudes.",
        "dependents_count": "Binary/discretized quantity, not an unrestricted count.",
    }
    return caveats.get(
        name,
        "Use only with the documented historical UCI semantics; model association is not causal.",
    )


def create_input_schema() -> dict[str, Any]:
    """Create the strict raw-input schema for all 17 predictors."""
    specs = {spec.name: spec for spec in FEATURE_SPECS}
    fields: list[dict[str, Any]] = []
    for name in PREDICTIVE_FEATURES:
        spec = specs[name]
        field: dict[str, Any] = {
            "name": name,
            "type": "finite_number" if name in NUMERICAL_FEATURES else "categorical",
            "required": True,
            "semantic_description": spec.official_meaning,
            "source_field": spec.uci_name,
            "caveat": _semantic_caveat(name),
        }
        if name in NUMERICAL_FEATURES:
            field["constraints"] = ["parseable as numeric", "finite", "not NaN"]
            field["source_code_mapping"] = None
        else:
            field["accepted_categories"] = list(CATEGORY_DOMAINS[name])
            labels = CATEGORY_LABELS.get(name)
            field["source_code_mapping"] = (
                labels
                if labels is not None
                else {
                    str(value): "documented discretized source level"
                    for value in CATEGORY_DOMAINS[name]
                }
            )
        fields.append(field)
    return {
        "schema_version": "1.0.0",
        "model_version": MODEL_VERSION,
        "strict": True,
        "additional_fields_allowed": False,
        "audit_only_fields_rejected": list(AUDIT_ATTRIBUTES),
        "field_order": list(PREDICTIVE_FEATURES),
        "fields": fields,
    }


def _records_from_frame(X: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {name: row[name] for name in PREDICTIVE_FEATURES} for _, row in X.iterrows()
    ]


def select_golden_fixtures(
    estimator: CalibratedClassifierCV, X: pd.DataFrame
) -> list[dict[str, Any]]:
    """Select five development fixtures through a fixed probability/diversity rule."""
    probabilities = pd.Series(estimator.predict_proba(X)[:, 1], index=X.index)
    candidates = pd.DataFrame(
        {
            "probability": probabilities.to_numpy(),
            ROW_ID_NAME: X.index.astype(str).to_numpy(),
        }
    )
    ordered_rules = [
        ("lowest_probability", candidates.sort_values(["probability", ROW_ID_NAME])),
        (
            "closest_to_threshold_0_16",
            candidates.assign(
                distance=np.abs(probabilities.to_numpy() - FROZEN_THRESHOLD)
            ).sort_values(["distance", ROW_ID_NAME]),
        ),
        (
            "closest_to_moderate_reference_0_35",
            candidates.assign(
                distance=np.abs(probabilities.to_numpy() - 0.35)
            ).sort_values(["distance", ROW_ID_NAME]),
        ),
        (
            "highest_probability",
            candidates.sort_values(
                ["probability", ROW_ID_NAME], ascending=[False, True]
            ),
        ),
    ]
    selected: list[str] = []
    fixture_rules: list[str] = []
    for label, ordered in ordered_rules:
        row_id = str(ordered[~ordered[ROW_ID_NAME].isin(selected)].iloc[0][ROW_ID_NAME])
        selected.append(row_id)
        fixture_rules.append(label)
    remaining = X.loc[~X.index.isin(selected)]
    selected_categories = X.loc[selected, list(CATEGORICAL_FEATURES)]
    diversity = remaining[list(CATEGORICAL_FEATURES)].apply(
        lambda row: min(
            (row != chosen).sum() for _, chosen in selected_categories.iterrows()
        ),
        axis=1,
    )
    max_diversity = diversity.max()
    diverse_id = min(diversity[diversity == max_diversity].index.astype(str))
    selected.append(diverse_id)
    fixture_rules.append("maximum_minimum_categorical_hamming_distance")

    records = _records_from_frame(X.loc[selected])
    probabilities = estimator.predict_proba(X.loc[selected])[:, 1]
    fixtures: list[dict[str, Any]] = []
    for rule, row_id, record, probability in zip(
        fixture_rules, selected, records, probabilities, strict=True
    ):
        predicted_class = int(float(probability) >= FROZEN_THRESHOLD)
        fixtures.append(
            {
                "selection_rule": rule,
                ROW_ID_NAME: row_id,
                "input": record,
                "expected": {
                    "probability_bad_risk": float(probability),
                    "predicted_class": predicted_class,
                    "predicted_label": (
                        "bad_credit_risk" if predicted_class else "good_credit_risk"
                    ),
                    "threshold": FROZEN_THRESHOLD,
                    "model_version": MODEL_VERSION,
                },
            }
        )
    return fixtures


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "pandas": version("pandas"),
        "scikit_learn": version("scikit-learn"),
        "xgboost": version("xgboost"),
        "joblib": version("joblib"),
    }


def build_frozen_model_artifacts(
    project_root: Path,
    *,
    output_dir: Path | None = None,
    examples_dir: Path | None = None,
) -> dict[str, Any]:
    """Fit, serialize, document, and round-trip the frozen policy."""
    hashes = verify_build_evidence(project_root)
    frozen_policy = json.loads(
        (project_root / "reports/stage6/frozen_model_policy.json").read_text(
            encoding="utf-8"
        )
    )
    if frozen_policy["model"]["hyperparameters"] != XGBOOST_PARAMETERS:
        raise RuntimeError("Frozen policy XGBoost parameters changed.")
    if frozen_policy["decision_threshold"] != FROZEN_THRESHOLD:
        raise RuntimeError("Frozen policy threshold changed.")
    if frozen_policy["feature_governance"]["predictive_features"] != list(
        PREDICTIVE_FEATURES
    ):
        raise RuntimeError("Frozen policy predictive features changed.")

    X, y, development_ids, folds = load_development_training_data(project_root)
    estimator = build_final_estimator(folds).fit(X, y)
    validate_frozen_architecture(estimator)
    fixtures = select_golden_fixtures(estimator, X)
    before = estimator.predict_proba(
        pd.concat(
            [pd.DataFrame([fixture["input"]]) for fixture in fixtures],
            ignore_index=True,
        )[list(PREDICTIVE_FEATURES)]
    )[:, 1]

    destination = output_dir or project_root / "artifacts" / "model"
    destination.mkdir(parents=True, exist_ok=True)
    artifact_path = destination / "creditscope_frozen_model.joblib"
    manifest_path = destination / "model_manifest.json"
    schema_path = destination / "input_schema.json"
    fixtures_path = destination / "golden_inference_fixtures.json"
    joblib.dump(estimator, artifact_path, compress=3)
    artifact_hash = file_sha256(artifact_path)
    schema_path.write_text(
        json.dumps(create_input_schema(), indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    fixtures_path.write_text(
        json.dumps(fixtures, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )

    manifest: dict[str, Any] = {
        "project_name": "CreditScope",
        "artifact_type": "trusted joblib-serialized calibrated classifier",
        "model_version": MODEL_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "artifact_filename": artifact_path.name,
        "artifact_sha256": artifact_hash,
        **hashes,
        "environment": _environment(),
        "model_family": "fixed untuned XGBoost",
        "model_hyperparameters": XGBOOST_PARAMETERS,
        "predictor_count": len(PREDICTIVE_FEATURES),
        "predictor_names": list(PREDICTIVE_FEATURES),
        "audit_only_exclusions": list(AUDIT_ATTRIBUTES),
        "target_semantics": {"0": "good credit risk", "1": "bad credit risk"},
        "positive_class": 1,
        "calibration": {
            "class": "sklearn.calibration.CalibratedClassifierCV",
            "method": "sigmoid",
            "ensemble": True,
            "cv": "exact persisted Stage 4 five-fold development splits",
            "n_jobs": None,
            "calibrated_members": 5,
            "aggregation": "arithmetic mean of calibrated member probabilities",
        },
        "decision_threshold": FROZEN_THRESHOLD,
        "comparison_operator": ">=",
        "decision_rule": "bad credit risk if probability >= 0.16; good credit risk otherwise",
        "cost_assumption": "5 * false_negative + false_positive",
        "training_population": {
            "split": "development",
            "rows": len(development_ids),
            "good_credit_risk": int((y == 0).sum()),
            "bad_credit_risk": int((y == 1).sum()),
            "holdout_rows_used": 0,
        },
        "dataset_provenance": "UCI Statlog German Credit Data ID 144; South German Credit ID 573 supplies semantic corrections",
        "intended_use": "Educational/research credit-risk decision-support prototype and portfolio demonstration",
        "prohibited_use": "Automated lending approval, rejection, eligibility, pricing, or real credit decisions",
        "serialization_security": "Joblib/pickle is unsafe for untrusted artifacts; load only this internally generated artifact after hash verification.",
        "binary_reproducibility": "The joblib hash is version-sensitive and is not guaranteed across Python/library rebuilds.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    loaded = load_frozen_model(artifact_path, manifest_path, strict_integrity=True)
    after_results = predict_batch(loaded, [fixture["input"] for fixture in fixtures])
    after = np.array([result["probability_bad_risk"] for result in after_results])
    if not np.allclose(before, after, rtol=0, atol=1e-12):
        raise RuntimeError("Serialization round-trip probability check failed.")
    if not np.array_equal(
        (before >= FROZEN_THRESHOLD).astype(int),
        np.array([result["predicted_class"] for result in after_results]),
    ):
        raise RuntimeError("Serialization round-trip classifications changed.")
    manifest["round_trip_validation"] = {
        "fixture_count": len(fixtures),
        "probability_atol": 1e-12,
        "maximum_absolute_probability_difference": float(
            np.max(np.abs(before - after))
        ),
        "predictions_exact": True,
        "labels_exact": True,
        "five_members_preserved": len(loaded.estimator.calibrated_classifiers_) == 5,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    checksums = {
        artifact_path.name: file_sha256(artifact_path),
        manifest_path.name: file_sha256(manifest_path),
        schema_path.name: file_sha256(schema_path),
        fixtures_path.name: file_sha256(fixtures_path),
    }
    (destination / "artifact_checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
        encoding="utf-8",
    )

    if examples_dir is not None:
        examples_dir.mkdir(parents=True, exist_ok=True)
        example = next(
            fixture
            for fixture in fixtures
            if fixture["selection_rule"] == "closest_to_moderate_reference_0_35"
        )
        (examples_dir / "sample_prediction_input.json").write_text(
            json.dumps(example["input"], indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        sample_output = dict(example["expected"])
        sample_output["meaning"] = (
            "Model-estimated calibrated bad-credit-risk probability and decision-support "
            "classification under the frozen threshold; not a lending decision."
        )
        (examples_dir / "sample_prediction_output.json").write_text(
            json.dumps(sample_output, indent=2) + "\n", encoding="utf-8"
        )

    return {
        "model_version": MODEL_VERSION,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_hash,
        "manifest_path": str(manifest_path),
        "input_schema_path": str(schema_path),
        "training_rows": len(development_ids),
        "holdout_rows_used": 0,
        "calibrated_members": 5,
        "round_trip": manifest["round_trip_validation"],
        "checksums": checksums,
    }


def main() -> None:
    """Build the canonical production artifact from the project root."""
    parser = argparse.ArgumentParser(description="Build frozen CreditScope model.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    result = build_frozen_model_artifacts(
        root,
        output_dir=args.output_dir,
        examples_dir=root / "examples" if args.output_dir is None else None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
