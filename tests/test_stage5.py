"""Tests for controlled Stage 5 nested XGBoost tuning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from creditscope import stage5
from creditscope.evaluation import DEFAULT_THRESHOLD, asymmetric_cost, make_cv_splits
from creditscope.modeling import AUDIT_ATTRIBUTES, PREDICTIVE_FEATURES
from creditscope.schema import FEATURE_SPECS


def make_model_data(rows: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    values: dict[str, list[object]] = {}
    for offset, spec in enumerate(FEATURE_SPECS, start=1):
        if spec.name not in PREDICTIVE_FEATURES:
            continue
        if spec.categories:
            codes = [code for code, _ in spec.categories]
            values[spec.name] = [
                codes[index % len(codes)] for index in range(rows)
            ]
        else:
            values[spec.name] = [offset + index % 11 for index in range(rows)]
    X = pd.DataFrame(values)[list(PREDICTIVE_FEATURES)]
    X.index = pd.Index([f"dev-{index:03d}" for index in range(rows)])
    y = pd.Series(
        [0] * (rows * 7 // 10) + [1] * (rows - rows * 7 // 10),
        index=X.index,
        name="bad_credit_risk",
        dtype="int8",
    )
    return X, y


def test_search_space_and_selection_policy_are_locked() -> None:
    assert stage5.SEARCH_ITERATIONS == 40
    assert stage5.PRIMARY_SCORER == "average_precision"
    assert stage5.RANDOM_STATE == 42
    assert stage5.INNER_SPLITS == 4
    assert stage5.SEARCH_SPACE == {
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


def test_search_configuration_is_deterministic_and_unweighted() -> None:
    search = stage5.make_randomized_search(cv=stage5.make_inner_cv(), n_jobs=1)
    assert search.n_iter == 40
    assert search.random_state == 42
    assert search.refit == "average_precision"
    classifier = search.estimator.named_steps["classifier"]
    params = classifier.get_params()
    assert params["n_jobs"] == 1
    assert params.get("scale_pos_weight") in (None, 1)
    assert params.get("early_stopping_rounds") is None
    assert not set(AUDIT_ATTRIBUTES) & set(PREDICTIVE_FEATURES)


def test_persisted_outer_folds_equal_stage4_membership() -> None:
    root = Path(__file__).resolve().parents[1]
    stage4 = pd.read_csv(root / "reports" / "stage4" / "fold_membership.csv")
    membership, folds = stage5.load_outer_folds(
        root, pd.Index(stage4["source_row_id"])
    )
    assert membership.equals(stage4)
    assert len(folds) == 5
    validation = np.concatenate([fold_validation for _, fold_validation in folds])
    assert sorted(validation.tolist()) == list(range(800))
    manifest = pd.read_csv(root / "reports" / "split_manifest.csv")
    development_ids = set(
        manifest.loc[manifest["split"] == "development", "source_row_id"]
    )
    final_test_ids = set(
        manifest.loc[manifest["split"] == "test", "source_row_id"]
    )
    assert set(membership["source_row_id"]) == development_ids
    assert not set(membership["source_row_id"]) & final_test_ids


def test_inner_cv_never_contains_outer_validation_rows() -> None:
    _, y = make_model_data()
    outer_folds = make_cv_splits(pd.DataFrame(index=y.index), y)
    for outer_train, outer_validation in outer_folds:
        stage5.validate_inner_outer_isolation(outer_train, outer_validation, y)


def test_nested_oof_is_complete_bounded_and_deterministic(monkeypatch) -> None:
    X, y = make_model_data()
    outer_folds = make_cv_splits(X, y)
    original = stage5.make_randomized_search

    def tiny_search(*, cv, refit=stage5.PRIMARY_SCORER, n_jobs=-1):
        search = original(cv=cv, refit=refit, n_jobs=1)
        search.set_params(
            n_iter=1,
            param_distributions={
                name: [values[0]] for name, values in stage5.SEARCH_SPACE.items()
            },
        )
        return search

    monkeypatch.setattr(stage5, "make_randomized_search", tiny_search)
    first = stage5.run_nested_tuning(X, y, outer_folds)
    second = stage5.run_nested_tuning(X, y, outer_folds)
    assert first.probabilities.index.equals(y.index)
    assert first.probabilities.notna().all()
    assert first.probabilities.between(0, 1).all()
    assert first.predictions.isin([0, 1]).all()
    np.testing.assert_allclose(first.probabilities, second.probabilities)


def test_threshold_and_cost_orientation_remain_fixed() -> None:
    assert DEFAULT_THRESHOLD == 0.50
    y_true = pd.Series([1, 1, 0, 0], dtype="int8")
    y_pred = pd.Series([0, 1, 1, 0], dtype="int8")
    cost = asymmetric_cost(y_true, y_pred)
    assert cost["false_negative"] == 1
    assert cost["false_positive"] == 1
    assert cost["total_cost"] == 6


def test_full_development_search_does_not_refit_automatically() -> None:
    search = stage5.make_randomized_search(cv=stage5.make_inner_cv(), refit=False)
    assert search.refit is False
