"""Governed feature separation and baseline preprocessing/model pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from creditscope.schema import TARGET_BINARY_NAME
from creditscope.splits import (
    DEVELOPMENT_SPLIT,
    ROW_ID_NAME,
    TEST_SPLIT,
    source_row_ids,
)

AUDIT_ATTRIBUTES = (
    "personal_status_sex",
    "age_years",
    "foreign_worker",
)

NUMERICAL_FEATURES = (
    "duration_months",
    "credit_amount",
)

CATEGORICAL_FEATURES = (
    "checking_account_status",
    "credit_history",
    "purpose",
    "savings_account_status",
    "employment_duration",
    "installment_rate_percent",
    "other_debtors_guarantors",
    "residence_duration",
    "property",
    "other_installment_plans",
    "housing",
    "existing_credits_count",
    "job",
    "dependents_count",
    "telephone",
)

PREDICTIVE_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

RANDOM_FOREST_PARAMETERS = {
    "n_estimators": 500,
    "random_state": 42,
    "n_jobs": -1,
}

XGBOOST_PARAMETERS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


@dataclass(frozen=True)
class ModelPartitions:
    """Development/test partitions with separate predictive and audit views."""

    X_development: pd.DataFrame
    y_development: pd.Series
    audit_development: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series
    audit_test: pd.DataFrame


def validate_feature_specification() -> None:
    """Protect the model boundary from sensitive or duplicate columns."""
    groups = [set(NUMERICAL_FEATURES), set(CATEGORICAL_FEATURES)]
    if any(
        left & right
        for index, left in enumerate(groups)
        for right in groups[index + 1 :]
    ):
        raise ValueError("Feature-type groups overlap.")
    if set(PREDICTIVE_FEATURES) & set(AUDIT_ATTRIBUTES):
        raise ValueError("Direct sensitive attributes entered predictive features.")
    if len(PREDICTIVE_FEATURES) != 17:
        raise ValueError("Expected 17 predictive features after governance exclusion.")


def partition_model_data(data: pd.DataFrame, manifest: pd.DataFrame) -> ModelPartitions:
    """Align predictors, target, and audit attributes to locked memberships."""
    validate_feature_specification()
    indexed = data.copy(deep=False)
    indexed.index = source_row_ids(data)
    membership = manifest.set_index(ROW_ID_NAME)["split"]
    if set(indexed.index) != set(membership.index):
        raise ValueError("Manifest identifiers do not align with the analysis dataset.")

    development_ids = membership[membership == DEVELOPMENT_SPLIT].index
    test_ids = membership[membership == TEST_SPLIT].index

    def feature_view(ids: pd.Index) -> pd.DataFrame:
        frame = indexed.loc[ids, list(PREDICTIVE_FEATURES)].copy()
        if set(AUDIT_ATTRIBUTES) & set(frame.columns):
            raise RuntimeError("Audit attributes leaked into predictive features.")
        return frame

    def target_view(ids: pd.Index) -> pd.Series:
        return indexed.loc[ids, TARGET_BINARY_NAME].astype("int8").copy()

    def audit_view(ids: pd.Index) -> pd.DataFrame:
        return indexed.loc[ids, list(AUDIT_ATTRIBUTES)].copy()

    return ModelPartitions(
        X_development=feature_view(development_ids),
        y_development=target_view(development_ids),
        audit_development=audit_view(development_ids),
        X_test=feature_view(test_ids),
        y_test=target_view(test_ids),
        audit_test=audit_view(test_ids),
    )


def build_preprocessor() -> ColumnTransformer:
    """Build deterministic preprocessing; fit only through a model pipeline."""
    validate_feature_specification()
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, list(NUMERICAL_FEATURES)),
            (
                "categorical",
                categorical_pipeline,
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_tree_preprocessor() -> ColumnTransformer:
    """Build tree-appropriate preprocessing without numeric scaling."""
    validate_feature_specification()
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, list(NUMERICAL_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_dummy_pipeline() -> Pipeline:
    """Build a non-informative prior-probability classifier."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", DummyClassifier(strategy="prior", random_state=42)),
        ]
    )


def build_logistic_pipeline() -> Pipeline:
    """Build the single untuned, unweighted Logistic Regression baseline."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    max_iter=2_000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """Build the single untuned Random Forest Stage 4 candidate."""
    return Pipeline(
        [
            ("preprocessor", build_tree_preprocessor()),
            ("classifier", RandomForestClassifier(**RANDOM_FOREST_PARAMETERS)),
        ]
    )


def build_xgboost_pipeline() -> Pipeline:
    """Build the single untuned XGBoost Stage 4 candidate."""
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise ImportError(
            "Stage 4 requires the optional project dependency 'xgboost'."
        ) from error
    return Pipeline(
        [
            ("preprocessor", build_tree_preprocessor()),
            ("classifier", XGBClassifier(**XGBOOST_PARAMETERS)),
        ]
    )
