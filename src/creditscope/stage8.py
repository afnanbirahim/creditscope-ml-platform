"""Stage 8 post-hoc explainability and responsible-AI audit.

This module reconstructs the already-frozen five-member calibrated ensemble on
development data solely for explanation. It never reads final-holdout rows or
changes the evaluated policy.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import brier_score_loss

from creditscope.analysis import build_analysis_dataset
from creditscope.data import load_verified_raw_snapshot
from creditscope.modeling import (
    AUDIT_ATTRIBUTES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PREDICTIVE_FEATURES,
)
from creditscope.schema import CATEGORY_LABELS, FEATURE_SPECS, TARGET_BINARY_NAME
from creditscope.splits import DEVELOPMENT_SPLIT, ROW_ID_NAME, source_row_ids
from creditscope.stage5 import load_outer_folds
from creditscope.stage7 import build_final_estimator

FROZEN_THRESHOLD = 0.16
EXPECTED_HASHES = {
    "raw_snapshot": (
        "data/raw/german_credit.csv",
        "4ce6007c9eb3dbcd67e7858773566b0eb3ff140033cf67b2d8a42b3a89edfb9d",
    ),
    "split": (
        "reports/split_manifest.csv",
        "32fb4c9ac2cdb358e3bffd145cd97e1a393c14946888d46212d0b395523e99af",
    ),
    "stage4_folds": (
        "reports/stage4/fold_membership.csv",
        "513fc9249a4c00be3ba67fa99ea20a7ac76b653694130cf66733ea08f97d0b92",
    ),
    "stage4_oof": (
        "reports/stage4/development_oof_predictions.csv",
        "c1e96b552ca7f5510f8aa9eda88db128e9cfbdd8ef30ca8535fac69646c486f9",
    ),
    "stage5_oof": (
        "reports/stage5/nested_oof_predictions.csv",
        "eecccdb9ae24d747562b5914c5f54bc5838688df4019b013552d1b87217460f5",
    ),
    "stage6_oof": (
        "reports/stage6/nested_policy_oof_predictions.csv",
        "a11703e95c1fd17c05de3c2636aba4bab74bdd3d93a992140cefd0b58bbda621",
    ),
    "stage6_threshold": (
        "reports/stage6/threshold_search.csv",
        "4f738292925c51242b34fad89d8a9c763c19014a3de3afea31f3968c3e7ac1a9",
    ),
    "frozen_policy": (
        "reports/stage6/frozen_model_policy.json",
        "afee2e4ea82d438fa0ba638a8b7bdadaecbce9a4ada7da1f72d66866e4313afc",
    ),
    "holdout_predictions": (
        "reports/stage7/final_holdout_predictions.csv",
        "dc43c8f81bfd993ec6735253be3fdc572ba44deda2743d4c7b9f8dbff41dce18",
    ),
    "final_metrics": (
        "reports/stage7/final_holdout_metrics.csv",
        "c676b0f0ea0a02a4bfe8b681cc29903831aef83631def31c38eeb599879b124d",
    ),
}

AGE_BINS = (-np.inf, 24, 34, 44, 54, np.inf)
AGE_LABELS = ("under 25", "25–34", "35–44", "45–54", "55+")
PROXY_FEATURES = (
    "checking_account_status",
    "savings_account_status",
    "employment_duration",
    "housing",
    "property",
    "telephone",
    "job",
)


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_locked_evidence(project_root: Path) -> dict[str, str]:
    """Require all frozen Stage 6P/7 evidence to remain byte-identical."""
    observed = {
        name: sha256(project_root / relative)
        for name, (relative, _) in EXPECTED_HASHES.items()
    }
    expected = {name: expected for name, (_, expected) in EXPECTED_HASHES.items()}
    if observed != expected:
        raise RuntimeError(f"Locked evidence changed: {observed}")
    return observed


def original_feature_name(encoded_name: str) -> str:
    """Map a ColumnTransformer output name to its governed source feature."""
    suffix = encoded_name.split("__", 1)[-1]
    if suffix in NUMERICAL_FEATURES:
        return suffix
    matches = [name for name in CATEGORICAL_FEATURES if suffix.startswith(f"{name}_")]
    if len(matches) != 1:
        raise ValueError(f"Cannot map encoded feature {encoded_name!r}.")
    return matches[0]


def extract_member_shap(
    fitted_ensemble: Any, X: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, Any]]]:
    """Explain each actual frozen-ensemble tree member on raw-margin scale."""
    encoded_records: list[dict[str, Any]] = []
    original_records: list[dict[str, Any]] = []
    member_payloads: dict[int, dict[str, Any]] = {}
    members = fitted_ensemble.calibrated_classifiers_
    if len(members) != 5:
        raise RuntimeError("Frozen explanation requires exactly five ensemble members.")
    for member_id, calibrated in enumerate(members, start=1):
        pipeline = calibrated.estimator
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["classifier"]
        transformed = preprocessor.transform(X)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=float)
        names = preprocessor.get_feature_names_out().astype(str)
        explainer = shap.TreeExplainer(model, model_output="raw")
        values = np.asarray(explainer.shap_values(transformed), dtype=float)
        if values.shape != transformed.shape or not np.isfinite(values).all():
            raise RuntimeError("SHAP output is incomplete or non-finite.")
        originals = np.array([original_feature_name(name) for name in names])
        for column, name in enumerate(names):
            encoded_records.append(
                {
                    "member": member_id,
                    "encoded_feature": name,
                    "original_feature": originals[column],
                    "mean_absolute_shap": float(np.abs(values[:, column]).mean()),
                    "signed_mean_shap": float(values[:, column].mean()),
                    "median_absolute_shap": float(np.median(np.abs(values[:, column]))),
                    "sd_shap": float(values[:, column].std(ddof=1)),
                }
            )
        for original in PREDICTIVE_FEATURES:
            columns = np.flatnonzero(originals == original)
            original_records.append(
                {
                    "member": member_id,
                    "original_feature": original,
                    "mean_absolute_shap": float(
                        np.abs(values[:, columns]).sum(axis=1).mean()
                    ),
                    "signed_mean_shap": float(values[:, columns].sum(axis=1).mean()),
                    "encoded_level_count": len(columns),
                }
            )
        encoded_total = sum(
            row["mean_absolute_shap"]
            for row in encoded_records
            if row["member"] == member_id
        )
        original_total = sum(
            row["mean_absolute_shap"]
            for row in original_records
            if row["member"] == member_id
        )
        if not np.isclose(encoded_total, original_total, rtol=0, atol=1e-10):
            raise RuntimeError(
                "Original-feature absolute SHAP aggregation does not reconcile."
            )
        member_payloads[member_id] = {
            "names": names,
            "originals": originals,
            "values": values,
            "data": transformed,
            "expected_value": float(
                np.asarray(explainer.expected_value).reshape(-1)[0]
            ),
            "calibrator_a": float(calibrated.calibrators[0].a_),
            "calibrator_b": float(calibrated.calibrators[0].b_),
        }
    return (
        pd.DataFrame(encoded_records),
        pd.DataFrame(original_records),
        member_payloads,
    )


def aggregate_encoded_importance(member_table: pd.DataFrame) -> pd.DataFrame:
    """Align encoded names and summarize only members in which a level exists."""
    result = member_table.groupby(
        ["encoded_feature", "original_feature"], as_index=False
    ).agg(
        members_present=("member", "nunique"),
        mean_absolute_shap=("mean_absolute_shap", "mean"),
        sd_member_mean_absolute_shap=("mean_absolute_shap", "std"),
        signed_mean_shap=("signed_mean_shap", "mean"),
        median_absolute_shap=("median_absolute_shap", "mean"),
    )
    result["members_absent"] = 5 - result["members_present"]
    return result.sort_values("mean_absolute_shap", ascending=False).reset_index(
        drop=True
    )


def aggregate_original_importance(member_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize member-level original-feature absolute SHAP contributions."""
    result = (
        member_table.groupby("original_feature", as_index=False)
        .agg(
            mean_absolute_shap=("mean_absolute_shap", "mean"),
            sd_member_mean_absolute_shap=("mean_absolute_shap", "std"),
            min_member_mean_absolute_shap=("mean_absolute_shap", "min"),
            max_member_mean_absolute_shap=("mean_absolute_shap", "max"),
            signed_mean_shap=("signed_mean_shap", "mean"),
        )
        .sort_values("mean_absolute_shap", ascending=False)
        .reset_index(drop=True)
    )
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def importance_stability(member_table: pd.DataFrame) -> pd.DataFrame:
    """Describe original-feature rank stability across five members."""
    ranked = member_table.copy()
    ranked["rank"] = ranked.groupby("member")["mean_absolute_shap"].rank(
        ascending=False, method="min"
    )
    return (
        ranked.groupby("original_feature", as_index=False)
        .agg(
            mean_rank=("rank", "mean"),
            sd_rank=("rank", "std"),
            minimum_rank=("rank", "min"),
            maximum_rank=("rank", "max"),
        )
        .sort_values("mean_rank")
        .reset_index(drop=True)
    )


def subgroup_metrics(
    audit: pd.DataFrame, y: pd.Series, probabilities: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate descriptive development-OOF audit metrics at frozen threshold."""
    prediction = (probabilities >= FROZEN_THRESHOLD).astype("int8")
    personal_labels = {
        code: f"{code} — {meaning}"
        for code, meaning in CATEGORY_LABELS["personal_status_sex"].items()
    }
    foreign_labels = {
        code: f"{code} — {meaning}"
        for code, meaning in CATEGORY_LABELS["foreign_worker"].items()
    }
    groups = {
        "personal_status_sex": (
            audit["personal_status_sex"].map(personal_labels),
            tuple(personal_labels.values()),
        ),
        "age_band": (
            pd.cut(audit["age_years"], bins=AGE_BINS, labels=AGE_LABELS, right=True),
            AGE_LABELS,
        ),
        "foreign_worker": (
            audit["foreign_worker"].map(foreign_labels),
            tuple(foreign_labels.values()),
        ),
    }
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for attribute, (labels, expected_groups) in groups.items():
        for group in expected_groups:
            mask = labels == group
            yt, yp, pr = y[mask], prediction[mask], probabilities[mask]
            if not mask.any():
                rows.append(
                    {
                        "audit_attribute": attribute,
                        "group": str(group),
                        "n": 0,
                        "bad_risk_n": 0,
                        "good_risk_n": 0,
                        "bad_risk_prevalence": np.nan,
                        "mean_predicted_bad_risk_probability": np.nan,
                        "predicted_positive_rate_at_0_16": np.nan,
                        "precision": np.nan,
                        "recall_sensitivity": np.nan,
                        "specificity": np.nan,
                        "false_positive_rate": np.nan,
                        "false_negative_rate": np.nan,
                        "balanced_accuracy": np.nan,
                        "brier_score": np.nan,
                    }
                )
                samples.append(
                    {"audit_attribute": attribute, "group": str(group), "n": 0}
                )
                continue
            n_pos, n_neg = int(yt.sum()), int((1 - yt).sum())
            tp = int(((yt == 1) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())
            tn = int(((yt == 0) & (yp == 0)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            precision = float(tp / (tp + fp)) if tp + fp else np.nan
            recall = float(tp / n_pos) if n_pos else np.nan
            specificity = float(tn / n_neg) if n_neg else np.nan
            rows.append(
                {
                    "audit_attribute": attribute,
                    "group": str(group),
                    "n": int(mask.sum()),
                    "bad_risk_n": n_pos,
                    "good_risk_n": n_neg,
                    "bad_risk_prevalence": float(yt.mean()),
                    "mean_predicted_bad_risk_probability": float(pr.mean()),
                    "predicted_positive_rate_at_0_16": float(yp.mean()),
                    "precision": precision,
                    "recall_sensitivity": recall,
                    "specificity": specificity,
                    "false_positive_rate": float(fp / n_neg) if n_neg else np.nan,
                    "false_negative_rate": float(fn / n_pos) if n_pos else np.nan,
                    "balanced_accuracy": float((recall + specificity) / 2)
                    if not (np.isnan(recall) or np.isnan(specificity))
                    else np.nan,
                    "brier_score": float(brier_score_loss(yt, pr)),
                }
            )
            samples.append(
                {
                    "audit_attribute": attribute,
                    "group": str(group),
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(samples)


def cramer_v(left: pd.Series, right: pd.Series) -> float:
    """Bias-unadjusted Cramer's V for transparent descriptive association."""
    table = pd.crosstab(left, right)
    if min(table.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(table, correction=False)[0]
    return float(np.sqrt((chi2 / table.to_numpy().sum()) / (min(table.shape) - 1)))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Return eta for categorical groups against a numeric variable."""
    grand = float(values.mean())
    numerator = sum(
        len(values[categories == group])
        * (float(values[categories == group].mean()) - grand) ** 2
        for group in categories.unique()
    )
    denominator = float(((values - grand) ** 2).sum())
    return float(np.sqrt(numerator / denominator)) if denominator else np.nan


def audit_feature_associations(
    audit: pd.DataFrame, X: pd.DataFrame, leading_features: Sequence[str]
) -> pd.DataFrame:
    """Assess a predefined limited set of leading frozen-model features."""
    records: list[dict[str, Any]] = []
    for audit_name in AUDIT_ATTRIBUTES:
        for feature in leading_features:
            if audit_name == "age_years" and feature in NUMERICAL_FEATURES:
                value = float(spearmanr(audit[audit_name], X[feature]).statistic)
                method = "spearman_rho"
            elif audit_name == "age_years":
                value = correlation_ratio(X[feature].astype(str), audit[audit_name])
                method = "correlation_ratio_eta"
            elif feature in NUMERICAL_FEATURES:
                value = correlation_ratio(audit[audit_name].astype(str), X[feature])
                method = "correlation_ratio_eta"
            else:
                value = cramer_v(audit[audit_name].astype(str), X[feature].astype(str))
                method = "cramers_v"
            records.append(
                {
                    "audit_attribute": audit_name,
                    "model_feature": feature,
                    "association_method": method,
                    "association_value": value,
                }
            )
    return pd.DataFrame(records)


def select_local_cases(oof: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen, probability-only deterministic case-selection rule."""
    frame = oof.copy()
    frame["prediction_at_0_16"] = (frame["probability"] >= FROZEN_THRESHOLD).astype(
        "int8"
    )
    frame["outcome_type"] = np.select(
        [
            (frame[TARGET_BINARY_NAME] == 1) & (frame["prediction_at_0_16"] == 1),
            (frame[TARGET_BINARY_NAME] == 0) & (frame["prediction_at_0_16"] == 0),
            (frame[TARGET_BINARY_NAME] == 0) & (frame["prediction_at_0_16"] == 1),
        ],
        ["true_positive", "true_negative", "false_positive"],
        default="false_negative",
    )
    rules = [
        ("low_probability", frame.sort_values(["probability", ROW_ID_NAME])),
        (
            "closest_to_threshold",
            frame.assign(
                distance=(frame["probability"] - FROZEN_THRESHOLD).abs()
            ).sort_values(["distance", ROW_ID_NAME]),
        ),
        (
            "high_probability",
            frame.sort_values(["probability", ROW_ID_NAME], ascending=[False, True]),
        ),
        (
            "false_positive",
            frame[frame["outcome_type"] == "false_positive"].sort_values(
                ["probability", ROW_ID_NAME], ascending=[False, True]
            ),
        ),
        (
            "false_negative",
            frame[frame["outcome_type"] == "false_negative"].sort_values(
                ["probability", ROW_ID_NAME]
            ),
        ),
    ]
    selected: list[pd.Series] = []
    used: set[str] = set()
    for label, candidates in rules:
        available = candidates[~candidates[ROW_ID_NAME].isin(used)]
        if available.empty:
            continue
        row = available.iloc[0].copy()
        row["selection_rule"] = label
        used.add(str(row[ROW_ID_NAME]))
        selected.append(row)
    return pd.DataFrame(selected)


def local_explanation_table(
    cases: pd.DataFrame,
    member_payloads: dict[int, dict[str, Any]],
    development_index: pd.Index,
) -> pd.DataFrame:
    """Add ensemble-mean underlying-tree contributors for selected cases."""
    positions = {row_id: position for position, row_id in enumerate(development_index)}
    output: list[dict[str, Any]] = []
    for _, case in cases.iterrows():
        position = positions[case[ROW_ID_NAME]]
        contributions: dict[str, list[float]] = {
            name: [] for name in PREDICTIVE_FEATURES
        }
        for payload in member_payloads.values():
            row_values = payload["values"][position]
            for original in PREDICTIVE_FEATURES:
                columns = np.flatnonzero(payload["originals"] == original)
                contributions[original].append(float(row_values[columns].sum()))
        ranked = sorted(
            ((name, float(np.mean(values))) for name, values in contributions.items()),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5]
        output.append(
            {
                "selection_rule": case["selection_rule"],
                ROW_ID_NAME: case[ROW_ID_NAME],
                TARGET_BINARY_NAME: int(case[TARGET_BINARY_NAME]),
                "stage6_oof_probability": float(case["probability"]),
                "prediction_at_frozen_threshold_0_16": int(case["prediction_at_0_16"]),
                "outcome_type": case["outcome_type"],
                "top_underlying_tree_score_contributors": "; ".join(
                    f"{name}={value:+.6f}" for name, value in ranked
                ),
            }
        )
    return pd.DataFrame(output)


def proxy_register(original_importance: pd.DataFrame) -> pd.DataFrame:
    """Create a qualitative, non-removal proxy-risk register."""
    risks = {
        "checking_account_status": "high proxy potential",
        "savings_account_status": "high proxy potential",
        "employment_duration": "high proxy potential",
        "housing": "high proxy potential",
        "property": "high proxy potential",
        "telephone": "contextual/moderate proxy potential",
        "job": "high proxy potential",
    }
    rationale = {
        "checking_account_status": "May encode financial access, liquidity, and socioeconomic position.",
        "savings_account_status": "Directly reflects accumulated financial resources or account access.",
        "employment_duration": "May reflect labour-market stability and socioeconomic circumstances.",
        "housing": "Tenure status may reflect wealth, stability, and access to housing markets.",
        "property": "Asset ownership is closely connected to socioeconomic resources.",
        "telephone": "Historical telephone registration may proxy access, stability, or wealth.",
        "job": "Occupation grouping can encode socioeconomic position and structural inequality.",
    }
    importance = original_importance.set_index("original_feature")
    return pd.DataFrame(
        [
            {
                "feature": feature,
                "semantic_description": next(
                    spec.official_meaning
                    for spec in FEATURE_SPECS
                    if spec.name == feature
                ),
                "proxy_risk_category": risks[feature],
                "rationale": rationale[feature],
                "mean_absolute_shap": float(
                    importance.loc[feature, "mean_absolute_shap"]
                ),
                "importance_rank": int(importance.loc[feature, "rank"]),
                "governance_concern": "Potential proxy remains despite direct demographic exclusions.",
                "recommended_monitoring": "Monitor subgroup performance and data meaning; do not infer fairness from exclusion alone.",
            }
            for feature in PROXY_FEATURES
        ]
    )


def error_analysis(
    X: pd.DataFrame, y: pd.Series, probabilities: pd.Series, leading: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Describe frozen-threshold development OOF error groups without optimization."""
    prediction = (probabilities >= FROZEN_THRESHOLD).astype("int8")
    labels = np.select(
        [
            (y == 1) & (prediction == 1),
            (y == 0) & (prediction == 0),
            (y == 0) & (prediction == 1),
        ],
        ["true_positive", "true_negative", "false_positive"],
        default="false_negative",
    )
    groups = pd.Series(labels, index=X.index, name="error_group")
    counts = groups.value_counts().rename_axis("error_group").reset_index(name="n")
    numeric_rows: list[dict[str, Any]] = []
    for group in ("true_positive", "true_negative", "false_positive", "false_negative"):
        for feature in NUMERICAL_FEATURES:
            values = X.loc[groups == group, feature]
            numeric_rows.append(
                {
                    "error_group": group,
                    "feature": feature,
                    "n": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "minimum": values.min(),
                    "maximum": values.max(),
                }
            )
    categorical_rows: list[dict[str, Any]] = []
    categorical_leaders = [name for name in leading if name in CATEGORICAL_FEATURES][:5]
    for group in ("true_positive", "true_negative", "false_positive", "false_negative"):
        for feature in categorical_leaders:
            values = X.loc[groups == group, feature].astype(str)
            for category, count in values.value_counts().items():
                categorical_rows.append(
                    {
                        "error_group": group,
                        "feature": feature,
                        "category": category,
                        "n": int(count),
                        "within_group_fraction": float(count / len(values)),
                    }
                )
    return counts, pd.DataFrame(numeric_rows), pd.DataFrame(categorical_rows)


def save_figures(
    output_dir: Path,
    encoded: pd.DataFrame,
    original: pd.DataFrame,
    stability: pd.DataFrame,
    payloads: dict[int, dict[str, Any]],
    X: pd.DataFrame,
    subgroup: pd.DataFrame,
) -> None:
    """Create focused development-only post-hoc figures."""
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    title_suffix = "\nPOST-HOC DEVELOPMENT EXPLAINABILITY"

    top = encoded.head(20).sort_values("mean_absolute_shap")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["encoded_feature"], top["mean_absolute_shap"], color="#356859")
    ax.set_xlabel("Mean absolute SHAP (raw tree margin)")
    ax.set_title("Encoded-feature importance" + title_suffix)
    fig.tight_layout()
    fig.savefig(figures / "encoded_shap_importance.png", dpi=180)
    plt.close(fig)

    top_original = original.head(17).sort_values("mean_absolute_shap")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        top_original["original_feature"],
        top_original["mean_absolute_shap"],
        color="#2F6690",
    )
    ax.set_xlabel("Mean sum of absolute encoded-level SHAP")
    ax.set_title("Original-feature aggregated importance" + title_suffix)
    fig.tight_layout()
    fig.savefig(figures / "original_feature_shap_importance.png", dpi=180)
    plt.close(fig)

    member_original = []
    for member, payload in payloads.items():
        for feature in PREDICTIVE_FEATURES:
            columns = np.flatnonzero(payload["originals"] == feature)
            member_original.append(
                (
                    member,
                    feature,
                    np.abs(payload["values"][:, columns]).sum(axis=1).mean(),
                )
            )
    pivot = pd.DataFrame(
        member_original, columns=["member", "feature", "importance"]
    ).pivot(index="feature", columns="member", values="importance")
    leaders = original.head(10)["original_feature"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for member in pivot.columns:
        ax.plot(
            leaders,
            pivot.loc[leaders, member],
            marker="o",
            alpha=0.75,
            label=f"Member {member}",
        )
    ax.tick_params(axis="x", rotation=55)
    ax.set_ylabel("Mean absolute SHAP")
    ax.legend(ncol=3)
    ax.set_title("Ensemble-member importance stability" + title_suffix)
    fig.tight_layout()
    fig.savefig(figures / "ensemble_member_stability.png", dpi=180)
    plt.close(fig)

    common = set(payloads[1]["names"])
    for payload in payloads.values():
        common &= set(payload["names"])
    common_order = [name for name in encoded["encoded_feature"] if name in common][:20]
    shap_blocks, data_blocks = [], []
    for payload in payloads.values():
        positions = [
            int(np.where(payload["names"] == name)[0][0]) for name in common_order
        ]
        shap_blocks.append(payload["values"][:, positions])
        data_blocks.append(payload["data"][:, positions])
    explanation = shap.Explanation(
        values=np.vstack(shap_blocks),
        data=np.vstack(data_blocks),
        feature_names=common_order,
    )
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title("Member-pooled common encoded features" + title_suffix)
    plt.tight_layout()
    plt.savefig(figures / "shap_beeswarm.png", dpi=180)
    plt.close()

    leaders3 = list(original.head(3)["original_feature"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, feature in zip(axes, leaders3, strict=True):
        signed = np.mean(
            [
                payload["values"][
                    :, np.flatnonzero(payload["originals"] == feature)
                ].sum(axis=1)
                for payload in payloads.values()
            ],
            axis=0,
        )
        if feature in NUMERICAL_FEATURES:
            ax.scatter(X[feature], signed, alpha=0.35, s=14)
        else:
            categories = X[feature].astype(str)
            positions = {
                value: index for index, value in enumerate(sorted(categories.unique()))
            }
            jitter = np.random.default_rng(42).normal(0, 0.04, len(categories))
            ax.scatter(categories.map(positions) + jitter, signed, alpha=0.3, s=12)
            ax.set_xticks(list(positions.values()), list(positions), rotation=45)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(feature)
        ax.set_ylabel("Mean member SHAP (raw margin)")
    fig.suptitle("Leading-feature dependence diagnostics" + title_suffix)
    fig.tight_layout()
    fig.savefig(figures / "leading_feature_dependence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, attribute in zip(
        axes, ("personal_status_sex", "age_band", "foreign_worker"), strict=True
    ):
        frame = subgroup[subgroup["audit_attribute"] == attribute]
        ax.bar(
            frame["group"], frame["predicted_positive_rate_at_0_16"], color="#B56576"
        )
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=55)
        ax.set_title(attribute)
        ax.set_ylabel("Predicted bad-credit-risk rate")
    fig.suptitle("Audit-group policy rates (descriptive)\nDEVELOPMENT OOF ONLY")
    fig.tight_layout()
    fig.savefig(figures / "subgroup_policy_rates.png", dpi=180)
    plt.close(fig)


def _write_reports(
    output_dir: Path,
    original: pd.DataFrame,
    stability: pd.DataFrame,
    local: pd.DataFrame,
    subgroup: pd.DataFrame,
    proxy: pd.DataFrame,
    associations: pd.DataFrame,
    error_counts: pd.DataFrame,
) -> None:
    """Write the three required governance reports."""
    leaders = ", ".join(original.head(5)["original_feature"])
    unstable = stability.sort_values("sd_rank", ascending=False).iloc[0]
    small_groups = subgroup[subgroup["n"] < 50][["audit_attribute", "group", "n"]]
    small_text = (
        "; ".join(
            f"{r.audit_attribute}/{r.group}: n={r.n}" for r in small_groups.itertuples()
        )
        or "none under n=50"
    )
    global_table = original.head(10)[
        [
            "rank",
            "original_feature",
            "mean_absolute_shap",
            "sd_member_mean_absolute_shap",
        ]
    ].copy()
    global_table[["mean_absolute_shap", "sd_member_mean_absolute_shap"]] = global_table[
        ["mean_absolute_shap", "sd_member_mean_absolute_shap"]
    ].round(6)
    global_markdown = global_table.to_markdown(index=False)
    local_markdown = local.to_markdown(index=False)
    subgroup_display = subgroup.copy()
    rate_columns = [
        column
        for column in subgroup.columns
        if column not in {"audit_attribute", "group", "n", "bad_risk_n", "good_risk_n"}
    ]
    subgroup_display[rate_columns] = subgroup_display[rate_columns].round(4)
    subgroup_markdown = subgroup_display.fillna("NA").to_markdown(index=False)
    proxy_display = proxy[
        [
            "feature",
            "proxy_risk_category",
            "importance_rank",
            "mean_absolute_shap",
            "governance_concern",
        ]
    ].copy()
    proxy_display["mean_absolute_shap"] = proxy_display["mean_absolute_shap"].round(6)
    proxy_markdown = proxy_display.to_markdown(index=False)
    association_display = associations.assign(
        absolute_association=associations["association_value"].abs()
    ).sort_values("absolute_association", ascending=False)
    association_display["association_value"] = association_display[
        "association_value"
    ].round(4)
    association_markdown = association_display.head(10)[
        [
            "audit_attribute",
            "model_feature",
            "association_method",
            "association_value",
        ]
    ].to_markdown(index=False)
    (output_dir / "explainability_report.md").write_text(
        "# Stage 8 explainability report\n\n"
        "## Frozen final model architecture\n\nThe frozen predictor is the Stage 3R tree preprocessor plus fixed untuned Stage 4 XGBoost inside a five-fold sigmoid-calibrated `CalibratedClassifierCV(ensemble=True)`, followed by threshold 0.16. The fitted Stage 7 Python object was not serialized; Stage 8 therefore deterministically reconstructs and refits that exact frozen architecture on the same 800 development rows solely to inspect its five fitted member pipelines. It does not fit an unrelated sixth model, refit for performance, or access holdout rows.\n\n"
        "## Why calibrated-ensemble explanation is nontrivial\n\nEach member has its own fitted preprocessing, XGBoost model, and sigmoid calibrator. Final probabilities average five calibrated member probabilities. Tree SHAP therefore does not exactly decompose the final probability.\n\n"
        '## Exact SHAP methodology and scale\n\n`shap.TreeExplainer(model_output="raw")` explains each underlying XGBoost member\'s raw tree margin. Encoded levels are aligned by canonical `ColumnTransformer` name. Statistics for an encoded level average only across members where that level exists; absence is reported separately and is not treated as zero effect.\n\n'
        f"## Global findings\n\nThe five leading original features by ensemble mean absolute raw-margin contribution were: {leaders}. Rankings describe this frozen model, not causal importance.\n\n{global_markdown}\n\n"
        "## Original-feature aggregation\n\nFor every member, absolute contributions across a source feature's one-hot levels were summed per observation and then averaged. These totals reconcile to encoded-level absolute totals within `1e-10`. This source-feature view is preferred because one-hot encoding fragments importance.\n\n"
        f"## Ensemble-member stability\n\nRankings varied across fold-specific members; the largest rank SD was for `{unstable.original_feature}` ({unstable.sd_rank:.3f}). This is descriptive evidence of finite-sample/model instability.\n\n"
        f"## Local development cases\n\nCases were selected before inspecting SHAP patterns: minimum OOF probability, closest to 0.16, maximum probability, highest-probability false positive, and lowest-probability false negative, without row reuse. Contributors are ensemble means of underlying raw tree-score SHAP values, not calibrated-probability effects. The OOF probability comes from the honest Stage 6 nested-policy artifact and is re-thresholded at 0.16; fold-selected calibration methods varied, so it is not a row-level decomposition of the final full-development sigmoid ensemble.\n\n{local_markdown}\n\n"
        "## Semantic cautions\n\n`credit_amount` is transformed historical DM-denominated information with an undocumented transformation. Employment duration, installment-rate category, residence duration, existing-credit count, job, and dependents are discretized/categorical concepts. No plotted pattern supports a literal continuous dose-response claim.\n\n"
        "## Non-causal interpretation\n\nAll explanations are conditional associations learned from a small historical dataset. They do not establish causes, lending entitlements, or individual counterfactual outcomes.\n",
        encoding="utf-8",
    )
    (output_dir / "responsible_ai_audit.md").write_text(
        "# Stage 8 responsible-AI audit\n\n"
        "## Scope and population\n\nThis is a post-freeze, development-only audit using Stage 6 honest nested-policy OOF probabilities re-applied at frozen threshold 0.16. Those probabilities reflect calibration selected separately inside each outer training fold, rather than one global sigmoid OOF series; they remain the prespecified honest OOF evidence available for responsible-AI diagnostics. The audit was not used to optimize the model. Audit attributes were joined only after prediction and never entered the predictive feature matrix.\n\n"
        "## Audit-only attributes and subgroup evidence\n\n`personal_status_sex`, age, and foreign-worker status remain excluded from prediction. Tables report n before rates and retain undefined metrics as NA. Groups smaller than 50 are especially unstable. Small or absent groups observed: "
        + small_text
        + f".\n\n{subgroup_markdown}\n\n"
        "## Compound personal-status/sex limitation\n\nThe source field combines historical personal-status and sex codes. Sex cannot be cleanly isolated, modern gender concepts cannot be inferred, and these descriptive categories cannot support clean gender-fairness conclusions.\n\n"
        "## Age audit\n\nAge bands were frozen before metric calculation: under 25, 25–34, 35–44, 45–54, and 55+. No alternative banding was searched.\n\n"
        "## Foreign-worker audit\n\nThe source-documented yes/no categories are reported with sample sizes. The smaller category produces unstable rate estimates and should not support strong conclusions.\n\n"
        f"## Proxy-risk register\n\nSeven predefined socioeconomic predictors are reviewed qualitatively. Direct demographic exclusion does not remove their potential to encode social or economic structure.\n\n{proxy_markdown}\n\n"
        f"## Limited audit-attribute associations\n\nThe three audit attributes are crossed only with the five leading frozen-model source features, using Spearman rho, correlation ratio, or Cramer's V as appropriate. These summaries identify plausible relationships; they neither prove proxy discrimination nor causality. The ten largest observed magnitudes are shown below.\n\n{association_markdown}\n\n"
        "## Error analysis\n\nAt 0.16, development OOF error groups were: "
        + ", ".join(f"{r.error_group}={r.n}" for r in error_counts.itertuples())
        + ". Numeric and selected categorical summaries are descriptive. This analysis occurred after policy freeze and was not used for feature engineering or optimization.\n\n"
        "## Interpretation and governance recommendations\n\nSubgroup performance differs, but fairness cannot be established from this dataset. Estimates are limited by small samples, historical compound coding, proxy predictors, 1970s West German context, oversampling, and uncertain transportability. Maintain human review, document context, monitor subgroup data quality and error rates in any new setting, and require jurisdiction-specific legal and ethical review before any real use. No subgroup-specific threshold is recommended.\n",
        encoding="utf-8",
    )
    final_metrics = pd.read_csv(
        output_dir.parent / "stage7" / "final_holdout_metrics.csv"
    ).iloc[0]
    (output_dir / "model_card.md").write_text(
        "# CreditScope model card\n\n"
        "## Purpose and intended use\n\nCreditScope is an educational/research decision-support prototype demonstrating reproducible credit-risk modelling, explainability, and governance. It may support learning and methodological review. It is **not** a production lending approval system and must not autonomously approve, reject, price, or otherwise determine credit access.\n\n"
        "## Data and target\n\nThe modelling data are UCI Statlog German Credit Data (ID 144), with South German Credit (ID 573) used for corrected semantic guidance. The target is good versus bad credit risk, not observed loan default. The data describe 1,000 historical West German credits from 1973–1975 and have severe temporal/geographic transportability limits.\n\n"
        '## Governed policy\n\nSeventeen predictors use Stage 3R preprocessing: duration and transformed credit amount remain numeric; fifteen categorical/discretized variables are one-hot encoded. `personal_status_sex`, `age_years`, and `foreign_worker` are audit-only. The model is fixed untuned `XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method="hist")` in five-fold sigmoid `CalibratedClassifierCV(ensemble=True, n_jobs=None)`. Member calibrated probabilities are averaged and threshold 0.16 is applied. The assumed prototype cost is `5 × FN + FP`.\n\n'
        "## Final locked evaluation\n\nThe one-time 200-row Stage 7 holdout results were ROC-AUC "
        + f"{final_metrics.roc_auc:.6f}, Average Precision {final_metrics.average_precision:.6f}, precision {final_metrics.precision_bad_credit_risk:.6f}, recall {final_metrics.recall_bad_credit_risk:.6f}, F1 {final_metrics.f1_bad_credit_risk:.6f}, balanced accuracy {final_metrics.balanced_accuracy:.6f}, specificity {final_metrics.specificity:.6f}, Brier {final_metrics.brier_score:.6f}, log loss {final_metrics.log_loss:.6f}, and accuracy {final_metrics.accuracy:.6f}. Confusion counts were TN {int(final_metrics.true_negative)}, FP {int(final_metrics.false_positive)}, FN {int(final_metrics.false_negative)}, TP {int(final_metrics.true_positive)}; 5:1 cost was {int(final_metrics.total_cost)}. No Stage 8 result changes these values.\n\n"
        "## Explainability\n\nTree SHAP is calculated on each underlying XGBoost member's raw-margin output and aggregated descriptively. It does not exactly decompose the calibrated ensemble probability. Importance and local contributions are associative model explanations, not causal effects.\n\n"
        "## Responsible-AI findings and limitations\n\nDirect demographic exclusions do not remove proxy risk. Checking/savings status, employment, housing, property, telephone, and job may encode socioeconomic position. Development OOF subgroup rates differ, but are descriptive, sometimes unstable, and cannot establish fairness, absence of discrimination, or legal compliance. The non-foreign-worker source group has only 31 development observations; one documented personal-status/sex category has no observations. The compound field cannot isolate sex or represent modern gender concepts. Credit amount has an undocumented monotonic transformation. Measurement timing is incomplete.\n\n"
        "## Human oversight\n\nAny research demonstration should expose uncertainty, explanations, policy assumptions, and avenues for human review. Real deployment would require representative contemporary data, jurisdiction-specific review, impact assessment, monitoring, recourse design, security controls, and independent validation.\n",
        encoding="utf-8",
    )


def generate_stage8_artifacts(project_root: Path) -> dict[str, Any]:
    """Generate all Stage 8 development-only explainability/audit evidence."""
    locked = verify_locked_evidence(project_root)
    output_dir = project_root / "reports" / "stage8"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = build_analysis_dataset(load_verified_raw_snapshot(project_root))
    data.index = source_row_ids(data)
    manifest = pd.read_csv(project_root / "reports" / "split_manifest.csv").set_index(
        ROW_ID_NAME
    )
    development_ids = manifest.index[manifest["split"] == DEVELOPMENT_SPLIT]
    if len(development_ids) != 800:
        raise RuntimeError("Development population changed.")
    X = data.loc[development_ids, list(PREDICTIVE_FEATURES)].copy()
    y = data.loc[development_ids, TARGET_BINARY_NAME].astype("int8").copy()
    audit = data.loc[development_ids, list(AUDIT_ATTRIBUTES)].copy()
    if set(X.columns) & set(AUDIT_ATTRIBUTES):
        raise RuntimeError("Audit-only attributes entered explanation model features.")
    _, folds = load_outer_folds(project_root, X.index)
    ensemble = build_final_estimator(folds).fit(X, y)
    encoded_members, original_members, payloads = extract_member_shap(ensemble, X)
    encoded = aggregate_encoded_importance(encoded_members)
    original = aggregate_original_importance(original_members)
    stability = importance_stability(original_members)

    oof = (
        pd.read_csv(
            project_root / "reports" / "stage6" / "nested_policy_oof_predictions.csv"
        )
        .set_index(ROW_ID_NAME)
        .loc[development_ids]
        .reset_index()
    )
    probabilities = oof.set_index(ROW_ID_NAME)["probability"].loc[development_ids]
    subgroup, samples = subgroup_metrics(audit, y, probabilities)
    leading = list(original.head(5)["original_feature"])
    associations = audit_feature_associations(audit, X, leading)
    proxy = proxy_register(original)
    cases = select_local_cases(oof)
    local = local_explanation_table(cases, payloads, X.index)
    error_counts, error_numeric, error_categorical = error_analysis(
        X, y, probabilities, leading
    )

    encoded_members.to_csv(
        output_dir / "ensemble_member_shap_importance.csv",
        index=False,
        lineterminator="\n",
    )
    encoded.to_csv(
        output_dir / "encoded_shap_importance.csv", index=False, lineterminator="\n"
    )
    original.to_csv(
        output_dir / "original_feature_shap_importance.csv",
        index=False,
        lineterminator="\n",
    )
    stability.to_csv(
        output_dir / "ensemble_importance_stability.csv",
        index=False,
        lineterminator="\n",
    )
    local.to_csv(
        output_dir / "local_explanation_cases.csv", index=False, lineterminator="\n"
    )
    samples.to_csv(
        output_dir / "subgroup_sample_sizes.csv", index=False, lineterminator="\n"
    )
    subgroup.to_csv(
        output_dir / "subgroup_metrics.csv", index=False, lineterminator="\n"
    )
    proxy.to_csv(
        output_dir / "proxy_risk_register.csv", index=False, lineterminator="\n"
    )
    associations.to_csv(
        output_dir / "audit_feature_associations.csv", index=False, lineterminator="\n"
    )
    error_counts.to_csv(
        output_dir / "error_group_counts.csv", index=False, lineterminator="\n"
    )
    error_numeric.to_csv(
        output_dir / "error_group_numeric_summary.csv", index=False, lineterminator="\n"
    )
    error_categorical.to_csv(
        output_dir / "error_group_categorical_distributions.csv",
        index=False,
        lineterminator="\n",
    )
    metadata = pd.DataFrame(
        [
            {
                "member": member,
                "encoded_feature_count": len(payload["names"]),
                "calibrator_method": "sigmoid",
                "calibrator_a": payload["calibrator_a"],
                "calibrator_b": payload["calibrator_b"],
                "shap_output": "raw_tree_margin",
            }
            for member, payload in payloads.items()
        ]
    )
    metadata.to_csv(
        output_dir / "ensemble_member_metadata.csv", index=False, lineterminator="\n"
    )
    save_figures(output_dir, encoded, original, stability, payloads, X, subgroup)
    _write_reports(
        output_dir,
        original,
        stability,
        local,
        subgroup,
        proxy,
        associations,
        error_counts,
    )
    summary = {
        "stage": "Stage 8 post-hoc explainability and responsible-AI audit",
        "population": "800 development observations only",
        "holdout_accessed_for_new_analysis": False,
        "frozen_policy_changed": False,
        "locked_hashes": locked,
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "scikit_learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
            "shap": version("shap"),
            "numba": version("numba"),
            "llvmlite": version("llvmlite"),
        },
        "explanation": {
            "members": 5,
            "model_output": "raw XGBoost margin",
            "calibrated_probability_decomposition": False,
            "encoded_abs_aggregation_atol": 1e-10,
        },
        "age_groups_frozen": list(AGE_LABELS),
        "local_case_selection": [
            "minimum OOF probability",
            "closest to 0.16",
            "maximum OOF probability",
            "highest-probability false positive",
            "lowest-probability false negative",
        ],
        "leading_original_features": original.head(5)["original_feature"].tolist(),
        "artifact_hashes": {},
    }
    for path in sorted(output_dir.glob("*.csv")):
        summary["artifact_hashes"][path.name] = sha256(path)
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    """CLI entry point."""
    summary = generate_stage8_artifacts(Path.cwd())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
