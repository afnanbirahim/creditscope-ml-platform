"""Generate reproducible Stage 2 dictionaries, summaries, figures, and report."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from creditscope.analysis import build_analysis_dataset
from creditscope.schema import (
    CATEGORICAL_SPECS,
    CATEGORY_LABELS,
    FEATURE_SPECS,
    NUMERICAL_SPECS,
    TARGET_BINARY_NAME,
    TARGET_LABELS,
    TARGET_ORIGINAL_NAME,
)

RARE_CATEGORY_THRESHOLD = 0.02
SOURCE_URL = "https://archive.ics.uci.edu/dataset/144/statlog%2Bgerman%2Bcredit%2Bdata"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_dictionary_frame() -> pd.DataFrame:
    """Return one row per documented feature/category code plus the target."""
    rows: list[dict[str, object]] = []
    for spec in FEATURE_SPECS:
        categories = spec.categories or (("", ""),)
        for code, meaning in categories:
            rows.append(
                {
                    "uci_field": spec.uci_name,
                    "internal_field": spec.name,
                    "official_meaning": spec.official_meaning,
                    "official_type": spec.official_type,
                    "analytical_type": spec.analytical_type,
                    "unit": spec.unit or "not documented",
                    "category_code": code,
                    "official_category_meaning": meaning,
                    "source": SOURCE_URL,
                }
            )
    rows.extend(
        [
            {
                "uci_field": "class",
                "internal_field": TARGET_ORIGINAL_NAME,
                "official_meaning": "Credit-risk class",
                "official_type": "Binary target",
                "analytical_type": "binary categorical",
                "unit": "not applicable",
                "category_code": value,
                "official_category_meaning": label,
                "source": SOURCE_URL,
            }
            for value, label in TARGET_LABELS.items()
        ]
    )
    return pd.DataFrame(rows)


def write_data_dictionary(project_root: Path) -> None:
    """Write machine- and human-readable versions of the authoritative schema."""
    dictionary = data_dictionary_frame()
    dictionary.to_csv(project_root / "reports" / "data_dictionary.csv", index=False)

    lines = [
        "# CreditScope data dictionary",
        "",
        (
            "Authoritative source: [UCI Statlog (German Credit Data), dataset ID 144]"
            f"({SOURCE_URL}). Category wording below follows UCI's `german.doc`. Blank category "
            "cells identify numerical fields. “Not documented” means no unit was supplied by UCI."
        ),
        "",
        (
            "The original target is preserved as `credit_risk_class`: 1 = good credit risk and "
            "2 = bad credit risk. The analysis-only `bad_credit_risk` maps these to 0 and 1."
        ),
        "",
        "| UCI field | Internal field | Official meaning | Official type | Analytical type | Unit |",
        "|---|---|---|---|---|---|",
    ]
    for spec in FEATURE_SPECS:
        lines.append(
            f"| {spec.uci_name} | `{spec.name}` | {spec.official_meaning} | "
            f"{spec.official_type} | {spec.analytical_type} | "
            f"{spec.unit or 'not documented'} |"
        )
    lines.extend(["", "## Official categorical codes", ""])
    for spec in CATEGORICAL_SPECS:
        lines.extend(
            [
                f"### {spec.uci_name}: `{spec.name}`",
                "",
                "| Code | Official meaning |",
                "|---|---|",
                *[f"| {code} | {meaning} |" for code, meaning in spec.categories],
                "",
            ]
        )
    lines.extend(
        [
            "## Target",
            "",
            "| Source value | Source meaning | `bad_credit_risk` |",
            "|---:|---|---:|",
            "| 1 | good credit risk | 0 |",
            "| 2 | bad credit risk | 1 |",
            "",
        ]
    )
    (project_root / "docs" / "data_dictionary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def governance_frame() -> pd.DataFrame:
    """Return the Stage 2 feature-governance review."""
    return pd.DataFrame(
        [
            {
                "internal_field": spec.name,
                "official_meaning": spec.official_meaning,
                "classification": spec.governance,
                "rationale": spec.governance_rationale,
                "stage3_action": spec.stage3_action,
            }
            for spec in FEATURE_SPECS
        ]
    )


def write_summary_tables(data: pd.DataFrame, reports_dir: Path) -> None:
    """Write observed descriptive and quality tables without modelling."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "metric": [
                "observations",
                "candidate_features",
                "source_target_columns",
                "analysis_binary_target_columns",
                "missing_cells",
                "duplicate_rows_including_targets",
            ],
            "value": [
                len(data),
                len(FEATURE_SPECS),
                1,
                1,
                int(data.isna().sum().sum()),
                int(data.duplicated().sum()),
            ],
        }
    ).to_csv(reports_dir / "dataset_overview.csv", index=False)

    target_counts = data[TARGET_ORIGINAL_NAME].value_counts().sort_index()
    target_summary = pd.DataFrame(
        {
            "credit_risk_class": target_counts.index.astype(int),
            "meaning": [TARGET_LABELS[int(value)] for value in target_counts.index],
            "bad_credit_risk": [0, 1],
            "count": target_counts.to_numpy(),
            "percentage": (target_counts / len(data)).to_numpy(),
        }
    )
    target_summary.to_csv(reports_dir / "target_summary.csv", index=False)

    numerical_names = [spec.name for spec in NUMERICAL_SPECS]
    numerical = data[numerical_names].describe().T.reset_index(names="feature")
    numerical.rename(columns={"25%": "q25", "50%": "median", "75%": "q75"}).to_csv(
        reports_dir / "numerical_summary.csv", index=False
    )

    frequency_parts = []
    cardinality_rows = []
    for spec in CATEGORICAL_SPECS:
        counts = (
            data[spec.name]
            .value_counts()
            .reindex([code for code, _ in spec.categories], fill_value=0)
        )
        frame = pd.DataFrame(
            {
                "feature": spec.name,
                "category_code": counts.index,
                "category_meaning": [
                    dict(spec.categories)[code] for code in counts.index
                ],
                "count": counts.to_numpy(),
                "percentage": (counts / len(data)).to_numpy(),
                "observed": counts.to_numpy() > 0,
            }
        )
        frequency_parts.append(frame)
        observed_counts = counts[counts > 0]
        cardinality_rows.append(
            {
                "feature": spec.name,
                "documented_cardinality": len(spec.categories),
                "observed_cardinality": int(data[spec.name].nunique()),
                "minimum_observed_category_count": int(observed_counts.min()),
            }
        )
    frequencies = pd.concat(frequency_parts, ignore_index=True)
    frequencies.to_csv(reports_dir / "categorical_frequencies.csv", index=False)
    pd.DataFrame(cardinality_rows).to_csv(
        reports_dir / "categorical_cardinality.csv", index=False
    )
    frequencies[
        (frequencies["count"] > 0)
        & (frequencies["percentage"] < RARE_CATEGORY_THRESHOLD)
    ].to_csv(reports_dir / "rare_categories.csv", index=False)

    quality_checks = [
        {
            "check": "missing cells",
            "flagged_rows_or_cells": int(data.isna().sum().sum()),
        },
        {
            "check": "duplicate complete rows",
            "flagged_rows_or_cells": int(data.duplicated().sum()),
        },
        {
            "check": "nonpositive duration_months",
            "flagged_rows_or_cells": int((data["duration_months"] <= 0).sum()),
        },
        {
            "check": "nonpositive credit_amount",
            "flagged_rows_or_cells": int((data["credit_amount"] <= 0).sum()),
        },
        {
            "check": "nonpositive age_years",
            "flagged_rows_or_cells": int((data["age_years"] <= 0).sum()),
        },
    ]
    for spec in CATEGORICAL_SPECS:
        allowed = {code for code, _ in spec.categories}
        quality_checks.append(
            {
                "check": f"out-of-domain codes: {spec.name}",
                "flagged_rows_or_cells": int((~data[spec.name].isin(allowed)).sum()),
            }
        )
    pd.DataFrame(quality_checks).to_csv(reports_dir / "quality_checks.csv", index=False)
    governance_frame().to_csv(reports_dir / "feature_governance.csv", index=False)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _risk_label(value: int) -> str:
    return TARGET_LABELS[int(value)].capitalize()


def _distribution_figure(
    data: pd.DataFrame, feature: str, title: str, xlabel: str
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    for value, color in ((1, "#2B6F8A"), (2, "#D97732")):
        subset = data.loc[data[TARGET_ORIGINAL_NAME] == value, feature]
        ax.hist(
            subset,
            bins=24,
            alpha=0.58,
            density=True,
            label=_risk_label(value),
            color=color,
        )
    ax.set(title=title, xlabel=xlabel, ylabel="Density")
    ax.legend(frameon=False)
    return fig


def _rate_figure(data: pd.DataFrame, feature: str, title: str) -> plt.Figure:
    labels = CATEGORY_LABELS[feature]
    rates = data.groupby(feature, observed=True)[TARGET_BINARY_NAME].agg(
        ["mean", "size"]
    )
    rates = rates.reindex([code for code in labels if code in rates.index])
    display_labels = [
        f"{code}: {labels[code]} (n={int(rates.loc[code, 'size'])})"
        for code in rates.index
    ]
    height = max(4.5, 0.55 * len(rates))
    fig, ax = plt.subplots(figsize=(10, height))
    bars = ax.barh(display_labels, rates["mean"], color="#447E9B")
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in rates["mean"]], padding=4)
    ax.set(title=title, xlabel="Observed bad-credit-risk rate", ylabel="")
    ax.set_xlim(0, max(0.6, float(rates["mean"].max()) + 0.08))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.invert_yaxis()
    return fig


def _box_figure(
    data: pd.DataFrame, feature: str, title: str, ylabel: str
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [
        data.loc[data[TARGET_ORIGINAL_NAME] == value, feature] for value in (1, 2)
    ]
    boxes = ax.boxplot(
        groups, tick_labels=[_risk_label(1), _risk_label(2)], patch_artist=True
    )
    for patch, color in zip(boxes["boxes"], ("#2B6F8A", "#D97732"), strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set(title=title, xlabel="Credit-risk class", ylabel=ylabel)
    return fig


def create_figures(data: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Create ten focused publication-quality EDA figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    created: list[Path] = []

    counts = data[TARGET_ORIGINAL_NAME].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [_risk_label(value) for value in counts.index],
        counts,
        color=["#2B6F8A", "#D97732"],
    )
    ax.bar_label(
        bars, labels=[f"{count} ({count / len(data):.0%})" for count in counts]
    )
    ax.set(title="Credit-risk class distribution", ylabel="Observations", xlabel="")
    path = figures_dir / "01_class_distribution.png"
    _save_figure(fig, path)
    created.append(path)

    specs = (
        (
            "credit_amount",
            "Credit amount distribution by credit-risk class",
            "Credit amount",
        ),
        (
            "duration_months",
            "Duration distribution by credit-risk class",
            "Duration (months)",
        ),
        ("age_years", "Age distribution by credit-risk class", "Age (years)"),
    )
    for number, (feature, title, xlabel) in enumerate(specs, start=2):
        path = figures_dir / f"{number:02d}_{feature}_distribution.png"
        _save_figure(_distribution_figure(data, feature, title, xlabel), path)
        created.append(path)

    rate_specs = (
        ("checking_account_status", "Observed risk rate by checking-account category"),
        ("savings_account_status", "Observed risk rate by savings category"),
        ("credit_history", "Observed risk rate by credit-history category"),
        ("purpose", "Observed risk rate by purpose category"),
    )
    for number, (feature, title) in enumerate(rate_specs, start=5):
        path = figures_dir / f"{number:02d}_{feature}_risk_rate.png"
        _save_figure(_rate_figure(data, feature, title), path)
        created.append(path)

    box_specs = (
        (
            "duration_months",
            "Duration differs across observed risk classes",
            "Duration (months)",
        ),
        (
            "credit_amount",
            "Credit amount differs across observed risk classes",
            "Credit amount",
        ),
    )
    for number, (feature, title, ylabel) in enumerate(box_specs, start=9):
        path = figures_dir / f"{number:02d}_{feature}_by_risk.png"
        _save_figure(_box_figure(data, feature, title, ylabel), path)
        created.append(path)
    return created


def _rate_range_sentence(data: pd.DataFrame, feature: str) -> str:
    labels = CATEGORY_LABELS[feature]
    grouped = data.groupby(feature)[TARGET_BINARY_NAME].agg(["mean", "size"])
    low_code = str(grouped["mean"].idxmin())
    high_code = str(grouped["mean"].idxmax())
    return (
        f"For `{feature}`, observed bad-credit-risk rates ranged from "
        f"{grouped.loc[low_code, 'mean']:.1%} for {low_code} ({labels[low_code]}, "
        f"n={int(grouped.loc[low_code, 'size'])}) to "
        f"{grouped.loc[high_code, 'mean']:.1%} for {high_code} "
        f"({labels[high_code]}, n={int(grouped.loc[high_code, 'size'])})."
    )


def write_findings_report(data: pd.DataFrame, reports_dir: Path) -> None:
    """Write a concise factual Stage 2 report from observed calculations."""
    counts = data[TARGET_ORIGINAL_NAME].value_counts().sort_index()
    rare = pd.read_csv(reports_dir / "rare_categories.csv")
    medians = data.groupby(TARGET_ORIGINAL_NAME)[
        ["duration_months", "credit_amount"]
    ].median()
    lines = [
        "# Stage 2 data understanding and EDA summary",
        "",
        "## 1. Dataset overview",
        "",
        (
            f"The verified UCI source contains {len(data):,} observations and {len(FEATURE_SPECS)} candidate "
            "features. It contains no missing cells and no duplicate complete rows. Stage 2 created no "
            "model and no train/test split."
        ),
        "",
        "## 2. Target semantics",
        "",
        (
            "UCI defines source class 1 as good credit risk and class 2 as bad credit risk. The analysis "
            "view preserves `credit_risk_class` and adds `bad_credit_risk` (0 = good, 1 = bad). The target "
            "must not be interpreted as an observed loan-default event."
        ),
        "",
        "## 3. Main descriptive findings",
        "",
        (
            f"There are {int(counts.loc[1])} good-credit-risk records ({counts.loc[1] / len(data):.1%}) "
            f"and {int(counts.loc[2])} bad-credit-risk records ({counts.loc[2] / len(data):.1%}). "
            f"The documented categorical schema contains {len(rare)} observed categories below the "
            f"predeclared {RARE_CATEGORY_THRESHOLD:.0%} rarity threshold. Detailed numerical ranges and "
            "category counts are in the generated CSV tables."
        ),
        "",
        "## 4. Important EDA observations",
        "",
        f"- {_rate_range_sentence(data, 'checking_account_status')}",
        f"- {_rate_range_sentence(data, 'savings_account_status')}",
        f"- {_rate_range_sentence(data, 'credit_history')}",
        f"- {_rate_range_sentence(data, 'purpose')}",
        (
            f"- Median duration was {medians.loc[1, 'duration_months']:.0f} months for good-credit-risk "
            f"records and {medians.loc[2, 'duration_months']:.0f} months for bad-credit-risk records."
        ),
        (
            f"- Median credit amount was {medians.loc[1, 'credit_amount']:.0f} for good-credit-risk "
            f"records and {medians.loc[2, 'credit_amount']:.0f} for bad-credit-risk records. UCI does not "
            "state a unit for this field in its current variable metadata."
        ),
        "",
        "These are sample associations, not causal effects. Small category counts make some rates unstable.",
        "",
        "## 5. Class imbalance",
        "",
        (
            "The 70%/30% split is a moderate imbalance: both classes are represented, but an all-good "
            "classifier would already achieve 70% accuracy. Later evaluation therefore needs class-specific "
            "metrics and cost-sensitive analysis. No resampling or class weighting was applied."
        ),
        "",
        "## 6. Sensitive-feature considerations",
        "",
        (
            "`personal_status_sex`, `age_years`, and `foreign_worker` are explicitly demographic or "
            "sensitive in context. Employment, job, housing, property, savings, checking-account status, "
            "residence duration, dependents, and telephone may act as proxies. These labels do not assert "
            "jurisdiction-specific protected-class status. No feature was removed in Stage 2."
        ),
        "",
        "## 7. Potential leakage concerns",
        "",
        (
            "No feature is an obvious copy of the target or explicit post-outcome field. However, UCI does "
            "not fully document measurement timing. Credit history, existing credits at this bank, checking "
            "status, savings, other installment plans, and debtors/guarantors require confirmation that they "
            "were known at the decision point. The compound personal-status/sex field cannot cleanly separate "
            "its components. These are Stage 3 review flags, not reasons for automatic removal."
        ),
        "",
        "## 8. Cost-matrix implications",
        "",
        (
            "UCI assigns cost 5 to a bad credit risk classified as good and cost 1 to a good credit risk "
            "classified as bad. Later evaluation and threshold selection must reflect this asymmetry. No "
            "threshold was selected in Stage 2."
        ),
        "",
        "## 9. Dataset limitations",
        "",
        (
            "The dataset is small, donated in 1994, geographically specific, and sparsely documented on "
            "sampling and measurement timing. Several values are opaque codes, some categories are rare, "
            "and demographic/proxy variables create fairness concerns. It cannot establish causal effects or "
            "demonstrate suitability for contemporary lending decisions."
        ),
        "",
        "## 10. Decisions required before modelling",
        "",
        "1. Define the intended decision-support use and evaluation population.",
        (
            "2. Choose whether sensitive fields are excluded from prediction, retained only for auditing, or "
            "used under a documented research rationale."
        ),
        "3. Decide how to treat likely proxy variables and the compound personal-status/sex field.",
        "4. Confirm application-time availability and define a leakage-safe feature set.",
        "5. Choose preprocessing for nominal, ordinal, special unknown/no-account, and rare categories.",
        "6. Predefine validation design, metrics, cost-sensitive evaluation, and final holdout isolation.",
        "",
    ]
    (reports_dir / "stage2_eda_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def generate_stage2_artifacts(
    project_root: Path, data: pd.DataFrame | None = None
) -> dict[str, object]:
    """Run the complete, non-modelling Stage 2 artifact pipeline."""
    raw_path = project_root / "data" / "raw" / "german_credit.csv"
    raw_hash_before = _sha256(raw_path) if raw_path.exists() else None
    data = build_analysis_dataset() if data is None else data
    write_data_dictionary(project_root)
    write_summary_tables(data, project_root / "reports")
    figures = create_figures(data, project_root / "reports" / "figures")
    write_findings_report(data, project_root / "reports")
    raw_hash_after = _sha256(raw_path) if raw_path.exists() else None
    if raw_hash_before != raw_hash_after:
        raise RuntimeError("Raw source snapshot changed during Stage 2 generation.")
    return {
        "rows": len(data),
        "candidate_features": len(FEATURE_SPECS),
        "figures_created": len(figures),
        "raw_snapshot_unchanged": raw_hash_before == raw_hash_after,
        "model_trained": False,
        "train_test_split_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    print(generate_stage2_artifacts(args.project_root.resolve()))


if __name__ == "__main__":
    main()
