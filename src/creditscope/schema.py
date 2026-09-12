"""Authoritative Stage 2 schema derived from UCI dataset 144 documentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    """Documented source semantics and internal governance metadata."""

    uci_name: str
    name: str
    official_meaning: str
    official_type: str
    analytical_type: str
    unit: str | None = None
    categories: tuple[tuple[str, str], ...] = ()
    governance: str = "ordinary candidate predictor"
    governance_rationale: str = "No specific concern identified at this stage."
    stage3_action: str = "Retain as a candidate; assess preprocessing before modelling."


FEATURE_SPECS = (
    FeatureSpec(
        "Attribute1",
        "checking_account_status",
        "Status of existing checking account",
        "Categorical",
        "ordered categorical with a special no-account category",
        categories=(
            ("A11", "... < 0 DM"),
            ("A12", "0 <= ... < 200 DM"),
            ("A13", "... >= 200 DM / salary assignments for at least 1 year"),
            ("A14", "no checking account"),
        ),
        governance="potential proxy; requires later modelling decision",
        governance_rationale="May encode financial access or socioeconomic position.",
        stage3_action="Assess necessity, encoding, and proxy risk before modelling.",
    ),
    FeatureSpec(
        "Attribute2",
        "duration_months",
        "Duration",
        "Integer",
        "discrete numerical",
        "months",
    ),
    FeatureSpec(
        "Attribute3",
        "credit_history",
        "Credit history",
        "Categorical",
        "nominal categorical",
        categories=(
            ("A30", "no credits taken / all credits paid back duly"),
            ("A31", "all credits at this bank paid back duly"),
            ("A32", "existing credits paid back duly till now"),
            ("A33", "delay in paying off in the past"),
            ("A34", "critical account / other credits existing (not at this bank)"),
        ),
        governance="ordinary candidate predictor; requires timing review",
        governance_rationale="Credit history is substantively relevant, but observation timing must precede the decision.",
        stage3_action="Confirm application-time availability and encode without imposing an undocumented order.",
    ),
    FeatureSpec(
        "Attribute4",
        "purpose",
        "Purpose",
        "Categorical",
        "nominal categorical",
        categories=(
            ("A40", "car (new)"),
            ("A41", "car (used)"),
            ("A42", "furniture/equipment"),
            ("A43", "radio/television"),
            ("A44", "domestic appliances"),
            ("A45", "repairs"),
            ("A46", "education"),
            ("A47", "(vacation - does not exist?)"),
            ("A48", "retraining"),
            ("A49", "business"),
            ("A410", "others"),
        ),
    ),
    FeatureSpec(
        "Attribute5",
        "credit_amount",
        "Credit amount",
        "Integer",
        "discrete numerical",
    ),
    FeatureSpec(
        "Attribute6",
        "savings_account_status",
        "Savings account/bonds",
        "Categorical",
        "ordered categorical with a special unknown/no-account category",
        categories=(
            ("A61", "... < 100 DM"),
            ("A62", "100 <= ... < 500 DM"),
            ("A63", "500 <= ... < 1000 DM"),
            ("A64", ".. >= 1000 DM"),
            ("A65", "unknown / no savings account"),
        ),
        governance="potential proxy; requires later modelling decision",
        governance_rationale="May encode socioeconomic position or access to financial products.",
        stage3_action="Assess necessity, missing-like semantics, encoding, and proxy risk.",
    ),
    FeatureSpec(
        "Attribute7",
        "employment_duration",
        "Present employment since",
        "Categorical",
        "ordinal categorical",
        categories=(
            ("A71", "unemployed"),
            ("A72", "... < 1 year"),
            ("A73", "1 <= ... < 4 years"),
            ("A74", "4 <= ... < 7 years"),
            ("A75", ".. >= 7 years"),
        ),
        governance="potential proxy; requires later modelling decision",
        governance_rationale="Employment duration may reflect socioeconomic circumstances.",
        stage3_action="Review fairness implications and encoding before modelling.",
    ),
    FeatureSpec(
        "Attribute8",
        "installment_rate_percent",
        "Installment rate in percentage of disposable income",
        "Integer",
        "discrete numerical percentage",
        "percentage of disposable income",
    ),
    FeatureSpec(
        "Attribute9",
        "personal_status_sex",
        "Personal status and sex",
        "Categorical",
        "nominal categorical (compound field)",
        categories=(
            ("A91", "male: divorced/separated"),
            ("A92", "female: divorced/separated/married"),
            ("A93", "male: single"),
            ("A94", "male: married/widowed"),
            ("A95", "female: single"),
        ),
        governance="potentially sensitive/protected; requires later modelling decision",
        governance_rationale="Directly combines sex and personal/marital status; legal treatment is jurisdiction-specific.",
        stage3_action="Decide exclusion, audit-only use, or constrained inclusion before modelling.",
    ),
    FeatureSpec(
        "Attribute10",
        "other_debtors_guarantors",
        "Other debtors / guarantors",
        "Categorical",
        "nominal categorical",
        categories=(
            ("A101", "none"),
            ("A102", "co-applicant"),
            ("A103", "guarantor"),
        ),
    ),
    FeatureSpec(
        "Attribute11",
        "residence_duration",
        "Present residence since",
        "Integer",
        "ordinal integer (unit not documented by UCI)",
        governance="potential proxy; requires later modelling decision",
        governance_rationale="Residential stability may proxy socioeconomic circumstances.",
        stage3_action="Clarify encoding and review proxy risk before modelling.",
    ),
    FeatureSpec(
        "Attribute12",
        "property",
        "Property",
        "Categorical",
        "nominal categorical",
        categories=(
            ("A121", "real estate"),
            (
                "A122",
                "if not A121: building society savings agreement / life insurance",
            ),
            ("A123", "if not A121/A122: car or other, not in Attribute6"),
            ("A124", "unknown / no property"),
        ),
        governance="potential proxy; requires later modelling decision",
        governance_rationale="Asset ownership may encode socioeconomic position.",
        stage3_action="Assess necessity, encoding, and proxy risk before modelling.",
    ),
    FeatureSpec(
        "Attribute13",
        "age_years",
        "Age",
        "Integer",
        "discrete numerical",
        "years",
        governance="potentially sensitive/protected; requires later modelling decision",
        governance_rationale="Age is demographic; legal treatment is jurisdiction-specific.",
        stage3_action="Decide exclusion, audit-only use, or constrained inclusion before modelling.",
    ),
    FeatureSpec(
        "Attribute14",
        "other_installment_plans",
        "Other installment plans",
        "Categorical",
        "nominal categorical",
        categories=(("A141", "bank"), ("A142", "stores"), ("A143", "none")),
    ),
    FeatureSpec(
        "Attribute15",
        "housing",
        "Housing",
        "Categorical",
        "nominal categorical",
        categories=(("A151", "rent"), ("A152", "own"), ("A153", "for free")),
        governance="potential proxy; requires later modelling decision",
        governance_rationale="Housing status may encode socioeconomic position.",
        stage3_action="Assess necessity and proxy risk before modelling.",
    ),
    FeatureSpec(
        "Attribute16",
        "existing_credits_count",
        "Number of existing credits at this bank",
        "Integer",
        "count",
        "credits",
        governance="ordinary candidate predictor; requires timing review",
        governance_rationale="The count may be relevant but must be available at application time.",
        stage3_action="Confirm application-time availability before modelling.",
    ),
    FeatureSpec(
        "Attribute17",
        "job",
        "Job",
        "Categorical",
        "ordinal categorical",
        categories=(
            ("A171", "unemployed / unskilled - non-resident"),
            ("A172", "unskilled - resident"),
            ("A173", "skilled employee / official"),
            (
                "A174",
                "management / self-employed / highly qualified employee / officer",
            ),
        ),
        governance="potential proxy; requires later modelling decision",
        governance_rationale="Occupation grouping may encode socioeconomic position.",
        stage3_action="Review fairness implications and encoding before modelling.",
    ),
    FeatureSpec(
        "Attribute18",
        "dependents_count",
        "Number of people being liable to provide maintenance for",
        "Integer",
        "count",
        "people",
        governance="potential proxy; requires later modelling decision",
        governance_rationale="Household dependency information may relate to family circumstances.",
        stage3_action="Review necessity and proxy risk before modelling.",
    ),
    FeatureSpec(
        "Attribute19",
        "telephone",
        "Telephone",
        "Binary",
        "binary categorical",
        categories=(
            ("A191", "none"),
            ("A192", "yes, registered under the customer's name"),
        ),
        governance="potential proxy; requires later modelling decision",
        governance_rationale="Telephone registration may proxy access, stability, or socioeconomic position.",
        stage3_action="Assess necessity and proxy risk before modelling.",
    ),
    FeatureSpec(
        "Attribute20",
        "foreign_worker",
        "Foreign worker",
        "Binary",
        "binary categorical",
        categories=(("A201", "yes"), ("A202", "no")),
        governance="potentially sensitive/protected; requires later modelling decision",
        governance_rationale="Foreign-worker status is demographic; legal treatment is jurisdiction-specific.",
        stage3_action="Decide exclusion, audit-only use, or constrained inclusion before modelling.",
    ),
)

FEATURE_NAME_MAP = {spec.uci_name: spec.name for spec in FEATURE_SPECS}
CATEGORICAL_SPECS = tuple(spec for spec in FEATURE_SPECS if spec.categories)
NUMERICAL_SPECS = tuple(spec for spec in FEATURE_SPECS if not spec.categories)
CATEGORY_LABELS = {spec.name: dict(spec.categories) for spec in CATEGORICAL_SPECS}

TARGET_ORIGINAL_NAME = "credit_risk_class"
TARGET_BINARY_NAME = "bad_credit_risk"
TARGET_LABELS = {1: "good credit risk", 2: "bad credit risk"}
TARGET_BINARY_MAP = {1: 0, 2: 1}

TARGET_UCI_NAME = "class"
TARGET_ORIGINAL_NAME = "credit_risk_class"
TARGET_BINARY_NAME = "bad_credit_risk"
TARGET_LABELS = {1: "good credit risk", 2: "bad credit risk"}
TARGET_BINARY_MAP = {1: 0, 2: 1}
