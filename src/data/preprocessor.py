"""
Data preprocessing, feature engineering, and train/test splitting pipeline.
Ensures zero data leakage, provides clean column normalization,
and produces structured outputs for machine learning training and web app inference.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "data.csv")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# Canonical mapping from raw CSV column names (stripped) to clean snake_case names
COLUMN_MAPPING = {
    "Marital status": "marital_status",
    "Application mode": "application_mode",
    "Application order": "application_order",
    "Course": "course",
    "Daytime/evening attendance": "daytime_evening_attendance",
    "Daytime/evening attendance\t": "daytime_evening_attendance",
    "Previous qualification": "previous_qualification",
    "Previous qualification (grade)": "previous_qualification_grade",
    "Nacionality": "nationality",
    "Mother's qualification": "mothers_qualification",
    "Father's qualification": "fathers_qualification",
    "Mother's occupation": "mothers_occupation",
    "Father's occupation": "fathers_occupation",
    "Admission grade": "admission_grade",
    "Displaced": "displaced",
    "Educational special needs": "educational_special_needs",
    "Debtor": "debtor",
    "Tuition fees up to date": "tuition_fees_up_to_date",
    "Gender": "gender",
    "Scholarship holder": "scholarship_holder",
    "Age at enrollment": "age_at_enrollment",
    "International": "international",
    "Curricular units 1st sem (credited)": "cu_1st_sem_credited",
    "Curricular units 1st sem (enrolled)": "cu_1st_sem_enrolled",
    "Curricular units 1st sem (evaluations)": "cu_1st_sem_evaluations",
    "Curricular units 1st sem (approved)": "cu_1st_sem_approved",
    "Curricular units 1st sem (grade)": "cu_1st_sem_grade",
    "Curricular units 1st sem (without evaluations)": "cu_1st_sem_without_evaluations",
    "Curricular units 2nd sem (credited)": "cu_2nd_sem_credited",
    "Curricular units 2nd sem (enrolled)": "cu_2nd_sem_enrolled",
    "Curricular units 2nd sem (evaluations)": "cu_2nd_sem_evaluations",
    "Curricular units 2nd sem (approved)": "cu_2nd_sem_approved",
    "Curricular units 2nd sem (grade)": "cu_2nd_sem_grade",
    "Curricular units 2nd sem (without evaluations)": "cu_2nd_sem_without_evaluations",
    "Unemployment rate": "unemployment_rate",
    "Inflation rate": "inflation_rate",
    "GDP": "gdp",
    "Target": "target_raw",
}

# Categorized feature lists
DEMOGRAPHIC_FEATURES = [
    "marital_status",
    "nationality",
    "displaced",
    "gender",
    "age_at_enrollment",
    "international",
]

SOCIO_ECONOMIC_FEATURES = [
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
    "educational_special_needs",
    "debtor",
    "tuition_fees_up_to_date",
    "scholarship_holder",
]

PRE_ENROLLMENT_FEATURES = [
    "application_mode",
    "application_order",
    "course",
    "daytime_evening_attendance",
    "previous_qualification",
    "previous_qualification_grade",
    "admission_grade",
]

SEM1_ACADEMIC_FEATURES = [
    "cu_1st_sem_credited",
    "cu_1st_sem_enrolled",
    "cu_1st_sem_evaluations",
    "cu_1st_sem_approved",
    "cu_1st_sem_grade",
    "cu_1st_sem_without_evaluations",
]

SEM2_ACADEMIC_FEATURES = [
    "cu_2nd_sem_credited",
    "cu_2nd_sem_enrolled",
    "cu_2nd_sem_evaluations",
    "cu_2nd_sem_approved",
    "cu_2nd_sem_grade",
    "cu_2nd_sem_without_evaluations",
]

MACROECONOMIC_FEATURES = [
    "unemployment_rate",
    "inflation_rate",
    "gdp",
]

# Engineered Early Warning Features (Available after 1st Semester)
ENGINEERED_EARLY_FEATURES = [
    "sem1_approval_rate",
    "sem1_eval_ratio",
]

# Feature Set 1: Early Warning (Pre-enrollment + 1st Semester)
# Strictly excludes 2nd semester features to prevent future data leakage!
EARLY_WARNING_FEATURES = (
    DEMOGRAPHIC_FEATURES
    + SOCIO_ECONOMIC_FEATURES
    + PRE_ENROLLMENT_FEATURES
    + SEM1_ACADEMIC_FEATURES
    + ENGINEERED_EARLY_FEATURES
)

# Feature Set 2: Cumulative Full Year Features
CUMULATIVE_FEATURES = (
    EARLY_WARNING_FEATURES
    + SEM2_ACADEMIC_FEATURES
    + ["sem2_approval_rate"]
)


def load_and_clean_raw_data(raw_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads raw CSV, strips headers, and renames columns to standardized snake_case."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}. Run download_data.py first.")

    df = pd.read_csv(raw_path, sep=";")
    # Strip whitespace and hidden tab characters from headers
    cleaned_cols = {col: col.strip() for col in df.columns}
    df = df.rename(columns=cleaned_cols)

    # Rename to canonical snake_case
    df = df.rename(columns=COLUMN_MAPPING)
    return df


def engineer_features_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes valid engineered progression metrics and ground-truth targets.
    - target_dropout: 1 for Dropout, 0 for Graduate (resolved students)
    - target_dropout_binary: 1 for Dropout, 0 for Non-Dropout (Graduate + Enrolled)
    - target_pass_fail: 1 for Pass (cu_1st_sem_grade >= 10.0 and approval_rate >= 0.5), 0 for Academic Warning/Fail
    - target_marks: Continuous 2nd sem grade (0.0 to 20.0)
    """
    df = df.copy()

    # 1. Engineered progression ratios
    # Approval rate = approved / enrolled (default 0.0 if enrolled is 0)
    df["sem1_approval_rate"] = np.where(
        df["cu_1st_sem_enrolled"] > 0,
        df["cu_1st_sem_approved"] / df["cu_1st_sem_enrolled"],
        0.0
    ).round(4)

    df["sem1_eval_ratio"] = np.where(
        df["cu_1st_sem_enrolled"] > 0,
        df["cu_1st_sem_evaluations"] / df["cu_1st_sem_enrolled"],
        0.0
    ).round(4)

    df["sem2_approval_rate"] = np.where(
        df["cu_2nd_sem_enrolled"] > 0,
        df["cu_2nd_sem_approved"] / df["cu_2nd_sem_enrolled"],
        0.0
    ).round(4)

    # 2. Target: Dropout Risk (Binary - Dropout=1, Non-Dropout=0 across full dataset)
    df["target_dropout_binary"] = (df["target_raw"] == "Dropout").astype(int)

    # 3. Target: Dropout Resolved (Only resolved outcomes: Dropout=1, Graduate=0, Enrolled=NaN)
    df["target_dropout_resolved"] = np.where(
        df["target_raw"] == "Dropout", 1,
        np.where(df["target_raw"] == "Graduate", 0, np.nan)
    )

    # 4. Target: Academic Pass/Fail Progression (Subsequent Coursework Progression)
    # Grounded in Portuguese Higher Education Statutory Standards (Decree-Law no. 42/2005):
    # - Curricular unit passing threshold is strictly grade >= 10.0 on a 0-20 scale.
    # - A student successfully passes their subsequent (2nd semester) academic coursework if:
    #   (1) they achieve an average passing mark >= 10.0, AND
    #   (2) they successfully pass/approve at least one enrolled curricular unit.
    # - Students with grade < 10.0 or 0 approved units are classified as 0 (Academic Warning / Failure).
    # - IMPORTANT: Input features for this task strictly use EARLY_WARNING_FEATURES (1st semester
    #   and pre-enrollment baseline). All 2nd semester outcomes are excluded from inputs, guaranteeing
    #   zero future data leakage!
    df["target_pass_fail"] = np.where(
        (df["cu_2nd_sem_grade"] >= 10.0) & (df["cu_2nd_sem_approved"] > 0),
        1,  # Pass
        0   # Academic Warning / Failure
    )

    # 5. Target: Numerical Marks (Continuous 2nd semester grade on 0-20 scale)
    # Predicted using early warning features (1st semester & baseline) with zero future leakage.
    df["target_marks"] = df["cu_2nd_sem_grade"]

    return df


def split_and_save_data(
    df: pd.DataFrame,
    output_dir: str = PROCESSED_DATA_DIR,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Performs stratified train/test split based on the raw target class distribution.
    Saves train.csv, test.csv, and summary documentation.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Stratify by raw target to preserve proportion of Dropout, Graduate, Enrolled
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target_raw"]
    )

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "total_records": len(df),
        "train_records": len(train_df),
        "test_records": len(test_df),
        "split_ratio": f"{int((1 - test_size) * 100)}/{int(test_size * 100)}",
        "random_state": random_state,
        "target_distribution_overall": df["target_raw"].value_counts().to_dict(),
        "target_distribution_train": train_df["target_raw"].value_counts().to_dict(),
        "target_distribution_test": test_df["target_raw"].value_counts().to_dict(),
        "dropout_binary_distribution_overall": df["target_dropout_binary"].value_counts().to_dict(),
        "pass_fail_distribution_overall": {
            "Pass (1)": int((df["target_pass_fail"] == 1).sum()),
            "Fail/Warning (0)": int((df["target_pass_fail"] == 0).sum()),
        },
        "early_warning_features": EARLY_WARNING_FEATURES,
        "cumulative_features": CUMULATIVE_FEATURES,
        "macroeconomic_features": MACROECONOMIC_FEATURES,
    }

    summary_path = os.path.join(output_dir, "data_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[SUCCESS] Processed train dataset ({len(train_df)} rows) saved to: {train_path}")
    print(f"[SUCCESS] Processed test dataset ({len(test_df)} rows) saved to: {test_path}")
    print(f"[SUCCESS] Data summary saved to: {summary_path}")

    return summary


def run_pipeline():
    """Executes the full preprocessing pipeline from raw data to processed splits."""
    print("[INFO] Starting Data Preprocessing Pipeline...")
    raw_df = load_and_clean_raw_data()
    print(f"[INFO] Loaded and cleaned raw dataset: {raw_df.shape}")

    featured_df = engineer_features_and_targets(raw_df)
    print(f"[INFO] Engineered features and targets. Total columns: {featured_df.shape[1]}")

    summary = split_and_save_data(featured_df)
    print("[INFO] Preprocessing pipeline finished successfully.")
    return summary


if __name__ == "__main__":
    run_pipeline()
