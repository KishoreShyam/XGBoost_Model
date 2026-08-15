import pandas as pd
import numpy as np

DATA_PATH = "data/raw/Synthea_Original_Dataset.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)
if "LABEL" in df.columns:
    df.drop(columns=["LABEL"], inplace=True)
riskscore_df = pd.read_csv("data/raw/Synthea_Dataset_with_RiskScore.csv", usecols=["PATIENT_ID", "LABEL"], low_memory=False)
df = df.merge(riskscore_df, on="PATIENT_ID", how="left")
df["LABEL"] = df["LABEL"].isin(["High", "Very High"]).astype(int)

print("=" * 70)
print("TEMPORAL DATA AUDIT")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))

# ------------------------------------------------------------
# 1. Check important temporal columns
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TEMPORAL / IDENTIFIER COLUMNS")
print("=" * 70)

important_columns = [
    "PATIENT_ID",
    "SNAPSHOT_DATE",
    "HISTORY_WINDOW_MONTHS",
    "DATA_SPLIT",
    "LABEL",
]

for column in important_columns:
    if column in df.columns:
        print(f"\n{column}")
        print("dtype:", df[column].dtype)
        print("unique values:", df[column].nunique())
        print("sample:")
        print(df[column].head(10).to_string(index=False))
    else:
        print(f"\n{column} → NOT FOUND")


# ------------------------------------------------------------
# 2. Check current clinical features
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CURRENT CLINICAL FEATURES")
print("=" * 70)

current_features = [
    "CURRENT_CLINICAL_CONCEPT_COUNT",
    "CURRENT_CLINICAL_CONCEPT_DENSITY",
    "CURRENT_CLINICAL_TRUE_COUNT",
    "CURRENT_CLINICAL_OBSERVED_COUNT",
    "CURRENT_CLINICAL_BURDEN_12M",
]

for column in current_features:
    if column not in df.columns:
        print(column, "→ NOT FOUND")
        continue

    print("\n" + column)
    print(
        df[column].describe()
    )


# ------------------------------------------------------------
# 3. Compare current clinical features by label
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CURRENT FEATURES VS LABEL")
print("=" * 70)

existing_current = [
    c for c in current_features
    if c in df.columns
]

if existing_current:
    summary = df.groupby("LABEL")[existing_current].mean()
    print(summary)


# ------------------------------------------------------------
# 4. Calculate correlation with LABEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CORRELATION WITH LABEL")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

correlations = (
    df[numeric_columns]
    .corr()["LABEL"]
    .drop("LABEL")
    .abs()
    .sort_values(ascending=False)
)

print(
    correlations.head(20)
)


# ------------------------------------------------------------
# 5. Check whether past/current values behave logically
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PAST vs CURRENT CHECK")
print("=" * 70)

pairs = [
    (
        "PAST_ENCOUNTER_COUNT_12M",
        "CURRENT_ENCOUNTER_COUNT_12M"
    ),
    (
        "PAST_CONDITION_COUNT_12M",
        "CURRENT_CONDITION_COUNT_12M"
    ),
    (
        "PAST_MEDICATION_COUNT_12M",
        "CURRENT_MEDICATION_COUNT_12M"
    ),
    (
        "PAST_PROCEDURE_COUNT_12M",
        "CURRENT_PROCEDURE_COUNT_12M"
    ),
    (
        "PAST_OBSERVATION_COUNT_12M",
        "CURRENT_OBSERVATION_COUNT_12M"
    ),
]

for past, current in pairs:
    if past not in df.columns or current not in df.columns:
        continue

    difference = (
        df[current] - df[past]
    )

    print(
        f"\n{past} → {current}"
    )

    print(
        "Mean change:",
        round(difference.mean(), 3)
    )

    print(
        "Median change:",
        round(difference.median(), 3)
    )

    print(
        "Current > Past:",
        round(
            (difference > 0).mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Current = Past:",
        round(
            (difference == 0).mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Current < Past:",
        round(
            (difference < 0).mean() * 100,
            2
        ),
        "%"
    )


# ------------------------------------------------------------
# 6. Check duplicate patients
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PATIENT STRUCTURE")
print("=" * 70)

if "PATIENT_ID" in df.columns:
    patient_counts = (
        df["PATIENT_ID"]
        .value_counts()
    )

    print(
        "Unique patients:",
        patient_counts.size
    )

    print(
        "Patients with >1 row:",
        (patient_counts > 1).sum()
    )

    print(
        "Maximum rows per patient:",
        patient_counts.max()
    )


# ------------------------------------------------------------
# 7. Label relationship with clinical features
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LABEL SEPARATION")
print("=" * 70)

for column in existing_current:
    print("\n", column)
    for label in [0, 1]:
        values = df.loc[
            df["LABEL"] == label,
            column
        ]
        print(
            f"LABEL {label}: "
            f"mean={values.mean():.3f}, "
            f"median={values.median():.3f}, "
            f"min={values.min():.3f}, "
            f"max={values.max():.3f}"
        )


print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
