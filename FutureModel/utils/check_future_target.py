import pandas as pd
import numpy as np

DATA_PATH = "FutureModel/data/Cognizant_14K_Model_Ready.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

TARGET = "FUTURE_TARGET"

print("=" * 70)
print("FUTURE TARGET LEAKAGE CHECK")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nTarget distribution:")
print(df[TARGET].value_counts())
print("\nTarget percentage:")
print(df[TARGET].value_counts(normalize=True).mul(100).round(2))

print("\n" + "=" * 70)
print("NUMERIC CORRELATION WITH TARGET")
print("=" * 70)

numeric = df.select_dtypes(include=np.number)
correlation = numeric.corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
print(correlation.head(20))

if correlation.iloc[0] > 0.90:
    print(f"\n*** WARNING: {correlation.index[0]} has correlation {correlation.iloc[0]:.3f} "
          "with the target. This is a strong leakage signal — investigate before training. ***")

print("\n" + "=" * 70)
print("STRUCTURAL LEAKAGE CHECK (CURRENT_* / CHANGE_* / GROWTH_*)")
print("=" * 70)

current_cols_present = [c for c in df.columns if c.startswith("CURRENT_")]
if current_cols_present:
    print("*** LEAKAGE DETECTED: CURRENT_* columns are present in the model-ready "
          "dataset. These were used to build the FUTURE_* targets and must NEVER "
          "be used as features. ***")
    for c in current_cols_present:
        print(" -", c)
else:
    print("OK: No CURRENT_* columns found in the feature set.")

suspect_trend_cols = [
    c for c in df.columns
    if (c.startswith("CHANGE_") or c.startswith("GROWTH_"))
]
print(f"\nCHANGE_*/GROWTH_* columns present ({len(suspect_trend_cols)}):")
for c in suspect_trend_cols:
    print(" -", c)
print("Confirm manually these were built from PAST_A -> PAST_B only, not CURRENT_* - PAST_*.")

print("\n" + "=" * 70)
print("NAME-PATTERN LEAKAGE CHECK (secondary)")
print("=" * 70)

suspicious = [
    column for column in df.columns
    if ("FUTURE" in column.upper() and column != TARGET)
    or "RISK_SCORE" in column.upper()
    or "PROB" in column.upper()
]
for column in suspicious:
    print("-", column)

print("\n" + "=" * 70)
print("TARGET BY DATA SPLIT")
print("=" * 70)

print(pd.crosstab(df["DATA_SPLIT"], df[TARGET], normalize="index").round(4))

metadata = [
    "PATIENT_ID", "SNAPSHOT_DATE", "HISTORY_WINDOW_MONTHS",
    "CENSUS_TRACT_GEOID", "COUNTY_FIPS", "DATA_SPLIT",
]

target_columns = [
    "FUTURE_TARGET", "FUTURE_HIGH_UTILIZATION",
    "FUTURE_CLINICAL_DETERIORATION", "FUTURE_HEALTHCARE_ESCALATION",
    "SYNTHETIC_FUTURE_HIGH_UTILIZATION_PROB",
    "SYNTHETIC_FUTURE_DETERIORATION_PROB",
    "SYNTHETIC_FUTURE_ESCALATION_PROB",
]

excluded = set(metadata + target_columns + current_cols_present)
features = [column for column in df.columns if column not in excluded]

print("\n" + "=" * 70)
print("FINAL MODEL FEATURES")
print("=" * 70)
print("Feature count:", len(features))
for feature in features:
    print("-", feature)

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
