import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "FutureModel/outputs/multi_target_calibrated_predictions.csv"
OUTPUT_DIR = "FutureModel/outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 70)
print("STEP 1 — LOADING PREDICTIONS")
print("=" * 70)

df = pd.read_csv(
    INPUT_PATH
)

print(
    "Patients:",
    len(df)
)

# ============================================================
# INTEGRATE CURRENT RISK IF AVAILABLE
# ============================================================
current_preds_path = "CurrentModel/outputs/calibrated_predictions.csv"
if os.path.exists(current_preds_path):
    print("Integrating current risk predictions from CurrentModel...")
    current_df = pd.read_csv(current_preds_path)
    df = df.merge(current_df[["PATIENT_ID", "CURRENT_RISK_CALIBRATED_PERCENT"]], on="PATIENT_ID", how="left")
    df["CURRENT_RISK_PROBABILITY"] = df["CURRENT_RISK_CALIBRATED_PERCENT"]
else:
    print("CurrentModel predictions not found. Defaulting current risk to 0.0.")
    df["CURRENT_RISK_PROBABILITY"] = 0.0

# ============================================================
# PROBABILITY COLUMNS
# ============================================================

UTILIZATION_ORIGINAL = "FUTURE_HIGH_UTILIZATION_ORIGINAL_PROB"
DETERIORATION = "FUTURE_CLINICAL_DETERIORATION_CALIBRATED_PERCENT"
ESCALATION = "FUTURE_HEALTHCARE_ESCALATION_CALIBRATED_PERCENT"

df["UTILIZATION_PROBABILITY"] = df[UTILIZATION_ORIGINAL] * 100
df["DETERIORATION_PROBABILITY"] = df[DETERIORATION]
df["ESCALATION_PROBABILITY"] = df[ESCALATION]


# ============================================================
# RISK CATEGORY FUNCTION
# ============================================================

def risk_category(probability):
    if probability < 20:
        return "VERY LOW"
    elif probability < 40:
        return "LOW"
    elif probability < 60:
        return "MODERATE"
    elif probability < 80:
        return "HIGH"
    else:
        return "VERY HIGH"


# ============================================================
# STEP 2 — INDIVIDUAL RISK CATEGORIES
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — INDIVIDUAL RISK CATEGORIES")
print("=" * 70)

df["CURRENT_RISK_LEVEL"] = df["CURRENT_RISK_PROBABILITY"].apply(risk_category)
df["UTILIZATION_RISK"] = df["UTILIZATION_PROBABILITY"].apply(risk_category)
df["DETERIORATION_RISK"] = df["DETERIORATION_PROBABILITY"].apply(risk_category)
df["ESCALATION_RISK"] = df["ESCALATION_PROBABILITY"].apply(risk_category)

# ============================================================
# STEP 3 — HIGHEST RISK
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — HIGHEST FUTURE RISK")
print("=" * 70)

risk_columns = [
    "UTILIZATION_PROBABILITY",
    "DETERIORATION_PROBABILITY",
    "ESCALATION_PROBABILITY"
]

df["HIGHEST_RISK_PROBABILITY"] = df[risk_columns].max(axis=1)


def highest_risk_type(row):
    values = {
        "HIGH_UTILIZATION": row["UTILIZATION_PROBABILITY"],
        "CLINICAL_DETERIORATION": row["DETERIORATION_PROBABILITY"],
        "HEALTHCARE_ESCALATION": row["ESCALATION_PROBABILITY"]
    }
    return max(values, key=values.get)


df["HIGHEST_RISK_TYPE"] = df.apply(highest_risk_type, axis=1)

# ============================================================
# STEP 4 — COUNT ELEVATED DOMAINS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — MULTI-DOMAIN RISK")
print("=" * 70)

df["ELEVATED_RISK_DOMAINS"] = (
        (df["UTILIZATION_PROBABILITY"] >= 50).astype(int) +
        (df["DETERIORATION_PROBABILITY"] >= 50).astype(int) +
        (df["ESCALATION_PROBABILITY"] >= 50).astype(int)
)


def overall_risk_profile(row):
    count = row["ELEVATED_RISK_DOMAINS"]
    current_high = row["CURRENT_RISK_PROBABILITY"] >= 60
    current_risk_prefix = "CURRENT CRITICAL RISK & " if current_high else ""
    if count == 0:
        if current_high:
            return "CURRENT CRITICAL RISK & LOW FUTURE RISK"
        return "LOW FUTURE RISK PROFILE"
    elif count == 1:
        return current_risk_prefix + "SINGLE-DOMAIN ELEVATED RISK"
    elif count == 2:
        return current_risk_prefix + "MULTI-DOMAIN ELEVATED RISK"
    else:
        return current_risk_prefix + "MULTI-DOMAIN HIGH RISK"


df["RISK_PROFILE"] = df.apply(overall_risk_profile, axis=1)

# ============================================================
# STEP 5 — RISK SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — RISK PROFILE SUMMARY")
print("=" * 70)

print(df["RISK_PROFILE"].value_counts())

print("\nHighest risk domain:")
print(df["HIGHEST_RISK_TYPE"].value_counts())

# ============================================================
# STEP 6 — DISPLAY SAMPLE PATIENTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 — SAMPLE PATIENT RISK PROFILES")
print("=" * 70)

display_columns = [
    "PATIENT_ID",
    "CURRENT_RISK_PROBABILITY",
    "UTILIZATION_PROBABILITY",
    "DETERIORATION_PROBABILITY",
    "ESCALATION_PROBABILITY",
    "CURRENT_RISK_LEVEL",
    "UTILIZATION_RISK",
    "DETERIORATION_RISK",
    "ESCALATION_RISK",
    "HIGHEST_RISK_PROBABILITY",
    "HIGHEST_RISK_TYPE",
    "ELEVATED_RISK_DOMAINS",
    "RISK_PROFILE"
]

print(
    df[display_columns]
    .head(10)
    .to_string(
        index=False
    )
)

# ============================================================
# STEP 7 — SAVE
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 — SAVING FUTURE RISK ENGINE OUTPUT")
print("=" * 70)

output_path = f"{OUTPUT_DIR}/future_risk_profiles.csv"
df[display_columns].to_csv(output_path, index=False)
print("Saved:", output_path)

complete_output = f"{OUTPUT_DIR}/future_risk_engine_complete.csv"
df.to_csv(complete_output, index=False)
print("Saved:", complete_output)

print("\n" + "=" * 70)
print("FUTURE RISK ENGINE COMPLETE")
print("=" * 70)
