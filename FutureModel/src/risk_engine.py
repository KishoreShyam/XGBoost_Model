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

df["FUTURE_RISK_PROBABILITY"] = df["FUTURE_TARGET_CALIBRATED_PERCENT"]


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
df["FUTURE_RISK_LEVEL"] = df["FUTURE_RISK_PROBABILITY"].apply(risk_category)

# ============================================================
# STEP 4 — OVERALL RISK PROFILE
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — OVERALL RISK PROFILING")
print("=" * 70)


def overall_risk_profile(row):
    current_high = row["CURRENT_RISK_PROBABILITY"] >= 60
    future_high = row["FUTURE_RISK_PROBABILITY"] >= 50
    
    current_prefix = "CURRENT CRITICAL RISK" if current_high else "CURRENT LOW RISK"
    future_suffix = "HIGH FUTURE RISK" if future_high else "LOW FUTURE RISK"
    
    return f"{current_prefix} & {future_suffix}"


df["RISK_PROFILE"] = df.apply(overall_risk_profile, axis=1)

# ============================================================
# STEP 5 — RISK SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — RISK PROFILE SUMMARY")
print("=" * 70)

print(df["RISK_PROFILE"].value_counts())

# ============================================================
# STEP 6 — DISPLAY SAMPLE PATIENTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 — SAMPLE PATIENT RISK PROFILES")
print("=" * 70)

display_columns = [
    "PATIENT_ID",
    "CURRENT_RISK_PROBABILITY",
    "FUTURE_RISK_PROBABILITY",
    "CURRENT_RISK_LEVEL",
    "FUTURE_RISK_LEVEL",
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
