import os
import numpy as np
import pandas as pd
import joblib

DATA_PATH = "data/processed/Cognizant_14K_Model_Ready.csv"
OUTPUT_PATH = "data/drifted/Cognizant_14K_Drifted.csv"
MODEL_DIR = "outputs/models"

print("=" * 70)
print("CREATING DRIFTED DATASET")
print("=" * 70)

# Load data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Source dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)
test_df = df[df["DATA_SPLIT"] == "test"].copy()

print(f"Loaded test split: {len(test_df)} rows")

# Apply synthetic population drift
np.random.seed(42)
rng = np.random.default_rng(42)

drifted_df = test_df.copy()

# Features to drift (simulating sicker population)
drift_multipliers = {
    "PAST_CHRONIC_CONDITION_COUNT_12M": 1.6,
    "PAST_CONDITION_COUNT_12M": 1.5,
    "PAST_MEDICATION_COUNT_12M": 1.7,
    "PAST_CLINICAL_BURDEN_12M": 1.6,
    "PAST_ENCOUNTER_COUNT_12M": 1.4,
    "PAST_HEALTHCARE_UTILIZATION_12M": 1.5,
}

for col, mult in drift_multipliers.items():
    if col in drifted_df.columns:
        noise = rng.normal(1.0, 0.05, size=len(drifted_df))
        drifted_df[col] = np.round(drifted_df[col] * mult * noise).clip(lower=0)

# Recalculate A/B splits and trend columns to maintain internal consistency
past_count_map = {
    "PAST_ENCOUNTER_COUNT_12M": "ENCOUNTER_COUNT",
    "PAST_CONDITION_COUNT_12M": "CONDITION_COUNT",
    "PAST_CHRONIC_CONDITION_COUNT_12M": "CHRONIC_CONDITION_COUNT",
    "PAST_MEDICATION_COUNT_12M": "MEDICATION_COUNT",
    "PAST_CLINICAL_BURDEN_12M": "CLINICAL_BURDEN",
    "PAST_HEALTHCARE_UTILIZATION_12M": "HEALTHCARE_UTILIZATION",
}

for full_col, short in past_count_map.items():
    total = drifted_df[full_col].values
    split_ratio = rng.beta(6, 6, size=len(drifted_df))
    b_share = np.minimum(np.round(total * split_ratio).astype(int), total)
    a_share = total - b_share
    
    if f"PAST_A_{short}" in drifted_df.columns:
        drifted_df[f"PAST_A_{short}"] = a_share
        drifted_df[f"PAST_B_{short}"] = b_share
    if f"CHANGE_{short}" in drifted_df.columns:
        drifted_df[f"CHANGE_{short}"] = b_share - a_share
        drifted_df[f"GROWTH_{short}"] = (b_share - a_share) / (a_share + 1)

# Set the data split flag to denote drifted data
drifted_df["DATA_SPLIT"] = "drifted"

# Save the dataset
drifted_df.to_csv(OUTPUT_PATH, index=False)
print(f"Drifted dataset successfully created and saved to: {OUTPUT_PATH}")
print(f"Total drifted rows: {len(drifted_df)}")
print("=" * 70)
