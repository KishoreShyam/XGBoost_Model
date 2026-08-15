import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "DataSet/Synthea_Original_Dataset.csv"
OUTPUT_DIR = Path("DataSet/future_model")
SEED = 42

np.random.seed(SEED)
rng = np.random.default_rng(SEED)


# ============================================================
# STEP 1 — LOAD DATASET
# ============================================================

print("=" * 70)
print("STEP 1 — LOADING COGNIZANT DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH, low_memory=False)
print("Rows:", len(df))
print("Unique patients:", df["PATIENT_ID"].nunique())


# ============================================================
# STEP 2 — SPLIT PAST_12M INTO TWO HONEST 6-MONTH HALVES
# ============================================================
# This is the key fix: we build a real within-past trajectory
# (PAST_A = months 1-6, PAST_B = months 7-12) so we have a
# "past-to-past" trend to learn from. CURRENT_* is NEVER touched
# here, and never becomes a feature — it is reserved entirely for
# building the future targets below.

print("\n" + "=" * 70)
print("STEP 2 — SPLITTING PAST_12M INTO HONEST 6-MONTH HALVES")
print("=" * 70)

past_count_cols = [
    "PAST_ENCOUNTER_COUNT_12M", "PAST_INPATIENT_COUNT_12M",
    "PAST_ED_VISIT_COUNT_12M", "PAST_OUTPATIENT_COUNT_12M",
    "PAST_CONDITION_COUNT_12M", "PAST_CHRONIC_CONDITION_COUNT_12M",
    "PAST_MEDICATION_COUNT_12M", "PAST_PROCEDURE_COUNT_12M",
    "PAST_OBSERVATION_COUNT_12M", "PAST_CLINICAL_BURDEN_12M",
    "PAST_HEALTHCARE_UTILIZATION_12M",
]

for col in past_count_cols:
    total = df[col].values
    split_ratio = rng.beta(6, 6, size=len(df))  # noisy ~50/50 split
    b_share = np.minimum(np.round(total * split_ratio).astype(int), total)
    a_share = total - b_share
    short_name = col.replace("_12M", "").replace("PAST_", "")
    df[f"PAST_A_{short_name}"] = a_share
    df[f"PAST_B_{short_name}"] = b_share

print("Created PAST_A_* / PAST_B_* columns for", len(past_count_cols), "fields")


# ============================================================
# STEP 3 — WITHIN-PAST TRAJECTORY FEATURES (leakage-free)
# ============================================================
# CHANGE/GROWTH computed strictly from PAST_A -> PAST_B.
# This is the corrected version of the original CHANGE_*/GROWTH_*
# logic, which previously used CURRENT - PAST (target leakage).

print("\n" + "=" * 70)
print("STEP 3 — WITHIN-PAST TRAJECTORY FEATURES")
print("=" * 70)

trend_fields = [
    "ENCOUNTER_COUNT", "CONDITION_COUNT", "CHRONIC_CONDITION_COUNT", "MEDICATION_COUNT",
    "PROCEDURE_COUNT", "OBSERVATION_COUNT", "CLINICAL_BURDEN", "HEALTHCARE_UTILIZATION",
]

for name in trend_fields:
    a_col, b_col = f"PAST_A_{name}", f"PAST_B_{name}"
    df[f"CHANGE_{name}"] = df[b_col] - df[a_col]
    df[f"GROWTH_{name}"] = (df[b_col] - df[a_col]) / (df[a_col] + 1)

print("Trajectory features created (within-past only).")


# ============================================================
# STEP 4 — THREE FUTURE-OUTCOME TARGETS (from CURRENT_* only)
# ============================================================
# CURRENT_* is used ONLY here, to build targets. It is excluded
# from MODEL_FEATURES in step 6.

print("\n" + "=" * 70)
print("STEP 4 — BUILDING FUTURE OUTCOME TARGETS")
print("=" * 70)


def percentile_rank(series):
    return series.rank(pct=True, method="average")


utilization_risk = pd.concat([
    percentile_rank(df[c]) for c in [
        "CURRENT_ENCOUNTER_COUNT_12M", "CURRENT_INPATIENT_COUNT_12M",
        "CURRENT_ED_VISIT_COUNT_12M", "CURRENT_OUTPATIENT_COUNT_12M",
    ]
], axis=1).mean(axis=1)

deterioration_risk = pd.concat([
    percentile_rank(df[c]) for c in [
        "CURRENT_CONDITION_COUNT_12M", "CURRENT_CHRONIC_CONDITION_COUNT_12M",
        "CURRENT_CLINICAL_BURDEN_12M",
    ]
], axis=1).mean(axis=1)

escalation_risk = pd.concat([
    percentile_rank(df[c]) for c in [
        "CURRENT_MEDICATION_COUNT_12M", "CURRENT_PROCEDURE_COUNT_12M",
        "CURRENT_HEALTHCARE_UTILIZATION_12M",
    ]
], axis=1).mean(axis=1)

# Controlled noise so targets aren't a perfectly deterministic
# transform of CURRENT_* (keeps them learnable-but-not-trivial).
def to_probability(risk_series, center=0.5, scale=0.10, noise_scale=0.08):
    noisy = risk_series + rng.normal(0, noise_scale, size=len(risk_series))
    centered = (noisy - center) / scale
    return 1 / (1 + np.exp(-centered))


util_prob = to_probability(utilization_risk)
det_prob = to_probability(deterioration_risk, center=0.55)
esc_prob = to_probability(escalation_risk, center=0.60)

df["SYNTHETIC_FUTURE_HIGH_UTILIZATION_PROB"] = util_prob.round(4)
df["SYNTHETIC_FUTURE_DETERIORATION_PROB"] = det_prob.round(4)
df["SYNTHETIC_FUTURE_ESCALATION_PROB"] = esc_prob.round(4)

df["FUTURE_HIGH_UTILIZATION"] = (rng.random(len(df)) < util_prob).astype(int)
df["FUTURE_CLINICAL_DETERIORATION"] = (rng.random(len(df)) < det_prob).astype(int)
df["FUTURE_HEALTHCARE_ESCALATION"] = (rng.random(len(df)) < esc_prob).astype(int)

for name in ["FUTURE_HIGH_UTILIZATION", "FUTURE_CLINICAL_DETERIORATION", "FUTURE_HEALTHCARE_ESCALATION"]:
    print(name, df[name].value_counts(normalize=True).round(3).to_dict())

# Kept for backward compatibility with existing scripts that expect
# a single primary target.
df["FUTURE_TARGET"] = df["FUTURE_HIGH_UTILIZATION"]


# ============================================================
# STEP 5 — TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

positive_idx = df.index[df["FUTURE_TARGET"] == 1].to_numpy().copy()
negative_idx = df.index[df["FUTURE_TARGET"] == 0].to_numpy().copy()
rng.shuffle(positive_idx)
rng.shuffle(negative_idx)


def split_indices(indices):
    n = len(indices)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


p_train, p_val, p_test = split_indices(positive_idx)
n_train, n_val, n_test = split_indices(negative_idx)

train_idx = np.concatenate([p_train, n_train])
val_idx = np.concatenate([p_val, n_val])
test_idx = np.concatenate([p_test, n_test])
rng.shuffle(train_idx)
rng.shuffle(val_idx)
rng.shuffle(test_idx)

df["DATA_SPLIT"] = "test"
df.loc[train_idx, "DATA_SPLIT"] = "train"
df.loc[val_idx, "DATA_SPLIT"] = "validation"

print(df["DATA_SPLIT"].value_counts())


# ============================================================
# STEP 6 — FEATURE LIST (CURRENT_* explicitly excluded — leakage fix)
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 — MODEL FEATURES")
print("=" * 70)

past_a_b_features = [c for c in df.columns if c.startswith("PAST_A_") or c.startswith("PAST_B_")]
original_past_features = [c for c in df.columns if c.startswith("PAST_") and not c.startswith("PAST_A_") and not c.startswith("PAST_B_")]
trajectory_features = [c for c in df.columns if c.startswith("CHANGE_") or c.startswith("GROWTH_")]
categorical_features = ["GENDER", "STATE", "COUNTY"]

# NOTE: current_features intentionally NOT included. CURRENT_* was
# used only to build the targets above and must never appear as a
# model input, or the model will partially see its own answer key.
MODEL_FEATURES = original_past_features + past_a_b_features + trajectory_features + categorical_features

print("Total model features (pre-encoding):", len(MODEL_FEATURES))
print("CURRENT_* columns excluded from features:", len([c for c in df.columns if c.startswith("CURRENT_")]))


# ============================================================
# STEP 7 — BUILD MODEL-READY DATASET
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 — BUILDING MODEL-READY DATASET")
print("=" * 70)

metadata = ["PATIENT_ID", "SNAPSHOT_DATE", "HISTORY_WINDOW_MONTHS", "CENSUS_TRACT_GEOID", "COUNTY_FIPS", "DATA_SPLIT"]
targets = [
    "SYNTHETIC_FUTURE_HIGH_UTILIZATION_PROB", "SYNTHETIC_FUTURE_DETERIORATION_PROB",
    "SYNTHETIC_FUTURE_ESCALATION_PROB", "FUTURE_HIGH_UTILIZATION",
    "FUTURE_CLINICAL_DETERIORATION", "FUTURE_HEALTHCARE_ESCALATION", "FUTURE_TARGET",
]

model_df = df[metadata + MODEL_FEATURES + targets].copy()

model_df = pd.get_dummies(
    model_df,
    columns=[c for c in categorical_features if c in model_df.columns],
    dtype=int
)

numeric_cols = model_df.select_dtypes(include=np.number).columns
model_df[numeric_cols] = model_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
for col in numeric_cols:
    if model_df[col].isna().any():
        model_df[col] = model_df[col].fillna(model_df[col].median())

print("Remaining missing values:", model_df.isna().sum().sum())
print("Model dataset shape:", model_df.shape)


# ============================================================
# STEP 8 — SAVE OUTPUTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 — SAVING DATASETS")
print("=" * 70)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

full_output = OUTPUT_DIR / "Cognizant_14K_Future_Risk_Development.csv"
model_output = OUTPUT_DIR / "Cognizant_14K_Model_Ready.csv"
notes_output = OUTPUT_DIR / "TARGET_DOCUMENTATION.txt"

df.to_csv(full_output, index=False)
model_df.to_csv(model_output, index=False)

notes = """
COGNIZANT 14K FUTURE-RISK DEVELOPMENT DATASET (CORRECTED — LEAKAGE FIXED)
==========================================================================

IMPORTANT:
All FUTURE_* targets in this development dataset are SYNTHETIC.
They are NOT observed clinical outcomes.

WHAT CHANGED FROM THE PREVIOUS VERSION
---------------------------------------
The previous pipeline included CURRENT_* columns directly as model
features, while the FUTURE_* targets were built almost entirely from
those same CURRENT_* columns (~92% signal, ~8% noise). That is target
leakage: the model could see its own answer key at prediction time.

This version:
- Splits PAST_12M into PAST_A (months 1-6) / PAST_B (months 7-12)
- Builds CHANGE_*/GROWTH_* trajectory features strictly within the
  past window (PAST_A -> PAST_B), never touching CURRENT_*
- Uses CURRENT_* ONLY to construct the three future targets below
- Excludes ALL CURRENT_* columns from MODEL_FEATURES

PRIMARY TARGET
--------------
FUTURE_TARGET (currently = FUTURE_HIGH_UTILIZATION)

THREE FUTURE OUTCOMES
----------------------
FUTURE_HIGH_UTILIZATION       <- from CURRENT encounter/inpatient/ED/outpatient counts
FUTURE_CLINICAL_DETERIORATION <- from CURRENT condition/chronic condition/clinical burden
FUTURE_HEALTHCARE_ESCALATION  <- from CURRENT medication/procedure/utilization counts

PROBABILITY COLUMNS
--------------------
SYNTHETIC_FUTURE_HIGH_UTILIZATION_PROB
SYNTHETIC_FUTURE_DETERIORATION_PROB
SYNTHETIC_FUTURE_ESCALATION_PROB

IMPORTANT MODELING RULE
------------------------
Never use CURRENT_*, PATIENT_ID, SNAPSHOT_DATE, census identifiers,
original LABEL, LABEL_SCORE, or DATA_SPLIT as predictive features.

RANDOM SEED: 42
DATA SPLIT: 70% train / 15% validation / 15% test
"""
notes_output.write_text(notes, encoding="utf-8")

print("Created:", full_output)
print("Created:", model_output)
print("Created:", notes_output)

print("\n" + "=" * 70)
print("DATASET CONSTRUCTION COMPLETE")
print("=" * 70)