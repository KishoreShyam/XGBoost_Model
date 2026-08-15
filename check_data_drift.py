import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "DataSet/future_model/Cognizant_14K_Model_Ready.csv"
MODEL_DIR = "model_outputs_multi"
OUTPUT_DIR = os.path.join(MODEL_DIR, "drift_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PSI_WARNING = 0.10   # PSI thresholds are an industry-standard rule of thumb:
PSI_CRITICAL = 0.25  # <0.10 stable, 0.10-0.25 moderate shift, >0.25 major shift

np.random.seed(42)
rng = np.random.default_rng(42)


class CalibratedXGBWrapper:
    """Must match the class definition in train_multi_target.py exactly —
    joblib needs this class importable under the same name to unpickle
    final_*.pkl files that used isotonic calibration."""
    def __init__(self, base_model, isotonic_regressor):
        self.base_model = base_model
        self.isotonic_regressor = isotonic_regressor

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        calibrated = np.clip(self.isotonic_regressor.predict(raw), 0, 1)
        return np.column_stack([1 - calibrated, calibrated])


# ============================================================
# STEP 1 — LOAD DATA + MODEL ARTIFACTS
# ============================================================

print("=" * 70)
print("STEP 1 — LOADING DATA AND MODELS")
print("=" * 70)

df = pd.read_csv(DATA_PATH, low_memory=False)
features = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))

train_df = df[df["DATA_SPLIT"] == "train"].copy()
test_df = df[df["DATA_SPLIT"] == "test"].copy()

print("Train rows (reference/baseline):", len(train_df))
print("Test rows (current batch, should show minimal drift):", len(test_df))
print("Features monitored:", len(features))

outcomes = ["utilization", "deterioration", "escalation"]
models = {o: joblib.load(os.path.join(MODEL_DIR, f"xgb_{o}.pkl")) for o in outcomes}
final_models = {o: joblib.load(os.path.join(MODEL_DIR, f"final_{o}.pkl")) for o in outcomes}
imputers = {o: joblib.load(os.path.join(MODEL_DIR, f"imputer_{o}.pkl")) for o in outcomes}


# ============================================================
# STEP 2 — BUILD A SYNTHETICALLY DRIFTED BATCH
# ============================================================
# Simulate a population shift: an older, sicker incoming cohort with
# meaningfully higher chronic condition burden, more medications, and
# more encounters than the training population. This should trigger
# clear drift signals so we can prove the detector actually works.

print("\n" + "=" * 70)
print("STEP 2 — GENERATING SYNTHETICALLY DRIFTED BATCH")
print("=" * 70)

drifted_df = test_df.sample(n=1500, random_state=42, replace=True).reset_index(drop=True)

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

# Re-derive PAST_A/PAST_B halves and CHANGE/GROWTH so drifted features
# stay internally consistent rather than contradicting each other.
past_count_map = {
    "PAST_ENCOUNTER_COUNT_12M": "ENCOUNTER_COUNT", "PAST_CONDITION_COUNT_12M": "CONDITION_COUNT",
    "PAST_CHRONIC_CONDITION_COUNT_12M": "CHRONIC_CONDITION_COUNT", "PAST_MEDICATION_COUNT_12M": "MEDICATION_COUNT",
    "PAST_CLINICAL_BURDEN_12M": "CLINICAL_BURDEN", "PAST_HEALTHCARE_UTILIZATION_12M": "HEALTHCARE_UTILIZATION",
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

print(f"Drifted batch created: {len(drifted_df)} patients")
print("Simulated shift: older/sicker incoming cohort (chronic conditions, "
      "medications, clinical burden, encounters all elevated).")


# ============================================================
# STEP 3 — POPULATION STABILITY INDEX (PSI) PER FEATURE
# ============================================================

def calculate_psi(reference, current, bins=10):
    """PSI compares the distribution of a feature between a reference
    (training) population and a current population, using shared
    quantile bins from the reference set."""
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if reference.nunique() < 2 or current.nunique() < 2:
        return 0.0

    breakpoints = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(breakpoints) < 3:
        return 0.0
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)

    ref_counts = np.where(ref_counts == 0, 1e-6, ref_counts)
    cur_counts = np.where(cur_counts == 0, 1e-6, cur_counts)

    psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))
    return psi


def run_feature_drift_report(reference_df, current_df, label):
    print("\n" + "=" * 70)
    print(f"FEATURE DRIFT REPORT — {label}")
    print("=" * 70)

    numeric_features = [f for f in features if f in reference_df.columns and pd.api.types.is_numeric_dtype(reference_df[f])]

    rows = []
    for feat in numeric_features:
        psi = calculate_psi(reference_df[feat], current_df[feat])
        ks_stat, ks_pval = ks_2samp(reference_df[feat].dropna(), current_df[feat].dropna())
        status = "CRITICAL" if psi > PSI_CRITICAL else ("WARNING" if psi > PSI_WARNING else "STABLE")
        rows.append({"feature": feat, "psi": round(psi, 4), "ks_stat": round(ks_stat, 4),
                      "ks_pvalue": round(ks_pval, 6), "status": status})

    report = pd.DataFrame(rows).sort_values("psi", ascending=False)
    print(report.head(15).to_string(index=False))

    n_critical = (report["status"] == "CRITICAL").sum()
    n_warning = (report["status"] == "WARNING").sum()
    print(f"\n{n_critical} features CRITICAL, {n_warning} features WARNING, "
          f"{len(report) - n_critical - n_warning} STABLE (out of {len(report)})")

    report.to_csv(os.path.join(OUTPUT_DIR, f"feature_drift_{label.lower().replace(' ', '_')}.csv"), index=False)
    return report


test_report = run_feature_drift_report(train_df, test_df, "Train vs Test (baseline)")
drift_report = run_feature_drift_report(train_df, drifted_df, "Train vs Drifted Batch")


# ============================================================
# STEP 4 — PREDICTION DRIFT (per model)
# ============================================================

def get_predictions(data_df, outcome):
    X = data_df[features].copy().replace([np.inf, -np.inf], np.nan)
    X_imp = imputers[outcome].transform(X)
    return final_models[outcome].predict_proba(X_imp)[:, 1]


print("\n" + "=" * 70)
print("STEP 4 — PREDICTION DRIFT (model output distributions)")
print("=" * 70)

prediction_drift_rows = []
for outcome in outcomes:
    train_preds = get_predictions(train_df, outcome)
    test_preds = get_predictions(test_df, outcome)
    drifted_preds = get_predictions(drifted_df, outcome)

    psi_test = calculate_psi(train_preds, test_preds)
    psi_drifted = calculate_psi(train_preds, drifted_preds)
    ks_test = ks_2samp(train_preds, test_preds)
    ks_drifted = ks_2samp(train_preds, drifted_preds)

    print(f"\n{outcome.upper()}")
    print(f"  Train mean prob:   {train_preds.mean():.4f}")
    print(f"  Test mean prob:    {test_preds.mean():.4f}  | PSI vs train: {psi_test:.4f}  | KS p-value: {ks_test.pvalue:.6f}")
    print(f"  Drifted mean prob: {drifted_preds.mean():.4f}  | PSI vs train: {psi_drifted:.4f}  | KS p-value: {ks_drifted.pvalue:.6f}")

    prediction_drift_rows.append({
        "outcome": outcome, "train_mean_prob": train_preds.mean(),
        "test_mean_prob": test_preds.mean(), "test_psi": psi_test, "test_ks_pvalue": ks_test.pvalue,
        "drifted_mean_prob": drifted_preds.mean(), "drifted_psi": psi_drifted, "drifted_ks_pvalue": ks_drifted.pvalue,
    })

pd.DataFrame(prediction_drift_rows).to_csv(os.path.join(OUTPUT_DIR, "prediction_drift_summary.csv"), index=False)


# ============================================================
# STEP 5 — OVERALL VERDICT
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — OVERALL DRIFT VERDICT")
print("=" * 70)

test_critical = (test_report["status"] == "CRITICAL").sum()
drift_critical = (drift_report["status"] == "CRITICAL").sum()

print(f"\nBaseline (Train vs Test):        {test_critical} critical features "
      f"-> {'DRIFT DETECTED (unexpected!)' if test_critical > 0 else 'NO DRIFT (expected — same source population)'}")
print(f"Simulated drift (Train vs Drifted): {drift_critical} critical features "
      f"-> {'DRIFT DETECTED (correctly caught)' if drift_critical > 0 else 'NO DRIFT (detector may be too lenient — investigate)'}")

print("\nDrift reports saved to:", OUTPUT_DIR)
print("=" * 70)
print("DRIFT CHECK COMPLETE")
print("=" * 70)