import os
import joblib
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

BASELINE_DATA_PATH = "FutureModel/data/Cognizant_14K_Model_Ready.csv"
MODEL_DIR = "FutureModel/outputs"
OUTPUT_DIR = "FutureModel/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Alerts thresholds
Z_SCORE_THRESHOLD = 3.0
RISK_SPIKE_THRESHOLD = 0.20  # absolute probability jump (20%)
CRITICAL_RISK_THRESHOLD = 0.80  # 80% probability

class CalibratedXGBWrapper:
    def __init__(self, base_model, isotonic_regressor):
        self.base_model = base_model
        self.isotonic_regressor = isotonic_regressor

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        calibrated = np.clip(self.isotonic_regressor.predict(raw), 0, 1)
        return np.column_stack([1 - calibrated, calibrated])


# ============================================================
# STEP 1 — LOAD MODELS & TRAINING REFERENCE STATISTICS
# ============================================================

print("=" * 75)
print("INDIVIDUAL PATIENT RISK & ANOMALY MONITORING")
print("=" * 75)

# Load feature names and pipeline files
features = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))

outcomes = ["utilization", "deterioration", "escalation"]
final_models = {}
imputers = {}

# Integrate current risk model if available
if os.path.exists("CurrentModel/outputs/final_current_risk.pkl"):
    print("Integrating current risk model from CurrentModel...")
    final_models["current_risk"] = joblib.load("CurrentModel/outputs/final_current_risk.pkl")
    imputers["current_risk"] = joblib.load("CurrentModel/outputs/imputer_current_risk.pkl")
    outcomes.insert(0, "current_risk")

for o in [x for x in outcomes if x != "current_risk"]:
    final_models[o] = joblib.load(os.path.join(MODEL_DIR, f"final_{o}.pkl"))
    imputers[o] = joblib.load(os.path.join(MODEL_DIR, f"imputer_{o}.pkl"))

# Load baseline dataset to extract reference statistics for Z-score checks
baseline_df = pd.read_csv(BASELINE_DATA_PATH, low_memory=False)
train_df = baseline_df[baseline_df["DATA_SPLIT"] == "train"].copy()

key_clinical_features = [
    "PAST_CHRONIC_CONDITION_COUNT_12M",
    "PAST_CONDITION_COUNT_12M",
    "PAST_MEDICATION_COUNT_12M",
    "PAST_CLINICAL_BURDEN_12M",
    "PAST_ENCOUNTER_COUNT_12M",
    "PAST_HEALTHCARE_UTILIZATION_12M"
]

feature_stats = {}
for feat in key_clinical_features:
    if feat in train_df.columns:
        feature_stats[feat] = {
            "mean": train_df[feat].mean(),
            "std": train_df[feat].std()
        }

print("Loaded model pipeline and calculated baseline training statistics.")

# ============================================================
# STEP 2 — PROCESS CURRENT/DRIFTED DATASET
# ============================================================

def monitor_dataset(data_path, label):
    print(f"\nProcessing {label} from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    
    X = df[features].copy().replace([np.inf, -np.inf], np.nan)
    
    for outcome in outcomes:
        X_imp = imputers[outcome].transform(X)
        df[f"{outcome.upper()}_PROB"] = final_models[outcome].predict_proba(X_imp)[:, 1]
    
    # Outlier Detection (Z-score based)
    outlier_flags = []
    outlier_reasons = []
    
    for idx, row in df.iterrows():
        reasons = []
        is_outlier = False
        for feat in key_clinical_features:
            if feat in df.columns:
                val = row[feat]
                mean = feature_stats[feat]["mean"]
                std = feature_stats[feat]["std"]
                if std > 0:
                    z = (val - mean) / std
                    if abs(z) > Z_SCORE_THRESHOLD:
                        is_outlier = True
                        reasons.append(f"{feat}={val} (Z={z:.1f})")
        
        outlier_flags.append(is_outlier)
        outlier_reasons.append("; ".join(reasons) if is_outlier else "")
        
    df["IS_OUTLIER"] = outlier_flags
    df["OUTLIER_DETAILS"] = outlier_reasons

    # Temporal Risk Spike Detection
    df = df.sort_values(by=["PATIENT_ID", "SNAPSHOT_DATE"]).reset_index(drop=True)
    
    spike_flags = []
    spike_details = []
    
    grouped = df.groupby("PATIENT_ID")
    diff_records = {}
    for pid, group in grouped:
        if len(group) > 1:
            group_sorted = group.sort_values("SNAPSHOT_DATE")
            for i in range(1, len(group_sorted)):
                prev_row = group_sorted.iloc[i-1]
                curr_row = group_sorted.iloc[i]
                curr_idx = group_sorted.index[i]
                
                spikes = []
                for outcome in outcomes:
                    prob_col = f"{outcome.upper()}_PROB"
                    diff = curr_row[prob_col] - prev_row[prob_col]
                    if diff > RISK_SPIKE_THRESHOLD:
                        spikes.append(f"{outcome.upper()} jump: +{diff*100:.1f}%")
                
                if spikes:
                    diff_records[curr_idx] = " & ".join(spikes)

    for idx in df.index:
        if idx in diff_records:
            spike_flags.append(True)
            spike_details.append(diff_records[idx])
        else:
            spike_flags.append(False)
            spike_details.append("")
            
    df["IS_RISK_SPIKE"] = spike_flags
    df["RISK_SPIKE_DETAILS"] = spike_details

    # Threshold Alerting (Critical Levels)
    critical_flags = []
    critical_details = []
    
    for idx, row in df.iterrows():
        crits = []
        is_critical = False
        for outcome in outcomes:
            prob_col = f"{outcome.upper()}_PROB"
            if row[prob_col] > CRITICAL_RISK_THRESHOLD:
                is_critical = True
                crits.append(f"{outcome.upper()}={row[prob_col]*100:.1f}%")
        
        critical_flags.append(is_critical)
        critical_details.append("; ".join(crits) if is_critical else "")
        
    df["IS_CRITICAL_ALERT"] = critical_flags
    df["CRITICAL_ALERT_DETAILS"] = critical_details

    # Filter and Save Alerts Report
    alerts_df = df[df["IS_OUTLIER"] | df["IS_RISK_SPIKE"] | df["IS_CRITICAL_ALERT"]].copy()
    
    columns_to_keep = [
        "PATIENT_ID", "SNAPSHOT_DATE", 
        "CURRENT_RISK_PROB", "UTILIZATION_PROB", "DETERIORATION_PROB", "ESCALATION_PROB",
        "IS_OUTLIER", "OUTLIER_DETAILS",
        "IS_RISK_SPIKE", "RISK_SPIKE_DETAILS",
        "IS_CRITICAL_ALERT", "CRITICAL_ALERT_DETAILS"
    ]
    columns_to_keep = [c for c in columns_to_keep if c in alerts_df.columns]
    alerts_df = alerts_df[columns_to_keep]
    
    output_filename = f"individual_alerts_{label.lower().replace(' ', '_')}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    alerts_df.to_csv(output_path, index=False)
    
    print(f"Audited {len(df)} patient records.")
    print(f"Identified {len(alerts_df)} patients triggering alerts.")
    print(f" - Outliers flagged: {df['IS_OUTLIER'].sum()}")
    print(f" - Risk Spikes flagged: {df['IS_RISK_SPIKE'].sum()}")
    print(f" - Critical Risk alerts: {df['IS_CRITICAL_ALERT'].sum()}")
    print(f"Saved patient alert report to: {output_path}")
    return df

test_results = monitor_dataset(BASELINE_DATA_PATH, "Test Dataset")
drifted_results = monitor_dataset("FutureModel/data/Cognizant_14K_Drifted.csv", "Drifted Dataset")

print("\n" + "=" * 75)
print("MONITORING COMPLETE")
print("=" * 75)
