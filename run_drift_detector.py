import os
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# ============================================================
# CONFIGURATION
# ============================================================
BASELINE_PATH = "DataSet/future_model/Cognizant_14K_Model_Ready.csv"
MODEL_DIR = "model_outputs_multi"
PSI_CRITICAL = 0.25

class CalibratedXGBWrapper:
    def __init__(self, base_model, isotonic_regressor):
        self.base_model = base_model
        self.isotonic_regressor = isotonic_regressor
    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        calibrated = np.clip(self.isotonic_regressor.predict(raw), 0, 1)
        return np.column_stack([1 - calibrated, calibrated])

def calculate_psi(reference, current, bins=10):
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_drift_detector.py <path_to_dataset_csv>")
        sys.exit(1)
        
    target_path = sys.argv[1]
    if not os.path.exists(target_path):
        print(f"Error: Target dataset file not found at '{target_path}'")
        sys.exit(1)
        
    print("=" * 75)
    print(f"RUNNING DRIFT DETECTOR ON: {target_path}")
    print("=" * 75)
    
    # Load reference training data
    df_base = pd.read_csv(BASELINE_PATH, low_memory=False)
    train_df = df_base[df_base["DATA_SPLIT"] == "train"].copy()
    
    # Load current target data
    curr_df = pd.read_csv(target_path, low_memory=False)
    # If the user passed the main file, filter to test split (representing the clean dataset)
    if target_path == BASELINE_PATH:
        curr_df = curr_df[curr_df["DATA_SPLIT"] == "test"].copy()
        print("Detected main dataset path: Auditing the clean 'test' split.")
        
    features = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))
    numeric_features = [f for f in features if f in train_df.columns and pd.api.types.is_numeric_dtype(train_df[f])]
    
    critical_features = []
    
    for feat in numeric_features:
        psi = calculate_psi(train_df[feat], curr_df[feat])
        if psi > PSI_CRITICAL:
            ks_stat, ks_pval = ks_2samp(train_df[feat].dropna(), curr_df[feat].dropna())
            critical_features.append((feat, psi, ks_pval))
            
    print("\n--- RESULTS ---")
    print(f"Baseline (Train) Size: {len(train_df)} rows")
    print(f"Target Dataset Size:   {len(curr_df)} rows")
    print(f"Features Monitored:    {len(numeric_features)}")
    
    if len(critical_features) > 0:
        print("\n🚨 VERDICT: DATA DRIFT DETECTED!")
        print(f"Found {len(critical_features)} features with CRITICAL population shift (PSI > 0.25):")
        for feat, psi, pval in sorted(critical_features, key=lambda x: x[1], reverse=True):
            print(f" - {feat:<40} | PSI: {psi:.4f} | KS p-value: {pval:.6f}")
    else:
        print("\n✅ VERDICT: NO DRIFT DETECTED (Data distributions are stable).")
        
    print("=" * 75)

if __name__ == "__main__":
    main()
