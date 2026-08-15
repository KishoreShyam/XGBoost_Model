import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "data/processed/Cognizant_14K_Model_Ready.csv"
MODEL_DIR = "outputs/models"

class CalibratedXGBWrapper:
    """Class wrapper to load final calibrated models successfully."""
    def __init__(self, base_model, isotonic_regressor):
        self.base_model = base_model
        self.isotonic_regressor = isotonic_regressor

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        calibrated = np.clip(self.isotonic_regressor.predict(raw), 0, 1)
        return np.column_stack([1 - calibrated, calibrated])


print("=" * 75)
print("EVALUATING MULTI-TARGET FUTURE & CURRENT RISK MODELS")
print("=" * 75)

# Load data and target features
df = pd.read_csv(DATA_PATH, low_memory=False)
test_df = df[df["DATA_SPLIT"] == "test"].copy()

print(f"Loaded test split size: {len(test_df)} rows")

features = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))
outcomes = ["current_risk", "utilization", "deterioration", "escalation"]

target_map = {
    "current_risk": "CURRENT_RISK",
    "utilization": "FUTURE_HIGH_UTILIZATION",
    "deterioration": "FUTURE_CLINICAL_DETERIORATION",
    "escalation": "FUTURE_HEALTHCARE_ESCALATION"
}

# Run evaluations
for outcome in outcomes:
    model_path = os.path.join(MODEL_DIR, f"final_{outcome}.pkl")
    imputer_path = os.path.join(MODEL_DIR, f"imputer_{outcome}.pkl")
    
    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    
    target_col = target_map[outcome]
    y_true = test_df[target_col].astype(int).values
    
    # Process features
    X = test_df[features].copy().replace([np.inf, -np.inf], np.nan)
    X_imp = imputer.transform(X)
    
    # Predict
    probs = model.predict_proba(X_imp)[:, 1]
    preds = (probs >= 0.50).astype(int)
    
    # Compute metrics
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    
    print(f"\nTarget: {target_col.upper()}")
    print("-" * 50)
    print(f"  Accuracy:          {acc:.4f} ({acc*100:.2f}%)")
    print(f"  ROC-AUC Score:     {auc:.4f}")
    print(f"  F1 Score:          {f1:.4f}")
    print(f"  Precision:         {prec:.4f}")
    print(f"  Recall (Sens):     {rec:.4f}")
    print(f"  Brier Score Loss:  {brier:.4f}")
    print("-" * 50)

print("\n" + "=" * 75)
print("EVALUATION COMPLETE")
print("=" * 75)
