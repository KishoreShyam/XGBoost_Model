import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, brier_score_loss
)


# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "DataSet/Synthea_Dataset_with_RiskScore.csv"
MODEL_DIR = "model_outputs"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_uc09_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.pkl")
IMPUTER_PATH = os.path.join(MODEL_DIR, "median_imputer.pkl")
OUTPUT_DIR = os.path.join(MODEL_DIR, "synthea_riskscore_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# STEP 1 — LOAD DATASET
# ============================================================
print("STEP 1 — LOADING DATASET")
df = pd.read_csv(DATA_PATH, low_memory=False)
print("Rows:", len(df), "| Unique patients:", df["PATIENT_ID"].nunique())


# ============================================================
# STEP 2 — CREATE BINARY GROUND TRUTH
# ============================================================
df["TEST_TARGET"] = df["LABEL"].isin(["High", "Very High"]).astype(int)
print(df["TEST_TARGET"].value_counts(normalize=True).round(4))


# ============================================================
# STEP 2.5 — REPLICATE TRAINING-TIME ENCODING (GENDER one-hot)
# ============================================================
# Must match training exactly, or feature names won't line up.
df = pd.get_dummies(df, columns=["GENDER"], prefix="GENDER", drop_first=True)


# ============================================================
# STEP 3 — RESTRICT TO TEST SPLIT ONLY (held-out, unseen data)
# ============================================================
# Important: evaluating on the same rows used for train/val would
# overstate performance. Use only the rows the model never trained on.
df = df[df["DATA_SPLIT"] == "test"].copy()
print("Test rows:", len(df))


# ============================================================
# STEP 4 — LOAD TRAINED MODEL
# ============================================================
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
imputer = joblib.load(IMPUTER_PATH)
print("Expected features:", len(features))


# ============================================================
# STEP 5 — FEATURE COMPATIBILITY
# ============================================================
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print("Missing features:", missing_features)
    raise ValueError("Dataset is not compatible with model.")
print("All model features are available.")


# ============================================================
# STEP 6 — PREPARE FEATURES
# ============================================================
X = df[features].copy()
y = df["TEST_TARGET"].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X = imputer.transform(X)
print("Feature matrix:", X.shape)


# ============================================================
# STEP 7 — PREDICT
# ============================================================
probabilities = model.predict_proba(X)[:, 1]
predictions = (probabilities >= 0.50).astype(int)


# ============================================================
# STEP 8 — EVALUATE
# ============================================================
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions, zero_division=0)
recall = recall_score(y, predictions, zero_division=0)
f1 = f1_score(y, predictions, zero_division=0)
roc_auc = roc_auc_score(y, probabilities)
pr_auc = average_precision_score(y, probabilities)
brier = brier_score_loss(y, probabilities)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
print(f"Brier:     {brier:.4f}")

print(confusion_matrix(y, predictions))
print(classification_report(y, predictions, target_names=["Low/Moderate", "High/Very High"], zero_division=0))


# ============================================================
# STEP 9 — SAVE PREDICTIONS + METRICS
# ============================================================
comparison = pd.DataFrame({
    "PATIENT_ID": df["PATIENT_ID"],
    "LABEL": df["LABEL"],
    "LABEL_SCORE": df["LABEL_SCORE"],
    "TRUE_BINARY_TARGET": y,
    "MODEL_PROBABILITY": probabilities,
    "MODEL_PREDICTION": predictions
})
comparison.to_csv(os.path.join(OUTPUT_DIR, "synthea_riskscore_predictions.csv"), index=False)

metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier Score"],
    "Value": [accuracy, precision, recall, f1, roc_auc, pr_auc, brier]
})
metrics.to_csv(os.path.join(OUTPUT_DIR, "synthea_riskscore_metrics.csv"), index=False)

print("Saved predictions and metrics to", OUTPUT_DIR)