import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, brier_score_loss, ConfusionMatrixDisplay
)

# Config
TRAIN_PATH = "DataSet/ML_Synthea_Training_Dataset.csv"
TEST_PATH = "DataSet/Synthea_Dataset_with_RiskScore.csv"
MODEL_DIR = "model_outputs"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_uc09_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.pkl")
IMPUTER_PATH = os.path.join(MODEL_DIR, "median_imputer.pkl")
OUTPUT_DIR = os.path.join(MODEL_DIR, "synthea_riskscore_reconstructed_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("STEP 1: Loading Datasets...")
df_train = pd.read_csv(TRAIN_PATH, low_memory=False)
df_test = pd.read_csv(TEST_PATH, low_memory=False)

print(f"Train set rows: {len(df_train)}")
print(f"Test set rows: {len(df_test)}")

# Identify numeric columns common to both datasets for prediction
exclude_cols = [
    "CURRENT_CLINICAL_CONCEPT_COUNT", "CURRENT_CLINICAL_CONCEPT_DENSITY",
    "CURRENT_CLINICAL_TRUE_COUNT", "CURRENT_CLINICAL_OBSERVED_COUNT",
    "LABEL", "LABEL_SCORE", "SCC", "PATIENT_ID", "SNAPSHOT_DATE", "DATA_SPLIT",
    "GENDER", "STATE", "COUNTY"
]
common_numeric_cols = [
    col for col in df_train.select_dtypes(include=np.number).columns
    if col in df_test.columns and col not in exclude_cols
]

print(f"Using {len(common_numeric_cols)} numeric columns to reconstruct missing features.")

# Fit ExtraTrees Regressors on training set
print("\nSTEP 2: Training helper regressors on training set...")
X_train_ref = df_train[common_numeric_cols].fillna(df_train[common_numeric_cols].median())

et_concept = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
et_concept.fit(X_train_ref, df_train["CURRENT_CLINICAL_CONCEPT_COUNT"])
print("Concept Count Regressor trained.")

et_true = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
et_true.fit(X_train_ref, df_train["CURRENT_CLINICAL_TRUE_COUNT"])
print("True Count Regressor trained.")

# Reconstruct features on test set
print("\nSTEP 3: Reconstructing missing features on test dataset...")
X_test_ref = df_test[common_numeric_cols].fillna(df_train[common_numeric_cols].median())

df_test["CURRENT_CLINICAL_CONCEPT_COUNT"] = et_concept.predict(X_test_ref)
df_test["CURRENT_CLINICAL_TRUE_COUNT"] = et_true.predict(X_test_ref)
df_test["CURRENT_CLINICAL_OBSERVED_COUNT"] = df_test["CURRENT_CLINICAL_CONCEPT_COUNT"]
df_test["CURRENT_CLINICAL_CONCEPT_DENSITY"] = df_test["CURRENT_CLINICAL_CONCEPT_COUNT"] / 762.0

print("Reconstruction complete.")

# 30-feature Model Evaluation
print("\nSTEP 4: Loading 30-feature XGBoost model...")
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
imputer = joblib.load(IMPUTER_PATH)

print(f"Loaded model expecting {len(features)} features.")

# Target encoding & filtering test split
df_test["TEST_TARGET"] = df_test["LABEL"].isin(["High", "Very High"]).astype(int)
df_test_eval = df_test[df_test["DATA_SPLIT"] == "test"].copy()
print(f"Unseen test split size: {len(df_test_eval)}")

# Encoding gender matching training model
if "GENDER_M" in features and "GENDER_M" not in df_test_eval.columns:
    df_test_eval = pd.get_dummies(df_test_eval, columns=["GENDER"], prefix="GENDER", drop_first=True)
    # Ensure column exists
    if "GENDER_M" not in df_test_eval.columns:
        df_test_eval["GENDER_M"] = 0

X_eval = df_test_eval[features].copy()
y_eval = df_test_eval["TEST_TARGET"].copy()

X_eval = X_eval.replace([np.inf, -np.inf], np.nan)
X_eval_imp = imputer.transform(X_eval)

print("\nSTEP 5: Evaluating Model...")
probabilities = model.predict_proba(X_eval_imp)[:, 1]
predictions = (probabilities >= 0.50).astype(int)

# Metrics
accuracy = accuracy_score(y_eval, predictions)
precision = precision_score(y_eval, predictions, zero_division=0)
recall = recall_score(y_eval, predictions, zero_division=0)
f1 = f1_score(y_eval, predictions, zero_division=0)
roc_auc = roc_auc_score(y_eval, probabilities)
pr_auc = average_precision_score(y_eval, probabilities)
brier = brier_score_loss(y_eval, probabilities)

print("-" * 50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
print(f"Brier:     {brier:.4f}")
print("-" * 50)

# Confusion Matrix
cm = confusion_matrix(y_eval, predictions)
print("Confusion Matrix:")
print(cm)

# Save Confusion Matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low/Mod", "High/Very High"])
disp.plot(cmap=plt.cm.Blues)
plt.title("XGBoost UC09 30-Feature Model on Reconstructed Test Set")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=200)
plt.close()

# Save predictions and metrics
comparison = pd.DataFrame({
    "PATIENT_ID": df_test_eval["PATIENT_ID"],
    "LABEL": df_test_eval["LABEL"],
    "LABEL_SCORE": df_test_eval["LABEL_SCORE"],
    "TRUE_BINARY_TARGET": y_eval,
    "MODEL_PROBABILITY": probabilities,
    "MODEL_PREDICTION": predictions
})
comparison.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier Score"],
    "Value": [accuracy, precision, recall, f1, roc_auc, pr_auc, brier]
})
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)
print(f"\nSaved metrics and predictions to {OUTPUT_DIR}")
