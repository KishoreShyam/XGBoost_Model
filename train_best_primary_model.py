import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, brier_score_loss, ConfusionMatrixDisplay
)

# Config
DATA_PATH = "DataSet/Synthea_Dataset_with_RiskScore.csv"
OUTPUT_DIR = "model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "xgboost_uc09_model.pkl")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "model_features.pkl")
IMPUTER_PATH = os.path.join(OUTPUT_DIR, "median_imputer.pkl")

print("=" * 70)
print("TRAINING BEST PRIMARY RISK MODEL (TRAJECTORY + DEMOGRAPHICS)")
print("=" * 70)

# Load data
df = pd.read_csv(DATA_PATH, low_memory=False)
print("Rows:", len(df), "| Unique patients:", df["PATIENT_ID"].nunique())

# Define binary target
df["TARGET"] = df["LABEL"].isin(["High", "Very High"]).astype(int)

# Create Trajectory Features (within past 12M)
rng = np.random.default_rng(42)
past_count_cols = [
    "PAST_ENCOUNTER_COUNT_12M", "PAST_INPATIENT_COUNT_12M",
    "PAST_ED_VISIT_COUNT_12M", "PAST_OUTPATIENT_COUNT_12M",
    "PAST_CONDITION_COUNT_12M", "PAST_CHRONIC_CONDITION_COUNT_12M",
    "PAST_MEDICATION_COUNT_12M", "PAST_PROCEDURE_COUNT_12M",
    "PAST_OBSERVATION_COUNT_12M", "PAST_CLINICAL_BURDEN_12M",
    "PAST_HEALTHCARE_UTILIZATION_12M",
]

print("\nGenerating past trajectory features (PAST_A / PAST_B)...")
for col in past_count_cols:
    total = df[col].values
    split_ratio = rng.beta(6, 6, size=len(df))
    b_share = np.minimum(np.round(total * split_ratio).astype(int), total)
    a_share = total - b_share
    short_name = col.replace("_12M", "").replace("PAST_", "")
    df[f"PAST_A_{short_name}"] = a_share
    df[f"PAST_B_{short_name}"] = b_share

trend_fields = [
    "ENCOUNTER_COUNT", "CONDITION_COUNT", "CHRONIC_CONDITION_COUNT", "MEDICATION_COUNT",
    "PROCEDURE_COUNT", "OBSERVATION_COUNT", "CLINICAL_BURDEN", "HEALTHCARE_UTILIZATION",
]
for name in trend_fields:
    a_col, b_col = f"PAST_A_{name}", f"PAST_B_{name}"
    df[f"CHANGE_{name}"] = df[b_col] - df[a_col]
    df[f"GROWTH_{name}"] = (df[b_col] - df[a_col]) / (df[a_col] + 1)

# Encode demographics (excluding leakages)
df = pd.get_dummies(df, columns=["GENDER", "STATE", "COUNTY"], drop_first=True)

# Select features (PAST_*, CHANGE_*, GROWTH_*, GENDER_*, STATE_*, COUNTY_*)
features = [
    c for c in df.columns
    if c.startswith("PAST_") or c.startswith("CHANGE_") or c.startswith("GROWTH_")
    or c.startswith("GENDER_") or c.startswith("STATE_") or c.startswith("COUNTY_")
]
features = [c for c in features if c not in ["TARGET", "LABEL", "LABEL_SCORE", "DATA_SPLIT"]]

print(f"Using {len(features)} leakage-free features.")

# Split datasets
train_df = df[df["DATA_SPLIT"] == "train"]
val_df   = df[df["DATA_SPLIT"] == "val"]
test_df  = df[df["DATA_SPLIT"] == "test"]

X_train, y_train = train_df[features].copy(), train_df["TARGET"].copy()
X_val,   y_val   = val_df[features].copy(),   val_df["TARGET"].copy()
X_test,  y_test  = test_df[features].copy(),  test_df["TARGET"].copy()

# Imputation
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train.replace([np.inf, -np.inf], np.nan))
X_val_imp   = imputer.transform(X_val.replace([np.inf, -np.inf], np.nan))
X_test_imp  = imputer.transform(X_test.replace([np.inf, -np.inf], np.nan))

# Handle class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print(f"Train size: {len(X_train_imp)} | Val size: {len(X_val_imp)} | Test size: {len(X_test_imp)}")
print("Training XGBoost Classifier...")

model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.05,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_imp, y_train,
    eval_set=[(X_val_imp, y_val)],
    verbose=False
)

print(f"Best iteration: {model.best_iteration}")

# Evaluate
probs = model.predict_proba(X_test_imp)[:, 1]
preds = (probs >= 0.50).astype(int)

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, zero_division=0)
recall = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)
roc_auc = roc_auc_score(y_test, probs)
pr_auc = average_precision_score(y_test, probs)
brier = brier_score_loss(y_test, probs)

print("\n" + "=" * 50)
print("TEST SPLIT METRICS")
print("=" * 50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
print(f"Brier:     {brier:.4f}")
print("=" * 50)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, preds)
print(cm)

# Save Confusion Matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low/Mod", "High/Very High"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Best XGBoost 65-Feature Model on Test Set")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=200)
plt.close()

# Save models and features
joblib.dump(model, MODEL_PATH)
joblib.dump(features, FEATURES_PATH)
joblib.dump(imputer, IMPUTER_PATH)

print(f"\nArtifacts saved successfully to {OUTPUT_DIR}/")

# Generate SHAP
try:
    import shap
    print("\nGenerating SHAP Summary plot...")
    explainer = shap.TreeExplainer(model)
    # Use a sample for speed
    shap_sample = X_test_imp[np.random.choice(len(X_test_imp), size=min(300, len(X_test_imp)), replace=False)]
    shap_values = explainer.shap_values(shap_sample)
    
    plt.figure()
    shap.summary_plot(shap_values, shap_sample, feature_names=features, show=False)
    plt.title("SHAP Feature Importance (Optimized Model)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("SHAP Summary plot saved.")
except Exception as e:
    print(f"SHAP failed: {e}")
