import os
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "DataSet/Synthea_Dataset_with_RiskScore.csv"
MODEL_DIR = "model_outputs"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_uc09_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.pkl")
IMPUTER_PATH = os.path.join(MODEL_DIR, "median_imputer.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# STEP 1 — LOAD DATASET
# ============================================================
print("STEP 1 — LOADING DATASET")
df = pd.read_csv(DATA_PATH, low_memory=False)
print("Rows:", len(df), "| Unique patients:", df["PATIENT_ID"].nunique())
print(df["DATA_SPLIT"].value_counts())


# ============================================================
# STEP 2 — BUILD BINARY TARGET
# ============================================================
# Low+Moderate = 0, High+Very High = 1
# LABEL/LABEL_SCORE were engineered from CURRENT_* columns, so those
# columns must NOT be used as features (that would be target leakage).
df["TARGET"] = df["LABEL"].isin(["High", "Very High"]).astype(int)
print(df["TARGET"].value_counts(normalize=True).round(4))


# ============================================================
# STEP 3 — SELECT FEATURES (PAST_* + demographics only)
# ============================================================
past_features = [c for c in df.columns if c.startswith("PAST_")]
df = pd.get_dummies(df, columns=["GENDER"], prefix="GENDER", drop_first=True)
gender_features = [c for c in df.columns if c.startswith("GENDER_")]
features = past_features + gender_features
print(f"Using {len(features)} features")


# ============================================================
# STEP 4 — TRAIN / VAL / TEST SPLIT
# ============================================================
train_df = df[df["DATA_SPLIT"] == "train"]
val_df   = df[df["DATA_SPLIT"] == "val"]
test_df  = df[df["DATA_SPLIT"] == "test"]

X_train, y_train = train_df[features].copy(), train_df["TARGET"].copy()
X_val,   y_val   = val_df[features].copy(),   val_df["TARGET"].copy()
X_test,  y_test  = test_df[features].copy(),  test_df["TARGET"].copy()

for X in (X_train, X_val, X_test):
    X.replace([np.inf, -np.inf], np.nan, inplace=True)


# ============================================================
# STEP 5 — IMPUTATION (fit on train only)
# ============================================================
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_val_imp   = imputer.transform(X_val)
X_test_imp  = imputer.transform(X_test)


# ============================================================
# STEP 6 — TRAIN XGBOOST
# ============================================================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
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
print("Best iteration:", model.best_iteration)


# ============================================================
# STEP 7 — EVALUATE ON HELD-OUT TEST SET
# ============================================================
probs = model.predict_proba(X_test_imp)[:, 1]
preds = (probs >= 0.50).astype(int)

print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
print(f"Precision: {precision_score(y_test, preds, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, preds, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, preds, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, probs):.4f}")
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds, target_names=["Low/Moderate", "High/Very High"], zero_division=0))


# ============================================================
# STEP 8 — SAVE ARTIFACTS
# ============================================================
joblib.dump(model, MODEL_PATH)
joblib.dump(features, FEATURES_PATH)
joblib.dump(imputer, IMPUTER_PATH)
print("Saved:", MODEL_PATH, FEATURES_PATH, IMPUTER_PATH)