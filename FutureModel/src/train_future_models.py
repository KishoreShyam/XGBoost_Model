import os
import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

DATA_PATH = "FutureModel/data/Cognizant_14K_Model_Ready.csv"
MODEL_DIR = "FutureModel/outputs"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, low_memory=False)

# Exclude metadata, targets, and (defensively) any stray CURRENT_* columns
metadata = ["PATIENT_ID", "SNAPSHOT_DATE", "HISTORY_WINDOW_MONTHS", "CENSUS_TRACT_GEOID", "COUNTY_FIPS", "DATA_SPLIT"]
target_cols = [
    "FUTURE_TARGET", "CURRENT_RISK",
]
excluded = set(metadata + target_cols + [c for c in df.columns if c.startswith("CURRENT_")])
features = [c for c in df.columns if c not in excluded]
print(f"Using {len(features)} features (CURRENT_* and CURRENT_RISK excluded)")

class CalibratedXGBWrapper:
    def __init__(self, base_model, isotonic_regressor):
        self.base_model = base_model
        self.isotonic_regressor = isotonic_regressor

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        calibrated = np.clip(self.isotonic_regressor.predict(raw), 0, 1)
        return np.column_stack([1 - calibrated, calibrated])


outcome_targets = {
    "FUTURE": "FUTURE_TARGET"
}

results = {}

for outcome, target_col in outcome_targets.items():
    print("=" * 70)
    print(f"TRAINING: {outcome}  (target = {target_col})")
    print("=" * 70)

    train_df = df[df["DATA_SPLIT"] == "train"]
    val_df = df[df["DATA_SPLIT"] == "validation"]
    test_df = df[df["DATA_SPLIT"] == "test"]

    X_train, y_train = train_df[features].copy(), train_df[target_col].copy()
    X_val, y_val = val_df[features].copy(), val_df[target_col].copy()
    X_test, y_test = test_df[features].copy(), test_df[target_col].copy()

    for X in (X_train, X_val, X_test):
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc", early_stopping_rounds=30,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train_imp, y_train, eval_set=[(X_val_imp, y_val)], verbose=False)
    print("Best iteration:", model.best_iteration)

    raw_probs_val = model.predict_proba(X_val_imp)[:, 1]
    raw_brier_val = brier_score_loss(y_val, raw_probs_val)

    isotonic_regressor = IsotonicRegression(out_of_bounds="clip")
    isotonic_regressor.fit(raw_probs_val, y_val)
    cal_probs_val = isotonic_regressor.predict(raw_probs_val)
    cal_brier_val = brier_score_loss(y_val, cal_probs_val)

    use_calibrated = cal_brier_val < raw_brier_val
    final_model = CalibratedXGBWrapper(model, isotonic_regressor) if use_calibrated else model
    print(f"Calibration: {'USED (isotonic)' if use_calibrated else 'SKIPPED (raw better)'} "
          f"(raw Brier={raw_brier_val:.4f}, cal Brier={cal_brier_val:.4f})")

    test_probs = final_model.predict_proba(X_test_imp)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)

    print(f"Test  Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"Test  ROC-AUC:  {auc:.4f}  PR-AUC: {pr_auc:.4f}  Brier: {brier:.4f}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_imp)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs_shap}).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(os.path.join(MODEL_DIR, f"shap_importance_{outcome.lower()}.csv"), index=False)
    print("Top 5 SHAP features:", importance_df.head(5)["feature"].tolist())

    joblib.dump(model, os.path.join(MODEL_DIR, f"xgb_{outcome.lower()}.pkl"))
    joblib.dump(final_model, os.path.join(MODEL_DIR, f"final_{outcome.lower()}.pkl"))
    joblib.dump(imputer, os.path.join(MODEL_DIR, f"imputer_{outcome.lower()}.pkl"))
    joblib.dump(use_calibrated, os.path.join(MODEL_DIR, f"use_calibrated_{outcome.lower()}.pkl"))

    results[outcome] = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "roc_auc": auc, "pr_auc": pr_auc, "brier": brier, "used_calibration": use_calibrated,
    }

joblib.dump(features, os.path.join(MODEL_DIR, "model_features.pkl"))

# Extract training medians for imputation in downstream tasks
train_medians = df[df["DATA_SPLIT"] == "train"][features].median()
joblib.dump(train_medians, os.path.join(MODEL_DIR, "training_medians.pkl"))

print("\n" + "=" * 70)
print("SUMMARY — FUTURE RISK MODEL")
print("=" * 70)
summary_df = pd.DataFrame(results).T
print(summary_df.round(4))
summary_df.to_csv(os.path.join(MODEL_DIR, "model_summary.csv"))

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
