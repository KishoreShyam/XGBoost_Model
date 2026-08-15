import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    log_loss
)

DATA_PATH = "FutureModel/data/Cognizant_14K_Model_Ready.csv"
MODEL_DIR = "FutureModel/outputs"
OUTPUT_DIR = "FutureModel/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = [
    "FUTURE_HIGH_UTILIZATION",
    "FUTURE_CLINICAL_DETERIORATION",
    "FUTURE_HEALTHCARE_ESCALATION"
]

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH, low_memory=False)
print("Rows:", len(df))

FEATURES = joblib.load(f"{MODEL_DIR}/model_features.pkl")
training_medians = joblib.load(f"{MODEL_DIR}/training_medians.pkl")
print("Features:", len(FEATURES))

X = df[FEATURES].copy().replace([np.inf, -np.inf], np.nan).fillna(training_medians)

train_df = df[df["DATA_SPLIT"] == "train"].copy()
validation_df = df[df["DATA_SPLIT"] == "validation"].copy()
test_df = df[df["DATA_SPLIT"] == "test"].copy()

X_validation = validation_df[FEATURES].copy().replace([np.inf, -np.inf], np.nan).fillna(training_medians)
X_test = test_df[FEATURES].copy().replace([np.inf, -np.inf], np.nan).fillna(training_medians)

print("Validation rows:", len(X_validation))
print("Test rows:", len(X_test))


def probability_to_logit(probabilities):
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities))


def calibrate_sigmoid(validation_probability, validation_target, test_probability):
    validation_logit = probability_to_logit(validation_probability).reshape(-1, 1)
    test_logit = probability_to_logit(test_probability).reshape(-1, 1)

    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(validation_logit, validation_target)

    calibrated_validation = calibrator.predict_proba(validation_logit)[:, 1]
    calibrated_test = calibrator.predict_proba(test_logit)[:, 1]

    return calibrator, calibrated_validation, calibrated_test


print("\n" + "=" * 70)
print("FUTURE RISK MODELS PROBABILITY CALIBRATION")
print("=" * 70)

all_predictions = pd.DataFrame({"PATIENT_ID": test_df["PATIENT_ID"].values})
summary = []

for target in TARGETS:
    print("\n\n")
    print("=" * 70)
    print(f"CALIBRATING: {target}")
    print("=" * 70)

    target_to_file = {
        "FUTURE_HIGH_UTILIZATION": "xgb_utilization.pkl",
        "FUTURE_CLINICAL_DETERIORATION": "xgb_deterioration.pkl",
        "FUTURE_HEALTHCARE_ESCALATION": "xgb_escalation.pkl"
    }
    model_path = os.path.join(MODEL_DIR, target_to_file[target])
    model = joblib.load(model_path)
    print("Model loaded.")

    y_validation = validation_df[target].astype(int).values
    y_test = test_df[target].astype(int).values

    validation_probability = model.predict_proba(X_validation)[:, 1]
    test_probability = model.predict_proba(X_test)[:, 1]

    before_brier = brier_score_loss(y_test, test_probability)
    before_auc = roc_auc_score(y_test, test_probability)
    before_logloss = log_loss(y_test, test_probability)

    print("\nBEFORE CALIBRATION")
    print(f"Brier Score : {before_brier:.4f}")
    print(f"ROC-AUC     : {before_auc:.4f}")
    print(f"Log Loss    : {before_logloss:.4f}")

    calibrator, calibrated_validation, calibrated_test = calibrate_sigmoid(
        validation_probability, y_validation, test_probability
    )

    after_brier = brier_score_loss(y_test, calibrated_test)
    after_auc = roc_auc_score(y_test, calibrated_test)
    after_logloss = log_loss(y_test, calibrated_test)

    print("\nAFTER CALIBRATION")
    print(f"Brier Score : {after_brier:.4f}")
    print(f"ROC-AUC     : {after_auc:.4f}")
    print(f"Log Loss    : {after_logloss:.4f}")

    brier_change = before_brier - after_brier
    logloss_change = before_logloss - after_logloss

    model_name = target.lower()
    joblib.dump(calibrator, f"{OUTPUT_DIR}/{model_name}_calibrator.pkl")

    all_predictions[f"{target}_ORIGINAL_PROB"] = np.round(test_probability, 6)
    all_predictions[f"{target}_CALIBRATED_PROB"] = np.round(calibrated_test, 6)
    all_predictions[f"{target}_CALIBRATED_PERCENT"] = np.round(calibrated_test * 100, 2)
    all_predictions[f"{target}_PREDICTION"] = (calibrated_test >= 0.50).astype(int)

    summary.append({
        "target": target,
        "before_brier": before_brier,
        "after_brier": after_brier,
        "brier_improvement": brier_change,
        "before_roc_auc": before_auc,
        "after_roc_auc": after_auc,
        "before_log_loss": before_logloss,
        "after_log_loss": after_logloss,
        "log_loss_improvement": logloss_change
    })

prediction_path = f"{OUTPUT_DIR}/multi_target_calibrated_predictions.csv"
all_predictions.to_csv(prediction_path, index=False)
print("\nSaved predictions to:", prediction_path)

summary_df = pd.DataFrame(summary)
summary_path = f"{OUTPUT_DIR}/calibration_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 70)
print("CALIBRATION SUMMARY")
print("=" * 70)
print(summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("FUTURE MODELS CALIBRATION COMPLETE")
print("=" * 70)
