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

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/Cognizant_14K_Model_Ready.csv"
MODEL_DIR = "outputs/models"
OUTPUT_DIR = "outputs/calibration"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# TARGETS
# ============================================================

TARGETS = [
    "CURRENT_RISK",
    "FUTURE_HIGH_UTILIZATION",
    "FUTURE_CLINICAL_DETERIORATION",
    "FUTURE_HEALTHCARE_ESCALATION"
]

# ============================================================
# STEP 1 — LOAD DATASET
# ============================================================

print("=" * 70)
print("STEP 1 — LOADING DATASET")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print(
    "Rows:",
    len(df)
)

# ============================================================
# STEP 2 — LOAD FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — LOADING MODEL FEATURES")
print("=" * 70)

FEATURES = joblib.load(
    f"{MODEL_DIR}/model_features.pkl"
)

training_medians = joblib.load(
    f"{MODEL_DIR}/training_medians.pkl"
)

print(
    "Features:",
    len(FEATURES)
)

# ============================================================
# PREPARE FEATURES
# ============================================================

X = df[FEATURES].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    training_medians
)

# ============================================================
# SPLIT DATA
# ============================================================

train_df = df[
    df["DATA_SPLIT"] == "train"
].copy()

validation_df = df[
    df["DATA_SPLIT"] == "validation"
].copy()

test_df = df[
    df["DATA_SPLIT"] == "test"
].copy()

X_validation = validation_df[
    FEATURES
].copy()

X_test = test_df[
    FEATURES
].copy()

X_validation = X_validation.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

X_validation = X_validation.fillna(
    training_medians
)

X_test = X_test.fillna(
    training_medians
)

print(
    "Validation rows:",
    len(X_validation)
)

print(
    "Test rows:",
    len(X_test)
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def probability_to_logit(probabilities):
    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6
    )
    return np.log(
        probabilities /
        (1 - probabilities)
    )


def calibrate_sigmoid(
    validation_probability,
    validation_target,
    test_probability
):
    # Convert XGBoost probabilities into
    # log-odds before fitting calibration model.

    validation_logit = probability_to_logit(
        validation_probability
    ).reshape(-1, 1)

    test_logit = probability_to_logit(
        test_probability
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        random_state=42
    )

    calibrator.fit(
        validation_logit,
        validation_target
    )

    calibrated_validation = calibrator.predict_proba(
        validation_logit
    )[:, 1]

    calibrated_test = calibrator.predict_proba(
        test_logit
    )[:, 1]

    return (
        calibrator,
        calibrated_validation,
        calibrated_test
    )


# ============================================================
# STEP 3 — CALIBRATE EACH MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — MULTI-TARGET PROBABILITY CALIBRATION")
print("=" * 70)

all_predictions = pd.DataFrame({
    "PATIENT_ID":
        test_df["PATIENT_ID"].values
})

summary = []

for target in TARGETS:
    print("\n\n")
    print("=" * 70)
    print(
        f"CALIBRATING: {target}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD MODEL
    # Map target names to filenames
    model_name = target.lower()
    target_to_file = {
        "CURRENT_RISK": "xgb_current_risk.pkl",
        "FUTURE_HIGH_UTILIZATION": "xgb_utilization.pkl",
        "FUTURE_CLINICAL_DETERIORATION": "xgb_deterioration.pkl",
        "FUTURE_HEALTHCARE_ESCALATION": "xgb_escalation.pkl"
    }
    model_path = os.path.join(MODEL_DIR, target_to_file[target])

    model = joblib.load(
        model_path
    )

    print(
        "Model loaded."
    )

    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

    y_validation = validation_df[
        target
    ].astype(int).values

    y_test = test_df[
        target
    ].astype(int).values

    # --------------------------------------------------------
    # ORIGINAL PROBABILITIES
    # --------------------------------------------------------

    validation_probability = model.predict_proba(
        X_validation
    )[:, 1]

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # BEFORE CALIBRATION
    # --------------------------------------------------------

    before_brier = brier_score_loss(
        y_test,
        test_probability
    )

    before_auc = roc_auc_score(
        y_test,
        test_probability
    )

    before_logloss = log_loss(
        y_test,
        test_probability
    )

    print("\nBEFORE CALIBRATION")
    print(
        f"Brier Score : {before_brier:.4f}"
    )
    print(
        f"ROC-AUC     : {before_auc:.4f}"
    )
    print(
        f"Log Loss    : {before_logloss:.4f}"
    )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    (
        calibrator,
        calibrated_validation,
        calibrated_test
    ) = calibrate_sigmoid(
        validation_probability,
        y_validation,
        test_probability
    )

    # --------------------------------------------------------
    # AFTER CALIBRATION
    # --------------------------------------------------------

    after_brier = brier_score_loss(
        y_test,
        calibrated_test
    )

    after_auc = roc_auc_score(
        y_test,
        calibrated_test
    )

    after_logloss = log_loss(
        y_test,
        calibrated_test
    )

    print("\nAFTER CALIBRATION")
    print(
        f"Brier Score : {after_brier:.4f}"
    )
    print(
        f"ROC-AUC     : {after_auc:.4f}"
    )
    print(
        f"Log Loss    : {after_logloss:.4f}"
    )

    # --------------------------------------------------------
    # IMPROVEMENT
    # --------------------------------------------------------

    brier_change = (
        before_brier -
        after_brier
    )

    logloss_change = (
        before_logloss -
        after_logloss
    )

    print("\nCALIBRATION IMPROVEMENT")
    print(
        f"Brier improvement : {brier_change:.4f}"
    )
    print(
        f"Log-loss improvement : {logloss_change:.4f}"
    )

    # --------------------------------------------------------
    # PROBABILITY DISTRIBUTION
    # --------------------------------------------------------

    print("\nProbability statistics:")
    print(
        "Original:"
    )
    print(
        pd.Series(
            test_probability
        ).describe()
    )

    print(
        "\nCalibrated:"
    )
    print(
        pd.Series(
            calibrated_test
        ).describe()
    )

    # --------------------------------------------------------
    # SAVE CALIBRATOR
    # --------------------------------------------------------

    calibrator_path = (
        f"{OUTPUT_DIR}/"
        f"{model_name}_calibrator.pkl"
    )

    joblib.dump(
        calibrator,
        calibrator_path
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    all_predictions[
        f"{target}_ORIGINAL_PROB"
    ] = np.round(
        test_probability,
        6
    )

    all_predictions[
        f"{target}_CALIBRATED_PROB"
    ] = np.round(
        calibrated_test,
        6
    )

    all_predictions[
        f"{target}_CALIBRATED_PERCENT"
    ] = np.round(
        calibrated_test * 100,
        2
    )

    all_predictions[
        f"{target}_PREDICTION"
    ] = (
        calibrated_test >= 0.50
    ).astype(int)

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary.append({
        "target":
            target,
        "before_brier":
            before_brier,
        "after_brier":
            after_brier,
        "brier_improvement":
            brier_change,
        "before_roc_auc":
            before_auc,
        "after_roc_auc":
            after_auc,
        "before_log_loss":
            before_logloss,
        "after_log_loss":
            after_logloss,
        "log_loss_improvement":
            logloss_change
    })

# ============================================================
# STEP 4 — SAVE COMBINED PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — SAVING CALIBRATED PREDICTIONS")
print("=" * 70)

prediction_path = (
    f"{OUTPUT_DIR}/"
    "multi_target_calibrated_predictions.csv"
)

all_predictions.to_csv(
    prediction_path,
    index=False
)

print(
    "Saved:",
    prediction_path
)

# ============================================================
# STEP 5 — SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary
)

summary_path = (
    f"{OUTPUT_DIR}/"
    "calibration_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print(
    "\n" + "=" * 70
)
print(
    "CALIBRATION SUMMARY"
)
print(
    "=" * 70
)

print(
    summary_df.to_string(
        index=False
    )
)

# ============================================================
# STEP 6 — RISK DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATED RISK DISTRIBUTION")
print("=" * 70)

for target in TARGETS:
    column = (
        f"{target}_CALIBRATED_PERCENT"
    )

    print(
        f"\n{target}"
    )

    bins = pd.cut(
        all_predictions[column],
        bins=[
            0,
            20,
            40,
            60,
            80,
            100
        ],
        labels=[
            "0-20%",
            "20-40%",
            "40-60%",
            "60-80%",
            "80-100%"
        ],
        include_lowest=True
    )

    print(
        bins.value_counts()
        .sort_index()
    )

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("MULTI-TARGET CALIBRATION COMPLETE")
print("=" * 70)

print(
    f"""
Created:

{OUTPUT_DIR}/
|
+-- current_risk_calibrator.pkl
+-- future_high_utilization_calibrator.pkl
+-- future_clinical_deterioration_calibrator.pkl
+-- future_healthcare_escalation_calibrator.pkl
|
+-- multi_target_calibrated_predictions.csv
+-- calibration_summary.csv
"""
)
