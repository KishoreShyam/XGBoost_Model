import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "DataSet/future_model/Cognizant_14K_Model_Ready.csv"

MODEL_PATH = "model_outputs_future/xgboost_future_model.pkl"

FEATURES_PATH = "model_outputs_future/model_features.pkl"

MEDIANS_PATH = "model_outputs_future/training_medians.pkl"

OUTPUT_DIR = "model_outputs_future"

TARGET = "FUTURE_TARGET"


# ============================================================
# STEP 1 — LOAD MODEL
# ============================================================

print("=" * 70)
print("STEP 1 — LOADING TRAINED MODEL")
print("=" * 70)

model = joblib.load(
    MODEL_PATH
)

FEATURES = joblib.load(
    FEATURES_PATH
)

training_medians = joblib.load(
    MEDIANS_PATH
)

print(
    "Model loaded successfully."
)

print(
    "Features:",
    len(FEATURES)
)


# ============================================================
# STEP 2 — LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — LOADING TEST DATA")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

test_df = df[
    df["DATA_SPLIT"] == "test"
].copy()


X_test = test_df[
    FEATURES
].copy()

y_test = test_df[
    TARGET
].copy()


# ============================================================
# STEP 3 — CLEAN FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — DATA CLEANING")
print("=" * 70)


X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.fillna(
    training_medians
)


print(
    "Missing values:",
    X_test.isna().sum().sum()
)


# ============================================================
# STEP 4 — GENERATE PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — GENERATING PROBABILITIES")
print("=" * 70)


probabilities = model.predict_proba(
    X_test
)[:, 1]


print(
    "Minimum probability:",
    round(probabilities.min(), 4)
)

print(
    "Maximum probability:",
    round(probabilities.max(), 4)
)

print(
    "Mean probability:",
    round(probabilities.mean(), 4)
)


# ============================================================
# STEP 5 — BRIER SCORE
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — PROBABILITY CALIBRATION")
print("=" * 70)


brier = brier_score_loss(
    y_test,
    probabilities
)


auc = roc_auc_score(
    y_test,
    probabilities
)


print(
    f"Brier Score: {brier:.4f}"
)

print(
    f"ROC-AUC:     {auc:.4f}"
)


# ============================================================
# STEP 6 — CALIBRATION CURVE
# ============================================================

fraction_positive, mean_predicted = (
    calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="quantile"
    )
)


print("\nCalibration bins:")

for predicted, actual in zip(
    mean_predicted,
    fraction_positive
):

    print(
        f"Predicted = {predicted:.3f}"
        f" | Actual = {actual:.3f}"
    )


# ============================================================
# STEP 7 — PLOT
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)


plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    label="XGBoost"
)


plt.xlabel(
    "Predicted Probability"
)

plt.ylabel(
    "Observed Frequency"
)

plt.title(
    "Future Risk Probability Calibration"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


calibration_path = (
    f"{OUTPUT_DIR}/"
    "calibration_curve.png"
)


plt.savefig(
    calibration_path,
    dpi=300
)

plt.close()


# ============================================================
# STEP 8 — SAVE PROBABILITY RESULTS
# ============================================================

results = pd.DataFrame({

    "PATIENT_ID":
        test_df["PATIENT_ID"].values,

    "ACTUAL_TARGET":
        y_test.values,

    "PREDICTED_PROBABILITY":
        probabilities,

    "PREDICTED_PERCENTAGE":
        probabilities * 100

})


results[
    "PREDICTED_PERCENTAGE"
] = results[
    "PREDICTED_PERCENTAGE"
].round(2)


results_path = (
    f"{OUTPUT_DIR}/"
    "calibrated_test_probabilities.csv"
)


results.to_csv(
    results_path,
    index=False
)


# ============================================================
# STEP 9 — PROBABILITY INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 — PROBABILITY DISTRIBUTION")
print("=" * 70)


bins = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
]


labels = [
    "0-20%",
    "20-40%",
    "40-60%",
    "60-80%",
    "80-100%"
]


results["RISK_RANGE"] = pd.cut(
    results["PREDICTED_PROBABILITY"],
    bins=bins,
    labels=labels,
    include_lowest=True
)


print(
    results[
        "RISK_RANGE"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("PROBABILITY CALIBRATION COMPLETE")
print("=" * 70)

print(
    f"""
Brier Score : {brier:.4f}
ROC-AUC     : {auc:.4f}

Saved:

{calibration_path}
{results_path}
"""
)