import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "DataSet/future_model/Cognizant_14K_Model_Ready.csv"

MODEL_PATH = "model_outputs_future/xgboost_future_model.pkl"

FEATURES_PATH = "model_outputs_future/model_features.pkl"

MEDIANS_PATH = "model_outputs_future/training_medians.pkl"

OUTPUT_DIR = "model_outputs_future"

TARGET = "FUTURE_TARGET"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# STEP 1 — LOAD MODEL
# ============================================================

print("=" * 70)
print("STEP 1 — LOADING XGBOOST MODEL")
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
# STEP 2 — LOAD DATA
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


X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.fillna(
    training_medians
)


print(
    "Test rows:",
    len(X_test)
)

print(
    "Missing values:",
    X_test.isna().sum().sum()
)


# ============================================================
# STEP 3 — CREATE SHAP EXPLAINER
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — CREATING SHAP EXPLAINER")
print("=" * 70)


explainer = shap.TreeExplainer(
    model
)


print(
    "SHAP TreeExplainer created."
)


# ============================================================
# STEP 4 — CALCULATE SHAP VALUES
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — CALCULATING SHAP VALUES")
print("=" * 70)


shap_values = explainer.shap_values(
    X_test
)


# Some SHAP/XGBoost versions return
# an array while others can return
# more complex structures.

if isinstance(
    shap_values,
    list
):

    shap_values = shap_values[1]


print(
    "SHAP matrix shape:",
    shap_values.shape
)


# ============================================================
# STEP 5 — GLOBAL SHAP IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — GLOBAL SHAP IMPORTANCE")
print("=" * 70)


mean_abs_shap = np.abs(
    shap_values
).mean(axis=0)


shap_importance = pd.DataFrame({

    "feature": FEATURES,

    "mean_absolute_shap":
        mean_abs_shap

}).sort_values(

    "mean_absolute_shap",

    ascending=False

)


print("\nTop 20 SHAP features:")

print(
    shap_importance
    .head(20)
    .to_string(index=False)
)


shap_importance.to_csv(

    f"{OUTPUT_DIR}/"
    "shap_feature_importance.csv",

    index=False

)


# ============================================================
# STEP 6 — SHAP SUMMARY PLOT
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 — SHAP SUMMARY")
print("=" * 70)


plt.figure(
    figsize=(10, 8)
)


shap.summary_plot(

    shap_values,

    X_test,

    show=False,

    max_display=20

)


plt.tight_layout()


summary_path = (
    f"{OUTPUT_DIR}/"
    "shap_summary_future.png"
)


plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "SHAP summary saved."
)


# ============================================================
# STEP 7 — SHAP BAR PLOT
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 — SHAP BAR PLOT")
print("=" * 70)


plt.figure(
    figsize=(10, 8)
)


shap.summary_plot(

    shap_values,

    X_test,

    plot_type="bar",

    show=False,

    max_display=20

)


plt.tight_layout()


bar_path = (
    f"{OUTPUT_DIR}/"
    "shap_bar_future.png"
)


plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "SHAP bar plot saved."
)


# ============================================================
# STEP 8 — INDIVIDUAL PATIENT EXPLANATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 — INDIVIDUAL PATIENT EXPLANATION")
print("=" * 70)


# Pick the first test patient.

patient_index = 0


patient_features = X_test.iloc[
    patient_index
]


patient_shap = shap_values[
    patient_index
]


patient_id = test_df.iloc[
    patient_index
]["PATIENT_ID"]


patient_probability = model.predict_proba(
    patient_features.to_frame().T
)[0, 1]


individual_explanation = pd.DataFrame({

    "feature": FEATURES,

    "feature_value":
        patient_features.values,

    "shap_value":
        patient_shap

})


individual_explanation[
    "absolute_shap"
] = np.abs(
    individual_explanation[
        "shap_value"
    ]
)


individual_explanation = (
    individual_explanation
    .sort_values(
        "absolute_shap",
        ascending=False
    )
)


print(
    "\nPatient:",
    patient_id
)

print(
    "Predicted future probability:",
    f"{patient_probability * 100:.2f}%"
)


print(
    "\nTop factors:"
)

print(
    individual_explanation[
        [
            "feature",
            "feature_value",
            "shap_value"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


individual_path = (
    f"{OUTPUT_DIR}/"
    "individual_patient_shap.csv"
)


individual_explanation.to_csv(
    individual_path,
    index=False
)


# ============================================================
# STEP 9 — SHAP WATERFALL
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 — PATIENT SHAP WATERFALL")
print("=" * 70)


try:

    explanation = shap.Explanation(

        values=patient_shap,

        base_values=explainer.expected_value,

        data=patient_features.values,

        feature_names=FEATURES

    )


    plt.figure(
        figsize=(10, 8)
    )


    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False
    )


    plt.tight_layout()


    waterfall_path = (
        f"{OUTPUT_DIR}/"
        "individual_patient_waterfall.png"
    )


    plt.savefig(
        waterfall_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Waterfall plot saved."
    )

except Exception as error:

    print(
        "Waterfall plot could not be generated:"
    )

    print(error)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"""
Saved:

{OUTPUT_DIR}/
│
├── shap_feature_importance.csv
├── shap_summary_future.png
├── shap_bar_future.png
├── individual_patient_shap.csv
└── individual_patient_waterfall.png
"""
)