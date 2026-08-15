import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "DataSet/future_model/Cognizant_14K_Model_Ready.csv"

MODEL_DIR = "model_outputs_multi"

OUTPUT_DIR = "model_outputs_multi/shap"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


TARGETS = [

    "FUTURE_HIGH_UTILIZATION",

    "FUTURE_CLINICAL_DETERIORATION",

    "FUTURE_HEALTHCARE_ESCALATION"

]


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 1 — LOADING DATA")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)


test_df = df[
    df["DATA_SPLIT"] == "test"
].copy()


print(
    "Test patients:",
    len(test_df)
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
    "Features:",
    len(FEATURES)
)

print(
    "Missing values:",
    X_test.isna().sum().sum()
)


# ============================================================
# STEP 3 — ANALYZE EACH TARGET
# ============================================================

for target in TARGETS:

    print("\n\n")

    print("=" * 70)

    print(
        f"SHAP ANALYSIS — {target}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model_name = target.lower()

    model_path = (
        f"{MODEL_DIR}/"
        f"{model_name}_xgboost.pkl"
    )


    model = joblib.load(
        model_path
    )


    print(
        "Model loaded."
    )


    # --------------------------------------------------------
    # CREATE EXPLAINER
    # --------------------------------------------------------

    print(
        "Creating TreeExplainer..."
    )


    explainer = shap.TreeExplainer(
        model
    )


    # --------------------------------------------------------
    # CALCULATE SHAP
    # --------------------------------------------------------

    print(
        "Calculating SHAP values..."
    )


    shap_values = explainer.shap_values(
        X_test
    )


    shap_values = np.asarray(
        shap_values
    )


    print(
        "SHAP matrix:",
        shap_values.shape
    )


    # --------------------------------------------------------
    # GLOBAL IMPORTANCE
    # --------------------------------------------------------

    mean_abs_shap = np.abs(
        shap_values
    ).mean(
        axis=0
    )


    importance = pd.DataFrame({

        "feature":
            FEATURES,

        "mean_absolute_shap":
            mean_abs_shap

    }).sort_values(

        "mean_absolute_shap",

        ascending=False

    )


    importance_path = (

        f"{OUTPUT_DIR}/"
        f"{model_name}_shap_importance.csv"

    )


    importance.to_csv(
        importance_path,
        index=False
    )


    print(
        "\nTop 15 SHAP features:"
    )


    print(
        importance
        .head(15)
        .to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # SHAP SUMMARY PLOT
    # --------------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )

    plt.title(
        target
    )

    plt.tight_layout()


    summary_path = (

        f"{OUTPUT_DIR}/"
        f"{model_name}_shap_summary.png"

    )


    plt.savefig(
        summary_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    # --------------------------------------------------------
    # SHAP BAR PLOT
    # --------------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False
    )

    plt.title(
        f"{target} — SHAP Importance"
    )

    plt.tight_layout()


    bar_path = (

        f"{OUTPUT_DIR}/"
        f"{model_name}_shap_bar.png"

    )


    plt.savefig(
        bar_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    # --------------------------------------------------------
    # INDIVIDUAL PATIENT
    # --------------------------------------------------------

    patient_index = 0


    patient_id = test_df.iloc[
        patient_index
    ]["PATIENT_ID"]


    patient_shap = shap_values[
        patient_index
    ]


    patient_features = X_test.iloc[
        patient_index
    ]


    patient_explanation = pd.DataFrame({

        "feature":
            FEATURES,

        "feature_value":
            patient_features.values,

        "shap_value":
            patient_shap

    })


    patient_explanation[
        "absolute_shap"
    ] = np.abs(
        patient_explanation[
            "shap_value"
        ]
    )


    patient_explanation = (
        patient_explanation
        .sort_values(
            "absolute_shap",
            ascending=False
        )
    )


    patient_path = (

        f"{OUTPUT_DIR}/"
        f"{model_name}_patient_explanation.csv"

    )


    patient_explanation.to_csv(
        patient_path,
        index=False
    )


    print(
        f"\nPatient: {patient_id}"
    )


    print(
        "\nTop SHAP contributors:"
    )


    print(
        patient_explanation
        .head(15)
        .to_string(
            index=False
        )
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("MULTI-TARGET SHAP ANALYSIS COMPLETE")
print("=" * 70)


print(
    f"""
Created:

{OUTPUT_DIR}/

For each target:
├── *_shap_importance.csv
├── *_shap_summary.png
├── *_shap_bar.png
└── *_patient_explanation.csv
"""
)