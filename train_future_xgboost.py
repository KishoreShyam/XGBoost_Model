import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "DataSet/future_model/Cognizant_14K_Model_Ready.csv"

OUTPUT_DIR = "model_outputs_future"

TARGET = "FUTURE_TARGET"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# STEP 1 — LOAD DATASET
# ============================================================

print("=" * 70)
print("STEP 1 — LOADING FUTURE-RISK DATASET")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nTarget distribution:")

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)

print("\nTarget percentage:")

print(
    df[TARGET]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# STEP 2 — DEFINE EXCLUDED COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — REMOVING LEAKAGE / METADATA")
print("=" * 70)


metadata_columns = [

    "PATIENT_ID",
    "SNAPSHOT_DATE",
    "HISTORY_WINDOW_MONTHS",
    "CENSUS_TRACT_GEOID",
    "COUNTY_FIPS",
    "DATA_SPLIT"

]


target_columns = [

    "FUTURE_TARGET",

    "FUTURE_HIGH_UTILIZATION",

    "FUTURE_CLINICAL_DETERIORATION",

    "FUTURE_HEALTHCARE_ESCALATION",

    "SYNTHETIC_FUTURE_RISK_SCORE",

    "SYNTHETIC_FUTURE_HIGH_UTILIZATION_PROB",

    "SYNTHETIC_FUTURE_DETERIORATION_PROB",

    "SYNTHETIC_FUTURE_ESCALATION_PROB"

]


excluded_columns = set(
    metadata_columns +
    target_columns
)


FEATURES = [

    column

    for column in df.columns

    if column not in excluded_columns
]


print("Final feature count:", len(FEATURES))

print("\nFeatures used by XGBoost:")

for feature in FEATURES:

    print(" -", feature)


# Safety check

leakage = [

    column

    for column in FEATURES

    if (
        "FUTURE" in column.upper()
        or "RISK_SCORE" in column.upper()
        or "PROB" in column.upper()
    )

]


if leakage:

    raise ValueError(
        f"TARGET LEAKAGE DETECTED: {leakage}"
    )


print("\nTarget leakage check: PASSED")


# ============================================================
# STEP 3 — USE PREDEFINED SPLITS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — TRAIN / VALIDATION / TEST")
print("=" * 70)


train_df = df[
    df["DATA_SPLIT"] == "train"
].copy()

val_df = df[
    df["DATA_SPLIT"] == "validation"
].copy()

test_df = df[
    df["DATA_SPLIT"] == "test"
].copy()


X_train = train_df[FEATURES]

y_train = train_df[TARGET]


X_val = val_df[FEATURES]

y_val = val_df[TARGET]


X_test = test_df[FEATURES]

y_test = test_df[TARGET]


print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))


print("\nTraining target:")

print(
    y_train.value_counts().sort_index()
)


print("\nValidation target:")

print(
    y_val.value_counts().sort_index()
)


print("\nTesting target:")

print(
    y_test.value_counts().sort_index()
)


# ============================================================
# STEP 4 — MISSING / INVALID VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — DATA QUALITY CHECK")
print("=" * 70)


X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_val = X_val.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


train_medians = X_train.median()


X_train = X_train.fillna(
    train_medians
)

X_val = X_val.fillna(
    train_medians
)

X_test = X_test.fillna(
    train_medians
)


print(
    "Training missing:",
    X_train.isna().sum().sum()
)

print(
    "Validation missing:",
    X_val.isna().sum().sum()
)

print(
    "Testing missing:",
    X_test.isna().sum().sum()
)


# ============================================================
# STEP 5 — CLASS BALANCE
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — CLASS BALANCE")
print("=" * 70)


negative = int(
    (y_train == 0).sum()
)

positive = int(
    (y_train == 1).sum()
)


scale_pos_weight = (
    negative / positive
)


print("Negative samples:", negative)
print("Positive samples:", positive)

print(
    "scale_pos_weight:",
    round(scale_pos_weight, 4)
)


# ============================================================
# STEP 6 — CREATE XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 — CREATING XGBOOST MODEL")
print("=" * 70)


model = XGBClassifier(

    n_estimators=500,

    max_depth=4,

    learning_rate=0.03,

    min_child_weight=5,

    subsample=0.80,

    colsample_bytree=0.80,

    reg_alpha=0.10,

    reg_lambda=2.0,

    scale_pos_weight=scale_pos_weight,

    objective="binary:logistic",

    eval_metric="auc",

    random_state=42,

    n_jobs=-1

)


# ============================================================
# STEP 7 — TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 — TRAINING XGBOOST")
print("=" * 70)


model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_train, y_train),
        (X_val, y_val)
    ],
    verbose=False
)


print("XGBoost training completed!")


# ============================================================
# STEP 8 — TRAIN / VALIDATION / TEST COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 — TRAIN vs VALIDATION vs TEST")
print("=" * 70)


def evaluate_split(
    name,
    X,
    y
):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    auc = roc_auc_score(
        y,
        probabilities
    )

    f1 = f1_score(
        y,
        predictions
    )

    print(
        f"{name} Accuracy: {accuracy:.4f}"
    )

    print(
        f"{name} F1:       {f1:.4f}"
    )

    print(
        f"{name} ROC-AUC:  {auc:.4f}"
    )

    return (
        predictions,
        probabilities
    )


train_pred, train_prob = evaluate_split(
    "Training",
    X_train,
    y_train
)

print()

val_pred, val_prob = evaluate_split(
    "Validation",
    X_val,
    y_val
)

print()

test_pred, test_prob = evaluate_split(
    "Test",
    X_test,
    y_test
)


# ============================================================
# STEP 9 — TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 — FINAL TEST EVALUATION")
print("=" * 70)


accuracy = accuracy_score(
    y_test,
    test_pred
)

precision = precision_score(
    y_test,
    test_pred
)

recall = recall_score(
    y_test,
    test_pred
)

f1 = f1_score(
    y_test,
    test_pred
)

roc_auc = roc_auc_score(
    y_test,
    test_prob
)

pr_auc = average_precision_score(
    y_test,
    test_prob
)


print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_pred,
        digits=4
    )
)


# ============================================================
# STEP 10 — CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 — CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(
    y_test,
    test_pred
)

print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Future High Utilization — Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 11 — FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 — FEATURE IMPORTANCE")
print("=" * 70)


importance_df = pd.DataFrame({

    "feature": FEATURES,

    "importance": model.feature_importances_

}).sort_values(

    "importance",
    ascending=False

)


print("\nTop 20 features:")

print(
    importance_df
    .head(20)
    .to_string(index=False)
)


importance_df.to_csv(

    f"{OUTPUT_DIR}/feature_importance.csv",

    index=False

)


# ============================================================
# STEP 12 — SAVE TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 12 — SAVING PREDICTIONS")
print("=" * 70)


predictions_df = pd.DataFrame({

    "PATIENT_ID":
        test_df["PATIENT_ID"].values,

    "ACTUAL_SYNTHETIC_TARGET":
        y_test.values,

    "PREDICTED_TARGET":
        test_pred,

    "PREDICTED_PROBABILITY":
        test_prob,

    "PREDICTED_PERCENTAGE":
        np.round(
            test_prob * 100,
            2
        )

})


predictions_df.to_csv(

    f"{OUTPUT_DIR}/test_predictions.csv",

    index=False

)


print("Predictions saved.")


# ============================================================
# STEP 13 — SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 13 — SAVING MODEL")
print("=" * 70)


joblib.dump(

    model,

    f"{OUTPUT_DIR}/xgboost_future_model.pkl"

)


joblib.dump(

    FEATURES,

    f"{OUTPUT_DIR}/model_features.pkl"

)


joblib.dump(

    train_medians,

    f"{OUTPUT_DIR}/training_medians.pkl"

)


print("Model saved.")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FUTURE-RISK MODEL TRAINING COMPLETE")
print("=" * 70)


print(
    f"""
Saved files:

{OUTPUT_DIR}/
│
├── xgboost_future_model.pkl
├── model_features.pkl
├── training_medians.pkl
├── test_predictions.csv
├── feature_importance.csv
└── confusion_matrix.png
"""
)