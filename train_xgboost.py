import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "DataSet/ML_Synthea_Training_Dataset.csv"
OUTPUT_DIR = "model_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "LABEL"

# These columns are identifiers, metadata, or intentionally excluded features.
# They should NOT be given to XGBoost as predictive features.
DROP_COLUMNS = [
    "PATIENT_ID",
    "SNAPSHOT_DATE",
    "HISTORY_WINDOW_MONTHS",
    "CENSUS_TRACT_GEOID",
    "COUNTY_FIPS",
    "COUNTY_ID",
    "DATA_SPLIT",
    "LABEL_SCORE",
    "SCC",  # Removed after feature-ablation testing
]


# ============================================================
# STEP 1 — LOAD DATASET
# ============================================================

print("=" * 70)
print("STEP 1 — LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Unique patients:", df["PATIENT_ID"].nunique())

print("\nTarget distribution:")
print(df[TARGET].value_counts())

print("\nTarget percentage:")
print(
    (df[TARGET].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# STEP 2 — REMOVE INVALID TARGET ROWS
# ============================================================

df = df[df[TARGET].isin([0, 1])].copy()

print("\nRows after target validation:", len(df))


# ============================================================
# STEP 3 — CREATE X AND y
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — SEPARATING FEATURES AND TARGET")
print("=" * 70)

columns_to_drop = [
    column
    for column in DROP_COLUMNS
    if column in df.columns
]

X = df.drop(
    columns=[TARGET] + columns_to_drop,
    errors="ignore"
).copy()

y = df[TARGET].astype(int)

# Remove any remaining non-numeric columns.
non_numeric = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()

if non_numeric:

    print("\nRemoving non-numeric columns:")

    for column in non_numeric:
        print(" -", column)

    X = X.drop(columns=non_numeric)


print("\nFinal feature count:", X.shape[1])

print("\nFeatures used by XGBoost:")

for column in X.columns:
    print(" -", column)


# ============================================================
# STEP 4 — TRAIN / VALIDATION / TEST
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

# IMPORTANT:
# Our dataset already contains DATA_SPLIT created at PATIENT level.
# Therefore we use it instead of randomly splitting individual rows.

split = df["DATA_SPLIT"].astype(str).str.lower()

train_mask = split == "train"
validation_mask = split == "validation"
test_mask = split == "test"

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_validation = X.loc[validation_mask]
y_validation = y.loc[validation_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_validation))
print("Testing rows:", len(X_test))

print("\nTraining target:")
print(y_train.value_counts())

print("\nValidation target:")
print(y_validation.value_counts())

print("\nTesting target:")
print(y_test.value_counts())


# ============================================================
# STEP 5 — HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — HANDLING MISSING VALUES")
print("=" * 70)

# Median imputation.
#
# IMPORTANT:
# Fit ONLY on training data.
# Then apply the same transformation to validation/test.

imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)

X_validation = pd.DataFrame(
    imputer.transform(X_validation),
    columns=X_validation.columns,
    index=X_validation.index,
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)

print(
    "Training missing values:",
    X_train.isna().sum().sum()
)

print(
    "Validation missing values:",
    X_validation.isna().sum().sum()
)

print(
    "Testing missing values:",
    X_test.isna().sum().sum()
)


# ============================================================
# STEP 6 — CLASS IMBALANCE
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — CLASS BALANCE")
print("=" * 70)

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = (
    negative_count / positive_count
)

print("Negative samples:", negative_count)
print("Positive samples:", positive_count)
print(
    "scale_pos_weight:",
    round(scale_pos_weight, 4)
)


# ============================================================
# STEP 7 — CREATE XGBOOST MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 — CREATING XGBOOST MODEL")
print("=" * 70)

model = XGBClassifier(

    n_estimators=500,

    max_depth=5,

    learning_rate=0.03,

    subsample=0.80,

    colsample_bytree=0.80,

    min_child_weight=3,

    reg_alpha=0.05,

    reg_lambda=1.0,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    n_jobs=-1
)


# ============================================================
# STEP 8 — TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 — TRAINING XGBOOST")
print("=" * 70)

model.fit(

    X_train,

    y_train,

    eval_set=[
        (X_train, y_train),
        (X_validation, y_validation)
    ],

    verbose=False
)

from sklearn.metrics import accuracy_score, roc_auc_score

# Training predictions
train_pred = model.predict(X_train)
train_prob = model.predict_proba(X_train)[:, 1]

# Validation predictions
val_pred = model.predict(X_validation)
val_prob = model.predict_proba(X_validation)[:, 1]

# Test predictions
test_pred = model.predict(X_test)
test_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 70)
print("TRAIN vs VALIDATION vs TEST")
print("=" * 70)

print(f"Training Accuracy:   {accuracy_score(y_train, train_pred):.4f}")
print(f"Training ROC-AUC:    {roc_auc_score(y_train, train_prob):.4f}")

print(f"\nValidation Accuracy: {accuracy_score(y_validation, val_pred):.4f}")
print(f"Validation ROC-AUC:  {roc_auc_score(y_validation, val_prob):.4f}")

print(f"\nTest Accuracy:       {accuracy_score(y_test, test_pred):.4f}")
print(f"Test ROC-AUC:        {roc_auc_score(y_test, test_prob):.4f}")


print("XGBoost training completed!")


# ============================================================
# STEP 9 — PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 — GENERATING PREDICTIONS")
print("=" * 70)

# 0 / 1 prediction
y_pred = model.predict(X_test)

# Probability of class 1
y_probability = model.predict_proba(
    X_test
)[:, 1]

# Convert to percentage
y_probability_percent = (
    y_probability * 100
)

print("\nExample predictions:")

for i in range(
    min(10, len(y_test))
):

    print(
        f"Actual = {y_test.iloc[i]} | "
        f"Predicted = {y_pred[i]} | "
        f"Probability = "
        f"{y_probability_percent[i]:.2f}%"
    )


# ============================================================
# STEP 10 — MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 — MODEL EVALUATION")
print("=" * 70)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
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
        y_pred,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# STEP 11 — CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 — CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

fig, ax = plt.subplots(
    figsize=(6, 5)
)

ax.imshow(cm)

ax.set_title(
    "XGBoost Confusion Matrix"
)

ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "Actual"
)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/confusion_matrix.png",
    dpi=200
)

plt.close()


# ============================================================
# STEP 12 — FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 — FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.DataFrame({

    "feature":
        X_train.columns,

    "importance":
        model.feature_importances_

})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nTop 15 features:")

print(
    importance.head(15)
    .to_string(index=False)
)

importance.to_csv(
    f"{OUTPUT_DIR}/feature_importance.csv",
    index=False
)


# ============================================================
# STEP 13 — PREDICTION FILE
# ============================================================

print("\n" + "=" * 70)
print("STEP 12 — SAVING PREDICTIONS")
print("=" * 70)

predictions = df.loc[
    X_test.index,
    ["PATIENT_ID"]
].copy()

predictions["ACTUAL_LABEL"] = y_test

predictions["PREDICTED_LABEL"] = y_pred

predictions["POSITIVE_PROBABILITY"] = (
    y_probability
)

predictions[
    "POSITIVE_PROBABILITY_PERCENT"
] = y_probability_percent


predictions.to_csv(
    f"{OUTPUT_DIR}/test_predictions.csv",
    index=False
)

print(
    "Prediction file saved."
)


# ============================================================
# STEP 14 — SHAP
# ============================================================

print("\n" + "=" * 70)
print("STEP 13 — SHAP EXPLAINABILITY")
print("=" * 70)

try:

    import shap

    sample_size = min(
        500,
        len(X_test)
    )

    shap_sample = X_test.sample(
        n=sample_size,
        random_state=42
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = (
        explainer.shap_values(
            shap_sample
        )
    )

    shap.summary_plot(
        shap_values,
        shap_sample,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/shap_summary.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "SHAP summary saved."
    )

except Exception as error:

    print(
        "SHAP could not be generated:"
    )

    print(error)


# ============================================================
# STEP 15 — SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 14 — SAVING MODEL")
print("=" * 70)

joblib.dump(
    model,
    f"{OUTPUT_DIR}/xgboost_uc09_model.pkl"
)

joblib.dump(
    imputer,
    f"{OUTPUT_DIR}/median_imputer.pkl"
)

joblib.dump(
    list(X_train.columns),
    f"{OUTPUT_DIR}/model_features.pkl"
)


print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE!")
print("=" * 70)

print("\nSaved files:")

print(
    f"""
{OUTPUT_DIR}/
│
├── xgboost_uc09_model.pkl
├── median_imputer.pkl
├── model_features.pkl
├── test_predictions.csv
├── feature_importance.csv
├── confusion_matrix.png
└── shap_summary.png
"""
)