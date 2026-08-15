import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DATA_PATH = "CurrentModel/data/ML_Synthea_Training_Dataset.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

TARGET = "LABEL"

DROP_COLUMNS = [
    "PATIENT_ID",
    "SNAPSHOT_DATE",
    "HISTORY_WINDOW_MONTHS",
    "CENSUS_TRACT_GEOID",
    "COUNTY_FIPS",
    "COUNTY_ID",
    "DATA_SPLIT",
    "LABEL_SCORE",
]

X = df.drop(
    columns=[TARGET] + [
        c for c in DROP_COLUMNS
        if c in df.columns
    ],
    errors="ignore"
)

y = df[TARGET].astype(int)
X = X.select_dtypes(include=[np.number])

split = df["DATA_SPLIT"].astype(str).str.lower()

train_mask = split == "train"
val_mask = split == "validation"
test_mask = split == "test"

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]
X_val = X.loc[val_mask]
y_val = y.loc[val_mask]
X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns, index=X_val.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale_pos_weight = negative / positive


def run_experiment(name, remove_features):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    features = [c for c in X_train.columns if c not in remove_features]
    train = X_train[features]
    val = X_val[features]
    test = X_test[features]

    model = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.80, colsample_bytree=0.80, min_child_weight=3,
        reg_alpha=0.05, reg_lambda=1.0, objective="binary:logistic",
        eval_metric="logloss", scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1
    )

    model.fit(train, y_train, eval_set=[(val, y_val)], verbose=False)

    pred = model.predict(test)
    prob = model.predict_proba(test)[:, 1]

    print("Features used:", len(features))
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("F1:", round(f1_score(y_test, pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))


run_experiment("MODEL A — ALL FEATURES", [])
run_experiment("MODEL B — WITHOUT SCC", ["SCC"])
run_experiment(
    "MODEL C — WITHOUT TOP CLINICAL FEATURES",
    [
        "SCC",
        "CURRENT_CLINICAL_CONCEPT_COUNT",
        "CURRENT_CLINICAL_CONCEPT_DENSITY",
        "CURRENT_CLINICAL_TRUE_COUNT",
        "CURRENT_CLINICAL_OBSERVED_COUNT"
    ]
)
