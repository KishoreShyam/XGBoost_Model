import pandas as pd
import numpy as np

DATA_PATH = "DataSet/future_model/Cognizant_14K_Model_Ready.csv"

TARGETS = [
    "FUTURE_HIGH_UTILIZATION",
    "FUTURE_CLINICAL_DETERIORATION",
    "FUTURE_HEALTHCARE_ESCALATION"
]

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("=" * 70)
print("MULTI-TARGET OUTCOME AUDIT")
print("=" * 70)

print("\nTarget distributions:")

for target in TARGETS:

    print("\n", target)

    print(
        df[target]
        .value_counts()
        .sort_index()
    )


# ============================================================
# TARGET CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("TARGET CORRELATION")
print("=" * 70)

correlation = df[TARGETS].corr()

print(
    correlation.round(4)
)


# ============================================================
# OUTCOME OVERLAP
# ============================================================

print("\n" + "=" * 70)
print("OUTCOME OVERLAP")
print("=" * 70)

for target_a in TARGETS:

    for target_b in TARGETS:

        if target_a >= target_b:
            continue

        both_positive = (
            (df[target_a] == 1) &
            (df[target_b] == 1)
        ).sum()

        a_positive = (
            df[target_a] == 1
        ).sum()

        b_positive = (
            df[target_b] == 1
        ).sum()

        print(
            f"\n{target_a}"
        )

        print(
            f"vs {target_b}"
        )

        print(
            "Both positive:",
            both_positive
        )

        print(
            f"P({target_b}=1 | {target_a}=1): "
            f"{both_positive / a_positive:.4f}"
        )

        print(
            f"P({target_a}=1 | {target_b}=1): "
            f"{both_positive / b_positive:.4f}"
        )


# ============================================================
# COMBINED OUTCOME PATTERNS
# ============================================================

print("\n" + "=" * 70)
print("COMBINED OUTCOME PATTERNS")
print("=" * 70)

patterns = (
    df[TARGETS]
    .astype(str)
    .agg("_".join, axis=1)
)

print(
    patterns
    .value_counts()
    .sort_index()
)


# ============================================================
# NUMBER OF POSITIVE FUTURE OUTCOMES
# ============================================================

print("\n" + "=" * 70)
print("NUMBER OF POSITIVE FUTURE OUTCOMES")
print("=" * 70)

positive_count = (
    df[TARGETS]
    .sum(axis=1)
)

print(
    positive_count
    .value_counts()
    .sort_index()
)

print(
    "\nPercentage:"
)

print(
    positive_count
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)