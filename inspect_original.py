import pandas as pd

DATA_PATH = "DataSet/Synthea_Original_Dataset.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

print("=" * 70)
print("ORIGINAL COGNIZANT DATASET — COMPLETE SCHEMA")
print("=" * 70)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nCOLUMN LIST:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes.to_string())

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isna().sum()

for column, count in missing.items():
    print(
        f"{column}: {count} "
        f"({count / len(df) * 100:.2f}%)"
    )

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print(
    df.head().to_string()
)

print("\n" + "=" * 70)
print("NUMERIC SUMMARY")
print("=" * 70)

print(
    df.describe().T.to_string()
)