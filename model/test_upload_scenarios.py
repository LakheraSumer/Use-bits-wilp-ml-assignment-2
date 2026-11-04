"""Exercises validate_uploaded_dataframe() (used by app.py) against the
scenarios the evaluator is expected to try: reordered columns, a valid
subset, a single-class file, and invalid data.
"""
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.common import PREDICTOR_COLUMNS, TARGET_COLUMN, TEST_DATA_PATH, validate_uploaded_dataframe  # noqa: E402

df = pd.read_csv(TEST_DATA_PATH)

print("1) Reordered columns (valid)")
reordered = df[[TARGET_COLUMN] + PREDICTOR_COLUMNS[::-1]]
clean, err = validate_uploaded_dataframe(reordered)
print("   error:", err, "| rows:", None if clean is None else len(clean))
assert err is None and len(clean) == len(df)

print("2) Valid subset (first 50 rows)")
subset = df.head(50)
clean, err = validate_uploaded_dataframe(subset)
print("   error:", err, "| rows:", None if clean is None else len(clean))
assert err is None and len(clean) == 50

print("3) Single-class file (all Other_Faults=0)")
single_class = df[df[TARGET_COLUMN] == 0].head(30)
clean, err = validate_uploaded_dataframe(single_class)
print("   error:", err, "| rows:", None if clean is None else len(clean), "| nunique:", clean[TARGET_COLUMN].nunique())
assert err is None and clean[TARGET_COLUMN].nunique() == 1

print("4) Empty file")
clean, err = validate_uploaded_dataframe(df.iloc[0:0])
print("   error:", err)
assert err is not None

print("5) Missing columns")
missing_cols = df.drop(columns=["X_Minimum", "Y_Minimum"])
clean, err = validate_uploaded_dataframe(missing_cols)
print("   error:", err)
assert err is not None

print("6) Invalid target labels (e.g. 2 instead of 0/1)")
bad_labels = df.copy()
bad_labels.loc[0, TARGET_COLUMN] = 2
clean, err = validate_uploaded_dataframe(bad_labels)
print("   error:", err)
assert err is not None

print("7) Non-numeric predictor value (simulated via raw CSV text, like a real upload)")
csv_text = df.head(5).to_csv(index=False).splitlines()
header = csv_text[0].split(",")
x_min_idx = header.index("X_Minimum")
row = csv_text[1].split(",")
row[x_min_idx] = "not_a_number"
csv_text[1] = ",".join(row)
bad_values = pd.read_csv(io.StringIO("\n".join(csv_text)))
clean, err = validate_uploaded_dataframe(bad_values)
print("   error:", err)
assert err is not None

print("8) Missing predictor value (empty cell, like a real upload)")
csv_text = df.head(5).to_csv(index=False).splitlines()
row = csv_text[1].split(",")
row[x_min_idx] = ""
csv_text[1] = ",".join(row)
bad_nan = pd.read_csv(io.StringIO("\n".join(csv_text)))
clean, err = validate_uploaded_dataframe(bad_nan)
print("   error:", err)
assert err is not None

print("\nALL UPLOAD SCENARIOS PASSED")
