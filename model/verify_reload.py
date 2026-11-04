"""Reload the saved pipelines and test_data.csv, recompute all 30 metric
values, and confirm they match model/training_metrics.csv within tolerance.

This is the acceptance check described in the assignment: the app must
reproduce the experiment's numbers exactly, not approximately.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.common import (  # noqa: E402
    MODEL_FILES,
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    compute_metrics,
    get_positive_score,
    round_metrics,
)

TOLERANCE = 1e-6


def main():
    df = pd.read_csv(TEST_DATA_PATH)
    X = df[PREDICTOR_COLUMNS]
    y = df[TARGET_COLUMN]

    expected = pd.read_csv(Path(__file__).resolve().parent / "training_metrics.csv", index_col=0)

    all_ok = True
    for name, path in MODEL_FILES.items():
        pipeline = joblib.load(path)
        y_pred = pipeline.predict(X)
        y_score = get_positive_score(pipeline, X)
        actual = round_metrics(compute_metrics(y, y_pred, y_score))

        for metric_name, actual_value in actual.items():
            expected_value = expected.loc[name, metric_name]
            diff = abs(actual_value - expected_value)
            status = "OK" if diff <= TOLERANCE else "MISMATCH"
            if status == "MISMATCH":
                all_ok = False
            print(f"{name:25s} {metric_name:10s} expected={expected_value:.4f} actual={actual_value:.4f} {status}")

    print("\nALL 30 VALUES MATCH" if all_ok else "\nSOME VALUES DID NOT MATCH")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
