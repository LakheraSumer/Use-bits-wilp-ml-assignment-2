"""Verify split integrity, preprocessing fit, saved-pipeline equivalence, and AUC orientation.

This script does not rewrite any artifact. It reconstructs the documented split from the raw data,
fits fresh deterministic pipelines on the training rows only, and compares them with the committed
test data and saved pipelines.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.common import (  # noqa: E402
    FAULT_COLUMNS,
    MODEL_FILES,
    POSITIVE_LABEL,
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    get_positive_score,
)
from model.train import RANDOM_STATE, TEST_SIZE, build_pipelines, load_raw_data  # noqa: E402


def main():
    raw = load_raw_data()
    assert len(raw) == 1941
    assert int(raw.isna().sum().sum()) == 0
    assert int(raw.duplicated().sum()) == 0
    assert not (set(PREDICTOR_COLUMNS) & set(FAULT_COLUMNS))

    X = raw[PREDICTOR_COLUMNS]
    y = raw[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    expected_test = X_test.copy()
    expected_test[TARGET_COLUMN] = y_test.to_numpy()
    committed_test = pd.read_csv(TEST_DATA_PATH)
    pd.testing.assert_frame_equal(
        committed_test.reset_index(drop=True),
        expected_test.reset_index(drop=True),
        check_dtype=False,
    )
    print("PASS: committed test_data.csv is the exact reconstructed 389-row held-out split")

    fresh_pipelines = build_pipelines()
    for name, fresh_pipeline in fresh_pipelines.items():
        saved_pipeline = joblib.load(MODEL_FILES[name])
        fresh_pipeline.fit(X_train, y_train)

        saved_prediction = saved_pipeline.predict(X_test)
        fresh_prediction = fresh_pipeline.predict(X_test)
        np.testing.assert_array_equal(saved_prediction, fresh_prediction)

        saved_score = get_positive_score(saved_pipeline, X_test)
        fresh_score = get_positive_score(fresh_pipeline, X_test)
        np.testing.assert_allclose(saved_score, fresh_score, rtol=1e-10, atol=1e-12)

        classes = list(saved_pipeline.classes_)
        assert classes == [0, POSITIVE_LABEL]
        if hasattr(saved_pipeline, "predict_proba"):
            positive_index = classes.index(POSITIVE_LABEL)
            manual_score = saved_pipeline.predict_proba(X_test)[:, positive_index]
            score_method = "predict_proba[:, class 1]"
        else:
            manual_score = saved_pipeline.decision_function(X_test)
            score_method = "decision_function toward classes_[1] = 1"
        np.testing.assert_allclose(saved_score, manual_score, rtol=1e-12, atol=1e-12)

        auc = roc_auc_score(y_test, saved_score)
        inverse_auc = roc_auc_score(y_test, -saved_score)
        print(
            f"PASS: {name:24s} saved=fresh training-only predictions/scores; "
            f"AUC={auc:.10f}, reversed={inverse_auc:.10f}, source={score_method}"
        )

        if "scaler" in saved_pipeline.named_steps:
            scaler = saved_pipeline.named_steps["scaler"]
            np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(scaler.var_, X_train.var(axis=0, ddof=0), rtol=1e-12, atol=1e-12)
            assert int(scaler.n_samples_seen_) == len(X_train)

    print("PASS: every fitted scaler contains statistics from the 1,552 training rows only")
    print("ALL INTEGRITY CHECKS PASSED")


if __name__ == "__main__":
    main()
