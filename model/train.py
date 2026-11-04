"""Training source for Assignment 2.

Loads the raw UCI Steel Plates Faults data (data_raw/Faults.NNA, downloaded
once from https://archive.ics.uci.edu/dataset/198/steel+plates+faults),
builds a stratified train/test split, fits five scikit-learn pipelines on the
training rows only, evaluates them on the held-out test rows, and saves:

  - test_data.csv              held-out rows (27 predictors + Other_Faults), at project root
  - model/*.joblib             five fitted pipelines
  - model/training_metrics.csv the six metrics for each model, on the held-out set

Run from the project root:
    .venv\\Scripts\\python.exe model\\train.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.common import (  # noqa: E402
    FAULT_COLUMNS,
    MODEL_DIR,
    MODEL_FILES,
    PREDICTOR_COLUMNS,
    PROJECT_ROOT,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    compute_metrics,
    get_positive_score,
    round_metrics,
)

RAW_DATA_DIR = PROJECT_ROOT / "data_raw"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw_data() -> pd.DataFrame:
    columns_path = RAW_DATA_DIR / "Faults27x7_var"
    data_path = RAW_DATA_DIR / "Faults.NNA"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Download Faults.NNA and Faults27x7_var from "
            "https://archive.ics.uci.edu/dataset/198/steel+plates+faults into data_raw/."
        )
    columns = columns_path.read_text().split()
    df = pd.read_csv(data_path, sep="\t", header=None, names=columns)
    return df


def build_pipelines():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Naive Bayes (Gaussian)": Pipeline([
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE)),
        ]),
    }


def main():
    df = load_raw_data()
    leaked = set(PREDICTOR_COLUMNS) & set(FAULT_COLUMNS)
    assert not leaked, f"Fault-label columns must not be used as predictors: {leaked}"

    X = df[PREDICTOR_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Save held-out rows exactly as the app/evaluator will consume them.
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"Saved {len(test_df)} held-out rows to {TEST_DATA_PATH}")
    print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")
    print(f"Train class balance:\n{y_train.value_counts()}")
    print(f"Test class balance:\n{y_test.value_counts()}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pipelines = build_pipelines()
    results = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = get_positive_score(pipeline, X_test)
        metrics = compute_metrics(y_test, y_pred, y_score)
        results[name] = round_metrics(metrics)

        joblib.dump(pipeline, MODEL_FILES[name])
        print(f"\n{name}: saved -> {MODEL_FILES[name].name}")
        for k, v in results[name].items():
            print(f"  {k}: {v}")

    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = "ML Model Name"
    metrics_path = MODEL_DIR / "training_metrics.csv"
    metrics_df.to_csv(metrics_path)
    print(f"\nSaved metrics table to {metrics_path}")
    print(metrics_df)


if __name__ == "__main__":
    main()
