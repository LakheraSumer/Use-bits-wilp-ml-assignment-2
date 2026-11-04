"""Shared feature/column definitions and evaluation logic.

Imported by both model/train.py (experiment) and app.py (Streamlit app) so that
metric computation is identical in both places.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"

FAULT_COLUMNS = [
    "Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults",
]

PREDICTOR_COLUMNS = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
    "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity",
    "Maximum_of_Luminosity", "Length_of_Conveyer", "TypeOfSteel_A300",
    "TypeOfSteel_A400", "Steel_Plate_Thickness", "Edges_Index", "Empty_Index",
    "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
    "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index",
    "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas",
]

TARGET_COLUMN = "Other_Faults"
POSITIVE_LABEL = 1
CLASS_NAMES = {0: "Named fault (one of six)", 1: "Other_Faults"}

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "K-Nearest Neighbors": MODEL_DIR / "knn.joblib",
    "Naive Bayes (Gaussian)": MODEL_DIR / "naive_bayes.joblib",
    "Support Vector Machine": MODEL_DIR / "svm.joblib",
}


def get_positive_score(pipeline, X):
    """Probability (or decision score) assigned to the positive class (Other_Faults=1)."""
    if hasattr(pipeline, "predict_proba"):
        classes = list(pipeline.classes_)
        idx = classes.index(POSITIVE_LABEL)
        return pipeline.predict_proba(X)[:, idx]
    return pipeline.decision_function(X)


def compute_metrics(y_true, y_pred, y_score=None):
    """Compute Accuracy, AUC, Precision, Recall, F1, MCC for the positive class (1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["AUC"] = roc_auc_score(y_true, y_score)
    else:
        metrics["AUC"] = float("nan")  # single-class upload: AUC undefined

    # Fixed column order for display
    return {k: metrics[k] for k in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}


def round_metrics(metrics, ndigits=4):
    return {k: (round(v, ndigits) if v == v else v) for k, v in metrics.items()}


def validate_uploaded_dataframe(df):
    """Align an uploaded dataframe to the expected schema. Returns (clean_df, error_message)."""
    if df is None or df.empty:
        return None, "The uploaded file is empty."

    missing_predictors = [c for c in PREDICTOR_COLUMNS if c not in df.columns]
    if missing_predictors:
        return None, f"Missing required predictor column(s): {', '.join(missing_predictors)}"

    if TARGET_COLUMN not in df.columns:
        return None, f"Missing required target column: '{TARGET_COLUMN}'"

    df = df[PREDICTOR_COLUMNS + [TARGET_COLUMN]].copy()

    if df[PREDICTOR_COLUMNS].isna().any().any():
        return None, "One or more predictor values are missing (NaN). Please provide complete rows."

    try:
        df[PREDICTOR_COLUMNS] = df[PREDICTOR_COLUMNS].astype(float)
    except (ValueError, TypeError):
        return None, "One or more predictor columns contain non-numeric values."

    valid_labels = {0, 1}
    try:
        target_values = set(df[TARGET_COLUMN].astype(int).unique())
    except (ValueError, TypeError):
        return None, f"Target column '{TARGET_COLUMN}' must contain only 0/1 values."

    if not target_values.issubset(valid_labels):
        return None, f"Target column '{TARGET_COLUMN}' must contain only 0/1 values (found {sorted(target_values)})."

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df, None
