"""Streamlit app for Assignment 2: Steel Plates Faults binary classification.

Predicts whether a steel plate's fault is "Other_Faults" (1) or one of the
six named fault types (0), using only the 27 measurement columns captured by
the plant's automated optical inspection system.

Loads the five pipelines and bundled test_data.csv already committed to the
repository - no retraining, no download of the original dataset at runtime.
"""
import io

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from model.common import (
    MODEL_FILES,
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    compute_metrics,
    get_positive_score,
    round_metrics,
    validate_uploaded_dataframe,
)

st.set_page_config(page_title="Steel Plates Fault Classifier", layout="wide")


@st.cache_resource
def load_pipelines():
    return {name: joblib.load(path) for name, path in MODEL_FILES.items()}


@st.cache_data
def load_default_test_data():
    return pd.read_csv(TEST_DATA_PATH)


@st.cache_data
def evaluate_all_models(_pipelines, df: pd.DataFrame):
    """Evaluate every pipeline on the given dataframe. Returns {name: metrics_dict}."""
    X = df[PREDICTOR_COLUMNS]
    y = df[TARGET_COLUMN]
    results = {}
    for name, pipeline in _pipelines.items():
        y_pred = pipeline.predict(X)
        y_score = get_positive_score(pipeline, X)
        results[name] = round_metrics(compute_metrics(y, y_pred, y_score))
    return results


pipelines = load_pipelines()

st.title("Steel Plates Fault Classifier")
st.caption(
    "Predicting whether a steel plate's fault is **Other_Faults (1)** or one of the "
    "six named fault types (0), from 27 optical-inspection measurements. "
    "Dataset: [UCI Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults)."
)

# --- Data source: bundled sample or user upload ---
st.sidebar.header("Evaluation data")
st.sidebar.download_button(
    "Download sample test_data.csv",
    data=TEST_DATA_PATH.read_bytes(),
    file_name="test_data.csv",
    mime="text/csv",
)
st.sidebar.markdown(
    "Upload a CSV with the same 27 predictor columns (any order) plus the "
    f"`{TARGET_COLUMN}` column (0 or 1). Missing columns, non-numeric values, "
    "or invalid labels will be reported as an error."
)
uploaded_file = st.sidebar.file_uploader("Upload labeled test CSV", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not re-raised
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()

    active_df, error = validate_uploaded_dataframe(raw_df)
    if error:
        st.error(f"Invalid upload: {error}")
        st.stop()

    data_source = f"Uploaded file: {uploaded_file.name}"
else:
    active_df = load_default_test_data()
    data_source = "Bundled sample (test_data.csv)"

st.subheader("Active dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Source", data_source)
col2.metric("Evaluated rows", len(active_df))
single_class = active_df[TARGET_COLUMN].nunique() < 2
col3.metric("Classes present", "1 (AUC unavailable)" if single_class else "2")

# --- Model selection ---
st.subheader("Model results")
model_name = st.selectbox("Select a model", list(pipelines.keys()))

all_results = evaluate_all_models(pipelines, active_df)
selected_metrics = all_results[model_name]

metric_cols = st.columns(6)
for col, (metric_name, value) in zip(metric_cols, selected_metrics.items()):
    display_value = "N/A" if value != value else f"{value:.4f}"  # NaN check
    col.metric(metric_name, display_value)

# --- Confusion matrix ---
X_active = active_df[PREDICTOR_COLUMNS]
y_active = active_df[TARGET_COLUMN]
y_pred_active = pipelines[model_name].predict(X_active)
cm = confusion_matrix(y_active, y_pred_active, labels=[0, 1])
cm_df = pd.DataFrame(
    cm,
    index=pd.MultiIndex.from_product([["Actual"], ["0: Named fault", "1: Other_Faults"]]),
    columns=pd.MultiIndex.from_product([["Predicted"], ["0: Named fault", "1: Other_Faults"]]),
)
st.write("Confusion matrix")
st.dataframe(cm_df, width="content")

# --- Comparison table across all five models ---
st.subheader("Comparison across all five models (active data)")
comparison_df = pd.DataFrame(all_results).T
comparison_df.index.name = "ML Model Name"
st.dataframe(comparison_df, width="stretch")

st.caption(
    "AUC shows as NaN/N/A when the active data contains only one class, "
    "since ROC AUC is undefined for a single-class evaluation set."
)
