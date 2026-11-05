"""Streamlit app for Assignment 2: Steel Plates Faults binary classification.

Predicts whether a steel plate's fault is "Other_Faults" (1) or one of the
six named fault types (0), using only the 27 measurement columns captured by
the plant's automated optical inspection system.

Loads the five pipelines and bundled test_data.csv already committed to the
repository - no retraining, no download of the original dataset at runtime.
"""
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics import confusion_matrix

from model.common import (
    CLASS_NAMES,
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

METRIC_HELP = {
    "Accuracy": "Share of all predictions that were correct.",
    "AUC": "Probability a random Other_Faults plate scores higher than a random named-fault plate. Undefined for single-class data.",
    "Precision": "Of plates predicted Other_Faults, the share that truly are.",
    "Recall": "Of plates that truly are Other_Faults, the share the model caught.",
    "F1": "Harmonic mean of Precision and Recall.",
    "MCC": "Matthews Correlation Coefficient — uses all four confusion-matrix counts; the primary metric for this assignment.",
}

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    div[data-testid="stMetric"] { background: rgba(127,127,127,0.08); border-radius: 8px; padding: 10px 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def style_comparison_table(df: pd.DataFrame):
    return df.style.highlight_max(
        axis=0, props="background-color: rgba(46, 160, 67, 0.28); font-weight: 600;"
    ).format("{:.4f}", na_rep="N/A")


def style_confusion_matrix(cm_df: pd.DataFrame):
    def _tint(_):
        styles = pd.DataFrame("", index=cm_df.index, columns=cm_df.columns)
        for i in range(2):
            for j in range(2):
                color = "rgba(46, 160, 67, 0.28)" if i == j else "rgba(220, 53, 69, 0.18)"
                styles.iloc[i, j] = f"background-color: {color};"
        return styles

    return cm_df.style.apply(_tint, axis=None)


pipelines = load_pipelines()

st.title("Steel Plates Fault Classifier")
st.caption(
    f"**0** = {CLASS_NAMES[0]} &nbsp;·&nbsp; **1** = {CLASS_NAMES[1]}. "
    "Dataset: [UCI Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults).",
    unsafe_allow_html=True,
)

with st.expander("New here? How to use this app", expanded=True):
    st.markdown(
        "1. This page already shows results on the **bundled held-out test set** (389 rows) — nothing to click to get started.\n"
        "2. **Selected model** tab (below): pick any of the five models from the dropdown to see its six metrics "
        "and confusion matrix.\n"
        "3. **Compare all five models** tab: see all five side by side on the same data, with the **best score per "
        "metric highlighted in green**.\n"
        "4. To test with your own labeled data: use **Download sample test_data.csv** in the sidebar to see the "
        "expected format, then **Upload labeled test CSV** there. Both tabs update together."
    )

# --- Sidebar: data source ---
with st.sidebar:
    st.header("Evaluation data")
    st.download_button(
        "Download sample test_data.csv",
        data=TEST_DATA_PATH.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        width="stretch",
    )
    st.caption(
        "Upload a CSV with the same 27 predictor columns (any order) plus the "
        f"`{TARGET_COLUMN}` column (0 or 1). Missing columns, non-numeric values, "
        "or invalid labels are reported as an error."
    )
    uploaded_file = st.file_uploader("Upload labeled test CSV", type=["csv"])

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

    data_source = f"Uploaded file — {uploaded_file.name}"
else:
    active_df = load_default_test_data()
    data_source = "Bundled sample — test_data.csv"

st.divider()

with st.container(border=True):
    st.subheader("Active dataset")
    left, mid, right = st.columns([2, 1, 1])
    left.markdown(f"**Source**  \n{data_source}")
    mid.metric("Evaluated rows", len(active_df))
    single_class = active_df[TARGET_COLUMN].nunique() < 2
    right.metric("Classes present", "1 (AUC unavailable)" if single_class else "2")

all_results = evaluate_all_models(pipelines, active_df)

tab_single, tab_compare = st.tabs(["Selected model", "Compare all five models"])

with tab_single:
    model_name = st.selectbox("Select a model", list(pipelines.keys()))
    selected_metrics = all_results[model_name]

    metric_row1 = st.columns(3)
    metric_row2 = st.columns(3)
    for col, (metric_name, value) in zip(metric_row1 + metric_row2, selected_metrics.items()):
        display_value = "N/A" if value != value else f"{value:.4f}"  # NaN check
        col.metric(metric_name, display_value, help=METRIC_HELP[metric_name])

    X_active = active_df[PREDICTOR_COLUMNS]
    y_active = active_df[TARGET_COLUMN]
    y_pred_active = pipelines[model_name].predict(X_active)
    cm = confusion_matrix(y_active, y_pred_active, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=pd.MultiIndex.from_product([["Actual"], [f"0: {CLASS_NAMES[0]}", f"1: {CLASS_NAMES[1]}"]]),
        columns=pd.MultiIndex.from_product([["Predicted"], [f"0: {CLASS_NAMES[0]}", f"1: {CLASS_NAMES[1]}"]]),
    )

    with st.container(border=True):
        st.markdown("**Confusion matrix** — green = correct, red = incorrect")
        st.dataframe(style_confusion_matrix(cm_df), width="content")

with tab_compare:
    with st.container(border=True):
        st.markdown("**Comparison across all five models (active data)** — best value per metric highlighted")
        comparison_df = pd.DataFrame(all_results).T
        comparison_df.index.name = "ML Model Name"
        st.dataframe(style_comparison_table(comparison_df), width="stretch")
        st.caption(
            "AUC shows as NaN/N/A when the active data contains only one class, "
            "since ROC AUC is undefined for a single-class evaluation set."
        )
