"""Final audit: recompute all 30 metric values at full (unrounded) precision
from the committed test_data.csv and saved pipelines, determine the winner
for EACH metric independently, and print supporting confusion-matrix counts
(TP/FP/TN/FN) for every model so observations can cite real counts.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.common import (  # noqa: E402
    MODEL_FILES,
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    compute_metrics,
    get_positive_score,
)

df = pd.read_csv(TEST_DATA_PATH)
X = df[PREDICTOR_COLUMNS]
y = df[TARGET_COLUMN]

rows = {}
for name, path in MODEL_FILES.items():
    pipeline = joblib.load(path)
    y_pred = pipeline.predict(X)
    y_score = get_positive_score(pipeline, X)
    metrics = compute_metrics(y, y_pred, y_score)  # UNROUNDED
    rows[name] = metrics

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    print(f"{name}")
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}  (total={tn+fp+fn+tp}, actual positives={tp+fn}, actual negatives={tn+fp})")
    for k, v in metrics.items():
        print(f"  {k:10s} = {v:.10f}")
    print()

print("=" * 70)
print("WINNER PER METRIC (unrounded)")
metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
for m in metric_names:
    best_model = max(rows, key=lambda n: rows[n][m])
    best_val = rows[best_model][m]
    print(f"  {m:10s} -> {best_model}  ({best_val:.10f})")
    # show runner-up for close calls
    sorted_models = sorted(rows, key=lambda n: rows[n][m], reverse=True)
    for n in sorted_models[:3]:
        print(f"      {n:25s} {rows[n][m]:.10f}")
