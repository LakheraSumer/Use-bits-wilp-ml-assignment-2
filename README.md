# Use-bits-wilp-ml-assignment-2

**Student:** Sumer Lakhera | **BITS ID:** 2025NS04051 | **Programme:** M.Tech (DSE), BITS Pilani WILP
**Course:** Machine Learning — Assignment 2 (classification models and Streamlit deployment)

## a. Problem statement

> *Draft below — please read and personalize before submission; it should reflect your own understanding.*

An automated optical inspection system on a steel production line scans every plate and records 27
measurements about each detected fault: its location and size on the plate, its perimeter, its
luminosity, the steel type, and several derived shape/edge indices. The system already recognizes
six specific, well-understood fault types (Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps).
This project asks: **can these 27 measurements alone tell apart the six known fault types from
"Other_Faults" — defects that don't fit any established category?**

This matters operationally because `Other_Faults` plates are exactly the ones that most likely need
a human inspector's judgement; a model that reliably flags them lets quality-control staff focus
their attention where automatic classification doesn't already help, while plates with well-understood
faults keep moving through the existing automated routing.

## b. Dataset description

- **Source:** [UCI Machine Learning Repository — Steel Plates Faults (ID 198)](https://archive.ics.uci.edu/dataset/198/steel+plates+faults), CC BY 4.0, Buscema, Terzi & Tastle (2010).
- **Rows:** 1,941 steel plate fault records (verified by direct download and inspection), 0 missing values, 0 duplicate rows.
- **Predictors:** 27 genuine numeric measurement columns (bounding-box coordinates, pixel area, perimeter, luminosity statistics, steel type indicators, plate thickness, and derived geometric/luminosity indices). No identifier columns are used as predictors.
- **Target:** `Other_Faults` (binary). The original dataset encodes seven mutually-exclusive one-hot fault columns (`Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`, `Bumps`, `Other_Faults`); we use `Other_Faults` as the target and **exclude all seven fault-label columns from the model inputs** to prevent leakage.
- **Class distribution:** 673 `Other_Faults` (34.7%) vs. 1,268 named-fault plates (65.3%) — mild imbalance.
- **Missing values / duplicates:** none.
- **Split:** stratified 80/20 train/test split, fixed `random_state=42` (no repeated entities or time ordering in this dataset that would require a different split strategy). Train: 1,552 rows (538 positive). Test: 389 rows (135 positive), saved verbatim as [`test_data.csv`](test_data.csv) with the original predictor names and the `Other_Faults` target column.
- **Preprocessing:** `StandardScaler` inside the pipeline for the three models sensitive to feature scale (Logistic Regression, KNN, SVM); Decision Tree and Naive Bayes use the raw measurements directly. All preprocessing is fit on the training split only.

## c. GitHub repository link

<https://github.com/LakheraSumer/Use-bits-wilp-ml-assignment-2>

## d. Models used

All five models share the same stratified 80/20 split and the same held-out `test_data.csv`, evaluated with one shared metrics function (`model/common.py`) used by both training and the Streamlit app. Positive class = `Other_Faults` (1).

**Table 1 — Metrics (held-out test set, 389 rows)**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7198 | 0.7623 | 0.6667 | 0.3852 | 0.4883 | 0.3363 |
| Decision Tree | 0.7661 | 0.7393 | 0.6667 | 0.6519 | 0.6592 | 0.4812 |
| K-Nearest Neighbors | 0.7815 | 0.8189 | 0.7273 | 0.5926 | 0.6531 | **0.5015** |
| Naive Bayes (Gaussian) | 0.5116 | 0.6838 | 0.4121 | 0.9556 | 0.5759 | 0.2775 |
| Support Vector Machine | 0.7661 | 0.7971 | 0.7391 | 0.5037 | 0.5991 | 0.4584 |

Primary metric: **MCC** (Matthews Correlation Coefficient) — chosen because it uses all four confusion-matrix cells and stays reliable under the dataset's mild class imbalance (65/35), unlike plain Accuracy.

**Table 2 — Observations**

> *Draft below — please read, verify against your own run of the app, and rewrite in your own words before submission.*

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Lowest MCC (0.3363) and lowest Recall (0.3852) of the five: it misses most true `Other_Faults` plates. A single linear boundary appears too simple to separate the classes well — several predictors are heavily skewed with very different scales (e.g. `Y_Minimum` ranges from 6,712 to ~13,000,000), which a linear model exploits less effectively than the non-linear models here. |
| Decision Tree | Second-best MCC (0.4812), with Precision and Recall close to each other (0.6667 / 0.6519) — a reasonably balanced model. It can carve out non-linear, axis-aligned regions suited to the geometric indices, but it was trained with no depth limit, so some of this held-out performance may not generalize as well to new plates as a pruned/tuned tree would. |
| K-Nearest Neighbors | Best MCC (0.5015), best Accuracy (0.7815) and best AUC (0.8189) of all five. With `StandardScaler` in the pipeline, every one of the 27 measurements contributes proportionally to distance instead of being dominated by large-magnitude ones (e.g. `Sum_of_Luminosity`). This suggests that plates with similar measurement profiles genuinely tend to share fault types — a strong signal for a distance-based model. |
| Naive Bayes (Gaussian) | Lowest MCC (0.2775) and lowest Accuracy (0.5116) — worse than always predicting the majority class — despite by far the *highest* Recall (0.9556). Gaussian Naive Bayes assumes the 27 features are independent given the class, but several are mathematically derived from each other (e.g. `LogOfAreas` is a log transform of `Pixels_Areas`). That broken independence assumption appears to bias the model toward over-predicting `Other_Faults`. |
| Support Vector Machine | Third-best MCC (0.4584), and the **highest Precision** of all five models (0.7391): when it predicts `Other_Faults`, it is usually right. With an RBF kernel on standardized features it finds a more conservative boundary than KNN or the Decision Tree, trading some Recall (0.5037) for that precision. |
| **Overall winner for this dataset** | **K-Nearest Neighbors** — highest MCC, Accuracy and AUC among the five models on the identical held-out test set, indicating that local similarity between plates' measurement profiles is the most useful signal for separating `Other_Faults` from the six named fault types in this dataset. |

## Live app

Streamlit Community Cloud: <https://bits-wilp-ml-assignment-2-sumer.streamlit.app/> — public, no sign-in required.

## Setup, training, and launching

### 1. First-time setup (BITS Virtual Lab VM)

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 2. Training (optional — the repository already ships five fitted pipelines and `test_data.csv`)

Only needed if you want to retrain from scratch. Download `Faults.NNA` and `Faults27x7_var` from the
[UCI Steel Plates Faults page](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) into a
`data_raw/` folder at the project root, then:

```bash
./.venv/Scripts/python.exe model/train.py
```

This rewrites `test_data.csv` and all five `model/*.joblib` pipelines.

### 3. Launch the app (one command, handles a path with spaces, no manual activation needed)

```bash
./.venv/Scripts/streamlit.exe run app.py
```

### 4. Testing with the sample CSV

Open the app, use the **Download sample test_data.csv** button in the sidebar, then re-upload that
same file via **Upload labeled test CSV** — the metrics and confusion matrix should be identical to
the default (no-upload) view, since it's the same held-out data the app loads by default.

### 5. Verification scripts

```bash
./.venv/Scripts/python.exe model/verify_reload.py          # reloads saved pipelines, checks all 30 metric values match training exactly
./.venv/Scripts/python.exe model/test_upload_scenarios.py  # exercises reordered/valid-subset/single-class/invalid upload handling
```
