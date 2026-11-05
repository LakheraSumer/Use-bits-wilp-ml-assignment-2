# Use-bits-wilp-ml-assignment-2

**Student:** Sumer Lakhera | **BITS ID:** 2025NS04051 | **Programme:** M.Tech (DSE), BITS Pilani WILP

**Course:** Machine Learning - Assignment 2 (classification models and Streamlit deployment)

## a. Problem statement

Every row in this dataset represents a steel plate fault that has already been detected. The task is
binary classification, not fault detection:

- Positive class `1` means `Other_Faults`.
- Negative class `0` means one of the six named fault types: `Pastry`, `Z_Scratch`, `K_Scatch`,
  `Stains`, `Dirtiness`, or `Bumps`.
- The models use the 27 measurement features recorded for each fault.
- All seven fault-label columns are excluded from the model inputs. `Other_Faults` is used only as
  the target, and the other six label columns are not used as predictors.

The experiment tests whether the 27 measurements can separate `Other_Faults` from the six named
fault types. A possible application would be to use the prediction to help prioritize cases for human
review. However, this dataset and experiment do not establish which plates actually require human
inspection or how predictions should be used in a production quality-control process.

## b. Dataset description

- **Source:** [UCI Machine Learning Repository - Steel Plates Faults (ID 198)](https://archive.ics.uci.edu/dataset/198/steel+plates+faults), CC BY 4.0, Buscema, Terzi & Tastle (2010).
- **Rows:** 1,941 steel plate fault records, 0 missing values, and 0 duplicate rows.
- **Predictors:** 27 numeric measurement columns covering bounding-box coordinates, pixel area, perimeter, luminosity statistics, steel type indicators, plate thickness, and derived geometric/luminosity indices. No identifier or fault-label column is used as a predictor.
- **Target:** `Other_Faults` (binary). The original data has seven mutually exclusive one-hot label columns: `Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`, `Bumps`, and `Other_Faults`.
- **Class distribution:** 673 `Other_Faults` records (34.7%) and 1,268 named-fault records (65.3%).
- **Split:** stratified 80/20 train/test split with `random_state=42`. Training has 1,552 rows (538 positive); testing has 389 rows (135 positive). The same held-out rows are saved as [`test_data.csv`](test_data.csv) and used for every model.
- **Preprocessing:** `StandardScaler` is inside the Logistic Regression, KNN, and SVM pipelines. Decision Tree and Gaussian Naive Bayes use the raw measurements. Each scaler and model is fit only on the training split.

## c. GitHub repository link

<https://github.com/LakheraSumer/Use-bits-wilp-ml-assignment-2>

## d. Models used

All five saved pipelines are evaluated on the same 389-row held-out `test_data.csv`. Training and the
Streamlit app use the shared metric implementation in `model/common.py`. The positive class for
Precision, Recall, F1, MCC, and AUC is `Other_Faults = 1`.

**Table 1 - Metrics (held-out test set, 389 rows)**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7198 | 0.7623 | 0.6667 | 0.3852 | 0.4883 | 0.3363 |
| Decision Tree | 0.7661 | 0.7393 | 0.6667 | 0.6519 | **0.6592** | 0.4812 |
| K-Nearest Neighbors | **0.7815** | **0.8189** | 0.7273 | 0.5926 | 0.6531 | **0.5015** |
| Naive Bayes (Gaussian) | 0.5116 | 0.6838 | 0.4121 | **0.9556** | 0.5759 | 0.2775 |
| Support Vector Machine | 0.7661 | 0.7971 | **0.7391** | 0.5037 | 0.5991 | 0.4584 |

Primary metric: **MCC** (Matthews Correlation Coefficient). It uses all four confusion-matrix cells,
which makes it more informative than Accuracy alone for this 65/35 class distribution.

**Table 2 - Observations**

All counts below are from the same 389-row held-out set: 135 records are `Other_Faults` and 254 are
one of the six named fault types.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Correctly identified 52 of 135 `Other_Faults` records and missed 83 (Recall 0.3852). It produced 26 false positives among the 254 named-fault records (Precision 0.6667). Its MCC is 0.3363, the **fourth-highest of the five models**, ahead of Gaussian Naive Bayes. A possible explanation for the lower score is that a linear decision boundary may not describe the class separation well; this explanation was not directly tested. |
| Decision Tree | Correctly identified 88 of 135 `Other_Faults` records and missed 47 (Recall 0.6519). It produced 44 false positives (Precision 0.6667). It has the second-highest MCC (0.4812) and the **highest F1 (0.6592)**. The tree was trained without a depth limit, so its performance should not be treated as evidence that an untuned tree will generalize equally well to new operational data. |
| K-Nearest Neighbors | Correctly identified 80 of 135 `Other_Faults` records and missed 55 (Recall 0.5926). It produced 30 false positives (Precision 0.7273). It has the **highest MCC (0.5015), Accuracy (0.7815), and AUC (0.8189)**. It does not lead every metric: Decision Tree has the highest F1, SVM has the highest Precision, and Gaussian Naive Bayes has the highest Recall. |
| Naive Bayes (Gaussian) | Correctly identified 129 of 135 `Other_Faults` records and missed only 6, giving the **highest Recall (0.9556)**. It also produced 184 false positives among 254 named-fault records, so it has the lowest Precision (0.4121), Accuracy (0.5116), and **MCC (0.2775)**. A possible explanation is that Gaussian Naive Bayes assumes conditional feature independence while several measurements are mathematically related; this explanation was not directly tested. |
| Support Vector Machine | Correctly identified 68 of 135 `Other_Faults` records and missed 67 (Recall 0.5037). It produced 24 false positives, the fewest of the five models, and therefore has the **highest Precision (0.7391)**. Its MCC (0.4584) is third-highest. |
| **Overall winner for this dataset** | **K-Nearest Neighbors**, using MCC as the primary metric. Its MCC is 0.5015, the highest of the five, and it also leads Accuracy and AUC. Decision Tree leads F1, SVM leads Precision, and Gaussian Naive Bayes leads Recall. |

## Live app

Streamlit Community Cloud: <https://bits-wilp-ml-assignment-2-sumer.streamlit.app/> - public, no sign-in required.

## Setup, training, and launching

### 1. First-time setup (BITS Virtual Lab VM)

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 2. Training (optional - the repository already contains five fitted pipelines and `test_data.csv`)

Download `Faults.NNA` and `Faults27x7_var` from the
[UCI Steel Plates Faults page](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) into a
`data_raw/` folder at the project root, then run:

```bash
./.venv/Scripts/python.exe model/train.py
```

This rewrites `test_data.csv`, `model/training_metrics.csv`, and the five `model/*.joblib` pipelines.

### 3. Launch the app

```bash
./.venv/Scripts/streamlit.exe run app.py
```

### 4. Testing with the sample CSV

Open the app, use **Download sample test_data.csv** in the sidebar, and upload that same file through
**Upload labeled test CSV**. The metrics and confusion matrix should be identical to the default view
because both use the same held-out rows.

### 5. Verification scripts

```bash
./.venv/Scripts/python.exe model/verify_reload.py
./.venv/Scripts/python.exe model/verify_integrity.py
./.venv/Scripts/python.exe model/test_upload_scenarios.py
```

