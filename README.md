# Fraud Detection — Data Drift Monitoring

A production-style data drift monitoring system for the fraud detection model.
Detects when incoming transaction data starts diverging from the training distribution — signaling that the model may need retraining.

---

## Why Drift Monitoring?

A fraud detection model is trained on historical data.
In production, transaction patterns evolve — new fraud techniques, seasonal spending behavior, or shifts in user demographics can cause the input distribution to change.

Without monitoring, the model silently degrades.
This project detects that degradation before it becomes a business problem.

---

## How It Works

```
Training Data (validation split)
        ↓
  Reference Dataset         ← what the model expects
        ↓
  PSI Calculation           ← compare feature distributions
        ↓
  Production Batch          ← what the model is receiving
        ↓
  Decision Layer            ← STABLE or ALERT
```

**PSI (Population Stability Index)** measures how much each feature's distribution has shifted:

| PSI | Interpretation |
|-----|----------------|
| < 0.1 | No drift |
| 0.1 – 0.25 | Moderate drift |
| ≥ 0.25 | Severe drift → retraining recommended |

---

## Project Structure

```
drift-monitoring/
├── data/
│   ├── raw/
│   │   └── creditcard.csv         (local only — excluded from version control)
│   ├── reference/
│   │   ├── features.parquet       (baseline feature distributions)
│   │   └── predictions.parquet    (baseline model predictions)
│   └── production/
│       └── batches/
│           └── batch_001.parquet  (simulated production data)
├── models/
│   ├── fraud_model.pkl            (local only)
│   └── standard_scaler.pkl        (local only)
├── notebooks/
│   ├── create_reference.ipynb         # Step 1
│   ├── create_production_batch.ipynb  # Step 2
│   └── data_drift_detection.ipynb     # Step 3
├── requirements.txt
└── README.md
```

---

## Notebooks

### 1. `create_reference.ipynb`
- Loads the original fraud dataset
- Recreates the exact train/validation split used during model training
- Applies the same preprocessing (StandardScaler on Time & Amount)
- Generates model predictions on the validation set
- Saves reference features and predictions as `.parquet` files

### 2. `create_production_batch.ipynb`
- Simulates incoming production data using the held-out test set
- Applies the same preprocessing pipeline
- Saves as `batch_001.parquet` — ready for drift comparison

### 3. `data_drift_detection.ipynb`
- Loads reference and production datasets
- Calculates PSI for all 30 features
- Visualizes distribution shifts for the most drifted features
- Applies a rule-based decision layer
- Outputs a final verdict: **STABLE** or **ALERT**

---

## Results

```
Severe drift features: 0/30
Severe drift ratio: 0.00%

Final Verdict: STABLE — Minor drift detected. No retraining required.
```

Expected result — reference and production data come from the same dataset (different splits).
In a real system, production batches would contain live transaction data, where drift is expected over time.

---

## Key Design Decisions

**PSI implemented from scratch** — using NumPy only, no external monitoring libraries.
Demonstrates understanding of the metric, not just library usage.

**Same preprocessing pipeline** — reference and production data go through identical transformations, matching real production constraints where the scaler is a fixed artifact.

**Rule-based decision layer** — drift severity is evaluated across all features collectively, not per-feature in isolation. A single drifted feature is noise; 20%+ drifted features is a signal.

**Parquet format** — used for efficient columnar storage, matching industry standards for feature stores and data pipelines.

---

## Dataset

Anonymized credit card transactions. Features V1–V28 are PCA-transformed by the dataset provider.
Time and Amount are the only raw features, scaled during preprocessing.

Source: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
Place `creditcard.csv` under `data/raw/` locally.

---

## Tech Stack

Python · NumPy · Pandas · Matplotlib · Scikit-learn · XGBoost · PyArrow

---

## Related Projects

- [fraud-detection-ml](https://github.com/msaad-dot/fraud-detection-ml) — Model training pipeline
- [fraud-detection-api](https://github.com/msaad-dot/fraud-detection-api) — FastAPI inference service

---

## Author

Mohamed Saad — [GitHub](https://github.com/msaad-dot)
