# SemiGuard — Semiconductor Defect Detection System

Production-grade ML/DL system to predict semiconductor chip failures from 590 sensor readings using the UCI SECOM dataset.

## 🎯 Problem Statement

Semiconductor manufacturing produces defective chips that cost millions in recalls. This system predicts chip failures from sensor data, enabling early defect detection before shipping.

## 📊 Dataset

- **Source**: UCI SECOM Dataset
- **Size**: 1,567 batches × 590 sensors
- **Class Imbalance**: 93.3% PASS / 6.7% FAIL

## 🔧 Approach

### 1. Feature Engineering
- Dropped sensors with >50% missing values (590 → 562)
- Variance thresholding removed near-constant sensors (562 → 297)
- Correlation filtering removed redundant sensors (297 → 195)
- **67% dimensionality reduction**

### 2. Class Imbalance Handling
- Applied SMOTE on training data only (leakage-free split: train/val separated *before* SMOTE)
- Threshold tuning to optimize FAIL detection

### 3. Models Trained

| Model | ROC-AUC | FAIL Recall | FAIL F1 |
|-------|---------|-------------|---------|
| Logistic Regression | 0.664 | 0.43 | 0.23 |
| Random Forest | **0.789** | 0.43 | 0.30 |
| XGBoost (default) | 0.760 | 0.52 | 0.35 |
| XGBoost (tuned) | 0.742 | 0.57 | 0.34 |
| ANN | 0.628 | 0.38 | 0.22 |
| LSTM | 0.650 | 0.40 | 0.24 |
| Stacking Ensemble | 0.761 | **0.67** | 0.26 |

### 4. Explainability
- SHAP for global feature importance
- LIME for individual prediction explanation
- Top predictive sensors identified: `sensor_486`, `sensor_511`, `sensor_59`

### 5. Deployment
- FastAPI REST API for real-time predictions
- Dockerized for platform-independent deployment
- MLflow for experiment tracking

## 🛠️ Tech Stack

**ML/DL**: Scikit-learn, XGBoost, TensorFlow, Keras, SMOTE
**Tuning**: Optuna, GridSearchCV
**Explainability**: SHAP, LIME
**Tracking**: MLflow
**Deployment**: FastAPI, Docker
**Language**: Python 3.12

## 📁 Project Structure

```
secom/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_classical_ml.ipynb
│   ├── 04_deep_learning.ipynb
│   └── 05_ensemble.ipynb
├── models/
├── api/
│   ├── main.py
│   └── requirements.txt
├── reports/
│   └── figures/
├── Dockerfile
└── README.md
```

## 🚀 Running the API

```bash
# Build Docker image
docker build -t semiguard-api .

# Run container
docker run -p 8000:8000 semiguard-api

# Access interactive API docs
http://localhost:8000/docs
```

### Example Request

```json
POST /predict
{
  "features": [0.5, -0.3, 1.2, ...]  // 195 sensor values
}
```

### Example Response

```json
{
  "prediction": "PASS",
  "fail_probability": 0.25
}
```

## 📈 Key Results

- Reduced 590 sensors to 195 (67% reduction) while maintaining predictive power
- Improved FAIL detection recall from 24% (baseline threshold) to 67% (ensemble, tuned threshold)
- Caught and fixed a critical SMOTE data leakage bug that was causing 99.9% fake validation accuracy
- Best overall model: Random Forest (AUC 0.789)
- Best FAIL detection: Stacking Ensemble (Recall 0.67)

## 🔍 Key Learnings

- **Data leakage vigilance**: Caught two significant leakage bugs — label column inclusion in features, and SMOTE applied before train/validation split
- **Class imbalance**: Combined SMOTE + class weighting + threshold tuning rather than relying on any single technique
- **Model selection is metric-dependent**: Random Forest wins on AUC, but the Stacking Ensemble wins on FAIL recall — the metric that matters most for catching defective chips

## 👤 Author

Jatin — Data Science Enthusiast

---
*Built as an end-to-end learning project covering the complete ML lifecycle from data exploration to production deployment.*
