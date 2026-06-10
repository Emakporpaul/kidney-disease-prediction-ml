---
title: CKD Prediction API
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🩺 Chronic Kidney Disease Prediction

[![CI](https://github.com/Emakporpaul/ckd-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Emakporpaul/ckd-prediction/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-Emakporpaul%2Fckd--prediction-yellow)](https://huggingface.co/Emakporpaul/ckd-prediction)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Space-ckd--prediction--api-blue)](https://huggingface.co/spaces/Emakporpaul/ckd-prediction-api)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-quality machine learning system that predicts **Chronic Kidney Disease (CKD)** from 24 routine lab measurements. Built to senior ML engineer standards — clean architecture, full test coverage, and a live REST API.

---

## Live Demo

| Resource | Link |
|----------|------|
| 🔗 API (Swagger UI) | https://huggingface.co/spaces/Emakporpaul/ckd-prediction-api/docs |
| 🤗 Model artefacts | https://huggingface.co/Emakporpaul/ckd-prediction |
| 📓 Notebooks | [`notebooks/`](notebooks/) |

---

## Results

Best model: **Random Forest** — ROC-AUC 1.000 | Accuracy 98.75% | Recall 100% (zero missed CKD patients)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|-------|----------|-----------|--------|----|---------|------------|
| Random Forest | 0.9875 | 0.9804 | 1.00 | 0.9901 | 1.0000 | 0.9990 ± 0.0021 |
| XGBoost | 0.9875 | 1.0000 | 0.98 | 0.9899 | 0.9993 | 0.9902 ± 0.0041 |
| Gradient Boosting | 0.9500 | 0.9792 | 0.94 | 0.9592 | 0.9980 | 0.9977 ± 0.0023 |
| Logistic Regression | 0.9625 | 1.0000 | 0.94 | 0.9691 | 0.9900 | 0.9998 ± 0.0004 |
| SVM | 0.9375 | 1.0000 | 0.90 | 0.9474 | 0.9900 | 0.9998 ± 0.0004 |
| Decision Tree | 0.9500 | 1.0000 | 0.92 | 0.9583 | 0.9827 | 0.9610 ± 0.0280 |
| KNN | 0.9375 | 1.0000 | 0.90 | 0.9474 | 0.9700 | 0.9818 ± 0.0154 |

> **Why not deep learning?** The dataset has 400 samples — too small for neural networks to generalise reliably. Tree-based ensembles are the correct tool at this scale.

---

## Project Structure

```
ckd-prediction/
├── data/
│   └── kidney_disease.csv       # Raw UCI dataset (400 patients, 26 cols)
├── model/                       # Artefacts downloaded from Hugging Face at runtime
│   ├── kidney_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_names.json
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory data analysis
│   ├── 02_Preprocessing.ipynb   # Step-by-step preprocessing with visualisations
│   └── 03_Modelling.ipynb       # Train 7 models, compare, save best
├── src/
│   ├── config.py                # All paths, column lists, hyperparameters
│   ├── preprocess.py            # Cleaning, imputation, encoding, scaling
│   └── evaluate.py              # Metrics, plots, model comparison
├── tests/
│   ├── test_preprocess.py       # 20+ unit tests for preprocessing functions
│   └── test_api.py              # Integration tests for all API endpoints
├── visualizations/              # Auto-generated plots from notebooks
├── app.py                       # FastAPI server
├── schemas.py                   # Pydantic request/response models
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Dev + test dependencies
└── .github/workflows/ci.yml     # GitHub Actions CI pipeline
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Emakporpaul/ckd-prediction.git
cd ckd-prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 2. Run the notebooks (optional — artefacts are on Hugging Face)

```bash
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Preprocessing.ipynb
jupyter notebook notebooks/03_Modelling.ipynb
```

### 3. Start the API

```bash
uvicorn app:app --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Reference

### `POST /predict`

Predict CKD from lab measurements. All 24 fields are optional — missing values are imputed automatically.

**Request body:**

```json
{
  "age": 48.0,
  "blood_pressure": 80.0,
  "specific_gravity": 1.020,
  "albumin": 1.0,
  "sugar": 0.0,
  "blood_glucose_random": 121.0,
  "blood_urea": 36.0,
  "serum_creatinine": 1.2,
  "sodium": 137.0,
  "potassium": 4.4,
  "haemoglobin": 15.4,
  "packed_cell_volume": 44.0,
  "white_blood_cell_count": 7800.0,
  "red_blood_cell_count": 5.2,
  "red_blood_cells": "normal",
  "pus_cell": "normal",
  "pus_cell_clumps": "notpresent",
  "bacteria": "notpresent",
  "hypertension": "no",
  "diabetes_mellitus": "no",
  "coronary_artery_disease": "no",
  "appetite": "good",
  "pedal_edema": "no",
  "anemia": "no"
}
```

**Response:**

```json
{
  "prediction": 0,
  "label": "Not CKD",
  "probability_ckd": 0.04,
  "probability_no_ckd": 0.96,
  "model": "RandomForestClassifier",
  "warning": null
}
```

| Field | Description |
|-------|-------------|
| `prediction` | `1` = CKD, `0` = Not CKD |
| `label` | Human-readable class name |
| `probability_ckd` | Model confidence the patient **has** CKD |
| `probability_no_ckd` | Model confidence the patient does **not** have CKD |
| `model` | Classifier used |
| `warning` | Set when confidence < 70% — suggests clinical review |

### Other endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check + model name |
| `GET` | `/health` | Liveness probe (for Docker / cloud) |
| `GET` | `/model-info` | Model name, metrics, all feature names |

---

## Hugging Face Integration

Model artefacts are stored at [Emakporpaul/ckd-prediction](https://huggingface.co/Emakporpaul/ckd-prediction) and downloaded automatically at API startup via `huggingface_hub`.

This keeps binary files out of Git while making them reproducibly accessible. To download manually:

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id   = "Emakporpaul/ckd-prediction",
    filename  = "kidney_model.pkl",
    local_dir = "model/",
)
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Preprocessing tests only
pytest tests/test_preprocess.py -v

# API tests only
pytest tests/test_api.py -v
```

---

## Dataset

[UCI CKD Dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) — 400 patients, 25 features, binary target.

| Attribute | Value |
|-----------|-------|
| Patients | 400 |
| Features | 24 (after target removal) |
| Target | CKD = 1 / Not CKD = 0 |
| CKD cases | 248 (62%) |
| Not CKD | 150 (38%) |
| Missing values | Yes — handled by random-sample + mode imputation |

**Key features by importance (Random Forest):** haemoglobin, packed cell volume, specific gravity, serum creatinine, albumin, red blood cell count, blood urea, blood glucose, hypertension, diabetes mellitus.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Split before impute | Prevents data leakage from test set statistics |
| One LabelEncoder per column | Shared encoder silently uses wrong class mapping |
| `fit=True/False` pattern | Same preprocessing functions used in training and inference |
| `StandardScaler` | Required by KNN (distance-based) and SVM (RBF kernel) |
| Artefacts on Hugging Face | Keeps binary files out of Git; reproducible across environments |
| Lifespan context manager | Artefacts loaded once at startup, not on every request |

---

## Deployment

The API is deployed as a Hugging Face Space using a `Dockerfile`:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

> **Note:** This project is for research and educational purposes. Do not use model predictions as a substitute for professional medical diagnosis.

---

## Author

**Paul Emakpor**
**AI/ML Engineer**
- GitHub: [@Emakporpaul](https://github.com/Emakporpaul)
- Hugging Face: [@Emakporpaul](https://huggingface.co/Emakporpaul)
- LinkedIn: [paulemakpor](https://www.linkedin.com/in/paulemakpor)

---

## License

MIT License — see [LICENSE](LICENSE) for details.