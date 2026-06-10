"""
FastAPI server for the Chronic Kidney Disease Prediction project.

Endpoints:
    GET  /              — health check + model info
    GET  /health        — liveness probe (for Docker/cloud deployments)
    GET  /model-info    — model name, metrics, feature list
    POST /predict       — predict CKD from 24 lab features
"""

import json
import os
import pickle
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import (
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    FEATURE_PATH,
    MODEL_RESULTS_PATH,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    MODEL_DIR,
)
from preprocess import preprocess_single_patient
from schemas import PatientFeatures, PredictionResult


# Hugging Face artefact download

HF_REPO_ID = "Emakporpaul/ckd-prediction"

HF_ARTEFACTS = [
    "kidney_model.pkl",
    "scaler.pkl",
    "label_encoders.pkl",
    "feature_names.json",
]


def _download_artefacts_from_hf() -> None:
    """
    Download model artefacts from Hugging Face Hub into MODEL_DIR.
    Called at startup before any artefact is loaded.
    Raises RuntimeError with a clear message if any download fails.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is not installed. Add it to requirements.txt."
        ) from e

    token = os.getenv("HF_TOKEN")  # optional for public repos
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"    HF repo : {HF_REPO_ID}")
    print(f"    Token   : {'set' if token else 'not set (public repo)'}")
    print(f"    Dest    : {MODEL_DIR}")

    for filename in HF_ARTEFACTS:
        dest = MODEL_DIR / filename
        if dest.exists():
            print(f"    ✓ {filename} (already cached)")
            continue
        print(f"    ⬇  Downloading {filename} ...")
        try:
            hf_hub_download(
                repo_id   = HF_REPO_ID,
                filename  = filename,
                local_dir = str(MODEL_DIR),
                token     = token,
            )
            size_kb = (MODEL_DIR / filename).stat().st_size // 1024
            print(f"    ✓ {filename} ({size_kb} KB)")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download '{filename}' from '{HF_REPO_ID}'.\n"
                f"Error: {e}\n"
                f"Make sure the repo exists and the file has been uploaded."
            ) from e


# Artefact store

class _Store:
    model         = None
    scaler        = None
    encoders      = None
    feature_names = None
    model_name    = "Unknown"
    model_metrics = {}


store = _Store()


# Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄  Loading model artefacts...")

    # Download from HF if not already cached locally
    _download_artefacts_from_hf()

    # Load trained model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}.")
    with open(MODEL_PATH, "rb") as f:
        store.model = pickle.load(f)
    store.model_name = type(store.model).__name__

    # Load scaler
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}.")
    with open(SCALER_PATH, "rb") as f:
        store.scaler = pickle.load(f)

    # Load label encoders
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Encoders not found at {ENCODER_PATH}.")
    with open(ENCODER_PATH, "rb") as f:
        store.encoders = pickle.load(f)

    # Load feature names
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature names not found at {FEATURE_PATH}.")
    with open(FEATURE_PATH) as f:
        store.feature_names = json.load(f)

    # Load model metrics (optional)
    if MODEL_RESULTS_PATH.exists():
        results_df = pd.read_csv(MODEL_RESULTS_PATH)
        CLASS_TO_FRIENDLY = {
            "LogisticRegression":         "Logistic Regression",
            "KNeighborsClassifier":       "KNN",
            "DecisionTreeClassifier":     "Decision Tree",
            "RandomForestClassifier":     "Random Forest",
            "XGBClassifier":              "XGBoost",
            "GradientBoostingClassifier": "Gradient Boosting",
            "SVC":                        "SVM",
        }
        friendly_name = CLASS_TO_FRIENDLY.get(store.model_name, store.model_name)
        match = results_df[results_df["model"] == friendly_name]
        if not match.empty:
            row = match.iloc[0]
            store.model_metrics = {
                "accuracy":  round(float(row.get("accuracy",  0)), 4),
                "precision": round(float(row.get("precision", 0)), 4),
                "recall":    round(float(row.get("recall",    0)), 4),
                "f1":        round(float(row.get("f1",        0)), 4),
                "roc_auc":   round(float(row.get("roc_auc",   0)), 4),
            }

    print(f"✅  Model    : {store.model_name}")
    print(f"    Features : {len(store.feature_names)}")
    print(f"    Encoders : {len(store.encoders)} columns")
    print(f"    Metrics  : {store.model_metrics}")
    print("🚀  API ready\n")

    yield

    print("👋  Shutting down")


# App instance

app = FastAPI(
    title       = "CKD Prediction API",
    description = (
        "Predict Chronic Kidney Disease from 24 lab measurements.\n\n"
        "All fields are optional — missing values are imputed automatically "
        "using the same strategy applied during model training.\n\n"
        "**Label encoding:** CKD = 1 | Not CKD = 0"
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# Helpers

def _impute_missing_inference(patient_dict: dict) -> dict:
    numerical_medians = {
        "age": 55.0, "blood_pressure": 80.0, "specific_gravity": 1.020,
        "albumin": 0.0, "sugar": 0.0, "blood_glucose_random": 121.0,
        "blood_urea": 42.0, "serum_creatinine": 1.3, "sodium": 138.0,
        "potassium": 4.4, "haemoglobin": 12.65, "packed_cell_volume": 40.0,
        "white_blood_cell_count": 8000.0, "red_blood_cell_count": 4.8,
    }
    categorical_modes = {
        "red_blood_cells": "normal", "pus_cell": "normal",
        "pus_cell_clumps": "notpresent", "bacteria": "notpresent",
        "hypertension": "no", "diabetes_mellitus": "no",
        "coronary_artery_disease": "no", "appetite": "good",
        "pedal_edema": "no", "anemia": "no",
    }
    imputed = dict(patient_dict)
    for col, val in numerical_medians.items():
        if imputed.get(col) is None:
            imputed[col] = val
    for col, val in categorical_modes.items():
        if imputed.get(col) is None:
            imputed[col] = val
    return imputed


def _patient_to_array(patient: PatientFeatures) -> np.ndarray:
    raw_dict = {
        k: (v.value if hasattr(v, "value") else v)
        for k, v in patient.model_dump().items()
    }
    raw_dict = _impute_missing_inference(raw_dict)
    return preprocess_single_patient(
        patient_data  = raw_dict,
        encoders      = store.encoders,
        scaler        = store.scaler,
        feature_names = store.feature_names,
    )


# Routes

@app.get("/", tags=["Status"])
def root():
    return {
        "service": "CKD Prediction API",
        "version": "1.0.0",
        "status":  "running",
        "model":   store.model_name,
        "docs":    "/docs",
    }


@app.get("/health", tags=["Status"])
def health():
    if store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.get("/model-info", tags=["Model"])
def model_info():
    return {
        "model_name":       store.model_name,
        "metrics":          store.model_metrics,
        "n_features":       len(store.feature_names),
        "feature_names":    store.feature_names,
        "numerical_cols":   NUMERICAL_COLS,
        "categorical_cols": CATEGORICAL_COLS,
    }


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(patient: PatientFeatures):
    """
    Predict whether a patient has Chronic Kidney Disease.

    **Input:** 24 lab features (all optional — missing values are imputed).
    **Output:** Predicted label, probabilities, warning if confidence < 70%.

    **Label encoding:** 1 = CKD | 0 = Not CKD

    *This API is for research purposes only. Always consult a physician.*
    """
    if store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        X           = _patient_to_array(patient)
        pred_label  = int(store.model.predict(X)[0])
        pred_proba  = store.model.predict_proba(X)[0]
        prob_ckd    = round(float(pred_proba[1]), 4)
        prob_no_ckd = round(float(pred_proba[0]), 4)
        label       = "CKD" if pred_label == 1 else "Not CKD"
        confidence  = max(prob_ckd, prob_no_ckd)
        warning     = None
        if confidence < 0.70:
            warning = (
                f"Model confidence is {confidence:.0%}. "
                "Borderline case — please consult a physician."
            )
        return PredictionResult(
            prediction         = pred_label,
            label              = label,
            probability_ckd    = prob_ckd,
            probability_no_ckd = prob_no_ckd,
            model              = store.model_name,
            warning            = warning,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)