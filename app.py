"""
FastAPI server for the Chronic Kidney Disease Prediction project.

Endpoints:
    GET  /              — health check + model info
    GET  /health        — liveness probe (for Docker/cloud deployments)
    GET  /model-info    — model name, metrics, feature list
    POST /predict       — predict CKD from 24 lab features

Design decisions:
  - Artefacts (model, scaler, encoders, feature_names) are loaded ONCE at
    startup via a FastAPI lifespan context manager — not on every request.
    Loading a pkl on every call would add ~100ms latency unnecessarily.

  - All preprocessing reuses the exact same functions from preprocess.py
    that were used during training. No copy-pasted logic in this file.

  - Missing values in the request are handled by imputation before the
    model sees the data, mirroring training behaviour.

  - A warning is added to the response when model confidence < 70%,
    prompting clinical review for borderline cases.

Usage:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Then open:
    http://localhost:8000/docs    ← interactive Swagger UI
    http://localhost:8000/redoc  ← ReDoc documentation

"""

import json
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

# Project imports
# sys.path manipulation is NOT needed here because app.py lives in the
# project root (same level as src/). We add src/ explicitly so config and
# preprocess can be imported cleanly.
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
)
from preprocess import preprocess_single_patient
from schemas import PatientFeatures, PredictionResult


# ARTEFACT STORE
# A simple namespace that holds loaded artefacts so they are available to
# all route handlers without being passed around as arguments.

class _Store:
    model         = None
    scaler        = None
    encoders      = None
    feature_names = None
    model_name    = "Unknown"
    model_metrics = {}


store = _Store()


# LIFESPAN — runs once at startup and once at shutdown

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all artefacts into memory when the server starts.
    This runs exactly once — not on every request.
    """
    print("🔄  Loading model artefacts...")

    # Load trained model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run 03_Modelling.ipynb first."
        )
    with open(MODEL_PATH, "rb") as f:
        store.model = pickle.load(f)

    store.model_name = type(store.model).__name__

    # Load scaler
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. "
            "Run 02_Preprocessing.ipynb first."
        )
    with open(SCALER_PATH, "rb") as f:
        store.scaler = pickle.load(f)

    # Load label encoders
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Encoders not found at {ENCODER_PATH}. "
            "Run 02_Preprocessing.ipynb first."
        )
    with open(ENCODER_PATH, "rb") as f:
        store.encoders = pickle.load(f)

    # Load feature names
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature names not found at {FEATURE_PATH}. "
            "Run 02_Preprocessing.ipynb first."
        )
    with open(FEATURE_PATH) as f:
        store.feature_names = json.load(f)

    # Load model metrics (from model_results.csv if available)
    if MODEL_RESULTS_PATH.exists():
        results_df = pd.read_csv(MODEL_RESULTS_PATH)

        # model_results.csv uses friendly names ("Random Forest") but
        # type(model).__name__ returns the sklearn/xgboost class name.
        # This explicit map handles every case without fragile string heuristics.
        CLASS_TO_FRIENDLY = {
            "LogisticRegression":           "Logistic Regression",
            "KNeighborsClassifier":         "KNN",
            "DecisionTreeClassifier":       "Decision Tree",
            "RandomForestClassifier":       "Random Forest",
            "XGBClassifier":                "XGBoost",
            "GradientBoostingClassifier":   "Gradient Boosting",
            "SVC":                          "SVM",
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

    print(f"✅  Model loaded     : {store.model_name}")
    print(f"    Features         : {len(store.feature_names)}")
    print(f"    Encoders         : {len(store.encoders)} columns")
    print(f"    Metrics          : {store.model_metrics}")
    print("🚀  API ready\n")

    yield   # ← server runs here

    # Shutdown (nothing to clean up for a simple ML server)
    print("👋  Shutting down")


# APP INSTANCE

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

# Allow all origins for local development.
# In production, replace ["*"] with your frontend's exact origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# HELPERS

def _impute_missing_inference(patient_dict: dict) -> dict:
    """
    Fill None values in the patient dict before encoding/scaling.

    Strategy mirrors training imputation:
      - Numerical → median of training distribution (a safe point estimate)
      - Categorical → most common value in training data

    Using medians/modes here (rather than reloading the training set) is
    appropriate for a single-sample API — we don't have a distribution to
    sample from. The values below come from the training set statistics
    computed in 02_Preprocessing.ipynb.
    """
    # Numerical medians from training data (from describe() output)
    numerical_medians = {
        "age":                    55.0,
        "blood_pressure":         80.0,
        "specific_gravity":       1.020,
        "albumin":                0.0,
        "sugar":                  0.0,
        "blood_glucose_random":   121.0,
        "blood_urea":             42.0,
        "serum_creatinine":       1.3,
        "sodium":                 138.0,
        "potassium":              4.4,
        "haemoglobin":            12.65,
        "packed_cell_volume":     40.0,
        "white_blood_cell_count": 8000.0,
        "red_blood_cell_count":   4.8,
    }

    # Categorical modes from training data
    categorical_modes = {
        "red_blood_cells":        "normal",
        "pus_cell":               "normal",
        "pus_cell_clumps":        "notpresent",
        "bacteria":               "notpresent",
        "hypertension":           "no",
        "diabetes_mellitus":      "no",
        "coronary_artery_disease":"no",
        "appetite":               "good",
        "pedal_edema":            "no",
        "anemia":                 "no",
    }

    imputed = dict(patient_dict)  # copy — don't mutate the original

    for col, median in numerical_medians.items():
        if imputed.get(col) is None:
            imputed[col] = median

    for col, mode in categorical_modes.items():
        if imputed.get(col) is None:
            imputed[col] = mode

    return imputed


def _patient_to_array(patient: PatientFeatures) -> np.ndarray:
    """
    Convert a PatientFeatures Pydantic model to a scaled numpy array
    ready for model.predict().

    Steps:
      1. Convert to dict (Enum values resolved to their string)
      2. Impute any missing (None) fields
      3. Encode categoricals using the fitted LabelEncoders
      4. Scale using the fitted StandardScaler
    """
    # Convert to plain dict; resolve Enum → string
    raw_dict = {
        k: (v.value if hasattr(v, "value") else v)
        for k, v in patient.model_dump().items()
    }

    # Impute None values
    raw_dict = _impute_missing_inference(raw_dict)

    # Delegate encoding + scaling to preprocess.py
    # (same function path used during training)
    array = preprocess_single_patient(
        patient_data  = raw_dict,
        encoders      = store.encoders,
        scaler        = store.scaler,
        feature_names = store.feature_names,
    )

    return array   # shape (1, n_features)


# ROUTES

@app.get("/", tags=["Status"])
def root():
    """
    Root endpoint — quick health check confirming the API is alive.
    """
    return {
        "service":     "CKD Prediction API",
        "version":     "1.0.0",
        "status":      "running",
        "model":       store.model_name,
        "docs":        "/docs",
    }


@app.get("/health", tags=["Status"])
def health():
    """
    Liveness probe — returns 200 OK when the model is loaded.
    Used by Docker health checks and cloud load balancers.
    """
    if store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.get("/model-info", tags=["Model"])
def model_info():
    """
    Return metadata about the loaded model: name, metrics, feature list.
    Useful for dashboards and automated monitoring.
    """
    return {
        "model_name":    store.model_name,
        "metrics":       store.model_metrics,
        "n_features":    len(store.feature_names),
        "feature_names": store.feature_names,
        "numerical_cols":   NUMERICAL_COLS,
        "categorical_cols": CATEGORICAL_COLS,
    }


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(patient: PatientFeatures):
    """
    Predict whether a patient has Chronic Kidney Disease.

    **Input:** 24 lab features (all optional — missing values are imputed).  
    **Output:** Predicted label, probabilities, and a warning if confidence < 70%.

    ---
    **Label encoding used by the model:**
    - `1` → CKD (disease present)
    - `0` → Not CKD (disease absent)

    **Note:** This API is for research purposes only.  
    Always consult a qualified physician for medical decisions.
    """
    if store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Preprocess the incoming patient data
        X = _patient_to_array(patient)

        # Run prediction
        pred_label = int(store.model.predict(X)[0])
        pred_proba = store.model.predict_proba(X)[0]   # [P(Not CKD), P(CKD)]

        prob_ckd    = round(float(pred_proba[1]), 4)
        prob_no_ckd = round(float(pred_proba[0]), 4)

        label = "CKD" if pred_label == 1 else "Not CKD"

        # Add a warning for low-confidence predictions
        confidence  = max(prob_ckd, prob_no_ckd)
        warning = None
        if confidence < 0.70:
            warning = (
                f"Model confidence is {confidence:.0%}. "
                "This is a borderline case — please consult a physician."
            )

        return PredictionResult(
            prediction      = pred_label,
            label           = label,
            probability_ckd = prob_ckd,
            probability_no_ckd = prob_no_ckd,
            model           = store.model_name,
            warning         = warning,
        )

    except Exception as exc:
        # Surface the underlying error with context
        raise HTTPException(
            status_code = 500,
            detail      = f"Prediction failed: {str(exc)}",
        ) from exc


# ENTRY POINT — for running directly with `python app.py`

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,   # auto-restart on code changes (dev only)
    )