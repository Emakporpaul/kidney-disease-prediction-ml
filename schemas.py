"""
Pydantic models for the CKD Prediction REST API.

Responsibilities:
  - Validate every incoming request field (type, range, allowed values)
  - Document the API automatically via FastAPI's OpenAPI schema
  - Provide a clear, user-friendly example payload in the docs UI

Two schemas:
  PatientFeatures  — request body (25 lab measurements)
  PredictionResult — response body (prediction + probabilities + metadata)

Why Pydantic?
  FastAPI runs these validators before your route handler is called.
  Bad input is rejected with a 422 response and a clear error message
  before it ever reaches the model — no silent wrong predictions.

"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# CATEGORICAL VALUE ENUMS
# Restrict string fields to exactly the values seen during training.
# Any other string would cause the LabelEncoder to raise an error at runtime.

class NormalAbnormal(str, Enum):
    normal   = "normal"
    abnormal = "abnormal"

class PresentNotPresent(str, Enum):
    present    = "present"
    notpresent = "notpresent"

class YesNo(str, Enum):
    yes = "yes"
    no  = "no"

class Appetite(str, Enum):
    good = "good"
    poor = "poor"


# REQUEST SCHEMA — PatientFeatures

class PatientFeatures(BaseModel):
    """
    All 24 lab features required to predict CKD for one patient.

    Numerical fields are validated against realistic clinical ranges.
    Categorical fields are restricted to the exact labels seen during training.
    All fields are optional (None) to support partial records — the API
    handles imputation just as the training pipeline did.
    """

    # Numerical features
    age: Optional[float] = Field(
        default=None,
        ge=0, le=120,
        description="Patient age in years",
        example=48.0,
    )
    blood_pressure: Optional[float] = Field(
        default=None,
        ge=0, le=300,
        description="Diastolic blood pressure (mm/Hg)",
        example=80.0,
    )
    specific_gravity: Optional[float] = Field(
        default=None,
        ge=1.000, le=1.030,
        description="Urine specific gravity",
        example=1.020,
    )
    albumin: Optional[float] = Field(
        default=None,
        ge=0, le=5,
        description="Albumin level (0–5 scale)",
        example=1.0,
    )
    sugar: Optional[float] = Field(
        default=None,
        ge=0, le=5,
        description="Sugar level (0–5 scale)",
        example=0.0,
    )
    blood_glucose_random: Optional[float] = Field(
        default=None,
        ge=0, le=600,
        description="Random blood glucose (mgs/dl)",
        example=121.0,
    )
    blood_urea: Optional[float] = Field(
        default=None,
        ge=0, le=500,
        description="Blood urea (mgs/dl)",
        example=36.0,
    )
    serum_creatinine: Optional[float] = Field(
        default=None,
        ge=0, le=100,
        description="Serum creatinine (mgs/dl)",
        example=1.2,
    )
    sodium: Optional[float] = Field(
        default=None,
        ge=0, le=200,
        description="Sodium level (mEq/L)",
        example=137.0,
    )
    potassium: Optional[float] = Field(
        default=None,
        ge=0, le=60,
        description="Potassium level (mEq/L)",
        example=4.4,
    )
    haemoglobin: Optional[float] = Field(
        default=None,
        ge=0, le=25,
        description="Haemoglobin (gms)",
        example=15.4,
    )
    packed_cell_volume: Optional[float] = Field(
        default=None,
        ge=0, le=60,
        description="Packed cell volume (%)",
        example=44.0,
    )
    white_blood_cell_count: Optional[float] = Field(
        default=None,
        ge=0, le=50000,
        description="White blood cell count (cells/cumm)",
        example=7800.0,
    )
    red_blood_cell_count: Optional[float] = Field(
        default=None,
        ge=0, le=15,
        description="Red blood cell count (millions/cmm)",
        example=5.2,
    )

    # Categorical features
    red_blood_cells: Optional[NormalAbnormal] = Field(
        default=None,
        description="Red blood cells in urine: 'normal' or 'abnormal'",
        example="normal",
    )
    pus_cell: Optional[NormalAbnormal] = Field(
        default=None,
        description="Pus cell in urine: 'normal' or 'abnormal'",
        example="normal",
    )
    pus_cell_clumps: Optional[PresentNotPresent] = Field(
        default=None,
        description="Pus cell clumps: 'present' or 'notpresent'",
        example="notpresent",
    )
    bacteria: Optional[PresentNotPresent] = Field(
        default=None,
        description="Bacteria in urine: 'present' or 'notpresent'",
        example="notpresent",
    )
    hypertension: Optional[YesNo] = Field(
        default=None,
        description="Hypertension: 'yes' or 'no'",
        example="no",
    )
    diabetes_mellitus: Optional[YesNo] = Field(
        default=None,
        description="Diabetes mellitus: 'yes' or 'no'",
        example="no",
    )
    coronary_artery_disease: Optional[YesNo] = Field(
        default=None,
        description="Coronary artery disease: 'yes' or 'no'",
        example="no",
    )
    appetite: Optional[Appetite] = Field(
        default=None,
        description="Appetite: 'good' or 'poor'",
        example="good",
    )
    pedal_edema: Optional[YesNo] = Field(
        default=None,
        description="Pedal edema: 'yes' or 'no'",
        example="no",
    )
    anemia: Optional[YesNo] = Field(
        default=None,
        description="Anemia: 'yes' or 'no'",
        example="no",
    )

    class Config:
        # Show a full realistic example in the /docs UI
        json_schema_extra = {
            "example": {
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
                "anemia": "no",
            }
        }


# RESPONSE SCHEMA — PredictionResult

class PredictionResult(BaseModel):
    """
    Prediction result returned by the /predict endpoint.

    Fields:
        prediction       — integer class label (1 = CKD, 0 = Not CKD)
        label            — human-readable class name
        probability_ckd  — model confidence that patient HAS CKD (0–1)
        probability_no_ckd — model confidence that patient does NOT have CKD
        model            — name of the model that produced this prediction
        warning          — optional clinical note when confidence is borderline
    """

    prediction: int = Field(
        description="Predicted class: 1 = CKD, 0 = Not CKD",
        example=0,
    )
    label: str = Field(
        description="Human-readable prediction label",
        example="Not CKD",
    )
    probability_ckd: float = Field(
        description="Probability the patient has CKD (0.0 – 1.0)",
        example=0.12,
    )
    probability_no_ckd: float = Field(
        description="Probability the patient does NOT have CKD (0.0 – 1.0)",
        example=0.88,
    )
    model: str = Field(
        description="Name of the model used for this prediction",
        example="Random Forest",
    )
    warning: Optional[str] = Field(
        default=None,
        description="Flagged when model confidence is below 70% — consult a physician",
        example=None,
    )
