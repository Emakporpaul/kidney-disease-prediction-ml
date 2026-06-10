"""
Integration tests for the FastAPI CKD Prediction API.

Run with:
    pytest tests/test_api.py -v

Uses FastAPI's TestClient (runs the app in-process, no server needed).
Artefacts must exist in model/ before running — execute the notebooks first.

Coverage:
  GET  /              — status code, key fields
  GET  /health        — liveness
  GET  /model-info    — fields, feature count
  POST /predict       — full payload, partial payload, edge cases
  POST /predict       — invalid values (expects 422)

"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Allow importing app and src/ 
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app import app


# CLIENT FIXTURE

@pytest.fixture(scope="module")
def client():
    """
    TestClient wraps the app and runs lifespan (startup/shutdown) automatically.
    scope="module" means artefacts are loaded once for the entire test file.
    """
    with TestClient(app) as c:
        yield c


# PAYLOADS

FULL_HEALTHY_PATIENT = {
    "age": 35.0,
    "blood_pressure": 70.0,
    "specific_gravity": 1.020,
    "albumin": 0.0,
    "sugar": 0.0,
    "blood_glucose_random": 100.0,
    "blood_urea": 25.0,
    "serum_creatinine": 0.9,
    "sodium": 140.0,
    "potassium": 4.2,
    "haemoglobin": 15.0,
    "packed_cell_volume": 45.0,
    "white_blood_cell_count": 8000.0,
    "red_blood_cell_count": 5.0,
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

FULL_CKD_PATIENT = {
    "age": 70.0,
    "blood_pressure": 100.0,
    "specific_gravity": 1.005,
    "albumin": 4.0,
    "sugar": 3.0,
    "blood_glucose_random": 380.0,
    "blood_urea": 180.0,
    "serum_creatinine": 9.4,
    "sodium": 110.0,
    "potassium": 6.1,
    "haemoglobin": 6.0,
    "packed_cell_volume": 20.0,
    "white_blood_cell_count": 14000.0,
    "red_blood_cell_count": 2.5,
    "red_blood_cells": "abnormal",
    "pus_cell": "abnormal",
    "pus_cell_clumps": "present",
    "bacteria": "present",
    "hypertension": "yes",
    "diabetes_mellitus": "yes",
    "coronary_artery_disease": "yes",
    "appetite": "poor",
    "pedal_edema": "yes",
    "anemia": "yes",
}


# STATUS ENDPOINTS

class TestRootEndpoint:

    def test_status_code(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_response_has_required_keys(self, client):
        data = client.get("/").json()
        for key in ("service", "version", "status", "model", "docs"):
            assert key in data, f"Missing key in root response: '{key}'"

    def test_status_is_running(self, client):
        data = client.get("/").json()
        assert data["status"] == "running"

    def test_docs_link(self, client):
        data = client.get("/").json()
        assert data["docs"] == "/docs"


class TestHealthEndpoint:

    def test_status_code(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"


# MODEL INFO

class TestModelInfoEndpoint:

    def test_status_code(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_has_required_fields(self, client):
        data = client.get("/model-info").json()
        for key in ("model_name", "metrics", "n_features", "feature_names",
                    "numerical_cols", "categorical_cols"):
            assert key in data, f"Missing key: '{key}'"

    def test_feature_count(self, client):
        data = client.get("/model-info").json()
        assert data["n_features"] == 24

    def test_feature_names_is_list_of_strings(self, client):
        data = client.get("/model-info").json()
        assert isinstance(data["feature_names"], list)
        assert all(isinstance(f, str) for f in data["feature_names"])

    def test_model_name_is_nonempty_string(self, client):
        data = client.get("/model-info").json()
        assert isinstance(data["model_name"], str)
        assert len(data["model_name"]) > 0


# PREDICT — valid inputs

class TestPredictEndpoint:

    def test_status_code_full_payload(self, client):
        response = client.post("/predict", json=FULL_HEALTHY_PATIENT)
        assert response.status_code == 200

    def test_response_schema_fields(self, client):
        data = client.post("/predict", json=FULL_HEALTHY_PATIENT).json()
        for key in ("prediction", "label", "probability_ckd",
                    "probability_no_ckd", "model"):
            assert key in data, f"Missing key in predict response: '{key}'"

    def test_prediction_is_binary(self, client):
        data = client.post("/predict", json=FULL_HEALTHY_PATIENT).json()
        assert data["prediction"] in (0, 1)

    def test_label_matches_prediction(self, client):
        data = client.post("/predict", json=FULL_HEALTHY_PATIENT).json()
        if data["prediction"] == 1:
            assert data["label"] == "CKD"
        else:
            assert data["label"] == "Not CKD"

    def test_probabilities_sum_to_one(self, client):
        data = client.post("/predict", json=FULL_HEALTHY_PATIENT).json()
        total = data["probability_ckd"] + data["probability_no_ckd"]
        assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}, expected 1.0"

    def test_probabilities_in_range(self, client):
        data = client.post("/predict", json=FULL_HEALTHY_PATIENT).json()
        assert 0.0 <= data["probability_ckd"] <= 1.0
        assert 0.0 <= data["probability_no_ckd"] <= 1.0

    def test_healthy_patient_predicted_not_ckd(self, client):
        """A clearly healthy patient should be predicted as Not CKD."""
        data = client.post("/predict", json=FULL_HEALTHY_PATIENT).json()
        assert data["prediction"] == 0, (
            f"Healthy patient predicted as CKD with P(CKD)={data['probability_ckd']}"
        )

    def test_ckd_patient_predicted_ckd(self, client):
        """A clearly CKD patient should be predicted as CKD."""
        data = client.post("/predict", json=FULL_CKD_PATIENT).json()
        assert data["prediction"] == 1, (
            f"CKD patient predicted as Not CKD with P(CKD)={data['probability_ckd']}"
        )

    def test_partial_payload_accepted(self, client):
        """All fields are optional — a minimal payload must not cause a 422."""
        minimal = {"age": 55.0, "haemoglobin": 10.0}
        response = client.post("/predict", json=minimal)
        assert response.status_code == 200

    def test_empty_payload_accepted(self, client):
        """Empty body — all fields imputed — must return a valid prediction."""
        response = client.post("/predict", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in (0, 1)

    def test_warning_field_present_in_response(self, client):
        """warning key must exist (may be None for high-confidence predictions)."""
        data = client.post("/predict", json=FULL_HEALTHY_PATIENT).json()
        assert "warning" in data   # key exists; value may be None

    def test_model_field_nonempty_string(self, client):
        data = client.post("/predict", json=FULL_HEALTHY_PATIENT).json()
        assert isinstance(data["model"], str)
        assert len(data["model"]) > 0


# PREDICT — invalid inputs (expect 422 Unprocessable Entity)

class TestPredictValidation:

    def test_invalid_categorical_value_rejected(self, client):
        """'maybe' is not a valid value for hypertension."""
        bad_payload = {**FULL_HEALTHY_PATIENT, "hypertension": "maybe"}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_age_below_minimum_rejected(self, client):
        bad_payload = {**FULL_HEALTHY_PATIENT, "age": -5.0}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_age_above_maximum_rejected(self, client):
        bad_payload = {**FULL_HEALTHY_PATIENT, "age": 200.0}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_specific_gravity_out_of_range_rejected(self, client):
        bad_payload = {**FULL_HEALTHY_PATIENT, "specific_gravity": 2.0}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_wrong_type_for_numeric_field_rejected(self, client):
        """Passing a string where a float is expected should be rejected."""
        bad_payload = {**FULL_HEALTHY_PATIENT, "age": "old"}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_invalid_appetite_value_rejected(self, client):
        bad_payload = {**FULL_HEALTHY_PATIENT, "appetite": "excellent"}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422