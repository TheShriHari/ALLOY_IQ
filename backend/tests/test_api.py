import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.db.models import Base
from backend.main import engine

# Setup test DB
Base.metadata.create_all(bind=engine)

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "alloy-iq-api"}

def test_predict_mechanical_steel_yield_strength():
    payload = {
        "alloy_family": "steel",
        "property": "yield_strength_mpa",
        "composition": {
            "Fe": 0.69,
            "Cr": 0.225,
            "Ni": 0.05,
            "Mo": 0.03,
            "Mn": 0.005
        },
        "confidence": 0.90
    }
    response = client.post("/api/v1/predict/mechanical", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "predictions" in data
    assert "yield_strength_mpa" in data["predictions"]
    assert "mean" in data["predictions"]["yield_strength_mpa"]
    assert "lower" in data["predictions"]["yield_strength_mpa"]
    assert "upper" in data["predictions"]["yield_strength_mpa"]
    assert "job_id" in data
    assert "corrosion_analysis" in data

def test_predict_explain_steel_yield_strength():
    payload = {
        "alloy_family": "steel",
        "property": "yield_strength_mpa",
        "composition": {
            "Fe": 0.69,
            "Cr": 0.225,
            "Ni": 0.05,
            "Mo": 0.03,
            "Mn": 0.005
        }
    }
    response = client.post("/api/v1/predict/explain", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "shap_values" in data
    assert "narrative" in data
