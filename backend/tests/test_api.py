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
        "property": "yield_strength",
        "composition": {
            "Fe": 0.69,
            "Cr": 0.225,
            "Ni": 0.05,
            "Mo": 0.03,
            "Mn": 0.005
        },
        "confidence": 0.90
    }
    response = client.post("/predict/mechanical", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "prediction" in data
    assert "lower" in data
    assert "upper" in data
    assert "job_id" in data
    assert data["unit"] == "MPa"

def test_predict_explain_steel_yield_strength():
    payload = {
        "alloy_family": "steel",
        "property": "yield_strength",
        "composition": {
            "Fe": 0.69,
            "Cr": 0.225,
            "Ni": 0.05,
            "Mo": 0.03,
            "Mn": 0.005
        }
    }
    response = client.post("/predict/explain", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "shap" in data
    shap_data = data["shap"]
    assert "waterfall" in shap_data
    assert "narrative" in shap_data
    assert "base_value" in shap_data
