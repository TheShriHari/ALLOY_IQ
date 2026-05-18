"""
ALLOY IQ - Sync Script
Generates openapi.json from the FastAPI app and logs metrics to agent_tracker.json.
"""

import json
from pathlib import Path
from backend.main import app
from fastapi.openapi.utils import get_openapi

def generate_openapi():
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )
    
    output_path = Path(__file__).parent.parent / "openapi.json"
    with open(output_path, "w") as f:
        json.dump(openapi_schema, f, indent=2)
    print(f"Generated openapi.json at {output_path}")

def update_agent_tracker(metrics: dict = None):
    tracker_path = Path(__file__).parent.parent / "agent_tracker.json"
    
    if tracker_path.exists():
        with open(tracker_path, "r") as f:
            try:
                tracker_data = json.load(f)
            except json.JSONDecodeError:
                tracker_data = {}
    else:
        tracker_data = {}

    tracker_data["api_endpoints"] = [
        "/predict/mechanical",
        "/predict/explain",
        "/inverse",
        "/history"
    ]
    
    if metrics:
        tracker_data.setdefault("ml_metrics", {}).update(metrics)
        
    tracker_data["artifact_paths"] = {
        "models": "backend/models/",
        "openapi": "openapi.json"
    }
    
    with open(tracker_path, "w") as f:
        json.dump(tracker_data, f, indent=2)
    print(f"Updated agent_tracker.json at {tracker_path}")

if __name__ == "__main__":
    generate_openapi()
    
    # Mock metrics, in a real scenario these would be passed after training
    mock_metrics = {
        "steel__yield_strength": {"r2": 0.94, "mae": 15.2, "rmse": 20.1},
        "hea__yield_strength": {"r2": 0.88, "mae": 45.3, "rmse": 55.4}
    }
    update_agent_tracker(mock_metrics)
