import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from backend.tasks.inverse_task import _normalize_targets, _normalize_constraints, run_inverse_optimization
from backend.ws.connection_manager import WebSocketConnectionManager

def test_normalize_targets_dict():
    """Verify dictionary-style targets are parsed into ObjectiveTarget instances."""
    targets_data = {
        "yield_strength": [">", 900],
        "corrosion_pren": ["<", 35]
    }
    normalized = _normalize_targets(targets_data)
    assert len(normalized) == 2
    
    ys = [t for t in normalized if t.property_name == "yield_strength"][0]
    assert ys.direction == "maximize"
    assert ys.min_val == 900
    assert ys.max_val is None

    pren = [t for t in normalized if t.property_name == "corrosion_pren"][0]
    assert pren.direction == "minimize"
    assert pren.min_val is None
    assert pren.max_val == 35

def test_normalize_targets_list():
    """Verify list of dicts targets are parsed into ObjectiveTarget instances."""
    targets_data = [
        {"property": "yield_strength", "direction": "maximize", "min_val": 900},
        {"property": "hardness", "direction": "minimize", "max_val": 300}
    ]
    normalized = _normalize_targets(targets_data)
    assert len(normalized) == 2
    
    ys = [t for t in normalized if t.property_name == "yield_strength"][0]
    assert ys.direction == "maximize"
    assert ys.min_val == 900
    assert ys.max_val is None

    hd = [t for t in normalized if t.property_name == "hardness"][0]
    assert hd.direction == "minimize"
    assert hd.min_val is None
    assert hd.max_val == 300

def test_normalize_constraints():
    """Verify element percentage boundaries are mapped to correct fractional bounds."""
    constraints_data = {
        "Cr": [15.0, 25.0],   # percentage format
        "frac_Ni": [0.08, 0.12] # fractional format
    }
    normalized = _normalize_constraints(constraints_data)
    assert "frac_Cr" in normalized
    assert "frac_Ni" in normalized
    
    # 15.0% maps to 0.15 decimal
    assert normalized["frac_Cr"] == (0.15, 0.25)
    assert normalized["frac_Ni"] == (0.08, 0.12)

@patch("backend.tasks.inverse_task.SessionLocal")
@patch("backend.tasks.inverse_task.r")
@patch("backend.tasks.inverse_task.redis_available", True)
def test_celery_task_success(mock_redis, mock_session_class):
    """
    Test successful execution of inverse design task:
    1. DB status transitions to 'running'
    2. Optimizer generates milestones
    3. Redis receives published progress states
    4. DB completes as 'done'
    """
    mock_db = MagicMock()
    mock_session_class.return_value = mock_db
    
    # Mock database job record
    mock_job = MagicMock()
    mock_job.id = "test-job-uuid"
    mock_job.alloy_family = "steel"
    mock_job.targets = {"yield_strength": [">", 900]}
    mock_job.constraints = {"Cr": [12, 20]}
    mock_job.n_generations = 2
    mock_job.pop_size = 8
    mock_job.status = "pending"
    
    mock_db.query().filter().first.return_value = mock_job
    
    # Run the Celery task locally
    res = run_inverse_optimization.apply(args=["test-job-uuid"])
    
    assert res.status == "SUCCESS"
    assert mock_job.status == "done"
    assert mock_job.n_candidates > 0
    assert mock_redis.publish.called
    
    # Verify that the final milestone payload was sent to the correct redis channel
    channel_arg = mock_redis.publish.call_args[0][0]
    payload_arg = json.loads(mock_redis.publish.call_args[0][1])
    
    assert channel_arg == "job:progress:test-job-uuid"
    assert payload_arg["status"] == "complete"
    assert "result" in payload_arg

@pytest.mark.asyncio
async def test_connection_manager_recovery_done():
    """Verify WebSocket manager closes completed jobs cleanly."""
    mock_ws = AsyncMock()
    
    mock_db = MagicMock()
    mock_job = MagicMock()
    mock_job.status = "done"
    mock_job.targets = {"yield_strength": [">", 900]}
    mock_job.n_generations = 50
    mock_job.n_candidates = 5
    mock_job.pareto_front = [{"composition": {"Cr": 0.18}}]
    mock_db.query().filter().first.return_value = mock_job
    
    manager = WebSocketConnectionManager()
    
    # Execute the async WebSocket connection lifecycle
    await manager.handle_connection(mock_ws, "test-done-job", mock_db)
    
    # WebSocket should have accepted, sent final recovery state, and closed instantly
    mock_ws.accept.assert_called_once()
    mock_ws.send_json.assert_called_once()
    
    payload = mock_ws.send_json.call_args[0][0]
    assert payload["status"] == "complete"
    assert payload["n_candidates"] == 5
    
    mock_ws.close.assert_called_once()
