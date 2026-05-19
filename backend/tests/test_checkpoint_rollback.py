import os
import shutil
import pytest
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.models import Base, InverseDesignCheckpoint
from backend.cache.checkpoint_service import CheckpointService

# Create an in-memory SQLite database specifically for test isolation
engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Setup temporary test checkpoint directory
TEST_DIR = "test_checkpoints"

@pytest.fixture(scope="function", autouse=True)
def setup_test_db_and_dir():
    # Setup
    Base.metadata.create_all(bind=engine)
    os.makedirs(TEST_DIR, exist_ok=True)
    yield
    # Teardown
    Base.metadata.drop_all(bind=engine)
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

def test_checkpoint_save_and_load():
    """Verify state data is serialized, compressed, hashed, written, and loaded correctly."""
    db = SessionLocal()
    service = CheckpointService(checkpoint_dir=TEST_DIR)
    
    state_data = {
        "population": [{"ind": [0.1, 0.2, 0.7], "fitness": [1200.0, 45.0]}],
        "start_generation": 5,
        "random_state": (3, (12345, 6789), None)
    }
    
    # Save checkpoint
    cp = service.save_checkpoint(
        db=db,
        job_id="test-job",
        generation=5,
        state_data=state_data
    )
    
    assert cp.generation == 5
    assert cp.job_id == "test-job"
    assert os.path.exists(cp.file_path)
    assert cp.checksum is not None
    
    # Load checkpoint
    loaded = service.load_checkpoint(cp.file_path, cp.checksum)
    assert loaded["start_generation"] == 5
    assert loaded["population"][0]["ind"] == [0.1, 0.2, 0.7]
    assert loaded["population"][0]["fitness"] == [1200.0, 45.0]
    
    db.close()

def test_checkpoint_corruption_checksum_validation():
    """Verify that altering a single byte in the binary file triggers a checksum validation error."""
    db = SessionLocal()
    service = CheckpointService(checkpoint_dir=TEST_DIR)
    
    state = {"generation": 10, "data": "dummy"}
    cp = service.save_checkpoint(db, "job-corrupt", 10, state)
    
    # Corrupt a byte in the saved binary file
    with open(cp.file_path, "r+b") as f:
        f.seek(5)
        # Flip a byte
        byte = f.read(1)
        flipped_byte = bytes([byte[0] ^ 0xFF])
        f.seek(5)
        f.write(flipped_byte)
        
    # Attempting to load must raise a ValueError due to cryptographic checksum mismatch
    with pytest.raises(ValueError, match="Cryptographic corruption!"):
        service.load_checkpoint(cp.file_path, cp.checksum)
        
    db.close()

def test_checkpoint_backwards_chain_rollback():
    """
    Test backwards chain scanning:
    Save gen 5, 10, 15. Corrupt gen 15.
    Assert rollback recovery rolls back to gen 10 and deletes the corrupt gen 15 row.
    """
    db = SessionLocal()
    service = CheckpointService(checkpoint_dir=TEST_DIR)
    
    # Save Gen 5
    cp5 = service.save_checkpoint(db, "job-rollback", 5, {"gen": 5})
    # Save Gen 10
    cp10 = service.save_checkpoint(db, "job-rollback", 10, {"gen": 10}, previous_checksum=cp5.checksum)
    # Save Gen 15
    cp15 = service.save_checkpoint(db, "job-rollback", 15, {"gen": 15}, previous_checksum=cp10.checksum)
    
    # Corrupt the Gen 15 physical file
    with open(cp15.file_path, "wb") as f:
        f.write(b"garbage-corrupted-binary-payload")
        
    # Run rollback recovery
    recovered = service.rollback_recovery(db, "job-rollback")
    
    assert recovered is not None
    recovered_gen, recovered_state, recovered_checksum = recovered
    
    # Must recover Gen 10 (since Gen 15 was corrupted)
    assert recovered_gen == 10
    assert recovered_state["gen"] == 10
    assert recovered_checksum == cp10.checksum
    
    # Assert Gen 15 DB row was deleted
    cp15_row = db.query(InverseDesignCheckpoint).filter(InverseDesignCheckpoint.generation == 15).first()
    assert cp15_row is None
    
    db.close()

def test_checkpoint_cleanup_gc():
    """Verify that cleanup_stale_checkpoints leaves only the last N files."""
    db = SessionLocal()
    service = CheckpointService(checkpoint_dir=TEST_DIR)
    
    # Save 5 checkpoints
    cps = []
    for g in [5, 10, 15, 20, 25]:
        cp = service.save_checkpoint(db, "job-gc", g, {"gen": g})
        cps.append(cp)
        
    # Clean up, keeping only last 2 checkpoints (gens 20, 25)
    service.cleanup_stale_checkpoints(db, "job-gc", keep_last=2)
    
    # Assert DB count
    db_cps = db.query(InverseDesignCheckpoint).filter(InverseDesignCheckpoint.job_id == "job-gc").all()
    assert len(db_cps) == 2
    active_gens = [c.generation for c in db_cps]
    assert 20 in active_gens
    assert 25 in active_gens
    assert 5 not in active_gens
    assert 10 not in active_gens
    
    # Assert older intermediate physical files are deleted, newer ones are preserved
    assert not os.path.exists(cps[0].file_path) # gen 5 deleted
    assert not os.path.exists(cps[1].file_path) # gen 10 deleted
    assert not os.path.exists(cps[2].file_path) # gen 15 deleted
    assert os.path.exists(cps[3].file_path)     # gen 20 preserved
    assert os.path.exists(cps[4].file_path)     # gen 25 preserved
    
    db.close()
