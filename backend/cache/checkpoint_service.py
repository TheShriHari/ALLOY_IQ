import os
import hashlib
import msgpack
import zstandard as zstd
from datetime import datetime
from loguru import logger
from sqlalchemy.orm import Session
from backend.db.models import InverseDesignCheckpoint

class CheckpointService:
    """
    Manages rollback-safe binary checkpoints for GA optimization jobs.
    Checkpoints are saved as zstd-compressed Msgpack binary files in a Docker-volume compatible directory.
    Cryptographic chaining (linking previous generation SHA-256) permits resilient rollback detection.
    """
    def __init__(self, checkpoint_dir: str = None):
        if checkpoint_dir is None:
            checkpoint_dir = os.getenv("CHECKPOINT_DIR", "checkpoints")
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info("Initialized CheckpointService with directory: {}", self.checkpoint_dir)

    def save_checkpoint(
        self,
        db: Session,
        job_id: str,
        generation: int,
        state_data: dict,
        previous_checksum: str = None
    ) -> InverseDesignCheckpoint:
        """
        Serializes and compresses the state data using Msgpack + Zstd,
        writes it to a Docker volume compatible file, and commits metadata to SQL.
        """
        try:
            # 1. Msgpack serialization
            serialized = msgpack.packb(state_data, use_bin_type=True)
            
            # 2. Zstd compression
            cctx = zstd.ZstdCompressor(level=3)
            compressed = cctx.compress(serialized)
            
            # 3. Cryptographic hash calculation (SHA-256)
            checksum = hashlib.sha256(compressed).hexdigest()
            
            # 4. Write to binary file
            filename = f"job_{job_id}_gen_{generation}.bin"
            file_path = os.path.join(self.checkpoint_dir, filename)
            
            with open(file_path, "wb") as f:
                f.write(compressed)
                
            logger.info("Saved binary checkpoint: {}, size={} bytes, checksum={}", file_path, len(compressed), checksum)
            
            # 5. Insert Pointer Row in PostgreSQL/SQLite (Metadata only - zero write amplification)
            checkpoint = InverseDesignCheckpoint(
                job_id=job_id,
                generation=generation,
                file_path=file_path,
                checksum=checksum,
                previous_checksum=previous_checksum
            )
            db.add(checkpoint)
            db.commit()
            db.refresh(checkpoint)
            
            return checkpoint
            
        except Exception as e:
            db.rollback()
            logger.exception("Failed to write checkpoint for job {} gen {}: {}", job_id, generation, e)
            raise e

    def load_checkpoint(self, file_path: str, expected_checksum: str) -> dict:
        """
        Reads a binary checkpoint file, validates its SHA-256 integrity,
        and decompresses/deserializes it.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint file not found: {file_path}")
            
        with open(file_path, "rb") as f:
            compressed_data = f.read()
            
        # Verify SHA-256 Checksum to validate against corruption
        actual_checksum = hashlib.sha256(compressed_data).hexdigest()
        if actual_checksum != expected_checksum:
            raise ValueError(f"Cryptographic corruption! Expected checksum {expected_checksum}, got {actual_checksum}")
            
        # Decompress Zstd
        dctx = zstd.ZstdDecompressor()
        serialized_data = dctx.decompress(compressed_data)
        
        # Deserialization
        state_data = msgpack.unpackb(serialized_data, raw=False)
        return state_data

    def rollback_recovery(self, db: Session, job_id: str) -> tuple[int, dict, str] | None:
        """
        Rollback Selection Scan:
        Scans backwards through the generation chain of checkpoints in DB.
        Verifies checksum integrity for each file. 
        Returns the first valid checkpoint state (generation, state_data, checksum).
        If all are corrupted or missing, returns None.
        """
        logger.info("Initiating rollback-safe recovery scan for job: {}", job_id)
        
        # Query checkpoints in reverse order of generations
        checkpoints = (
            db.query(InverseDesignCheckpoint)
            .filter(InverseDesignCheckpoint.job_id == job_id)
            .order_index(InverseDesignCheckpoint.generation.desc())
            if hasattr(InverseDesignCheckpoint, "order_index")
            else db.query(InverseDesignCheckpoint)
            .filter(InverseDesignCheckpoint.job_id == job_id)
            .order_by(InverseDesignCheckpoint.generation.desc())
        ).all()
        
        for cp in checkpoints:
            try:
                logger.info("Verifying checkpoint: gen={}, path={}", cp.generation, cp.file_path)
                state_data = self.load_checkpoint(cp.file_path, cp.checksum)
                logger.info("Successfully verified and recovered checkpoint at generation {}", cp.generation)
                return cp.generation, state_data, cp.checksum
            except Exception as err:
                logger.warning(
                    "Skipping corrupt/missing checkpoint at generation {}: {}. Rolling back further...",
                    cp.generation,
                    err
                )
                # Cleanup DB row of corrupt checkpoint to keep chain consistent
                try:
                    db.delete(cp)
                    db.commit()
                except Exception:
                    db.rollback()
                    
        logger.error("No valid checkpoint found in chain for job: {}", job_id)
        return None

    def cleanup_stale_checkpoints(self, db: Session, job_id: str, keep_last: int = 3):
        """
        Deletes old intermediate binary files from the Docker volume 
        and prunes DB rows, leaving only the last `keep_last` checkpoints.
        """
        try:
            checkpoints = (
                db.query(InverseDesignCheckpoint)
                .filter(InverseDesignCheckpoint.job_id == job_id)
                .order_by(InverseDesignCheckpoint.generation.desc())
            ).all()
            
            if len(checkpoints) <= keep_last:
                return
                
            stale_cps = checkpoints[keep_last:]
            logger.info("Cleaning up {} stale intermediate checkpoints for job: {}", len(stale_cps), job_id)
            
            for cp in stale_cps:
                # Remove file from disk
                if os.path.exists(cp.file_path):
                    try:
                        os.remove(cp.file_path)
                    except Exception as fe:
                        logger.error("Failed to delete physical checkpoint file {}: {}", cp.file_path, fe)
                # Remove pointer row from DB
                db.delete(cp)
                
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error("Failed to clean up stale checkpoints for job {}: {}", job_id, e)

# Global instance
checkpoint_service = CheckpointService()
