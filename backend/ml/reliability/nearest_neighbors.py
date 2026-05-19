import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Any
from loguru import logger

class AlloyEvidenceFinder:
    """
    Empirical nearest-neighbor evidence collector.
    Finds matching historical alloy formulations in the training corpus
    to serve as decision-support proofs with bibliographic citations.
    """
    def __init__(self):
        self.nn_model = None
        self.metadata = []
        self.X_train = None

    def fit(self, X: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Fits spatial nearest neighbors indexing over composition+processing spaces
        and links each sample to its database metadata record.
        """
        logger.info("Initializing AlloyEvidenceFinder index with {} database records.", len(metadata))
        self.X_train = X
        self.metadata = metadata
        
        # Fit KDTree or BallTree search index
        self.nn_model = NearestNeighbors(n_neighbors=min(5, len(X)), metric="euclidean")
        self.nn_model.fit(X)
        return self

    def find_evidence(self, x: np.ndarray) -> List[Dict[str, Any]]:
        """
        Queries top 5 closest empirical matches and formats evidence footprints.
        Returns elements, DOI, processing route, and distance metrics.
        """
        if self.nn_model is None:
            raise ValueError("AlloyEvidenceFinder index has not been fitted!")
            
        distances, indices = self.nn_model.kneighbors(x.reshape(1, -1))
        
        evidence_list = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            record_meta = self.metadata[idx]
            
            # Format custom summary route
            ht = record_meta.get("heat_treatment_category") or "unknown"
            cool = record_meta.get("cooling_method") or "unknown"
            route = record_meta.get("manufacturing_route") or "unknown"
            temp = record_meta.get("annealing_temperature")
            temp_str = f" @ {temp}C" if temp and temp > 0 else ""
            
            proc_route_str = f"{route.upper()} -> {ht.upper()}{temp_str} -> {cool.upper()}"
            
            evidence_list.append({
                "rank": rank + 1,
                "composition": record_meta.get("composition") or {},
                "paper_doi": record_meta.get("paper_doi") or "unknown",
                "processing_route": proc_route_str,
                "distance": float(dist)
            })
            
        return evidence_list
