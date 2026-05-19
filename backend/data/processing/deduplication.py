import hashlib
import json
from typing import List, Dict, Any
from loguru import logger

class AlloyDeduplicator:
    """
    Performs multi-criteria deduplication.
    Metallurgical properties depend heavily on the processing history. Therefore, 
    duplicate detection uses: Composition + Processing Parameters + Target Properties + DOI.
    Composition-only duplicates are preserved to support different heat-treatment study comparison.
    """
    
    @staticmethod
    def generate_fingerprint(record: Dict[str, Any]) -> str:
        """
        Generates a deterministic cryptographic hash fingerprint of an alloy entry.
        Rounds floats and standardizes keys to handle noise.
        """
        # 1. Composition Fingerprint (sorted, rounded)
        raw_comp = record.get("composition") or {}
        normalized_comp = []
        for el, frac in sorted(raw_comp.items()):
            try:
                rounded_frac = round(float(frac), 5)
                if rounded_frac > 0:
                    normalized_comp.append((el, rounded_frac))
            except (ValueError, TypeError):
                continue
                
        # 2. Processing Fingerprint (normalized fields)
        heat_treat = str(record.get("heat_treatment_category") or "unknown").strip().lower()
        
        try:
            temp_val = record.get("annealing_temperature")
            temp = float(temp_val) if temp_val is not None else 0.0
            temp = round(temp, 1)
        except (ValueError, TypeError):
            temp = 0.0
            
        cooling = str(record.get("cooling_method") or "unknown").strip().lower()
        route = str(record.get("manufacturing_route") or "unknown").strip().lower()
        
        # 3. Target Property Value Fingerprint
        target_prop = str(record.get("property_target") or "unknown").strip().lower()
        
        try:
            prop_val = record.get("prediction")
            prop = float(prop_val) if prop_val is not None else 0.0
            prop = round(prop, 2)
        except (ValueError, TypeError):
            prop = 0.0

        # 4. Bibliographic DOI
        doi = str(record.get("paper_doi") or "").strip().lower()

        # Combine into a deterministic structural dictionary
        fingerprint_structure = {
            "composition": normalized_comp,
            "heat_treatment": heat_treat,
            "annealing_temperature": temp,
            "cooling_method": cooling,
            "manufacturing_route": route,
            "property_target": target_prop,
            "property_val": prop,
            "doi": doi
        }
        
        serialized = json.dumps(fingerprint_structure, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def deduplicate(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters and removes absolute duplicates based on the composite fingerprint.
        Logs the duplicate pruning statistics.
        """
        logger.info("Starting multi-criteria deduplication on {} records.", len(records))
        
        seen_fingerprints = set()
        deduplicated_records = []
        
        for record in records:
            fingerprint = self.generate_fingerprint(record)
            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                deduplicated_records.append(record)
                
        pruned_count = len(records) - len(deduplicated_records)
        duplicate_rate = (pruned_count / len(records) * 100) if records else 0.0
        
        logger.info(
            "Deduplication completed. Remaining records: {}, Pruned: {} (Rate: {:.2f}%)",
            len(deduplicated_records),
            pruned_count,
            duplicate_rate
        )
        
        return deduplicated_records
