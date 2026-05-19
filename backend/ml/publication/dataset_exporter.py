import os
import pandas as pd
from typing import Dict, Any, List
from loguru import logger

class DatasetExporter:
    """
    Exports clean datasets, processing parameters, features, and splits to CSV format.
    Enforces standardized files structure without hardcoded absolute paths.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export_cleaned_dataset(self, df: pd.DataFrame, filename: str = "cleaned_dataset.csv") -> str:
        """Exports the core sanitized dataset."""
        path = os.path.join(self.output_dir, filename)
        # Drop sensitive columns or database indices if present
        export_df = df.copy()
        if "user_id" in export_df.columns:
            export_df.drop(columns=["user_id"], inplace=True)
            
        export_df.to_csv(path, index=False)
        logger.info("Exported cleaned dataset to: {}", path)
        return path

    def export_processing_metadata(self, metadata: List[Dict[str, Any]], filename: str = "processing_metadata.csv") -> str:
        """Exports categorical heat treatments and thermal budget metrics."""
        path = os.path.join(self.output_dir, filename)
        df = pd.DataFrame(metadata)
        df.to_csv(path, index=False)
        logger.info("Exported processing metadata to: {}", path)
        return path

    def export_feature_definitions(self, features_dict: Dict[str, str], filename: str = "feature_definitions.csv") -> str:
        """Saves a descriptors features registry mapping names to chemical descriptions."""
        path = os.path.join(self.output_dir, filename)
        records = [{"feature_name": k, "definition": v} for k, v in features_dict.items()]
        df = pd.DataFrame(records)
        df.to_csv(path, index=False)
        logger.info("Exported feature definitions to: {}", path)
        return path

    def export_split_records(self, train_indices: List[int], test_indices: List[int], filename: str = "split_records.csv") -> str:
        """Saves data split records to guarantee leakage-free validation reproduction."""
        path = os.path.join(self.output_dir, filename)
        records = []
        for idx in train_indices:
            records.append({"data_index": idx, "split_partition": "train"})
        for idx in test_indices:
            records.append({"data_index": idx, "split_partition": "test"})
            
        df = pd.DataFrame(records)
        df.to_csv(path, index=False)
        logger.info("Exported data split records to: {}", path)
        return path
