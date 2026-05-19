import os
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import GroupKFold
from loguru import logger
from sqlalchemy.orm import Session

from backend.ml.lifecycle.ensemble_manager import EnsembleManager
from backend.ml.validation.experiment_registry import ExperimentRegistry

class ModelTrainer:
    """
    Orchestrates robust, leakage-free model training via GroupKFold.
    Integrates with cryptographic registry ledger and validates processing-aware features.
    """
    def __init__(
        self,
        registry_path: str = "./data/experiments/registry.json",
        confidence: float = 0.90
    ):
        self.registry = ExperimentRegistry(registry_path=registry_path)
        self.confidence = confidence

    @staticmethod
    def compute_feature_hash(features: List[str]) -> str:
        """Helper to compute feature hash consistently."""
        feature_str = ",".join(sorted(str(f).strip().lower() for f in features))
        return hashlib.sha256(feature_str.encode("utf-8")).hexdigest()

    def verify_features(self, df: pd.DataFrame, features: List[str], expected_feature_hash: Optional[str] = None):
        """
        Validates the presence of processing-aware features, and
        asserts feature hash matches to prevent layout mismatches.
        """
        # Ensure at least some processing-aware columns are present in feature names
        processing_keywords = {"cooling", "annealing", "aging", "temp", "route", "thermal", "treatment"}
        has_processing = any(any(kw in f.lower() for kw in processing_keywords) for f in features)
        if not has_processing:
            logger.warning("No standard processing-aware features found in feature set!")

        # Check missing columns in DataFrame
        missing = [col for col in features if col not in df.columns]
        if missing:
            raise ValueError(f"Features missing from training dataset: {missing}")

        # Check expected feature hash
        if expected_feature_hash:
            actual_hash = self.compute_feature_hash(features)
            if actual_hash != expected_feature_hash:
                raise ValueError(
                    f"Feature hash mismatch! Expected: {expected_feature_hash}, Got: {actual_hash}"
                )

    def run_group_kfold_training(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_columns: List[str],
        group_column: str,
        coverage: str = "rich",
        n_splits: int = 3,
        expected_feature_hash: Optional[str] = None,
        checkpoint_callback: Optional[callable] = None
    ) -> Tuple[EnsembleManager, Dict[str, Any]]:
        """
        Performs GroupKFold cross-validation, saves checkpoint metrics,
        trains the final production model, and returns (fitted_manager, global_metrics).
        """
        self.verify_features(df, features, expected_feature_hash)
        
        if group_column not in df.columns:
            raise ValueError(f"Group column {group_column} not found in training dataset.")

        X = df[features]
        y = df[target_columns]
        groups = df[group_column]

        gkf = GroupKFold(n_splits=n_splits)
        fold_checkpoints = []

        logger.info("Starting GroupKFold training (n_splits={}) grouped by {}", n_splits, group_column)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # Fit intermediate fold ensemble
            fold_manager = EnsembleManager(coverage=coverage, confidence=self.confidence)
            # Use split within train for calibration
            cal_split = int(len(X_tr) * 0.20)
            if cal_split < 2:
                cal_split = 2
            X_tr_fit, y_tr_fit = X_tr.iloc[:-cal_split], y_tr.iloc[:-cal_split]
            X_tr_cal, y_tr_cal = X_tr.iloc[-cal_split:], y_tr.iloc[-cal_split:]

            fold_manager.fit_and_calibrate(X_tr_fit, y_tr_fit, X_tr_cal, y_tr_cal)

            # Evaluate on validation split
            y_pred_log = fold_manager.pipeline.predict(X_val)
            y_pred = np.expm1(y_pred_log)

            # Calculate fold MAE & R2
            fold_metrics = {}
            for idx, target in enumerate(target_columns):
                ss_res = np.sum((y_val.iloc[:, idx] - y_pred[:, idx]) ** 2)
                ss_tot = np.sum((y_val.iloc[:, idx] - y_val.iloc[:, idx].mean()) ** 2)
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                mae = float(np.mean(np.abs(y_val.iloc[:, idx] - y_pred[:, idx])))
                fold_metrics[f"{target}_r2"] = float(r2)
                fold_metrics[f"{target}_mae"] = float(mae)

            checkpoint = {
                "fold": fold,
                "metrics": fold_metrics
            }
            fold_checkpoints.append(checkpoint)
            logger.info("Fold {} completed. yield_strength_mpa_r2: {:.3f}", fold, fold_metrics.get("yield_strength_mpa_r2", 0.0))

            if checkpoint_callback:
                checkpoint_callback(fold, checkpoint)

        # Train final production model on full dataset
        logger.info("Training final production ensemble on complete dataset.")
        prod_manager = EnsembleManager(coverage=coverage, confidence=self.confidence)
        
        # 80/20 split for fit vs calibration
        cal_split = int(len(X) * 0.20)
        if cal_split < 2:
            cal_split = 2
        X_fit, y_fit = X.iloc[:-cal_split], y.iloc[:-cal_split]
        X_cal, y_cal = X.iloc[-cal_split:], y.iloc[-cal_split:]

        prod_manager.fit_and_calibrate(X_fit, y_fit, X_cal, y_cal)

        # Compute production global metrics on calibration split
        prod_pred_log = prod_manager.pipeline.predict(X_cal)
        prod_pred = np.expm1(prod_pred_log)

        global_metrics = {
            "n_train_total": len(df),
            "fold_checkpoints": fold_checkpoints,
        }
        for idx, target in enumerate(target_columns):
            ss_res = np.sum((y_cal.iloc[:, idx] - prod_pred[:, idx]) ** 2)
            ss_tot = np.sum((y_cal.iloc[:, idx] - y_cal.iloc[:, idx].mean()) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            mae = float(np.mean(np.abs(y_cal.iloc[:, idx] - prod_pred[:, idx])))
            global_metrics[f"{target}_r2"] = float(r2)
            global_metrics[f"{target}_mae"] = float(mae)

        return prod_manager, global_metrics

    def log_to_registry(
        self,
        df: pd.DataFrame,
        features: List[str],
        model_bytes: bytes,
        metrics: Dict[str, Any],
        training_config: Optional[Dict[str, Any]] = None,
        db_session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """Saves execution run metrics to cryptographic ledger file & ExperimentRun DB table."""
        return self.registry.log_run(
            train_df=df,
            features=features,
            model_bytes=model_bytes,
            metrics=metrics,
            training_config=training_config,
            db_session=db_session
        )
