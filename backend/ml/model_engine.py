"""
ALLOY IQ — ML Model Engine
===========================
Implements the multi-output stacking ensemble strategy for alloy families:
  - Predicts YS, UTS, HV, Elongation simultaneously.
  - Conformal prediction via MAPIE wrapper.
  - SHAP plain-English narrative generation.
  - MLflow experiment tracking.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.multioutput import MultiOutputRegressor

# Assuming xgboost is installed
try:
    from xgboost import XGBRegressor
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor as XGBRegressor

import shap

from backend.ml.uncertainty import AlloyUncertainty
from backend.ml.narrative import generate_full_report, SHAPNarrativeGenerator
from backend.ml.mlflow_config import setup_mlflow, log_training_run

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_NAMES = ["yield_strength_mpa", "tensile_strength_mpa", "hardness_hv", "elongation_pct"]

# Cell coverage classification
CELL_COVERAGE: Dict[str, str] = {
    "steel": "rich",
    "hea": "moderate",
    "aluminum": "sparse",
}

MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Family Model — multi-output stacking ensemble
# ---------------------------------------------------------------------------
class FamilyModel:
    """Stacking ensemble for an alloy family predicting all 4 core properties."""

    def __init__(self, family: str, coverage: Literal["rich", "moderate", "sparse"], confidence: float = 0.90):
        self.family = family
        self.coverage = coverage
        self.confidence = confidence
        self._stack: Optional[Pipeline] = None
        self._conformal: Optional[AlloyUncertainty] = None
        self._explainer: Optional[shap.Explainer] = None
        self._feature_names: List[str] = []
        self._base_values: List[float] = [0.0]*4

    def _build_stack(self, params: Optional[Dict] = None) -> Pipeline:
        params = params or {}
        
        xgb = MultiOutputRegressor(XGBRegressor(
            n_estimators=params.get("xgb_n_estimators", 100),
            max_depth=params.get("xgb_max_depth", 6),
            learning_rate=params.get("xgb_lr", 0.1),
            random_state=42,
        ))
        
        # RF is natively multi-output
        rf = RandomForestRegressor(
            n_estimators=params.get("rf_n_estimators", 100),
            max_depth=params.get("rf_max_depth", None),
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        
        mlp = MultiOutputRegressor(MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=200,
            random_state=42,
        ))
        
        stack = StackingRegressor(
            estimators=[("xgb", xgb), ("rf", rf), ("mlp", mlp)],
            final_estimator=MultiOutputRegressor(Ridge(alpha=1.0)),
            cv=3,
            passthrough=True,
            n_jobs=-1,
        )

        steps = []
        if self.coverage in ["moderate", "sparse"]:
            imputer = KNNImputer(n_neighbors=5) if self.coverage == "moderate" else SimpleImputer(strategy="mean")
            steps.append(("imputer", imputer))
            
        steps.append(("scaler", StandardScaler()))
        steps.append(("stack", stack))
        
        return Pipeline(steps)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict:
        """Train ensemble + calibrate MAPIE layer. y must have all 4 TARGET_NAMES."""
        self._feature_names = list(X.columns)
        
        setup_mlflow()
        
        # Split: 80% train, 10% val (conformal), 10% test
        X_tr, X_cal, y_tr, y_cal = train_test_split(X, y[TARGET_NAMES], test_size=0.20, random_state=42)
        X_cal, X_test, y_cal, y_test = train_test_split(X_cal, y_cal, test_size=0.50, random_state=42)

        # Log transform targets
        y_tr_log = np.log1p(y_tr)
        y_cal_log = np.log1p(y_cal)

        # Build & Fit stack
        self._stack = self._build_stack()
        self._stack.fit(X_tr, y_tr_log)
        
        # Store summary stats for PDP
        self._X_median = X_tr.median(axis=0).values
        self._X_lo = X_tr.quantile(0.05).values
        self._X_hi = X_tr.quantile(0.95).values
        
        # Conformal Calibrate
        self._conformal = AlloyUncertainty(base_model=self._stack)
        self._conformal.calibrate(X_cal, y_cal_log.values, alpha=1.0 - self.confidence)

        # SHAP explainer (using RF for multi-output SHAP approximations, or XGB)
        rf_model = self._stack.named_steps["stack"].estimators_[1]
        X_tr_transformed = X_tr.copy()
        for name, transformer in self._stack.steps[:-1]:
            X_tr_transformed = pd.DataFrame(transformer.transform(X_tr_transformed), columns=self._feature_names)
        
        try:
            self._explainer = shap.TreeExplainer(rf_model)
            self._base_values = [float(ev) for ev in self._explainer.expected_value] if hasattr(self._explainer, 'expected_value') else [0.0]*4
        except Exception:
            self._base_values = [0.0]*4

        # Test metrics
        y_test_pred_log = self._stack.predict(X_test)
        y_test_pred = np.expm1(y_test_pred_log)
        
        # Sanity check correlation
        try:
            r, p = pearsonr(y_test_pred[:, 0], y_test_pred[:, 3]) # YS vs elong
            if r >= -0.2:
                print(f"WARNING: Expected negative YS-elongation correlation, got r={r:.2f}")
        except Exception:
            pass

        metrics = {"n_train": len(X_tr)}
        for i, target in enumerate(TARGET_NAMES):
            ss_res = np.sum((y_test.iloc[:, i] - y_test_pred[:, i]) ** 2)
            ss_tot = np.sum((y_test.iloc[:, i] - y_test.iloc[:, i].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            metrics[f"{target}_r2"] = float(r2)

        log_training_run(params={"coverage": self.coverage}, metrics=metrics, model=self._stack, artifacts=[])

        return metrics

    def predict(self, X: pd.DataFrame) -> Dict:
        confidence_map = {"rich": "high", "moderate": "moderate", "sparse": "low"}
        
        if self._stack is None or self._conformal is None:
            # Fallback mock for dev
            res = {}
            for t in TARGET_NAMES:
                res[t] = {"mean": 800.0, "lower": 750.0, "upper": 850.0}
            return {
                "predictions": res,
                "confidence_level": self.confidence,
                "data_confidence": confidence_map[self.coverage],
                "shap": {"frac_C": 100.0, "frac_Fe": 50.0},
                "narrative": "Model not loaded. This is a placeholder.",
            }

        # Conformal predictions
        uq_res = self._conformal.predict(X)
        
        # SHAP
        scaler = self._stack.named_steps["scaler"]
        X_scaled = pd.DataFrame(scaler.transform(X[:1]), columns=self._feature_names)
        rf_model = self._stack.named_steps["stack"].estimators_[1]
        
        try:
            shap_values = shap.TreeExplainer(rf_model).shap_values(X_scaled)
            # shap_values could be list of arrays per target
            shap_dict = {
                t: {f: float(shap_values[i][0][j]) for j, f in enumerate(self._feature_names)}
                for i, t in enumerate(TARGET_NAMES)
            }
        except Exception:
            shap_dict = {t: {} for t in TARGET_NAMES}

        # Generate narratives
        narratives = generate_full_report(shap_dict, uq_res, uq_res)

        # Average interval width logic for data confidence
        mean_val = uq_res["yield_strength_mpa"]["mean"]
        interval_width = uq_res["yield_strength_mpa"]["upper"] - uq_res["yield_strength_mpa"]["lower"]
        avg_width = mean_val * 0.18
        if interval_width < avg_width * 1.5:
            data_conf = "high"
        elif interval_width < avg_width * 3.0:
            data_conf = "moderate"
        else:
            data_conf = "low"

        return {
            "predictions": uq_res,
            "confidence_level": self.confidence,
            "data_confidence": data_conf,
            "shap_dicts": shap_dict,
            "narratives": narratives,
        }

    def save(self) -> None:
        cell_id = f"{self.family}__multi"
        joblib.dump(self._stack, MODEL_DIR / f"{cell_id}__stack.pkl")
        self._conformal.save(MODEL_DIR / f"{cell_id}__conformal.pkl")
        meta = {
            "family": self.family,
            "coverage": self.coverage,
            "confidence": self.confidence,
            "feature_names": self._feature_names,
            "base_values": self._base_values,
            "X_median": self._X_median.tolist() if hasattr(self, "_X_median") else None,
            "X_lo": self._X_lo.tolist() if hasattr(self, "_X_lo") else None,
            "X_hi": self._X_hi.tolist() if hasattr(self, "_X_hi") else None,
        }
        (MODEL_DIR / f"{cell_id}__meta.json").write_text(json.dumps(meta, indent=2))
        (MODEL_DIR / "target_names.json").write_text(json.dumps(TARGET_NAMES))

    @classmethod
    def load(cls, family: str) -> "FamilyModel":
        cell_id = f"{family}__multi"
        meta = json.loads((MODEL_DIR / f"{cell_id}__meta.json").read_text())
        model = cls(family, meta["coverage"], meta["confidence"])
        model._stack = joblib.load(MODEL_DIR / f"{cell_id}__stack.pkl")
        
        # Load conformal predictor properly
        from backend.ml.uncertainty import AlloyUncertainty
        conformal_obj = joblib.load(MODEL_DIR / f"{cell_id}__conformal.pkl")
        model._conformal = conformal_obj
        
        model._feature_names = meta["feature_names"]
        model._base_values = meta.get("base_values", [0.0]*4)
        if meta.get("X_median") is not None:
            model._X_median = np.array(meta["X_median"])
            model._X_lo = np.array(meta["X_lo"])
            model._X_hi = np.array(meta["X_hi"])
        return model

# ---------------------------------------------------------------------------
# AlloyModelEngine — top-level registry & router
# ---------------------------------------------------------------------------
class AlloyModelEngine:
    """Loads / trains / routes predictions across all families."""

    def __init__(self):
        self._models: Dict[str, FamilyModel] = {}

    @staticmethod
    def wait_for_ingestion(timeout_sec: int = 600, check_interval: int = 5) -> bool:
        tracker_path = Path(__file__).parent.parent.parent / "agent_tracker.json"
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            if tracker_path.exists():
                try:
                    with open(tracker_path, "r") as f:
                        tracker_data = json.load(f)
                    parquets = tracker_data.get("parquet_files", {})
                    if all(t in str(parquets) for t in ["tier1", "tier2", "tier3", "tier4"]):
                        return True
                except json.JSONDecodeError:
                    pass
            time.sleep(check_interval)
        return False

    def load_or_create(self, family: str) -> FamilyModel:
        if family in self._models:
            return self._models[family]

        # Try loading persisted model
        meta_path = MODEL_DIR / f"{family}__multi__meta.json"
        if meta_path.exists():
            m = FamilyModel.load(family)
        else:
            coverage = CELL_COVERAGE.get(family, "sparse")
            m = FamilyModel(family, coverage)

        self._models[family] = m
        return m

    def predict(self, family: str, X: pd.DataFrame) -> Dict:
        model = self.load_or_create(family)
        return model.predict(X)

    def train(self, family: str, X: pd.DataFrame, y: pd.DataFrame) -> Dict:
        coverage = CELL_COVERAGE.get(family, "sparse")
        model = FamilyModel(family, coverage)
        metrics = model.fit(X, y)
        model.save()
        self._models[family] = model
        return {"family": family, **metrics}

    def get_model(self, family: str):
        return self.load_or_create(family)
