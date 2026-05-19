import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

try:
    from xgboost import XGBRegressor
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor as XGBRegressor

from backend.ml.uncertainty import AlloyUncertainty

class EnsembleManager:
    """
    Manages building, training, calibrating, and saving the multi-output stacking ensemble.
    Compatible with the existing FamilyModel architecture.
    """
    def __init__(self, coverage: str = "rich", confidence: float = 0.90):
        self.coverage = coverage
        self.confidence = confidence
        self.pipeline: Optional[Pipeline] = None
        self.conformal: Optional[AlloyUncertainty] = None

    def build_pipeline(self, params: Optional[Dict] = None) -> Pipeline:
        params = params or {}
        
        xgb = XGBRegressor(
            n_estimators=params.get("xgb_n_estimators", 100),
            max_depth=params.get("xgb_max_depth", 6),
            learning_rate=params.get("xgb_lr", 0.1),
            random_state=42,
        )
        
        rf = RandomForestRegressor(
            n_estimators=params.get("rf_n_estimators", 100),
            max_depth=params.get("rf_max_depth", None),
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=200,
            random_state=42,
        )
        
        stack = MultiOutputRegressor(StackingRegressor(
            estimators=[("xgb", xgb), ("rf", rf), ("mlp", mlp)],
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            passthrough=True,
            n_jobs=-1,
        ))

        steps = []
        if self.coverage in ["moderate", "sparse"]:
            imputer = KNNImputer(n_neighbors=5) if self.coverage == "moderate" else SimpleImputer(strategy="mean")
            steps.append(("imputer", imputer))
            
        steps.append(("scaler", StandardScaler()))
        steps.append(("stack", stack))
        
        self.pipeline = Pipeline(steps)
        return self.pipeline

    def fit_and_calibrate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_cal: pd.DataFrame,
        y_cal: pd.DataFrame,
        params: Optional[Dict] = None
    ) -> Pipeline:
        """
        Trains the full multi-output stacking ensemble on train data, and
        calibrates the Conformal Predictors (AlloyUncertainty) on calibration data.
        """
        if self.pipeline is None:
            self.build_pipeline(params)

        # Fit stack (targets must be log-transformed)
        y_train_log = np.log1p(y_train)
        self.pipeline.fit(X_train, y_train_log)

        # Calibrate Conformal Predictor
        y_cal_log = np.log1p(y_cal)
        self.conformal = AlloyUncertainty(base_model=self.pipeline)
        self.conformal.calibrate(X_cal, y_cal_log.values, alpha=1.0 - self.confidence)

        return self.pipeline

    def save(self, base_path: str):
        """Saves the pipeline and conformal objects to disk."""
        os.path.dirname(base_path) and os.makedirs(os.path.dirname(base_path), exist_ok=True)
        joblib.dump(self.pipeline, f"{base_path}_stack.pkl")
        if self.conformal:
            self.conformal.save(f"{base_path}_conformal.pkl")

    def load(self, base_path: str):
        """Loads both components from disk."""
        self.pipeline = joblib.load(f"{base_path}_stack.pkl")
        if os.path.exists(f"{base_path}_conformal.pkl"):
            self.conformal = AlloyUncertainty.load(f"{base_path}_conformal.pkl")
