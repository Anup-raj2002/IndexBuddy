"""
models/predict.py
─────────────────
Loads the trained XGBoost speedup regressor from ``models/xgb_speedup.joblib``
and predicts ``log_speedup`` and ``speedup = exp(log_speedup)`` for candidate
indexes, along with per-candidate SHAP feature attributions.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

import config
from features.extract import FEATURE_NAMES


class SpeedupPredictor:
    """Wrapper around the trained XGBoost log-speedup regression model."""

    def __init__(self, model_path: str = config.MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._bundle: Optional[Dict[str, Any]] = None

    def load(self) -> Dict[str, Any]:
        if self._bundle is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Trained model not found at {self.model_path}. "
                    "Run `python -m models.train` first."
                )
            self._bundle = joblib.load(self.model_path)
        return self._bundle

    @property
    def model(self) -> Any:
        return self.load()["model"]

    @property
    def feature_names(self) -> List[str]:
        return list(self.load().get("feature_names", FEATURE_NAMES))

    @property
    def metadata(self) -> Dict[str, Any]:
        b = self.load()
        return {
            "train_templates": b.get("train_templates", []),
            "heldout_templates": b.get("heldout_templates", []),
            "metrics": b.get("metrics", {}),
            "top_shap_features": b.get("top_shap_features", []),
            "speedup_threshold": b.get("speedup_threshold", config.SPEEDUP_THRESHOLD),
        }

    def predict_one(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Return ``(predicted_speedup, predicted_log_speedup)`` for a single candidate."""
        row = [[float(features.get(col, 0.0)) for col in self.feature_names]]
        df_row = pd.DataFrame(row, columns=self.feature_names)
        pred_log = float(self.model.predict(df_row)[0])
        pred_speedup = float(math.exp(pred_log))
        return pred_speedup, pred_log

    def predict_batch(
        self, feature_dicts: Sequence[Dict[str, float]]
    ) -> List[Tuple[float, float]]:
        """Return list of ``(predicted_speedup, predicted_log_speedup)`` for a batch."""
        if not feature_dicts:
            return []
        rows = [
            [float(fd.get(col, 0.0)) for col in self.feature_names]
            for fd in feature_dicts
        ]
        df_batch = pd.DataFrame(rows, columns=self.feature_names)
        pred_logs = self.model.predict(df_batch)
        return [(float(math.exp(pl)), float(pl)) for pl in pred_logs]

    def explain_one(
        self, features: Dict[str, float], top_k: int = 6
    ) -> List[Dict[str, Any]]:
        """Return top-k per-feature SHAP contributions for a single candidate."""
        try:
            import shap

            row = [[float(features.get(col, 0.0)) for col in self.feature_names]]
            df_row = pd.DataFrame(row, columns=self.feature_names)
            explainer = shap.TreeExplainer(self.model)
            sv = explainer.shap_values(df_row)[0]
            order = np.argsort(np.abs(sv))[::-1][:top_k]
            return [
                {
                    "feature": self.feature_names[i],
                    "feature_value": float(df_row.iloc[0, i]),
                    "shap_log_impact": float(sv[i]),
                }
                for i in order
            ]
        except Exception:
            return []
