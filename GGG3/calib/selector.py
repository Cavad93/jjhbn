# calib/selector.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from .platt import PlattCalibrator, TemperatureScaler
from .beta import BetaCalibrator
from .isotonic import IsotonicCalibrator
from .bbq import BBQCalibrator

# ← ИСПРАВЛЕНИЕ: Импортируем ВСЕ метрики из одного места
from metrics.calibration import ece, nll, brier


@dataclass
class CalibratorSelector:
    min_n_platt: int = 50
    min_n_iso: int = 800
    pick_by: str = "nll"

    def _candidates(self, n):
        cands = {"platt": PlattCalibrator(), "temp": TemperatureScaler(), "beta": BetaCalibrator()}
        if n >= self.min_n_iso:
            cands["isotonic"] = IsotonicCalibrator()
            cands["bbq"] = BBQCalibrator(n_bins=20)
        return cands

    def fit_pick(self, raw_scores, y, input_is_logit=False, weights=None) -> Tuple[str, Any, Dict[str, float]]:
        raw_scores = np.asarray(raw_scores).ravel()
        y = np.asarray(y).ravel()
        
        if len(raw_scores) == 0 or len(y) == 0:
            raise ValueError("Empty input data")
        
        metrics = {}
        best_name, best_obj, best_val = None, None, float('inf')
        
        for name, Cal in self._candidates(len(y)).items():
            try:
                obj = Cal.fit(raw_scores, y, sample_weight=weights)
                
                if name in ["platt", "temp"]:
                    p = obj.predict_proba(raw_scores, input_is_logit=input_is_logit)
                else:
                    p = obj.predict_proba(raw_scores)
                
                cur = nll(y, p, w=weights) if self.pick_by == "nll" else brier(y, p, w=weights)
                metrics[name] = float(cur)
                
                if cur < best_val:
                    best_val, best_name, best_obj = cur, name, obj
            except Exception as e:
                print(f"[CalibratorSelector] Failed to fit {name}: {e}")
                continue
        
        if best_obj is None:
            raise RuntimeError("All calibrators failed to fit")
        
        if best_name in ["platt", "temp"]:
            p_best = best_obj.predict_proba(raw_scores, input_is_logit=input_is_logit)
        else:
            p_best = best_obj.predict_proba(raw_scores)
        
        metrics["ece"] = float(ece(y, p_best))
        metrics["nll"] = float(nll(y, p_best, w=weights))
        metrics["brier"] = float(brier(y, p_best, w=weights))  # ← ДОБАВЛЕНО для полноты
        
        return best_name, best_obj, metrics
