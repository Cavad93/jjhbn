import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from .platt import PlattCalibrator, TemperatureScaler
from .beta import BetaCalibrator
from .isotonic import IsotonicCalibrator
from .bbq import BBQCalibrator

def brier(y, p, w=None):
    y = np.asarray(y).ravel().astype(float)
    p = np.asarray(p).ravel().astype(float)
    if w is None: w = np.ones_like(y)
    return np.average((p - y)**2, weights=w)

def nll(y, p, w=None):
    y = np.asarray(y).ravel().astype(float)
    p = np.clip(np.asarray(p).ravel(), 1e-12, 1-1e-12)
    if w is None: w = np.ones_like(y)
    return -np.average(y*np.log(p) + (1-y)*np.log(1-p), weights=w)

def ece(y, p, n_bins=15):
    y = np.asarray(y).ravel().astype(float)
    p = np.asarray(p).ravel().astype(float)
    bins = np.linspace(0,1,n_bins+1)
    idx = np.digitize(p, bins) - 1
    e = 0.0
    for b in range(n_bins):
        m = idx==b
        if not np.any(m): continue
        conf = p[m].mean()
        acc = y[m].mean()
        e += (m.sum() / len(y)) * abs(acc - conf)
    return e

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
                
                cur = nll(y, p) if self.pick_by == "nll" else brier(y, p)
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
        metrics["nll"] = float(nll(y, p_best))
        
        return best_name, best_obj, metrics