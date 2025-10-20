import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression

def _eps(p):
    return np.clip(p, 1e-6, 1-1e-6)

@dataclass
class BetaCalibrator:
    lr_: LogisticRegression = None

    def fit(self, p, y, sample_weight=None):
        p = np.asarray(p).ravel()
        y = np.asarray(y).ravel()
        
        if len(p) == 0 or len(y) == 0:
            raise ValueError("Empty input data")
        if len(p) != len(y):
            raise ValueError(f"Shape mismatch: p has {len(p)} samples, y has {len(y)}")
        
        p_clipped = _eps(p)
        X = np.column_stack([np.log(p_clipped), np.log(1 - p_clipped)])
        
        self.lr_ = LogisticRegression(solver="lbfgs", max_iter=200)
        self.lr_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, p):
        if self.lr_ is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        
        p = np.asarray(p).ravel()
        if len(p) == 0:
            return np.array([])
        
        p_clipped = _eps(p)
        X = np.column_stack([np.log(p_clipped), np.log(1 - p_clipped)])
        return self.lr_.predict_proba(X)[:, 1]