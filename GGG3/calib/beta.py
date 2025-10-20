import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression

def _eps(p):
    return np.clip(p, 1e-6, 1-1e-6)

@dataclass
class BetaCalibrator:
    lr_: LogisticRegression = None
    add_bias: bool = True

    def fit(self, p, y, sample_weight=None):
        p = _eps(np.asarray(p).ravel())
        y = np.asarray(y).ravel()
        X = np.column_stack([np.log(p), np.log(1 - p)])
        self.lr_ = LogisticRegression(solver="lbfgs", max_iter=200)
        self.lr_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, p):
        p = _eps(np.asarray(p).ravel())
        X = np.column_stack([np.log(p), np.log(1 - p)])
        return self.lr_.predict_proba(X)[:, 1]
