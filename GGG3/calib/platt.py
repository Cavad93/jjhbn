import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

def _as_col(x):
    x = np.asarray(x)
    return x.reshape(-1, 1) if x.ndim == 1 else x

@dataclass
class PlattCalibrator:
    C: float = 1.0
    max_iter: int = 200
    lr_: LogisticRegression = None

    def fit(self, s, y, sample_weight=None, input_is_logit=False):
        s = np.asarray(s).ravel()
        z = s if input_is_logit else np.log(np.clip(s, 1e-6, 1-1e-6) / (1 - np.clip(s, 1e-6, 1-1e-6)))
        y = np.asarray(y).ravel()
        self.lr_ = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")
        self.lr_.fit(_as_col(z), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, s, input_is_logit=False):
        z = s if input_is_logit else np.log(np.clip(s, 1e-6, 1-1e-6) / (1 - np.clip(s, 1e-6, 1-1e-6)))
        return self.lr_.predict_proba(_as_col(z))[:, 1]

@dataclass
class TemperatureScaler:
    T_: float = 1.0

    def _nll(self, T, logits, y, w=None):
        z = logits / max(T[0], 1e-6)
        p = 1/(1+np.exp(-z))
        eps = 1e-12
        ll = y*np.log(np.clip(p,eps,1)) + (1-y)*np.log(np.clip(1-p,eps,1))
        return -np.average(ll, weights=w)

    def fit(self, logits, y, sample_weight=None):
        logits = np.asarray(logits).ravel()
        y = np.asarray(y).ravel()
        res = minimize(self._nll, x0=[1.0], args=(logits, y, sample_weight), bounds=[(1e-3, 100)])
        self.T_ = float(res.x[0])
        return self

    def predict_proba(self, logits):
        z = np.asarray(logits).ravel() / max(self.T_, 1e-6)
        return 1/(1+np.exp(-z))
