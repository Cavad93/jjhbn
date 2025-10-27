import numpy as np
from dataclasses import dataclass

@dataclass
class BBQCalibrator:
    n_bins: int = 20
    qs_: np.ndarray = None
    rates_: np.ndarray = None

    def fit(self, p, y, sample_weight=None):
        p = np.asarray(p).ravel()
        y = np.asarray(y).ravel().astype(float)
        if sample_weight is None:
            sample_weight = np.ones_like(p)
        w = np.asarray(sample_weight).ravel()
        qs = np.quantile(p, np.linspace(0, 1, self.n_bins+1))
        qs[0] = 0.0; qs[-1] = 1.0
        rates = []
        for i in range(self.n_bins):
            m = (p >= qs[i]) & (p <= qs[i+1] if i==self.n_bins-1 else p < qs[i+1])
            if not np.any(m):
                rates.append(np.mean(y))
            else:
                rates.append(np.average(y[m], weights=w[m]))
        self.qs_, self.rates_ = qs, np.asarray(rates)
        return self

    def predict_proba(self, p):
        p = np.asarray(p).ravel()
        idx = np.searchsorted(self.qs_[1:], p, side="right")
        return self.rates_[np.clip(idx, 0, len(self.rates_)-1)]
