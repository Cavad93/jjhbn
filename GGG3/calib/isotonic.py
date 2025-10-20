import numpy as np
from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression

@dataclass
class IsotonicCalibrator:
    iso_: IsotonicRegression = None

    def fit(self, p, y, sample_weight=None):
        p = np.asarray(p).ravel()
        y = np.asarray(y).ravel()
        self.iso_ = IsotonicRegression(out_of_bounds="clip")
        self.iso_.fit(p, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, p):
        return self.iso_.predict(np.asarray(p).ravel())
