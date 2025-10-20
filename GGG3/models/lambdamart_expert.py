import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence
import lightgbm as lgb

@dataclass
class LambdaMARTExpert:
    params: dict
    model: lgb.LGBMRanker = None
    feature_names: Optional[Sequence[str]] = None

    def fit(self, X, y, groups):
        self.model = lgb.LGBMRanker(**self.params)
        self.model.fit(X, y, group=groups)
        return self

    def predict_score(self, X):
        return self.model.predict(X)

    def predict_proba_sigmoid(self, X):
        s = self.predict_score(X)
        return 1/(1+np.exp(-s))
