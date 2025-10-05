# -*- coding: utf-8 -*-
from __future__ import annotations
import os, pickle, math
from typing import Optional, List
import numpy as np

# опциональные зависимости
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    HAVE_SK = True
except Exception:
    IsotonicRegression = None
    LogisticRegression = None
    HAVE_SK = False

def _logit(p: float) -> float:
    p = float(min(max(p, 1e-6), 1.0 - 1e-6))
    return math.log(p/(1.0 - p))

class _BaseCal:
    def __init__(self, window: int = 2000):
        self.window = int(window)
        self.P: List[float] = []
        self.Y: List[int] = []
        self.ready: bool = False

    def observe(self, p_raw: float, y: int):
        self.P.append(float(p_raw))
        self.Y.append(int(y))
        if len(self.P) > self.window:
            self.P = self.P[-self.window:]
            self.Y = self.Y[-self.window:]

    def save(self, path: str):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self, f)
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> Optional["_BaseCal"]:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def transform(self, p_raw: float) -> float:
        raise NotImplementedError

    def maybe_fit(self, min_samples: int = 200, every: int = 100) -> bool:
        raise NotImplementedError

# --- 1) Platt: sigmoid(A * logit(p_raw) + B) ---
class PlattCalibrator(_BaseCal):
    def __init__(self, window: int = 2000):
        super().__init__(window=window)
        self.A: float = 1.0
        self.B: float = 0.0
        self._since = 0
        self._clf = None  # sklearn LR (если доступен)

    def transform(self, p_raw: float) -> float:
        z = self.A * _logit(p_raw) + self.B
        z = max(min(z, 60.0), -60.0)
        return 1.0/(1.0 + math.exp(-z))

    def maybe_fit(self, min_samples: int = 200, every: int = 100) -> bool:
        self._since += 1
        if len(self.P) < min_samples or self._since < every:
            return False
        self._since = 0
        try:
            X = np.array([[_logit(p)] for p in self.P], dtype=np.float64)
            y = np.array(self.Y, dtype=np.int32)
            if HAVE_SK and LogisticRegression is not None:
                clf = LogisticRegression(solver="lbfgs")
                clf.fit(X, y)
                self._clf = clf
                self.A = float(clf.coef_[0,0])
                self.B = float(clf.intercept_[0])
            else:
                # простой градиентный спуск по A,B
                A, B = self.A, self.B
                lr = 0.05
                for _ in range(200):
                    z = A*X[:,0] + B
                    p = 1.0/(1.0 + np.exp(-np.clip(z, -60, 60)))
                    gA = np.mean((p - y) * X[:,0])
                    gB = np.mean(p - y)
                    A -= lr * gA
                    B -= lr * gB
                self.A, self.B = float(A), float(B)
            self.ready = True
            return True
        except Exception:
            return False

# --- 2) Isotonic: p_cal = iso(p_raw) ---
class IsotonicCalibrator(_BaseCal):
    def __init__(self, window: int = 4000):
        super().__init__(window=window)
        self.iso = IsotonicRegression(out_of_bounds="clip") if HAVE_SK and IsotonicRegression else None
        self._since = 0

    def transform(self, p_raw: float) -> float:
        if self.iso is None or not self.ready:
            return float(min(max(p_raw, 1e-6), 1.0 - 1e-6))
        return float(self.iso.predict([p_raw])[0])

    def maybe_fit(self, min_samples: int = 400, every: int = 200) -> bool:
        self._since += 1
        if self.iso is None or len(self.P) < min_samples or self._since < every:
            return False
        self._since = 0
        try:
            self.iso.fit(np.array(self.P, dtype=np.float64), np.array(self.Y, dtype=np.int32))
            self.ready = True
            return True
        except Exception:
            return False

# --- 3) Temperature: p_cal = sigmoid(logit(p_raw)/T) ---
class TemperatureCalibrator(_BaseCal):
    def __init__(self, window: int = 2000):
        super().__init__(window=window)
        self.T: float = 1.0
        self._since = 0

    def transform(self, p_raw: float) -> float:
        z = _logit(p_raw) / max(1e-6, self.T)
        z = max(min(z, 60.0), -60.0)
        return 1.0/(1.0 + math.exp(-z))

    def maybe_fit(self, min_samples: int = 200, every: int = 100) -> bool:
        self._since += 1
        if len(self.P) < min_samples or self._since < every:
            return False
        self._since = 0
        try:
            z = np.array([_logit(p) for p in self.P], dtype=np.float64)
            y = np.array(self.Y, dtype=np.float64)
            best_T, best_loss = self.T, float("inf")
            for T in np.linspace(0.5, 3.0, 26):
                zz = np.clip(z / T, -60, 60)
                p = 1.0/(1.0 + np.exp(-zz))
                eps = 1e-9
                loss = -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
                if loss < best_loss:
                    best_loss, best_T = loss, float(T)
            self.T = float(best_T)
            self.ready = True
            return True
        except Exception:
            return False

def make_calibrator(method: str = "platt") -> _BaseCal:
    m = (method or "platt").lower()
    if m == "isotonic":
        return IsotonicCalibrator()
    if m == "temp" or m == "temperature":
        return TemperatureCalibrator()
    return PlattCalibrator()
