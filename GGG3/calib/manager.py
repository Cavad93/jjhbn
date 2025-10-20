import os, time
import numpy as np
from .selector import CalibratorSelector, nll, brier
from .holdout_manager import HoldoutCalibManager  # НОВОЕ

class OnlineCalibManager:
    def __init__(self):
        self.global_cal = None
        self.roll_cal = None
        self.roll_buf = []
        self.roll_y = []
        self.holdout_mgr = HoldoutCalibManager()
        self.selector = CalibratorSelector()  # ДОБАВИТЬ
        self.roll_hours = int(os.getenv("CALIB_ROLLING_H","72"))
        self.min_n = int(os.getenv("CALIB_MIN_N","200"))

    def update(self, p_raw, y, ts):
        self.roll_buf.append((ts, float(p_raw)))
        self.roll_y.append((ts, int(y)))
        horizon = self.roll_hours*3600
        now = ts
        self.roll_buf = [(t,p) for (t,p) in self.roll_buf if now - t <= horizon]
        self.roll_y   = [(t,y) for (t,y) in self.roll_y   if now - t <= horizon]

        if len(self.roll_y) >= self.min_n:
            pr = np.array([p for _,p in self.roll_buf])
            yr = np.array([y for _,y in self.roll_y])
            
            # ИЗМЕНЕНО: используем hold-out валидацию
            cal, holdout_nll = self.holdout_mgr.fit_with_holdout(pr, yr)
            if cal is not None:
                self.roll_cal = cal
                print(f"[calib] Retrained with hold-out NLL={holdout_nll:.4f}")

    def fit_global(self, p_raw_hist, y_hist):
        _, self.global_cal, _ = self.selector.fit_pick(p_raw_hist, y_hist)

    def transform(self, p_raw):
        if self.roll_cal is not None:
            return self.roll_cal.predict_proba([p_raw])[0]
        if self.global_cal is not None:
            return self.global_cal.predict_proba([p_raw])[0]
        return float(p_raw)
