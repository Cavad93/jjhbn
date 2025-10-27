# gating_no_r.py
# -*- coding: utf-8 -*-
"""
Три EV-free фильтра, не использующие payout r:
1) prob_lcb  — Wilson LCB по эмпирическому win rate (учитываем неопределённость).
2) conformal — квантили ошибок |y-p| (онлайн-конформал).
3) disagree  — расхождение с толпой I_up=bet_up/(bet_up+bet_down).
"""

from __future__ import annotations
import os, csv, math, time
from collections import deque
from typing import Tuple, List, Optional
from error_logger import log_exception

try:
    import numpy as np
    HAVE_NP = True
except Exception:
    HAVE_NP = False

class _Hist:
    def __init__(self, maxlen: int = 4000):
        self.maxlen = maxlen
        self.mt = 0.0
        self.items: deque = deque(maxlen=maxlen)
        
    def load(self, csv_path: str):
        shadow_path = os.path.join(os.path.dirname(csv_path), "trades_shadow.csv")

        mt_main = 0.0
        mt_shadow = 0.0
        try:
            if os.path.exists(csv_path):
                mt_main = os.stat(csv_path).st_mtime
        except Exception:
            log_exception(f"gating_no_r: stat failed for {csv_path}")
        try:
            if os.path.exists(shadow_path):
                mt_shadow = os.stat(shadow_path).st_mtime
        except Exception:
            log_exception(f"gating_no_r: stat failed for {shadow_path}")

        mt = max(mt_main, mt_shadow)
        if mt <= self.mt:
            return
        self.mt = mt

        items: List[Tuple[float,int]] = []
        seen_epochs: set = set()

        def append_from(path: str):
            if not os.path.exists(path):
                return
            try:
                with open(path, "r", encoding="utf-8-sig", newline="") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        out = str(row.get("outcome","")).strip().lower()
                        if out not in ("win","loss"):
                            continue
                        ep = str(row.get("epoch","")).strip()
                        if ep and ep in seen_epochs:
                            continue
                        try:
                            p_up = float(row.get("p_up", "nan"))
                        except Exception:
                            continue
                        side = str(row.get("side","")).strip().upper()
                        if side not in ("UP","DOWN"):
                            continue
                        p_pred = p_up if side == "UP" else (1.0 - p_up)
                        y = 1 if out == "win" else 0
                        if math.isfinite(p_pred):
                            items.append((max(1e-6, min(1.0-1e-6, float(p_pred))), int(y)))
                            if ep:
                                seen_epochs.add(ep)
            except Exception:
                log_exception("gating_no_r: CSV parse item failed")

        if mt_main > 0.0:
            append_from(csv_path)
        if mt_shadow > 0.0:
            append_from(shadow_path)

        if len(items) > self.maxlen:
            items = items[-self.maxlen:]
        self.items = deque(items, maxlen=self.maxlen)


_H = _Hist(maxlen=4000)

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _wilson_margin(p: float, n: int, z: float) -> float:
    p = _clamp(p, 1e-6, 1.0-1e-6)
    n = max(1, int(n))
    se = math.sqrt(p * (1.0 - p) / n)
    return float(z * se)

def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    if HAVE_NP:
        return float(np.quantile(np.array(xs, dtype=float), _clamp(q,0.0,1.0)))
    xs2 = sorted(xs)
    k = int(_clamp(q,0.0,1.0) * (len(xs2)-1))
    return float(xs2[k])

def compute_p_thr_no_r(
    mode: str,
    csv_path: str,
    p_up: float,
    side: str,
    bet_up: Optional[float] = None,
    bet_down: Optional[float] = None,
    q: float = None
) -> Tuple[float, str]:
    mode = (mode or "").strip().lower() or "prob_lcb"
    side = (side or "").strip().upper()
    p_up = float(p_up)
    p_up = _clamp(p_up, 1e-6, 1.0-1e-6)
    p_side = p_up if side == "UP" else (1.0 - p_up)

    if mode == "prob_lcb":
        z = float(os.getenv("NO_R_Z", "1.04"))
        n_min = int(os.getenv("NO_R_MINN", "600"))
        _H.load(csv_path)
        n_eff = max(n_min, len(_H.items))
        
        if len(_H.items) > 0:
            empirical_wr = sum(y for (p, y) in _H.items) / len(_H.items)
        else:
            empirical_wr = 0.5
        
        moe = _wilson_margin(empirical_wr, n_eff, z)
        thr = 0.5 + moe
        return (_clamp(thr, 0.5, 0.9), f"noR_probLCB(z={z:.2f}, n_eff={n_eff}, wr={empirical_wr:.3f})")

    elif mode == "conformal":
        q_level = float(os.getenv("NO_R_Q", "0.15")) if (q is None) else float(q)
        win = int(os.getenv("NO_R_WIN", "2000"))
        _H.load(csv_path)
        items = list(_H.items)[-win:]
        errs = [abs(y - p) for (p, y) in items]
        q_err = _quantile(errs, 1.0 - _clamp(q_level, 0.01, 0.5))
        thr = 0.5 + q_err
        return (_clamp(thr, 0.5, 0.9), f"noR_conformal(q={q_level:.2f}, win={win}, q_err={q_err:.3f})")

    elif mode == "disagree":
        tau = float(os.getenv("NO_R_TAU", "0.06"))
        if (bet_up is None) or (bet_down is None) or (bet_up + bet_down) <= 0.0:
            return (0.5 + tau, f"noR_disagree(no_pool, τ={tau:.3f})")
        total = float(bet_up + bet_down)
        I_up = float(bet_up / total)
        I_side = I_up if side == "UP" else (1.0 - I_up)
        diff = float(p_side - I_side)
        
        # Если сильно выше толпы — повышаем порог (требуем уверенность)
        thr = 0.5 + tau + max(0.0, diff - tau)
        return (_clamp(thr, 0.5, 0.9), f"noR_disagree(Δ={diff:+.3f}, I_side={I_side:.3f}, τ={tau:.3f})")

    else:
        return (0.51, f"fixed(0.51; mode={mode})")
