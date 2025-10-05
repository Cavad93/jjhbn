# extra_features.py
# -*- coding: utf-8 -*-
import math
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

def realized_metrics(close: pd.Series, window: int) -> Tuple[float, float, float, int]:
    """
    Возвращает (RV, BV, RQ, n) на последнем окне длиной window (в барах).
    r_t = log(C_t/C_{t-1})
    RV = sum r^2
    BV = (pi/2) * sum |r_t||r_{t-1}|, t=2..n
    RQ = (n/3) * sum r^4
    """
    if len(close) < window+1:
        return 0.0, 0.0, 0.0, 0
    s = close.iloc[-(window+1):]
    r = np.log(s/s.shift(1)).dropna().values
    n = len(r)
    if n < 2:
        return 0.0, 0.0, 0.0, 0
    RV = float(np.sum(r*r))
    BV = float((math.pi/2.0)*np.sum(np.abs(r[1:])*np.abs(r[:-1])))
    RQ = float((n/3.0)*np.sum(r**4))
    return RV, BV, RQ, n

def jump_flag_from_rv_bv_rq(RV: float, BV: float, RQ: float, n: int, z_thr: float = 3.0) -> int:
    """
    Простой BN–S тест: Z ≈ (RV - BV) / sqrt(const * RQ/n). Сигнал jump=1, если Z > z_thr.
    Константы упрощены (под практику). Для онлайн-торговли достаточно индикатора.
    """
    if n <= 1 or RQ <= 0:
        return 0
    const = 1.0  # упрощённо
    z = (RV - BV)/max(1e-12, math.sqrt(const*RQ/max(1, n)))
    return int(z > z_thr)

def amihud_illiq(df: pd.DataFrame, win: int = 20) -> float:
    """
    Amihud ≈ mean(|ret|/DollarVol) на окне win.
    """
    if len(df) < win+1:
        return 0.0
    sub = df.iloc[-(win+1):].copy()
    sub["ret"] = np.log(sub["close"]/sub["close"].shift(1))
    sub["dvol"] = sub["close"]*sub["volume"]
    x = (sub["ret"].abs()/sub["dvol"].replace(0.0, np.nan)).dropna().values
    if x.size == 0:
        return 0.0
    return float(np.mean(np.clip(x, 0, 1e6)))

def kyle_lambda(df: pd.DataFrame, win: int = 20) -> float:
    """
    Kyle's λ: регрессия |Δp| на объём за последние win минут (псевдо-оценка).
    """
    if len(df) < win+1:
        return 0.0
    sub = df.iloc[-(win+1):].copy()
    dp = (sub["close"] - sub["close"].shift(1)).abs().dropna()
    v = sub["volume"].iloc[1:len(sub)].values
    x = v.astype(float)
    y = dp.values.astype(float)
    vx = np.var(x)
    if vx <= 0:
        return 0.0
    cov = np.mean((x - x.mean())*(y - y.mean()))
    return float(max(0.0, cov/max(1e-12, vx)))

def intraday_time_features(ts_utc: pd.Timestamp) -> Dict[str, float]:
    """
    sin/cos time-of-day, dummies EU/US/ASIA, day-of-week one-hot (7).
    """
    tod = ts_utc.hour*60 + ts_utc.minute
    s = math.sin(2*math.pi * tod/1440.0)
    c = math.cos(2*math.pi * tod/1440.0)
    dow = ts_utc.weekday()  # 0=Mon
    onehot = {f"dow_{i}": (1.0 if i==dow else 0.0) for i in range(7)}
    # грубая разметка сессий (UTC):
    eu = 1.0 if 7 <= ts_utc.hour < 16 else 0.0
    us = 1.0 if (13 < ts_utc.hour < 21) or (ts_utc.hour==13 and ts_utc.minute>=30) else 0.0
    asia = 1.0 if 0 <= ts_utc.hour < 8 else 0.0
    feat = dict(tod_sin=s, tod_cos=c, EU=eu, US=us, ASIA=asia)
    feat.update(onehot)
    return feat

def idio_features(bnb_df: pd.DataFrame, btc_df: Optional[pd.DataFrame], eth_df: Optional[pd.DataFrame],
                  look_min: int = 240) -> Dict[str, float]:
    """
    Очищаем ретёрн BNB от рынка: OLS на окне 3–4ч.
    Возвращаем resid_ret_1m, beta_sum и Δbeta_sum за 60 мин.
    """
    out = dict(resid_ret_1m=0.0, beta_sum=0.0, beta_sum_d60=0.0)
    if bnb_df is None or bnb_df.empty or btc_df is None or btc_df.empty or eth_df is None or eth_df.empty:
        return out
    # синхронизация индексов
    df = pd.DataFrame({
        "bnb": np.log(bnb_df["close"]/bnb_df["close"].shift(1)),
        "btc": np.log(btc_df["close"]/btc_df["close"].shift(1)).reindex(bnb_df.index).ffill(),
        "eth": np.log(eth_df["close"]/eth_df["close"].shift(1)).reindex(bnb_df.index).ffill()
    }).dropna()
    if len(df) < look_min+2:
        return out
    sub = df.iloc[-look_min:]
    X = sub[["btc","eth"]].values
    y = sub["bnb"].values
    # беты через нормальные уравнения
    XtX = X.T @ X
    try:
        beta = np.linalg.solve(XtX + 1e-9*np.eye(2), X.T @ y)
    except np.linalg.LinAlgError:
        return out
    out["beta_sum"] = float(beta.sum())

    # текущий resid (последняя точка)
    cur = df.iloc[-1]
    yhat = beta[0]*cur["btc"] + beta[1]*cur["eth"]
    out["resid_ret_1m"] = float(cur["bnb"] - yhat)

    # изменение «динамической» беты за 60 мин (наивно: сравним суммы бета на оконцах)
    if len(df) >= look_min + 60:
        sub2 = df.iloc[-(look_min+60):-60]
        X2 = sub2[["btc","eth"]].values
        y2 = sub2["bnb"].values
        XtX2 = X2.T @ X2
        try:
            beta2 = np.linalg.solve(XtX2 + 1e-9*np.eye(2), X2.T @ y2)
            out["beta_sum_d60"] = float(beta.sum() - beta2.sum())
        except np.linalg.LinAlgError:
            pass
    return out

class GasHistory:
    def __init__(self, maxlen: int = 600):
        self.hist = deque(maxlen=maxlen)  # list of (ts, gwei)

    def push(self, ts: int, gwei: float):
        self.hist.append((ts, float(gwei)))

    def features(self, now_ts: int) -> Dict[str, float]:
        out = dict(gas_d1m=0.0, gas_vol5m=0.0)
        if not self.hist:
            return out
        # d1m
        tgt = now_ts - 60
        prev = [x for x in self.hist if x[0] <= tgt]
        if prev:
            out["gas_d1m"] = float(self.hist[-1][1] - prev[-1][1])
        # vol5m
        prev5 = [x[1] for x in self.hist if x[0] >= now_ts - 300]
        if prev5:
            out["gas_vol5m"] = float(np.std(np.array(prev5, dtype=float)))
        return out

def pack_vector(feat_dicts: Dict[str, float], names: Optional[list] = None) -> Tuple[np.ndarray, list]:
    keys = names if names is not None else list(feat_dicts.keys())
    vals = [float(feat_dicts.get(k, 0.0)) for k in keys]
    return np.array(vals, dtype=float), keys
