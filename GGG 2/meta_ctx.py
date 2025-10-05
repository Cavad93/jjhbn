# meta_ctx.py
# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd

# Лёгкие утилиты (локальные, чтобы не плодить импортов)
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=int(max(2, n)), adjust=False).mean()

def macd_hist(close: pd.Series, fast=12, slow=26, sig=9) -> pd.Series:
    macd = ema(close, fast) - ema(close, slow)
    signal = ema(macd, sig)
    return macd - signal

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first","high": "max","low": "min","close": "last","volume": "sum"}
    out = df[["open","high","low","close","volume"]].resample(rule, label="right", closed="right").agg(agg)
    return out.dropna(how="any")

def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)

def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def build_regime_ctx(
    df_1m: pd.DataFrame,
    feats: dict,
    tstamp: pd.Timestamp,
    micro_feats: dict,
    fut_feats: dict,
    jump_flag: float = 0.0,
    htf_rule: str = "15min"
) -> dict:
    """
    Возвращает ψ (контекст) ИЗ ДАННЫХ ДО lock:
      - trend_sign, trend_abs (|MACD_hist| z-норм к 100 барам HTF)
      - vol_ratio = atr_now/atr_sma
      - jump_flag (0/1)
      - ofi_sign (из ofi_15s)
      - book_imb ([-1..1])
      - basis_sign (знак basis_now)
      - funding_sign (-1/0/1)
    """
    # --- 1) Тренд (HTF MACD-hist)
    try:
        htf = resample_ohlc(df_1m, htf_rule)
        i = htf.index.get_indexer([tstamp], method="pad")[0]
        if i <= 0:
            trend_sign = 0.0; trend_abs = 0.0
        else:
            h = macd_hist(htf["close"])
            h_win = h.iloc[max(0, i-100):i]  # только прошлое
            std = float(h_win.std(ddof=0)) if len(h_win) else 0.0
            h_val = float(h.iloc[i-1])
            trend_sign = _sign(h_val)
            trend_abs = 0.0 if std == 0.0 else _clip(abs(h_val)/std, 0.0, 3.0)
    except Exception:
        trend_sign, trend_abs = 0.0, 0.0

    # --- 2) Вола
    try:
        j = feats["atr"].index.get_indexer([tstamp], method="pad")[0]
        atr_now = float(feats["atr"].iloc[j])
        atr_sma = float(feats["atr_sma"].iloc[j])
        vol_ratio = 0.0 if atr_sma <= 0 else _clip(atr_now/atr_sma, 0.0, 5.0)
    except Exception:
        vol_ratio = 0.0

    # --- 3) Микроструктура/лента
    ofi15 = float(micro_feats.get("ofi_15s", 0.0))
    ofi_sign = _sign(ofi15)
    book_imb = _clip(float(micro_feats.get("book_imb", 0.0)), -1.0, 1.0)

    # --- 4) Фьючи
    basis_sign = _sign(float(fut_feats.get("basis_now", 0.0)))
    funding_sign = _sign(float(fut_feats.get("funding_sign", 0.0)))

    # jump_flag уже посчитан снаружи (BN–S тест/прокси)
    jf = 1.0 if bool(jump_flag) else 0.0

    return dict(
        trend_sign=trend_sign,
        trend_abs=trend_abs,
        vol_ratio=vol_ratio,
        jump_flag=jf,
        ofi_sign=ofi_sign,
        book_imb=book_imb,
        basis_sign=basis_sign,
        funding_sign=funding_sign,
    )

def pack_ctx(ctx: dict):
    """
    Фиксированный порядок ψ для гейтера.
    Возвращает (вектор numpy, имена).
    """
    names = [
        "trend_sign","trend_abs","vol_ratio","jump_flag",
        "ofi_sign","book_imb","basis_sign","funding_sign","bias"
    ]
    vec = np.array([
        float(ctx.get("trend_sign", 0.0)),
        float(ctx.get("trend_abs", 0.0)),
        float(ctx.get("vol_ratio", 0.0)),
        float(ctx.get("jump_flag", 0.0)),
        float(ctx.get("ofi_sign", 0.0)),
        float(ctx.get("book_imb", 0.0)),
        float(ctx.get("basis_sign", 0.0)),
        float(ctx.get("funding_sign", 0.0)),
        1.0  # bias
    ], dtype=float)
    return vec, names

# Простая жёсткая фазовая разметка (~6 фаз) для EXP4
def phase_from_ctx(ctx: dict) -> int:
    """
    0: bull_low, 1: bull_high, 2: bear_low, 3: bear_high, 4: flat_low, 5: flat_high
    """
    ts = float(ctx.get("trend_sign", 0.0))
    vol = float(ctx.get("vol_ratio", 0.0))
    high = (vol >= 1.3)
    if ts > 0:
        return 1 if high else 0
    if ts < 0:
        return 3 if high else 2
    # flat
    return 5 if high else 4
