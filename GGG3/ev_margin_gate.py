# ev_margin_gate.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

def p_thr_from_ev(r_hat: float, stake: float, gb_hat: float, gc_hat: float, delta: float) -> float:
    """
    EV = p*r_hat - 1 - (gb+gc)/stake > δ  ⇒  p > (1 + (gb+gc)/stake + δ) / r_hat
    
    ВАЖНО: delta должна быть >= 0 для корректной работы защитного порога.
    Возвращает 1.0 если сделка невозможна (порог >= 1.0).
    """
    if delta < 0:
        delta = 0.0
    
    r = float(max(1e-9, r_hat))
    gas_per_stake = float(max(0.0, (float(gb_hat) + float(gc_hat)) / max(1e-9, float(stake))))
    thr = (1.0 + gas_per_stake + float(delta)) / r
    
    if thr >= 1.0:
        return 1.0
    return float(max(0.0, min(thr, 0.999999)))

def loss_margin_q(csv_path: str, max_epoch_exclusive: Optional[int] = None, q: float = 0.90) -> float:
    """
    Q-quantile по margin = p_side - 1/r на ИСТОРИЧЕСКИХ ПРОИГРЫШАХ.
    Только строки до max_epoch_exclusive.
    
    Возвращает float('inf') если недостаточно данных (< 10 проигрышей),
    что блокирует прохождение margin-фильтра.
    """
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        return float('inf')
    
    if df is None or df.empty:
        return float('inf')
    
    df = df.dropna(subset=["outcome","p_up","side"])
    
    if "epoch" in df.columns and max_epoch_exclusive is not None:
        try:
            df = df[df["epoch"] < int(max_epoch_exclusive)]
        except Exception:
            try:
                from error_logger import log_exception
                log_exception("Unhandled exception")
            except (ImportError, Exception):
                pass
    
    if df.empty:
        return float('inf')

    p_up = pd.to_numeric(df["p_up"], errors="coerce").astype(float)
    p_side = np.where(df["side"].astype(str).str.upper() == "UP", p_up, 1.0 - p_up)
    
    if "r_hat_used" in df.columns:
        r = pd.to_numeric(df["r_hat_used"], errors="coerce").astype(float)
    else:
        r = pd.to_numeric(df["payout_ratio"], errors="coerce").astype(float)
    
    loss = df["outcome"].astype(str).str.lower() == "loss"

    margin = p_side - (1.0 / np.maximum(1e-9, r))
    arr = margin[loss].to_numpy()
    arr = arr[np.isfinite(arr)]
    
    if arr.size < 10:
        return float('inf')
    
    q = float(min(max(q, 0.0), 1.0))
    return float(np.quantile(arr, q))
