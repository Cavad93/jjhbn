# ctx_calibration.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import math

def _safe_float(x, default=np.nan):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _bin_edges_quantiles(vals: np.ndarray, nbins: int) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.linspace(0.0, 1.0, nbins+1)
    qs = np.linspace(0, 1, nbins+1)
    try:
        edges = np.quantile(vals, qs)
    except Exception:
        edges = np.linspace(np.nanmin(vals), np.nanmax(vals), nbins+1)
    for i in range(1, len(edges)):
        if not np.isfinite(edges[i]) or edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-12
    return edges

def _digitize_with_edges(x: float, edges: np.ndarray) -> int:
    i = int(np.searchsorted(edges, x, side="right") - 1)
    i = max(0, min(i, len(edges)-2))
    return i

def p_ctx_calibrated(p_raw: float,
                     r_hat: float,
                     csv_path: str,
                     max_epoch_exclusive: Optional[int] = None,
                     nbins_p: int = 10,
                     nbins_r: int = 4,
                     laplace_alpha: float = 1.0) -> float:
    """
    p_ctx = E[win | bin(p_side), bin(r)]  (только история до текущего раунда)
    Laplace (α=laplace_alpha).
    """
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        return float(_clip01(p_raw))
    if df is None or df.empty:
        return float(_clip01(p_raw))

    df = df.dropna(subset=["outcome"]).copy()
    if "epoch" in df.columns and max_epoch_exclusive is not None:
        try:
            df = df[df["epoch"] < int(max_epoch_exclusive)]
        except Exception:
            pass
    if df.empty:
        return float(_clip01(p_raw))

    p_up = pd.to_numeric(df.get("p_up", pd.Series(dtype=float)), errors="coerce")
    side = df.get("side", pd.Series(dtype=str)).astype(str).str.upper()
    p_side_hist = np.where(side == "UP", p_up, 1.0 - p_up)
    p_side_hist = np.clip(p_side_hist.astype(float), 1e-9, 1.0 - 1e-9)

    r_real = pd.to_numeric(df.get("payout_ratio", pd.Series(dtype=float)), errors="coerce").astype(float)
    mask = np.isfinite(p_side_hist) & np.isfinite(r_real)
    p_side_hist = p_side_hist[mask]
    r_real = r_real[mask]
    y = (df.loc[mask, "outcome"].astype(str).str.lower() == "win").astype(int).to_numpy()

    if p_side_hist.size == 0 or y.size == 0:
        return float(_clip01(p_raw))

    p_edges = _bin_edges_quantiles(p_side_hist, nbins_p)
    r_edges = _bin_edges_quantiles(r_real, nbins_r)

    p_idx = np.searchsorted(p_edges, p_side_hist, side="right") - 1
    p_idx = np.clip(p_idx, 0, len(p_edges)-2)
    r_idx = np.searchsorted(r_edges, r_real, side="right") - 1
    r_idx = np.clip(r_idx, 0, len(r_edges)-2)

    wins = np.zeros((len(p_edges)-1, len(r_edges)-1), dtype=float)
    total = np.zeros_like(wins)
    for i, j, yi in zip(p_idx, r_idx, y):
        wins[i, j] += float(yi)
        total[i, j] += 1.0

    i = _digitize_with_edges(float(p_raw), p_edges)
    j = _digitize_with_edges(float(r_hat), r_edges)

    a = float(laplace_alpha)
    post = (wins[i, j] + a) / (total[i, j] + 2.0*a) if total[i, j] > 0 else np.nan

    if not np.isfinite(post):
        Wp = wins[i, :].sum()
        Tp = total[i, :].sum()
        post = (Wp + a) / (Tp + 2.0*a) if Tp > 0 else np.nan

    if not np.isfinite(post):
        post = float(p_raw)

    return float(_clip01(post))
