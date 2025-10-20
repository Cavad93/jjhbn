# rhat_quantile2d.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import json, math, os, time
from error_logger import log_exception

def _safe_float(x, default=np.nan):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default

def _log_bin(x: float, edges: List[float]) -> int:
    if not math.isfinite(x) or x <= 0:
        return 0
    lx = math.log10(max(1e-12, x))
    i = int(np.searchsorted(edges, lx, side="right") - 1)
    return max(0, min(i, len(edges)-2))

def _lin_bin(x: float, edges: List[float]) -> int:
    i = int(np.searchsorted(edges, x, side="right") - 1)
    return max(0, min(i, len(edges)-2))

@dataclass
class RHat2D:
    state_path: str = "rhat2d_state.json"
    pending_path: str = "rhat2d_pending.json"
    t_edges: List[float] = field(default_factory=lambda: [0, 7, 15, 30, 60, 9999])  # сек до lock
    pool_log_edges: List[float] = field(default_factory=lambda: [-3, -2, -1, 0, 1, 3])  # лог10(BNB)
    max_per_bucket: int = 500

    def _load_state(self) -> Dict[str, List[float]]:
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return {k: list(map(float, v)) for k, v in obj.get("buckets", {}).items()}
        except Exception:
            return {}

    def _save_state(self, buckets: Dict[str, List[float]]) -> None:
        try:
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"buckets": buckets}, f, ensure_ascii=False)
            os.replace(tmp, self.state_path)
        except Exception:
            log_exception(f"RHat2D: failed to save state to {self.state_path}")

    def _load_pending(self) -> Dict[str, Dict[str, float]]:
        try:
            with open(self.pending_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_pending(self, pend: Dict[str, Dict[str, float]]) -> None:
        try:
            tmp = self.pending_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(pend, f, ensure_ascii=False)
            os.replace(tmp, self.pending_path)
        except Exception:
            from error_logger import log_exception
            log_exception("Failed to load JSON")

    def _key(self, i: int, j: int) -> str:
        return f"{i}:{j}"

    def observe_epoch(self, epoch: int, t_rem_s: int, pool_total_bnb: float) -> None:
        i = _lin_bin(float(t_rem_s), self.t_edges)
        j = _log_bin(float(pool_total_bnb), self.pool_log_edges)
        pend = self._load_pending()
        pend[str(int(epoch))] = {"i": int(i), "j": int(j), "ts": int(time.time())}
        self._save_pending(pend)

    def ingest_settled(self, csv_path: str) -> None:
        pend = self._load_pending()
        if not pend:
            return
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception:
            return
        if df is None or df.empty:
            return
        df = df.dropna(subset=["epoch","payout_ratio","outcome"])
        df = df[df["outcome"].isin(["win","loss","draw"])]
        buckets = self._load_state()
        for _, row in df.iterrows():
            ep = str(int(_safe_float(row.get("epoch"))))
            if ep not in pend:
                continue
            pr = _safe_float(row.get("payout_ratio"))
            if not math.isfinite(pr) or pr <= 0:
                pend.pop(ep, None)
                continue
            i = int(pend[ep]["i"])
            j = int(pend[ep]["j"])
            key = self._key(i, j)
            arr = buckets.get(key, [])
            arr.append(float(pr))
            if len(arr) > self.max_per_bucket:
                arr = arr[-self.max_per_bucket:]
            buckets[key] = arr
            pend.pop(ep, None)
        self._save_state(buckets)
        self._save_pending(pend)

    def estimate(self,
                 side: str,
                 lock_ts: Optional[int] = None,
                 ewma_lambda: float = 0.25,
                 fallback_hours: int = 24,
                 csv_path: str = "trades_prediction.csv",
                 i_hint: Optional[int] = None,
                 j_hint: Optional[int] = None) -> Optional[float]:
        buckets = self._load_state()
        key = None
        if (i_hint is not None) and (j_hint is not None):
            key = f"{int(i_hint)}:{int(j_hint)}"
        if key is not None and key in buckets and len(buckets[key]) >= 5:
            try:
                return float(np.quantile(np.array(buckets[key], dtype=float), 0.25))
            except Exception:
                log_exception("RHat2D: quantile compute failed")

        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception:
            return None
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["outcome","payout_ratio"])
        df = df[df["outcome"].isin(["win","loss","draw"])]
        side = (side or "").upper()
        if "side" in df.columns:
            df_side = df[df["side"].astype(str).str.upper() == side]
        else:
            df_side = df
        rr = pd.to_numeric(df_side.get("payout_ratio", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
        if rr.size >= 3:
            lam = float(max(0.01, min(0.99, ewma_lambda)))
            w = np.power(1.0 - lam, np.arange(rr.size-1, -1, -1))
            w /= w.sum()
            r_ewma = float(np.sum(w * rr))
        else:
            r_ewma = float(np.nan)
        if math.isfinite(r_ewma) and r_ewma > 1.0:
            return r_ewma

        if lock_ts is not None and "lock_ts" in df.columns:
            df = df.assign(h=((pd.to_numeric(df["lock_ts"], errors="coerce") // 3600) % 24))
            try:
                h = int((int(lock_ts) // 3600) % 24)
            except Exception:
                h = None
            if h is not None:
                arr = pd.to_numeric(df[df["h"] == h]["payout_ratio"], errors="coerce").dropna().to_numpy()
                if arr.size >= 5:
                    return float(np.quantile(arr, 0.25))

        arr = pd.to_numeric(df["payout_ratio"], errors="coerce").dropna().to_numpy()
        if arr.size >= 1:
            return float(np.quantile(arr, 0.25))
        return None
