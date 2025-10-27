# futures_ctx.py
# -*- coding: utf-8 -*-
import time
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from error_logger import log_exception
except ImportError:
    def log_exception(msg: str):
        pass

BINANCE_FUT = "https://fapi.binance.com"

class FuturesContext:
    """
    Использует:
      - /fapi/v1/premiumIndex (markPrice, lastFundingRate, nextFundingTime)
      - /fapi/v1/openInterest (текущий OI)
    Кэширует историю OI для Δ за 1м/5м. Обновляй через refresh(spot_mid).
    """
    def __init__(self, session, symbol: str, min_refresh_sec: int = 30):
        self.s = session
        self.symbol = symbol.upper()
        self.min_refresh = int(min_refresh_sec)
        self.last_ts: int = 0
        self.mark_price: Optional[float] = None
        self.last_funding_rate: Optional[float] = None
        self.next_funding_time_ms: Optional[int] = None
        self.oi_hist = deque(maxlen=600)  # (ts, oi)

    def _get_premium_index(self) -> Optional[Dict]:
        try:
            r = self.s.get(f"{BINANCE_FUT}/fapi/v1/premiumIndex",
                           params={"symbol": self.symbol}, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def _get_open_interest(self) -> Optional[Tuple[int, float]]:
        try:
            r = self.s.get(f"{BINANCE_FUT}/fapi/v1/openInterest",
                           params={"symbol": self.symbol}, timeout=10)
            r.raise_for_status()
            js = r.json()
            oi = float(js["openInterest"])
            return int(time.time()), oi
        except Exception:
            return None

    def refresh(self):
        now = int(time.time())
        if now - self.last_ts < self.min_refresh:
            return
        pi = self._get_premium_index()
        if pi:
            try:
                self.mark_price = float(pi.get("markPrice", self.mark_price or 0.0))
            except Exception:
                log_exception("FuturesContext: markPrice parse failed")
            try:
                self.last_funding_rate = float(pi.get("lastFundingRate", self.last_funding_rate or 0.0))
            except Exception:
                log_exception("FuturesContext: lastFundingRate parse failed")
            try:
                self.next_funding_time_ms = int(pi.get("nextFundingTime", self.next_funding_time_ms or 0))
            except Exception:
                log_exception("FuturesContext: nextFundingTime parse failed")
        oi = self._get_open_interest()
        if oi:
            self.oi_hist.append(oi)
        self.last_ts = now

    def _delta_oi(self, secs: int) -> Optional[float]:
        if not self.oi_hist:
            return None
        now_ts, now_oi = self.oi_hist[-1]
        target = now_ts - secs
        # найдём ближайшую слева
        prev_candidates = [x for x in self.oi_hist if x[0] <= target]
        if not prev_candidates:
            return None
        prev_ts, prev_oi = prev_candidates[-1]
        if prev_oi == 0:
            return None
        return float((now_oi - prev_oi)/prev_oi)

    def features(self, spot_mid: Optional[float]) -> Dict[str, float]:
        """
        Возвращает:
          funding_sign, funding_timeleft, dOI_1m, dOI_5m, basis_now
        """
        out = dict(funding_sign=0.0, funding_timeleft=0.0, dOI_1m=0.0, dOI_5m=0.0, basis_now=0.0)
        if self.last_funding_rate is not None:
            out["funding_sign"] = 1.0 if self.last_funding_rate > 0 else (-1.0 if self.last_funding_rate < 0 else 0.0)
        if self.next_funding_time_ms:
            out["funding_timeleft"] = max(0.0, (self.next_funding_time_ms/1000.0) - time.time())
        d1 = self._delta_oi(60)
        d5 = self._delta_oi(300)
        out["dOI_1m"] = float(d1 or 0.0)
        out["dOI_5m"] = float(d5 or 0.0)
        if self.mark_price is not None and spot_mid and spot_mid > 0:
            out["basis_now"] = float((self.mark_price - spot_mid)/spot_mid)
        return out
