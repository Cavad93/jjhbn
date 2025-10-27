# microstructure.py
# -*- coding: utf-8 -*-
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

BINANCE_SPOT = "https://api.binance.com"

@dataclass
class DepthSnapshot:
    ts_ms: int
    bid1: float
    ask1: float
    bids: List[Tuple[float, float]]  # (price, vol)
    asks: List[Tuple[float, float]]

class MicrostructureClient:
    """
    Работает только на публичных Spot REST.
    - depth(limit<=5) для L1/L5
    - aggTrades(startTime,endTime) для OFI в окнах 5/15/30с
    """
    def __init__(self, session, symbol: str):
        self.s = session
        self.symbol = symbol.upper()
        self.prev_microprice: Optional[float] = None

    def fetch_depth(self, limit: int = 5, ts_ms: Optional[int] = None) -> Optional[DepthSnapshot]:
        try:
            r = self.s.get(f"{BINANCE_SPOT}/api/v3/depth",
                           params={"symbol": self.symbol, "limit": int(limit)},
                           timeout=10)
            r.raise_for_status()
            js = r.json()
            bids = [(float(p), float(q)) for p, q in js.get("bids", [])[:limit]]
            asks = [(float(p), float(q)) for p, q in js.get("asks", [])[:limit]]
            if not bids or not asks:
                return None
            bid1, ask1 = bids[0][0], asks[0][0]
            return DepthSnapshot(
                ts_ms=ts_ms if ts_ms is not None else int(time.time()*1000),
                bid1=bid1, ask1=ask1,
                bids=bids, asks=asks
            )
        except Exception:
            return None

    @staticmethod
    def _rel_spread(bid1: float, ask1: float) -> float:
        mid = 0.5*(bid1+ask1)
        return float((ask1 - bid1)/max(1e-12, mid))

    @staticmethod
    def _book_imbalance(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> float:
        sb = sum(v for _, v in bids)
        sa = sum(v for _, v in asks)
        if sb+sa <= 0:
            return 0.0
        return float((sb - sa)/(sb + sa))

    @staticmethod
    def _microprice(bid1: float, ask1: float, bidv1: float, askv1: float) -> float:
        denom = max(1e-12, bidv1 + askv1)
        return float((ask1*bidv1 + bid1*askv1)/denom)

    @staticmethod
    def _ob_slope(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> float:
        # приблизительный наклон кривой кумулятивной ликвидности по обеим сторонам
        def side_slope(levels: List[Tuple[float, float]], best: float, is_bid: bool) -> float:
            if not levels:
                return 0.0
            mid_like = best  # локальная нормализация
            cumv, dists = [], []
            run = 0.0
            for p, v in levels:
                run += max(0.0, v)
                cumv.append(run)
                dist = (best - p)/max(1e-12, mid_like) if is_bid else (p - best)/max(1e-12, mid_like)
                dists.append(max(1e-9, dist))
            x = np.array(dists, dtype=float)
            y = np.array(cumv, dtype=float)
            if x.size < 2:
                return 0.0
            # slope = cov(x,y)/var(x)
            vx = float(np.var(x))
            if vx <= 0:
                return 0.0
            cov = float(np.mean((x - x.mean())*(y - y.mean())))
            return float(cov/max(1e-12, vx))
        bid1 = bids[0][0] if bids else 0.0
        ask1 = asks[0][0] if asks else 0.0
        s_b = side_slope(bids, bid1, True)
        s_a = side_slope(asks, ask1, False)
        return 0.5*(s_b + s_a)

    def _ofi_in_window(self, end_ts_ms: int, window_sec: int) -> Optional[float]:
        try:
            start_ms = end_ts_ms - window_sec*1000
            r = self.s.get(f"{BINANCE_SPOT}/api/v3/aggTrades",
                           params={"symbol": self.symbol, "startTime": start_ms, "endTime": end_ts_ms, "limit": 1000},
                           timeout=10)
            r.raise_for_status()
            trades = r.json()  # [{p, q, T, m, ...}]
            if not trades:
                return 0.0
            s = 0.0
            vol = 0.0
            for t in trades:
                if "q" not in t or "m" not in t:
                    continue
                qty = float(t["q"])
                is_buyer_maker = bool(t["m"])
                # m=True => buyer is maker => агрессивный продавец => знак -1
                sign = -1.0 if is_buyer_maker else 1.0
                s += sign*qty
                vol += qty
            if vol <= 0:
                return 0.0
            return float(s/vol)
        except Exception:
            return None

    def compute(self, end_ts_ms: int) -> Dict[str, float]:
        """
        Возвращает словарь:
          rel_spread, book_imb, microprice_delta, ofi_5s, ofi_15s, ofi_30s, ob_slope, mid
        """
        out = dict(rel_spread=0.0, book_imb=0.0, microprice_delta=0.0,
                   ofi_5s=0.0, ofi_15s=0.0, ofi_30s=0.0, ob_slope=0.0, mid=0.0)
        dp = self.fetch_depth(limit=5, ts_ms=end_ts_ms)
        if dp is None:
            return out
        mid = 0.5*(dp.bid1 + dp.ask1)
        out["mid"] = float(mid)
        out["rel_spread"] = self._rel_spread(dp.bid1, dp.ask1)
        book_imb_val = self._book_imbalance(dp.bids, dp.asks)
        out["book_imbalance"] = book_imb_val
        out["book_imb"] = book_imb_val
        bidv1 = dp.bids[0][1] if dp.bids else 0.0
        askv1 = dp.asks[0][1] if dp.asks else 0.0
        mp = self._microprice(dp.bid1, dp.ask1, bidv1, askv1)
        if self.prev_microprice is not None:
            out["microprice_delta"] = float((mp - self.prev_microprice)/max(1e-12, mid))
        self.prev_microprice = mp
        out["ob_slope"] = self._ob_slope(dp.bids, dp.asks)

        ofi_5 = self._ofi_in_window(end_ts_ms, 5)
        ofi_15 = self._ofi_in_window(end_ts_ms, 15)
        ofi_30 = self._ofi_in_window(end_ts_ms, 30)
        out["ofi_5s"] = float(ofi_5 or 0.0)
        out["ofi_15s"] = float(ofi_15 or 0.0)
        out["ofi_30s"] = float(ofi_30 or 0.0)
        return out
