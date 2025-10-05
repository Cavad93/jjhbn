# pool_features.py
# -*- coding: utf-8 -*-
import math
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

import numpy as np

def _to_logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p/(1.0-p))

class PoolFeaturesCtx:
    """
    Копит снимки пулов bull/bear для активного epoch:
      - pool_logit = logit(bull/(bull+bear))
      - Δpool_logit за 30/60с
      - late_money_share за последние 12с до lock
      - last_k_outcomes_mean: среднее по последним k исходам (UP=1, DOWN=0)
      - last_k_payout_median: медианный payout по последним k
    """
    def __init__(self, k: int = 10, late_sec: int = 30):
        self.obs: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)  # epoch -> list[(ts,bull,bear)]
        self.k = int(k)
        self.late = int(late_sec)
        self.outcomes = deque(maxlen=max(5, k))  # 1=UP win, 0=DOWN win (ничья игнор)
        self.payouts = deque(maxlen=max(5, k))
        self.late_deltas = deque(maxlen=120)  # скользящее окно притоков за последние late секунд


    def observe(self, epoch: int, ts: int, bull_amount: float, bear_amount: float):
        self.obs[epoch].append((ts, float(bull_amount), float(bear_amount)))

    def update_streak_from_rounds(self, get_round_fn, cur_epoch: int):
        """
        get_round_fn(epoch)->round with fields (oracle_called, lock_price, close_price, reward_base, reward_amt)
        """
        # соберём 2*k исторических завершённых раундов (с запасом)
        for e in range(cur_epoch-1, max(1, cur_epoch-2*self.k)-1, -1):
            try:
                rd = get_round_fn(e)
                if not rd.oracle_called:
                    continue
                if rd.lock_price == rd.close_price:
                    continue
                up_won = rd.close_price > rd.lock_price
                self.outcomes.append(1 if up_won else 0)
                ratio = (rd.reward_amt/rd.reward_base) if (rd.reward_base and rd.reward_base>0) else None
                if ratio:
                    self.payouts.append(float(ratio))
                if len(self.outcomes) >= self.k:
                    break
            except Exception:
                continue

    def _last_obs_before(self, epoch: int, ts_cut: int) -> Optional[Tuple[int,float,float]]:
        arr = self.obs.get(epoch, [])
        if not arr:
            return None
        # ищем последнюю запись со временем <= ts_cut
        cand = [x for x in arr if x[0] <= ts_cut]
        return cand[-1] if cand else None

    def features(self, epoch: int, lock_ts: int) -> Dict[str, float]:
        out = dict(pool_logit=0.0, pool_logit_d30=0.0, pool_logit_d60=0.0,
                   late_money_share=0.0, last_k_outcomes_mean=0.0, last_k_payout_median=0.0)
        arr = self.obs.get(epoch, [])
        if arr:
            # текущее значение к lock-1s
            cur = self._last_obs_before(epoch, lock_ts-1) or arr[-1]
            _, bull, bear = cur
            tot = max(1e-12, bull + bear)
            p_bull = float(bull/tot)
            out["pool_logit"] = _to_logit(p_bull)

            # Δ за 30/60с
            for W, key in [(30, "pool_logit_d30"), (60, "pool_logit_d60")]:
                prev = self._last_obs_before(epoch, lock_ts-1-W)
                if prev:
                    _, b2, a2 = prev
                    tot2 = max(1e-12, b2+a2)
                    p2 = b2/tot2
                    out[key] = float(out["pool_logit"] - _to_logit(p2))

            # поздние вливания: доля средств, зашедших за последние late секунд
            prev_late = self._last_obs_before(epoch, lock_ts-1-self.late)
            if prev_late:
                _, b0, a0 = prev_late
                add = (bull + bear) - (b0 + a0)
                out["late_money_share"] = float(max(0.0, add)/max(1e-12, bull+bear))

        if self.outcomes:
            out["last_k_outcomes_mean"] = float(sum(self.outcomes)/len(self.outcomes))
        if self.payouts:
            out["last_k_payout_median"] = float(np.median(self.payouts))
        return out

    def finalize_epoch(self, epoch: int, lock_ts: int) -> None:
        """
        Вызывается один раз, когда раунд переходит в locked/settled.
        Берём последнюю запись ≤ lock_ts и запись ≤ lock_ts - late, считаем приток и кладём в историю.
        """
        try:
            last = self._last_obs_before(epoch, lock_ts) or (self.obs.get(epoch, [])[-1] if self.obs.get(epoch) else None)
            prev = self._last_obs_before(epoch, lock_ts - 1 - self.late)

            # мягкий фолбэк: если prev не нашли (редкие лаги/редкие тики), берём ближайший ≤ lock_ts-1
            if (prev is None) and self.obs.get(epoch):
                prev = self._last_obs_before(epoch, lock_ts - 1)

            if not last or not prev:
                return

            _, b_last, a_last = last
            _, b_prev, a_prev = prev
            delta = (b_last + a_last) - (b_prev + a_prev)

            # не выбрасываем информацию окончательно — клиппуем отрицательное к 0,
            # чтобы история обновлялась (и «медиана» честно показывала отсутствие позднего притока)
            if math.isfinite(delta):
                self.late_deltas.append(float(max(0.0, delta)))
        except Exception:
            return

    def late_delta_quantile(self, q: float = 0.5) -> float:
        """Квантиль (по умолчанию медиана) притока за последние `late` секунд по истории."""
        if not self.late_deltas:
            return 0.0
        q = min(max(q, 0.0), 1.0)
        return float(np.quantile(np.array(self.late_deltas, dtype=float), q))

