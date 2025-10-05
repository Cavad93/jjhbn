# -*- coding: utf-8 -*-
"""
performance_metrics.py — монитор реального брейк-ивена p_BE и rolling log-growth.

Что делает:
- На каждом сеттле принимает запись сделки (тот же dict, что вы пишете в CSV).
- Поддерживает скользящее окно последних N сделок (по умолчанию 500).
- Считает:
    WR_rolling, W̄ (средний выигрыш), L̄ (средний проигрыш, как положительное),
    PF (profit factor), p_BE, g_rolling (средний лог-рост на сделку).
- Раз в час (по UTC) отправляет короткий отчёт в Telegram с диагнозом:
    EV-положительно/отрицательно, лог-рост положительный/отрицательный, динамика vs прошлый час.
- Хранит состояние в JSON (последний час, последние метрики) — устойчив к рестартам.
"""

from __future__ import annotations
import os, json, math, time
from dataclasses import dataclass, asdict
from typing import Optional, Deque, Dict, Any
from collections import deque


# t-критические значения для 97.5% (двусторонний 95% ДИ)
_TCRIT_975 = {
    1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447, 7:2.365, 8:2.306, 9:2.262, 10:2.228,
    11:2.201, 12:2.179, 13:2.160, 14:2.145, 15:2.131, 16:2.120, 17:2.110, 18:2.101, 19:2.093, 20:2.086,
    21:2.080, 22:2.074, 23:2.069, 24:2.064, 25:2.060, 26:2.056, 27:2.052, 28:2.048, 29:2.045, 30:2.042,
}

def _tcrit_975(df: int) -> float:
    if df <= 0:
        return float('nan')
    if df in _TCRIT_975:
        return _TCRIT_975[df]
    # нормальное приближение для df>30
    return 1.96


# ------------- утилиты -------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)

def _clip01(p: float) -> float:
    return float(min(max(p, 1e-9), 1.0 - 1e-9))

def _fmt(v: Optional[float], q: int = 4, pct: bool = False) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v*100:.2f}%"
    return f"{v:.{q}f}"

# ------------- состояние -------------

@dataclass
class PerfState:
    last_hour_bucket: Optional[int] = None  # int(hour UTC) как floor(ts/3600)
    last_report: Optional[Dict[str, float]] = None  # метрики прошлого отчёта

    def to_json(self) -> Dict[str, Any]:
        return {"last_hour_bucket": self.last_hour_bucket, "last_report": self.last_report or {}}

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "PerfState":
        st = PerfState()
        st.last_hour_bucket = obj.get("last_hour_bucket")
        lr = obj.get("last_report") or {}
        if isinstance(lr, dict):
            st.last_report = {k: float(v) for k, v in lr.items() if isinstance(v, (int, float))}
        return st

# ------------- монитор -------------

class PerfMonitor:
    """
    fees_net=True: считаем, что pnl в строке уже NET (комиссии/газ учтены),
    тогда p_BE = L̄ / (W̄ + L̄). Если выставить fees_net=False, учтём среднюю комиссию/газ.
    """
    def __init__(self,
                 path: str = "perf_state.json",
                 window_trades: int = 500,
                 min_trades_for_report: int = 50,
                 fees_net: bool = True):
        self.path = path
        self.window = int(window_trades)
        self.min_trades_for_report = int(min_trades_for_report)
        self.fees_net = bool(fees_net)

        self.pnls: Deque[float] = deque(maxlen=self.window)    # pnl по сделке (в тех же ед., что capital)
        self.gas:  Deque[float] = deque(maxlen=self.window)    # gas_bet+gas_claim (если есть в строке)
        self.cap_before: Deque[float] = deque(maxlen=self.window)  # capital_before (если есть)

        self.state = self._load_state()
        self._seen: Deque[str] = deque(maxlen=5000)  # ключи обработанных сделок

    # ---------- persist ----------
    def _load_state(self) -> PerfState:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return PerfState.from_json(json.load(f))
        except Exception:
            return PerfState()

    def _row_key(self, row: Dict[str, Any]) -> str:
        # попытки взять стабильный идентификатор сделки
        tid = row.get("trade_id") or row.get("id")
        if tid:
            return f"id:{tid}"
        ep  = row.get("epoch")
        st  = row.get("settled_ts") or row.get("ts") or ""
        sd  = row.get("side") or ""
        return f"epoch:{ep}|ts:{st}|side:{sd}"


    def _save_state(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state.to_json(), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    # ---------- приём сделки ----------
    def on_trade_settled(self, row: Dict[str, Any]) -> None:
        """
        row — та же запись, что вы пишете в CSV (или очень похожая).
        Пытаемся извлечь:
        - 'pnl' или 'pnl_net' → pnl (float)
        - 'gas_bet_bnb', 'gas_claim_bnb' → gas (float, в тех же ед., что pnl)
        - 'capital_before' и/или 'capital_after' → для лог-роста
        """
        # --- дедупликация одной и той же сделки ---
        try:
            key = self._row_key(row)  # например: id:..., либо epoch:...|ts:...|side:...
        except Exception:
            key = None
        if key is not None and key in self._seen:
            return  # уже учитывали эту сделку
        if key is not None:
            self._seen.append(key)

        # pnl (стараемся взять net)
        pnl = None
        for k in ("pnl_net", "pnl"):
            if k in row:
                pnl = _safe_float(row.get(k))
                break
        if pnl is None:
            # попробуем из outcome/ratio/stake, но это редко нужно
            pnl = _safe_float(row.get("pnl"), 0.0)

        # gas
        gas_bet   = _safe_float(row.get("gas_bet_bnb"), 0.0)
        gas_claim = _safe_float(row.get("gas_claim_bnb"), 0.0)
        gas_sum = gas_bet + gas_claim

        # capital_before / after
        cap_before = None
        for k in ("capital_before", "cap_before", "capital_prev"):
            if k in row:
                cap_before = _safe_float(row.get(k))
                break

        cap_after = None
        for k in ("capital_after", "cap_after", "capital"):
            if k in row:
                cap_after = _safe_float(row.get(k))
                break

        # накапливаем
        self.pnls.append(float(pnl))
        self.gas.append(float(gas_sum))
        self.cap_before.append(_safe_float(cap_before, 0.0))

        # сохранять состояние здесь не обязательно, делаем при репорте


    # ---------- расчёт метрик ----------
    def _metrics(self) -> Dict[str, float]:
        n = len(self.pnls)
        if n == 0:
            return {}

        wins = [p for p in self.pnls if p > 0]
        losses = [-p for p in self.pnls if p < 0]  # положительные величины
        w_cnt = len(wins)
        l_cnt = len(losses)

        wr = w_cnt / n if n > 0 else 0.0
        Wbar = (sum(wins) / w_cnt) if w_cnt > 0 else 0.0
        Lbar = (sum(losses) / l_cnt) if l_cnt > 0 else 0.0
        pf = (sum(wins) / max(1e-12, sum(losses))) if l_cnt > 0 else float("inf")

        # средняя комиссия/газ (если fees_net=False учтём её в p_BE)
        c_avg = (sum(self.gas) / n) if (not self.fees_net and n > 0) else 0.0

        # p_BE (защита от деления на 0)
        denom = Wbar + Lbar
        p_be = (Lbar + c_avg) / denom if denom > 1e-12 else 1.0

        # rolling log-growth
        # лучший способ — через capital_before/after, но если нет, используем pnl/ capital_before
        # лог-рост на сделку и 95% ДИ среднего
        g_vals: list[float] = []
        for pnl_i, cb in zip(self.pnls, self.cap_before):
            if cb and cb > 0:
                gi = math.log(max(1e-12, 1.0 + (pnl_i / cb)))
                g_vals.append(gi)

        if g_vals:
            g_rolling = sum(g_vals) / len(g_vals)
            if len(g_vals) > 1:
                # несмещённая оценка дисперсии и SE
                mean = g_rolling
                var = sum((x - mean) ** 2 for x in g_vals) / (len(g_vals) - 1)
                se = (var ** 0.5) / (len(g_vals) ** 0.5)
                tcrit = _tcrit_975(len(g_vals) - 1)
                g_ci_low = mean - tcrit * se
                g_ci_high = mean + tcrit * se
            else:
                g_ci_low = float('nan')
                g_ci_high = float('nan')
        else:
            g_rolling = 0.0
            g_ci_low = float('nan')
            g_ci_high = float('nan')

        return {
            "n": float(n),
            "wr": float(wr),
            "Wbar": float(Wbar),
            "Lbar": float(Lbar),
            "pf": float(pf),
            "c_avg": float(c_avg),
            "p_be": float(p_be),
            "g_rolling": float(g_rolling),
            "g_ci_low": float(g_ci_low),
            "g_ci_high": float(g_ci_high),
        }


    # ---------- почасовой отчёт ----------
    def maybe_hourly_report(self, now_ts: int, tg_send_fn) -> Optional[Dict[str, float]]:
        """
        Раз в час (UTC) шлём отчёт. Возвращаем метрики, если отправили.
        tg_send_fn(text: str, html: bool=True) — ваша функция отправки в Telegram.
        """
        hour_bucket = int(now_ts // 3600)
        if self.state.last_hour_bucket is not None and self.state.last_hour_bucket >= hour_bucket:
            return None  # этот час уже отчитывались

        m = self._metrics()
        if not m or m.get("n", 0.0) < float(self.min_trades_for_report):
            # мало наблюдений — просто отметим час, но отчёт не шлём
            self.state.last_hour_bucket = hour_bucket
            self._save_state()
            return None

        # диагнозы
        # диагнозы
        ev_pos = (m["wr"] > m["p_be"])
        g_pos  = (m["g_rolling"] > 0.0)
        g95_pos = (("g_ci_low" in m) and (m["g_ci_low"] is not None) and (m["g_ci_low"] > 0.0))

        # динамика vs прошлый отчёт
        prev = self.state.last_report or {}
        def _delta(k: str) -> Optional[float]:
            return (m[k] - prev[k]) if (k in m and k in prev) else None

        d_wr   = _delta("wr")
        d_pbe  = _delta("p_be")
        d_g    = _delta("g_rolling")
        d_pf   = _delta("pf")

        # текст
        lines = []
        lines.append("📊 <b>Часовой отчёт (EV & Log-growth)</b>")
        lines.append(f"N={int(m['n'])} | WR={_fmt(m['wr'], pct=True)} | PF={_fmt(m['pf'], 3)}")
        lines.append(f"W̄={_fmt(m['Wbar'], 6)} | L̄={_fmt(m['Lbar'], 6)} | p_BE={_fmt(m['p_be'], pct=True)}")
        if not self.fees_net:
            lines.append(f"Fees(avg)={_fmt(m['c_avg'], 6)} (в ед. PnL)")
        # E[log(1+Δ)] и 95% ДИ
        if ("g_ci_low" in m) and ("g_ci_high" in m):
            lines.append(f"ḡ={_fmt(m['g_rolling'], 6)} | CI95=[{_fmt(m['g_ci_low'], 6)}, {_fmt(m['g_ci_high'], 6)}]")
        else:
            lines.append(f"ḡ={_fmt(m['g_rolling'], 6)}")

        # дельты (если есть прошлый отчёт)
        dparts = []
        if d_wr is not None:  dparts.append(f"ΔWR={_fmt(d_wr, 4, pct=True)}")
        if d_pbe is not None: dparts.append(f"Δp_BE={_fmt(d_pbe, 4, pct=True)}")
        if d_g is not None:   dparts.append(f"Δg={_fmt(d_g, 6)}")
        if d_pf is not None:  dparts.append(f"ΔPF={_fmt(d_pf, 3)}")
        if dparts:
            lines.append(" | ".join(dparts))

        # вердикты
        if ev_pos:
            lines.append("✅ <b>EV положительное</b>: фактический WR выше брейк-ивена.")
        else:
            lines.append("❌ <b>EV отрицательное</b>: WR ≤ p_BE — вероятность «сжигания» на комиссиях/раскладе.")

        # итог по геометрическому росту с учётом 95% ДИ
        if g95_pos:
            lines.append("✅ <b>E[log(1+Δ)]>0 с 95% дов.</b> Геометрический рост статистически доказан.")
        elif g_pos:
            lines.append("⚠️ <b>Геометрический рост положительный</b>, но не доказан на 95% (CI пересекает 0).")
        else:
            lines.append("📉 <b>Геометрический рост отрицательный</b>: капитал в среднем сжимается.")

        # мягкие рекомендации
        tips = []
        if not ev_pos:
            tips.append("→ Повышайте порог входа (p_thr/δ) или улучшайте калибровку вероятностей.")
        if not g_pos:
            tips.append("→ Проверьте сайзинг (Kelly/вол-таргетинг), фильтры chop/news, издержки (gas/fee).")
        if tips:
            lines.append("\n".join(tips))

        text = "\n".join(lines)

        # отправка в TG
        try:
            tg_send_fn(text, html=True)
        except Exception:
            pass

        # обновим состояние
        self.state.last_hour_bucket = hour_bucket
        self.state.last_report = {
            "n": m["n"], "wr": m["wr"], "Wbar": m["Wbar"], "Lbar": m["Lbar"],
            "pf": m["pf"], "c_avg": m["c_avg"], "p_be": m["p_be"], "g_rolling": m["g_rolling"],
            "g_ci_low": m.get("g_ci_low"), "g_ci_high": m.get("g_ci_high"),
        }
        self._save_state()
        return m
