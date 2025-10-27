
# -*- coding: utf-8 -*-
"""
wr_pnl_tracker.py — устойчивый трекер WR/PNL + анти-повторная логика "отдыха".
Автор: ChatGPT

Ключевые идеи:
- Всегда работаем с UTC-временем. "Сутки/7д/30д" = скользящие окна по UTC, а не "календарный день".
- Читаем trades_prediction.csv, ДЕДУПЛИКИРУЕМ по столбцу epoch, берём только строчки с зафиксированным исходом.
- WR_Δ24h считаем как разницу между двумя НЕПЕРЕКРЫВАЮЩИМИСЯ окнами: (48..24ч назад) vs (последние 24ч).
- Состояние отдыха (REST4H/REST24H/ACTIVE) и время rest_until_utc сохраняем в JSON.
- Не допускаем повторного входа в REST4H, пока не случится полноценный выход в ACTIVE и не будет выполнен минимум сделок/времени.
- Защита от "штормов" после перезагрузки ПК: при старте перечитываем CSV и корректно восстанавливаем состояние.

Интеграция в ваш код:
1) from wr_pnl_tracker import StatsTracker, RestState, RestConfig
2) Инициализация (один раз при запуске):
       stats = StatsTracker(csv_path="GGG/trades_prediction.csv")
       rest = RestState.load(path="GGG/rest_state.json")
3) Перед каждой ставкой проверять:
       if not rest.can_trade_now():
           log("REST until", rest.rest_until_utc)
           return  # пропускаем ставку
4) После КАЖДОГО закрытия сделки (после записи в CSV):
       stats.reload()
       rest.update_from_stats(stats)
       rest.save()

По желанию параметризуйте пороги в RestConfig(...).
"""
from __future__ import annotations
import json, os
from dataclasses import dataclass, asdict
from typing import Optional, Literal
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

StateName = Literal["ACTIVE", "REST4H", "REST24H"]

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

@dataclass
class RestConfig:
    min_interval_between_rests: timedelta = timedelta(hours=24)
    last_rest_start_utc: Optional[str] = None  # ISO8601, когда мы вошли в REST в последний раз
   # Пороги падения винрейта (в ПРОЦЕНТНЫХ ПУНКТАХ, например 0.10 = -10 п.п.)
    drop_for_rest4h: float = 0.10
    drop_for_rest24h: float = 0.15
    # Длительности:
    dur_rest4h: timedelta = timedelta(hours=4)
    dur_rest24h: timedelta = timedelta(hours=24)
    # Минимальный размер выборки в каждом 24h-окне, чтобы триггерить отдых
    min_trades_per_window: int = 40
    # Гистерезис: после REST4H требуется выполнить хотя бы N сделок,
    # прежде чем снова рассматривать вход в любой REST
    min_trades_after_rest4h: int = 10
    # Либо минимум времени, если сделок мало
    min_time_after_rest4h: timedelta = timedelta(hours=2)
    # Использовать "двухоконную" оценку WR_Δ24h (48..24h vs 24..0h)
    use_two_window_drop: bool = True
    # 🚫 Иммунитет от отдыха до достижения общего числа завершённых сделок
    # (после дедупа по epoch и только с outcome)
    min_total_trades_for_rest: int = 500

@dataclass
class RestState:
    state: StateName = "ACTIVE"
    rest_until_utc: Optional[str] = None            # ISO8601 в UTC
    last_exit_to_active_utc: Optional[str] = None   # ISO8601 в UTC
    last_rest_start_utc: Optional[str] = None       # ← ДОБАВЛЕНО
    trades_since_last_rest4h_exit: int = 0
    decided_at_utc: Optional[str] = None            # служебное: когда принимали решение

    @staticmethod
    def load(path: str) -> "RestState":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return RestState(**data)
        except FileNotFoundError:
            return RestState()
        except Exception as e:
            # Повреждённый файл? Стартуем в ACTIVE, но не триггерим мгновенно отдых.
            return RestState()

    def save(self, path: str = "rest_state.json") -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _parse_dt(self, s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        return datetime.fromisoformat(s)

    def _set_until(self, dt_until: datetime) -> None:
        self.rest_until_utc = dt_until.isoformat()
        now_iso = utcnow().isoformat()
        self.decided_at_utc      = now_iso
        self.last_rest_start_utc = now_iso


    def can_trade_now(self) -> bool:
        """TRUE если можно торговать (мы не в REST или время вышло)."""
        if self.state == "ACTIVE":
            return True
        until = self._parse_dt(self.rest_until_utc)
        if not until:
            # безопасно: если непонятно, не торгуем только текущий тик
            return False
        if utcnow() >= until:
            # Выходим из отдыха.
            self.state = "ACTIVE"
            self.last_exit_to_active_utc = utcnow().isoformat()
            self.trades_since_last_rest4h_exit = 0
            return True
        return False

    def notify_trade_executed(self) -> None:
        """Вызывайте КАЖДЫЙ раз, когда сделка реально закрылась (после записи в CSV)."""
        if self.state == "ACTIVE":
            if self.last_exit_to_active_utc:
                self.trades_since_last_rest4h_exit += 1

    def update_from_stats(self, stats: "StatsTracker", cfg: RestConfig = RestConfig()) -> None:
        """Обновить состояние на основе свежей статистики."""
        now = utcnow()

        # 🚫 Стартовый иммунитет: до N_total завершённых сделок — НИКАКОГО отдыха.
        # Берём число строк после дедупа и только с outcome.
        try:
            n_total = int(getattr(stats, "df", None).shape[0])  # type: ignore[attr-defined]
        except Exception:
            n_total = 0

        if n_total < getattr(cfg, "min_total_trades_for_rest", 0):
            # Если вдруг уже в REST — принудительно выходим в ACTIVE
            if self.state in ("REST4H", "REST24H"):
                self.state = "ACTIVE"
                self.rest_until_utc = None
                self.last_exit_to_active_utc = now.isoformat()
                self.trades_since_last_rest4h_exit = 0
            self.decided_at_utc = now.isoformat()
            return

        # 🚫 Частота: не чаще 1 раза за min_interval_between_rests
        if self.state == "ACTIVE":
            last_start = self._parse_dt(self.last_rest_start_utc)
            if last_start and (utcnow() - last_start) < cfg.min_interval_between_rests:
                # Рано снова входить в REST — ждём окончания суточного интервала
                return


        # Если сейчас отдых и время не вышло — ничего не решаем повторно.
        if self.state in ("REST4H", "REST24H"):
            until = self._parse_dt(self.rest_until_utc)
            if until and now < until:
                return
            else:
                pass


        # Гистерезис после REST4H: не даём тут же снова влететь в отдых
        if self.last_exit_to_active_utc:
            elapsed = now - datetime.fromisoformat(self.last_exit_to_active_utc)
            if (self.trades_since_last_rest4h_exit < cfg.min_trades_after_rest4h) and (elapsed < cfg.min_time_after_rest4h):
                return  # рано судить, дайте модели "вдохнуть"

        drop, n_old, n_new = (
            stats.wr_drop_24h_two_windows() 
            if getattr(cfg, "use_two_window_drop", True) 
            else stats.wr_drop_24h_single_baseline()
        )


        # Не триггерим, если выборки слишком малы
        if (n_old < cfg.min_trades_per_window) or (n_new < cfg.min_trades_per_window) or (drop is None):
            return

        # Эскалации/решения
        if drop >= cfg.drop_for_rest24h:
            self.state = "REST24H"
            self._set_until(now + cfg.dur_rest24h)
        elif drop >= cfg.drop_for_rest4h:
            # В REST4H входим только если мы действительно в ACTIVE сейчас
            if self.state == "ACTIVE":
                self.state = "REST4H"
                self._set_until(now + cfg.dur_rest4h)
        # Иначе — остаёмся ACTIVE

class StatsTracker:
    def __init__(self, csv_path: str = "GGG/trades_prediction.csv"):
        self.csv_path = csv_path
        self.df = None
        self.reload()

    def reload(self) -> None:
        df = pd.read_csv(self.csv_path)
        # Требуем обязательные поля
        required = ["settled_ts", "epoch", "outcome", "pnl"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f"В CSV нет обязательных колонок: {missing}")
        # UTC-время в секундах
        df["_ts"] = pd.to_datetime(df["settled_ts"], unit="s", utc=True, errors="coerce")
        # Только завершённые сделки (outcome заполнен)
        mask_done = df["outcome"].astype(str).str.len() > 0
        df = df.loc[mask_done].copy()
        # Дедуп по epoch (на случай повторной записи)
        df.sort_values(by=["epoch", "_ts"], inplace=True)
        df = df.drop_duplicates(subset=["epoch"], keep="last")
        # Удобные поля
        df["_win"] = df["outcome"].astype(str).str.lower().isin(["win", "won", "true", "1"])
        df["_pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
        self.df = df

    def _window(self, end_utc: Optional[datetime], days: int):
        if end_utc is None:
            end_utc = utcnow()
        start_utc = end_utc - timedelta(days=days)
        sub = self.df[(self.df["_ts"] >= start_utc) & (self.df["_ts"] < end_utc)]
        n = int(len(sub))
        wr = float(sub["_win"].mean()) if n else None
        pnl = float(sub["_pnl"].sum()) if n else None
        return n, wr, pnl

    def stats_24h(self):
        return self._window(utcnow(), 1)

    def stats_7d(self):
        return self._window(utcnow(), 7)

    def stats_30d(self):
        return self._window(utcnow(), 30)

    def wr_drop_24h_two_windows(self):
        """Вернёт (drop, n_old, n_new), где drop = WR(48..24h) - WR(24..0h)."""
        end = utcnow()
        n_new, wr_new, _ = self._window(end, 1)  # последние 24h
        n_old, wr_old, _ = self._window(end - timedelta(days=1), 1)  # 48..24h
        if (wr_old is None) or (wr_new is None):
            return None, n_old, n_new
        return float(wr_old - wr_new), n_old, n_new

    def wr_drop_24h_single_baseline(self):
        """Альтернатива: сравнить текущие 24h с глобальным скользящим baseline за 7д."""
        n_new, wr_new, _ = self.stats_24h()
        n_base, wr_base, _ = self.stats_7d()
        if (wr_new is None) or (wr_base is None):
            return None, n_base, n_new
        return float(wr_base - wr_new), n_base, n_new
