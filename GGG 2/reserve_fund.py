# -*- coding: utf-8 -*-
"""
reserve_fund.py — учёт «резервного фонда» с ежедневной (23:00 UTC) фиксацией:
- Если капитал вырос за сутки: 50% прироста → в резерв (капитал уменьшается).
- Если капитал упал: компенсируем падение из резерва на доступную сумму.
- Состояние хранится в JSON (баланс резерва, капитал на последней отсечке, время отсечки).
- Расчёт выполняется «при первом тике» после 23:00 UTC (как сейчас у delta_daily).

Интеграция (в bnbusdrt6.py):
    from reserve_fund import ReserveFund
    reserve = ReserveFund(path=os.path.join(os.path.dirname(__file__), "reserve_state.json"))

    # внутри главного while True, рядом с daily-логикой:
    event = reserve.maybe_eod_rebalance(now_ts=now, capital=capital)
    if event and event.get("changed"):
        capital = float(event["capital"])
        capital_state.save(capital, ts=now)
        try:
            tg_send(event["message"])
        except Exception:
            pass

    # в сообщении статистики после каждого раунда:
    # резерв уже появится (мы добавим одну строку в build_stats_message).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json, os
from typing import Optional, Dict, Any
from datetime import datetime, date, time, timezone

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)

def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

@dataclass
class ReserveState:
    reserve: float = 0.0
    last_checkpoint_utc: Optional[str] = None
    capital_at_checkpoint: Optional[float] = None

    @staticmethod
    def load(path: str) -> "ReserveState":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ReserveState(
                reserve=float(data.get("reserve", 0.0)),
                last_checkpoint_utc=data.get("last_checkpoint_utc"),
                capital_at_checkpoint=(float(data["capital_at_checkpoint"])
                                       if data.get("capital_at_checkpoint") is not None else None),
            )
        except FileNotFoundError:
            return ReserveState()
        except Exception:
            # Повреждение файла — начинаем «с нуля», не ломаясь.
            return ReserveState()

    def save(self, path: str) -> None:
        _atomic_write_json(path, asdict(self))

class ReserveFund:
    """
    Хранит состояние и выполняет дневной «ребаланс» в 23:00 UTC.
    """
    def __init__(self, path: str = "reserve_state.json", checkpoint_hour: int = 23):
        self.path = path
        self.checkpoint_hour = int(checkpoint_hour)
        self.state = ReserveState.load(self.path)

    @staticmethod
    def _anchor_for_day(d: date, hour: int) -> datetime:
        # 23:00 UTC данного календарного дня
        return datetime(d.year, d.month, d.day, hour, 0, 0, tzinfo=timezone.utc)

    @property
    def balance(self) -> float:
        return float(self.state.reserve or 0.0)

    def read_balance_fast(self) -> float:
        # Удобный вызов для «ленивого» чтения баланса в чужом коде.
        try:
            s = ReserveState.load(self.path)
            return float(s.reserve or 0.0)
        except Exception:
            return 0.0

    def maybe_eod_rebalance(self, now_ts: int, capital: float) -> Optional[Dict[str, Any]]:
        """
        Выполнить дневную процедуру, если мы пересекли «якорь» 23:00 UTC и
        текущий день ещё не обработан.

        Возвращает:
            None — ничего не делали;
            dict с полями:
                changed: bool
                capital: float (возможно изменённый)
                reserve: float (новый баланс резерва)
                moved: float (переведено в резерв при профите)
                covered: float (компенсировано из резерва при убытке)
                message: str (краткий отчёт для TG)
                checkpoint_utc: str (ISO времени отсечки)
        """
        now = datetime.fromtimestamp(int(now_ts), tz=timezone.utc)
        today_anchor = self._anchor_for_day(now.date(), self.checkpoint_hour)

        # Если текущее время ещё НЕ достигло 23:00 UTC — ничего не делаем.
        if now < today_anchor:
            return None

        # Уже делали сегодня?
        last_iso = self.state.last_checkpoint_utc
        if last_iso:
            try:
                last_dt = datetime.fromisoformat(last_iso)
                if last_dt >= today_anchor:
                    return None  # уже обработано сегодня
            except Exception:
                # корявый формат — сделаем вид, что не было отсечки
                pass

        # Первая инициализация (нет базового capital_at_checkpoint)
        if self.state.capital_at_checkpoint is None:
            self.state.capital_at_checkpoint = float(capital)
            self.state.last_checkpoint_utc = today_anchor.isoformat()
            self.state.reserve = float(self.state.reserve or 0.0)
            self.state.save(self.path)
            return {
                "changed": False,
                "capital": float(capital),
                "reserve": float(self.state.reserve),
                "moved": 0.0,
                "covered": 0.0,
                "message": f"🏦 Резерв: первая отсечка {today_anchor.isoformat()} — базовый капитал зафиксирован.",
                "checkpoint_utc": today_anchor.isoformat(),
            }

        prev_cap = float(self.state.capital_at_checkpoint)
        cur_cap = float(capital)
        delta = cur_cap - prev_cap

        moved = 0.0
        covered = 0.0
        changed = False
        reserve_before = float(self.state.reserve or 0.0)
        lines = [f"🏦 Резерв (дневная отсечка 23:00 UTC)"]

        if delta > 0.0:
            moved = 0.5 * delta
            self.state.reserve = reserve_before + moved
            cur_cap = cur_cap - moved
            changed = True
            lines.append(f"Профит за сутки: +{delta:.6f} BNB → 50% в резерв: {moved:.6f} BNB.")
        elif delta < 0.0:
            loss = -delta
            covered = min(loss, reserve_before)
            if covered > 0.0:
                self.state.reserve = reserve_before - covered
                cur_cap = cur_cap + covered
                changed = True
                lines.append(f"Убыток за сутки: -{loss:.6f} BNB → компенсировано из резерва: {covered:.6f} BNB.")
            else:
                lines.append(f"Убыток за сутки: -{loss:.6f} BNB; резерв пуст — компенсации нет.")
        else:
            lines.append("За сутки изменения капитала отсутствуют.")

        self.state.capital_at_checkpoint = float(cur_cap)  # новая база = ПОСЛЕ перевода/компенсации
        self.state.last_checkpoint_utc = today_anchor.isoformat()
        self.state.save(self.path)

        lines.append(f"Баланс резерва: {self.state.reserve:.6f} BNB. Рабочий капитал: {cur_cap:.6f} BNB.")
        return {
            "changed": changed,
            "capital": float(cur_cap),
            "reserve": float(self.state.reserve),
            "moved": float(moved),
            "covered": float(covered),
            "message": "\n".join(lines),
            "checkpoint_utc": today_anchor.isoformat(),
        }
