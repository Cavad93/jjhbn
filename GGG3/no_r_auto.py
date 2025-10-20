# no_r_auto.py
# -*- coding: utf-8 -*-
"""
Авто-выбор EV-free режима на основе «прокс-прибыли» за последний час.
Не использует payout r — только исход (win/loss) и вероятность p_side,
а также те же пороги, что вы применяете онлайном (из gating_no_r.py).

Идея метрики (EV-free):
    Для каждого раунда и каждого режима считаем margin = p_side - (p_thr+δ_now).
    Если margin <= 0 → "не ставим".
    Если margin > 0 → "ставим" прокс-ставку f = clip(k * margin, f_min, f_max).
    Прокс-прибыль g = (+1 при win, −1 при loss) * f.
Суммируем g по последнему часу для каждого режима и берём лучший.

Параметры через .env:
    NO_R_AUTO=1                   — вкл/выкл авто-режим
    NO_R_AUTO_MINN=40             — минимум наблюдений (финализированных раундов) за час
    NO_R_AUTO_COOLDOWN=900        — секунд, минимальный интервал между переключениями
    NO_R_AUTO_HYST=0.0005         — абсолютный порог преимущества по g (гистерезис)
    NO_R_AUTO_K=1.0               — множитель для f = k * margin
    NO_R_AUTO_FMIN=0.0            — нижняя граница прокс-ставки
    NO_R_AUTO_FMAX=0.005          — верхняя граница прокс-ставки (0.5% капитала эквивалент)
    NO_R_AUTO_STATE_PATH=no_r_auto_state.json — файл состояния (последний выбор)
"""

from __future__ import annotations
import os, json, math, time, csv
from typing import Dict, Tuple, List, Optional

from gating_no_r import compute_p_thr_no_r

MODES = ("prob_lcb", "conformal", "disagree")

def _utcnow() -> int:
    return int(time.time())

def _safe_int(x, default: Optional[int]=None) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

def _safe_float(x, default: Optional[float]=None) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _read_last_hour_rows(csv_path: str, now_utc: Optional[int]=None) -> List[Dict]:
    now_utc = now_utc or _utcnow()
    since = now_utc - 3600
    rows: List[Dict] = []
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                out = str(row.get("outcome","")).strip().lower()
                if out not in ("win","loss"):
                    continue
                ts = _safe_int(row.get("settled_ts") or row.get("ts") or row.get("time"))
                if ts is None or ts < since or ts > now_utc:
                    continue
                rows.append(row)
    except Exception:
        from error_logger import log_exception
        log_exception("Failed to read file")
    return rows

def _mode_score_for_row(mode: str, row: Dict, delta_eff: float) -> Optional[float]:
    """
    Возвращает прокс-прибыль g для конкретного режима на одной строке лога или None.
    """
    side = str(row.get("side","")).strip().upper()
    if side not in ("UP","DOWN"):
        return None
    p_up = _safe_float(row.get("p_up"))
    if p_up is None:
        return None
    p_up = _clamp(p_up, 1e-6, 1.0 - 1e-6)
    p_side = p_up if side == "UP" else (1.0 - p_up)

    # Пул (может отсутствовать — тогда disagree использует "no_pool" ветку)
    bet_up = _safe_float(row.get("bet_up"))
    bet_down = _safe_float(row.get("bet_down"))

    p_thr, _ = compute_p_thr_no_r(
        mode=mode,
        csv_path=os.getenv("CSV_PATH", ""),  # если есть глобальная, передадим
        p_up=p_up,
        side=side,
        bet_up=bet_up,
        bet_down=bet_down
    )
    margin = p_side - (p_thr + float(delta_eff or 0.0))
    if margin <= 0.0:
        return 0.0  # "не ставим" — нулевая прокс-прибыль

    # Прокс-ставка (EV-free) — одинаковая для всех режимов, чтобы сравнение было честным
    k = _safe_float(os.getenv("NO_R_AUTO_K","1.0"), 1.0)
    f_min = _safe_float(os.getenv("NO_R_AUTO_FMIN","0.0"), 0.0)
    f_max = _safe_float(os.getenv("NO_R_AUTO_FMAX","0.005"), 0.005)
    f = _clamp(k * margin, f_min, f_max)

    y = 1 if str(row.get("outcome","")).strip().lower() == "win" else 0
    g = (1 if y == 1 else -1) * f
    return g

def _eval_modes(csv_path: str, delta_eff: float) -> Tuple[Dict[str, float], int]:
    scores = {m: 0.0 for m in MODES}
    rows = _read_last_hour_rows(csv_path)
    n = 0
    for row in rows:
        n += 1
        for m in MODES:
            g = _mode_score_for_row(m, row, delta_eff=delta_eff)
            if g is not None:
                scores[m] += g
    return scores, n

def _load_state(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_state(path: str, obj: Dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        from error_logger import log_exception
        log_exception("Failed to load JSON")

def pick_no_r_mode(
    csv_path: str,
    current_mode: str,
    delta_eff: float,
    now_utc: Optional[int]=None
) -> Tuple[str, str]:
    """
    Возвращает (mode, reason).
    Учитывает: минимум наблюдений, cooldown, гистерезис.
    """
    now_utc = now_utc or _utcnow()
    state_path = os.getenv("NO_R_AUTO_STATE_PATH", "no_r_auto_state.json")

    min_n = int(os.getenv("NO_R_AUTO_MINN", "40"))
    cooldown = int(os.getenv("NO_R_AUTO_COOLDOWN", "900"))
    hyst = _safe_float(os.getenv("NO_R_AUTO_HYST", "0.0005"), 0.0005)

    scores, n = _eval_modes(csv_path, delta_eff=delta_eff)
    state = _load_state(state_path)
    last_switch = _safe_int(state.get("last_switch_ts"), 0)
    cur = (current_mode or "prob_lcb").strip().lower()
    if cur not in MODES:
        cur = "prob_lcb"

    if n < min_n:
        return (cur, f"auto=hold(min_n={min_n}, n={n}) scores={scores}")

    # анти-флаппер: cooldown
    if (now_utc - (last_switch or 0)) < cooldown:
        return (cur, f"auto=hold(cooldown ⩽ {cooldown}s) scores={scores}")

    # выбор лучшего
    best = max(MODES, key=lambda m: scores.get(m, float("-inf")))
    gain = scores.get(best, 0.0) - scores.get(cur, 0.0)

    if best != cur and gain > hyst:
        # переключаем
        state["last_switch_ts"] = now_utc
        state["last_mode"] = best
        state["last_scores"] = scores
        _save_state(state_path, state)
        return (best, f"auto=switch({cur}→{best}, Δ={gain:+.6f}) scores={scores}")
    else:
        # остаёмся
        return (cur, f"auto=stay({cur}, Δ={gain:+.6f} ≤ hyst={hyst}) scores={scores}")
