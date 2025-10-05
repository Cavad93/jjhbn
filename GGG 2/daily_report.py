# daily_report.py
from __future__ import annotations
import csv, math, time, statistics as st
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta

from calib.selector import brier as _brier_ext, nll as _nll_ext, ece as _ece_ext
from metrics.reliability import reliability_curve


B = 1e9  # gwei scale
WEI = 10**18

# --- Утилиты ---
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None: return default
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip()
        if not s: return default
        return float(s)
    except Exception:
        return default

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None: return default
        if isinstance(x, (int, float)): return int(x)
        s = str(x).strip()
        if not s: return default
        return int(float(s))
    except Exception:
        return default

def _utcnow() -> int:
    return int(time.time())

def _sparkline(xs: List[float], buckets: int = 40) -> str:
    # мини-спарклайн на юникод-блоках
    if not xs:
        return "—"
    n = min(len(xs), buckets)
    xs = xs[-n:]
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return "▁" * n
    blocks = "▁▂▃▄▅▆▇█"
    out = []
    for v in xs:
        t = (v - lo) / (hi - lo)
        k = min(len(blocks)-1, max(0, int(round(t * (len(blocks)-1)))))
        out.append(blocks[k])
    return "".join(out)

def _pct(x: float) -> str:
    return f"{x*100:+.2f}%"

def _fmt(x: float, nd: int = 6) -> str:
    return f"{x:.{nd}f}"

def _brier(p: List[float], y: List[int]) -> Optional[float]:
    if not p or not y or len(p) != len(y): return None
    return sum((pi - yi)**2 for pi, yi in zip(p, y)) / len(p)

def _roi_from_capitals(cap_before: float, cap_after: float) -> Optional[float]:
    if cap_before and cap_after and cap_before > 0:
        return (cap_after / cap_before) - 1.0
    return None

# --- чтение трейдов из CSV ---
def read_trades(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    try:
        # utf-8-sig удаляет BOM у первого заголовка (в твоём файле это \ufeffsettled_ts)
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            r = csv.DictReader(f)
            for row in r:
                # На всякий случай уберём BOM и пробелы у всех ключей
                clean = {}
                for k, v in row.items():
                    if isinstance(k, str):
                        k = k.lstrip("\ufeff").strip()
                    clean[k] = v
                rows.append(clean)
    except Exception:
        return []
    return rows


# --- основной расчёт за 24ч ---
def compute_24h_metrics(csv_path: str, now_utc: Optional[int] = None) -> Dict[str, Any]:
    now_utc = now_utc or _utcnow()
    since = now_utc - 24 * 3600

    rows = read_trades(csv_path)

    # фильтруем только завершённые за 24ч
    filt: List[Dict[str, Any]] = []
    for r in rows:
        ts = _safe_int(r.get("settled_ts") or r.get("ts") or r.get("time"))
        if ts >= since and ts <= now_utc:
            # исключим «черновики» без outcome
            out = (r.get("outcome") or "").strip().lower()
            if out in ("win", "loss") or ("pnl" in r or "pnl_net" in r):
                filt.append(r)

    n = len(filt)

    # агрегаты
    wins = 0
    losses = 0
    pnl_raw = 0.0
    pnl_net = 0.0
    gas_sum = 0.0
    stakes: List[float] = []
    edges: List[float] = []
    caps_bef: List[float] = []
    caps_aft: List[float] = []
    probs_used: List[float] = []
    labels: List[int] = []
    gas_gwei: List[float] = []

    best = None  # type: Optional[Tuple[float, Dict[str, Any]]]
    worst = None  # type: Optional[Tuple[float, Dict[str, Any]]]
    series_cap: List[float] = []  # для спарклайна

    for r in filt:
        out = (r.get("outcome") or "").strip().lower()
        y = 1 if out == "win" else 0 if out == "loss" else None
        if y is not None:
            wins += y
            losses += (1 - y)
            labels.append(y)

        p_final = r.get("p_final_used") or r.get("p_side") or r.get("p_up")
        if p_final is not None:
            p = max(0.0, min(1.0, _safe_float(p_final, 0.5)))
            probs_used.append(p)

        stake = _safe_float(r.get("stake"))
        if math.isfinite(stake) and stake > 0:
            stakes.append(stake)

        edge = _safe_float(r.get("edge_at_entry"))
        if math.isfinite(edge):
            edges.append(edge)

        gb = _safe_float(r.get("gas_bet_bnb"))
        gc = _safe_float(r.get("gas_claim_bnb"))
        gas_sum += max(0.0, gb) + max(0.0, gc)

        gwei1 = _safe_float(r.get("gas_price_bet_gwei"))
        gwei2 = _safe_float(r.get("gas_price_claim_gwei"))
        if gwei1:
            gas_gwei.append(gwei1)
        if gwei2:
            gas_gwei.append(gwei2)

        pnl_r = _safe_float(r.get("pnl"))
        pnl_n = _safe_float(r.get("pnl_net"), pnl_r - (gb + gc if (gb or gc) else 0.0))

        pnl_raw += pnl_r
        pnl_net += pnl_n

        cb = _safe_float(r.get("capital_before"))
        ca = _safe_float(r.get("capital_after"))
        if cb:
            caps_bef.append(cb)
        if ca:
            caps_aft.append(ca)
            series_cap.append(ca)

        # best/worst по net
        if best is None or pnl_n > best[0]:
            best = (pnl_n, r)
        if worst is None or pnl_n < worst[0]:
            worst = (pnl_n, r)

    wr = wins / n if n else 0.0
    ev_brier = _brier(probs_used, labels)

    # --- новые метрики калибровки ---
    try:
        ev_nll  = _nll_ext(labels, probs_used) if probs_used else None
        ev_ece  = _ece_ext(labels, probs_used, n_bins=15) if probs_used else None
        conf, acc = reliability_curve(labels, probs_used, n_bins=15)
    except Exception:
        ev_nll, ev_ece, conf, acc = None, None, None, None

    # ROI по капиталу если есть
    roi_cap = None
    if caps_bef and caps_aft:
        roi_cap = _roi_from_capitals(min(caps_bef), max(caps_aft))

    # DD & восстановление по capital_after
    dd = None
    recov = None
    if series_cap:
        peak = -1e30
        max_dd = 0.0
        trough_after_peak = None
        last = series_cap[-1]
        for v in series_cap:
            peak = max(peak, v)
            drawdown = 0.0 if peak <= 0 else (v / peak - 1.0)
            if drawdown < max_dd:
                max_dd = drawdown
                trough_after_peak = v
        dd = max_dd
        if trough_after_peak and peak > 0:
            recov = last / peak - 1.0

    # медиана EV-порога (если писали)
    p_thr_eff_list: List[float] = []
    for r in filt:
        val = r.get("p_thr_eff") or r.get("p_thr")
        v = _safe_float(val, float("nan"))
        if math.isfinite(v):
            p_thr_eff_list.append(v)
    p_thr_med = st.median(p_thr_eff_list) if p_thr_eff_list else None

    # агрегаты газа / стейков / edge
    gwei_med = st.median(gas_gwei) if gas_gwei else None
    stake_med = st.median(stakes) if stakes else None
    edge_med = st.median(edges) if edges else None

    # ---- новые метрики ----
    payout_med = None
    kelly_med = None
    gas_pct_med = None
    hold_med_s = None

    try:
        payouts = [_safe_float(r.get("payout_ratio"), float("nan")) for r in filt]
        payouts = [x for x in payouts if math.isfinite(x) and x > 0]
        payout_med = (st.median(payouts) if payouts else None)
    except Exception:
        payout_med = None

    try:
        pairs = []
        for r in filt:
            cap_b = _safe_float(r.get("capital_before"), float("nan"))
            stv = _safe_float(r.get("stake"), float("nan"))
            if math.isfinite(cap_b) and math.isfinite(stv) and cap_b > 0 and stv >= 0:
                pairs.append(stv / cap_b)
        kelly_med = (st.median(pairs) if pairs else None)
    except Exception:
        kelly_med = None

    try:
        gpairs = []
        for r in filt:
            stv = _safe_float(r.get("stake"), float("nan"))
            gb = _safe_float(r.get("gas_bet_bnb"), 0.0)
            gc = _safe_float(r.get("gas_claim_bnb"), 0.0)
            gsum = gb + gc
            if math.isfinite(stv) and stv > 0 and math.isfinite(gsum) and gsum >= 0:
                gpairs.append(gsum / stv)
        gas_pct_med = (st.median(gpairs) if gpairs else None)
    except Exception:
        gas_pct_med = None

    try:
        durs = []
        for r in filt:
            lt = _safe_int(r.get("lock_ts"))
            ct = _safe_int(r.get("close_ts"))
            if lt > 0 and ct > 0 and ct >= lt:
                durs.append(ct - lt)
        hold_med_s = (st.median(durs) if durs else None)
    except Exception:
        hold_med_s = None

    return {
        "n": n, "text": "За последние 24 часа завершённых сделок нет.",
    } if n == 0 else {
        "n": n,
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "pnl_raw": pnl_raw,
        "pnl_net": pnl_net,
        "roi_cap": roi_cap,
        "dd": dd,
        "recov": recov,
        "edge_med": edge_med,
        "stake_med": stake_med,
        "p_thr_med": p_thr_med,
        "ev_brier": ev_brier,
        "ev_nll": float(ev_nll) if ev_nll is not None else None,
        "ev_ece": float(ev_ece) if ev_ece is not None else None,
        "reliability_bins": (
            {"confidence": list(map(float, conf)), "accuracy": list(map(float, acc))}
            if (conf is not None and acc is not None) else None
        ),
        "gas_sum": gas_sum,
        "gwei_med": gwei_med,
        "cap_spark": _sparkline(series_cap, 40),
        "best": best,
        "worst": worst,
        # новые поля:
        "payout_med": payout_med,
        "kelly_med": kelly_med,
        "gas_pct_med": gas_pct_med,
        "hold_med_s": hold_med_s,
    }


# --- рендер текста ---
def render_report(m: Dict[str, Any], tz_name: str = "Europe/Berlin") -> str:
    if not m or m.get("n", 0) == 0:
        return "📊 Ежедневный отчёт: за последние 24 часа завершённых сделок нет."

    n = m["n"]; wr = m["wr"]
    parts = []
    parts.append("📊 *Ежедневный отчёт за 24ч*")
    parts.append(f"Сделок: *{n}*, Winrate: *{wr*100:.2f}%*")
    # --- добавили строки NLL/ECE (если посчитались) ---
    if m.get("ev_nll") is not None:
        parts.append(f"NLL: *{m['ev_nll']:.4f}*  _(меньше — лучше)_")
    if m.get("ev_ece") is not None:
        parts.append(f"ECE(15): *{m['ev_ece']:.4f}*  _(меньше — лучше)_")
    if m.get("roi_cap") is not None:
        parts.append(f"ROI по капиталу: *{_pct(m['roi_cap'])}*")
    parts.append(f"PnL: валовый *{_fmt(m['pnl_raw'],6)}* BNB, чистый *{_fmt(m['pnl_net'],6)}* BNB")
    if m.get("dd") is not None:
        parts.append(f"Макс. просадка: *{_pct(m['dd'])}*, восстановление: *{_pct(m['recov'] or 0.0)}*")
    parts.append(f"Средний стейк (мед.): *{_fmt(m.get('stake_med') or 0.0,6)}* BNB")
    if m.get("edge_med") is not None:
        parts.append(f"Медианный edge при входе: *{_fmt(m['edge_med'],4)}*")
    if m.get("p_thr_med") is not None:
        parts.append(f"Медианный порог p_thr_eff: *{_fmt(m['p_thr_med'],3)}*")
    if m.get("ev_brier") is not None:
        parts.append(f"Калибровка (Brier): *{m['ev_brier']:.4f}*  _(меньше — лучше)_")
    if m.get("gwei_med") is not None:
        parts.append(f"Газ (медиана): *{_fmt(m['gwei_med'],1)}* gwei, всего газа: *{_fmt(m['gas_sum'],6)}* BNB")

    # Новые параметры
    if m.get("payout_med") is not None:
        parts.append(f"Реальный payout (мед.): *{_fmt(m['payout_med'],3)}*")
    if m.get("kelly_med") is not None:
        parts.append(f"Kelly f (мед.): *{_fmt(m['kelly_med']*100,2)}%*")
    if m.get("gas_pct_med") is not None:
        parts.append(f"Газ/стейк (мед.): *{_fmt(m['gas_pct_med']*100,2)}%*")
    if m.get("hold_med_s") is not None:
        parts.append(f"Длительность раунда (мед.): *{int(m['hold_med_s']//60)} мин*")


    parts.append("")
    parts.append(f"📈 Капитал: `{m['cap_spark']}`")

    # Лучшее/худшее
    best = m.get("best"); worst = m.get("worst")
    if best and best[1]:
        r = best[1]
        parts.append(f"🥇 Лучшая сделка: *{_fmt(best[0],6)}* BNB  (epoch {r.get('epoch','?')}, side={r.get('side','?')})")
    if worst and worst[1]:
        r = worst[1]
        parts.append(f"🥶 Худшая сделка: *{_fmt(worst[0],6)}* BNB  (epoch {r.get('epoch','?')}, side={r.get('side','?')})")

    # Короткая «живая» интерпретация
    hints = []
    if m["wr"] < 0.5 and (m.get("edge_med") or 0) > 0:
        hints.append("Положительный edge при низком winrate — возможно, EV-порог высоковат.")
    if (m.get("gwei_med") or 0) > 20:
        hints.append("Газ высокий — проверь, не съедает ли он мелкие edge'и.")
    if m.get("dd") and m["dd"] < -0.05:
        hints.append("Просадка >5% — стоит снизить плечо/стейк до стабилизации.")
    if not hints:
        hints.append("Система работает штатно. Keep calm & let the bot cook 😎")

    parts.append("")
    parts.append("🧠 Заметки:")
    for h in hints:
        parts.append(f"• {h}")

    return "\n".join(parts)

# --- внешнее API: отправка (с троттлингом 1 раз/сутки) ---
class _Throttle:
    last_sent_utc: Optional[int] = None

throttle = _Throttle()

def try_send(csv_path: str, tg_send_fn, *, now_utc: Optional[int] = None, force: bool = False) -> Optional[str]:
    """
    tg_send_fn: функция вида tg_send(text: str, html=True|False, parse_mode="Markdown")
    """
    now_utc = now_utc or _utcnow()
    if not force:
        # не чаще 1 раза в 20 часов
        if throttle.last_sent_utc and (now_utc - throttle.last_sent_utc) < 20*3600:
            return None

        # шлём около 00:05 UTC+01/UTC+02? — упростим: если сейчас минут <10 и час в [0..2] UTC
        tm = datetime.fromtimestamp(now_utc, tz=timezone.utc)
        if not (tm.minute < 10 and tm.hour in (0,1,2)):
            return None

    m = compute_24h_metrics(csv_path, now_utc)
    text = render_report(m)
    ok = False
    try:
        # по умолчанию Markdown
        ok = bool(tg_send_fn(text, html=False, parse_mode="Markdown"))
    except TypeError:
        # если у tg_send нет parse_mode
        ok = bool(tg_send_fn(text, html=False))
    except Exception:
        ok = False

    if ok:
        throttle.last_sent_utc = now_utc
    return text if ok else None
