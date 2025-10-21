# proj_scenarios.py
from __future__ import annotations
import csv, math, time, statistics as st
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None

WEI = 10**18

# -------- УТИЛИТЫ --------
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

def _utcdate(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

# -------- ЧТЕНИЕ ТРЕЙДОВ И ДНЕВНЫЕ ЛОГ-РЕТЁРНЫ --------
def read_trades(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
    except Exception:
        return []
    return rows

def daily_log_returns(csv_path: str, lookback_days: int = 30) -> List[Tuple[str, float]]:
    """
    Возвращает список (YYYY-MM-DD, r_day), где r_day — дневной лог-рост капитала,
    агрегированный из сделок дня: r_day = sum(ln(cap_after/cap_before)) по сделкам дня.
    Если в дне нет пары cap_before/after — пропускаем.
    """
    now = int(time.time())
    since = now - lookback_days*86400

    rows = read_trades(csv_path)
    per_day: Dict[str, List[float]] = {}
    for r in rows:
        ts = _safe_int(r.get("settled_ts") or r.get("ts") or r.get("time"))
        if ts <= 0 or ts < since:
            continue
        cb = _safe_float(r.get("capital_before"), float("nan"))
        ca = _safe_float(r.get("capital_after"),  float("nan"))
        if not (math.isfinite(cb) and math.isfinite(ca) and cb > 0 and ca > 0):
            # запасной вариант: если нет capital_* берем pnl ~ относительный к before
            pnl_net = _safe_float(r.get("pnl") or r.get("pnl_net"), float("nan"))
            if math.isfinite(pnl_net) and math.isfinite(cb) and cb > 0:
                ca = cb + pnl_net
            else:
                continue
        r_log = math.log(ca / cb)
        d = _utcdate(ts)
        per_day.setdefault(d, []).append(r_log)

    days = []
    for d, arr in per_day.items():
        if arr:
            days.append((d, sum(arr)))
    days.sort(key=lambda x: x[0])
    return days

# -------- ОЦЕНКА ПАРАМЕТРОВ И SHINKAGE --------
def estimate_mu_sigma(day_logs: List[Tuple[str, float]]) -> Tuple[float, float]:
    """Простая оценка μ, σ дневных лог-ретёрнов."""
    if not day_logs:
        return 0.0, 0.0
    xs = [v for _, v in day_logs]
    n = len(xs)
    mu = st.mean(xs)
    sigma = (st.pstdev(xs) if n > 1 else 0.0)
    return mu, sigma

def shrink_mu(mu: float, sigma: float, n: int, prior_var: float = 0.0001) -> float:
    """
    Мягкая усадка μ к 0 (байесовская логика): mu_post = mu * (n*sigma^2)/(n*sigma^2 + tau^2),
    где tau^2 = prior_var — дисперсия «приора» относительно среднего 0.
    """
    if n <= 1:
        return 0.0
    tau2 = prior_var
    s2 = sigma*sigma
    w = (n * s2) / (n * s2 + tau2) if (n*s2 + tau2) > 0 else 0.0
    return w * mu

# -------- БЛОЧНЫЙ БУТСТРАП СИМУЛЯЦИЯ --------
def block_bootstrap_paths(day_logs: List[Tuple[str, float]],
                          horizon_days: int,
                          n_paths: int = 10000,
                          block_len: int = 3,
                          rng_seed: Optional[int] = None) -> Tuple[List[float], List[float]]:
    """
    Возвращает:
      cum_logs  — список суммарных лог-ростов за горизонт;
      mdds      — список max drawdown (в логах).
    """
    xs = [v for _, v in day_logs]
    n = len(xs)
    if n == 0:
        return [], []
    
    if np is None:
        # деградация без numpy
        import random
        rnd = random.Random(rng_seed)
        cum_logs = []
        mdds = []
        for _ in range(n_paths):
            seq = []
            remain = horizon_days
            while remain > 0:
                j = rnd.randrange(0, n)
                for k in range(block_len):
                    if remain <= 0: break
                    seq.append(xs[(j+k) % n])
                    remain -= 1
            total = sum(seq[:horizon_days])
            # max DD в уровнях:
            level = 0.0
            peak  = 0.0
            max_dd = 0.0
            for r in seq[:horizon_days]:
                level += r
                peak = max(peak, level)
                dd = level - peak
                if dd < max_dd:
                    max_dd = dd
            cum_logs.append(total)
            mdds.append(max_dd)
        return cum_logs, mdds
    else:
        rng = np.random.default_rng(rng_seed)
        xs_np = np.array(xs, dtype=float)
        cum_logs = np.empty(n_paths, dtype=float)
        mdds = np.empty(n_paths, dtype=float)
        for i in range(n_paths):
            seq = []
            remain = horizon_days
            while remain > 0:
                j = rng.integers(0, n)
                # Циклическое взятие блока
                blk_indices = np.arange(j, j + block_len) % n
                blk = xs_np[blk_indices]
                take = min(remain, block_len)
                seq.extend(blk[:take])
                remain -= take
            seq = np.array(seq, dtype=float)
            total = float(seq.sum())
            # max DD
            level = np.cumsum(seq)
            peak = np.maximum.accumulate(level)
            dd_series = level - peak
            cum_logs[i] = total
            mdds[i] = float(dd_series.min())
        return list(cum_logs), list(mdds)

# -------- ПЕРЕВОД В ПРОЦЕНТЫ И СЦЕНАРИИ --------
def logs_to_mult(logx: float) -> float:
    return math.exp(logx)

def logdd_to_pct(logdd: float) -> float:
    return math.exp(logdd) - 1.0

def summarize_scenarios(cum_logs: List[float], mdds: List[float]) -> Dict[str, Any]:
    if not cum_logs:
        return {"ok": False, "msg": "недостаточно данных"}
    
    def q(p, arr_sorted):
        k = int(round((len(arr_sorted)-1)*p))
        return arr_sorted[k]
    
    mults_sorted = sorted([logs_to_mult(x) for x in cum_logs])
    dd_pct_sorted = sorted([logdd_to_pct(d) for d in mdds])
    
    res = {
        "ok": True,
        "mult": {
            "p10": q(0.10, mults_sorted),
            "p50": q(0.50, mults_sorted),
            "p90": q(0.90, mults_sorted),
        },
        "dd": {
            "p10": q(0.10, dd_pct_sorted),
            "p50": q(0.50, dd_pct_sorted),
            "p90": q(0.90, dd_pct_sorted),
        }
    }
    return res

# -------- ПРОЕКЦИЯ С УЧЁТОМ ПРАВИЛА 35 BNB --------
def apply_threshold_payout(start_cap: float, mult: float, daily_rate_guess: float,
                           horizon_days: int, threshold: float = 35.0) -> Dict[str, float]:
    """
    Грубая аппроксимация политики «реинвест до 35, далее — вывод ежедневно».
    mult — итоговый мультипликатор за горизонт (из симуляции)
    daily_rate_guess — оценка средней дневной доходности (геометрической)
    Возвращает: итоговый капитал (min(35, ...)) и суммарные выводы за горизонт.
    """
    cap_end_nothold = start_cap * mult
    if cap_end_nothold <= threshold:
        return {"final_cap": cap_end_nothold, "withdraw": 0.0}
    
    # Если стартовый капитал уже >= порога
    if start_cap >= threshold:
        total_withdraw = cap_end_nothold - threshold
        return {"final_cap": threshold, "withdraw": max(0.0, total_withdraw)}
    
    if daily_rate_guess <= -1.0:
        return {"final_cap": start_cap, "withdraw": 0.0}
    if daily_rate_guess <= 0:
        return {"final_cap": min(threshold, cap_end_nothold), "withdraw": 0.0}
    
    d_star = math.ceil(math.log(threshold/start_cap) / math.log(1.0 + daily_rate_guess))
    d_star = max(0, min(horizon_days, d_star))
    cap_at_d = start_cap * ((1.0 + daily_rate_guess) ** d_star)
    overshoot = max(0.0, cap_at_d - threshold)
    remain = max(0, horizon_days - d_star)
    daily_withdraw = threshold * daily_rate_guess
    total_withdraw = overshoot + remain * daily_withdraw
    return {"final_cap": threshold, "withdraw": total_withdraw}

# -------- ВНЕШНЕЕ API ДЛЯ TG --------
def build_projection_text(csv_path: str,
                          horizons=(30, 90, 365),
                          start_cap: float = 2.0,
                          threshold: float = 35.0,
                          lookback_days: int = 30,
                          n_paths: int = 10000,
                          block_len: int = 3,
                          prior_var: float = 0.0001,
                          rng_seed: Optional[int] = None) -> str:
    days = daily_log_returns(csv_path, lookback_days=lookback_days)
    if not days or len(days) < max(5, block_len):
        return "📈 Проекция: недостаточно данных за последний месяц (нужно ≥5 торговых дней)."

    mu, sigma = estimate_mu_sigma(days)
    n = len(days)
    mu_shr = shrink_mu(mu, sigma, n, prior_var=prior_var)
    daily_g = mu_shr

    lines = []
    lines.append("🔮 *Проекция доходности (3 сценария)*")
    lines.append(f"Окно: *{n}* дней, μ̂(shrink)={mu_shr:+.4f}, σ={sigma:.4f} (в логах/день)")
    lines.append(f"Стартовый капитал: *{start_cap:.6f}* BNB, порог реинвеста: *{threshold:.3f}* BNB")
    lines.append("")

    for H in horizons:
        cum_logs, mdds = block_bootstrap_paths(days, horizon_days=H,
                                               n_paths=n_paths, block_len=block_len, rng_seed=rng_seed)
        summ = summarize_scenarios(cum_logs, mdds)
        if not summ.get("ok"):
            lines.append(f"⏳ Горизонт {H}д: данных мало")
            continue
        mults = summ["mult"]; dds = summ["dd"]
        scen_text = []
        for tag, mult in (("Пессимистичный (10%)", mults["p10"]),
                          ("Реалистичный (50%)", mults["p50"]),
                          ("Оптимистичный (90%)", mults["p90"])):
            pol = apply_threshold_payout(start_cap=start_cap, mult=mult,
                                         daily_rate_guess=(math.exp(daily_g)-1.0),
                                         horizon_days=H, threshold=threshold)
            cap = pol["final_cap"]; wd = pol["withdraw"]
            annualized = (mult ** (365.0 / H)) - 1.0 if H > 0 else 0.0
            scen_text.append(
                f"• *{tag}:* итог ×{mult:.3f} ({annualized*100:.1f}% годовых экв.), "
                f"выводов за период: *{wd:.6f}* BNB, капитал в конце: *{cap:.6f}* BNB"
            )
        lines.append(f"⏱ Горизонт: *{H} дней*")
        lines.extend(scen_text)
        lines.append(f"  Max DD (10/50/90%): {dds['p10']*100:.1f}% / {dds['p50']*100:.1f}% / {dds['p90']*100:.1f}%")
        lines.append("")

    return "\n".join(lines)

def try_send_projection(csv_path: str, tg_send_fn, **kwargs) -> Optional[str]:
    text = build_projection_text(csv_path, **kwargs)
    ok = False
    try:
        ok = bool(tg_send_fn(text, html=False, parse_mode="Markdown"))
    except TypeError:
        ok = bool(tg_send_fn(text, html=False))
    except Exception:
        ok = False
    return text if ok else None
