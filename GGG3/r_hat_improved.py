# r_hat_improved.py
# -*- coding: utf-8 -*-
"""
Улучшенная оценка r̂ (payout ratio) для PancakeSwap Prediction.
Приоритет: IMPLIED из текущего пула → исторические методы.
"""

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from error_logger import log_exception


def estimate_late_money_direction(pool, rd, our_side_up: bool) -> float:
    """
    Анализирует последние N эпох из pool.obs для определения паттерна поздних денег.
    Определяет идут ли поздние деньги за толпой или против неё.
    
    Args:
        pool: PoolFeaturesCtx с историей снимков в pool.obs
        rd: RoundInfo текущего раунда
        our_side_up: True если ставим на UP
    
    Returns:
        0.0 = все против нас, 0.5 = нейтрально, 1.0 = все за нас
    """
    try:
        if not hasattr(pool, 'obs') or not pool.obs:
            return 0.5
        
        # Получаем последние 30 завершённых эпох (исключая текущую)
        all_epochs = sorted(pool.obs.keys())
        current_epoch = int(getattr(rd, 'epoch', 0))
        recent_epochs = [e for e in all_epochs if e < current_epoch][-30:]
        
        if len(recent_epochs) < 10:
            return 0.5
        
        patterns = []
        
        for epoch in recent_epochs:
            snapshots = pool.obs[epoch]
            if len(snapshots) < 2:
                continue
            
            # Первый и последний снимок: (ts, bull, bear)
            initial = snapshots[0]
            final = snapshots[-1]
            
            ts_init, bull_init, bear_init = initial
            ts_final, bull_final, bear_final = final
            
            total_init = bull_init + bear_init
            total_final = bull_final + bear_final
            
            if total_init < 1e-6 or total_final < 1e-6:
                continue
            
            # Начальный bias (кто доминирует в начале)
            bias_init = bull_init / total_init
            
            # Приток денег за весь период
            bull_flow = max(0, bull_final - bull_init)
            bear_flow = max(0, bear_final - bear_init)
            total_flow = bull_flow + bear_flow
            
            if total_flow < 1e-6:
                continue
            
            # Bias притока (куда идут новые деньги)
            flow_bias = bull_flow / total_flow
            
            patterns.append({
                'bias_init': bias_init,
                'flow_bias': flow_bias
            })
        
        if len(patterns) < 10:
            return 0.5
        
        # Анализ: поздние деньги идут за толпой или против?
        follow_count = 0
        contrarian_count = 0
        
        for p in patterns:
            bi = p['bias_init']
            fb = p['flow_bias']
            
            # "Толпа за толпой" (оба смещены в одну сторону)
            if (bi > 0.55 and fb > 0.55) or (bi < 0.45 and fb < 0.45):
                follow_count += 1
            # "Умные деньги против толпы" (поздние против начального смещения)
            elif (bi > 0.55 and fb < 0.45) or (bi < 0.45 and fb > 0.55):
                contrarian_count += 1
        
        # Текущий bias пула
        current_bull_bias = float(rd.bull_amount) / max(1e-9, float(rd.bull_amount + rd.bear_amount))
        
        # Если доминирует "толпа за толпой" → поздние деньги пойдут туда же куда pool
        if follow_count > contrarian_count * 1.5:
            return current_bull_bias if our_side_up else (1.0 - current_bull_bias)
        
        # Если доминирует "умные деньги" → поздние деньги пойдут против pool
        elif contrarian_count > follow_count * 1.5:
            return (1.0 - current_bull_bias) if our_side_up else current_bull_bias
        
        # Нейтрально (нет явного паттерна)
        return 0.5
        
    except Exception:
        log_exception("estimate_late_money_direction failed")
        return 0.5


def adaptive_quantile(csv_path: str, n: int = 100, max_epoch_exclusive: Optional[int] = None) -> float:
    """
    Выбирает перцентиль динамически на основе точности прогнозов.
    
    Args:
        csv_path: Путь к CSV с историей сделок
        n: Размер окна для анализа
        max_epoch_exclusive: Не учитывать эпохи >= этого значения (для честности)
    
    Returns:
        0.30 (консервативно) .. 0.50 (медиана)
    """
    try:
        from bnbusdrt6 import rolling_winrate_laplace, rolling_calib_error
        
        wr = rolling_winrate_laplace(csv_path, n=n, max_epoch_exclusive=max_epoch_exclusive)
        calib_err = rolling_calib_error(csv_path, n=n)
        
        # Высокая точность → менее консервативны
        if wr and wr > 0.55 and calib_err < 0.08:
            return 0.50  # медиана
        elif wr and wr > 0.52 and calib_err < 0.10:
            return 0.40
        else:
            return 0.30  # консервативно (по умолчанию)
    except Exception:
        return 0.30


def volatility_penalty(rv: float, threshold: float = 0.025) -> float:
    """
    При высокой волатильности цены пулы перекашиваются сильнее.
    Консервативно снижаем ожидаемую выплату.
    
    Args:
        rv: Realized volatility (например, RV60)
        threshold: Порог волатильности (2.5% по умолчанию)
    
    Returns:
        Множитель 0.85..1.0
    """
    try:
        if not math.isfinite(rv) or rv < 0:
            return 1.0
        
        if rv > threshold:
            # Снижаем до 15% при очень высокой волатильности
            penalty = 1.0 - min(0.15, (rv - threshold) * 3)
            return float(max(0.85, penalty))
        
        return 1.0
    except Exception:
        return 1.0


def estimate_r_hat_improved(
    rd,  # RoundInfo
    bet_up: bool,
    epoch: int,
    pool,  # PoolFeaturesCtx
    csv_path: str,
    kl_df: pd.DataFrame,
    treasury_fee: float = 0.03,
    use_stress_r15: bool = True,
    r2d = None
) -> Tuple[float, str]:
    """
    Улучшенная оценка r̂ с приоритетом на IMPLIED из текущего пула.
    
    Args:
        rd: RoundInfo текущего раунда
        bet_up: True если ставим на UP
        epoch: Номер эпохи (для честной оценки без заглядывания)
        pool: PoolFeaturesCtx для анализа поздних денег
        csv_path: Путь к CSV с историей
        kl_df: DataFrame с OHLCV для расчёта волатильности
        treasury_fee: Комиссия протокола (0.03 = 3%)
        use_stress_r15: Использовать корректировку на поздние деньги
        r2d: RHat2D объект для 2D-оценки (опционально)
    
    Returns:
        (r_hat, source_description)
    """
    from bnbusdrt6 import (
        implied_payout_ratio, r_ewma_by_side, r_tod_percentile,
        last3_ev_estimates
    )
    from extra_features import realized_metrics
    
    source_parts = []
    
    # ШАГ 1: IMPLIED из текущего состояния пула (ПРИОРИТЕТ!)
    r_implied = implied_payout_ratio(bet_up, rd, treasury_fee)
    
    if r_implied and math.isfinite(r_implied) and r_implied > 1.0:
        r_base = r_implied
        source_parts.append(f"implied={r_implied:.3f}")
        
        # ШАГ 2: Корректировка на "поздние деньги"
        if use_stress_r15 and hasattr(pool, 'late_delta_quantile'):
            try:
                delta15 = float(pool.late_delta_quantile(q=0.5) or 0.0)
                
                # Оцениваем направление поздних денег на основе истории
                late_fraction = estimate_late_money_direction(pool, rd, bet_up)
                
                side_now = float(rd.bull_amount if bet_up else rd.bear_amount)
                total_now = float(rd.bull_amount + rd.bear_amount)
                
                # Реалистичный сценарий (не 100% к нам, а late_fraction)
                side_adj = side_now + late_fraction * delta15
                total_adj = total_now + delta15
                
                r_adjusted = (total_adj / max(1e-12, side_adj)) * (1.0 - treasury_fee)
                
                if r_adjusted > 1.0 and r_adjusted < r_base:
                    r_base = r_adjusted
                    source_parts.append(f"late_adj={r_adjusted:.3f}(frac={late_fraction:.2f})")
            except Exception:
                log_exception("r_hat: late_adj calc failed")
        
        # ШАГ 3: Blend с исторической EWMA (30% веса истории)
        try:
            r_hist = r_ewma_by_side(
                path=csv_path,
                side_up=bet_up,
                alpha=0.25,
                max_epoch_exclusive=epoch
            )
            
            if r_hist and math.isfinite(r_hist) and r_hist > 1.0:
                r_blended = 0.70 * r_base + 0.30 * r_hist
                source_parts.append(f"ewma={r_hist:.3f}")
                source_parts.append(f"blend70/30={r_blended:.3f}")
                r_base = r_blended
        except Exception:
            log_exception("r_hat: ewma blend failed")
        
        # ШАГ 4: Penalty за волатильность
        try:
            RV60 = realized_metrics(kl_df["close"], 60)[0]
            if RV60 and math.isfinite(RV60):
                vol_mult = volatility_penalty(RV60, threshold=0.025)
                if vol_mult < 1.0:
                    r_base *= vol_mult
                    source_parts.append(f"vol_penalty={vol_mult:.3f}")
        except Exception:
            log_exception("r_hat: volatility penalty calc failed")
        
        # ВАЛИДАЦИЯ: ограничиваем разумным диапазоном
        r_final = float(np.clip(r_base, 1.1, 5.0))
        source = " | ".join(source_parts)
        return r_final, f"improved({source})"
    
    # FALLBACK: исторические методы
    try:
        q = adaptive_quantile(csv_path, n=100, max_epoch_exclusive=epoch)
        
        # 1) Перцентиль по часу суток
        try:
            lock_ts = int(getattr(rd, "lock_ts", int(time.time())))
            hour_utc = int(pd.to_datetime(lock_ts, unit="s", utc=True).hour)
        except Exception:
            hour_utc = 0
        
        r_hat = r_tod_percentile(
            path=csv_path,
            side_up=bet_up,
            hour_utc=hour_utc,
            q=q,
            max_epoch_exclusive=epoch
        )
        
        if r_hat and math.isfinite(r_hat) and r_hat > 1.0:
            r_hat = float(np.clip(r_hat, 1.1, 5.0))
            return r_hat, f"tod_q{int(q*100)}={r_hat:.3f}"
        
        # 2) EWMA
        r_hat = r_ewma_by_side(
            path=csv_path,
            side_up=bet_up,
            alpha=0.25,
            max_epoch_exclusive=epoch
        )
        
        if r_hat and math.isfinite(r_hat) and r_hat > 1.0:
            r_hat = float(np.clip(r_hat, 1.1, 5.0))
            return r_hat, f"ewma={r_hat:.3f}"
        
        # 3) 2D-таблица
        if r2d is not None:
            try:
                r_hat = r2d.estimate(
                    side=("UP" if bet_up else "DOWN"),
                    lock_ts=int(getattr(rd, "lock_ts", int(time.time()))),
                    csv_path=csv_path
                )
                if r_hat and math.isfinite(r_hat) and r_hat > 1.0:
                    r_hat = float(np.clip(r_hat, 1.1, 5.0))
                    return r_hat, f"r2d={r_hat:.3f}"
            except Exception:
                log_exception("r_hat: r2d estimation failed")
        
        # 4) Медиана последних 3
        r_med, _, _ = last3_ev_estimates(csv_path)
        if r_med and math.isfinite(r_med) and r_med > 1.0:
            r_med = float(np.clip(r_med, 1.1, 5.0))
            return r_med, f"last3_med={r_med:.3f}"
        
        # 5) IMPLIED как последний fallback
        r_imp = implied_payout_ratio(bet_up, rd, treasury_fee)
        if r_imp and math.isfinite(r_imp) and r_imp > 1.0:
            r_imp = float(np.clip(r_imp, 1.1, 5.0))
            return r_imp, f"implied_fallback={r_imp:.3f}"
        
    except Exception:
        log_exception("r_hat: all methods failed; using hardcoded fallback 1.90")
    
    return 1.90, "hardcoded_fallback=1.90"


def analyze_r_hat_accuracy(csv_path: str, n: int = 200) -> Optional[Dict[str, float]]:
    """
    Анализирует точность оценки r̂ по последним n сделкам.
    
    Args:
        csv_path: Путь к CSV с историей сделок
        n: Размер окна анализа
    
    Returns:
        dict с метриками: mae, mae_pct, rmse, bias, bias_pct, n_samples
        або None если данных недостаточно
    """
    try:
        from bnbusdrt6 import _read_csv_df
        
        df = _read_csv_df(csv_path)
        if df.empty:
            return None
        
        df = df.dropna(subset=["r_hat_used", "payout_ratio"])
        df = df[df["outcome"].isin(["win", "loss", "draw"])]
        df = df.tail(n)
        
        if len(df) < 10:
            return None
        
        r_hat = pd.to_numeric(df["r_hat_used"], errors="coerce").dropna()
        r_actual = pd.to_numeric(df["payout_ratio"], errors="coerce").dropna()
        
        # Убедимся что индексы совпадают
        common_idx = r_hat.index.intersection(r_actual.index)
        if len(common_idx) < 10:
            return None
        
        r_hat = r_hat.loc[common_idx].to_numpy()
        r_actual = r_actual.loc[common_idx].to_numpy()
        
        errors = np.abs(r_actual - r_hat)
        rel_errors = errors / np.maximum(r_actual, 1e-9)
        bias = (r_hat - r_actual).mean()
        
        return {
            "mae": float(errors.mean()),
            "mae_pct": float(rel_errors.mean() * 100),
            "rmse": float(np.sqrt((errors ** 2).mean())),
            "bias": float(bias),
            "bias_pct": float(bias / r_actual.mean() * 100) if r_actual.mean() > 0 else 0.0,
            "n_samples": len(r_hat)
        }
    except Exception:
        return None
