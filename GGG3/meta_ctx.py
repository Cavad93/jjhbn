# meta_ctx.py
# -*- coding: utf-8 -*-
"""
Модуль построения контекстных признаков для META-модели

Формирует вектор ψ (контекст) из рыночных условий:
- Тренд (MACD histogram на HTF с z-нормализацией)
- Волатильность (ATR ratio)
- Скачки цены (jump detection via BN-S test)
- Микроструктура (order flow, book imbalance)
- Фьючерсы (basis, funding rate)

Все фичи вычисляются СТРОГО из данных ДО lock timestamp,
чтобы избежать look-ahead bias.
"""
import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from collections import defaultdict

# ========== КОНСТАНТЫ ==========
VOL_THRESHOLD = 1.3  # Порог высокой волатильности

# ========== ГЛОБАЛЬНЫЕ СЧЕТЧИКИ ОШИБОК ==========
_ERROR_COUNTS = defaultdict(int)
_FEATURE_STATS = defaultdict(list)

# ========== УТИЛИТЫ ==========

def ema(series: pd.Series, n: int) -> pd.Series:
    """Экспоненциальное скользящее среднее"""
    return series.ewm(span=int(max(2, n)), adjust=False).mean()


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    """
    MACD Histogram: (MACD - Signal)
    
    Положительный = бычий момент, отрицательный = медвежий
    """
    macd = ema(close, fast) - ema(close, slow)
    signal = ema(macd, sig)
    return macd - signal


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Ресемплирует OHLCV данные в более высокий таймфрейм
    
    Args:
        df: DataFrame с колонками [open, high, low, close, volume]
        rule: правило ресемплинга ('5min', '15min', '1H' и т.д.)
    
    Returns:
        Ресемплированный DataFrame без NaN строк
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }
    out = df[["open", "high", "low", "close", "volume"]].resample(
        rule, label="left", closed="left"
    ).agg(agg)
    return out.dropna(how="any")


def _sign(x: float) -> float:
    """Знаковая функция: +1, 0, или -1"""
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def _clip(x: float, lo: float, hi: float) -> float:
    """Клиппинг значения в диапазон [lo, hi]"""
    return float(max(lo, min(hi, x)))


# ========== ОСНОВНАЯ ФУНКЦИЯ ==========

def build_regime_ctx(
    df_1m: pd.DataFrame,
    feats: dict,
    tstamp: pd.Timestamp,
    micro_feats: Optional[dict],
    fut_feats: Optional[dict],
    jump_flag: float = 0.0,
    htf_rule: str = "15min",
    enable_logging: bool = False
) -> dict:
    """
    Строит контекстный вектор ψ для META-модели из рыночных условий
    
    ⚠️ КРИТИЧНО: Все данные берутся СТРОГО ДО tstamp (no look-ahead bias)
    
    Args:
        df_1m: DataFrame с минутными OHLCV свечами (индекс = pd.DatetimeIndex)
        feats: Словарь с предвычисленными фичами (должен содержать 'atr', 'atr_sma')
        tstamp: Timestamp момента lock (используем данные ДО этого момента)
        micro_feats: Словарь с микроструктурными фичами (ofi_15s, book_imb)
        fut_feats: Словарь с фьючерсными фичами (basis_now, funding_sign)
        jump_flag: Флаг детекции скачка цены (0.0 или 1.0)
        htf_rule: Правило ресемплинга для HTF тренда (по умолчанию 15min)
        enable_logging: Включить логирование ошибок (для отладки)
    
    Returns:
        Словарь с 8 контекстными признаками
    """
    global _ERROR_COUNTS
    
    # --- 1) ТРЕНД (HTF MACD-hist с z-нормализацией) ---
    try:
        htf = resample_ohlc(df_1m, htf_rule)
        i = htf.index.get_indexer([tstamp], method="ffill")[0]
        
        # Защита от текущей незакрытой свечи
        if i >= 0 and htf.index[i] >= tstamp:
            i -= 1
        
        if i < 1:
            trend_sign = 0.0
            trend_abs = 0.0
        else:
            h = macd_hist(htf["close"])
            h_win = h.iloc[max(0, i - 100):i + 1]
            std = float(h_win.std(ddof=0)) if len(h_win) > 1 else 0.0
            h_val = float(h.iloc[i])
            trend_sign = _sign(h_val)
            trend_abs = 0.0 if std == 0.0 else _clip(abs(h_val) / std, 0.0, 3.0)
            
        if enable_logging:
            _FEATURE_STATS['trend_sign'].append(trend_sign)
            _FEATURE_STATS['trend_abs'].append(trend_abs)
            
    except Exception as e:
        _ERROR_COUNTS['trend'] += 1
        if enable_logging:
            print(f"[regime_ctx] trend computation failed: {e}")
        trend_sign, trend_abs = 0.0, 0.0

    # --- 2) ВОЛАТИЛЬНОСТЬ (ATR ratio) ---
    try:
        j = feats["atr"].index.get_indexer([tstamp], method="ffill")[0]
        
        if j >= 0 and feats["atr"].index[j] >= tstamp:
            j -= 1
        
        if j < 0:
            vol_ratio = 0.0
        else:
            atr_now = float(feats["atr"].iloc[j])
            atr_sma = float(feats["atr_sma"].iloc[j])
            vol_ratio = _clip(atr_now / atr_sma, 0.0, 5.0) if atr_sma > 0 else 0.0
        
        if enable_logging:
            _FEATURE_STATS['vol_ratio'].append(vol_ratio)
            
    except Exception as e:
        _ERROR_COUNTS['vol_ratio'] += 1
        if enable_logging:
            print(f"[regime_ctx] vol_ratio computation failed: {e}")
        vol_ratio = 0.0

    # --- 3) МИКРОСТРУКТУРА ---
    if micro_feats is not None and isinstance(micro_feats, dict):
        try:
            ofi15 = float(micro_feats.get("ofi_15s", 0.0))
            ofi_sign = _sign(ofi15)
            book_imb_raw = float(micro_feats.get("book_imb", 0.0))
            book_imb = _clip(book_imb_raw, -1.0, 1.0)
            
            if enable_logging:
                _FEATURE_STATS['ofi_sign'].append(ofi_sign)
                _FEATURE_STATS['book_imb'].append(book_imb)
                
        except Exception as e:
            _ERROR_COUNTS['microstructure'] += 1
            if enable_logging:
                print(f"[regime_ctx] microstructure computation failed: {e}")
            ofi_sign = 0.0
            book_imb = 0.0
    else:
        if enable_logging and micro_feats is None:
            _ERROR_COUNTS['microstructure_missing'] += 1
        ofi_sign = 0.0
        book_imb = 0.0

    # --- 4) ФЬЮЧЕРСЫ ---
    if fut_feats is not None and isinstance(fut_feats, dict):
        try:
            basis_raw = float(fut_feats.get("basis_now", 0.0))
            basis_sign = _sign(basis_raw)
            funding_raw = float(fut_feats.get("funding_sign", 0.0))
            funding_sign = _sign(funding_raw)
            
            if enable_logging:
                _FEATURE_STATS['basis_sign'].append(basis_sign)
                _FEATURE_STATS['funding_sign'].append(funding_sign)
                
        except Exception as e:
            _ERROR_COUNTS['futures'] += 1
            if enable_logging:
                print(f"[regime_ctx] futures computation failed: {e}")
            basis_sign = 0.0
            funding_sign = 0.0
    else:
        if enable_logging and fut_feats is None:
            _ERROR_COUNTS['futures_missing'] += 1
        basis_sign = 0.0
        funding_sign = 0.0

    # --- 5) СКАЧКИ ЦЕНЫ ---
    jf = 1.0 if bool(jump_flag) else 0.0
    
    if enable_logging:
        _FEATURE_STATS['jump_flag'].append(jf)

    return dict(
        trend_sign=trend_sign,
        trend_abs=trend_abs,
        vol_ratio=vol_ratio,
        jump_flag=jf,
        ofi_sign=ofi_sign,
        book_imb=book_imb,
        basis_sign=basis_sign,
        funding_sign=funding_sign,
    )


# ========== УПАКОВКА КОНТЕКСТА ==========

def pack_ctx(ctx: dict) -> Tuple[np.ndarray, list]:
    """
    Упаковывает контекстный словарь в numpy вектор с фиксированным порядком
    """
    names = [
        "trend_sign", "trend_abs", "vol_ratio", "jump_flag",
        "ofi_sign", "book_imb", "basis_sign", "funding_sign", "bias"
    ]
    
    vec = np.array([
        float(ctx.get("trend_sign", 0.0)),
        float(ctx.get("trend_abs", 0.0)),
        float(ctx.get("vol_ratio", 0.0)),
        float(ctx.get("jump_flag", 0.0)),
        float(ctx.get("ofi_sign", 0.0)),
        float(ctx.get("book_imb", 0.0)),
        float(ctx.get("basis_sign", 0.0)),
        float(ctx.get("funding_sign", 0.0)),
        1.0
    ], dtype=float)
    
    return vec, names


# ========== ОПРЕДЕЛЕНИЕ ФАЗ ==========

def phase_from_ctx(ctx: dict) -> int:
    """
    Определяет фазу рынка из контекста (6 фаз)
    
    Returns:
        Номер фазы [0-5]:
            0: bull_low   (бычий тренд, низкая вола)
            1: bull_high  (бычий тренд, высокая вола)
            2: bear_low   (медвежий тренд, низкая вола)
            3: bear_high  (медвежий тренд, высокая вола)
            4: flat_low   (флэт, низкая вола)
            5: flat_high  (флэт, высокая вола)
    """
    trend_sign = float(ctx.get("trend_sign", 0.0))
    vol_ratio = float(ctx.get("vol_ratio", 0.0))
    
    high_vol = (vol_ratio >= VOL_THRESHOLD)
    
    if trend_sign > 0:
        return 1 if high_vol else 0
    if trend_sign < 0:
        return 3 if high_vol else 2
    return 5 if high_vol else 4


# ========== ДИАГНОСТИКА И МОНИТОРИНГ ==========

def get_error_stats() -> dict:
    """Возвращает статистику ошибок при формировании контекста"""
    return dict(_ERROR_COUNTS)


def reset_error_stats() -> None:
    """Сбрасывает счетчики ошибок"""
    global _ERROR_COUNTS
    _ERROR_COUNTS.clear()


def get_feature_distributions() -> dict:
    """Возвращает статистику распределений контекстных фичей"""
    stats = {}
    
    for name, values in _FEATURE_STATS.items():
        if len(values) > 0:
            arr = np.array(values)
            stats[name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'n': len(values)
            }
        else:
            stats[name] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'n': 0}
    
    return stats


def reset_feature_stats() -> None:
    """Сбрасывает накопленную статистику фичей"""
    global _FEATURE_STATS
    _FEATURE_STATS.clear()


def validate_context(ctx: dict, strict: bool = False) -> Tuple[bool, list]:
    """
    Проверяет корректность сформированного контекста
    
    Returns:
        (is_valid, warnings)
    """
    warnings = []
    
    required = ['trend_sign', 'trend_abs', 'vol_ratio', 'jump_flag',
                'ofi_sign', 'book_imb', 'basis_sign', 'funding_sign']
    
    missing = [f for f in required if f not in ctx]
    if missing:
        warnings.append(f"Missing required fields: {missing}")
        return False, warnings
    
    if not (0.0 <= ctx['vol_ratio'] <= 5.0):
        warnings.append(f"vol_ratio out of range: {ctx['vol_ratio']}")
    
    if not (-1.0 <= ctx['book_imb'] <= 1.0):
        warnings.append(f"book_imb out of range: {ctx['book_imb']}")
    
    if ctx['trend_abs'] < 0.0 or ctx['trend_abs'] > 3.0:
        warnings.append(f"trend_abs out of range: {ctx['trend_abs']}")
    
    if ctx['jump_flag'] not in [0.0, 1.0]:
        warnings.append(f"jump_flag should be 0 or 1, got: {ctx['jump_flag']}")
    
    if strict:
        non_zero = sum(1 for k, v in ctx.items() if k != 'jump_flag' and abs(v) > 1e-6)
        if non_zero == 0:
            warnings.append("All features are zero - possible data source issue")
        
        if ctx['vol_ratio'] == 0.0:
            warnings.append("vol_ratio is zero - check ATR computation")
    
    is_valid = len([w for w in warnings if 'out of range' in w or 'Missing' in w]) == 0
    return is_valid, warnings


def format_context_summary(ctx: dict) -> str:
    """
    Форматирует контекст в человекочитаемую строку
    
    Пример:
        "Phase=2 Trend=↓(1.2σ) Vol=1.5x Jump=0 OFI=↓ Book=0.3 Basis=↑ Fund=↑"
    """
    phase = phase_from_ctx(ctx)
    
    trend_arrow = "↑" if ctx['trend_sign'] > 0 else ("↓" if ctx['trend_sign'] < 0 else "→")
    ofi_arrow = "↑" if ctx['ofi_sign'] > 0 else ("↓" if ctx['ofi_sign'] < 0 else "→")
    basis_arrow = "↑" if ctx['basis_sign'] > 0 else ("↓" if ctx['basis_sign'] < 0 else "→")
    fund_arrow = "↑" if ctx['funding_sign'] > 0 else ("↓" if ctx['funding_sign'] < 0 else "→")
    
    return (
        f"Phase={phase} "
        f"Trend={trend_arrow}({ctx['trend_abs']:.1f}σ) "
        f"Vol={ctx['vol_ratio']:.1f}x "
        f"Jump={int(ctx['jump_flag'])} "
        f"OFI={ofi_arrow} "
        f"Book={ctx['book_imb']:+.2f} "
        f"Basis={basis_arrow} "
        f"Fund={fund_arrow}"
    )


# ========== ЭКСПОРТ ==========

__all__ = [
    'build_regime_ctx',
    'pack_ctx',
    'phase_from_ctx',
    'get_error_stats',
    'reset_error_stats',
    'get_feature_distributions',
    'reset_feature_stats',
    'validate_context',
    'format_context_summary',
    'ema',
    'macd_hist',
    'resample_ohlc',
]
