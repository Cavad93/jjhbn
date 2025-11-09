#!/usr/bin/env python3
"""
BASE Logic - Multi-Timeframe Probability

Baseline предсказание вероятности роста/падения на основе:
- Multi-timeframe анализ (5m, 15m, 30m, 4h)
- 4 сигнала: M (Momentum), S (VWAP Slope), B (Breakout), R (Reversion)
- Weighted voting с адаптивной температурой

Перенесено из bnbusdrt6.py (полная реализация)

Usage:
    from features.base_logic import BaseLogicMultiTF

    base_logic = BaseLogicMultiTF()
    p_up, p_down = base_logic.predict(df)
"""

import numpy as np
import pandas as pd
import math
from typing import Dict, Tuple, Optional


# ============================================================================
# КОНСТАНТЫ
# ============================================================================

# Параметры Momentum
M1, M2, M3 = 1, 3, 5

# Range-based volatility
RB_LEN = 20

# ATR
ATR_LEN = 14
ATR_SMOOTH = 50

# VWAP Slope
VWAP_LOOK = 10

# Keltner Channels (Breakout)
KC_LEN = 20
KC_MULT = 2.0

# Bollinger Bands (Reversion)
BB_LEN = 20
BB_Z = 1.2  # Порог для реверсии

# Volume
VOL_LEN = 50
VOL_BOOST = 0.15

# Normalization
USE_LORENTZ = True
C_M, C_S, C_B, C_R = 2.5, 3.0, 2.0, 1.6


# ============================================================================
# PHASE DETECTION
# ============================================================================

# Market phase constants
PHASE_TRENDING = 1
PHASE_RANGING = 2

# Phase detection parameters
PHASE_ADX_PERIOD = 14
PHASE_TREND_THRESHOLD = 25  # ADX > 25 = trending


def detect_market_phase(df: pd.DataFrame) -> int:
    """
    Определяет фазу рынка: TRENDING (тренд) или RANGING (боковик)

    Использует упрощённый ADX-подобный индикатор:
    - Высокий trend strength = TRENDING
    - Низкий trend strength = RANGING

    Args:
        df: DataFrame с OHLCV данными

    Returns:
        PHASE_TRENDING или PHASE_RANGING
    """
    if len(df) < PHASE_ADX_PERIOD + 5:
        return PHASE_RANGING  # Default для малых данных

    # Вычисляем True Range
    tr = true_range(df)
    atr = rma(tr, PHASE_ADX_PERIOD)

    # Directional movement
    high_diff = df['high'] - df['high'].shift(1)
    low_diff = df['low'].shift(1) - df['low']

    pos_dm = pd.Series(0.0, index=df.index)
    neg_dm = pd.Series(0.0, index=df.index)

    # +DM when high_diff > low_diff and > 0
    pos_mask = (high_diff > low_diff) & (high_diff > 0)
    pos_dm[pos_mask] = high_diff[pos_mask]

    # -DM when low_diff > high_diff and > 0
    neg_mask = (low_diff > high_diff) & (low_diff > 0)
    neg_dm[neg_mask] = low_diff[neg_mask]

    # Smoothed directional indicators
    pos_di = 100 * rma(pos_dm, PHASE_ADX_PERIOD) / (atr + 1e-10)
    neg_di = 100 * rma(neg_dm, PHASE_ADX_PERIOD) / (atr + 1e-10)

    # DX = |+DI - -DI| / (+DI + -DI)
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-10)

    # ADX = smoothed DX
    adx = rma(dx, PHASE_ADX_PERIOD)

    # Классификация на основе последнего значения ADX
    current_adx = adx.iloc[-1] if len(adx) > 0 else 0

    if current_adx > PHASE_TREND_THRESHOLD:
        return PHASE_TRENDING
    else:
        return PHASE_RANGING


def get_phase_modulation(phase: int) -> np.ndarray:
    """
    Возвращает модуляционные множители для весов [M, S, B, R] в зависимости от фазы

    Эти множители УМНОЖАЮТСЯ на калиброванные веса, а не заменяют их!

    Args:
        phase: PHASE_TRENDING или PHASE_RANGING

    Returns:
        Модуляционные множители [M, S, B, R]
    """
    if phase == PHASE_TRENDING:
        # Тренд: усиливаем Momentum и Breakout, ослабляем Reversion
        return np.array([1.3, 1.0, 1.2, 0.6], dtype=float)
    else:  # PHASE_RANGING
        # Боковик: усиливаем Reversion, ослабляем Momentum
        return np.array([0.7, 1.0, 0.8, 1.4], dtype=float)


# ============================================================================
# ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
# ============================================================================

def ema(series: pd.Series, n: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=n, adjust=False).mean()


def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(n, min_periods=1).mean()


def stdev(series: pd.Series, n: int) -> pd.Series:
    """Standard Deviation"""
    return series.rolling(n, min_periods=1).std(ddof=0)


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range"""
    prev_close = df["close"].shift(1)
    tr = pd.DataFrame({
        "hl": df["high"] - df["low"],
        "hc": (df["high"] - prev_close).abs(),
        "lc": (df["low"] - prev_close).abs()
    }).max(axis=1)
    return tr


def rma(x: pd.Series, n: int) -> pd.Series:
    """Wilder's Smoothing (RMA)"""
    alpha = 1.0 / float(n)
    r = x.copy()
    if len(x) == 0:
        return x
    r.iloc[0] = x.iloc[:n].mean() if len(x) >= n else x.iloc[0]
    for i in range(1, len(x)):
        r.iloc[i] = alpha * x.iloc[i] + (1 - alpha) * r.iloc[i-1]
    return r


def atr_wilder(df: pd.DataFrame, n: int) -> pd.Series:
    """ATR с Wilder's smoothing"""
    return rma(true_range(df), n)


def session_vwap(df: pd.DataFrame, src: pd.Series) -> pd.Series:
    """Session VWAP (сбрасывается каждый день UTC)"""
    try:
        # Пробуем работать с timezone-aware индексом
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            day = df.index.tz_convert("UTC").date
        else:
            day = df.index.date
    except (TypeError, AttributeError):
        # Fallback: просто используем дату без timezone
        day = df.index.date

    grp = pd.Series(day, index=df.index)
    pv = (src * df["volume"]).groupby(grp).cumsum()
    vv = (df["volume"]).groupby(grp).cumsum().replace(0.0, np.nan)
    vwap = (pv / vv).ffill()
    return vwap


def rolling_vwap(df: pd.DataFrame, src: pd.Series, period: int) -> pd.Series:
    """
    Rolling VWAP с фиксированным окном

    Для высоких таймфреймов (4h) более подходит, чем session VWAP,
    так как даёт постоянный lookback период вместо сброса каждый день

    Args:
        df: DataFrame с OHLCV
        src: Цена для VWAP (обычно типичная цена)
        period: Количество свечей для расчета (например, 24 свечи = 4 дня на 4h)

    Returns:
        Rolling VWAP Series
    """
    pv = (src * df["volume"]).rolling(window=period, min_periods=1).sum()
    vv = df["volume"].rolling(window=period, min_periods=1).sum()
    vwap = (pv / vv.replace(0.0, np.nan)).ffill()
    return vwap


def detect_timeframe(df: pd.DataFrame) -> str:
    """
    Определяет таймфрейм DataFrame по интервалу между свечами

    Args:
        df: DataFrame с DatetimeIndex

    Returns:
        Строка таймфрейма: '5m', '15m', '30m', '4h', etc.
    """
    if len(df) < 2:
        return '5m'  # Default

    # Вычисляем средний интервал между свечами
    time_diffs = df.index[1:] - df.index[:-1]
    avg_diff = pd.Timedelta(time_diffs.mean())

    # Классифицируем
    if avg_diff <= pd.Timedelta(minutes=5):
        return '5m'
    elif avg_diff <= pd.Timedelta(minutes=15):
        return '15m'
    elif avg_diff <= pd.Timedelta(minutes=30):
        return '30m'
    elif avg_diff <= pd.Timedelta(hours=1):
        return '1h'
    elif avg_diff <= pd.Timedelta(hours=4):
        return '4h'
    else:
        return '1d'


# ============================================================================
# НОРМАЛИЗАЦИЯ И SOFTMAX
# ============================================================================

def lorentz(x: np.ndarray, c: float) -> np.ndarray:
    """Lorentz normalization (более плавная чем tanh)"""
    return x / (1.0 + (x / c) ** 2)


def _tanh(x: np.ndarray) -> np.ndarray:
    """Tanh normalization"""
    return np.tanh(x)


def norm_feat(x: np.ndarray, gain: float, c: float, use_lorentz: bool) -> np.ndarray:
    """Нормализация фич"""
    return lorentz(x, c) if use_lorentz else _tanh(gain * x)


def softmax2_with_temperature(z_up: float, z_dn: float, temperature: float = 1.0) -> Tuple[float, float]:
    """
    Softmax с температурным коэффициентом:
    - T > 1: более «размытые» вероятности (консервативнее)
    - T < 1: более «острые» вероятности (агрессивнее)
    - T = 1: обычный softmax
    """
    z_up_scaled = z_up / max(1e-3, float(temperature))
    z_dn_scaled = z_dn / max(1e-3, float(temperature))

    m = max(z_up_scaled, z_dn_scaled)
    e_up = math.exp(z_up_scaled - m)
    e_dn = math.exp(z_dn_scaled - m)
    s = e_up + e_dn

    return e_up / s, e_dn / s


def adaptive_temperature(volAmp: float, atr_norm: float) -> float:
    """
    Адаптивный выбор температуры на основе волатильности:
    - Высокая волатильность → T > 1 (консервативнее)
    - Низкая волатильность → T ≈ 1 (обычно)
    - Очень низкая → T < 1 (агрессивнее)
    """
    volAmp = max(0.5, min(2.0, float(volAmp)))
    atr_norm = max(0.3, min(3.0, float(atr_norm)))

    if atr_norm > 1.5 or volAmp > 1.4:
        return 1.2  # более размытые вероятности
    elif atr_norm < 0.7 and volAmp < 1.0:
        return 0.9  # более острые вероятности
    else:
        return 1.0  # дефолт


def adaptive_kc_multiplier(atr_norm: float) -> float:
    """
    Адаптивный множитель для Keltner Channels на основе волатильности

    Идея: В высокую волатильность расширяем каналы, чтобы избежать ложных пробоев
    В низкую волатильность сужаем каналы для более чувствительного обнаружения пробоев

    Args:
        atr_norm: Нормализованный ATR (atr / atr_sma)

    Returns:
        KC_MULT: Множитель для Keltner Channels (default 2.0)
    """
    atr_norm = max(0.3, min(3.0, float(atr_norm)))

    if atr_norm > 1.5:
        # Высокая волатильность → расширяем каналы (2.5-3.0)
        return 2.0 + (atr_norm - 1.5) * 1.0  # Linear scaling
    elif atr_norm < 0.7:
        # Низкая волатильность → сужаем каналы (1.5-2.0)
        return 2.0 - (0.7 - atr_norm) * 0.8
    else:
        # Нормальная волатильность → дефолт
        return 2.0


def adaptive_bb_threshold(phase: int, atr_norm: float) -> float:
    """
    Адаптивный порог для Bollinger Bands mean reversion

    Идея:
    - В трендовой фазе: выше порог (меньше ложных реверсий)
    - В боковике: ниже порог (больше возможностей для реверсии)
    - Высокая волатильность: выше порог (избегаем шума)

    Args:
        phase: PHASE_TRENDING или PHASE_RANGING
        atr_norm: Нормализованный ATR

    Returns:
        BB_Z: Порог для срабатывания реверсии (default 1.2)
    """
    atr_norm = max(0.3, min(3.0, float(atr_norm)))

    # Базовый порог зависит от фазы
    if phase == PHASE_TRENDING:
        base_threshold = 1.5  # Выше в тренде
    else:  # PHASE_RANGING
        base_threshold = 1.0  # Ниже в боковике

    # Корректируем на волатильность
    if atr_norm > 1.5:
        # Высокая волатильность → увеличиваем порог
        return base_threshold + (atr_norm - 1.5) * 0.4
    elif atr_norm < 0.7:
        # Низкая волатильность → уменьшаем порог
        return base_threshold - (0.7 - atr_norm) * 0.3
    else:
        return base_threshold


# ============================================================================
# РАСЧЕТ СИГНАЛОВ
# ============================================================================

def features_from_binance(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Вычисляет 4 сигнала (M, S, B, R) из OHLCV данных:

    M (Momentum): Weighted momentum с тремя периодами
    S (VWAP Slope): Наклон VWAP
    B (Breakout): Keltner Channels breakout (ADAPTIVE KC_MULT)
    R (Reversion): Bollinger Bands mean reversion (ADAPTIVE BB_Z)

    Args:
        df: DataFrame с колонками ['open', 'high', 'low', 'close', 'volume']
            и DatetimeIndex

    Returns:
        dict: Словарь с сигналами и доп. фичами
    """
    # === PHASE & TIMEFRAME DETECTION ===
    # Вычисляем заранее для адаптивных параметров
    timeframe = detect_timeframe(df)
    phase = detect_market_phase(df)

    # Range-based volatility normalization
    ln_hl = np.log(df["high"] / df["low"]).clip(lower=1e-12)
    sigP = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (ln_hl ** 2))

    ln_co = np.log(df["close"] / df["open"]).fillna(0.0)
    sigGK = np.sqrt(np.maximum(0.0, 0.5 * (ln_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (ln_co ** 2)))

    ln_hc = np.log(df["high"] / df["close"]).clip(lower=1e-12)
    ln_ho = np.log(df["high"] / df["open"]).clip(lower=1e-12)
    ln_lc = np.log(df["low"] / df["close"]).clip(lower=1e-12)
    ln_lo = np.log(df["low"] / df["open"]).clip(lower=1e-12)
    rsVar = ln_hc * ln_ho + ln_lc * ln_lo
    sigRS = np.sqrt(np.maximum(0.0, rsVar))

    sigRB = pd.Series((sigP + sigGK + sigRS) / 3.0, index=df.index)
    sigRB = ema(sigRB, RB_LEN)
    normGain = 1.0 / np.maximum(sigRB.values, 1e-10)

    # ATR (нужен для adaptive параметров)
    atr_series = atr_wilder(df, ATR_LEN)
    atr_sma = sma(atr_series, ATR_SMOOTH)

    # === ADAPTIVE PARAMETERS ===
    # Вычисляем адаптивные параметры на основе текущей волатильности
    atr_norm_last = float(atr_series.iloc[-1] / (atr_sma.iloc[-1] + 1e-12))
    kc_mult_adaptive = adaptive_kc_multiplier(atr_norm_last)
    bb_z_adaptive = adaptive_bb_threshold(phase, atr_norm_last)

    # === M (Momentum) ===
    r1 = np.log(df["close"] / df["close"].shift(M1)).fillna(0.0)
    r2 = np.log(df["close"] / df["close"].shift(M2)).fillna(0.0)
    r3 = np.log(df["close"] / df["close"].shift(M3)).fillna(0.0)
    Mraw = 0.6 * r1 + 0.3 * r2 + 0.1 * r3
    M_up = norm_feat(Mraw.values * normGain, 2.5, C_M, USE_LORENTZ)
    M_dn = norm_feat(-Mraw.values * normGain, 2.5, C_M, USE_LORENTZ)

    # === S (VWAP Slope) - ADAPTIVE ===
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    if timeframe == '4h':
        # Rolling VWAP для 4h (24 свечи = 4 дня)
        vwap = rolling_vwap(df, tp, period=24)
    else:
        # Session VWAP для более низких таймфреймов
        vwap = session_vwap(df, tp)
    vslp = ((vwap - vwap.shift(VWAP_LOOK)) / vwap.replace(0.0, np.nan)).fillna(0.0)
    S_up = norm_feat(vslp.values * normGain, 3.0, C_S, USE_LORENTZ)
    S_dn = norm_feat(-vslp.values * normGain, 3.0, C_S, USE_LORENTZ)

    # === B (Breakout - Keltner) - ADAPTIVE KC_MULT ===
    basisKC = ema(df["close"], KC_LEN)
    rngKC = atr_wilder(df, KC_LEN)
    upKC = basisKC + kc_mult_adaptive * rngKC
    dnKC = basisKC - kc_mult_adaptive * rngKC
    distUp = ((df["close"] - upKC) / (kc_mult_adaptive * rngKC.replace(0, np.nan))).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    distDn = ((dnKC - df["close"]) / (kc_mult_adaptive * rngKC.replace(0, np.nan))).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    B_up = norm_feat(distUp.values, 2.0, C_B, USE_LORENTZ)
    B_dn = norm_feat(distDn.values, 2.0, C_B, USE_LORENTZ)

    # === R (Reversion - Bollinger) - ADAPTIVE BB_Z ===
    bb_basis = sma(df["close"], BB_LEN)
    bb_dev = stdev(df["close"], BB_LEN).replace(0.0, np.nan)
    Zs = ((df["close"] - bb_basis) / bb_dev).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    R_up = norm_feat(np.maximum(0.0, -Zs - bb_z_adaptive).values, 1.6, C_R, USE_LORENTZ)
    R_dn = norm_feat(np.maximum(0.0,  Zs - bb_z_adaptive).values, 1.6, C_R, USE_LORENTZ)

    # === Volume Amplitude ===
    vol_usd = df["volume"] * df["close"]
    vMean = sma(vol_usd, VOL_LEN)
    vStd = stdev(vol_usd, VOL_LEN).replace(0.0, np.nan)
    volZ = ((vol_usd - vMean) / vStd).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    volAmp = np.clip(1.0 + VOL_BOOST * np.maximum(0.0, volZ.values), 0.8, 1.8)

    return dict(
        M_up=pd.Series(M_up, index=df.index),
        M_dn=pd.Series(M_dn, index=df.index),
        S_up=pd.Series(S_up, index=df.index),
        S_dn=pd.Series(S_dn, index=df.index),
        B_up=pd.Series(B_up, index=df.index),
        B_dn=pd.Series(B_dn, index=df.index),
        R_up=pd.Series(R_up, index=df.index),
        R_dn=pd.Series(R_dn, index=df.index),
        atr=pd.Series(atr_series, index=df.index),
        atr_sma=pd.Series(atr_sma, index=df.index),
        volAmp=pd.Series(volAmp, index=df.index),
        Zs=Zs,
        phase=phase,  # Фаза рынка для phase-adaptive весов
        timeframe=timeframe,  # Таймфрейм для отладки
    )


def _index_pad(series: pd.Series, t: pd.Timestamp) -> Optional[int]:
    """Находит индекс ближайшего элемента (pad)"""
    idx = series.index.get_indexer([t], method="pad")
    return None if idx[0] == -1 else int(idx[0])


def prob_up_down_at_time(
    feats: Dict[str, pd.Series],
    tstamp: pd.Timestamp,
    w_dyn: Optional[np.ndarray] = None,
    volAmp: Optional[float] = None,
    atr_norm: Optional[float] = None
) -> Tuple[float, float, Dict]:
    """
    Вычисляет вероятности роста/падения на основе 4 сигналов

    Args:
        feats: Словарь с сигналами (из features_from_binance)
        tstamp: Временная метка
        w_dyn: Динамические веса [M, S, B, R] (опционально)
        volAmp: Volume amplitude (опционально, извлекается из feats)
        atr_norm: ATR normalization (опционально, извлекается из feats)

    Returns:
        (P_up, P_dn, context): Вероятности и контекст для Walk-Forward
    """
    i = _index_pad(feats["M_up"], tstamp)
    if i is None:
        return 0.5, 0.5, {}

    M_up = float(feats["M_up"].iloc[i])
    M_dn = float(feats["M_dn"].iloc[i])
    S_up = float(feats["S_up"].iloc[i])
    S_dn = float(feats["S_dn"].iloc[i])
    B_up = float(feats["B_up"].iloc[i])
    B_dn = float(feats["B_dn"].iloc[i])
    R_up = float(feats["R_up"].iloc[i])
    R_dn = float(feats["R_dn"].iloc[i])

    if volAmp is None:
        volAmp = float(feats["volAmp"].iloc[i])
    if atr_norm is None:
        atr_norm = float(feats["atr"].iloc[i] / (feats["atr_sma"].iloc[i] + 1e-12))

    # Веса сигналов
    w_default = np.array([0.35, 0.20, 0.20, 0.25], dtype=float)  # [M, S, B, R]

    if w_dyn is None or len(w_dyn) != 4:
        w = w_default
    else:
        w = np.array(w_dyn, dtype=float)
        w_norm = np.linalg.norm(w)

        # Плавная интерполяция при усушке
        if w_norm < 0.15:
            alpha_shrink = max(0.0, w_norm / 0.15)
            w = alpha_shrink * w + (1 - alpha_shrink) * w_default
            w_norm = np.linalg.norm(w)

        # L1-нормализация если веса сбились
        w_sum = np.sum(np.abs(w))
        if w_sum < 0.8 or w_sum > 1.5:
            w = w / (w_sum + 1e-12)

    # === PHASE-ADAPTIVE WEIGHT MODULATION ===
    # Применяем модуляцию весов в зависимости от фазы рынка
    # Это НЕ заменяет калибровку, а дополняет её!
    phase = feats.get('phase', PHASE_RANGING)  # Default to ranging if not present
    phase_mod = get_phase_modulation(phase)

    # Применяем модуляцию
    w_modulated = w * phase_mod

    # Ре-нормализуем (L1 norm = 1)
    w_sum_mod = np.sum(np.abs(w_modulated))
    if w_sum_mod > 1e-8:
        w = w_modulated / w_sum_mod
    # Если w_sum_mod слишком мал, оставляем w без изменений

    w_mom, w_vwp, w_brk, w_rev = w

    # Линейная комбинация сигналов
    Z_up = (w_mom * M_up * volAmp) + (w_vwp * S_up) + (w_brk * B_up) + (w_rev * R_up)
    Z_dn = (w_mom * M_dn * volAmp) + (w_vwp * S_dn) + (w_brk * B_dn) + (w_rev * R_dn)

    # Softmax с адаптивной температурой
    T = adaptive_temperature(volAmp, atr_norm)
    P_up, P_dn = softmax2_with_temperature(Z_up, Z_dn, temperature=T)

    # Контекст для Walk-Forward learning
    phi_wf = np.array([
        (M_up - M_dn) * volAmp,
        (S_up - S_dn),
        (B_up - B_dn),
        (R_up - R_dn)
    ], dtype=float)

    return P_up, P_dn, {
        "phi_wf0": phi_wf[0],
        "phi_wf1": phi_wf[1],
        "phi_wf2": phi_wf[2],
        "phi_wf3": phi_wf[3],
        "temperature": T
    }


# ============================================================================
# MAIN CLASS
# ============================================================================

class BaseLogicMultiTF:
    """
    Baseline предсказание на основе multi-timeframe анализа

    Полная реализация с 4 сигналами (M, S, B, R)
    """

    def __init__(self):
        """Инициализация BASE logic"""
        self.w_dyn = None  # Динамические веса (могут обновляться через Walk-Forward)

    def predict(self, df: pd.DataFrame, timestamp: Optional[pd.Timestamp] = None) -> Tuple[float, float]:
        """
        Предсказание вероятности роста/падения

        Args:
            df: DataFrame с OHLCV данными
            timestamp: Временная метка (если None, используется последняя)

        Returns:
            (p_up, p_down): Вероятности роста и падения
        """
        if timestamp is None:
            timestamp = df.index[-1]

        # Вычисляем сигналы
        feats = features_from_binance(df)

        # Получаем вероятности
        p_up, p_down, context = prob_up_down_at_time(feats, timestamp, self.w_dyn)

        return p_up, p_down

    def predict_from_features(self, features: np.ndarray) -> float:
        """
        FALLBACK: Возвращает нейтральное предсказание

        Этот метод используется, когда у нас есть только 68D feature vector
        без доступа к raw OHLCV данным

        Args:
            features: 68D feature vector

        Returns:
            p_up: Вероятность роста (0.5 - нейтрально)
        """
        # TODO: Извлечь сигналы из 68D features если возможно
        return 0.5

    def predict_from_experts(
        self,
        p_xgb: float,
        p_rf: float,
        p_arf: float,
        p_nn: float
    ) -> float:
        """
        FALLBACK: Возвращает среднее экспертов

        Args:
            p_xgb: XGBoost prediction
            p_rf: RandomForest prediction
            p_arf: AdaptiveRF prediction
            p_nn: NeuralNet prediction

        Returns:
            p_base: Среднее экспертов
        """
        preds = [p_xgb, p_rf, p_arf, p_nn]
        valid_preds = [p for p in preds if p is not None and 0 <= p <= 1]

        if not valid_preds:
            return 0.5

        return float(np.mean(valid_preds))

    def set_weights(self, w: np.ndarray):
        """Устанавливает динамические веса сигналов"""
        if len(w) == 4:
            self.w_dyn = np.array(w, dtype=float)

    def get_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Вычисляет и возвращает все сигналы"""
        return features_from_binance(df)


# Alias для совместимости
BaseLogic = BaseLogicMultiTF
