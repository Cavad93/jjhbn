#!/usr/bin/env python3
"""
Технические индикаторы на чистом pandas/numpy (БЕЗ ta-lib)

Все индикаторы реализованы с нуля для полной совместимости с Windows
и устранения зависимости от ta-lib C библиотеки.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union


def ema(data: Union[np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Экспоненциальная скользящая средняя (EMA)

    Args:
        data: Ценовые данные
        period: Период

    Returns:
        np.ndarray: EMA значения
    """
    if isinstance(data, pd.Series):
        data = data.values

    ema_vals = np.zeros_like(data, dtype=float)
    if len(data) == 0:
        return ema_vals

    # Первое значение = SMA
    ema_vals[0] = data[0]

    # Коэффициент сглаживания
    alpha = 2.0 / (period + 1.0)

    # Вычисление EMA
    for i in range(1, len(data)):
        ema_vals[i] = data[i] * alpha + ema_vals[i-1] * (1 - alpha)

    return ema_vals


def sma(data: Union[np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Простая скользящая средняя (SMA)

    Args:
        data: Ценовые данные
        period: Период

    Returns:
        np.ndarray: SMA значения
    """
    if isinstance(data, pd.Series):
        data = data.values

    return pd.Series(data).rolling(window=period, min_periods=1).mean().values


def rsi(close: Union[np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
    """
    Relative Strength Index (RSI)

    Args:
        close: Цены закрытия
        period: Период (по умолчанию 14)

    Returns:
        np.ndarray: RSI значения (0-100)
    """
    if isinstance(close, pd.Series):
        close = close.values

    # Вычисление изменений цен
    deltas = np.diff(close, prepend=close[0])

    # Разделение на прирост и убыль
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Первое среднее = SMA
    avg_gain = np.zeros_like(close, dtype=float)
    avg_loss = np.zeros_like(close, dtype=float)

    if len(close) >= period:
        avg_gain[period-1] = np.mean(gains[:period])
        avg_loss[period-1] = np.mean(losses[:period])

        # EMA для остальных значений
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period

    # Вычисление RS и RSI
    rs = np.divide(avg_gain, avg_loss, where=(avg_loss != 0), out=np.ones_like(avg_gain))
    rsi_vals = 100.0 - (100.0 / (1.0 + rs))

    # Заполнение начальных значений
    rsi_vals[:period] = 50.0

    return rsi_vals


def macd(
    close: Union[np.ndarray, pd.Series],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Moving Average Convergence Divergence (MACD)

    Args:
        close: Цены закрытия
        fast_period: Быстрый период (по умолчанию 12)
        slow_period: Медленный период (по умолчанию 26)
        signal_period: Период сигнальной линии (по умолчанию 9)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (MACD, Signal, Histogram)
    """
    if isinstance(close, pd.Series):
        close = close.values

    # Вычисление EMA
    ema_fast = ema(close, fast_period)
    ema_slow = ema(close, slow_period)

    # MACD линия
    macd_line = ema_fast - ema_slow

    # Сигнальная линия
    signal_line = ema(macd_line, signal_period)

    # Гистограмма
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def stochastic(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    k_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator

    Args:
        high: Максимальные цены
        low: Минимальные цены
        close: Цены закрытия
        k_period: Период для %K (по умолчанию 14)
        k_smooth: Сглаживание для %K (по умолчанию 3)
        d_smooth: Сглаживание для %D (по умолчанию 3)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (%K, %D)
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    # Вычисление %K
    lowest_low = pd.Series(low).rolling(window=k_period, min_periods=1).min().values
    highest_high = pd.Series(high).rolling(window=k_period, min_periods=1).max().values

    # Избежание деления на ноль
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1.0, range_hl)

    fast_k = 100.0 * (close - lowest_low) / range_hl

    # Сглаживание %K
    slow_k = sma(fast_k, k_smooth)

    # Сглаживание %D
    slow_d = sma(slow_k, d_smooth)

    return slow_k, slow_d


def atr(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    period: int = 14
) -> np.ndarray:
    """
    Average True Range (ATR)

    Args:
        high: Максимальные цены
        low: Минимальные цены
        close: Цены закрытия
        period: Период (по умолчанию 14)

    Returns:
        np.ndarray: ATR значения
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    # True Range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    # ATR = EMA of True Range
    atr_vals = np.zeros_like(close, dtype=float)

    if len(true_range) >= period:
        # Первое значение = SMA
        atr_vals[period-1] = np.mean(true_range[:period])

        # Остальные = EMA
        alpha = 1.0 / period
        for i in range(period, len(true_range)):
            atr_vals[i] = true_range[i] * alpha + atr_vals[i-1] * (1 - alpha)

    return atr_vals


def bollinger_bands(
    close: Union[np.ndarray, pd.Series],
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands

    Args:
        close: Цены закрытия
        period: Период (по умолчанию 20)
        std_dev: Количество стандартных отклонений (по умолчанию 2.0)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (Upper, Middle, Lower)
    """
    if isinstance(close, pd.Series):
        close = close.values

    # Средняя линия (SMA)
    middle = sma(close, period)

    # Стандартное отклонение
    std = pd.Series(close).rolling(window=period, min_periods=1).std().values

    # Верхняя и нижняя полосы
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


def adx(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    period: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average Directional Index (ADX) с +DI и -DI

    Args:
        high: Максимальные цены
        low: Минимальные цены
        close: Цены закрытия
        period: Период (по умолчанию 14)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (ADX, +DI, -DI)
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    # True Range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    # Directional Movement
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]

    plus_dm = np.where((high - prev_high) > (prev_low - low),
                       np.maximum(high - prev_high, 0), 0)
    minus_dm = np.where((prev_low - low) > (high - prev_high),
                        np.maximum(prev_low - low, 0), 0)

    # Smoothed TR, +DM, -DM
    atr_vals = np.zeros_like(close, dtype=float)
    plus_dm_smooth = np.zeros_like(close, dtype=float)
    minus_dm_smooth = np.zeros_like(close, dtype=float)

    if len(close) >= period:
        # Первые значения = сумма
        atr_vals[period-1] = np.sum(true_range[:period])
        plus_dm_smooth[period-1] = np.sum(plus_dm[:period])
        minus_dm_smooth[period-1] = np.sum(minus_dm[:period])

        # Сглаживание Wilder
        for i in range(period, len(close)):
            atr_vals[i] = atr_vals[i-1] - (atr_vals[i-1] / period) + true_range[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]

    # +DI и -DI
    plus_di = 100 * np.divide(plus_dm_smooth, atr_vals,
                              where=(atr_vals != 0),
                              out=np.zeros_like(atr_vals))
    minus_di = 100 * np.divide(minus_dm_smooth, atr_vals,
                               where=(atr_vals != 0),
                               out=np.zeros_like(atr_vals))

    # DX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = 100 * np.divide(di_diff, di_sum,
                         where=(di_sum != 0),
                         out=np.zeros_like(di_sum))

    # ADX = EMA of DX
    adx_vals = np.zeros_like(close, dtype=float)

    if len(close) >= period * 2:
        # Первое значение ADX = среднее DX
        adx_vals[period*2-2] = np.mean(dx[period-1:period*2-1])

        # Сглаживание
        for i in range(period*2-1, len(close)):
            adx_vals[i] = (adx_vals[i-1] * (period - 1) + dx[i]) / period

    return adx_vals, plus_di, minus_di


def williams_r(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    period: int = 14
) -> np.ndarray:
    """
    Williams %R

    Args:
        high: Максимальные цены
        low: Минимальные цены
        close: Цены закрытия
        period: Период (по умолчанию 14)

    Returns:
        np.ndarray: Williams %R значения (-100 до 0)
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    # Максимум и минимум за период
    highest_high = pd.Series(high).rolling(window=period, min_periods=1).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=1).min().values

    # Williams %R
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1.0, range_hl)

    willr = -100.0 * (highest_high - close) / range_hl

    return willr


def cci(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    period: int = 20
) -> np.ndarray:
    """
    Commodity Channel Index (CCI)

    Args:
        high: Максимальные цены
        low: Минимальные цены
        close: Цены закрытия
        period: Период (по умолчанию 20)

    Returns:
        np.ndarray: CCI значения
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    # Typical Price
    typical_price = (high + low + close) / 3.0

    # SMA of Typical Price
    sma_tp = sma(typical_price, period)

    # Mean Deviation
    mad = pd.Series(typical_price).rolling(window=period, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values

    # CCI
    mad = np.where(mad == 0, 1.0, mad)
    cci_vals = (typical_price - sma_tp) / (0.015 * mad)

    return cci_vals


def roc(
    close: Union[np.ndarray, pd.Series],
    period: int = 10
) -> np.ndarray:
    """
    Rate of Change (ROC)

    Args:
        close: Цены закрытия
        period: Период (по умолчанию 10)

    Returns:
        np.ndarray: ROC значения (в процентах)
    """
    if isinstance(close, pd.Series):
        close = close.values

    # ROC = (close - close[n]) / close[n] * 100
    prev_close = np.roll(close, period)
    prev_close[:period] = close[0]

    roc_vals = np.divide(
        (close - prev_close),
        prev_close,
        where=(prev_close != 0),
        out=np.zeros_like(close, dtype=float)
    ) * 100.0

    # Заполнение начальных значений
    roc_vals[:period] = 0.0

    return roc_vals
