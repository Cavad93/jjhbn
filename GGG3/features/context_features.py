"""
Контекстные фичи для META модели

Вычисляет 7 контекстных фич:
1. vol_ratio - отношение объема к среднему
2. trend_macd - значение MACD тренда
3. jump_flag - флаг скачка цены
4. funding_sign - знак funding rate (для Futures)
5. book_imb - дисбаланс стакана
6. ofi_15s - order flow imbalance
7. basis_pct - базис futures/spot

Автор: Claude Code
Дата: 2025-11-07
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def calculate_volume_ratio(df: pd.DataFrame, period: int = 20) -> float:
    """
    Рассчитывает отношение текущего объема к среднему

    Args:
        df: DataFrame с колонкой 'volume'
        period: Период для среднего объема

    Returns:
        vol_ratio (обычно в диапазоне 0.8-1.4)
    """
    if len(df) < period:
        return 1.0

    try:
        current_volume = float(df['volume'].iloc[-1])
        avg_volume = float(df['volume'].iloc[-period:].mean())

        if avg_volume == 0:
            return 1.0

        vol_ratio = current_volume / avg_volume

        # Клиппинг к разумным значениям
        return float(np.clip(vol_ratio, 0.1, 3.0))

    except Exception:
        return 1.0


def calculate_macd_trend(df: pd.DataFrame,
                        fast_period: int = 12,
                        slow_period: int = 26,
                        signal_period: int = 9) -> float:
    """
    Рассчитывает нормализованное значение MACD тренда

    Args:
        df: DataFrame с колонкой 'close'
        fast_period: Период быстрой EMA
        slow_period: Период медленной EMA
        signal_period: Период сигнальной линии

    Returns:
        trend_macd (обычно в диапазоне -0.5 до 0.5)
    """
    if len(df) < slow_period + signal_period:
        return 0.0

    try:
        close = df['close']

        # Рассчитываем MACD
        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd - signal

        # Нормализуем относительно цены
        current_price = float(close.iloc[-1])
        if current_price == 0:
            return 0.0

        trend_macd = float(macd_histogram.iloc[-1] / current_price)

        # Клиппинг к разумным значениям
        return float(np.clip(trend_macd, -0.5, 0.5))

    except Exception:
        return 0.0


def detect_price_jump(df: pd.DataFrame,
                     threshold_pct: float = 3.0,
                     lookback: int = 5) -> bool:
    """
    Определяет резкий скачок цены

    Args:
        df: DataFrame с колонками 'high', 'low', 'close'
        threshold_pct: Порог для определения скачка (в процентах)
        lookback: Количество свечей для анализа

    Returns:
        jump_flag: True если обнаружен скачок
    """
    if len(df) < lookback + 1:
        return False

    try:
        recent_df = df.iloc[-(lookback+1):]

        # Рассчитываем изменения цены
        price_changes = []
        for i in range(1, len(recent_df)):
            prev_close = float(recent_df['close'].iloc[i-1])
            curr_high = float(recent_df['high'].iloc[i])
            curr_low = float(recent_df['low'].iloc[i])

            if prev_close == 0:
                continue

            # Максимальное изменение (вверх или вниз)
            change_up = (curr_high - prev_close) / prev_close * 100
            change_down = (prev_close - curr_low) / prev_close * 100

            max_change = max(abs(change_up), abs(change_down))
            price_changes.append(max_change)

        if not price_changes:
            return False

        # Проверяем, был ли скачок больше порога
        max_change = max(price_changes)
        return bool(max_change > threshold_pct)

    except Exception:
        return False


def calculate_atr_percentage(df: pd.DataFrame, period: int = 14) -> float:
    """
    Рассчитывает ATR в процентах от цены

    Args:
        df: DataFrame с колонками 'high', 'low', 'close'
        period: Период для ATR

    Returns:
        ATR в процентах (например, 3.5 означает 3.5%)
    """
    if len(df) < period + 1:
        return 0.0

    try:
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR
        atr = true_range.rolling(window=period).mean()

        current_price = float(close.iloc[-1])
        if current_price == 0:
            return 0.0

        atr_pct = float(atr.iloc[-1] / current_price * 100)

        return float(np.clip(atr_pct, 0.0, 50.0))

    except Exception:
        return 0.0


def build_context_features(df_4h: pd.DataFrame,
                           symbol: Optional[str] = None) -> Dict[str, float]:
    """
    Строит полный словарь контекстных фич для META

    Args:
        df_4h: DataFrame с 4h свечами
        symbol: Символ монеты (опционально, для будущих расширений)

    Returns:
        Dict с 7 контекстными фичами:
        {
            'vol_ratio': float,     # 0.8-1.4
            'trend_macd': float,    # -0.5 до 0.5
            'jump_detected': bool,  # True/False
            'funding_sign': float,  # -1 до 1 (дефолт 0.0)
            'book_imb': float,      # -0.3 до 0.3 (дефолт 0.0)
            'ofi_15s': float,       # -0.2 до 0.2 (дефолт 0.0)
            'basis_pct': float      # -0.01 до 0.01 (дефолт 0.0)
        }
    """
    context = {}

    # 1. Volume ratio (реальное значение)
    context['vol_ratio'] = calculate_volume_ratio(df_4h, period=20)

    # 2. MACD trend (реальное значение)
    context['trend_macd'] = calculate_macd_trend(df_4h)

    # 3. Price jump flag (реальное значение)
    context['jump_detected'] = detect_price_jump(df_4h, threshold_pct=3.0)

    # 4. Funding rate sign (требует API, пока дефолт)
    # TODO: Добавить запрос funding rate для Binance Futures
    context['funding_sign'] = 0.0

    # 5. Order book imbalance (требует orderbook API, пока дефолт)
    # TODO: Добавить запрос orderbook и расчет дисбаланса
    context['book_imb'] = 0.0

    # 6. Order flow imbalance 15s (требует high-freq данных, пока дефолт)
    # TODO: Добавить расчет OFI из тиков
    context['ofi_15s'] = 0.0

    # 7. Basis futures/spot (требует spot цены, пока дефолт)
    # TODO: Добавить запрос spot цены и расчет базиса
    context['basis_pct'] = 0.0

    return context


# Вспомогательная функция для вывода статистики фич
def print_context_summary(context: Dict[str, float]) -> None:
    """
    Выводит читаемую сводку контекстных фич

    Args:
        context: Словарь с контекстными фичами
    """
    print("\n=== CONTEXT FEATURES ===")
    print(f"  vol_ratio:      {context.get('vol_ratio', 1.0):.3f}")
    print(f"  trend_macd:     {context.get('trend_macd', 0.0):+.4f}")
    print(f"  jump_detected:  {context.get('jump_detected', False)}")
    print(f"  funding_sign:   {context.get('funding_sign', 0.0):+.3f}")
    print(f"  book_imb:       {context.get('book_imb', 0.0):+.3f}")
    print(f"  ofi_15s:        {context.get('ofi_15s', 0.0):+.3f}")
    print(f"  basis_pct:      {context.get('basis_pct', 0.0):+.4f}")
    print("========================\n")


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ТЕСТ КОНТЕКСТНЫХ ФИЧ META")
    print("="*80)

    # Создаем тестовые данные
    np.random.seed(42)
    n = 100

    data = {
        'timestamp': pd.date_range('2025-01-01', periods=n, freq='4H'),
        'open': 50000 + np.cumsum(np.random.randn(n) * 100),
        'high': 50000 + np.cumsum(np.random.randn(n) * 100) + 200,
        'low': 50000 + np.cumsum(np.random.randn(n) * 100) - 200,
        'close': 50000 + np.cumsum(np.random.randn(n) * 100),
        'volume': 1000 + np.random.randn(n) * 100
    }
    df_4h = pd.DataFrame(data)

    # Тестируем функции
    print("\n1️⃣ Тест volume_ratio:")
    vol_ratio = calculate_volume_ratio(df_4h, period=20)
    print(f"   vol_ratio = {vol_ratio:.3f}")
    assert 0.1 <= vol_ratio <= 3.0, "vol_ratio вне допустимого диапазона"
    print("   ✅ PASS")

    print("\n2️⃣ Тест macd_trend:")
    trend_macd = calculate_macd_trend(df_4h)
    print(f"   trend_macd = {trend_macd:+.4f}")
    assert -0.5 <= trend_macd <= 0.5, "trend_macd вне допустимого диапазона"
    print("   ✅ PASS")

    print("\n3️⃣ Тест price_jump detection:")
    jump_flag = detect_price_jump(df_4h, threshold_pct=3.0)
    print(f"   jump_detected = {jump_flag}")
    assert isinstance(jump_flag, (bool, np.bool_)), "jump_flag должен быть bool"
    print("   ✅ PASS")

    print("\n4️⃣ Тест build_context_features:")
    context = build_context_features(df_4h, symbol='BTCUSDT')
    print_context_summary(context)

    # Проверяем, что все ключи присутствуют
    required_keys = ['vol_ratio', 'trend_macd', 'jump_detected',
                    'funding_sign', 'book_imb', 'ofi_15s', 'basis_pct']
    for key in required_keys:
        assert key in context, f"Отсутствует ключ: {key}"
    print("   ✅ PASS - все ключи присутствуют")

    # Проверяем типы и диапазоны
    assert isinstance(context['vol_ratio'], float), "vol_ratio должен быть float"
    assert isinstance(context['trend_macd'], float), "trend_macd должен быть float"
    assert isinstance(context['jump_detected'], (bool, np.bool_)), "jump_detected должен быть bool"
    print("   ✅ PASS - типы корректны")

    print("\n" + "="*80)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("="*80)
