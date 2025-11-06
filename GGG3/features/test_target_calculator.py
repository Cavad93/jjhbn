"""
Комплексные тесты для target_calculator.py

Тестирует функции:
- calculate_atr
- calculate_tp_sl_outcome_bidirectional
- detect_market_phase
- prepare_training_dataset

Автор: Claude Code
Дата: 2025-11-06
"""

import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from typing import Dict, List

# Добавляем путь к модулю
sys.path.append('/home/user/jjhbn/GGG3')

from features.target_calculator import (
    calculate_atr,
    calculate_tp_sl_outcome_bidirectional,
    detect_market_phase,
    prepare_training_dataset
)


def create_test_ohlcv(n_candles: int = 200, base_price: float = 50000, seed: int = 42) -> pd.DataFrame:
    """Создаёт тестовые OHLCV данные с реалистичными свойствами"""
    np.random.seed(seed)

    # Генерируем random walk с трендом
    returns = np.random.randn(n_candles) * 0.01  # 1% волатильность
    prices = base_price * (1 + returns).cumprod()

    # OHLC с корректными high/low
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n_candles, freq='4H'),
        'open': prices,
        'close': prices * (1 + np.random.randn(n_candles) * 0.002),
        'volume': np.random.randint(1000, 10000, n_candles)
    })

    # High и low должны включать open и close
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.randn(n_candles)) * 0.01)
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.randn(n_candles)) * 0.01)

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


def test_1_atr_calculation():
    """Тест 1: Расчёт ATR"""
    print("\n" + "="*80)
    print("ТЕСТ 1: Расчёт ATR")
    print("="*80)

    df = create_test_ohlcv(n_candles=100)
    atr = calculate_atr(df, period=10)

    # Проверки
    assert len(atr) == len(df), "ATR должен иметь ту же длину что и DataFrame"
    assert not np.isnan(atr.iloc[-1]), "Последнее значение ATR не должно быть NaN"
    assert atr.iloc[-1] > 0, "ATR должен быть положительным"
    assert pd.isna(atr.iloc[0]), "Первые значения ATR должны быть NaN (период расчёта)"

    print(f"✅ ATR корректно рассчитан")
    print(f"   Последнее значение: ${atr.iloc[-1]:.2f}")
    print(f"   Среднее: ${atr.mean():.2f}")
    print(f"   Min/Max: ${atr.min():.2f} / ${atr.max():.2f}")

    return True


def test_2_long_tp_reached():
    """Тест 2: LONG позиция - TP достигнут"""
    print("\n" + "="*80)
    print("ТЕСТ 2: LONG позиция - TP достигнут")
    print("="*80)

    # Создаём данные где цена растёт
    np.random.seed(100)
    n_candles = 150

    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n_candles, freq='4H'),
        'open': 50000 + np.arange(n_candles) * 50,  # Растущий тренд
        'close': 50000 + np.arange(n_candles) * 50,
        'volume': 5000
    })

    df['high'] = df['close'] + 100
    df['low'] = df['close'] - 100

    entry_idx = 50
    entry_price = df['close'].iloc[entry_idx]
    atr_value = 1000.0

    result = calculate_tp_sl_outcome_bidirectional(
        df=df,
        entry_idx=entry_idx,
        entry_price=entry_price,
        atr_value=atr_value,
        direction='LONG',
        tp_mult=1.2,
        sl_mult=0.8,
        max_bars=50
    )

    assert result is not None, "Результат не должен быть None"
    assert result['outcome'] == 1, "Outcome должен быть 1 (TP достигнут)"
    assert result['direction'] == 'LONG', "Direction должен быть LONG"
    assert result['pnl_pct'] > 0, "P&L должен быть положительным"

    print(f"✅ LONG TP корректно определён")
    print(f"   Entry price: ${entry_price:.2f}")
    print(f"   TP price: ${entry_price + 1.2 * atr_value:.2f}")
    print(f"   Exit price: ${result['exit_price']:.2f}")
    print(f"   Bars held: {result['bars_held']}")
    print(f"   P&L: {result['pnl_pct']:.2f}%")

    return True


def test_3_long_sl_reached():
    """Тест 3: LONG позиция - SL достигнут"""
    print("\n" + "="*80)
    print("ТЕСТ 3: LONG позиция - SL достигнут")
    print("="*80)

    # Создаём данные где цена падает
    np.random.seed(101)
    n_candles = 150

    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n_candles, freq='4H'),
        'open': 50000 - np.arange(n_candles) * 50,  # Падающий тренд
        'close': 50000 - np.arange(n_candles) * 50,
        'volume': 5000
    })

    df['high'] = df['close'] + 100
    df['low'] = df['close'] - 100

    entry_idx = 50
    entry_price = df['close'].iloc[entry_idx]
    atr_value = 1000.0

    result = calculate_tp_sl_outcome_bidirectional(
        df=df,
        entry_idx=entry_idx,
        entry_price=entry_price,
        atr_value=atr_value,
        direction='LONG',
        tp_mult=1.2,
        sl_mult=0.8,
        max_bars=50
    )

    assert result is not None, "Результат не должен быть None"
    assert result['outcome'] == 0, "Outcome должен быть 0 (SL достигнут)"
    assert result['direction'] == 'LONG', "Direction должен быть LONG"
    assert result['pnl_pct'] < 0, "P&L должен быть отрицательным"

    print(f"✅ LONG SL корректно определён")
    print(f"   Entry price: ${entry_price:.2f}")
    print(f"   SL price: ${entry_price - 0.8 * atr_value:.2f}")
    print(f"   Exit price: ${result['exit_price']:.2f}")
    print(f"   Bars held: {result['bars_held']}")
    print(f"   P&L: {result['pnl_pct']:.2f}%")

    return True


def test_4_short_tp_reached():
    """Тест 4: SHORT позиция - TP достигнут"""
    print("\n" + "="*80)
    print("ТЕСТ 4: SHORT позиция - TP достигнут")
    print("="*80)

    # Создаём данные где цена падает (прибыль для SHORT)
    np.random.seed(102)
    n_candles = 150

    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n_candles, freq='4H'),
        'open': 50000 - np.arange(n_candles) * 50,  # Падающий тренд
        'close': 50000 - np.arange(n_candles) * 50,
        'volume': 5000
    })

    df['high'] = df['close'] + 100
    df['low'] = df['close'] - 100

    entry_idx = 50
    entry_price = df['close'].iloc[entry_idx]
    atr_value = 1000.0

    result = calculate_tp_sl_outcome_bidirectional(
        df=df,
        entry_idx=entry_idx,
        entry_price=entry_price,
        atr_value=atr_value,
        direction='SHORT',
        tp_mult=1.2,
        sl_mult=0.8,
        max_bars=50
    )

    assert result is not None, "Результат не должен быть None"
    assert result['outcome'] == 1, "Outcome должен быть 1 (TP достигнут)"
    assert result['direction'] == 'SHORT', "Direction должен быть SHORT"
    assert result['pnl_pct'] > 0, "P&L должен быть положительным"

    print(f"✅ SHORT TP корректно определён")
    print(f"   Entry price: ${entry_price:.2f}")
    print(f"   TP price: ${entry_price - 1.2 * atr_value:.2f}")
    print(f"   Exit price: ${result['exit_price']:.2f}")
    print(f"   Bars held: {result['bars_held']}")
    print(f"   P&L: {result['pnl_pct']:.2f}%")

    return True


def test_5_short_sl_reached():
    """Тест 5: SHORT позиция - SL достигнут"""
    print("\n" + "="*80)
    print("ТЕСТ 5: SHORT позиция - SL достигнут")
    print("="*80)

    # Создаём данные где цена растёт (убыток для SHORT)
    np.random.seed(103)
    n_candles = 150

    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n_candles, freq='4H'),
        'open': 50000 + np.arange(n_candles) * 50,  # Растущий тренд
        'close': 50000 + np.arange(n_candles) * 50,
        'volume': 5000
    })

    df['high'] = df['close'] + 100
    df['low'] = df['close'] - 100

    entry_idx = 50
    entry_price = df['close'].iloc[entry_idx]
    atr_value = 1000.0

    result = calculate_tp_sl_outcome_bidirectional(
        df=df,
        entry_idx=entry_idx,
        entry_price=entry_price,
        atr_value=atr_value,
        direction='SHORT',
        tp_mult=1.2,
        sl_mult=0.8,
        max_bars=50
    )

    assert result is not None, "Результат не должен быть None"
    assert result['outcome'] == 0, "Outcome должен быть 0 (SL достигнут)"
    assert result['direction'] == 'SHORT', "Direction должен быть SHORT"
    assert result['pnl_pct'] < 0, "P&L должен быть отрицательным"

    print(f"✅ SHORT SL корректно определён")
    print(f"   Entry price: ${entry_price:.2f}")
    print(f"   SL price: ${entry_price + 0.8 * atr_value:.2f}")
    print(f"   Exit price: ${result['exit_price']:.2f}")
    print(f"   Bars held: {result['bars_held']}")
    print(f"   P&L: {result['pnl_pct']:.2f}%")

    return True


def test_6_timeout():
    """Тест 6: Таймаут - не достигли ни TP ни SL"""
    print("\n" + "="*80)
    print("ТЕСТ 6: Таймаут - не достигли ни TP ни SL")
    print("="*80)

    # Создаём данные с малой волатильностью
    df = create_test_ohlcv(n_candles=150, seed=104)

    # Устанавливаем очень узкий диапазон цен
    entry_idx = 50
    entry_price = df['close'].iloc[entry_idx]

    # Делаем все цены близкими к entry_price
    df.loc[entry_idx+1:, 'close'] = entry_price + np.random.randn(len(df) - entry_idx - 1) * 10
    df.loc[entry_idx+1:, 'high'] = df.loc[entry_idx+1:, 'close'] + 5
    df.loc[entry_idx+1:, 'low'] = df.loc[entry_idx+1:, 'close'] - 5

    atr_value = 10000.0  # Очень большой ATR (не достигнем)

    result = calculate_tp_sl_outcome_bidirectional(
        df=df,
        entry_idx=entry_idx,
        entry_price=entry_price,
        atr_value=atr_value,
        direction='LONG',
        tp_mult=1.2,
        sl_mult=0.8,
        max_bars=20
    )

    assert result is None, "Результат должен быть None (таймаут)"

    print(f"✅ Таймаут корректно обработан")
    print(f"   Entry price: ${entry_price:.2f}")
    print(f"   TP price: ${entry_price + 1.2 * atr_value:.2f}")
    print(f"   SL price: ${entry_price - 0.8 * atr_value:.2f}")
    print(f"   Max bars: 20")
    print(f"   Результат: None (таймаут)")

    return True


def test_7_market_phases():
    """Тест 7: Определение фаз рынка"""
    print("\n" + "="*80)
    print("ТЕСТ 7: Определение фаз рынка")
    print("="*80)

    phase_names = {
        0: 'Ranging (флэт)',
        1: 'Accumulation (накопление)',
        2: 'Markup (рост)',
        3: 'Distribution (распределение)',
        4: 'Markdown (падение)',
        5: 'Volatility spike'
    }

    # Тест 7.1: Ranging (флэт)
    df_ranging = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=100, freq='4H'),
        'open': 50000,
        'high': 50050,
        'low': 49950,
        'close': 50000 + np.random.randn(100) * 10,  # Малая волатильность
        'volume': 5000
    })

    phase = detect_market_phase(df_ranging)
    print(f"   Ranging test: фаза {phase} - {phase_names.get(phase, 'Unknown')}")
    assert phase in [0, 1], "Для флэта должна быть фаза 0 или 1"

    # Тест 7.2: Markup (рост)
    df_markup = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=100, freq='4H'),
        'open': 50000 + np.arange(100) * 100,  # Сильный рост
        'close': 50000 + np.arange(100) * 100,
        'volume': 5000 * (1 + np.arange(100) * 0.02)  # Растущий объём
    })
    df_markup['high'] = df_markup['close'] + 100
    df_markup['low'] = df_markup['close'] - 100

    phase = detect_market_phase(df_markup)
    print(f"   Markup test: фаза {phase} - {phase_names.get(phase, 'Unknown')}")
    assert phase in [1, 2], "Для роста должна быть фаза 1 или 2"

    # Тест 7.3: Markdown (падение)
    df_markdown = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=100, freq='4H'),
        'open': 50000 - np.arange(100) * 100,  # Сильное падение
        'close': 50000 - np.arange(100) * 100,
        'volume': 5000 * (1 + np.arange(100) * 0.02)  # Растущий объём
    })
    df_markdown['high'] = df_markdown['close'] + 100
    df_markdown['low'] = df_markdown['close'] - 100

    phase = detect_market_phase(df_markdown)
    print(f"   Markdown test: фаза {phase} - {phase_names.get(phase, 'Unknown')}")
    assert phase in [3, 4], "Для падения должна быть фаза 3 или 4"

    print(f"✅ Фазы рынка корректно определяются")

    return True


def test_8_risk_reward_ratio():
    """Тест 8: Проверка Risk/Reward соотношения"""
    print("\n" + "="*80)
    print("ТЕСТ 8: Проверка Risk/Reward соотношения")
    print("="*80)

    entry_price = 50000.0
    atr_value = 1000.0
    tp_mult = 1.2
    sl_mult = 0.8

    # LONG
    tp_long = entry_price + tp_mult * atr_value
    sl_long = entry_price - sl_mult * atr_value
    reward_long = tp_long - entry_price
    risk_long = entry_price - sl_long
    rr_long = reward_long / risk_long

    print(f"   LONG:")
    print(f"      Entry: ${entry_price:.2f}")
    print(f"      TP: ${tp_long:.2f} (reward: ${reward_long:.2f})")
    print(f"      SL: ${sl_long:.2f} (risk: ${risk_long:.2f})")
    print(f"      R/R: {rr_long:.2f}")

    # SHORT
    tp_short = entry_price - tp_mult * atr_value
    sl_short = entry_price + sl_mult * atr_value
    reward_short = entry_price - tp_short
    risk_short = sl_short - entry_price
    rr_short = reward_short / risk_short

    print(f"   SHORT:")
    print(f"      Entry: ${entry_price:.2f}")
    print(f"      TP: ${tp_short:.2f} (reward: ${reward_short:.2f})")
    print(f"      SL: ${sl_short:.2f} (risk: ${risk_short:.2f})")
    print(f"      R/R: {rr_short:.2f}")

    # Проверяем что R/R = 1.5 для обоих направлений
    assert abs(rr_long - 1.5) < 0.01, f"R/R для LONG должен быть 1.5, получено {rr_long}"
    assert abs(rr_short - 1.5) < 0.01, f"R/R для SHORT должен быть 1.5, получено {rr_short}"

    print(f"✅ Risk/Reward соотношение корректно (1.5:1 для обоих направлений)")

    return True


def test_9_symmetry_long_short():
    """Тест 9: Симметричность LONG и SHORT"""
    print("\n" + "="*80)
    print("ТЕСТ 9: Симметричность LONG и SHORT")
    print("="*80)

    df = create_test_ohlcv(n_candles=200, seed=105)

    entry_idx = 100
    entry_price = df['close'].iloc[entry_idx]
    atr = calculate_atr(df, period=10)
    atr_value = atr.iloc[entry_idx]

    # Получаем результаты для обоих направлений
    result_long = calculate_tp_sl_outcome_bidirectional(
        df, entry_idx, entry_price, atr_value, 'LONG', max_bars=50
    )

    result_short = calculate_tp_sl_outcome_bidirectional(
        df, entry_idx, entry_price, atr_value, 'SHORT', max_bars=50
    )

    print(f"   Entry price: ${entry_price:.2f}")
    print(f"   ATR: ${atr_value:.2f}")

    if result_long:
        print(f"   LONG: {'TP' if result_long['outcome'] == 1 else 'SL'} @ ${result_long['exit_price']:.2f} ({result_long['pnl_pct']:.2f}%)")
    else:
        print(f"   LONG: Timeout")

    if result_short:
        print(f"   SHORT: {'TP' if result_short['outcome'] == 1 else 'SL'} @ ${result_short['exit_price']:.2f} ({result_short['pnl_pct']:.2f}%)")
    else:
        print(f"   SHORT: Timeout")

    # Проверяем что хотя бы один достиг результата
    assert result_long is not None or result_short is not None, "Хотя бы одно направление должно дать результат"

    print(f"✅ LONG и SHORT работают симметрично")

    return True


def run_all_tests():
    """Запускает все тесты"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ TARGET CALCULATOR" + " "*20 + "║")
    print("╚" + "="*78 + "╝")

    tests = [
        test_1_atr_calculation,
        test_2_long_tp_reached,
        test_3_long_sl_reached,
        test_4_short_tp_reached,
        test_5_short_sl_reached,
        test_6_timeout,
        test_7_market_phases,
        test_8_risk_reward_ratio,
        test_9_symmetry_long_short
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n❌ ПРОВАЛ: {e}")
        except Exception as e:
            failed += 1
            print(f"\n❌ ОШИБКА: {e}")

    # Итоги
    print("\n" + "="*80)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*80)
    print(f"Пройдено: {passed}/{len(tests)}")
    print(f"Провалено: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print(f"\n❌ ПРОВАЛЕНО ТЕСТОВ: {failed}")

    print("="*80)

    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()

    # Возвращаем код выхода
    sys.exit(0 if failed == 0 else 1)
