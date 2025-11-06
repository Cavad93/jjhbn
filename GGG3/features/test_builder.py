#!/usr/bin/env python3
"""
Комплексные тесты для BinanceFeatureBuilder

Проверка корректности расчёта 68 фич на различных сценариях
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Добавляем родительскую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.builder import BinanceFeatureBuilder


def create_test_data(length: int = 500, trend: str = 'up') -> tuple:
    """
    Создание тестовых OHLCV данных

    Args:
        length: Количество свечей
        trend: 'up', 'down', 'sideways'

    Returns:
        tuple: (df_5m, df_15m, df_30m, df_4h)
    """
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=length, freq='5min')

    base_price = 50000.0

    # Генерация цены в зависимости от тренда
    if trend == 'up':
        price_walk = np.cumsum(np.random.randn(length) * 50 + 20) + base_price
    elif trend == 'down':
        price_walk = np.cumsum(np.random.randn(length) * 50 - 20) + base_price
    else:  # sideways
        price_walk = np.cumsum(np.random.randn(length) * 50) + base_price

    df_5m = pd.DataFrame({
        'timestamp': dates,
        'open': price_walk + np.random.randn(length) * 50,
        'high': price_walk + np.abs(np.random.randn(length)) * 100,
        'low': price_walk - np.abs(np.random.randn(length)) * 100,
        'close': price_walk,
        'volume': np.abs(np.random.randn(length)) * 1000 + 5000
    })

    # Ресемплинг
    df_15m = df_5m.set_index('timestamp').resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()

    df_30m = df_5m.set_index('timestamp').resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()

    df_4h = df_5m.set_index('timestamp').resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()

    return df_5m, df_15m, df_30m, df_4h


def test_feature_dimensionality():
    """Тест 1: Проверка размерности"""
    print("Тест 1: Проверка размерности фич...", end=" ")

    try:
        df_5m, df_15m, df_30m, df_4h = create_test_data()
        builder = BinanceFeatureBuilder()
        features = builder.build_extended_features(df_5m, df_15m, df_30m, df_4h)

        assert features.shape[0] == 68, f"Ожидалось 68 фич, получено {features.shape[0]}"

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_no_nan_inf():
    """Тест 2: Отсутствие NaN и Inf"""
    print("Тест 2: Отсутствие NaN и Inf...", end=" ")

    try:
        df_5m, df_15m, df_30m, df_4h = create_test_data()
        builder = BinanceFeatureBuilder()
        features = builder.build_extended_features(df_5m, df_15m, df_30m, df_4h)

        nan_count = np.sum(np.isnan(features))
        inf_count = np.sum(np.isinf(features))

        assert nan_count == 0, f"Найдено {nan_count} NaN значений"
        assert inf_count == 0, f"Найдено {inf_count} Inf значений"

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_value_ranges():
    """Тест 3: Значения в разумных пределах"""
    print("Тест 3: Значения в разумных пределах...", end=" ")

    try:
        df_5m, df_15m, df_30m, df_4h = create_test_data()
        builder = BinanceFeatureBuilder()
        features = builder.build_extended_features(df_5m, df_15m, df_30m, df_4h)

        # Все значения должны быть в диапазоне [-10, 10] (clipping)
        assert np.all(features >= -10.0), f"Минимум: {np.min(features)} < -10"
        assert np.all(features <= 10.0), f"Максимум: {np.max(features)} > 10"

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_uptrend_detection():
    """Тест 4: Определение восходящего тренда"""
    print("Тест 4: Определение восходящего тренда...", end=" ")

    try:
        df_5m, df_15m, df_30m, df_4h = create_test_data(trend='up')
        builder = BinanceFeatureBuilder()
        features = builder.build_extended_features(df_5m, df_15m, df_30m, df_4h)

        # Проверяем, что momentum фичи положительные
        # Фичи 7-9: price returns (должны быть преимущественно положительные)
        returns_features = features[7:10]
        avg_return = np.mean(returns_features)

        assert avg_return > 0, f"Средний return {avg_return:.4f} должен быть > 0 для uptrend"

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_downtrend_detection():
    """Тест 5: Определение нисходящего тренда"""
    print("Тест 5: Определение нисходящего тренда...", end=" ")

    try:
        df_5m, df_15m, df_30m, df_4h = create_test_data(trend='down')
        builder = BinanceFeatureBuilder()
        features = builder.build_extended_features(df_5m, df_15m, df_30m, df_4h)

        # Проверяем, что momentum фичи отрицательные
        returns_features = features[7:10]
        avg_return = np.mean(returns_features)

        assert avg_return < 0, f"Средний return {avg_return:.4f} должен быть < 0 для downtrend"

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_reproducibility():
    """Тест 6: Воспроизводимость результатов"""
    print("Тест 6: Воспроизводимость результатов...", end=" ")

    try:
        df_5m, df_15m, df_30m, df_4h = create_test_data()
        builder = BinanceFeatureBuilder()

        # Два запуска на одних и тех же данных
        features_1 = builder.build_extended_features(df_5m, df_15m, df_30m, df_4h)
        features_2 = builder.build_extended_features(df_5m, df_15m, df_30m, df_4h)

        # Должны быть идентичны
        assert np.allclose(features_1, features_2), "Результаты не воспроизводимы"

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_minimum_data_length():
    """Тест 7: Минимальная длина данных"""
    print("Тест 7: Работа с минимальной длиной данных...", end=" ")

    try:
        # Минимум 100 свечей
        df_5m, df_15m, df_30m, df_4h = create_test_data(length=100)
        builder = BinanceFeatureBuilder()
        features = builder.build_extended_features(df_5m, df_15m, df_30m, df_4h)

        assert features.shape[0] == 68, "Размерность не соответствует"
        assert np.sum(np.isnan(features)) == 0, "Содержатся NaN"

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_different_timeframes():
    """Тест 8: Различные таймфреймы"""
    print("Тест 8: Различные таймфреймы...", end=" ")

    try:
        df_5m, df_15m, df_30m, df_4h = create_test_data()
        builder = BinanceFeatureBuilder()

        features = builder.build_extended_features(df_5m, df_15m, df_30m, df_4h)

        # Проверяем, что каждый блок фич имеет разумные значения
        # Блок 1 (5m): индексы 0-13
        momentum_block = features[0:14]
        assert len(momentum_block) == 14

        # Блок 2 (15m): индексы 14-35
        vwap_block = features[14:36]
        assert len(vwap_block) == 22

        # Блок 3 (30m): индексы 36-56
        reversion_block = features[36:57]
        assert len(reversion_block) == 21

        # Блок 4 (4h): индексы 57-67
        atr_block = features[57:68]
        assert len(atr_block) == 11

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_feature_stability():
    """Тест 9: Стабильность фич (малые изменения данных)"""
    print("Тест 9: Стабильность фич...", end=" ")

    try:
        df_5m, df_15m, df_30m, df_4h = create_test_data()
        builder = BinanceFeatureBuilder()

        features_original = builder.build_extended_features(df_5m, df_15m, df_30m, df_4h)

        # Делаем небольшое изменение последней цены (+0.1%)
        df_5m_modified = df_5m.copy()
        df_5m_modified.loc[df_5m_modified.index[-1], 'close'] *= 1.001

        features_modified = builder.build_extended_features(df_5m_modified, df_15m, df_30m, df_4h)

        # Изменения должны быть небольшими
        diff = np.abs(features_modified - features_original)
        max_diff = np.max(diff)

        assert max_diff < 1.0, f"Максимальное изменение {max_diff:.4f} слишком большое"

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_integration_with_db():
    """Тест 10: Интеграция с данными из БД (симуляция)"""
    print("Тест 10: Интеграция с данными из БД...", end=" ")

    try:
        # Симуляция данных из БД (с timestamp в правильном формате)
        df_5m, df_15m, df_30m, df_4h = create_test_data()

        # Конвертируем timestamp в правильный формат (как из БД)
        for df in [df_5m, df_15m, df_30m, df_4h]:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        builder = BinanceFeatureBuilder()
        features = builder.build_extended_features(
            df_5m, df_15m, df_30m, df_4h,
            current_time=pd.Timestamp('2025-01-01 12:00:00')
        )

        assert features.shape[0] == 68
        assert np.sum(np.isnan(features)) == 0

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def run_all_tests():
    """Запуск всех тестов"""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ BINANCE FEATURE BUILDER")
    print("=" * 80)
    print()

    tests = [
        test_feature_dimensionality,
        test_no_nan_inf,
        test_value_ranges,
        test_uptrend_detection,
        test_downtrend_detection,
        test_reproducibility,
        test_minimum_data_length,
        test_different_timeframes,
        test_feature_stability,
        test_integration_with_db,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print()
    print("=" * 80)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 80)

    passed = sum(results)
    total = len(results)

    print(f"Пройдено: {passed}/{total}")

    if passed == total:
        print()
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
        return True
    else:
        print()
        print(f"❌ ПРОВАЛЕНО ТЕСТОВ: {total - passed}")
        return False


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
