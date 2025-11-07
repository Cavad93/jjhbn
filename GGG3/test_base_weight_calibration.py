#!/usr/bin/env python3
"""
Тест калибровки весов BASE Logic

Проверяет работу BaseWeightCalibrator на синтетических данных
"""

import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# Импортируем Position из position_manager
import sys
sys.path.insert(0, '.')

from features.weight_calibrator import BaseWeightCalibrator
from portfolio.position_manager import Position


def create_test_position(symbol: str, pnl: float, base_signals: Dict[str, float]) -> Position:
    """Создаёт тестовую позицию с заданными сигналами и результатом"""
    return Position(
        symbol=symbol,
        direction='LONG',
        entry_time=datetime.now(),
        entry_price=100.0,
        amount=1.0,
        position_value=100.0,
        tp_price=110.0,
        sl_price=95.0,
        atr_value=5.0,
        entry_snapshot={
            'base_signals': base_signals,
            'timestamp': datetime.now().timestamp()
        },
        exit_time=datetime.now(),
        exit_price=100.0 + pnl,
        exit_reason='TP' if pnl > 0 else 'SL',
        pnl=pnl,
        pnl_pct=(pnl / 100.0) * 100,
        status='CLOSED'
    )


def test_basic_calibration():
    """Тест базовой калибровки"""
    print("\n" + "="*80)
    print("ТЕСТ 1: Базовая калибровка")
    print("="*80)

    calibrator = BaseWeightCalibrator(min_samples=10)

    # Создаём синтетические данные
    # Идея: сигнал M сильно коррелирует с успехом, остальные - слабо
    positions = []

    for i in range(50):
        # Momentum сильный → прибыль
        if i % 2 == 0:
            base_signals = {'M': 0.8, 'S': 0.1, 'B': 0.0, 'R': -0.1}
            pnl = 5.0  # Профит
        else:
            base_signals = {'M': -0.7, 'S': 0.0, 'B': 0.1, 'R': 0.2}
            pnl = -3.0  # Убыток

        pos = create_test_position(f"TEST{i}USDT", pnl, base_signals)
        positions.append(pos)

    # Калибруем
    old_weights = np.array([0.25, 0.25, 0.25, 0.25])  # Равные веса
    result = calibrator.calibrate(positions, old_weights)

    if result:
        print(f"\n✅ Калибровка успешна!")
        print(f"  Старые веса: {old_weights}")
        print(f"  Новые веса:  {result.weights}")
        print(f"  Samples:     {result.n_samples}")
        print(f"  Win rate:    {result.win_rate:.2%}")
        print(f"  Improvement: {result.improvement:+.1f}%")

        # Проверяем что вес M увеличился (он самый важный)
        if result.weights[0] > old_weights[0]:
            print(f"\n✓ Momentum вес увеличился: {old_weights[0]:.3f} → {result.weights[0]:.3f}")
        else:
            print(f"\n✗ WARNING: Momentum вес НЕ увеличился")

        return True
    else:
        print("\n❌ Калибровка не выполнена")
        return False


def test_insufficient_data():
    """Тест с недостаточным количеством данных"""
    print("\n" + "="*80)
    print("ТЕСТ 2: Недостаточно данных")
    print("="*80)

    calibrator = BaseWeightCalibrator(min_samples=50)

    # Только 10 позиций (меньше min_samples)
    positions = []
    for i in range(10):
        base_signals = {'M': 0.5, 'S': 0.3, 'B': 0.2, 'R': 0.1}
        pnl = 2.0 if i % 2 == 0 else -1.0
        pos = create_test_position(f"TEST{i}USDT", pnl, base_signals)
        positions.append(pos)

    old_weights = np.array([0.25, 0.25, 0.25, 0.25])
    result = calibrator.calibrate(positions, old_weights)

    if result is None:
        print("\n✅ Правильно: калибровка пропущена (недостаточно данных)")
        return True
    else:
        print("\n✗ WARNING: Калибровка выполнена при недостатке данных")
        return False


def test_missing_signals():
    """Тест с отсутствующими сигналами"""
    print("\n" + "="*80)
    print("ТЕСТ 3: Позиции без BASE сигналов")
    print("="*80)

    calibrator = BaseWeightCalibrator(min_samples=10)

    # Создаём позиции БЕЗ base_signals
    positions = []
    for i in range(20):
        pos = Position(
            symbol=f"TEST{i}USDT",
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=100.0,
            amount=1.0,
            position_value=100.0,
            tp_price=110.0,
            sl_price=95.0,
            atr_value=5.0,
            entry_snapshot={},  # Пустой snapshot
            exit_time=datetime.now(),
            exit_price=105.0,
            exit_reason='TP',
            pnl=5.0,
            pnl_pct=5.0,
            status='CLOSED'
        )
        positions.append(pos)

    old_weights = np.array([0.25, 0.25, 0.25, 0.25])
    result = calibrator.calibrate(positions, old_weights)

    if result is None:
        print("\n✅ Правильно: калибровка пропущена (нет сигналов)")
        return True
    else:
        print("\n✗ WARNING: Калибровка выполнена без сигналов")
        return False


def test_smoothing():
    """Тест EMA сглаживания"""
    print("\n" + "="*80)
    print("ТЕСТ 4: EMA сглаживание весов")
    print("="*80)

    calibrator = BaseWeightCalibrator(min_samples=10, smoothing_alpha=0.5)

    # Синтетические данные где S (VWAP Slope) важнее всего
    positions = []
    for i in range(50):
        if i % 2 == 0:
            base_signals = {'M': 0.1, 'S': 0.9, 'B': 0.0, 'R': 0.0}
            pnl = 10.0
        else:
            base_signals = {'M': 0.0, 'S': -0.8, 'B': 0.1, 'R': 0.1}
            pnl = -5.0

        pos = create_test_position(f"TEST{i}USDT", pnl, base_signals)
        positions.append(pos)

    old_weights = np.array([0.5, 0.2, 0.2, 0.1])  # M доминирует
    result = calibrator.calibrate(positions, old_weights)

    if result:
        print(f"\n  Старые веса: {old_weights}")
        print(f"  Новые веса:  {result.weights}")

        # С сглаживанием 0.5, новые веса должны быть между старыми и чистой калибровкой
        # Проверяем что S увеличился, но не полностью доминирует
        if result.weights[1] > old_weights[1] and result.weights[0] > 0.1:
            print(f"\n✅ Сглаживание работает:")
            print(f"  S увеличился: {old_weights[1]:.3f} → {result.weights[1]:.3f}")
            print(f"  M сохранил значимость: {result.weights[0]:.3f}")
            return True
        else:
            print(f"\n✗ WARNING: Сглаживание работает некорректно")
            return False
    else:
        print("\n❌ Калибровка не выполнена")
        return False


def test_stats():
    """Тест получения статистики"""
    print("\n" + "="*80)
    print("ТЕСТ 5: Статистика калибровок")
    print("="*80)

    calibrator = BaseWeightCalibrator(min_samples=10)

    # Первая калибровка
    positions = []
    for i in range(20):
        base_signals = {'M': 0.5, 'S': 0.3, 'B': 0.2, 'R': 0.1}
        pnl = 3.0 if i % 2 == 0 else -2.0
        pos = create_test_position(f"TEST{i}USDT", pnl, base_signals)
        positions.append(pos)

    old_weights = np.array([0.25, 0.25, 0.25, 0.25])
    result1 = calibrator.calibrate(positions, old_weights)

    # Вторая калибровка
    for i in range(20, 40):
        base_signals = {'M': 0.6, 'S': 0.2, 'B': 0.1, 'R': 0.1}
        pnl = 4.0 if i % 2 == 0 else -1.0
        pos = create_test_position(f"TEST{i}USDT", pnl, base_signals)
        positions.append(pos)

    result2 = calibrator.calibrate(positions, result1.weights if result1 else old_weights)

    # Получаем статистику
    stats = calibrator.get_stats()

    print(f"\n  Всего калибровок: {stats['total_calibrations']}")
    print(f"  Текущие веса: {stats['current_weights']}")
    print(f"  История (последние):")
    for i, h in enumerate(stats['weights_history'], 1):
        print(f"    {i}. weights={h['weights']}, samples={h['n_samples']}, "
              f"improvement={h['improvement']:+.1f}%")

    if stats['total_calibrations'] == 2:
        print(f"\n✅ Статистика работает корректно")
        return True
    else:
        print(f"\n✗ WARNING: Ожидалось 2 калибровки, получено {stats['total_calibrations']}")
        return False


def main():
    """Запуск всех тестов"""
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ BaseWeightCalibrator")
    print("="*80)

    results = {
        'Базовая калибровка': test_basic_calibration(),
        'Недостаточно данных': test_insufficient_data(),
        'Позиции без сигналов': test_missing_signals(),
        'EMA сглаживание': test_smoothing(),
        'Статистика': test_stats()
    }

    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<30} {status}")

    all_passed = all(results.values())
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    else:
        print("⚠️  НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ")
    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
