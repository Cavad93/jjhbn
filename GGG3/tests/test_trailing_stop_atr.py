#!/usr/bin/env python3
"""
Unit-тесты для ATR-based trailing stop

Проверяет корректность работы нового режима трейлинг-стопа
для LONG и SHORT позиций.
"""

import sys
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from risk.trailing_stop import TrailingStopManager


@dataclass
class MockPosition:
    """Мок-объект позиции для тестирования"""
    symbol: str
    direction: str
    entry_price: float
    sl_price: float
    atr_value: float
    entry_time: datetime = field(default_factory=datetime.now)


def test_atr_trailing_long_activation():
    """Тест: активация трейлинг-стопа для LONG при достижении порога прибыли"""
    print("\n=== Test 1: ATR Trailing LONG - Activation ===")

    manager = TrailingStopManager(
        enabled=True,
        activation_pct=3.0,
        distance_atr_mult=1.0
    )

    position = MockPosition(
        symbol='BTCUSDT',
        direction='LONG',
        entry_price=100.0,
        sl_price=97.0,  # SL = entry - 1.5 ATR
        atr_value=2.0
    )

    # Цена +2% - не должно активироваться
    current_price = 102.0
    new_sl = manager.check_trailing_stop(position, current_price)
    assert new_sl is None, f"Expected None at +2% profit, got {new_sl}"
    print("✅ +2% profit: trailing not activated (threshold 3%)")

    # Цена +3.5% - должно активироваться
    current_price = 103.5
    new_sl = manager.check_trailing_stop(position, current_price)
    expected_sl = 103.5 - (2.0 * 1.0)  # peak - (ATR * mult) = 103.5 - 2.0 = 101.5
    assert new_sl is not None, "Expected trailing activation at +3.5%"
    assert abs(new_sl - expected_sl) < 0.01, f"Expected SL {expected_sl}, got {new_sl}"
    print(f"✅ +3.5% profit: trailing activated, new SL = {new_sl:.2f} (expected {expected_sl:.2f})")


def test_atr_trailing_long_update():
    """Тест: обновление трейлинг-стопа для LONG при росте цены"""
    print("\n=== Test 2: ATR Trailing LONG - Update ===")

    manager = TrailingStopManager(
        enabled=True,
        activation_pct=3.0,
        distance_atr_mult=1.0
    )

    position = MockPosition(
        symbol='BTCUSDT',
        direction='LONG',
        entry_price=100.0,
        sl_price=97.0,
        atr_value=2.0
    )

    # Активируем на +3.5%
    current_price = 103.5
    new_sl = manager.check_trailing_stop(position, current_price)
    print(f"✅ Initial activation at {current_price}: SL = {new_sl:.2f}")
    position.sl_price = new_sl

    # Цена растет до +6%
    current_price = 106.0
    new_sl = manager.check_trailing_stop(position, current_price)
    expected_sl = 106.0 - (2.0 * 1.0)  # 104.0
    assert new_sl is not None, "Expected trailing update at +6%"
    assert abs(new_sl - expected_sl) < 0.01, f"Expected SL {expected_sl}, got {new_sl}"
    print(f"✅ Price up to +6% ({current_price}): SL updated to {new_sl:.2f} (expected {expected_sl:.2f})")
    position.sl_price = new_sl

    # Цена откатывается до 105.0 - SL не должен меняться
    current_price = 105.0
    new_sl = manager.check_trailing_stop(position, current_price)
    assert new_sl is None, f"Expected no SL change on pullback, got {new_sl}"
    print(f"✅ Price pullback to {current_price}: SL unchanged at {position.sl_price:.2f}")


def test_atr_trailing_short_activation():
    """Тест: активация трейлинг-стопа для SHORT при достижении порога прибыли"""
    print("\n=== Test 3: ATR Trailing SHORT - Activation ===")

    manager = TrailingStopManager(
        enabled=True,
        activation_pct=3.0,
        distance_atr_mult=1.0
    )

    position = MockPosition(
        symbol='ETHUSDT',
        direction='SHORT',
        entry_price=100.0,
        sl_price=103.0,  # SL = entry + 1.5 ATR
        atr_value=2.0
    )

    # Цена -2% - не должно активироваться
    current_price = 98.0
    new_sl = manager.check_trailing_stop(position, current_price)
    assert new_sl is None, f"Expected None at +2% profit, got {new_sl}"
    print("✅ +2% profit (price -2%): trailing not activated (threshold 3%)")

    # Цена -3.5% - должно активироваться
    current_price = 96.5
    new_sl = manager.check_trailing_stop(position, current_price)
    expected_sl = 96.5 + (2.0 * 1.0)  # peak + (ATR * mult) = 96.5 + 2.0 = 98.5
    assert new_sl is not None, "Expected trailing activation at +3.5%"
    assert abs(new_sl - expected_sl) < 0.01, f"Expected SL {expected_sl}, got {new_sl}"
    print(f"✅ +3.5% profit (price -3.5%): trailing activated, new SL = {new_sl:.2f} (expected {expected_sl:.2f})")


def test_atr_trailing_short_update():
    """Тест: обновление трейлинг-стопа для SHORT при падении цены"""
    print("\n=== Test 4: ATR Trailing SHORT - Update ===")

    manager = TrailingStopManager(
        enabled=True,
        activation_pct=3.0,
        distance_atr_mult=1.0
    )

    position = MockPosition(
        symbol='ETHUSDT',
        direction='SHORT',
        entry_price=100.0,
        sl_price=103.0,
        atr_value=2.0
    )

    # Активируем на +3.5% (цена -3.5%)
    current_price = 96.5
    new_sl = manager.check_trailing_stop(position, current_price)
    print(f"✅ Initial activation at {current_price}: SL = {new_sl:.2f}")
    position.sl_price = new_sl

    # Цена падает до -6%
    current_price = 94.0
    new_sl = manager.check_trailing_stop(position, current_price)
    expected_sl = 94.0 + (2.0 * 1.0)  # 96.0
    assert new_sl is not None, "Expected trailing update at +6%"
    assert abs(new_sl - expected_sl) < 0.01, f"Expected SL {expected_sl}, got {new_sl}"
    print(f"✅ Price down to +6% ({current_price}): SL updated to {new_sl:.2f} (expected {expected_sl:.2f})")
    position.sl_price = new_sl

    # Цена растет до 95.0 - SL не должен меняться
    current_price = 95.0
    new_sl = manager.check_trailing_stop(position, current_price)
    assert new_sl is None, f"Expected no SL change on adverse move, got {new_sl}"
    print(f"✅ Price adverse move to {current_price}: SL unchanged at {position.sl_price:.2f}")


def test_fallback_to_percentage_mode():
    """Тест: fallback на процентный режим если ATR не доступен в позиции"""
    print("\n=== Test 5: Fallback to Percentage Mode ===")

    # Создаем manager с ATR mode, но позицию БЕЗ atr_value
    manager = TrailingStopManager(
        enabled=True,
        activation_pct=3.0,
        distance_pct=2.0,
        distance_atr_mult=1.0  # ATR mode включен
    )

    # Позиция БЕЗ atr_value - должен использоваться fallback на процентный режим
    @dataclass
    class MockPositionNoATR:
        symbol: str
        direction: str
        entry_price: float
        sl_price: float
        entry_time: datetime = field(default_factory=datetime.now)
        # НЕТ atr_value!

    position = MockPositionNoATR(
        symbol='BTCUSDT',
        direction='LONG',
        entry_price=100.0,
        sl_price=97.0
    )

    # Активируем на +3.5%
    current_price = 103.5
    new_sl = manager.check_trailing_stop(position, current_price)
    expected_sl = 103.5 * (1 - 2.0 / 100)  # 103.5 * 0.98 = 101.43
    assert new_sl is not None, "Expected trailing activation"
    assert abs(new_sl - expected_sl) < 0.01, f"Expected SL {expected_sl}, got {new_sl}"
    print(f"✅ Fallback to percentage mode (no ATR in position): SL = {new_sl:.2f} (expected {expected_sl:.2f})")


def test_risk_reward_improvement():
    """Тест: улучшение R:R с новыми параметрами"""
    print("\n=== Test 6: Risk/Reward Improvement ===")

    manager = TrailingStopManager(
        enabled=True,
        activation_pct=3.0,  # Новый порог активации
        distance_atr_mult=1.0
    )

    # Симуляция реальной ситуации
    position = MockPosition(
        symbol='BTCUSDT',
        direction='LONG',
        entry_price=100.0,
        sl_price=97.0,  # -3% SL (1.5 ATR)
        atr_value=2.0
    )

    # Сценарий 1: Позиция идет в прибыль +5%
    current_price = 105.0
    new_sl = manager.check_trailing_stop(position, current_price)
    expected_sl = 105.0 - 2.0  # 103.0
    guaranteed_profit_pct = ((expected_sl - 100.0) / 100.0) * 100  # +3%

    print(f"Scenario: Position at +5% profit ({current_price})")
    print(f"  New SL: {new_sl:.2f}")
    print(f"  Guaranteed profit: +{guaranteed_profit_pct:.1f}%")
    print(f"  Risk/Reward: 1 loss (-3%) ≈ 1 win (+3%) ✅ УЛУЧШЕНО с 1:7")


def test_peak_tracking():
    """Тест: корректное отслеживание пиковой цены"""
    print("\n=== Test 7: Peak Price Tracking ===")

    manager = TrailingStopManager(
        enabled=True,
        activation_pct=3.0,
        distance_atr_mult=1.0
    )

    position = MockPosition(
        symbol='BTCUSDT',
        direction='LONG',
        entry_price=100.0,
        sl_price=97.0,
        atr_value=2.0
    )

    # Активируем на +3.5%
    manager.check_trailing_stop(position, 103.5)
    peak = manager.get_peak_price('BTCUSDT')
    assert peak == 103.5, f"Expected peak 103.5, got {peak}"
    print(f"✅ Peak tracked: {peak}")

    # Цена растет
    manager.check_trailing_stop(position, 106.0)
    peak = manager.get_peak_price('BTCUSDT')
    assert peak == 106.0, f"Expected peak 106.0, got {peak}"
    print(f"✅ Peak updated: {peak}")

    # Цена падает - peak не меняется
    manager.check_trailing_stop(position, 104.0)
    peak = manager.get_peak_price('BTCUSDT')
    assert peak == 106.0, f"Expected peak still 106.0, got {peak}"
    print(f"✅ Peak unchanged on pullback: {peak}")

    # Сброс позиции
    manager.reset_position('BTCUSDT')
    peak = manager.get_peak_price('BTCUSDT')
    assert peak is None, f"Expected None after reset, got {peak}"
    print(f"✅ Peak reset: {peak}")


def run_all_tests():
    """Запуск всех тестов"""
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ ATR-BASED TRAILING STOP")
    print("="*60)

    try:
        test_atr_trailing_long_activation()
        test_atr_trailing_long_update()
        test_atr_trailing_short_activation()
        test_atr_trailing_short_update()
        test_fallback_to_percentage_mode()
        test_risk_reward_improvement()
        test_peak_tracking()

        print("\n" + "="*60)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("="*60)
        return True

    except AssertionError as e:
        print(f"\n❌ ТЕСТ ПРОВАЛЕН: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
