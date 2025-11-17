#!/usr/bin/env python3
"""
Тест: проверка что trailing stop может брать ATR из metadata (для paper_bot)
"""

import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent))

from risk.trailing_stop import TrailingStopManager


def test_atr_from_metadata():
    """Тест: trailing stop берёт ATR из metadata (dict позиция)"""
    print("\n=== Test: ATR from metadata (Paper Bot) ===")

    # Создаём trailing stop manager
    manager = TrailingStopManager(
        enabled=True,
        activation_pct=3.0,
        distance_atr_mult=1.0  # ATR-based режим
    )

    # Симулируем позицию из PaperExchange (dict, не Position объект)
    position = {
        'symbol': 'BTCUSDT',
        'direction': 'LONG',
        'entry_price': 100.0,
        'sl_price': 95.0,
        'metadata': {
            'atr_value': 2.5  # ✅ ATR сохранён в metadata
        }
    }

    # Текущая цена с 5% прибылью (должна активировать trailing)
    current_price = 105.0

    # Проверяем trailing stop
    new_sl = manager.check_trailing_stop(position, current_price)

    print(f"Position: LONG @ {position['entry_price']}")
    print(f"Current price: {current_price}")
    print(f"Current SL: {position['sl_price']}")
    print(f"ATR from metadata: {position['metadata']['atr_value']}")

    if new_sl is not None:
        print(f"✅ New SL calculated: {new_sl:.2f}")

        # Проверяем что это ATR-based расчёт
        # Expected: peak_price (105) - (ATR 2.5 × 1.0) = 102.5
        expected_sl = 105.0 - 2.5
        assert abs(new_sl - expected_sl) < 0.01, f"Expected {expected_sl}, got {new_sl}"
        print(f"✅ ATR-based mode ACTIVE: {new_sl:.2f} (distance = 1 ATR)")

        return True
    else:
        print(f"❌ No SL update - should have activated!")
        return False


def test_atr_from_position_object():
    """Тест: trailing stop берёт ATR из position.atr_value (основной бот)"""
    print("\n=== Test: ATR from Position object (Main Bot) ===")

    manager = TrailingStopManager(
        enabled=True,
        activation_pct=3.0,
        distance_atr_mult=1.0
    )

    # Симулируем Position объект с атрибутом atr_value
    class MockPosition:
        def __init__(self):
            self.symbol = 'ETHUSDT'
            self.direction = 'LONG'
            self.entry_price = 2000.0
            self.sl_price = 1900.0
            self.atr_value = 50.0  # ✅ ATR как атрибут объекта

    position = MockPosition()
    current_price = 2100.0  # 5% прибыль

    new_sl = manager.check_trailing_stop(position, current_price)

    print(f"Position: LONG @ {position.entry_price}")
    print(f"Current price: {current_price}")
    print(f"ATR from position.atr_value: {position.atr_value}")

    if new_sl is not None:
        expected_sl = 2100.0 - 50.0  # peak - ATR
        assert abs(new_sl - expected_sl) < 0.01, f"Expected {expected_sl}, got {new_sl}"
        print(f"✅ ATR-based mode ACTIVE: {new_sl:.2f}")
        return True
    else:
        print(f"❌ No SL update")
        return False


def test_fallback_to_percent_mode():
    """Тест: fallback к процентному режиму если ATR не найден"""
    print("\n=== Test: Fallback to percent mode (no ATR) ===")

    manager = TrailingStopManager(
        enabled=True,
        activation_pct=3.0,
        distance_pct=1.0,  # Процентный режим
        distance_atr_mult=1.0  # ATR включен, но ATR не будет найден
    )

    # Позиция БЕЗ ATR
    position = {
        'symbol': 'SOLUSDT',
        'direction': 'LONG',
        'entry_price': 50.0,
        'sl_price': 48.0,
        'metadata': {}  # ❌ Нет atr_value
    }

    current_price = 52.0  # 4% прибыль

    new_sl = manager.check_trailing_stop(position, current_price)

    print(f"Position: LONG @ {position['entry_price']}")
    print(f"Current price: {current_price}")
    print(f"ATR: NOT FOUND (should fallback to percent)")

    if new_sl is not None:
        # Expected: peak (52) × (1 - 0.01) = 51.48
        expected_sl = 52.0 * 0.99
        assert abs(new_sl - expected_sl) < 0.01, f"Expected {expected_sl}, got {new_sl}"
        print(f"✅ Fallback to PERCENT mode: {new_sl:.2f} (1% from peak)")
        return True
    else:
        print(f"❌ No SL update")
        return False


if __name__ == '__main__':
    print("=" * 80)
    print("TRAILING STOP ATR EXTRACTION TESTS")
    print("=" * 80)

    try:
        result1 = test_atr_from_metadata()
        result2 = test_atr_from_position_object()
        result3 = test_fallback_to_percent_mode()

        if result1 and result2 and result3:
            print("\n" + "=" * 80)
            print("✅ ALL TESTS PASSED!")
            print("=" * 80)
            print("\nKey findings:")
            print("  ✅ ATR extraction from metadata (paper_bot) works")
            print("  ✅ ATR extraction from Position.atr_value (main bot) works")
            print("  ✅ Fallback to percent mode works")
            print("\n  🎯 Both paper_bot and main bot now use ATR-based trailing!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
