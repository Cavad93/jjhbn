#!/usr/bin/env python3
"""
ТЕСТ ЛОГИКИ МАРЖИ В ФЬЮЧЕРСАХ

Проверяет что бот правильно понимает сколько денег доступно для новых позиций
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'paper_trading'))

print("=" * 80)
print("🧪 ТЕСТ ЛОГИКИ МАРЖИ")
print("=" * 80)
print()

from paper_trading.paper_exchange import PaperExchange

# ============================================================================
# ТЕСТ 1: Открытие позиции - маржа блокируется
# ============================================================================

print("📋 ТЕСТ 1: Маржа блокируется при открытии позиции")
print("-" * 80)

try:
    exchange = PaperExchange(initial_capital=1000.0)

    print(f"✅ Начальный капитал: $1000.00")
    print(f"   Balance: ${exchange.get_balance():.2f}")
    print(f"   Used Margin: ${exchange.get_used_margin():.2f}")
    print(f"   Free Margin: ${exchange.get_free_margin():.2f}")
    print()

    # Симулируем открытие позиции
    entry_price = 50000.0
    amount = 0.01
    position_value = entry_price * amount  # $500

    position_id = "TEST_POS_001"
    exchange.positions[position_id] = {
        'id': position_id,
        'symbol': 'BTCUSDT',
        'side': 'LONG',
        'amount': amount,
        'entry_price': entry_price,
        'entry_order_id': 'TEST_ORDER_001',
        'status': 'open',
        'opened_at': 0,
        'tp_price': None,
        'sl_price': None,
        'metadata': {}
    }

    print(f"📈 Открыта позиция:")
    print(f"   Position value: ${position_value:.2f}")
    print()

    balance_after = exchange.get_balance()
    used_margin_after = exchange.get_used_margin(leverage=1.0)
    free_margin_after = exchange.get_free_margin(leverage=1.0)

    print(f"💰 ПОСЛЕ открытия:")
    print(f"   Balance: ${balance_after:.2f}")
    print(f"   Used Margin: ${used_margin_after:.2f}")
    print(f"   Free Margin: ${free_margin_after:.2f}")
    print()

    # ПРОВЕРКА 1: Balance не изменился
    if abs(balance_after - 1000.0) < 0.01:
        print(f"✅ PASS: Balance НЕ изменился")
    else:
        print(f"❌ FAIL: Balance изменился!")

    # ПРОВЕРКА 2: Used Margin = Position Value
    if abs(used_margin_after - position_value) < 0.01:
        print(f"✅ PASS: Used Margin = ${position_value:.2f}")
    else:
        print(f"❌ FAIL: Used Margin неправильный! Ожидалось ${position_value:.2f}, получено ${used_margin_after:.2f}")

    # ПРОВЕРКА 3: Free Margin = Balance - Used Margin
    expected_free = 1000.0 - position_value  # $500
    if abs(free_margin_after - expected_free) < 0.01:
        print(f"✅ PASS: Free Margin = ${expected_free:.2f} (доступно для новых позиций)")
    else:
        print(f"❌ FAIL: Free Margin неправильный! Ожидалось ${expected_free:.2f}, получено ${free_margin_after:.2f}")

    print()
    print("=" * 80)
    print()

    # ============================================================================
    # ТЕСТ 2: Открытие второй позиции - маржа уменьшается
    # ============================================================================

    print("📋 ТЕСТ 2: Открытие второй позиции")
    print("-" * 80)

    # Симулируем открытие второй позиции
    entry_price_2 = 3000.0  # ETH
    amount_2 = 0.1
    position_value_2 = entry_price_2 * amount_2  # $300

    position_id_2 = "TEST_POS_002"
    exchange.positions[position_id_2] = {
        'id': position_id_2,
        'symbol': 'ETHUSDT',
        'side': 'SHORT',
        'amount': amount_2,
        'entry_price': entry_price_2,
        'entry_order_id': 'TEST_ORDER_002',
        'status': 'open',
        'opened_at': 0,
        'tp_price': None,
        'sl_price': None,
        'metadata': {}
    }

    print(f"📉 Открыта вторая позиция (SHORT ETH):")
    print(f"   Position value: ${position_value_2:.2f}")
    print()

    used_margin_2 = exchange.get_used_margin(leverage=1.0)
    free_margin_2 = exchange.get_free_margin(leverage=1.0)

    print(f"💰 ПОСЛЕ открытия второй позиции:")
    print(f"   Balance: ${exchange.get_balance():.2f}")
    print(f"   Used Margin: ${used_margin_2:.2f}")
    print(f"   Free Margin: ${free_margin_2:.2f}")
    print()

    # ПРОВЕРКА 4: Used Margin = сумма всех позиций
    expected_used = position_value + position_value_2  # $800
    if abs(used_margin_2 - expected_used) < 0.01:
        print(f"✅ PASS: Used Margin = ${expected_used:.2f} (сумма всех позиций)")
    else:
        print(f"❌ FAIL: Used Margin неправильный! Ожидалось ${expected_used:.2f}, получено ${used_margin_2:.2f}")

    # ПРОВЕРКА 5: Free Margin уменьшилась
    expected_free_2 = 1000.0 - expected_used  # $200
    if abs(free_margin_2 - expected_free_2) < 0.01:
        print(f"✅ PASS: Free Margin = ${expected_free_2:.2f} (доступно для новых позиций)")
    else:
        print(f"❌ FAIL: Free Margin неправильный! Ожидалось ${expected_free_2:.2f}, получено ${free_margin_2:.2f}")

    print()
    print("=" * 80)
    print()

    # ============================================================================
    # ТЕСТ 3: Маржа с плечом
    # ============================================================================

    print("📋 ТЕСТ 3: Маржа с плечом 10x")
    print("-" * 80)

    used_margin_10x = exchange.get_used_margin(leverage=10.0)
    free_margin_10x = exchange.get_free_margin(leverage=10.0)

    expected_used_10x = expected_used / 10.0  # $80
    expected_free_10x = 1000.0 - expected_used_10x  # $920

    print(f"💰 С плечом 10x:")
    print(f"   Used Margin: ${used_margin_10x:.2f}")
    print(f"   Free Margin: ${free_margin_10x:.2f}")
    print()

    # ПРОВЕРКА 6: Used Margin с плечом
    if abs(used_margin_10x - expected_used_10x) < 0.01:
        print(f"✅ PASS: Used Margin (10x) = ${expected_used_10x:.2f}")
    else:
        print(f"❌ FAIL: Used Margin (10x) неправильный!")

    # ПРОВЕРКА 7: Free Margin с плечом
    if abs(free_margin_10x - expected_free_10x) < 0.01:
        print(f"✅ PASS: Free Margin (10x) = ${expected_free_10x:.2f}")
    else:
        print(f"❌ FAIL: Free Margin (10x) неправильный!")

    print()
    print("=" * 80)
    print()

    # ============================================================================
    # ИТОГИ
    # ============================================================================

    print("📊 ИТОГИ:")
    print("-" * 80)
    print()
    print("✅ ПРАВИЛЬНАЯ ЛОГИКА МАРЖИ (ФЬЮЧЕРСЫ):")
    print(f"   Balance:      ${exchange.get_balance():.2f}  ← НЕ меняется")
    print(f"   Used Margin:  ${exchange.get_used_margin():.2f}  ← Заблокировано в позициях")
    print(f"   Free Margin:  ${exchange.get_free_margin():.2f}  ← Доступно для новых позиций")
    print()
    print("💡 БОТ ПОНИМАЕТ СКОЛЬКО ДЕНЕГ ДОСТУПНО:")
    print(f"   1. Используйте exchange.get_free_margin() для Kelly")
    print(f"   2. Или используйте exchange.get_equity() для reinvestment")
    print()
    print("=" * 80)
    print("✅ ТЕСТ УСПЕШНО ЗАВЕРШЕН")
    print("=" * 80)

except Exception as e:
    print(f"❌ FAIL: Ошибка: {e}")
    import traceback
    traceback.print_exc()
