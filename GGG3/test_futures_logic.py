#!/usr/bin/env python3
"""
ТЕСТ ФЬЮЧЕРСНОЙ ЛОГИКИ PAPEREXCHANGE

Проверяет что новая фьючерсная логика работает правильно:
1. Balance НЕ меняется при открытии позиции
2. Equity = balance + unrealized_pnl
3. Balance меняется только при закрытии (добавляется PnL)
"""

import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent / 'paper_trading'))

print("=" * 80)
print("🧪 ТЕСТ ФЬЮЧЕРСНОЙ ЛОГИКИ PAPEREXCHANGE")
print("=" * 80)
print()

from paper_trading.paper_exchange import PaperExchange

# ============================================================================
# ТЕСТ 1: Открытие LONG позиции - баланс НЕ меняется
# ============================================================================

print("📋 ТЕСТ 1: Открытие LONG позиции (баланс НЕ должен меняться)")
print("-" * 80)

try:
    exchange = PaperExchange(initial_capital=1000.0)

    print(f"✅ Начальный капитал: $1000.00")
    print(f"   Balance: ${exchange.get_balance():.2f}")
    print(f"   Equity: ${exchange.get_equity():.2f}")
    print()

    # Открываем LONG позицию
    print("📈 Открываем LONG позицию на BTCUSDT...")
    position = exchange.open_position(
        symbol='BTCUSDT',
        side='LONG',
        amount=0.01,
        tp_price=None,
        sl_price=None
    )

    balance_after_open = exchange.get_balance()
    equity_after_open = exchange.get_equity()

    print()
    print(f"💰 ПОСЛЕ открытия позиции:")
    print(f"   Balance: ${balance_after_open:.2f}")
    print(f"   Equity: ${equity_after_open:.2f}")
    print()

    # ПРОВЕРКА 1: Balance НЕ должен измениться (фьючерсы)
    if abs(balance_after_open - 1000.0) < 0.01:
        print(f"✅ PASS: Balance НЕ изменился после открытия (фьючерсная логика)")
    else:
        print(f"❌ FAIL: Balance изменился! Ожидалось $1000.00, получено ${balance_after_open:.2f}")

    # ПРОВЕРКА 2: Equity должен быть примерно равен начальному капиталу
    # (может быть чуть меньше из-за комиссии на вход)
    if abs(equity_after_open - 1000.0) < 10.0:  # допуск 10 USDT на комиссию
        print(f"✅ PASS: Equity примерно равен начальному капиталу")
    else:
        print(f"❌ FAIL: Equity сильно отличается! Ожидалось ~$1000.00, получено ${equity_after_open:.2f}")

    print()
    print("=" * 80)
    print()

    # ============================================================================
    # ТЕСТ 2: Закрытие позиции - баланс меняется на PnL
    # ============================================================================

    print("📋 ТЕСТ 2: Закрытие позиции (баланс должен измениться на PnL)")
    print("-" * 80)

    # Закрываем позицию
    print("🔻 Закрываем позицию...")
    position_id = position['id']
    closed_position = exchange.close_position(position_id, reason='manual')

    balance_after_close = exchange.get_balance()
    equity_after_close = exchange.get_equity()
    pnl_after_commission = closed_position['pnl_after_commission']

    print()
    print(f"💰 ПОСЛЕ закрытия позиции:")
    print(f"   Balance: ${balance_after_close:.2f}")
    print(f"   Equity: ${equity_after_close:.2f}")
    print(f"   PnL (после комиссий): ${pnl_after_commission:.2f}")
    print()

    # ПРОВЕРКА 3: Balance должен измениться на PnL
    expected_balance = 1000.0 + pnl_after_commission
    if abs(balance_after_close - expected_balance) < 0.01:
        print(f"✅ PASS: Balance изменился на PnL ({pnl_after_commission:+.2f})")
    else:
        print(f"❌ FAIL: Balance неправильный! Ожидалось ${expected_balance:.2f}, получено ${balance_after_close:.2f}")

    # ПРОВЕРКА 4: Equity должен равняться balance (нет открытых позиций)
    if abs(equity_after_close - balance_after_close) < 0.01:
        print(f"✅ PASS: Equity = Balance (нет открытых позиций)")
    else:
        print(f"❌ FAIL: Equity ≠ Balance! Equity=${equity_after_close:.2f}, Balance=${balance_after_close:.2f}")

    print()
    print("=" * 80)
    print()

    # ============================================================================
    # ТЕСТ 3: Открытие SHORT позиции
    # ============================================================================

    print("📋 ТЕСТ 3: Открытие SHORT позиции (баланс НЕ должен меняться)")
    print("-" * 80)

    balance_before_short = exchange.get_balance()

    # Открываем SHORT позицию
    print("📉 Открываем SHORT позицию на ETHUSDT...")
    short_position = exchange.open_position(
        symbol='ETHUSDT',
        side='SHORT',
        amount=0.5,
        tp_price=None,
        sl_price=None
    )

    balance_after_short_open = exchange.get_balance()
    equity_after_short_open = exchange.get_equity()

    print()
    print(f"💰 ПОСЛЕ открытия SHORT позиции:")
    print(f"   Balance: ${balance_after_short_open:.2f}")
    print(f"   Equity: ${equity_after_short_open:.2f}")
    print()

    # ПРОВЕРКА 5: Balance НЕ должен измениться
    if abs(balance_after_short_open - balance_before_short) < 0.01:
        print(f"✅ PASS: Balance НЕ изменился после открытия SHORT")
    else:
        print(f"❌ FAIL: Balance изменился! Было ${balance_before_short:.2f}, стало ${balance_after_short_open:.2f}")

    print()

    # Закрываем SHORT позицию
    print("🔺 Закрываем SHORT позицию...")
    short_position_id = short_position['id']
    closed_short = exchange.close_position(short_position_id, reason='manual')

    balance_final = exchange.get_balance()
    pnl_short = closed_short['pnl_after_commission']

    print()
    print(f"💰 ФИНАЛЬНЫЙ баланс:")
    print(f"   Balance: ${balance_final:.2f}")
    print(f"   PnL SHORT: ${pnl_short:+.2f}")
    print()

    # ПРОВЕРКА 6: Balance должен измениться на PnL SHORT
    expected_final = balance_before_short + pnl_short
    if abs(balance_final - expected_final) < 0.01:
        print(f"✅ PASS: Balance изменился на PnL SHORT ({pnl_short:+.2f})")
    else:
        print(f"❌ FAIL: Balance неправильный! Ожидалось ${expected_final:.2f}, получено ${balance_final:.2f}")

    print()
    print("=" * 80)
    print("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 80)

except Exception as e:
    print(f"❌ FAIL: Ошибка при тестировании: {e}")
    import traceback
    traceback.print_exc()
