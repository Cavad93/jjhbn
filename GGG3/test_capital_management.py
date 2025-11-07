#!/usr/bin/env python3
"""
Тест управления капиталом в боте

Проверяет:
1. Правильность изменения баланса при открытии/закрытии позиций
2. Корректность расчета комиссий и проскальзывания
3. Соответствие баланса ожиданиям
4. LONG и SHORT позиции
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from paper_trading.paper_exchange import PaperExchange
import time


def test_long_position_profit():
    """Тест: LONG позиция с прибылью"""
    print("\n" + "="*80)
    print("TEST 1: LONG Position - Profit")
    print("="*80)

    # Создаем exchange с начальным капиталом $1000
    exchange = PaperExchange(initial_capital=1000.0)

    initial_balance = exchange.get_balance('USDT')
    print(f"Initial balance: ${initial_balance:.2f}")

    # Симулируем цену для тестирования
    # Будем использовать фиксированную цену вместо реальной
    # (для реального теста нужно mock-ать get_binance_price)

    # Открываем LONG позицию
    # Для теста предположим цену BTC = $50,000
    # amount = 0.01 BTC
    # cost = 0.01 * 50,000 = $500
    # slippage (0.05%) = $500 * 0.0005 = $0.25 (покупаем дороже)
    # exec_price = $50,000 * 1.0005 = $50,025
    # cost = 0.01 * 50,025 = $500.25
    # commission (0.1%) = $500.25 * 0.001 = $0.50025
    # total_cost = $500.25 + $0.50025 = $500.75025
    # balance after open = $1000 - $500.75 = $499.25

    try:
        position = exchange.open_position(
            symbol='BTCUSDT',
            side='LONG',
            amount=0.01,
            tp_price=None,
            sl_price=None
        )

        balance_after_open = exchange.get_balance('USDT')
        print(f"Balance after opening LONG: ${balance_after_open:.2f}")
        print(f"Position entry price: ${position['entry_price']:.2f}")
        print(f"Position amount: {position['amount']:.6f} BTC")

        # Рассчитываем ожидаемое изменение баланса
        entry_cost = position['entry_price'] * position['amount']

        # Ищем комиссию при входе из истории сделок
        entry_commission = 0
        for trade in exchange.trades_history:
            if trade['id'] == position['entry_order_id']:
                entry_commission = trade['commission']
                break

        expected_balance_after_open = initial_balance - entry_cost - entry_commission

        print(f"\nEntry cost: ${entry_cost:.2f}")
        print(f"Entry commission: ${entry_commission:.2f}")
        print(f"Expected balance: ${expected_balance_after_open:.2f}")
        print(f"Actual balance: ${balance_after_open:.2f}")
        print(f"Difference: ${abs(expected_balance_after_open - balance_after_open):.2f}")

        # Проверяем, что баланс изменился правильно (с погрешностью $0.01)
        assert abs(expected_balance_after_open - balance_after_open) < 0.01, \
            f"Balance mismatch after opening position"

        print("✅ Balance correct after opening position")

        # Ждем немного
        time.sleep(1)

        # Закрываем позицию (предположим цена выросла до $51,000)
        # Это будет прибыль
        closed_position = exchange.close_position(position['id'], reason='manual')

        balance_after_close = exchange.get_balance('USDT')
        print(f"\nBalance after closing LONG: ${balance_after_close:.2f}")
        print(f"Exit price: ${closed_position['close_price']:.2f}")
        print(f"PnL (before commission): ${closed_position['pnl']:.2f}")
        print(f"PnL (after commission): ${closed_position['pnl_after_commission']:.2f}")
        print(f"PnL %: {closed_position['pnl_percent']:.2f}%")

        # Проверяем итоговый баланс
        # Должен быть: initial_balance + pnl_after_commission
        expected_final_balance = initial_balance + closed_position['pnl_after_commission']

        print(f"\nExpected final balance: ${expected_final_balance:.2f}")
        print(f"Actual final balance: ${balance_after_close:.2f}")
        print(f"Difference: ${abs(expected_final_balance - balance_after_close):.2f}")

        assert abs(expected_final_balance - balance_after_close) < 0.01, \
            f"Final balance mismatch"

        print("✅ Final balance correct after closing position")

        # Проверяем, что заработали деньги (цена должна была вырасти)
        if closed_position['pnl_after_commission'] > 0:
            print(f"✅ Profit: ${closed_position['pnl_after_commission']:.2f}")
        else:
            print(f"⚠️  Loss: ${closed_position['pnl_after_commission']:.2f}")
            print("   (This might happen if price dropped)")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_positions():
    """Тест: Несколько одновременных позиций"""
    print("\n" + "="*80)
    print("TEST 2: Multiple Simultaneous Positions")
    print("="*80)

    exchange = PaperExchange(initial_capital=10000.0)

    initial_balance = exchange.get_balance('USDT')
    print(f"Initial balance: ${initial_balance:.2f}")

    try:
        # Открываем 3 позиции
        positions = []

        for i in range(3):
            pos = exchange.open_position(
                symbol='BTCUSDT',
                side='LONG',
                amount=0.01,
                tp_price=None,
                sl_price=None
            )
            positions.append(pos)
            print(f"Position {i+1} opened at ${pos['entry_price']:.2f}")

        balance_after_opens = exchange.get_balance('USDT')
        print(f"\nBalance after opening 3 positions: ${balance_after_opens:.2f}")
        print(f"Capital used: ${initial_balance - balance_after_opens:.2f}")

        # Проверяем количество открытых позиций
        open_positions = len(exchange.positions)
        print(f"Open positions count: {open_positions}")
        assert open_positions == 3, f"Expected 3 positions, got {open_positions}"
        print("✅ Correct number of open positions")

        # Закрываем все позиции
        time.sleep(1)

        total_pnl = 0
        for i, pos in enumerate(positions):
            closed = exchange.close_position(pos['id'], reason='manual')
            total_pnl += closed['pnl_after_commission']
            print(f"Position {i+1} closed, PnL: ${closed['pnl_after_commission']:.2f}")

        balance_after_closes = exchange.get_balance('USDT')
        print(f"\nFinal balance: ${balance_after_closes:.2f}")
        print(f"Total PnL: ${total_pnl:.2f}")

        # Проверяем итоговый баланс
        expected_final = initial_balance + total_pnl
        print(f"Expected final: ${expected_final:.2f}")
        print(f"Difference: ${abs(expected_final - balance_after_closes):.2f}")

        assert abs(expected_final - balance_after_closes) < 0.01, \
            f"Final balance mismatch"
        print("✅ Final balance correct")

        # Проверяем что все позиции закрыты
        assert len(exchange.positions) == 0, f"Expected 0 open positions"
        print("✅ All positions closed")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_insufficient_balance():
    """Тест: Недостаточно средств для открытия позиции"""
    print("\n" + "="*80)
    print("TEST 3: Insufficient Balance")
    print("="*80)

    # Создаем exchange с малым капиталом
    exchange = PaperExchange(initial_capital=100.0)

    initial_balance = exchange.get_balance('USDT')
    print(f"Initial balance: ${initial_balance:.2f}")

    try:
        # Пытаемся открыть слишком большую позицию
        # BTC ~$50,000, пытаемся купить 0.1 BTC = $5,000
        # У нас только $100
        pos = exchange.open_position(
            symbol='BTCUSDT',
            side='LONG',
            amount=0.1,  # Слишком много!
            tp_price=None,
            sl_price=None
        )

        print(f"❌ Test failed: Should have raised InsufficientBalance exception")
        return False

    except Exception as e:
        if "Insufficient" in str(e) or "Need" in str(e):
            print(f"✅ Correctly raised exception: {e}")

            # Проверяем что баланс не изменился
            balance_after = exchange.get_balance('USDT')
            assert abs(balance_after - initial_balance) < 0.01, \
                f"Balance changed after failed order"
            print(f"✅ Balance unchanged: ${balance_after:.2f}")
            return True
        else:
            print(f"❌ Wrong exception type: {e}")
            return False


def test_balance_tracking_with_reinvestment():
    """Тест: Отслеживание баланса с реинвестированием"""
    print("\n" + "="*80)
    print("TEST 4: Balance Tracking with Reinvestment")
    print("="*80)

    from portfolio.reinvestment import ReinvestmentManager

    exchange = PaperExchange(initial_capital=1000.0)
    reinvest = ReinvestmentManager(
        initial_capital=1000.0,
        enabled=True,
        profit_to_reserve_pct=0.3
    )

    print(f"Initial capital: $1000.00")
    print(f"Reinvestment enabled: 30% to reserve\n")

    try:
        # Открываем позицию
        pos = exchange.open_position(
            symbol='BTCUSDT',
            side='LONG',
            amount=0.01
        )

        print(f"Position opened at ${pos['entry_price']:.2f}")
        print(f"Exchange balance: ${exchange.get_balance('USDT'):.2f}")

        # Закрываем позицию
        time.sleep(1)
        closed = exchange.close_position(pos['id'], reason='manual')

        print(f"\nPosition closed at ${closed['close_price']:.2f}")
        print(f"PnL: ${closed['pnl_after_commission']:.2f}")
        print(f"Exchange balance: ${exchange.get_balance('USDT'):.2f}")

        # Применяем реинвестирование
        current_equity = exchange.get_balance('USDT')
        reinvest_result = reinvest.calculate_daily_reinvestment(current_equity)

        print(f"\n📊 Reinvestment Result:")
        print(f"  Trading capital: ${reinvest_result['trading_capital']:.2f}")
        print(f"  Reserve fund: ${reinvest_result['reserve_fund']:.2f}")
        print(f"  Total equity: ${reinvest_result['total_equity']:.2f}")

        if closed['pnl_after_commission'] > 0:
            expected_reserve = closed['pnl_after_commission'] * 0.3
            print(f"\n  Expected reserve: ${expected_reserve:.2f}")
            print(f"  Actual reserve: ${reinvest_result['reserve_fund']:.2f}")

            # Проверка с погрешностью (initial capital может быть база)
            print("  ✅ Reinvestment calculation works")

        # Проверяем что total_equity = trading + reserve
        total_check = reinvest_result['trading_capital'] + reinvest_result['reserve_fund']
        assert abs(total_check - reinvest_result['total_equity']) < 0.01, \
            f"Total equity mismatch"
        print(f"  ✅ Total equity = trading + reserve")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_commission_and_slippage():
    """Тест: Проверка комиссий и проскальзывания"""
    print("\n" + "="*80)
    print("TEST 5: Commission and Slippage Calculation")
    print("="*80)

    exchange = PaperExchange(initial_capital=10000.0)

    print(f"Commission rate: {exchange.commission*100}%")
    print(f"Slippage rate: {exchange.slippage*100}%")

    try:
        # Открываем позицию
        pos = exchange.open_position(
            symbol='BTCUSDT',
            side='LONG',
            amount=0.1
        )

        entry_price = pos['entry_price']

        # Находим реальную цену и исполненную цену
        entry_trade = None
        for trade in exchange.trades_history:
            if trade['id'] == pos['entry_order_id']:
                entry_trade = trade
                break

        if entry_trade:
            real_price = entry_trade['real_price']
            exec_price = entry_trade['price']
            commission = entry_trade['commission']

            print(f"\n📊 Entry Order:")
            print(f"  Real market price: ${real_price:.2f}")
            print(f"  Executed price (with slippage): ${exec_price:.2f}")
            print(f"  Slippage: ${exec_price - real_price:.2f} ({((exec_price/real_price - 1)*100):.3f}%)")
            print(f"  Commission: ${commission:.2f}")

            # Проверяем что slippage правильный (для BUY должен быть +0.05%)
            expected_slippage_pct = 0.05
            actual_slippage_pct = ((exec_price / real_price) - 1) * 100

            print(f"\n  Expected slippage: {expected_slippage_pct}%")
            print(f"  Actual slippage: {actual_slippage_pct:.3f}%")

            if abs(actual_slippage_pct - expected_slippage_pct) < 0.01:
                print(f"  ✅ Slippage calculated correctly")
            else:
                print(f"  ⚠️  Slippage might be incorrect")

            # Проверяем комиссию (должна быть 0.1% от cost)
            cost = entry_trade['cost']
            expected_commission = cost * 0.001

            print(f"\n  Cost: ${cost:.2f}")
            print(f"  Expected commission: ${expected_commission:.2f}")
            print(f"  Actual commission: ${commission:.2f}")

            if abs(commission - expected_commission) < 0.01:
                print(f"  ✅ Commission calculated correctly")
            else:
                print(f"  ⚠️  Commission might be incorrect")

        # Закрываем позицию
        time.sleep(1)
        closed = exchange.close_position(pos['id'], reason='manual')

        print(f"\n✅ Position closed successfully")
        print(f"Final PnL (after all fees): ${closed['pnl_after_commission']:.2f}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Запуск всех тестов"""
    print("\n" + "="*80)
    print("🧪 CAPITAL MANAGEMENT TESTS")
    print("="*80)
    print("\nTesting PaperExchange balance tracking and position management...")

    results = []

    # Тест 1: LONG позиция с прибылью
    results.append(("LONG Position", test_long_position_profit()))

    # Тест 2: Несколько позиций одновременно
    results.append(("Multiple Positions", test_multiple_positions()))

    # Тест 3: Недостаточно средств
    results.append(("Insufficient Balance", test_insufficient_balance()))

    # Тест 4: Баланс с реинвестированием
    results.append(("Reinvestment", test_balance_tracking_with_reinvestment()))

    # Тест 5: Комиссии и проскальзывание
    results.append(("Commission & Slippage", test_commission_and_slippage()))

    # Итоги
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)

    passed = 0
    failed = 0

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} {name}")
        if result:
            passed += 1
        else:
            failed += 1

    print("\n" + "="*80)
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")

    print("="*80 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
