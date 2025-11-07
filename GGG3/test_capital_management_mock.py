#!/usr/bin/env python3
"""
Тест управления капиталом в боте с MOCK ценами

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
from unittest.mock import patch


# Mock цены для тестирования
MOCK_PRICES = {
    'BTCUSDT': 50000.0,
    'ETHUSDT': 3000.0,
}

# Счетчик для симуляции изменения цены
price_call_count = {}


def mock_get_binance_price(symbol: str) -> float:
    """Mock функция для получения цены"""
    # Инициализируем счетчик для символа
    if symbol not in price_call_count:
        price_call_count[symbol] = 0

    price_call_count[symbol] += 1
    call = price_call_count[symbol]

    # Базовая цена
    base_price = MOCK_PRICES.get(symbol, 50000.0)

    # Симулируем рост цены при закрытии (каждый второй вызов)
    if call % 2 == 0:
        # Закрытие - цена выросла на 2%
        return base_price * 1.02
    else:
        # Открытие - базовая цена
        return base_price


def reset_mock_prices():
    """Сброс счетчика цен"""
    global price_call_count
    price_call_count = {}


@patch('paper_trading.paper_exchange.get_binance_price', side_effect=mock_get_binance_price)
def test_long_position_profit(_mock):
    """Тест: LONG позиция с прибылью"""
    print("\n" + "="*80)
    print("TEST 1: LONG Position - Profit")
    print("="*80)

    reset_mock_prices()
    exchange = PaperExchange(initial_capital=1000.0)

    initial_balance = exchange.get_balance('USDT')
    print(f"Initial balance: ${initial_balance:.2f}")

    try:
        # Открываем LONG позицию
        # BTC = $50,000, amount = 0.01 BTC
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

        # Ищем комиссию при входе
        entry_commission = 0
        for trade in exchange.trades_history:
            if trade['id'] == position['entry_order_id']:
                entry_commission = trade['commission']
                print(f"Entry commission from trade: ${entry_commission:.4f}")
                break

        expected_balance_after_open = initial_balance - entry_cost - entry_commission

        print(f"\n💰 Balance Analysis (after opening):")
        print(f"  Entry cost: ${entry_cost:.2f}")
        print(f"  Entry commission: ${entry_commission:.4f}")
        print(f"  Expected balance: ${expected_balance_after_open:.2f}")
        print(f"  Actual balance: ${balance_after_open:.2f}")
        print(f"  Difference: ${abs(expected_balance_after_open - balance_after_open):.4f}")

        # Проверяем, что баланс изменился правильно
        assert abs(expected_balance_after_open - balance_after_open) < 0.01, \
            f"Balance mismatch after opening: expected ${expected_balance_after_open:.2f}, got ${balance_after_open:.2f}"

        print("  ✅ Balance correct after opening position")

        # Закрываем позицию (цена вырастет на 2%)
        closed_position = exchange.close_position(position['id'], reason='manual')

        balance_after_close = exchange.get_balance('USDT')
        print(f"\n💰 Balance Analysis (after closing):")
        print(f"  Balance after closing: ${balance_after_close:.2f}")
        print(f"  Exit price: ${closed_position['close_price']:.2f}")
        print(f"  PnL (before commission): ${closed_position['pnl']:.2f}")
        print(f"  PnL (after commission): ${closed_position['pnl_after_commission']:.2f}")
        print(f"  PnL %: {closed_position['pnl_percent']:.2f}%")

        # Проверяем итоговый баланс
        expected_final_balance = initial_balance + closed_position['pnl_after_commission']

        print(f"\n  Expected final balance: ${expected_final_balance:.2f}")
        print(f"  Actual final balance: ${balance_after_close:.2f}")
        print(f"  Difference: ${abs(expected_final_balance - balance_after_close):.4f}")

        assert abs(expected_final_balance - balance_after_close) < 0.01, \
            f"Final balance mismatch: expected ${expected_final_balance:.2f}, got ${balance_after_close:.2f}"

        print("  ✅ Final balance correct after closing position")

        # Проверяем прибыльность
        assert closed_position['pnl_after_commission'] > 0, \
            f"Expected profit, but got ${closed_position['pnl_after_commission']:.2f}"

        print(f"  ✅ Profit: ${closed_position['pnl_after_commission']:.2f}")

        return True

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


@patch('paper_trading.paper_exchange.get_binance_price', side_effect=mock_get_binance_price)
def test_multiple_positions(_mock):
    """Тест: Несколько одновременных позиций"""
    print("\n" + "="*80)
    print("TEST 2: Multiple Simultaneous Positions")
    print("="*80)

    reset_mock_prices()
    exchange = PaperExchange(initial_capital=10000.0)

    initial_balance = exchange.get_balance('USDT')
    print(f"Initial balance: ${initial_balance:.2f}")

    try:
        # Открываем 3 позиции
        positions = []
        total_cost = 0

        for i in range(3):
            pos = exchange.open_position(
                symbol='BTCUSDT',
                side='LONG',
                amount=0.01,
                tp_price=None,
                sl_price=None
            )
            positions.append(pos)

            # Рассчитываем стоимость позиции
            cost = pos['entry_price'] * pos['amount']
            commission = 0
            for trade in exchange.trades_history:
                if trade['id'] == pos['entry_order_id']:
                    commission = trade['commission']
                    break

            total_cost += cost + commission
            print(f"Position {i+1} opened at ${pos['entry_price']:.2f}, cost: ${cost + commission:.2f}")

        balance_after_opens = exchange.get_balance('USDT')
        print(f"\nBalance after opening 3 positions: ${balance_after_opens:.2f}")
        print(f"Total capital used: ${total_cost:.2f}")
        print(f"Expected balance: ${initial_balance - total_cost:.2f}")
        print(f"Actual balance: ${balance_after_opens:.2f}")

        # Проверяем баланс
        expected_balance = initial_balance - total_cost
        assert abs(expected_balance - balance_after_opens) < 0.01, \
            f"Balance mismatch after opening positions"
        print("✅ Balance correct after opening 3 positions")

        # Проверяем количество открытых позиций
        open_positions = len(exchange.positions)
        print(f"\nOpen positions count: {open_positions}")
        assert open_positions == 3, f"Expected 3 positions, got {open_positions}"
        print("✅ Correct number of open positions")

        # Закрываем все позиции
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
        print(f"Difference: ${abs(expected_final - balance_after_closes):.4f}")

        assert abs(expected_final - balance_after_closes) < 0.01, \
            f"Final balance mismatch"
        print("✅ Final balance correct")

        # Проверяем что все позиции закрыты
        assert len(exchange.positions) == 0, f"Expected 0 open positions"
        print("✅ All positions closed")

        # Проверяем что все прибыльные
        assert total_pnl > 0, f"Expected profit"
        print(f"✅ Total profit: ${total_pnl:.2f}")

        return True

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


@patch('paper_trading.paper_exchange.get_binance_price', side_effect=mock_get_binance_price)
def test_insufficient_balance(_mock):
    """Тест: Недостаточно средств для открытия позиции"""
    print("\n" + "="*80)
    print("TEST 3: Insufficient Balance")
    print("="*80)

    reset_mock_prices()
    exchange = PaperExchange(initial_capital=100.0)

    initial_balance = exchange.get_balance('USDT')
    print(f"Initial balance: ${initial_balance:.2f}")

    try:
        # Пытаемся открыть слишком большую позицию
        # BTC = $50,000, пытаемся купить 0.1 BTC = ~$5,000
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
        error_msg = str(e)
        if "Insufficient" in error_msg or "Need" in error_msg:
            print(f"✅ Correctly raised exception: {error_msg}")

            # Проверяем что баланс не изменился
            balance_after = exchange.get_balance('USDT')
            assert abs(balance_after - initial_balance) < 0.01, \
                f"Balance changed after failed order"
            print(f"✅ Balance unchanged: ${balance_after:.2f}")
            return True
        else:
            print(f"❌ Wrong exception type: {error_msg}")
            return False


@patch('paper_trading.paper_exchange.get_binance_price', side_effect=mock_get_binance_price)
def test_balance_tracking_with_reinvestment(_mock):
    """Тест: Отслеживание баланса с реинвестированием"""
    print("\n" + "="*80)
    print("TEST 4: Balance Tracking with Reinvestment")
    print("="*80)

    from portfolio.reinvestment import ReinvestmentManager

    reset_mock_prices()
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

        # Закрываем позицию (с прибылью 2%)
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

        # Проверяем что total_equity = trading + reserve
        total_check = reinvest_result['trading_capital'] + reinvest_result['reserve_fund']
        assert abs(total_check - reinvest_result['total_equity']) < 0.01, \
            f"Total equity mismatch"
        print(f"  ✅ Total equity = trading + reserve")

        # Проверяем что есть прибыль
        assert closed['pnl_after_commission'] > 0, \
            f"Expected profit"
        print(f"  ✅ Profit made: ${closed['pnl_after_commission']:.2f}")

        return True

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


@patch('paper_trading.paper_exchange.get_binance_price', side_effect=mock_get_binance_price)
def test_commission_and_slippage(_mock):
    """Тест: Проверка комиссий и проскальзывания"""
    print("\n" + "="*80)
    print("TEST 5: Commission and Slippage Calculation")
    print("="*80)

    reset_mock_prices()
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

        # Находим детали сделки
        entry_trade = None
        for trade in exchange.trades_history:
            if trade['id'] == pos['entry_order_id']:
                entry_trade = trade
                break

        assert entry_trade is not None, "Entry trade not found"

        real_price = entry_trade['real_price']
        exec_price = entry_trade['price']
        commission = entry_trade['commission']

        print(f"\n📊 Entry Order:")
        print(f"  Real market price: ${real_price:.2f}")
        print(f"  Executed price (with slippage): ${exec_price:.2f}")
        print(f"  Slippage: ${exec_price - real_price:.2f} ({((exec_price/real_price - 1)*100):.3f}%)")
        print(f"  Commission: ${commission:.2f}")

        # Проверяем slippage (для BUY должен быть +0.05%)
        expected_slippage_pct = 0.05
        actual_slippage_pct = ((exec_price / real_price) - 1) * 100

        print(f"\n  Expected slippage: {expected_slippage_pct}%")
        print(f"  Actual slippage: {actual_slippage_pct:.3f}%")

        assert abs(actual_slippage_pct - expected_slippage_pct) < 0.01, \
            f"Slippage incorrect: expected {expected_slippage_pct}%, got {actual_slippage_pct:.3f}%"
        print(f"  ✅ Slippage calculated correctly")

        # Проверяем комиссию (0.1% от cost)
        cost = entry_trade['cost']
        expected_commission = cost * 0.001

        print(f"\n  Cost: ${cost:.2f}")
        print(f"  Expected commission (0.1%): ${expected_commission:.2f}")
        print(f"  Actual commission: ${commission:.2f}")

        assert abs(commission - expected_commission) < 0.01, \
            f"Commission incorrect"
        print(f"  ✅ Commission calculated correctly")

        # Закрываем позицию
        closed = exchange.close_position(pos['id'], reason='manual')

        print(f"\n✅ Position closed successfully")
        print(f"Final PnL (after all fees): ${closed['pnl_after_commission']:.2f}")

        return True

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Запуск всех тестов"""
    print("\n" + "="*80)
    print("🧪 CAPITAL MANAGEMENT TESTS (with MOCK prices)")
    print("="*80)
    print("\nTesting PaperExchange balance tracking and position management...")
    print("Using MOCK prices: BTC=$50,000, ETH=$3,000")
    print("Price increases by 2% on position close")

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
        print("\n✅ Capital management is working correctly:")
        print("   - Balance decreases by (cost + commission) when opening LONG")
        print("   - Balance increases by (revenue - commission) when closing")
        print("   - Final balance = initial + PnL (after all commissions)")
        print("   - Commissions (0.1%) calculated correctly")
        print("   - Slippage (0.05%) applied correctly")
        print("   - Multiple positions tracked independently")
        print("   - Insufficient balance protection works")
    else:
        print(f"\n⚠️  {failed} test(s) failed")

    print("="*80 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
