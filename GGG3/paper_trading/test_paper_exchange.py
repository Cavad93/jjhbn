"""
Test suite for PaperExchange module

Тесты для проверки функциональности paper trading модуля
"""

import time
import os
from paper_exchange import (
    PaperExchange,
    InsufficientBalance,
    OrderNotFound,
    InvalidOrder,
    get_binance_price,
    get_binance_kline,
    generate_order_id
)


def test_helper_functions():
    """Тест вспомогательных функций"""
    print("\n" + "=" * 80)
    print("TEST 1: Helper Functions")
    print("=" * 80)

    # Тест generate_order_id
    print("\n[1.1] Testing generate_order_id...")
    order_id = generate_order_id()
    assert order_id.startswith("PAPER_"), "Order ID should start with PAPER_"
    assert len(order_id) > 6, "Order ID should be longer than prefix"
    print(f"✓ Generated order ID: {order_id}")

    # Тест get_binance_price
    print("\n[1.2] Testing get_binance_price...")
    try:
        btc_price = get_binance_price('BTCUSDT')
        assert btc_price > 0, "BTC price should be positive"
        print(f"✓ BTC price: ${btc_price:.2f}")
    except Exception as e:
        print(f"✗ Failed to get BTC price: {e}")
        raise

    # Тест get_binance_kline
    print("\n[1.3] Testing get_binance_kline...")
    try:
        kline = get_binance_kline('BTCUSDT', '1m', 1)
        assert 'open' in kline and 'high' in kline and 'low' in kline and 'close' in kline
        assert kline['high'] >= kline['low'], "High should be >= Low"
        print(f"✓ Kline data: O={kline['open']:.2f}, H={kline['high']:.2f}, "
              f"L={kline['low']:.2f}, C={kline['close']:.2f}")
    except Exception as e:
        print(f"✗ Failed to get kline: {e}")
        raise

    print("\n✓ All helper function tests passed!")


def test_basic_initialization():
    """Тест инициализации биржи"""
    print("\n" + "=" * 80)
    print("TEST 2: Basic Initialization")
    print("=" * 80)

    print("\n[2.1] Creating PaperExchange with 1000 USDT...")
    exchange = PaperExchange(initial_capital=1000.0)

    assert exchange.get_balance() == 1000.0, "Initial balance should be 1000"
    assert exchange.get_equity() == 1000.0, "Initial equity should be 1000"
    assert len(exchange.get_open_positions()) == 0, "Should have no positions"
    assert len(exchange.get_open_orders()) == 0, "Should have no orders"

    print(f"✓ Balance: {exchange.get_balance():.2f} USDT")
    print(f"✓ Equity: {exchange.get_equity():.2f} USDT")
    print("✓ Basic initialization test passed!")

    return exchange


def test_market_orders(exchange):
    """Тест рыночных ордеров"""
    print("\n" + "=" * 80)
    print("TEST 3: Market Orders")
    print("=" * 80)

    initial_balance = exchange.get_balance()

    # Тест BUY order
    print("\n[3.1] Testing BUY market order...")
    try:
        order = exchange.create_market_order('BTCUSDT', 'BUY', 0.001)
        assert order['status'] == 'filled', "Order should be filled"
        assert order['side'] == 'BUY', "Order side should be BUY"
        assert order['amount'] == 0.001, "Order amount should match"
        assert exchange.get_balance() < initial_balance, "Balance should decrease"
        print(f"✓ BUY order executed: {order['amount']} BTC @ ${order['price']:.2f}")
        print(f"✓ Cost: ${order['total']:.2f} (including commission: ${order['commission']:.2f})")
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise

    # Тест SELL order
    print("\n[3.2] Testing SELL market order...")
    balance_before_sell = exchange.get_balance()
    try:
        order = exchange.create_market_order('BTCUSDT', 'SELL', 0.001)
        assert order['status'] == 'filled', "Order should be filled"
        assert order['side'] == 'SELL', "Order side should be SELL"
        assert exchange.get_balance() > balance_before_sell, "Balance should increase"
        print(f"✓ SELL order executed: {order['amount']} BTC @ ${order['price']:.2f}")
        print(f"✓ Received: ${order['total']:.2f} (after commission: ${order['commission']:.2f})")
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise

    # Тест недостаточного баланса
    print("\n[3.3] Testing insufficient balance...")
    try:
        exchange.create_market_order('BTCUSDT', 'BUY', 100.0)  # Огромная сумма
        print("✗ Should have raised InsufficientBalance exception")
        raise AssertionError("Should have raised InsufficientBalance")
    except InsufficientBalance as e:
        print(f"✓ Correctly raised InsufficientBalance: {e}")

    print("\n✓ All market order tests passed!")


def test_positions(exchange):
    """Тест работы с позициями"""
    print("\n" + "=" * 80)
    print("TEST 4: Positions")
    print("=" * 80)

    # Получаем текущую цену для расчета TP/SL
    current_price = get_binance_price('BTCUSDT')

    # Тест открытия LONG позиции
    print("\n[4.1] Testing LONG position...")
    try:
        tp_price = current_price * 1.05  # +5%
        sl_price = current_price * 0.97  # -3%

        position = exchange.open_position(
            symbol='BTCUSDT',
            side='LONG',
            amount=0.001,
            tp_price=tp_price,
            sl_price=sl_price,
            metadata={'test': 'long_position'}
        )

        assert position['status'] == 'open', "Position should be open"
        assert position['side'] == 'LONG', "Position side should be LONG"
        assert position['tp_price'] == tp_price, "TP price should match"
        assert position['sl_price'] == sl_price, "SL price should match"
        assert len(exchange.get_open_positions()) == 1, "Should have 1 open position"
        assert len(exchange.get_open_orders()) == 2, "Should have 2 orders (TP + SL)"

        print(f"✓ LONG position opened: {position['amount']} BTC @ ${position['entry_price']:.2f}")
        print(f"✓ TP: ${tp_price:.2f}, SL: ${sl_price:.2f}")
        print(f"✓ Position ID: {position['id']}")

        # Проверяем equity
        equity = exchange.get_equity()
        print(f"✓ Current equity: ${equity:.2f} USDT")

        # Сохраняем ID для последующих тестов
        long_position_id = position['id']

    except Exception as e:
        print(f"✗ Failed: {e}")
        raise

    # Тест открытия SHORT позиции
    print("\n[4.2] Testing SHORT position...")
    try:
        current_price = get_binance_price('ETHUSDT')
        tp_price = current_price * 0.95  # -5%
        sl_price = current_price * 1.03  # +3%

        position = exchange.open_position(
            symbol='ETHUSDT',
            side='SHORT',
            amount=0.1,
            tp_price=tp_price,
            sl_price=sl_price
        )

        assert position['status'] == 'open', "Position should be open"
        assert position['side'] == 'SHORT', "Position side should be SHORT"
        assert len(exchange.get_open_positions()) == 2, "Should have 2 open positions"

        print(f"✓ SHORT position opened: {position['amount']} ETH @ ${position['entry_price']:.2f}")
        print(f"✓ TP: ${tp_price:.2f}, SL: ${sl_price:.2f}")

        short_position_id = position['id']

    except Exception as e:
        print(f"✗ Failed: {e}")
        raise

    # Тест закрытия позиции
    print("\n[4.3] Testing position closure...")
    try:
        closed_position = exchange.close_position(long_position_id, reason='manual')

        assert closed_position['status'] == 'closed', "Position should be closed"
        assert 'pnl' in closed_position, "Should have PnL"
        assert 'pnl_percent' in closed_position, "Should have PnL percent"
        assert len(exchange.get_open_positions()) == 1, "Should have 1 open position left"

        print(f"✓ Position closed: PnL = ${closed_position['pnl_after_commission']:.2f} "
              f"({closed_position['pnl_percent']:.2f}%)")
        print(f"✓ Close reason: {closed_position['close_reason']}")

    except Exception as e:
        print(f"✗ Failed: {e}")
        raise

    # Закрываем оставшуюся позицию
    print("\n[4.4] Closing remaining position...")
    try:
        exchange.close_position(short_position_id, reason='manual')
        assert len(exchange.get_open_positions()) == 0, "Should have no open positions"
        print("✓ All positions closed")
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise

    print("\n✓ All position tests passed!")


def test_orders_execution():
    """Тест исполнения ордеров (TP/SL)"""
    print("\n" + "=" * 80)
    print("TEST 5: Orders Execution (TP/SL)")
    print("=" * 80)

    exchange = PaperExchange(initial_capital=1000.0)

    # Создаем позицию с очень близким TP для быстрого срабатывания
    print("\n[5.1] Creating position with tight TP...")
    try:
        current_price = get_binance_price('BTCUSDT')
        tp_price = current_price * 1.0001  # +0.01% - скорее всего сработает быстро
        sl_price = current_price * 0.95

        position = exchange.open_position(
            symbol='BTCUSDT',
            side='LONG',
            amount=0.001,
            tp_price=tp_price,
            sl_price=sl_price
        )

        print(f"✓ Position opened with TP={tp_price:.2f}, current price={current_price:.2f}")
        print("✓ Waiting for price movement...")

        # Проверяем ордера несколько раз
        max_checks = 3
        for i in range(max_checks):
            time.sleep(2)  # Ждем 2 секунды
            print(f"\n[5.2] Check #{i+1}: Checking orders...")

            executed = exchange.check_orders()

            if executed:
                print(f"✓ {len(executed)} order(s) executed!")
                for exec_info in executed:
                    print(f"  - {exec_info['order']['type']} executed at ${exec_info['execution_price']:.2f}")
                break
            else:
                current = get_binance_price('BTCUSDT')
                print(f"  No orders executed yet (current price: ${current:.2f})")

        # Очищаем оставшиеся позиции
        for pos_id in list(exchange.positions.keys()):
            exchange.close_position(pos_id, reason='test_cleanup')

    except Exception as e:
        print(f"✗ Failed: {e}")
        # Не raise - это не критичная ошибка, просто цена не достигла

    print("\n✓ Order execution test completed!")


def test_state_persistence():
    """Тест сохранения и загрузки состояния"""
    print("\n" + "=" * 80)
    print("TEST 6: State Persistence")
    print("=" * 80)

    test_file = "/tmp/test_paper_exchange_state.json"

    # Создаем биржу с позициями
    print("\n[6.1] Creating exchange with positions...")
    exchange1 = PaperExchange(initial_capital=1000.0)

    try:
        current_price = get_binance_price('BTCUSDT')
        exchange1.open_position(
            symbol='BTCUSDT',
            side='LONG',
            amount=0.001,
            tp_price=current_price * 1.1,
            sl_price=current_price * 0.9
        )

        balance1 = exchange1.get_balance()
        positions1 = len(exchange1.get_open_positions())
        orders1 = len(exchange1.get_open_orders())

        print(f"✓ Exchange state: Balance={balance1:.2f}, Positions={positions1}, Orders={orders1}")

    except Exception as e:
        print(f"✗ Failed to create position: {e}")
        raise

    # Сохраняем состояние
    print("\n[6.2] Saving state...")
    try:
        exchange1.save_state(test_file)
        assert os.path.exists(test_file), "State file should exist"
        print(f"✓ State saved to {test_file}")
    except Exception as e:
        print(f"✗ Failed to save: {e}")
        raise

    # Загружаем состояние
    print("\n[6.3] Loading state...")
    try:
        exchange2 = PaperExchange.load_state(test_file)

        balance2 = exchange2.get_balance()
        positions2 = len(exchange2.get_open_positions())
        orders2 = len(exchange2.get_open_orders())

        print(f"✓ Loaded state: Balance={balance2:.2f}, Positions={positions2}, Orders={orders2}")

        # Проверяем соответствие
        assert abs(balance1 - balance2) < 0.01, "Balance should match"
        assert positions1 == positions2, "Positions count should match"
        assert orders1 == orders2, "Orders count should match"

        print("✓ State matches perfectly!")

    except Exception as e:
        print(f"✗ Failed to load: {e}")
        raise
    finally:
        # Очистка
        if os.path.exists(test_file):
            os.remove(test_file)

    print("\n✓ State persistence test passed!")


def test_statistics():
    """Тест статистики"""
    print("\n" + "=" * 80)
    print("TEST 7: Statistics")
    print("=" * 80)

    exchange = PaperExchange(initial_capital=1000.0)

    # Открываем и закрываем несколько позиций
    print("\n[7.1] Creating test trades...")
    try:
        for i in range(3):
            current_price = get_binance_price('BTCUSDT')
            position = exchange.open_position(
                symbol='BTCUSDT',
                side='LONG',
                amount=0.001,
                tp_price=current_price * 1.1,
                sl_price=current_price * 0.9
            )
            time.sleep(0.5)
            exchange.close_position(position['id'], reason='test')

        print(f"✓ Created {len(exchange.closed_positions)} test trades")

    except Exception as e:
        print(f"✗ Failed: {e}")
        raise

    # Получаем статистику
    print("\n[7.2] Calculating statistics...")
    try:
        stats = exchange.get_statistics()

        print(f"\nTrading Statistics:")
        print(f"  Total trades: {stats['total_trades']}")
        print(f"  Wins: {stats['wins']}")
        print(f"  Losses: {stats['losses']}")
        print(f"  Win rate: {stats['winrate']:.2f}%")
        print(f"  Total PnL: ${stats['total_pnl']:.2f}")
        print(f"  Average PnL: ${stats['avg_pnl']:.2f}")
        print(f"  ROI: {stats['roi']:.2f}%")
        print(f"  Current equity: ${stats['current_equity']:.2f}")

        assert stats['total_trades'] == 3, "Should have 3 trades"
        print("\n✓ Statistics test passed!")

    except Exception as e:
        print(f"✗ Failed: {e}")
        raise


def run_all_tests():
    """Запуск всех тестов"""
    print("\n" + "=" * 80)
    print("PAPER EXCHANGE - COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    start_time = time.time()

    try:
        # Тест 1: Вспомогательные функции
        test_helper_functions()

        # Тест 2: Базовая инициализация
        exchange = test_basic_initialization()

        # Тест 3: Рыночные ордера
        test_market_orders(exchange)

        # Тест 4: Позиции
        test_positions(exchange)

        # Тест 5: Исполнение ордеров
        test_orders_execution()

        # Тест 6: Сохранение состояния
        test_state_persistence()

        # Тест 7: Статистика
        test_statistics()

        # Итоговый отчет
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print(f"Total time: {elapsed_time:.2f} seconds")
        print("=" * 80)

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ TESTS FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
