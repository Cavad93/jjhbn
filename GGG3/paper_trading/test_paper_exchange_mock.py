"""
Test suite for PaperExchange module with mocked API calls

Тесты с mock'ами для проверки логики без зависимости от реального API
"""

import time
import os
import json
from unittest.mock import patch, MagicMock


# Mock для get_binance_price
def mock_get_binance_price(symbol):
    """Mock функция для получения цены"""
    prices = {
        'BTCUSDT': 45000.0,
        'ETHUSDT': 2500.0,
        'BNBUSDT': 350.0
    }
    return prices.get(symbol, 1000.0)


# Mock для get_binance_kline
def mock_get_binance_kline(symbol, interval='1m', limit=1):
    """Mock функция для получения свечи"""
    base_price = mock_get_binance_price(symbol)
    return {
        'open': base_price * 0.999,
        'high': base_price * 1.002,
        'low': base_price * 0.998,
        'close': base_price,
        'volume': 1000000.0,
        'timestamp': int(time.time() * 1000)
    }


# Применяем патчи
with patch('paper_exchange.get_binance_price', side_effect=mock_get_binance_price):
    with patch('paper_exchange.get_binance_kline', side_effect=mock_get_binance_kline):
        from paper_exchange import (
            PaperExchange,
            InsufficientBalance,
            OrderNotFound,
            InvalidOrder,
            generate_order_id
        )


def test_basic_initialization():
    """Тест инициализации биржи"""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Initialization")
    print("=" * 80)

    print("\n[1.1] Creating PaperExchange with 1000 USDT...")
    exchange = PaperExchange(initial_capital=1000.0)

    assert exchange.get_balance() == 1000.0, "Initial balance should be 1000"
    assert exchange.get_equity() == 1000.0, "Initial equity should be 1000"
    assert len(exchange.get_open_positions()) == 0, "Should have no positions"
    assert len(exchange.get_open_orders()) == 0, "Should have no orders"

    print(f"✓ Balance: {exchange.get_balance():.2f} USDT")
    print(f"✓ Equity: {exchange.get_equity():.2f} USDT")
    print("✓ Basic initialization test passed!")

    return exchange


@patch('paper_exchange.get_binance_price', side_effect=mock_get_binance_price)
@patch('paper_exchange.get_binance_kline', side_effect=mock_get_binance_kline)
def test_market_orders(mock_kline, mock_price):
    """Тест рыночных ордеров"""
    print("\n" + "=" * 80)
    print("TEST 2: Market Orders")
    print("=" * 80)

    exchange = PaperExchange(initial_capital=1000.0)
    initial_balance = exchange.get_balance()

    # Тест BUY order
    print("\n[2.1] Testing BUY market order...")
    try:
        order = exchange.create_market_order('BTCUSDT', 'BUY', 0.01)
        assert order['status'] == 'filled', "Order should be filled"
        assert order['side'] == 'BUY', "Order side should be BUY"
        assert order['amount'] == 0.01, "Order amount should match"
        assert exchange.get_balance() < initial_balance, "Balance should decrease"
        print(f"✓ BUY order executed: {order['amount']} BTC @ ${order['price']:.2f}")
        print(f"✓ Cost: ${order['total']:.2f} (including commission: ${order['commission']:.2f})")
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise

    # Тест SELL order
    print("\n[2.2] Testing SELL market order...")
    balance_before_sell = exchange.get_balance()
    try:
        order = exchange.create_market_order('BTCUSDT', 'SELL', 0.01)
        assert order['status'] == 'filled', "Order should be filled"
        assert order['side'] == 'SELL', "Order side should be SELL"
        assert exchange.get_balance() > balance_before_sell, "Balance should increase"
        print(f"✓ SELL order executed: {order['amount']} BTC @ ${order['price']:.2f}")
        print(f"✓ Received: ${order['total']:.2f} (after commission: ${order['commission']:.2f})")
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise

    # Тест недостаточного баланса
    print("\n[2.3] Testing insufficient balance...")
    try:
        exchange.create_market_order('BTCUSDT', 'BUY', 100.0)  # Огромная сумма
        print("✗ Should have raised InsufficientBalance exception")
        raise AssertionError("Should have raised InsufficientBalance")
    except InsufficientBalance as e:
        print(f"✓ Correctly raised InsufficientBalance: {e}")

    # Тест некорректного side
    print("\n[2.4] Testing invalid order side...")
    try:
        exchange.create_market_order('BTCUSDT', 'INVALID', 0.01)
        print("✗ Should have raised InvalidOrder exception")
        raise AssertionError("Should have raised InvalidOrder")
    except InvalidOrder as e:
        print(f"✓ Correctly raised InvalidOrder: {e}")

    print("\n✓ All market order tests passed!")


@patch('paper_exchange.get_binance_price', side_effect=mock_get_binance_price)
@patch('paper_exchange.get_binance_kline', side_effect=mock_get_binance_kline)
def test_positions(mock_kline, mock_price):
    """Тест работы с позициями"""
    print("\n" + "=" * 80)
    print("TEST 3: Positions")
    print("=" * 80)

    exchange = PaperExchange(initial_capital=10000.0)

    # Получаем "текущую" цену для расчета TP/SL
    current_price = mock_get_binance_price('BTCUSDT')

    # Тест открытия LONG позиции
    print("\n[3.1] Testing LONG position...")
    try:
        tp_price = current_price * 1.05  # +5%
        sl_price = current_price * 0.97  # -3%

        position = exchange.open_position(
            symbol='BTCUSDT',
            side='LONG',
            amount=0.1,
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

        long_position_id = position['id']

    except Exception as e:
        print(f"✗ Failed: {e}")
        raise

    # Тест открытия SHORT позиции
    print("\n[3.2] Testing SHORT position...")
    try:
        current_price = mock_get_binance_price('ETHUSDT')
        tp_price = current_price * 0.95  # -5%
        sl_price = current_price * 1.03  # +3%

        position = exchange.open_position(
            symbol='ETHUSDT',
            side='SHORT',
            amount=1.0,
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

    # Тест получения позиции
    print("\n[3.3] Testing get_position...")
    try:
        pos = exchange.get_position(long_position_id)
        assert pos is not None, "Position should exist"
        assert pos['id'] == long_position_id, "Position ID should match"
        print(f"✓ Retrieved position: {pos['symbol']} {pos['side']}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise

    # Тест закрытия позиции
    print("\n[3.4] Testing position closure...")
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

    # Тест ошибки при закрытии несуществующей позиции
    print("\n[3.5] Testing close non-existent position...")
    try:
        exchange.close_position('FAKE_ID', reason='manual')
        print("✗ Should have raised OrderNotFound exception")
        raise AssertionError("Should have raised OrderNotFound")
    except OrderNotFound as e:
        print(f"✓ Correctly raised OrderNotFound: {e}")

    # Закрываем оставшуюся позицию
    print("\n[3.6] Closing remaining position...")
    try:
        exchange.close_position(short_position_id, reason='manual')
        assert len(exchange.get_open_positions()) == 0, "Should have no open positions"
        print("✓ All positions closed")
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise

    print("\n✓ All position tests passed!")


@patch('paper_exchange.get_binance_price', side_effect=mock_get_binance_price)
@patch('paper_exchange.get_binance_kline', side_effect=mock_get_binance_kline)
def test_state_persistence(mock_kline, mock_price):
    """Тест сохранения и загрузки состояния"""
    print("\n" + "=" * 80)
    print("TEST 4: State Persistence")
    print("=" * 80)

    test_file = "/tmp/test_paper_exchange_mock.json"

    # Создаем биржу с позициями
    print("\n[4.1] Creating exchange with positions...")
    exchange1 = PaperExchange(initial_capital=1000.0)

    try:
        current_price = mock_get_binance_price('BTCUSDT')
        exchange1.open_position(
            symbol='BTCUSDT',
            side='LONG',
            amount=0.01,
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
    print("\n[4.2] Saving state...")
    try:
        exchange1.save_state(test_file)
        assert os.path.exists(test_file), "State file should exist"
        print(f"✓ State saved to {test_file}")

        # Проверяем содержимое файла
        with open(test_file, 'r') as f:
            state_data = json.load(f)
        print(f"✓ State file contains {len(state_data)} keys")

    except Exception as e:
        print(f"✗ Failed to save: {e}")
        raise

    # Загружаем состояние
    print("\n[4.3] Loading state...")
    try:
        exchange2 = PaperExchange.load_state(test_file)

        balance2 = exchange2.get_balance()
        positions2 = len(exchange2.get_open_positions())
        orders2 = len(exchange2.get_open_orders())

        print(f"✓ Loaded state: Balance={balance2:.2f}, Positions={positions2}, Orders={orders2}")

        # Проверяем соответствие
        assert abs(balance1 - balance2) < 0.01, f"Balance mismatch: {balance1} vs {balance2}"
        assert positions1 == positions2, f"Positions mismatch: {positions1} vs {positions2}"
        assert orders1 == orders2, f"Orders mismatch: {orders1} vs {orders2}"

        print("✓ State matches perfectly!")

    except Exception as e:
        print(f"✗ Failed to load: {e}")
        raise
    finally:
        # Очистка
        if os.path.exists(test_file):
            os.remove(test_file)
            print("✓ Test file cleaned up")

    print("\n✓ State persistence test passed!")


@patch('paper_exchange.get_binance_price', side_effect=mock_get_binance_price)
@patch('paper_exchange.get_binance_kline', side_effect=mock_get_binance_kline)
def test_statistics(mock_kline, mock_price):
    """Тест статистики"""
    print("\n" + "=" * 80)
    print("TEST 5: Statistics")
    print("=" * 80)

    exchange = PaperExchange(initial_capital=10000.0)

    # Открываем и закрываем несколько позиций
    print("\n[5.1] Creating test trades...")
    try:
        for i in range(5):
            current_price = mock_get_binance_price('BTCUSDT')
            position = exchange.open_position(
                symbol='BTCUSDT',
                side='LONG',
                amount=0.01,
                tp_price=current_price * 1.1,
                sl_price=current_price * 0.9
            )
            time.sleep(0.1)
            exchange.close_position(position['id'], reason='test')

        print(f"✓ Created {len(exchange.closed_positions)} test trades")

    except Exception as e:
        print(f"✗ Failed: {e}")
        raise

    # Получаем статистику
    print("\n[5.2] Calculating statistics...")
    try:
        stats = exchange.get_statistics()

        print(f"\nTrading Statistics:")
        print(f"  Total trades: {stats['total_trades']}")
        print(f"  Wins: {stats['wins']}")
        print(f"  Losses: {stats['losses']}")
        print(f"  Win rate: {stats['winrate']:.2f}%")
        print(f"  Total PnL: ${stats['total_pnl']:.2f}")
        print(f"  Average PnL: ${stats['avg_pnl']:.2f}")
        print(f"  Average Win: ${stats['avg_win']:.2f}")
        print(f"  Average Loss: ${stats['avg_loss']:.2f}")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  ROI: {stats['roi']:.2f}%")
        print(f"  Current equity: ${stats['current_equity']:.2f}")

        assert stats['total_trades'] == 5, "Should have 5 trades"
        assert 'winrate' in stats, "Should have winrate"
        assert 'roi' in stats, "Should have ROI"

        print("\n✓ Statistics test passed!")

    except Exception as e:
        print(f"✗ Failed: {e}")
        raise


def run_all_tests():
    """Запуск всех тестов"""
    print("\n" + "=" * 80)
    print("PAPER EXCHANGE - MOCK TEST SUITE")
    print("=" * 80)

    start_time = time.time()

    try:
        # Тест 1: Базовая инициализация
        test_basic_initialization()

        # Тест 2: Рыночные ордера
        test_market_orders()

        # Тест 3: Позиции
        test_positions()

        # Тест 4: Сохранение состояния
        test_state_persistence()

        # Тест 5: Статистика
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
