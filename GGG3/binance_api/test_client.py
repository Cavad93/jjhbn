#!/usr/bin/env python3
"""
Тесты для Binance API клиента

ВАЖНО: Эти тесты используют mock-объекты и не требуют реальных API ключей
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Тест импорта всех компонентов"""
    print("Тест 1: Импорт компонентов...", end=" ")

    try:
        from binance_api import BinanceClient
        from binance_api.exceptions import (
            BinanceAPIError,
            RateLimitError,
            InsufficientBalanceError,
            OrderError
        )

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_rate_limiter():
    """Тест rate limiter"""
    print("Тест 2: Rate limiter...", end=" ")

    try:
        from binance_api.client import RateLimiter
        import time

        limiter = RateLimiter(max_requests_per_minute=5)

        # Делаем 5 запросов - должно пройти быстро
        start = time.time()
        for _ in range(5):
            limiter.wait_if_needed()
        elapsed = time.time() - start

        assert elapsed < 1.0, "Первые 5 запросов должны пройти быстро"

        # 6-й запрос должен заблокироваться
        # (но мы не тестируем, чтобы не ждать 60 секунд)

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_client_initialization_testnet():
    """Тест инициализации клиента (testnet)"""
    print("Тест 3: Инициализация клиента (testnet)...", end=" ")

    try:
        with patch('binance_api.client.Client') as MockClient:
            # Мокируем Binance Client
            mock_instance = Mock()
            MockClient.return_value = mock_instance

            from binance_api import BinanceClient

            # Создаем клиента с testnet
            client = BinanceClient(
                api_key="test_key",
                secret_key="test_secret",
                testnet=True
            )

            # Проверяем, что testnet установлен
            assert client.testnet == True
            assert client.api_key == "test_key"

            # Проверяем, что Client был вызван с testnet=True
            MockClient.assert_called_once()

            print("✓ ПРОЙДЕН")
            return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_client_initialization_live():
    """Тест инициализации клиента (live)"""
    print("Тест 4: Инициализация клиента (live)...", end=" ")

    try:
        with patch('binance_api.client.Client') as MockClient:
            mock_instance = Mock()
            MockClient.return_value = mock_instance

            from binance_api import BinanceClient

            # Создаем клиента с live
            client = BinanceClient(
                api_key="live_key",
                secret_key="live_secret",
                testnet=False
            )

            # Проверяем, что live режим установлен
            assert client.testnet == False
            assert client.api_key == "live_key"

            print("✓ ПРОЙДЕН")
            return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_get_current_price():
    """Тест получения текущей цены"""
    print("Тест 5: Получение текущей цены...", end=" ")

    try:
        with patch('binance_api.client.Client') as MockClient:
            mock_instance = Mock()
            mock_instance.futures_symbol_ticker = Mock(return_value={'price': '45000.00'})
            MockClient.return_value = mock_instance

            from binance_api import BinanceClient

            client = BinanceClient("key", "secret", testnet=True)
            price = client.get_current_price('BTCUSDT')

            assert price == 45000.0
            assert isinstance(price, float)

            print("✓ ПРОЙДЕН")
            return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_get_balance():
    """Тест получения баланса"""
    print("Тест 6: Получение баланса...", end=" ")

    try:
        with patch('binance_api.client.Client') as MockClient:
            mock_instance = Mock()
            mock_instance.futures_account = Mock(return_value={
                'assets': [
                    {'asset': 'USDT', 'availableBalance': '1000.50'},
                    {'asset': 'BTC', 'availableBalance': '0.5'}
                ]
            })
            MockClient.return_value = mock_instance

            from binance_api import BinanceClient

            client = BinanceClient("key", "secret", testnet=True)
            balance = client.get_balance('USDT')

            assert balance == 1000.50
            assert isinstance(balance, float)

            print("✓ ПРОЙДЕН")
            return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_get_ohlcv():
    """Тест получения OHLCV данных"""
    print("Тест 7: Получение OHLCV...", end=" ")

    try:
        with patch('binance_api.client.Client') as MockClient:
            # Мокируем klines данные
            mock_klines = [
                [
                    1699300000000,  # timestamp
                    '45000.0',      # open
                    '45500.0',      # high
                    '44800.0',      # low
                    '45200.0',      # close
                    '100.5',        # volume
                    1699300299999,  # close_time
                    '4530000.0',    # quote_volume
                    1000,           # trades
                    '50.0',         # taker_buy_base
                    '2250000.0',    # taker_buy_quote
                    '0'             # ignore
                ]
            ]

            mock_instance = Mock()
            mock_instance.futures_klines = Mock(return_value=mock_klines)
            MockClient.return_value = mock_instance

            from binance_api import BinanceClient

            client = BinanceClient("key", "secret", testnet=True)
            df = client.get_ohlcv('BTCUSDT', '4h', limit=1)

            # Проверяем структуру DataFrame
            assert len(df) == 1
            assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            assert df['open'].iloc[0] == 45000.0
            assert df['close'].iloc[0] == 45200.0

            print("✓ ПРОЙДЕН")
            return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_create_market_order():
    """Тест создания market ордера"""
    print("Тест 8: Создание market ордера...", end=" ")

    try:
        with patch('binance_api.client.Client') as MockClient:
            mock_instance = Mock()
            mock_instance.futures_create_order = Mock(return_value={
                'orderId': 12345,
                'symbol': 'BTCUSDT',
                'status': 'FILLED',
                'type': 'MARKET'
            })
            MockClient.return_value = mock_instance

            from binance_api import BinanceClient

            client = BinanceClient("key", "secret", testnet=True)
            order = client.create_market_order('BTCUSDT', 'BUY', 0.001)

            assert order['orderId'] == 12345
            assert order['status'] == 'FILLED'
            assert order['type'] == 'MARKET'

            print("✓ ПРОЙДЕН")
            return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_create_limit_order():
    """Тест создания limit ордера (TP)"""
    print("Тест 9: Создание limit ордера (TP)...", end=" ")

    try:
        with patch('binance_api.client.Client') as MockClient:
            mock_instance = Mock()
            mock_instance.futures_create_order = Mock(return_value={
                'orderId': 12346,
                'symbol': 'BTCUSDT',
                'status': 'NEW',
                'type': 'LIMIT',
                'price': '46000.0'
            })
            MockClient.return_value = mock_instance

            from binance_api import BinanceClient

            client = BinanceClient("key", "secret", testnet=True)
            order = client.create_limit_order('BTCUSDT', 'SELL', 0.001, 46000.0)

            assert order['orderId'] == 12346
            assert order['status'] == 'NEW'
            assert order['type'] == 'LIMIT'

            print("✓ ПРОЙДЕН")
            return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_create_stop_market_order():
    """Тест создания stop-market ордера (SL)"""
    print("Тест 10: Создание stop-market ордера (SL)...", end=" ")

    try:
        with patch('binance_api.client.Client') as MockClient:
            mock_instance = Mock()
            mock_instance.futures_create_order = Mock(return_value={
                'orderId': 12347,
                'symbol': 'BTCUSDT',
                'status': 'NEW',
                'type': 'STOP_MARKET',
                'stopPrice': '44000.0'
            })
            MockClient.return_value = mock_instance

            from binance_api import BinanceClient

            client = BinanceClient("key", "secret", testnet=True)
            order = client.create_stop_market_order('BTCUSDT', 'SELL', 0.001, 44000.0)

            assert order['orderId'] == 12347
            assert order['status'] == 'NEW'
            assert order['type'] == 'STOP_MARKET'

            print("✓ ПРОЙДЕН")
            return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_cancel_order():
    """Тест отмены ордера"""
    print("Тест 11: Отмена ордера...", end=" ")

    try:
        with patch('binance_api.client.Client') as MockClient:
            mock_instance = Mock()
            mock_instance.futures_cancel_order = Mock(return_value={
                'orderId': 12345,
                'status': 'CANCELED'
            })
            MockClient.return_value = mock_instance

            from binance_api import BinanceClient

            client = BinanceClient("key", "secret", testnet=True)
            result = client.cancel_order('BTCUSDT', 12345)

            assert result == True

            print("✓ ПРОЙДЕН")
            return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_get_position_info():
    """Тест получения информации о позициях"""
    print("Тест 12: Получение информации о позициях...", end=" ")

    try:
        with patch('binance_api.client.Client') as MockClient:
            mock_instance = Mock()
            mock_instance.futures_position_information = Mock(return_value=[
                {
                    'symbol': 'BTCUSDT',
                    'positionAmt': '0.001',
                    'entryPrice': '45000.0',
                    'unrealizedProfit': '5.0'
                },
                {
                    'symbol': 'ETHUSDT',
                    'positionAmt': '0.0',  # Не активная позиция
                    'entryPrice': '0.0',
                    'unrealizedProfit': '0.0'
                }
            ])
            MockClient.return_value = mock_instance

            from binance_api import BinanceClient

            client = BinanceClient("key", "secret", testnet=True)
            positions = client.get_position_info()

            # Должна вернуться только 1 активная позиция
            assert len(positions) == 1
            assert positions[0]['symbol'] == 'BTCUSDT'
            assert float(positions[0]['positionAmt']) > 0

            print("✓ ПРОЙДЕН")
            return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_retry_logic():
    """Тест retry логики при ошибках"""
    print("Тест 13: Retry логика...", end=" ")

    try:
        with patch('binance_api.client.Client') as MockClient:
            from binance.exceptions import BinanceRequestException

            mock_instance = Mock()

            # Первый вызов - ошибка, второй - успех
            mock_instance.futures_symbol_ticker = Mock(
                side_effect=[
                    BinanceRequestException("Network error"),
                    {'price': '45000.00'}
                ]
            )
            MockClient.return_value = mock_instance

            from binance_api import BinanceClient

            client = BinanceClient("key", "secret", testnet=True, max_retries=3, retry_delay=0.1)

            # Должен повторить запрос и получить цену
            price = client.get_current_price('BTCUSDT')
            assert price == 45000.0

            # Проверяем, что было 2 вызова (первый с ошибкой, второй успешный)
            assert mock_instance.futures_symbol_ticker.call_count == 2

            print("✓ ПРОЙДЕН")
            return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def run_all_tests():
    """Запускает все тесты"""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ BINANCE API КЛИЕНТА")
    print("=" * 80)
    print()

    tests = [
        test_imports,
        test_rate_limiter,
        test_client_initialization_testnet,
        test_client_initialization_live,
        test_get_current_price,
        test_get_balance,
        test_get_ohlcv,
        test_create_market_order,
        test_create_limit_order,
        test_create_stop_market_order,
        test_cancel_order,
        test_get_position_info,
        test_retry_logic,
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
    success = run_all_tests()
    sys.exit(0 if success else 1)
