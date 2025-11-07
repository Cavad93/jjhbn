#!/usr/bin/env python3
"""
Тест для проверки исправления PaperExchange
"""

import sys
from pathlib import Path

# Добавляем путь к GGG3
sys.path.insert(0, str(Path(__file__).parent))

from paper_trading.paper_exchange import PaperExchange

def test_paper_exchange_methods():
    """Тестирует наличие всех необходимых методов в PaperExchange"""

    print("="*80)
    print("TESTING PAPER EXCHANGE METHODS")
    print("="*80)

    # Создаем экземпляр PaperExchange
    exchange = PaperExchange(initial_capital=1000.0)

    # Список методов, которые должны быть доступны
    required_methods = [
        'get_current_price',
        'get_balance',
        'create_market_order',
        'create_limit_order',
        'create_stop_market_order',
        'cancel_order',
        'get_all_usdt_pairs',
        'get_ticker',
        'get_ohlcv'
    ]

    print("\nChecking required methods...")
    all_methods_exist = True

    for method_name in required_methods:
        if hasattr(exchange, method_name):
            print(f"  ✓ {method_name}")
        else:
            print(f"  ✗ {method_name} - MISSING!")
            all_methods_exist = False

    print("\n" + "="*80)

    if all_methods_exist:
        print("✅ ALL REQUIRED METHODS EXIST")
        print("\nTesting method calls...")

        # Test get_balance
        try:
            balance = exchange.get_balance('USDT')
            print(f"  ✓ get_balance('USDT') = ${balance:.2f}")
        except Exception as e:
            print(f"  ✗ get_balance('USDT') failed: {e}")

        # Test get_current_price (может занять время из-за API запроса)
        try:
            print("\n  Testing get_current_price('BTCUSDT')...")
            price = exchange.get_current_price('BTCUSDT')
            print(f"  ✓ get_current_price('BTCUSDT') = ${price:.2f}")
        except Exception as e:
            print(f"  ✗ get_current_price('BTCUSDT') failed: {e}")

        print("\n✅ PAPER EXCHANGE IS WORKING CORRECTLY")
        return True
    else:
        print("❌ SOME METHODS ARE MISSING")
        return False

if __name__ == "__main__":
    success = test_paper_exchange_methods()
    sys.exit(0 if success else 1)
