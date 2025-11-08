#!/usr/bin/env python3
"""
КРИТИЧЕСКИЙ ТЕСТ: Откуда бот берёт цены?
Проверяем что ИМЕННО возвращают методы PaperExchange
"""

import sys
sys.path.insert(0, '/home/user/jjhbn/GGG3')

from paper_trading.paper_exchange import PaperExchange
from datetime import datetime

print("=" * 80)
print("🔬 КРИТИЧЕСКИЙ ТЕСТ: Проверка источника данных")
print("=" * 80)
print()

# Создаём exchange как в боте
exchange = PaperExchange(initial_capital=1000.0)

# Тестовые символы (те которые были в логе)
test_symbols = [
    'KSMUSDT',    # Было: $13.42
    'ALICEUSDT',  # Было: $0.29
    'ILVUSDT',    # Было: $11.64
    'ICPUSDT',    # Было: $9.14
    'BNBUSDT',    # Было: $993.09 (правильно!)
    'SOLUSDT',    # Было: $157.00 (правильно!)
    'XRPUSDT',    # Было: $2.27 (правильно!)
]

print(f"🕐 Текущее время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

print("📊 ТЕСТ 1: Проверка get_ticker() - что возвращает?")
print("-" * 80)

for symbol in test_symbols:
    try:
        ticker = exchange.get_ticker(symbol)
        price = ticker.get('lastPrice', 0)
        volume = ticker.get('volume', 0)
        print(f"✅ {symbol:<12s} Price: ${price:>12.4f}  Volume: {volume:>12.2f}")
    except Exception as e:
        print(f"❌ {symbol:<12s} ERROR: {str(e)[:60]}")

print()
print("📊 ТЕСТ 2: Проверка get_ohlcv() - последняя 4h свеча")
print("-" * 80)

for symbol in test_symbols:
    try:
        df = exchange.get_ohlcv(symbol, '4h', 1)
        if len(df) > 0:
            last = df.iloc[-1]
            close = last['close']
            timestamp = last.name if hasattr(last, 'name') else 'N/A'
            print(f"✅ {symbol:<12s} Close: ${close:>12.4f}  Time: {timestamp}")
        else:
            print(f"⚠️ {symbol:<12s} Empty DataFrame!")
    except Exception as e:
        print(f"❌ {symbol:<12s} ERROR: {str(e)[:60]}")

print()
print("📊 ТЕСТ 3: Проверка create_market_order() - цена входа")
print("-" * 80)
print("Симулируем открытие позиций как в боте...")
print()

for symbol in test_symbols[:3]:  # Только первые 3 для теста
    try:
        # Симулируем покупку как в боте
        amount = 1.0  # Покупаем 1 монету
        order = exchange.create_market_order(symbol, 'BUY', amount)

        entry_price = order['price']
        print(f"✅ {symbol:<12s} Entry price: ${entry_price:>12.4f}")

        # Отменяем сразу чтобы не захламлять баланс
        exchange.balance += amount * entry_price * 1.001  # Возвращаем деньги

    except Exception as e:
        print(f"❌ {symbol:<12s} ERROR: {str(e)[:60]}")

print()
print("=" * 80)
print("📋 ИТОГОВЫЙ ВЫВОД")
print("=" * 80)
print()

# Сравнение с логом
expected_prices = {
    'KSMUSDT': 13.42,
    'ALICEUSDT': 0.29,
    'ILVUSDT': 11.64,
    'ICPUSDT': 9.14,
    'BNBUSDT': 993.09,
    'SOLUSDT': 157.00,
    'XRPUSDT': 2.27,
}

print("Сравнение с ценами из лога бота (19:14 МСК):")
print()

for symbol in test_symbols:
    try:
        ticker = exchange.get_ticker(symbol)
        current_price = ticker.get('lastPrice', 0)
        expected_price = expected_prices.get(symbol, 0)

        diff_pct = abs(current_price - expected_price) / expected_price * 100 if expected_price > 0 else 0

        if diff_pct < 5:
            status = "✅ СОВПАДАЕТ"
        elif diff_pct < 20:
            status = "⚠️ НЕБОЛЬШОЕ ОТКЛОНЕНИЕ"
        else:
            status = "❌ ОГРОМНАЯ РАЗНИЦА"

        print(f"{status}  {symbol:<12s}")
        print(f"           Лог бота:  ${expected_price:>10.4f}")
        print(f"           Сейчас:    ${current_price:>10.4f}")
        print(f"           Разница:   {diff_pct:>10.2f}%")
        print()

    except Exception as e:
        print(f"❌ ERROR  {symbol:<12s} {e}")
        print()

print("=" * 80)
print("🔍 ВОЗМОЖНЫЕ ПРИЧИНЫ РАСХОЖДЕНИЯ:")
print()
print("1. ❌ Бот использует старые кэшированные данные")
print("2. ❌ Где-то в коде есть хардкод или импорт старых данных")
print("3. ❌ PaperExchange возвращает не те данные")
print("4. ✅ Цены изменились между 19:14 и сейчас (нормально)")
print()
print("Если разница > 50% - это КРИТИЧЕСКАЯ проблема!")
print("Если разница < 10% - это нормальное изменение цен")
print("=" * 80)
