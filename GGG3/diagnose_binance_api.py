#!/usr/bin/env python3
"""
Диагностика Binance API - проверка endpoints и данных
"""
import requests
import time
from datetime import datetime

print("=" * 80)
print("🔍 ДИАГНОСТИКА BINANCE API")
print("=" * 80)
print()

# Тестовые символы
test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']

# MAINNET endpoints
spot_endpoints = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api-gcp.binance.com",
    "https://data-api.binance.vision"
]

# Futures endpoints (для сравнения)
futures_endpoints = [
    "https://fapi.binance.com",
]

print("📊 ТЕСТ 1: Проверка доступности SPOT API endpoints")
print("-" * 80)

working_spot_endpoint = None

for endpoint in spot_endpoints:
    try:
        url = f"{endpoint}/api/v3/ping"
        start = time.time()
        response = requests.get(url, timeout=5)
        latency = (time.time() - start) * 1000

        if response.status_code == 200:
            print(f"✅ {endpoint:<45s} OK ({latency:.0f}ms)")
            if working_spot_endpoint is None:
                working_spot_endpoint = endpoint
        else:
            print(f"❌ {endpoint:<45s} HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ {endpoint:<45s} ERROR: {str(e)[:40]}")

print()
print("📊 ТЕСТ 2: Проверка доступности FUTURES API endpoints")
print("-" * 80)

working_futures_endpoint = None

for endpoint in futures_endpoints:
    try:
        url = f"{endpoint}/fapi/v1/ping"
        start = time.time()
        response = requests.get(url, timeout=5)
        latency = (time.time() - start) * 1000

        if response.status_code == 200:
            print(f"✅ {endpoint:<45s} OK ({latency:.0f}ms)")
            working_futures_endpoint = endpoint
        else:
            print(f"❌ {endpoint:<45s} HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ {endpoint:<45s} ERROR: {str(e)[:40]}")

print()
print("📊 ТЕСТ 3: Сравнение SPOT vs FUTURES цен")
print("-" * 80)

if working_spot_endpoint and working_futures_endpoint:
    for symbol in test_symbols:
        # SPOT цена
        spot_price = None
        try:
            url = f"{working_spot_endpoint}/api/v3/ticker/price"
            response = requests.get(url, params={'symbol': symbol}, timeout=5)
            if response.status_code == 200:
                spot_price = float(response.json()['price'])
        except Exception as e:
            pass

        # FUTURES цена
        futures_price = None
        try:
            url = f"{working_futures_endpoint}/fapi/v1/ticker/price"
            response = requests.get(url, params={'symbol': symbol}, timeout=5)
            if response.status_code == 200:
                futures_price = float(response.json()['price'])
        except Exception as e:
            pass

        # Вывод
        if spot_price and futures_price:
            diff_pct = abs(spot_price - futures_price) / spot_price * 100
            status = "✅" if diff_pct < 1 else "⚠️"
            print(f"{status} {symbol:<12s} SPOT: ${spot_price:>12.4f}  FUTURES: ${futures_price:>12.4f}  (Δ {diff_pct:.2f}%)")
        elif spot_price:
            print(f"⚠️ {symbol:<12s} SPOT: ${spot_price:>12.4f}  FUTURES: N/A")
        elif futures_price:
            print(f"⚠️ {symbol:<12s} SPOT: N/A            FUTURES: ${futures_price:>12.4f}")
        else:
            print(f"❌ {symbol:<12s} Обе цены недоступны")
else:
    print("❌ Не удалось получить работающие endpoints для сравнения")

print()
print("📊 ТЕСТ 4: Проверка exchangeInfo (список активных пар)")
print("-" * 80)

if working_spot_endpoint:
    try:
        url = f"{working_spot_endpoint}/api/v3/exchangeInfo"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            total_symbols = len(data['symbols'])
            usdt_pairs = [s['symbol'] for s in data['symbols']
                         if s['symbol'].endswith('USDT') and s['status'] == 'TRADING']

            print(f"✅ Всего символов: {total_symbols}")
            print(f"✅ USDT пар (активных): {len(usdt_pairs)}")
            print(f"✅ Примеры: {', '.join(usdt_pairs[:10])}")
        else:
            print(f"❌ HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
else:
    print("❌ Нет работающих SPOT endpoints")

print()
print("📊 ТЕСТ 5: Проверка klines (OHLCV данные)")
print("-" * 80)

if working_spot_endpoint:
    symbol = 'BTCUSDT'
    try:
        url = f"{working_spot_endpoint}/api/v3/klines"
        params = {'symbol': symbol, 'interval': '1h', 'limit': 1}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data:
                kline = data[0]
                timestamp = datetime.fromtimestamp(kline[0] / 1000)
                o, h, l, c, v = float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4]), float(kline[5])

                print(f"✅ {symbol} - последняя 1h свеча:")
                print(f"   Время: {timestamp}")
                print(f"   OHLC: O=${o:.2f} H=${h:.2f} L=${l:.2f} C=${c:.2f}")
                print(f"   Volume: {v:.2f}")
            else:
                print("❌ Нет данных")
        else:
            print(f"❌ HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
else:
    print("❌ Нет работающих SPOT endpoints")

print()
print("=" * 80)
print("📋 ИТОГОВАЯ ДИАГНОСТИКА")
print("=" * 80)

if working_spot_endpoint:
    print(f"✅ SPOT API работает: {working_spot_endpoint}")
else:
    print("❌ SPOT API недоступен!")

if working_futures_endpoint:
    print(f"✅ FUTURES API работает: {working_futures_endpoint}")
else:
    print("❌ FUTURES API недоступен!")

print()
print("🔍 ВОЗМОЖНЫЕ ПРОБЛЕМЫ:")
print()

if not working_spot_endpoint:
    print("❌ КРИТИЧНО: SPOT API endpoints заблокированы или недоступны")
    print("   Возможные причины:")
    print("   - Geo-restriction (VPN/прокси требуется)")
    print("   - Firewall блокирует запросы")
    print("   - Rate limit превышен")
    print()

if working_spot_endpoint and not working_futures_endpoint:
    print("⚠️ ВНИМАНИЕ: FUTURES API недоступен, но SPOT работает")
    print("   Бот должен работать на SPOT ценах")
    print()

print("💡 РЕКОМЕНДАЦИИ:")
print()
print("1. Если все endpoints недоступны:")
print("   - Проверьте интернет соединение")
print("   - Попробуйте использовать VPN")
print("   - Проверьте антивирус/firewall")
print()
print("2. Если есть большая разница SPOT vs FUTURES:")
print("   - Убедитесь что бот использует правильный endpoint")
print("   - Проверьте конфигурацию в binance_config.py")
print()
print("3. Для production использования:")
print("   - Используйте API ключи (выше rate limits)")
print("   - Настройте retry логику с экспоненциальным backoff")
print("   - Мониторьте rate limits через response headers")
print()
print("=" * 80)
