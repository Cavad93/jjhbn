# Binance API Information

## ✅ ВАЖНО: Используется РЕАЛЬНЫЙ Binance Mainnet

Модуль paper_trading использует **РЕАЛЬНЫЕ** данные с Binance mainnet, **НЕ testnet**.

## Endpoints

### Mainnet URLs (используются)

```
✓ Primary:    https://api.binance.com
✓ Fallback 1: https://api1.binance.com
✓ Fallback 2: https://api-gcp.binance.com
✓ Fallback 3: https://data-api.binance.vision
```

### Testnet URLs (НЕ используются!)

```
✗ https://testnet.binance.vision          - TESTNET (не используем)
✗ https://testnet.binancefuture.com       - TESTNET (не используем)
```

## Public Endpoints (без API ключей)

### 1. Ticker Price

**Endpoint:** `GET /api/v3/ticker/price`

**Пример запроса:**
```bash
curl -X GET "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
```

**Ответ:**
```json
{
  "symbol": "BTCUSDT",
  "price": "45000.00"
}
```

**Использование в коде:**
```python
from paper_trading import get_binance_price

price = get_binance_price('BTCUSDT')
print(f"BTC: ${price:,.2f}")
```

### 2. Klines (Candlestick Data)

**Endpoint:** `GET /api/v3/klines`

**Параметры:**
- `symbol` (required) - торговая пара (например, BTCUSDT)
- `interval` (required) - интервал: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
- `limit` (optional) - количество свечей (по умолчанию 500, максимум 1000)
- `startTime` (optional) - начальное время в миллисекундах
- `endTime` (optional) - конечное время в миллисекундах

**Пример запроса:**
```bash
curl -X GET "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1"
```

**Ответ:**
```json
[
  [
    1499040000000,      // Open time
    "0.01634790",       // Open
    "0.80000000",       // High
    "0.01575800",       // Low
    "0.01577100",       // Close
    "148976.11427815",  // Volume
    1499644799999,      // Close time
    "2434.19055334",    // Quote asset volume
    308,                // Number of trades
    "1756.87402397",    // Taker buy base asset volume
    "28.46694368",      // Taker buy quote asset volume
    "0"                 // Unused field, ignore
  ]
]
```

**Использование в коде:**
```python
from paper_trading import get_binance_kline

kline = get_binance_kline('BTCUSDT', '1h', 1)
print(f"Open: {kline['open']}, High: {kline['high']}, Low: {kline['low']}, Close: {kline['close']}")
```

## API Keys

**❌ API ключи НЕ требуются!**

Оба endpoint'а являются публичными и предоставляют рыночные данные без аутентификации.

## Rate Limits

### Без API ключей:
- **Weight limit:** 1200 запросов в минуту на IP
- **ticker/price weight:** 1 (очень легкий)
- **klines weight:** 1 (очень легкий)

Наше использование находится в безопасных пределах для публичных endpoints.

## Fallback Mechanism

Модуль автоматически пробует все mainnet endpoints по порядку:

1. Пробует `api.binance.com` (primary)
2. Если ошибка → пробует `api1.binance.com` (alternative)
3. Если ошибка → пробует `api-gcp.binance.com` (GCP CDN)
4. Если ошибка → пробует `data-api.binance.vision` (historical data)
5. Если все failed → выбрасывает исключение

Это обеспечивает надежность при временных проблемах с одним из endpoints.

## Проверка подключения

Для проверки подключения к реальному Binance API:

```bash
cd /home/user/jjhbn/GGG3/paper_trading
python test_binance_connection.py
```

Этот тест проверяет:
- ✓ Используются правильные mainnet URLs
- ✓ Нет ссылок на testnet в коде
- ✓ API работает без ключей
- ✓ Получаемые цены реалистичные (mainnet)

## Официальная документация

- **Binance Spot API:** https://developers.binance.com/docs/binance-spot-api-docs
- **Market Data Endpoints:** https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
- **GitHub Docs:** https://github.com/binance/binance-spot-api-docs

## Примеры использования

### Базовое использование

```python
from paper_trading import PaperExchange

# Создаем биржу - будет использовать РЕАЛЬНЫЕ цены Binance mainnet
exchange = PaperExchange(initial_capital=1000.0)

# Открываем позицию - цена берется с РЕАЛЬНОГО mainnet
position = exchange.open_position(
    symbol='BTCUSDT',
    side='LONG',
    amount=0.01
)

print(f"Opened at real price: ${position['entry_price']:,.2f}")
```

### Прямой запрос цены

```python
from paper_trading import get_binance_price, get_binance_kline

# Получить текущую цену (MAINNET)
btc_price = get_binance_price('BTCUSDT')
eth_price = get_binance_price('ETHUSDT')

print(f"BTC: ${btc_price:,.2f}")
print(f"ETH: ${eth_price:,.2f}")

# Получить свечу (MAINNET)
kline = get_binance_kline('BTCUSDT', '4h', 1)
print(f"4h candle: O={kline['open']}, H={kline['high']}, L={kline['low']}, C={kline['close']}")
```

## Troubleshooting

### 403 Forbidden Error

Если получаете ошибку 403:
- **Причина:** IP адрес заблокирован Cloudflare/Binance
- **Решение:** Модуль автоматически попробует альтернативные endpoints
- **Примечание:** В production среде с нормальным интернетом это не должно происходить

### Timeout Errors

Если получаете timeout:
- **Причина:** Медленное соединение или проблемы с сетью
- **Решение:** Модуль автоматически попробует другой endpoint
- **Timeout:** По умолчанию 10 секунд на запрос

## Безопасность

✅ **Нет API ключей** - используются только публичные данные
✅ **Только чтение** - нет операций изменения
✅ **Mainnet URLs** - только официальные Binance endpoints
✅ **Rate limits** - автоматическое соблюдение лимитов

## Версия

**API Version:** v3 (Spot API)
**Last Updated:** 2025-11-06
**Status:** Production Ready
