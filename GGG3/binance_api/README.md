# Binance API Client

Полнофункциональный клиент для работы с Binance Futures API.

## Особенности

✅ **Futures API** (не Spot) - поддержка LONG и SHORT позиций
✅ **Leverage 1x** - консервативная торговля без плеча
✅ **Testnet и Live** - поддержка тестовой и боевой среды
✅ **Retry логика** - автоматические повторные попытки при ошибках
✅ **Rate limiting** - защита от превышения лимитов API
✅ **Логирование** - детальное логирование всех операций
✅ **Обработка ошибок** - специфичная обработка различных типов ошибок

## Установка

```bash
pip install python-binance
```

## Быстрый старт

### Testnet (рекомендуется для тестирования)

```python
from binance_api import BinanceClient

# Создание клиента для testnet
client = BinanceClient(
    api_key="your_testnet_api_key",
    secret_key="your_testnet_secret_key",
    testnet=True
)

# Получить текущую цену
price = client.get_current_price('BTCUSDT')
print(f"BTC price: ${price:,.2f}")

# Получить баланс
balance = client.get_balance('USDT')
print(f"Balance: ${balance:,.2f}")
```

### Live (боевой режим)

```python
from binance_api import BinanceClient

# ВНИМАНИЕ: Используйте реальные ключи ТОЛЬКО для боевого режима!
client = BinanceClient(
    api_key="your_live_api_key",
    secret_key="your_live_secret_key",
    testnet=False  # LIVE режим
)
```

## Основные методы

### Получение данных

#### get_all_usdt_pairs()
Получает все USDT пары с фильтрацией по объему

```python
# Получить все пары с объемом > $1M за 24ч
pairs = client.get_all_usdt_pairs(min_volume_24h=1_000_000)
print(f"Found {len(pairs)} pairs")  # ~300-400 пар

# Примеры: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', ...]
```

**Фильтры:**
- ✅ Только USDT пары
- ✅ Только активные (status='TRADING')
- ✅ Минимальный объем 24h
- ❌ Исключены стейблкоины (USDCUSDT, BUSDUSDT, ...)

#### get_ohlcv()
Получает OHLCV данные (свечи)

```python
import pandas as pd

# Получить 100 свечей по 4 часа
df = client.get_ohlcv('BTCUSDT', timeframe='4h', limit=100)

print(df.head())
#        timestamp      open      high       low     close    volume
# 0  2025-11-06... 45000.0  45500.0  44800.0  45200.0  1500.5
```

**Поддерживаемые таймфреймы:**
- `'1m'`, `'5m'`, `'15m'`, `'30m'`
- `'1h'`, `'4h'`, `'12h'`
- `'1d'`, `'1w'`, `'1M'`

#### get_current_price()
Получает текущую цену символа

```python
price = client.get_current_price('ETHUSDT')
print(f"ETH: ${price:,.2f}")
```

#### get_orderbook()
Получает стакан ордеров

```python
book = client.get_orderbook('BTCUSDT', limit=10)

# Топ-3 bids и asks
for i in range(3):
    bid_price, bid_qty = book['bids'][i]
    ask_price, ask_qty = book['asks'][i]
    print(f"Bid: ${bid_price:,.2f} x {bid_qty} | Ask: ${ask_price:,.2f} x {ask_qty}")
```

#### get_balance()
Получает доступный баланс

```python
usdt_balance = client.get_balance('USDT')
print(f"Available USDT: ${usdt_balance:,.2f}")
```

#### get_account_info()
Получает полную информацию об аккаунте

```python
account = client.get_account_info()

print(f"Total wallet balance: ${account['totalWalletBalance']}")
print(f"Total unrealized profit: ${account['totalUnrealizedProfit']}")
```

#### get_position_info()
Получает информацию о позициях

```python
# Все активные позиции
positions = client.get_position_info()

for pos in positions:
    print(f"{pos['symbol']}: {pos['positionAmt']} @ ${pos['entryPrice']}")
    print(f"  Unrealized PnL: ${pos['unrealizedProfit']}")

# Позиция по конкретному символу
btc_position = client.get_position_info(symbol='BTCUSDT')
```

#### get_open_orders()
Получает открытые ордера

```python
# Все открытые ордера
orders = client.get_open_orders()

for order in orders:
    print(f"{order['symbol']} {order['type']} {order['side']}")
    print(f"  Price: ${order['price']} | Qty: {order['origQty']}")

# Ордера по конкретному символу
btc_orders = client.get_open_orders(symbol='BTCUSDT')
```

### Создание ордеров

#### create_market_order()
Создает market ордер (мгновенное исполнение)

```python
# LONG позиция - покупка
order = client.create_market_order(
    symbol='BTCUSDT',
    side='BUY',
    quantity=0.001  # 0.001 BTC
)

print(f"Order ID: {order['orderId']}")
print(f"Status: {order['status']}")  # FILLED

# SHORT позиция - продажа
order = client.create_market_order(
    symbol='ETHUSDT',
    side='SELL',
    quantity=0.1,  # 0.1 ETH
    reduce_only=False  # False = открыть позицию, True = закрыть
)
```

#### create_limit_order()
Создает limit ордер (Take Profit)

```python
# Take Profit для LONG позиции
tp_order = client.create_limit_order(
    symbol='BTCUSDT',
    side='SELL',  # Продать при достижении цели
    quantity=0.001,
    price=46000.0,  # Цель
    reduce_only=True  # Закрыть позицию
)

print(f"TP Order ID: {tp_order['orderId']}")
```

#### create_stop_market_order()
Создает stop-market ордер (Stop Loss)

```python
# Stop Loss для LONG позиции
sl_order = client.create_stop_market_order(
    symbol='BTCUSDT',
    side='SELL',  # Продать при падении
    quantity=0.001,
    stop_price=44000.0,  # Стоп цена
    reduce_only=True  # Закрыть позицию
)

print(f"SL Order ID: {sl_order['orderId']}")
```

### Управление ордерами

#### cancel_order()
Отменяет конкретный ордер

```python
success = client.cancel_order(
    symbol='BTCUSDT',
    order_id=12345
)

if success:
    print("Order canceled successfully")
```

#### cancel_all_orders()
Отменяет все открытые ордера для символа

```python
client.cancel_all_orders('BTCUSDT')
print("All orders for BTCUSDT canceled")
```

### Управление leverage

#### set_leverage_for_symbol()
Устанавливает leverage для конкретного символа

```python
# Установить leverage 1x (без плеча)
client.set_leverage_for_symbol('BTCUSDT', leverage=1)

# Для более агрессивной торговли (НЕ рекомендуется)
# client.set_leverage_for_symbol('BTCUSDT', leverage=5)
```

## Полный пример: Открытие позиции с TP/SL

```python
from binance_api import BinanceClient

# Инициализация
client = BinanceClient(
    api_key="your_api_key",
    secret_key="your_secret_key",
    testnet=True
)

# Параметры
symbol = 'BTCUSDT'
quantity = 0.001

# 1. Получить текущую цену
current_price = client.get_current_price(symbol)
print(f"Current price: ${current_price:,.2f}")

# 2. Рассчитать TP/SL (пример: TP=+2%, SL=-1%)
tp_price = current_price * 1.02  # +2%
sl_price = current_price * 0.99  # -1%

# 3. Открыть LONG позицию (market order)
entry_order = client.create_market_order(
    symbol=symbol,
    side='BUY',
    quantity=quantity
)
print(f"✅ Position opened: {entry_order['orderId']}")

# 4. Установить Take Profit (limit order)
tp_order = client.create_limit_order(
    symbol=symbol,
    side='SELL',
    quantity=quantity,
    price=tp_price,
    reduce_only=True
)
print(f"✅ TP set at ${tp_price:,.2f}: {tp_order['orderId']}")

# 5. Установить Stop Loss (stop-market order)
sl_order = client.create_stop_market_order(
    symbol=symbol,
    side='SELL',
    quantity=quantity,
    stop_price=sl_price,
    reduce_only=True
)
print(f"✅ SL set at ${sl_price:,.2f}: {sl_order['orderId']}")

print("\n📊 Position summary:")
print(f"Symbol: {symbol}")
print(f"Entry: ${current_price:,.2f}")
print(f"TP: ${tp_price:,.2f} (+2%)")
print(f"SL: ${sl_price:,.2f} (-1%)")
print(f"R/R: 2:1")
```

## Обработка ошибок

Модуль предоставляет специфичные исключения для разных типов ошибок:

```python
from binance_api import BinanceClient
from binance_api.exceptions import (
    BinanceAPIError,
    RateLimitError,
    InsufficientBalanceError,
    OrderError,
    NetworkError
)

try:
    client = BinanceClient(api_key="key", secret_key="secret", testnet=True)

    # Попытка создать ордер
    order = client.create_market_order('BTCUSDT', 'BUY', 1000.0)

except RateLimitError as e:
    print(f"⚠️ Rate limit exceeded: {e}")
    # Подождать и повторить

except InsufficientBalanceError as e:
    print(f"❌ Insufficient balance: {e}")
    # Проверить баланс

except OrderError as e:
    print(f"❌ Order error: {e}")
    # Проверить параметры ордера

except NetworkError as e:
    print(f"⚠️ Network error: {e}")
    # Проверить соединение

except BinanceAPIError as e:
    print(f"❌ Binance API error: {e}")
    # Общая ошибка API
```

## Retry логика

Клиент автоматически повторяет запросы при временных ошибках:

```python
client = BinanceClient(
    api_key="key",
    secret_key="secret",
    testnet=True,
    max_retries=3,       # Максимум попыток (по умолчанию 3)
    retry_delay=1.0      # Задержка между попытками в секундах (по умолчанию 1.0)
)

# Если запрос провалился - автоматически повторит до 3 раз
# с увеличивающейся задержкой: 1s, 2s, 3s
price = client.get_current_price('BTCUSDT')
```

**Автоматически обрабатываемые ошибки:**
- Сетевые ошибки (timeout, connection error)
- Timestamp errors (-1021, -1022) - автоматическая синхронизация
- Internal errors (-1000, -1001) - повторные попытки
- Другие временные ошибки

**НЕ повторяются:**
- Rate limit exceeded (-1003) → выбрасывает RateLimitError
- Insufficient balance (-2010) → выбрасывает InsufficientBalanceError

## Rate Limiting

Binance имеет лимиты:
- **1200 запросов в минуту** (weight-based)
- **10 ордеров в секунду**

Клиент автоматически отслеживает количество запросов и ждет при достижении лимита:

```python
from binance_api.client import RateLimiter

# Встроенный rate limiter (используется автоматически)
limiter = RateLimiter(max_requests_per_minute=1000)  # Запас 200 req

# При каждом запросе:
# 1. Проверяется количество запросов за последнюю минуту
# 2. Если лимит достигнут - автоматически ждет
# 3. Запрос выполняется
```

## Логирование

Модуль использует стандартный Python logging:

```python
import logging

# Включить детальное логирование
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG для детального, INFO для основного
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from binance_api import BinanceClient

client = BinanceClient("key", "secret", testnet=True)

# Теперь все операции будут логироваться:
# 2025-11-06 18:00:00 - binance_api.client - INFO - Initialized Binance Futures TESTNET client
# 2025-11-06 18:00:01 - binance_api.client - DEBUG - Current price for BTCUSDT: 45000.0
```

## Тестирование

Модуль включает полный набор тестов:

```bash
cd /home/user/jjhbn/GGG3
python3 binance_api/test_client.py
```

**Результат:**
```
================================================================================
ТЕСТИРОВАНИЕ BINANCE API КЛИЕНТА
================================================================================

Тест 1: Импорт компонентов... ✓ ПРОЙДЕН
Тест 2: Rate limiter... ✓ ПРОЙДЕН
Тест 3: Инициализация клиента (testnet)... ✓ ПРОЙДЕН
Тест 4: Инициализация клиента (live)... ✓ ПРОЙДЕН
Тест 5: Получение текущей цены... ✓ ПРОЙДЕН
Тест 6: Получение баланса... ✓ ПРОЙДЕН
Тест 7: Получение OHLCV... ✓ ПРОЙДЕН
Тест 8: Создание market ордера... ✓ ПРОЙДЕН
Тест 9: Создание limit ордера (TP)... ✓ ПРОЙДЕН
Тест 10: Создание stop-market ордера (SL)... ✓ ПРОЙДЕН
Тест 11: Отмена ордера... ✓ ПРОЙДЕН
Тест 12: Получение информации о позициях... ✓ ПРОЙДЕН
Тест 13: Retry логика... ✓ ПРОЙДЕН

Пройдено: 13/13
✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ
```

## Структура модуля

```
binance_api/
├── __init__.py          # Экспорты модуля
├── client.py            # Основной клиент BinanceClient
├── exceptions.py        # Кастомные исключения
├── test_client.py       # Тесты (13 тестов)
└── README.md            # Документация
```

## API Reference

### BinanceClient

**Инициализация:**
```python
BinanceClient(
    api_key: str,
    secret_key: str,
    testnet: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0
)
```

**Методы данных:**
- `get_all_usdt_pairs(min_volume_24h: float = 1_000_000) -> List[str]`
- `get_ohlcv(symbol, timeframe, limit=500, start_time=None, end_time=None) -> pd.DataFrame`
- `get_current_price(symbol: str) -> float`
- `get_orderbook(symbol: str, limit: int = 20) -> dict`
- `get_balance(asset: str = 'USDT') -> float`
- `get_account_info() -> dict`
- `get_position_info(symbol: Optional[str] = None) -> List[dict]`
- `get_open_orders(symbol: Optional[str] = None) -> List[dict]`

**Методы ордеров:**
- `create_market_order(symbol, side, quantity, reduce_only=False) -> dict`
- `create_limit_order(symbol, side, quantity, price, reduce_only=False, time_in_force='GTC') -> dict`
- `create_stop_market_order(symbol, side, quantity, stop_price, reduce_only=False) -> dict`
- `cancel_order(symbol: str, order_id: int) -> bool`
- `cancel_all_orders(symbol: str) -> bool`

**Методы управления:**
- `set_leverage_all(leverage: int = 1)`
- `set_leverage_for_symbol(symbol: str, leverage: int = 1)`

## Важные замечания

⚠️ **TESTNET vs LIVE:**
- Всегда начинайте с testnet
- Testnet использует отдельные API ключи
- Testnet имеет виртуальные деньги
- Переход на live только после полного тестирования

⚠️ **LEVERAGE:**
- По умолчанию используется 1x (без плеча)
- Это консервативная стратегия
- Высокий leverage = высокий риск ликвидации

⚠️ **RATE LIMITS:**
- Соблюдайте лимиты API
- Используйте встроенный rate limiter
- При достижении лимита - клиент автоматически ждет

⚠️ **БЕЗОПАСНОСТЬ:**
- НЕ коммитьте API ключи в git
- Используйте переменные окружения
- Ограничьте права API ключей (только торговля, без вывода)

## Полезные ссылки

- [Binance Futures API Documentation](https://binance-docs.github.io/apidocs/futures/en/)
- [Binance Testnet](https://testnet.binancefuture.com/)
- [python-binance Library](https://python-binance.readthedocs.io/)

## Лицензия

См. основной проект.
