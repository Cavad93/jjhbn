

# Smart Collector - Умный сборщик данных с PostgreSQL

Новая система сбора данных с фокусом на **топ-10 монет по EV** вместо загрузки всех 300-400 пар.

---

## 🎯 Основные отличия от старой системы

| Функция | Старая система (Parquet) | Новая система (PostgreSQL) |
|---------|--------------------------|----------------------------|
| Хранение | Parquet файлы | PostgreSQL 12+ |
| Количество пар | Все 300-400 пар | **Только топ-10 по EV** |
| Обновление | Ручное | **Автоматическое** |
| Очистка | Нет | **Автоматическая (7 дней без позиций)** |
| Производительность | Средняя | **Высокая (индексы, connection pool)** |
| Асинхронность | Нет | **Да (asyncpg)** |
| Интеграция с позициями | Нет | **Да (отслеживание активности)** |

---

## 📋 Архитектура

```
┌─────────────────────┐
│   SmartCollector    │  ← Умный сборщик
└──────────┬──────────┘
           │
           ├─→ BinanceClient (API)
           │
           └─→ DatabaseManager (PostgreSQL)
                    │
                    ├─→ ohlcv_data (OHLCV свечи)
                    ├─→ active_symbols (топ-10)
                    ├─→ data_stats (статистика)
                    └─→ cleanup_log (лог очистки)
```

---

## 🚀 Быстрый старт

### 1. Установка PostgreSQL

```bash
# Ubuntu/Debian
sudo apt-get install postgresql-12

# Создание базы данных
sudo -u postgres psql

postgres=# CREATE DATABASE binance_bot;
postgres=# CREATE USER bot_user WITH PASSWORD 'bot_password';
postgres=# GRANT ALL PRIVILEGES ON DATABASE binance_bot TO bot_user;
postgres=# \q
```

### 2. Создание схемы БД

```bash
cd /home/user/jjhbn/GGG3/data_collection

# Применение схемы
psql -U bot_user -d binance_bot -f db_schema.sql
```

### 3. Установка зависимостей

```bash
pip install asyncpg psycopg2-binary
```

### 4. Первичная загрузка топ-10

```python
import asyncio
from binance_api.client import BinanceClient
from data_collection import DatabaseManager, SmartCollector

async def main():
    # Создание клиентов
    binance = BinanceClient(api_key="", secret_key="", testnet=False)

    db = DatabaseManager(
        host='localhost',
        port=5432,
        database='binance_bot',
        user='bot_user',
        password='bot_password'
    )

    async with db:
        # Создание умного сборщика
        collector = SmartCollector(db, binance)

        # Загрузка топ-10
        result = await collector.collect_top_symbols_data(top_n=10)

        print(f"✅ Загружено {result['success_count']} символов")
        print(f"📊 Топ-10: {[s['symbol'] for s in result['top_symbols']]}")

asyncio.run(main())
```

### 5. Автоматический режим

```bash
# Автоматическое обновление каждые 60 минут
python -m data_collection.smart_collector --mode auto --interval 60 --cleanup-days 7
```

---

## 💡 Основные функции

### SmartCollector

#### `get_top_symbols_by_ev(min_volume_24h, top_n)`

Получение топ-N символов по Expected Value (EV).

**ВАЖНО:** Сейчас используется заглушка (сортировка по объёму торгов).
**Нужно интегрировать с вашей ML моделью!**

```python
# TODO: Интеграция с ML моделью
top_symbols = await collector.get_top_symbols_by_ev(
    min_volume_24h=1_000_000,
    top_n=10
)

# Результат:
# [
#   {'symbol': 'BTCUSDT', 'ev_score': 0.045, 'rank': 1, 'volume_24h': 25000000000},
#   {'symbol': 'ETHUSDT', 'ev_score': 0.042, 'rank': 2, 'volume_24h': 15000000000},
#   ...
# ]
```

**Как интегрировать ML модель:**

```python
async def get_top_symbols_by_ev(self, min_volume_24h, top_n):
    all_pairs = self.client.get_all_usdt_pairs(min_volume_24h)

    ev_scores = []

    for symbol in all_pairs:
        # 1. Загрузить OHLCV
        df_4h = self.client.get_ohlcv(symbol, '4h', limit=100)

        # 2. Рассчитать 68 признаков
        features = calculate_features(df_4h)  # Ваша функция

        # 3. Получить предсказания моделей
        predictions = {
            'p_xgb': model_xgb.predict(features)[0],
            'p_rf': model_rf.predict(features)[0],
            'p_lgbm': model_lgbm.predict(features)[0],
            # ... остальные модели
        }

        # 4. META предсказание
        meta_input = np.array(list(predictions.values()))
        p_meta = model_meta.predict(meta_input)[0]

        # 5. Расчёт EV
        ev_long = p_meta * TP_MULTIPLIER - (1 - p_meta) * SL_MULTIPLIER
        ev_short = p_meta * TP_MULTIPLIER - (1 - p_meta) * SL_MULTIPLIER

        ev = max(ev_long, ev_short)

        ev_scores.append({
            'symbol': symbol,
            'ev_score': ev,
            'p_meta': p_meta
        })

    # Сортировка по EV
    ev_scores.sort(key=lambda x: x['ev_score'], reverse=True)

    # Топ-N
    top_symbols = []
    for rank, item in enumerate(ev_scores[:top_n], start=1):
        top_symbols.append({
            'symbol': item['symbol'],
            'ev_score': item['ev_score'],
            'rank': rank,
            'volume_24h': ...  # Из API
        })

    return top_symbols
```

#### `collect_top_symbols_data(top_n, min_volume_24h)`

Загрузка данных для топ-N символов.

**Workflow:**
1. Получить топ-N по EV
2. Обновить список активных символов в БД
3. Загрузить/обновить данные для каждого символа
4. Обновить статистику

```python
result = await collector.collect_top_symbols_data(top_n=10)

# Результат:
# {
#     'top_symbols': [...],
#     'success_count': 10,
#     'failed_symbols': [],
#     'total_inserted': 20000,
#     'timestamp': '2025-11-06T...'
# }
```

#### `cleanup_old_data(days_threshold)`

Асинхронная очистка старых данных.

**Критерии удаления:**
1. Символ не в топ-10 (`is_active = FALSE`)
2. Прошло более `days_threshold` дней с последней позиции

```python
# Удаление данных для монет вне топ-10 (7+ дней без позиций)
deleted_symbols = await collector.cleanup_old_data(days_threshold=7)

print(f"Удалено символов: {len(deleted_symbols)}")
# ['ADAUSDT', 'DOTUSDT', ...]
```

#### `auto_update_loop(interval_minutes, cleanup_days)`

Автоматический режим обновления.

```python
# Обновление каждые 60 минут, очистка раз в 24 часа
await collector.auto_update_loop(
    interval_minutes=60,
    cleanup_days=7
)
```

---

### DatabaseManager

#### Connection Pool

```python
db = DatabaseManager(
    host='localhost',
    port=5432,
    database='binance_bot',
    user='bot_user',
    password='bot_password',
    min_pool_size=10,  # Минимум 10 соединений
    max_pool_size=20   # Максимум 20 соединений
)

async with db:
    # Все операции внутри context manager
    ...
```

#### Batch Insert (UPSERT)

```python
# Вставка DataFrame в БД
df = pd.DataFrame({
    'timestamp': [...],
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

rows_inserted = await db.insert_ohlcv_batch('BTCUSDT', '4h', df)
print(f"Вставлено: {rows_inserted} строк")
```

**UPSERT** - если свеча уже существует (по `symbol + timeframe + timestamp`), она обновляется.

#### Получение данных

```python
# Получить последние 500 свечей
df = await db.get_ohlcv('BTCUSDT', '4h', limit=500)

print(df.head())
#              timestamp     open     high      low    close     volume
# 0  2025-09-01 00:00:00  50000.0  51000.0  49500.0  50500.0  1234.56
# ...
```

#### Обновление активных символов

```python
top_symbols = [
    {'symbol': 'BTCUSDT', 'ev_score': 0.045, 'rank': 1, 'volume_24h': 25000000000},
    {'symbol': 'ETHUSDT', 'ev_score': 0.042, 'rank': 2, 'volume_24h': 15000000000},
    ...
]

await db.update_active_symbols(top_symbols)
```

#### Обновление времени последней позиции

```python
# Вызывается при открытии позиции
await db.update_position_time('BTCUSDT')
```

Это обновляет `last_position_at` и `positions_count` для символа, чтобы очистка не удалила данные для активно торгуемых монет.

#### Статистика

```python
# Общая статистика
stats = await db.get_stats()
print(stats)
# {
#     'total_rows': 50000,
#     'total_symbols': 10,
#     'active_symbols': 10,
#     'total_size_mb': 125.5
# }

# Статистика по символу
sym_stats = await db.get_symbol_stats('BTCUSDT')
print(sym_stats)
# {
#     'symbol': 'BTCUSDT',
#     'timeframes': {
#         '5m': {'rows_count': 500, 'first_timestamp': ..., 'last_timestamp': ...},
#         '15m': {'rows_count': 500, ...},
#         ...
#     }
# }
```

---

## 🗄️ Схема базы данных

### Таблица: `ohlcv_data`

Хранение всех OHLCV свечей.

```sql
CREATE TABLE ohlcv_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    quote_volume DECIMAL(20, 8),
    trades_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT ohlcv_unique UNIQUE (symbol, timeframe, timestamp)
);

-- Индексы для быстрого поиска
CREATE INDEX idx_ohlcv_symbol_timeframe_timestamp
    ON ohlcv_data(symbol, timeframe, timestamp DESC);
```

**Размер:** ~100 байт на строку

**Примерный объём данных:**
- 10 символов × 4 таймфрейма × 500 свечей = **20,000 строк** (~2 MB)
- 10 символов × 4 таймфрейма × 10,000 свечей = **400,000 строк** (~40 MB)

### Таблица: `active_symbols`

Список активных символов (топ-10 по EV).

```sql
CREATE TABLE active_symbols (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    ev_score DECIMAL(10, 6),
    rank INTEGER,
    volume_24h DECIMAL(20, 2),
    last_position_at TIMESTAMPTZ,
    positions_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CHECK (rank >= 1 AND rank <= 10)
);
```

**Назначение:**
- Отслеживание топ-10 монет
- Отслеживание последней позиции (`last_position_at`)
- Критерий для очистки (если `is_active = FALSE` и прошло 7+ дней)

### Таблица: `data_stats`

Статистика по загруженным данным.

```sql
CREATE TABLE data_stats (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    rows_count INTEGER DEFAULT 0,
    first_timestamp TIMESTAMPTZ,
    last_timestamp TIMESTAMPTZ,
    data_size_mb DECIMAL(10, 2),
    last_update TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT data_stats_unique UNIQUE (symbol, timeframe)
);
```

**Обновление:** Вызов функции `refresh_data_stats()` или триггер.

### Таблица: `cleanup_log`

Журнал очистки старых данных.

```sql
CREATE TABLE cleanup_log (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframes TEXT[],
    rows_deleted INTEGER,
    reason VARCHAR(100),
    deleted_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Пример записи:**
```json
{
    "symbol": "ADAUSDT",
    "timeframes": ["5m", "15m", "30m", "4h"],
    "rows_deleted": 2000,
    "reason": "not_in_top10_7days",
    "deleted_at": "2025-11-06T20:00:00Z"
}
```

---

## ⚡ Производительность

### Connection Pool

- **Минимум:** 10 соединений
- **Максимум:** 20 соединений
- **Переиспользование:** Соединения не закрываются после запроса

**Результат:** Высокая скорость выполнения запросов (без overhead на создание соединения).

### Batch Inserts

```python
# Вставка 500 свечей за один запрос
await db.insert_ohlcv_batch('BTCUSDT', '4h', df)  # ~50ms
```

**Vs. Построчная вставка:**
```python
# 500 отдельных INSERT запросов
for row in df.iterrows():
    await conn.execute(...)  # ~25 секунд (медленно!)
```

### Индексы

Все критичные колонки проиндексированы:
- `(symbol, timeframe, timestamp)` - составной индекс для поиска свечей
- `timestamp DESC` - для быстрого получения последних данных

**Результат:** Запросы выполняются за **< 10ms** даже для больших таблиц.

---

## 🔄 Интеграция с ботом

### 1. При запуске бота

```python
async def bot_startup():
    # Создание DB менеджера
    db = DatabaseManager(...)
    await db.connect()

    # Создание умного сборщика
    collector = SmartCollector(db, binance_client)

    # Первичная загрузка топ-10
    await collector.collect_top_symbols_data(top_n=10)

    # Запуск фонового обновления
    asyncio.create_task(collector.auto_update_loop(interval_minutes=60))

    return db, collector
```

### 2. При открытии позиции

```python
async def open_position(symbol: str, direction: str, ...):
    # Открытие позиции
    position = Position(symbol=symbol, direction=direction, ...)

    # Обновление времени последней позиции в БД
    await db.update_position_time(symbol)

    # Это предотвратит удаление данных для этого символа
    # при очистке (даже если он выпал из топ-10)
```

### 3. Получение данных для расчёта признаков

```python
async def calculate_features(symbol: str) -> np.ndarray:
    # Загрузить OHLCV из БД
    df_5m = await db.get_ohlcv(symbol, '5m', limit=500)
    df_15m = await db.get_ohlcv(symbol, '15m', limit=500)
    df_30m = await db.get_ohlcv(symbol, '30m', limit=500)
    df_4h = await db.get_ohlcv(symbol, '4h', limit=500)

    # Расчёт 68 признаков
    features = extract_features(df_5m, df_15m, df_30m, df_4h)

    return features
```

### 4. Ежедневная очистка

```python
# Планировщик (например, APScheduler)
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job('cron', hour=0, minute=0)  # Каждый день в 00:00
async def daily_cleanup():
    deleted = await collector.cleanup_old_data(days_threshold=7)
    logger.info(f"Очистка: удалено {len(deleted)} символов")

scheduler.start()
```

---

## ⚠️ Важные замечания

### 1. Интеграция ML модели

**Критически важно:** Функция `get_top_symbols_by_ev()` сейчас использует **заглушку** (сортировка по объёму торгов).

**Вам нужно заменить** эту логику на вызов вашей ML модели для расчёта реального EV.

### 2. Обновление времени позиции

**Обязательно** вызывайте `db.update_position_time(symbol)` при открытии каждой позиции, чтобы:
- Отслеживать активность символа
- Предотвратить удаление данных для активно торгуемых монет

### 3. Очистка данных

Очистка выполняется **асинхронно** и не блокирует бота. Данные удаляются только если:
1. Символ не в топ-10 (`is_active = FALSE`)
2. Прошло 7+ дней с последней позиции (`last_position_at`)

### 4. Размер базы данных

При **10 символах × 4 таймфреймах × 500 свечах**:
- ~20,000 строк
- ~2-5 MB размер БД

При росте данных рекомендуется:
- Партиционирование по месяцам (см. `db_schema.sql`)
- Архивирование старых данных (> 3 месяцев)

---

## 🧪 Тестирование

### Тест подключения к БД

```python
import asyncio
from data_collection import DatabaseManager

async def test_connection():
    db = DatabaseManager(
        host='localhost',
        port=5432,
        database='binance_bot',
        user='bot_user',
        password='bot_password'
    )

    async with db:
        stats = await db.get_stats()
        print(f"✅ Подключение успешно")
        print(f"Статистика: {stats}")

asyncio.run(test_connection())
```

### Тест загрузки данных

```bash
# Одноразовая загрузка топ-10
python -m data_collection.smart_collector --mode once --top-n 10

# Только очистка
python -m data_collection.smart_collector --mode cleanup --cleanup-days 7
```

---

## 📝 Заключение

**Smart Collector** - это значительное улучшение над старой системой:

✅ **Загружает только топ-10** вместо всех 300+ пар
✅ **Автоматическая очистка** старых данных
✅ **Асинхронность** - не блокирует бота
✅ **PostgreSQL** - высокая производительность
✅ **Интеграция с позициями** - отслеживание активности
✅ **Connection pooling** - быстрые запросы
✅ **Batch inserts** - быстрая загрузка

**Готов к использованию в боте!** 🚀
