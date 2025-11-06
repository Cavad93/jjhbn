# Data Collection Module

Модуль для сбора и обновления исторических данных с Binance Futures API.

## 📋 Возможности

- ✅ **Первичная загрузка данных** - загрузка исторических OHLCV данных для всех торгуемых пар
- ✅ **Обновление данных** - дозагрузка новых свечей к существующим данным
- ✅ **Мульти-таймфрейм** - поддержка 5m, 15m, 30m, 4h и любых других таймфреймов
- ✅ **Автоматическая фильтрация** - исключение стейблкоинов и пар с низким объёмом
- ✅ **Формат Parquet** - быстрое сжатие и чтение данных
- ✅ **Индексирование** - автоматическое создание индекса данных
- ✅ **Rate limiting** - защита от превышения лимитов API
- ✅ **Целостность данных** - автоматическое удаление дубликатов и сортировка

---

## 🚀 Быстрый старт

### Первичная загрузка данных

```python
from data_collection import collect_all_data

# Загрузка данных для всех пар
result = collect_all_data(
    output_dir='/home/user/jjhbn/GGG3/data/historical',
    timeframes=['5m', '15m', '30m', '4h'],
    min_volume_24h=1_000_000,  # Минимум 1M USDT объём
    limit=500  # 500 свечей для каждого таймфрейма
)

print(f"Загружено пар: {result['success_pairs']}/{result['total_pairs']}")
```

### Обновление существующих данных

```python
from data_collection import update_all_data

# Обновление данных (дозагрузка новых свечей)
result = update_all_data(
    data_dir='/home/user/jjhbn/GGG3/data/historical',
    timeframes=['5m', '15m', '30m', '4h']
)

print(f"Обновлено символов: {result['success_symbols']}/{result['total_symbols']}")
```

### Использование из командной строки

```bash
# Первичная загрузка
python -m data_collection.collect_binance_data --mode collect \
    --output-dir /home/user/jjhbn/GGG3/data/historical \
    --min-volume 1000000 \
    --limit 500

# Обновление данных
python -m data_collection.collect_binance_data --mode update \
    --output-dir /home/user/jjhbn/GGG3/data/historical

# Статистика
python -m data_collection.collect_binance_data --mode stats \
    --output-dir /home/user/jjhbn/GGG3/data/historical
```

---

## 📂 Структура данных

### Структура директорий

```
data/
└── historical/
    ├── BTCUSDT_5m.parquet
    ├── BTCUSDT_15m.parquet
    ├── BTCUSDT_30m.parquet
    ├── BTCUSDT_4h.parquet
    ├── ETHUSDT_5m.parquet
    ├── ETHUSDT_15m.parquet
    ├── ...
    └── data_index.json
```

### Формат OHLCV файлов

Каждый parquet файл содержит следующие колонки:

| Колонка   | Тип       | Описание                      |
|-----------|-----------|-------------------------------|
| timestamp | datetime  | Временная метка (UTC)         |
| open      | float     | Цена открытия                 |
| high      | float     | Максимальная цена             |
| low       | float     | Минимальная цена              |
| close     | float     | Цена закрытия                 |
| volume    | float     | Объём торгов (в базовой валюте) |

### Формат индекса (data_index.json)

```json
{
  "BTCUSDT": {
    "5m": "BTCUSDT_5m.parquet",
    "15m": "BTCUSDT_15m.parquet",
    "30m": "BTCUSDT_30m.parquet",
    "4h": "BTCUSDT_4h.parquet",
    "rows_5m": 500,
    "rows_15m": 500,
    "rows_30m": 500,
    "rows_4h": 500,
    "first_timestamp": "2025-08-15T10:00:00",
    "last_timestamp": "2025-11-06T18:00:00",
    "last_update": "2025-11-06T19:00:00"
  },
  ...
}
```

---

## 🔧 API Reference

### get_all_tradeable_pairs()

Получает список всех торгуемых пар с фильтрами.

```python
def get_all_tradeable_pairs(
    client: BinanceClient,
    min_volume_24h: float = 1_000_000,
    exclude_stablecoins: bool = True
) -> List[str]
```

**Параметры:**
- `client` - Binance API клиент
- `min_volume_24h` - Минимальный объём торгов за 24ч (USDT)
- `exclude_stablecoins` - Исключить стейблкоины

**Возвращает:**
- `List[str]` - Список символов (обычно ~300-400 пар)

**Фильтры:**
- Только USDT пары
- Объём 24h > min_volume_24h
- Исключение стейблкоинов (USDCUSDT, BUSDUSDT и т.д.)
- Исключение blacklist пар

---

### download_ohlcv_data()

Скачивает OHLCV данные для символа и таймфреймов.

```python
def download_ohlcv_data(
    client: BinanceClient,
    symbol: str,
    timeframes: List[str],
    output_dir: str,
    limit: int = 500
) -> bool
```

**Параметры:**
- `client` - Binance API клиент
- `symbol` - Символ пары (например, 'BTCUSDT')
- `timeframes` - Список таймфреймов (например, ['5m', '15m', '30m', '4h'])
- `output_dir` - Директория для сохранения
- `limit` - Количество свечей (max 1000, default 500)

**Возвращает:**
- `bool` - True если все таймфреймы загружены успешно

**Сохраняет:**
- `{output_dir}/{symbol}_5m.parquet`
- `{output_dir}/{symbol}_15m.parquet`
- и т.д.

---

### update_ohlcv_data()

Обновляет существующие OHLCV данные (дозагружает новые свечи).

```python
def update_ohlcv_data(
    client: BinanceClient,
    symbol: str,
    timeframes: List[str],
    output_dir: str
) -> bool
```

**Логика обновления:**
1. Читает существующий файл
2. Определяет последнюю временную метку
3. Загружает новые свечи после последней метки
4. Объединяет старые и новые данные
5. Удаляет дубликаты по timestamp
6. Сортирует по времени
7. Сохраняет обновлённый файл

**Параметры:**
- `client` - Binance API клиент
- `symbol` - Символ пары
- `timeframes` - Список таймфреймов
- `output_dir` - Директория с данными

**Возвращает:**
- `bool` - True если все таймфреймы обновлены успешно

---

### collect_all_data()

Первичная загрузка всех данных.

```python
def collect_all_data(
    output_dir: str = '/home/user/jjhbn/GGG3/data/historical',
    timeframes: Optional[List[str]] = None,
    min_volume_24h: float = 1_000_000,
    limit: int = 500,
    use_testnet: bool = False
) -> Dict[str, Any]
```

**Процесс:**
1. Получение списка торгуемых пар
2. Загрузка OHLCV данных для всех пар
3. Создание индекса данных

**Параметры:**
- `output_dir` - Директория для сохранения
- `timeframes` - Список таймфреймов (default: ['5m', '15m', '30m', '4h'])
- `min_volume_24h` - Минимальный объём торгов
- `limit` - Количество свечей
- `use_testnet` - Использовать testnet

**Возвращает:**
```python
{
    'total_pairs': int,           # Всего пар
    'success_pairs': int,         # Успешно загружено
    'failed_pairs': List[str],    # Не удалось загрузить
    'timeframes': List[str],      # Таймфреймы
    'output_dir': str,            # Директория
    'timestamp': str,             # Время завершения
    'index_info': Dict            # Информация об индексе
}
```

---

### update_all_data()

Обновление всех существующих данных.

```python
def update_all_data(
    data_dir: str = '/home/user/jjhbn/GGG3/data/historical',
    timeframes: Optional[List[str]] = None,
    use_testnet: bool = False
) -> Dict[str, Any]
```

**Логика:**
1. Читает индекс данных (data_index.json)
2. Для каждого символа обновляет данные по всем таймфреймам
3. Обновляет индекс

**Параметры:**
- `data_dir` - Директория с данными
- `timeframes` - Список таймфреймов (default: ['5m', '15m', '30m', '4h'])
- `use_testnet` - Использовать testnet

**Возвращает:**
```python
{
    'total_symbols': int,         # Всего символов
    'success_symbols': int,       # Успешно обновлено
    'failed_symbols': List[str],  # Не удалось обновить
    'timeframes': List[str],      # Таймфреймы
    'data_dir': str,              # Директория
    'timestamp': str,             # Время завершения
    'index_info': Dict            # Информация об индексе
}
```

---

### create_data_index()

Создаёт индекс данных.

```python
def create_data_index(data_dir: str) -> Dict[str, Any]
```

**Параметры:**
- `data_dir` - Директория с данными

**Возвращает:**
```python
{
    'index_path': str,            # Путь к индексу
    'total_symbols': int,         # Всего символов
    'timeframe_stats': Dict       # Статистика по таймфреймам
}
```

**Сохраняет:**
- `{data_dir}/data_index.json` - Индекс всех данных

---

### get_data_stats()

Получает статистику по загруженным данным.

```python
def get_data_stats(
    data_dir: str = '/home/user/jjhbn/GGG3/data/historical'
) -> Dict[str, Any]
```

**Параметры:**
- `data_dir` - Директория с данными

**Возвращает:**
```python
{
    'total_symbols': int,     # Всего символов
    'total_files': int,       # Всего файлов
    'total_size_mb': float,   # Размер данных (MB)
    'data_dir': str,          # Директория
    'index_path': str         # Путь к индексу
}
```

---

## 💡 Примеры использования

### Пример 1: Загрузка данных для топ-100 монет

```python
from data_collection import collect_all_data

result = collect_all_data(
    output_dir='./data/historical',
    min_volume_24h=5_000_000,  # Топ монеты с объёмом > 5M
    limit=1000  # Больше исторических данных
)

print(f"✅ Загружено {result['success_pairs']} пар")
```

### Пример 2: Ежедневное обновление данных

```python
from data_collection import update_all_data
import schedule
import time

def daily_update():
    """Ежедневное обновление данных в 00:00 UTC"""
    print("Запуск обновления данных...")
    result = update_all_data()
    print(f"✅ Обновлено {result['success_symbols']} символов")

# Запланировать обновление каждый день в 00:00
schedule.every().day.at("00:00").do(daily_update)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Пример 3: Чтение данных

```python
import pandas as pd

# Чтение OHLCV данных
df = pd.read_parquet('data/historical/BTCUSDT_4h.parquet')

print(f"Загружено {len(df)} свечей")
print(f"Период: {df['timestamp'].min()} - {df['timestamp'].max()}")
print(df.head())
```

### Пример 4: Использование индекса

```python
import json

# Чтение индекса
with open('data/historical/data_index.json', 'r') as f:
    index = json.load(f)

# Доступные символы
symbols = list(index.keys())
print(f"Доступно символов: {len(symbols)}")

# Информация о конкретном символе
btc_info = index['BTCUSDT']
print(f"BTCUSDT:")
print(f"  - Таймфреймы: {[k for k in btc_info.keys() if not k.startswith('rows_')]}")
print(f"  - Строк (4h): {btc_info['rows_4h']}")
print(f"  - Последнее обновление: {btc_info['last_update']}")
```

---

## ⚙️ Настройки

### Фильтры пар

Можно настроить фильтры в файле `collect_binance_data.py`:

```python
# Список стейблкоинов (исключаются из торговли)
STABLECOINS = [
    'USDCUSDT',
    'BUSDUSDT',
    'TUSDUSDT',
    # Добавьте свои...
]

# Blacklist (проблемные монеты)
BLACKLIST = [
    # Добавьте проблемные пары
]
```

### Rate Limiting

Модуль автоматически ограничивает скорость запросов:
- 0.1 сек между таймфреймами
- 0.2 сек между парами

Для изменения измените значения `time.sleep()` в коде.

---

## 🧪 Тестирование

Запуск тестов:

```bash
python data_collection/test_collect_binance_data.py
```

Результаты: **10/12 тестов пройдено** ✅

Тесты покрывают:
- Импорты и базовые функции
- Фильтрацию пар
- Загрузку и обновление данных
- Создание индекса
- Целостность данных
- Формат файлов

---

## ⚠️ Важные замечания

### API ключи не требуются

Для загрузки исторических данных API ключи Binance **не требуются**. Модуль использует публичные endpoints.

### Объём данных

Примерный размер данных:
- 300 пар × 4 таймфрейма × 500 свечей ≈ **50-100 MB** (в формате Parquet с сжатием)

### Ограничения Binance API

- Максимум 1000 свечей за один запрос
- Rate limit: ~1200 запросов в минуту
- Модуль автоматически соблюдает эти ограничения

### Обновление данных

Рекомендуется запускать `update_all_data()` один раз в день (в 00:00 UTC) для получения свежих данных.

---

## 🔄 Интеграция с ботом

Пример использования в торговом боте:

```python
from data_collection import update_all_data, get_data_stats
import pandas as pd

# Обновление данных при запуске бота
print("Обновление исторических данных...")
update_all_data()

# Проверка статистики
stats = get_data_stats()
print(f"Доступно данных: {stats['total_symbols']} символов, {stats['total_size_mb']} MB")

# Загрузка данных для анализа
def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    """Загружает OHLCV данные для символа"""
    file_path = f"data/historical/{symbol}_{timeframe}.parquet"
    return pd.read_parquet(file_path)

# Использование в стратегии
btc_4h = load_ohlcv('BTCUSDT', '4h')
# ... расчёт индикаторов и сигналов
```

---

## 📝 Лицензия

Часть проекта GGG3 Binance Trading Bot.

---

## 🤝 Вклад

При добавлении новых фильтров или функций:
1. Добавьте тесты в `test_collect_binance_data.py`
2. Обновите документацию в README
3. Проверьте, что все тесты проходят

---

## 📞 Поддержка

При возникновении проблем:
1. Проверьте логи (уровень INFO/DEBUG)
2. Убедитесь, что установлены все зависимости (pyarrow, pandas, tqdm)
3. Проверьте доступность Binance API
