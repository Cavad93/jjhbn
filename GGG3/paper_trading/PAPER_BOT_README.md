# Paper Trading Bot

Полная копия реального торгового бота с использованием PaperExchange для симуляции торговли с реальными ценами Binance.

## Описание

`paper_bot_main.py` - главный файл paper trading бота, который:
- ✅ Использует **РЕАЛЬНЫЕ цены** с Binance mainnet
- ✅ Загружает ML модели (XGBoost, RF, ARF, NN, META)
- ✅ Реализует **EV Filter** для отбора топ-10 монет
- ✅ Поддерживает **LONG и SHORT** торговлю
- ✅ Использует **Snapshot Pattern** для temporal consistency
- ✅ Применяет **Online Learning** при закрытии позиций
- ✅ Логирует все операции
- ✅ Сохраняет историю в CSV

## Архитектура

```
paper_bot_main.py
├── Config                  # Конфигурация бота
├── ModelsManager           # Загрузка ML моделей
├── FeaturesCalculator      # Расчет 68D фич
├── EVFilter                # Expected Value фильтр
├── ATRCalculator           # Расчет ATR для TP/SL
└── PaperTradingBot         # Главный класс бота
    ├── get_binance_universe()  # Получение списка монет
    ├── update_portfolio()      # Обновление портфеля (каждые 4h)
    ├── open_position()         # Открытие позиции
    ├── check_orders()          # Проверка TP/SL (каждые 5m)
    └── run()                   # Главный цикл
```

## Функционал

### 1. Загрузка моделей

```python
models = {
    'xgb': XGBoost model,      # xgb_model.pkl
    'rf': RandomForest model,   # rf_model.pkl
    'arf': AdaptiveRF model,    # arf_model.pkl
    'nn': NeuralNet model,      # nn_model.pkl
    'meta': META model          # meta_model.pkl
}
```

### 2. Торговый цикл

**Каждые 4 часа:**
- Сканирование ~400 монет (universe)
- EV Filter: отбор топ-10 по Expected Value
- Закрытие позиций вне топ-10
- Открытие новых позиций

**Каждые 5 минут:**
- Проверка TP/SL ордеров
- Исполнение достигнутых ордеров
- Online learning на закрытых позициях

### 3. EV Filter

Рассчитывает Expected Value для каждой монеты:

```python
# EV = P(TP) × Reward - P(SL) × Risk
# При R/R = 1.5:

EV_long = 2.5 * p_up - 1.0
EV_short = 1.5 - 2.5 * p_up

# Break-even:
# LONG: p_up > 0.40
# SHORT: p_up < 0.60
```

Выбирает топ-10 монет с максимальным EV (min 0.02).

### 4. TP/SL расчет

На основе ATR (Average True Range):

```python
ATR = calculate_atr(symbol, period=10, timeframe='4h')

# LONG:
TP = entry + 1.2 * ATR  # Reward: 1.2R
SL = entry - 0.8 * ATR  # Risk: 0.8R

# SHORT:
TP = entry - 1.2 * ATR
SL = entry + 0.8 * ATR

# R/R = 1.5:1
```

### 5. Snapshot Pattern (Temporal Consistency)

При открытии позиции сохраняется snapshot:

```python
entry_snapshot = {
    'features': features_68D,     # Фичи на момент входа (t0)
    'predictions': {              # Предсказания экспертов
        'p_xgb': 0.62,
        'p_rf': 0.58,
        ...
    },
    'ev': 0.0500,                # Expected Value
    'direction': 'LONG',
    'timestamp': 1699291200
}
```

При закрытии используется **СОХРАНЕННЫЙ** snapshot для online learning (избегаем look-ahead bias).

### 6. Online Learning

```python
def update_models_online(position):
    snapshot = position['metadata']['entry_snapshot']
    features = snapshot['features']  # t0 (момент входа)
    outcome = 1 if position['close_reason'] == 'tp' else 0  # t1 (момент выхода)

    # Обучаем модели на (X_t0, y_t1)
    expert.train(features, outcome)
```

## Установка

### Требования

```bash
# Базовые зависимости
pip install numpy pandas requests

# ML библиотеки (опционально, для моделей)
pip install xgboost scikit-learn river torch
```

### Структура файлов

```
GGG3/paper_trading/
├── paper_bot_main.py          # Главный файл бота
├── paper_exchange.py          # PaperExchange модуль
├── test_paper_bot.py          # Тесты бота
├── PAPER_BOT_README.md        # Эта документация
│
├── logs/                      # Логи работы (создается автоматически)
│   └── paper_bot_20251106.log
│
└── data/                      # Данные (создается автоматически)
    ├── paper_bot_state.json   # Состояние бота
    ├── paper_exchange_state.json  # Состояние биржи
    ├── paper_trades.csv       # История сделок
    └── paper_portfolio.csv    # История портфеля
```

## Использование

### Быстрый старт

```bash
cd /home/user/jjhbn/GGG3/paper_trading

# Запуск с параметрами по умолчанию (1000 USDT, 10 позиций)
python3 paper_bot_main.py

# Запуск с кастомными параметрами
python3 paper_bot_main.py --initial-capital 5000 --max-positions 15

# Запуск с детальным логированием
python3 paper_bot_main.py --log-level DEBUG
```

### Параметры командной строки

```bash
--initial-capital FLOAT   # Начальный капитал в USDT (default: 1000)
--max-positions INT       # Макс. количество позиций (default: 10)
--log-level STR          # Уровень логов: DEBUG|INFO|WARNING|ERROR (default: INFO)
```

### Пример запуска

```bash
python3 paper_bot_main.py --initial-capital 10000 --max-positions 20
```

Вывод:
```
================================================================================
LOADING ML MODELS
================================================================================
✓ [XGBoost] Loaded from xgb_model.pkl
✓ [RandomForest] Loaded from rf_model.pkl
✗ [AdaptiveRF] Model file not found
...
Models loaded: 3/5
================================================================================

================================================================================
PAPER TRADING BOT INITIALIZED
================================================================================
Initial Capital: 10000.0 USDT
Max Positions: 20
Models Loaded: 3
================================================================================

================================================================================
STARTING PAPER TRADING BOT
================================================================================
Press Ctrl+C to stop
================================================================================

================================================================================
PORTFOLIO UPDATE
================================================================================
[EV Filter] Scanning 400 coins...
[EV Filter] Selected 10 coins:
  BTCUSDT      LONG  EV=+0.0812 p_up=0.632
  ETHUSDT      SHORT EV=+0.0654 p_up=0.374
  ...

[OPENING POSITION] BTCUSDT LONG
  Price: 45123.45
  ATR: 987.65
  TP: 46308.63
  SL: 44333.33
  Size: 0.0221
  EV: +0.0812
  ✓ Position opened: PAPER_A5B638BCF3F3
...
```

## Тестирование

```bash
# Запуск всех тестов
python3 test_paper_bot.py

# Вывод:
# ✓ Imports ..................... PASSED
# ✓ Configuration ............... PASSED
# ✓ Models Manager .............. PASSED
# ✓ EV Filter ................... PASSED
# ✓ ATR Calculator .............. PASSED
# ✓ Bot Initialization .......... PASSED
# ✓ Get Universe ................ PASSED
#
# ✓ ALL TESTS PASSED (7/7)
```

## Конфигурация

Основные параметры в классе `Config`:

```python
# Trading parameters
MAX_POSITIONS = 10              # Макс. кол-во позиций
RISK_PER_TRADE = 0.01          # Риск 1% на сделку

# ATR parameters
ATR_PERIOD = 10                 # Период ATR
ATR_TIMEFRAME = '4h'           # Таймфрейм для ATR
TP_MULTIPLIER = 1.2            # TP = entry ± 1.2*ATR
SL_MULTIPLIER = 0.8            # SL = entry ∓ 0.8*ATR

# Threshold
BASE_THRESHOLD = 0.51          # Базовый порог
MIN_THRESHOLD = 0.45           # Минимум
MAX_THRESHOLD = 0.70           # Максимум

# EV Filter
MIN_EV = 0.02                  # Мин. EV для входа
TOP_N_COINS = 10               # Топ-10 монет

# Timings
PORTFOLIO_UPDATE_INTERVAL = 4 * 3600   # 4 часа
CHECK_ORDERS_INTERVAL = 5 * 60         # 5 минут

# Blacklist
BLACKLIST = ['BTCDOMUSDT', 'DEFIUSDT', ...]
```

## Логи и данные

### Логи

`logs/paper_bot_YYYYMMDD.log`:
```
2025-11-06 17:00:00 [INFO] PORTFOLIO UPDATE
2025-11-06 17:00:01 [INFO] [EV Filter] Selected 10 coins...
2025-11-06 17:00:02 [INFO] [OPENING POSITION] BTCUSDT LONG
2025-11-06 17:05:00 [INFO] [ORDERS CHECK] 1 order(s) executed
2025-11-06 17:05:01 [INFO]   BTCUSDT LIMIT @ 46308.63
2025-11-06 17:05:01 [INFO]   PnL: +12.45 USDT (+1.23%)
```

### Сделки CSV

`data/paper_trades.csv`:
```csv
timestamp,action,symbol,direction,price,pnl,pnl_percent,reason
2025-11-06T17:00:00,OPEN,BTCUSDT,LONG,45123.45,,,
2025-11-06T17:05:00,CLOSE,BTCUSDT,LONG,46308.63,+12.45,+1.23,tp
```

## Статистика

Бот выводит статистику каждые 5 минут:

```
================================================================================
BOT STATUS
================================================================================
Balance: 987.65 USDT
Equity: 1012.34 USDT
Open Positions: 8/10
Total Trades: 123
Win Rate: 58.54%
Total PnL: +123.45 USDT
ROI: +12.35%
================================================================================
```

## Сравнение с реальным ботом

| Параметр | Реальный бот | Paper bot |
|----------|-------------|-----------|
| Биржа | Binance Futures API | PaperExchange |
| Цены | Реальные | Реальные (Binance mainnet) |
| Ордера | Реальные | Симуляция |
| Комиссии | Реальные (0.04%) | Симуляция (0.1%) |
| Проскальзывание | Реальное | Симуляция (0.05%) |
| ML модели | Да | Да |
| EV Filter | Да | Да |
| TP/SL | Да | Да |
| Online learning | Да | Да |
| Логирование | Да | Да |

**Вывод:** Paper bot идентичен реальному, но без реальных денег!

## Переход на реальную торговлю

Когда paper bot показывает хорошие результаты:

1. Убедитесь что Win Rate > 55%
2. ROI стабильно положительный
3. Profit Factor > 1.5

Тогда можно перейти на реальную торговлю:

```python
# Вместо PaperExchange используйте реальный Binance API
from binance.client import Client

client = Client(api_key, api_secret)
# ... (остальной код идентичен)
```

## Troubleshooting

### Проблема: No module named 'numpy'

**Решение:**
```bash
pip install numpy pandas
```

### Проблема: Models not found

**Решение:** Модели опциональны для тестирования. Для полноценной работы нужны обученные модели в `GGG3/*.pkl`.

### Проблема: 403 Forbidden from Binance

**Решение:** Используется fallback механизм с 4 альтернативными endpoints. В production это не должно происходить.

## Дополнительно

- **Мастер-план:** `BINANCE_BOT_MASTER_PLAN.md`
- **API документация:** `BINANCE_API_INFO.md`
- **PaperExchange README:** `README.md`

## Версия

**1.0.0** - Первая версия paper trading бота

## Лицензия

Часть проекта Binance Trading Bot (GGG3)
