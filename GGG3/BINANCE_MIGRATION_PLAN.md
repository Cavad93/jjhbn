# 📋 ПЛАН МИГРАЦИИ БОТА НА BINANCE SPOT/FUTURES

**Версия:** 1.0
**Дата:** 2025-11-06
**Цель:** Переход с PancakeSwap Prediction на Binance торговлю с 10 одновременными позициями

---

## 🎯 КРАТКОЕ ОПИСАНИЕ ИЗМЕНЕНИЙ

### Было (PancakeSwap Prediction):
- **Рынок:** BSC Prediction Market (бинарные ставки UP/DOWN)
- **Таймфрейм:** 5 минут (1 раунд)
- **Позиций:** 1
- **Риск:** 1-3% капитала
- **Контроль риска:** Невозможен (SL нет)
- **Свечи:** 1 минута для анализа

### Станет (Binance):
- **Рынок:** Binance Spot (реальная торговля)
- **Таймфрейм:** 4 часа (для расчета ATR)
- **Позиций:** 10 одновременно
- **Риск:** 0.5-2.5% на позицию (через Kelly)
- **TP/SL:** TP = +1.2×ATR, SL = -0.8×ATR
- **ATR:** Период 10 на 4h свечах
- **Контроль риска:** Автоматический через SL
- **Свечи:** 5-30 минут (мультитаймфрейм для разных фич)

### Архитектура (сохраняется):
- ✅ Базовая логика (4 сигнала: M, S, B, R)
- ✅ 4 ML эксперта (XGBoost, RF, ARF, NN)
- ✅ META нейросеть (CMA-ES обучение)
- ✅ 68 фич для ML
- ✅ Фазовая система (6 фаз рынка)
- ✅ Kelly criterion для риск-менеджмента

### Изменения:
- ⚡ EV → фильтр для отбора топ-10 монет из ~400
- ⚡ Target (y) → достиг TP раньше SL (вместо "цена выше через 5 мин")
- ⚡ Свечи → мультитаймфрейм (5m, 15m, 30m, 4h)
- ⚡ Управление портфелем из 10 позиций
- ⚡ Интеграция с Binance API

---

## 📊 ЭТАПЫ РЕАЛИЗАЦИИ

### Этап 1: Подготовка инфраструктуры (1-2 дня)
### Этап 2: Сбор исторических данных (2-3 дня)
### Этап 3: Адаптация фич и target (2-3 дня)
### Этап 4: EV-фильтр для отбора монет (2-3 дня)
### Этап 5: Обучение моделей (3-4 дня)
### Этап 6: Управление портфелем (2-3 дня)
### Этап 7: Backtesting (3-5 дней)
### Этап 8: Paper trading (7-14 дней)
### Этап 9: Live trading (бесконечно)

**Общее время:** 3-4 недели разработки + 2-4 недели тестирования

---

# 🚀 ЭТАП 1: ПОДГОТОВКА ИНФРАСТРУКТУРЫ

## Цель
Настроить окружение для работы с Binance API и создать базовые модули

## Задачи
1. Установить библиотеки для работы с Binance
2. Создать модуль подключения к Binance API
3. Создать базовые структуры данных для управления позициями
4. Создать конфигурационный файл

---

## 📝 ПРОМТ 1.1: Установка зависимостей

```
Установи библиотеки для работы с Binance:

1. python-binance (официальная библиотека)
2. ccxt (универсальная библиотека для бирж)

Проверь что установлены:
- pandas
- numpy
- requests

Создай файл requirements_binance.txt со всеми зависимостями.

Путь к проекту: /home/user/jjhbn/GGG3/
```

---

## 📝 ПРОМТ 1.2: Создание модуля Binance API

```
Создай модуль для работы с Binance API: /home/user/jjhbn/GGG3/binance_client.py

Функционал:
1. Класс BinanceClient с методами:
   - get_all_usdt_pairs() -> List[str]  # Получить все торговые пары USDT
   - get_ohlcv(symbol, timeframe, limit) -> pd.DataFrame  # OHLCV данные
   - get_current_price(symbol) -> float
   - get_orderbook(symbol, limit=20) -> dict
   - create_market_order(symbol, side, amount) -> dict
   - create_limit_order(symbol, side, amount, price) -> dict
   - create_stop_loss(symbol, amount, stop_price) -> dict
   - create_take_profit(symbol, amount, tp_price) -> dict
   - get_open_positions() -> List[dict]
   - get_balance(asset='USDT') -> float
   - cancel_order(symbol, order_id) -> bool

2. Обработка ошибок и retry логика
3. Rate limiting (не более 1200 запросов в минуту)
4. Логирование всех операций

Используй библиотеку python-binance или ccxt (на твой выбор).
Добавь поддержку testnet для безопасного тестирования.
```

---

## 📝 ПРОМТ 1.3: Структуры данных для позиций

```
Создай модуль для управления позициями: /home/user/jjhbn/GGG3/position_manager.py

Классы:

1. Position (dataclass):
   - symbol: str
   - entry_price: float
   - entry_time: datetime
   - amount: float
   - tp_price: float
   - sl_price: float
   - tp_order_id: Optional[str]
   - sl_order_id: Optional[str]
   - status: str  # 'open', 'tp_hit', 'sl_hit', 'timeout'
   - pnl: float
   - p_up: float  # вероятность от модели при входе
   - atr_value: float

2. PositionManager:
   - positions: Dict[str, Position]
   - add_position(position: Position)
   - remove_position(symbol: str)
   - get_position(symbol: str) -> Optional[Position]
   - get_all_open() -> List[Position]
   - update_position_status(symbol: str, status: str, pnl: float)
   - save_to_file(path: str)
   - load_from_file(path: str)
   - get_total_exposure() -> float  # сумма всех позиций
   - get_pnl_summary() -> dict  # статистика PnL

Сохранение позиций в JSON файл для персистентности.
```

---

## 📝 ПРОМТ 1.4: Конфигурационный файл

```
Создай конфигурационный файл: /home/user/jjhbn/GGG3/binance_config.py

Параметры:

# API ключи
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
USE_TESTNET = os.getenv("USE_TESTNET", "1") == "1"  # по умолчанию testnet

# Торговые параметры
MAX_POSITIONS = 10
MIN_POSITION_SIZE_USDT = 15.0  # минимальный размер позиции
MAX_POSITION_SIZE_USDT = 1000.0  # максимальный размер позиции

# ATR параметры
ATR_PERIOD = 10
ATR_TIMEFRAME = "4h"
TP_ATR_MULTIPLIER = 1.2
SL_ATR_MULTIPLIER = 0.8

# Риск менеджмент
MIN_RISK_PER_POSITION = 0.005  # 0.5%
MAX_RISK_PER_POSITION = 0.025  # 2.5%
DEFAULT_RISK = 0.009  # 0.9%

# Фильтры ликвидности
MIN_VOLUME_24H_USDT = 5_000_000  # $5M минимум
MAX_SPREAD_PCT = 0.002  # 0.2% максимальный спред
MIN_ATR_PCT = 0.02  # 2% минимальная волатильность
MAX_ATR_PCT = 0.15  # 15% максимальная волатильность

# Таймауты
MAX_POSITION_DURATION_HOURS = 168  # 7 дней
POSITION_CHECK_INTERVAL_SEC = 300  # проверка каждые 5 минут

# EV фильтр
MIN_EV_THRESHOLD = 0.02  # минимальный EV для входа
TOP_N_COINS = 10  # количество монет для торговли

# Свечи для анализа
CANDLE_TIMEFRAMES = {
    "fast": "5m",    # быстрые индикаторы (momentum, RSI)
    "medium": "15m",  # средние (VWAP, Bollinger)
    "slow": "30m",    # медленные (тренды)
    "atr": "4h"      # ATR и фазы рынка
}
CANDLE_LIMIT = 500  # количество свечей для анализа

# Paths
DATA_DIR = "/home/user/jjhbn/GGG3/binance_data"
MODELS_DIR = "/home/user/jjhbn/GGG3/binance_models"
POSITIONS_FILE = "/home/user/jjhbn/GGG3/positions.json"
TRADES_CSV = "/home/user/jjhbn/GGG3/binance_trades.csv"

Также добавь функцию validate_config() для проверки корректности параметров.
```

---

# 🚀 ЭТАП 2: СБОР ИСТОРИЧЕСКИХ ДАННЫХ

## Цель
Собрать исторические OHLCV данные для всех USDT пар на Binance

## Задачи
1. Получить список всех торгуемых пар
2. Скачать исторические данные (минимум 3 месяца)
3. Сохранить в структурированном формате
4. Создать индекс для быстрого доступа

---

## 📝 ПРОМТ 2.1: Модуль сбора данных

```
Создай модуль для сбора исторических данных: /home/user/jjhbn/GGG3/collect_binance_data.py

Функционал:

1. Функция get_all_tradeable_pairs() -> List[str]:
   - Получить все USDT пары
   - Отфильтровать по минимальному объему (> $1M за 24h)
   - Отфильтровать стабильные монеты (USDC, BUSD, etc)
   - Вернуть ~300-400 пар

2. Функция download_ohlcv_data(
    symbol: str,
    timeframes: List[str],  # ['5m', '15m', '30m', '4h']
    start_date: str,  # '2024-08-01'
    end_date: str     # '2024-11-06'
   ) -> Dict[str, pd.DataFrame]:
   - Скачать OHLCV для каждого таймфрейма
   - Обработать пропуски данных
   - Валидация (нет дублей, монотонность времени)
   - Сохранить в parquet формат для быстрого доступа

3. Функция collect_all_data(
    pairs: List[str],
    output_dir: str = '/home/user/jjhbn/GGG3/binance_data'
   ):
   - Скачать данные для всех пар
   - Показывать прогресс (progress bar)
   - Retry при ошибках
   - Логировать статистику

4. Создать индекс: data_index.json
   {
     "BTCUSDT": {
       "5m": "binance_data/BTCUSDT_5m.parquet",
       "15m": "binance_data/BTCUSDT_15m.parquet",
       "30m": "binance_data/BTCUSDT_30m.parquet",
       "4h": "binance_data/BTCUSDT_4h.parquet",
       "last_update": "2024-11-06T12:00:00",
       "rows": 12000
     },
     ...
   }

Период: последние 3 месяца (достаточно для обучения).
Обработка rate limits: задержки между запросами.
```

---

## 📝 ПРОМТ 2.2: Запуск сбора данных

```
Запусти сбор исторических данных:

python3 /home/user/jjhbn/GGG3/collect_binance_data.py

Параметры:
- Период: 2024-08-01 до 2024-11-06 (3 месяца)
- Таймфреймы: 5m, 15m, 30m, 4h
- Папка: /home/user/jjhbn/GGG3/binance_data/

Ожидаемый результат:
- ~400 пар × 4 таймфрейма = 1600 файлов
- Размер: ~500-1000 MB
- Время: 2-4 часа (из-за rate limits)

После завершения:
1. Проверь количество скачанных файлов
2. Проверь data_index.json
3. Выведи статистику: сколько монет, общий размер данных
```

---

## 📝 ПРОМТ 2.3: Валидация данных

```
Создай скрипт валидации данных: /home/user/jjhbn/GGG3/validate_binance_data.py

Проверки:
1. Все ли файлы существуют (из data_index.json)
2. Нет ли пропусков во времени (gaps)
3. Нет ли дублей timestamp
4. Корректность OHLCV (low <= close <= high, etc)
5. Минимальный объем данных (>= 1000 свечей)

Выведи отчет:
- Количество корректных пар
- Количество пар с проблемами
- Детали проблем для каждой пары

Если есть проблемные пары - перезапусти сбор только для них.
```

---

# 🚀 ЭТАП 3: АДАПТАЦИЯ ФИЧ И TARGET

## Цель
Адаптировать существующие фичи под новый таймфрейм и создать новый target (TP/SL)

## Задачи
1. Модифицировать расчет фич для мультитаймфрейма
2. Создать функцию расчета ATR
3. Создать функцию определения target (TP/SL)
4. Адаптировать фазовую систему

---

## 📝 ПРОМТ 3.1: Адаптация базовых фич

```
Модифицируй модуль фич для Binance: создай /home/user/jjhbn/GGG3/binance_features.py

На основе текущего кода (bnbusdrt6.py и market_features.py), но адаптированный:

1. Класс BinanceFeatureBuilder:
   - Принимает мультитаймфрейм данные (5m, 15m, 30m, 4h)
   - Рассчитывает базовые фичи:
     * Momentum (M_up, M_dn) - на 5m свечах
     * VWAP slope (S_up, S_dn) - на 15m свечах
     * Bollinger bands (B_up, B_dn, R_up, R_dn) - на 15m свечах
     * Keltner channels (kc_pos, kc_w) - на 30m свечах
     * ATR (atr, atr_norm) - на 4h свечах
     * RSI (rsi, rsi_norm) - на 15m свечах
     * Volume indicators - комбинация всех таймфреймов

2. Метод build_extended_features(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_30m: pd.DataFrame,
    df_4h: pd.DataFrame,
    current_time: datetime
   ) -> np.ndarray:
   - Возвращает 68 фич (те же что в prepare_training_features.py)
   - Но рассчитанные на разных таймфреймах

3. Функция calculate_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
   - True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
   - ATR = EMA(True Range, period)

4. Функция calculate_regime_context(df_4h: pd.DataFrame) -> dict:
   - trend_sign: знак тренда (MACD на дневных свечах)
   - trend_abs: абсолютная величина тренда
   - vol_ratio: волатильность относительно средней (ATR/ATR_SMA)
   - phase: фаза рынка (0-5)

Важно: сохранить совместимость с текущими 68 фичами!
```

---

## 📝 ПРОМТ 3.2: Функция определения target (TP/SL)

```
Создай модуль для расчета target: /home/user/jjhbn/GGG3/binance_target.py

Функция calculate_tp_sl_outcome(
    df: pd.DataFrame,  # OHLCV данные (4h timeframe)
    entry_idx: int,    # индекс свечи входа
    entry_price: float,
    atr_value: float,
    tp_mult: float = 1.2,
    sl_mult: float = 0.8,
    max_bars: int = 100  # максимум 100 свечей (400 часов = 16 дней)
) -> Optional[dict]:
    """
    Определяет исход сделки: достиг TP или SL раньше

    Returns:
        {
          'outcome': 1 (TP) или 0 (SL),
          'bars_held': количество свечей до закрытия,
          'exit_price': цена выхода,
          'pnl_pct': PnL в процентах
        }
        или None если ни TP ни SL не достигнуты за max_bars
    """
    # Уровни
    tp_price = entry_price + tp_mult * atr_value
    sl_price = entry_price - sl_mult * atr_value

    # Смотрим следующие свечи
    for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]

        # Проверяем TP
        if high >= tp_price:
            return {
                'outcome': 1,
                'bars_held': i - entry_idx,
                'exit_price': tp_price,
                'pnl_pct': (tp_price - entry_price) / entry_price
            }

        # Проверяем SL
        if low <= sl_price:
            return {
                'outcome': 0,
                'bars_held': i - entry_idx,
                'exit_price': sl_price,
                'pnl_pct': (sl_price - entry_price) / entry_price
            }

    return None  # таймаут

Также создай функцию prepare_training_dataset() которая:
1. Проходит по всем свечам
2. Для каждой рассчитывает фичи
3. Определяет outcome (TP/SL)
4. Сохраняет в датасет
```

---

## 📝 ПРОМТ 3.3: Создание датасета для обучения

```
Создай скрипт подготовки датасета: /home/user/jjhbn/GGG3/prepare_binance_dataset.py

Функционал:

1. Загрузить данные для всех ~400 монет
2. Для каждой монеты и каждой свечи:
   - Рассчитать 68 фич
   - Рассчитать ATR
   - Определить TP/SL outcome
   - Определить фазу рынка
   - Сохранить: (features, outcome, phase, symbol, timestamp)

3. Фильтрация:
   - Пропустить свечи без outcome (таймаут)
   - Пропустить аномальные значения (ATR < 0.5% или > 20%)
   - Пропустить низколиквидные периоды

4. Сохранить в формате:
   - binance_dataset_train.parquet (первые 80% времени)
   - binance_dataset_test.parquet (последние 20% времени)
   - dataset_stats.json (статистика)

Ожидаемый размер датасета:
- ~400 монет × ~1000 свечей (4h за 3 месяца) = 400,000 примеров
- После фильтрации: ~300,000 примеров
- Соотношение TP/SL: должно быть примерно 50/50

Запусти подготовку:
python3 /home/user/jjhbn/GGG3/prepare_binance_dataset.py

Выведи статистику:
- Количество примеров
- Распределение TP/SL (винрейт)
- Распределение по фазам
- Среднее количество баров до закрытия
```

---

# 🚀 ЭТАП 4: EV-ФИЛЬТР ДЛЯ ОТБОРА МОНЕТ

## Цель
Создать систему отбора топ-10 монет с наибольшим Expected Value

## Задачи
1. Создать функцию расчета EV для каждой монеты
2. Реализовать фильтры ликвидности
3. Создать ранжирование монет
4. Реализовать динамическое обновление топ-10

---

## 📝 ПРОМТ 4.1: Модуль EV-фильтра

```
Создай модуль EV-фильтра: /home/user/jjhbn/GGG3/coin_selector.py

Класс CoinSelector:

1. Метод calculate_ev(
    symbol: str,
    p_up: float,  # вероятность от META модели
    current_price: float,
    atr: float
   ) -> float:
    """
    Рассчитывает Expected Value для монеты

    EV = P(TP) × Reward - P(SL) × Risk

    Где:
    - P(TP) = p_up
    - Reward = 1.2 × ATR = 1.5R (в единицах риска R = 0.8×ATR)
    - P(SL) = 1 - p_up
    - Risk = 1.0R

    EV = p_up × 1.5R - (1 - p_up) × 1.0R
       = p_up × 1.5R - 1.0R + p_up × 1.0R
       = p_up × 2.5R - 1.0R

    Для прибыльности: EV > 0 => p_up > 0.4 (40%)
    """
    ev = p_up * 2.5 - 1.0
    return ev

2. Метод apply_filters(symbol: str) -> bool:
    """Фильтры ликвидности и качества"""
    # Получить данные монеты
    df = get_ohlcv(symbol, '4h', limit=100)

    # Фильтр 1: Объем 24h > $5M
    volume_24h = df['volume'].iloc[-6:].sum() * df['close'].iloc[-1]
    if volume_24h < 5_000_000:
        return False

    # Фильтр 2: Спред < 0.2%
    orderbook = get_orderbook(symbol)
    spread = (orderbook['asks'][0][0] - orderbook['bids'][0][0]) / orderbook['bids'][0][0]
    if spread > 0.002:
        return False

    # Фильтр 3: ATR в разумных пределах (2-15%)
    atr = calculate_atr(df, period=10)
    atr_pct = atr.iloc[-1] / df['close'].iloc[-1]
    if not (0.02 < atr_pct < 0.15):
        return False

    return True

3. Метод select_top_coins(
    all_pairs: List[str],
    experts: dict,  # 4 эксперта
    meta: MetaNeuralCEM,
    base_model,
    top_n: int = 10
   ) -> List[dict]:
    """
    Отбирает топ-N монет с наибольшим EV

    Для каждой монеты:
    1. Рассчитать фичи
    2. Получить предсказания от 4 экспертов
    3. Получить предсказание базовой модели
    4. Получить финальное предсказание от META
    5. Рассчитать EV
    6. Применить фильтры
    7. Отсортировать по EV
    8. Вернуть топ-N
    """
    results = []

    for symbol in all_pairs:
        # Применяем фильтры
        if not self.apply_filters(symbol):
            continue

        # Получаем данные
        df_5m = get_ohlcv(symbol, '5m', 200)
        df_15m = get_ohlcv(symbol, '15m', 200)
        df_30m = get_ohlcv(symbol, '30m', 200)
        df_4h = get_ohlcv(symbol, '4h', 100)

        # Рассчитываем фичи
        features = build_extended_features(df_5m, df_15m, df_30m, df_4h)

        # Предсказания экспертов
        p_xgb = experts['xgb'].predict(features)
        p_rf = experts['rf'].predict(features)
        p_arf = experts['arf'].predict(features)
        p_nn = experts['nn'].predict(features)

        # Базовая модель
        p_base = base_model.predict(features)

        # META
        ctx = calculate_regime_context(df_4h)
        p_meta = meta.predict(p_xgb, p_rf, p_arf, p_nn, p_base, ctx['phase'])

        # EV
        atr = calculate_atr(df_4h, 10).iloc[-1]
        ev = self.calculate_ev(symbol, p_meta, df_4h['close'].iloc[-1], atr)

        results.append({
            'symbol': symbol,
            'p_up': p_meta,
            'ev': ev,
            'atr_pct': atr / df_4h['close'].iloc[-1],
            'volume_24h': df_4h['volume'].iloc[-24:].sum() * df_4h['close'].iloc[-1]
        })

    # Сортируем по EV
    results.sort(key=lambda x: x['ev'], reverse=True)

    return results[:top_n]

Также добавь метод для диверсификации по секторам (максимум 3 монеты из одного сектора).
```

---

## 📝 ПРОМТ 4.2: Тестирование EV-фильтра

```
Создай тест EV-фильтра: /home/user/jjhbn/GGG3/test_coin_selector.py

Задачи:
1. Загрузить обученные модели (эксперты + META)
2. Запустить отбор монет на последних данных
3. Вывести топ-10 с метриками:
   - Symbol
   - P_up (вероятность роста)
   - EV (expected value)
   - ATR % (волатильность)
   - Volume 24h
   - Current price
   - TP price
   - SL price

4. Проверить:
   - Все ли монеты прошли фильтры
   - EV > 0.02 для всех
   - Диверсификация по секторам

Пример вывода:
```
=== ТОП-10 МОНЕТ ДЛЯ ТОРГОВЛИ ===

Rank | Symbol    | P_up   | EV     | ATR%  | Volume 24h  | Price   | TP      | SL
-----|-----------|--------|--------|-------|-------------|---------|---------|--------
1    | BTCUSDT   | 0.6234 | 0.0585 | 3.2%  | $450M       | $42,500 | $44,632 | $41,416
2    | ETHUSDT   | 0.5987 | 0.0468 | 4.1%  | $320M       | $2,250  | $2,360  | $2,177
3    | SOLUSDT   | 0.6123 | 0.0558 | 5.5%  | $180M       | $98.50  | $105.0  | $94.12
...
```

Запусти тест:
python3 /home/user/jjhbn/GGG3/test_coin_selector.py
```

---

# 🚀 ЭТАП 5: ОБУЧЕНИЕ МОДЕЛЕЙ

## Цель
Переобучить все 4 эксперта и META на новом датасете с TP/SL targets

## Задачи
1. Адаптировать обучение экспертов под новый target
2. Обучить все 4 эксперта на Binance датасете
3. Обучить META нейросеть
4. Валидация качества моделей

---

## 📝 ПРОМТ 5.1: Адаптация обучения экспертов

```
Модифицируй скрипт обучения: создай /home/user/jjhbn/GGG3/train_binance_experts.py

На основе train_bot_full.py, но с изменениями:

1. Загрузка датасета:
   - Загрузить binance_dataset_train.parquet
   - Разделить по фазам (0-5)
   - Статистика по классам (TP vs SL)

2. Обучение экспертов:
   a) XGBoost:
      - Параметры: max_depth=6, n_estimators=200, learning_rate=0.05
      - Scale_pos_weight для балансировки классов
      - Раздельные модели по фазам

   b) RandomForest:
      - n_estimators=150, max_depth=12
      - CalibratedClassifierCV для калибровки
      - class_weight='balanced'

   c) River ARF:
      - Онлайн обучение (последовательно)
      - n_models=10
      - grace_period=50

   d) NNExpert:
      - Архитектура: 68 → 32 → 16 → 1
      - Batch training
      - Temperature scaling для калибровки

3. Метрики на тесте:
   - Accuracy
   - Precision / Recall
   - ROC-AUC
   - Calibration error (ECE)
   - Brier score

4. Сохранение:
   - /home/user/jjhbn/GGG3/binance_models/xgb/
   - /home/user/jjhbn/GGG3/binance_models/rf/
   - /home/user/jjhbn/GGG3/binance_models/arf/
   - /home/user/jjhbn/GGG3/binance_models/nn/

Ожидаемые результаты:
- Accuracy > 52% (better than baseline 40%)
- ROC-AUC > 0.55
- ECE < 0.05 (хорошая калибровка)

Запусти обучение:
python3 /home/user/jjhbn/GGG3/train_binance_experts.py
```

---

## 📝 ПРОМТ 5.2: Обучение META нейросети

```
Создай скрипт обучения META: /home/user/jjhbn/GGG3/train_binance_meta.py

На основе текущей meta_neural_cem.py:

1. Загрузка:
   - Загрузить обученных экспертов
   - Загрузить датасет

2. Генерация предсказаний экспертов:
   - Для каждого примера получить p_xgb, p_rf, p_arf, p_nn, p_base
   - Сохранить матрицу предсказаний

3. Обучение META:
   - Архитектура: та же (36D → 64 → 32 → 16 → 1)
   - Обучение через CMA-ES
   - Раздельные головы по фазам
   - MC Dropout для uncertainty

4. Валидация:
   - Accuracy META vs среднее экспертов
   - Улучшение должно быть > +2%
   - Calibration (ECE)

5. Режим:
   - Оставить META в режиме "SHADOW"
   - Это значит: META предсказывает, но НЕ влияет на решения
   - Решения принимаются базовой логикой + 4 эксперта

Ожидаемые результаты:
- META accuracy: 54-56%
- Improvement над средним экспертов: +2-3%
- ECE < 0.04

Запусти обучение:
python3 /home/user/jjhbn/GGG3/train_binance_meta.py --epochs 50

Важно: META должна быть в SHADOW режиме для начала, чтобы собрать статистику!
```

---

## 📝 ПРОМТ 5.3: Валидация моделей

```
Создай скрипт валидации: /home/user/jjhbn/GGG3/validate_binance_models.py

Тесты:

1. Загрузка моделей:
   - Проверить что все модели загружаются без ошибок
   - Проверить размерности

2. Тест на синтетических данных:
   - Создать 1000 случайных примеров
   - Проверить что предсказания в диапазоне [0, 1]
   - Проверить что нет NaN/Inf

3. Тест на реальных данных:
   - Загрузить test dataset
   - Получить предсказания от всех моделей
   - Вычислить метрики:
     * Accuracy
     * ROC-AUC
     * Precision/Recall
     * Calibration (ECE)
     * Brier score

4. Тест на новых данных (свежие):
   - Скачать последние 7 дней данных
   - Сделать предсказания
   - Проверить стабильность (нет деградации)

5. Сравнение моделей:
   - Таблица с метриками всех экспертов + META
   - Определить лучшую модель
   - Проверить что META улучшает результаты

Запусти валидацию:
python3 /home/user/jjhbn/GGG3/validate_binance_models.py

Пример вывода:
```
=== ВАЛИДАЦИЯ МОДЕЛЕЙ ===

Model        | Accuracy | ROC-AUC | Precision | Recall | ECE    | Brier
-------------|----------|---------|-----------|--------|--------|-------
XGBoost      | 53.2%    | 0.568   | 0.545     | 0.612  | 0.042  | 0.238
RandomForest | 52.8%    | 0.562   | 0.538     | 0.598  | 0.038  | 0.241
ARF          | 51.9%    | 0.551   | 0.529     | 0.585  | 0.051  | 0.247
NN           | 52.5%    | 0.559   | 0.541     | 0.592  | 0.045  | 0.242
-------------|----------|---------|-----------|--------|--------|-------
Avg Experts  | 52.6%    | 0.560   | 0.538     | 0.597  | 0.044  | 0.242
META (SHADOW)| 54.8%    | 0.582   | 0.562     | 0.619  | 0.036  | 0.229
-------------|----------|---------|-----------|--------|--------|-------
Improvement  | +2.2%    | +0.022  | +0.024    | +0.022 | -0.008 | -0.013

✅ ВСЕ МОДЕЛИ ВАЛИДНЫ
✅ META УЛУЧШАЕТ НА +2.2%
```
```

---

# 🚀 ЭТАП 6: УПРАВЛЕНИЕ ПОРТФЕЛЕМ

## Цель
Реализовать логику управления портфелем из 10 одновременных позиций

## Задачи
1. Создать основной торговый цикл
2. Реализовать вход в позиции
3. Реализовать мониторинг позиций
4. Реализовать выход (TP/SL/Timeout)
5. Kelly criterion для размера позиций

---

## 📝 ПРОМТ 6.1: Основной торговый бот

```
Создай главный файл бота: /home/user/jjhbn/GGG3/binance_trading_bot.py

Класс BinanceTradingBot:

1. Инициализация:
   - Загрузить модели (эксперты + META)
   - Подключиться к Binance
   - Загрузить позиции (если есть)
   - Инициализировать PositionManager
   - Инициализировать CoinSelector
   - Загрузить конфиг

2. Основной цикл (run()):

   while True:
       # Каждые 4 часа (когда закрывается 4h свеча)
       if is_4h_candle_closed():

           # Шаг 1: Обновить топ-10 монет
           top_coins = coin_selector.select_top_coins(
               all_pairs=get_all_usdt_pairs(),
               experts=experts,
               meta=meta,
               base_model=base_model,
               top_n=10
           )

           # Шаг 2: Закрыть позиции не в топ-10
           current_positions = position_manager.get_all_open()
           for pos in current_positions:
               if pos.symbol not in [c['symbol'] for c in top_coins]:
                   close_position(pos, reason='not_in_top10')

           # Шаг 3: Открыть новые позиции
           for coin_data in top_coins:
               if position_manager.has_position(coin_data['symbol']):
                   continue  # уже открыта

               # Открыть позицию
               open_position(
                   symbol=coin_data['symbol'],
                   p_up=coin_data['p_up'],
                   ev=coin_data['ev']
               )

       # Каждые 5 минут: проверить позиции
       if time.time() - last_check > 300:
           check_positions()
           last_check = time.time()

       time.sleep(60)  # спать 1 минуту

3. Метод open_position():
   - Рассчитать размер позиции (Kelly)
   - Получить текущую цену
   - Рассчитать ATR
   - Рассчитать TP/SL уровни
   - Создать market order
   - Установить TP и SL orders
   - Сохранить в PositionManager
   - Записать в лог

4. Метод check_positions():
   - Для каждой открытой позиции:
     * Проверить статус ордеров (TP/SL)
     * Если TP исполнен → закрыть позицию, записать успех
     * Если SL исполнен → закрыть позицию, записать неудачу
     * Если таймаут (> 7 дней) → закрыть по рынку
   - Обновить статистику
   - Обучить модели на новых результатах

5. Метод close_position():
   - Отменить TP/SL ордера
   - Создать market sell order
   - Обновить PositionManager
   - Записать результат для обучения
   - Логировать

6. Kelly criterion для размера позиции:
   def calculate_position_size(
       capital: float,
       p_up: float,
       ev: float,
       recent_wr: float
   ) -> float:
       # Классическая Kelly формула
       kelly_fraction = ev / 1.5  # R/R = 1.5

       # Half Kelly для консерватизма
       kelly_half = kelly_fraction * 0.5

       # Ограничения
       risk_pct = np.clip(kelly_half, 0.005, 0.025)  # 0.5-2.5%

       # Адаптация по винрейту
       if recent_wr < 0.50:
           risk_pct *= 0.7  # снизить при плохом WR

       position_value = capital * risk_pct
       return position_value

Добавь подробное логирование всех операций.
```

---

## 📝 ПРОМТ 6.2: Обучение на живых данных

```
Добавь в binance_trading_bot.py методы для онлайн обучения:

1. Метод record_trade_result():
   - Когда позиция закрывается (TP/SL/Timeout)
   - Сохранить:
     * Фичи на момент входа (68D)
     * Outcome: 1 (TP) или 0 (SL)
     * Фаза рынка
     * Время удержания
     * PnL
   - Передать результат всем экспертам:
     expert.record_result(x_raw=features, y_up=outcome, ...)
   - Передать результат META:
     meta.record_result(p_xgb, p_rf, p_arf, p_nn, p_base, y_up=outcome, ...)

2. Метод retrain_models():
   - Вызывается каждые 100 закрытых сделок
   - Переобучает всех экспертов
   - Переобучает META (если достаточно данных)
   - Сохраняет новые модели
   - Логирует метрики

3. Метод save_checkpoint():
   - Сохранить состояние бота
   - Сохранить модели
   - Сохранить позиции
   - Сохранить историю сделок

Важно:
- TP закрытия = успех (y=1)
- SL закрытия = неудача (y=0)
- Это обеспечит правильное обучение моделей!
```

---

## 📝 ПРОМТ 6.3: Риск-менеджмент

```
Создай модуль риск-менеджмента: /home/user/jjhbn/GGG3/risk_manager.py

Класс RiskManager:

1. Метод check_daily_drawdown(trades_today: List[dict]) -> bool:
   - Если дневная просадка > 5% → остановить торговлю на сегодня
   - Логировать предупреждение

2. Метод check_max_drawdown(capital_history: List[float]) -> bool:
   - Если просадка от ATH > 15% → остановить бота
   - Требовать ручное разрешение для продолжения

3. Метод calculate_total_exposure(positions: List[Position]) -> float:
   - Общая стоимость всех позиций
   - Не должна превышать 95% капитала

4. Метод check_correlation(positions: List[Position]) -> float:
   - Рассчитать корреляцию между монетами
   - Если средняя корреляция > 0.7 → предупреждение

5. Метод adjust_risk_per_position(
    base_risk: float,
    recent_wr: float,
    volatility: float,
    correlation: float
   ) -> float:
   - Динамическая адаптация риска
   - При высокой волатильности → снизить
   - При высокой корреляции → снизить
   - При низком винрейте → снизить

Интегрировать в основной бот:
- Проверки перед каждой сделкой
- Автоматическая остановка при нарушениях
- Алерты в Telegram
```

---

# 🚀 ЭТАП 7: BACKTESTING

## Цель
Протестировать стратегию на исторических данных

## Задачи
1. Создать движок бэктеста
2. Симуляция портфеля из 10 позиций
3. Расчет метрик производительности
4. Анализ результатов

---

## 📝 ПРОМТ 7.1: Движок бэктеста

```
Создай бэктест: /home/user/jjhbn/GGG3/backtest_binance_strategy.py

Класс BinanceBacktest:

1. Параметры:
   - Начальный капитал: $10,000
   - Период: 2024-08-01 до 2024-11-06 (3 месяца)
   - Комиссия: 0.1% (Spot) на вход и выход
   - Проскальзывание: 0.05%

2. Логика:
   - Симуляция работает по 4h свечам
   - Каждые 4h:
     * Обновить топ-10 монет
     * Закрыть позиции не в топ-10
     * Открыть новые позиции
   - Между свечами:
     * Проверить TP/SL для открытых позиций
     * Использовать high/low для точности

3. Метод run():
   - Пройти по всем 4h свечам в периоде
   - Для каждой свечи:
     * Обновить капитал
     * Выбрать топ-10
     * Управлять позициями
     * Записать состояние
   - Вернуть историю сделок и капитала

4. Метрики:
   - Total Return (%)
   - Annualized Return (%)
   - Max Drawdown (%)
   - Sharpe Ratio
   - Sortino Ratio
   - Win Rate (%)
   - Profit Factor
   - Average Win / Average Loss
   - Total Trades
   - Average Trade Duration (hours)
   - Количество одновременных позиций (среднее)

5. Визуализация:
   - График капитала
   - График просадки
   - Распределение PnL
   - Win rate по монетам
   - Win rate по месяцам

Запусти бэктест:
python3 /home/user/jjhbn/GGG3/backtest_binance_strategy.py \
    --start-date 2024-08-01 \
    --end-date 2024-11-06 \
    --initial-capital 10000 \
    --output backtest_results.html

Ожидаемые результаты (при винрейте 50%):
- Total Return: +15-20%
- Max Drawdown: < 20%
- Sharpe Ratio: > 1.5
- Win Rate: 48-52%
```

---

## 📝 ПРОМТ 7.2: Анализ результатов бэктеста

```
После запуска бэктеста, проанализируй результаты:

1. Загрузи отчет: backtest_results.html
2. Проверь ключевые метрики:
   - Win Rate должен быть > 48% (выше break-even 40%)
   - Max Drawdown < 20%
   - Sharpe > 1.5
   - Profit Factor > 1.3

3. Анализ по периодам:
   - Есть ли периоды с большими потерями?
   - Какая производительность по месяцам?
   - Есть ли сезонность?

4. Анализ по монетам:
   - Какие монеты дают лучший результат?
   - Есть ли монеты с постоянными потерями?
   - Стоит ли добавить blacklist?

5. Анализ портфеля:
   - Среднее количество позиций: должно быть ~8-10
   - Корреляция между позициями
   - Экспозиция капитала

6. Если результаты хуже ожидаемых:
   - Проверить качество моделей (переобучение?)
   - Проверить параметры TP/SL (может 1.2/0.8 не оптимально?)
   - Проверить фильтры EV (слишком строгие/мягкие?)
   - Добавить дополнительные фильтры

7. Оптимизация:
   - Попробовать разные TP/SL (1.5/1.0, 1.0/0.7, etc)
   - Попробовать разные топ-N (5, 15, 20 монет)
   - Попробовать разные пороги EV

Сделай выводы и рекомендации для улучшения.
```

---

# 🚀 ЭТАП 8: PAPER TRADING

## Цель
Запустить бота в paper trading режиме на реальных данных без реальных денег

## Задачи
1. Настроить paper trading режим
2. Запустить бота
3. Мониторить производительность
4. Собрать статистику

---

## 📝 ПРОМТ 8.1: Настройка paper trading

```
Добавь в binance_trading_bot.py режим paper trading:

1. Параметр --paper-trading:
   - Если True:
     * Использовать виртуальный капитал (не реальный баланс)
     * Симулировать ордера (не отправлять на биржу)
     * Использовать реальные цены
     * Проверять исполнение через реальные high/low

2. Класс PaperTradingExchange:
   - Симулирует биржу
   - Хранит виртуальный баланс
   - Симулирует исполнение ордеров
   - Проверяет TP/SL по реальным ценам

3. Логирование:
   - Все сделки записываются в paper_trades.csv
   - Состояние капитала в paper_capital.json
   - Позиции в paper_positions.json

4. Отчеты:
   - Ежедневный отчет в Telegram
   - График капитала
   - Текущие позиции
   - Метрики (винрейт, PnL, etc)

Запусти paper trading:
python3 /home/user/jjhbn/GGG3/binance_trading_bot.py \
    --paper-trading \
    --initial-capital 10000 \
    --config binance_config.py

Запусти на 2-4 недели для сбора статистики.
```

---

## 📝 ПРОМТ 8.2: Мониторинг paper trading

```
Создай дашборд для мониторинга: /home/user/jjhbn/GGG3/paper_trading_dashboard.py

Используй Streamlit или создай HTML страницу с:

1. Текущее состояние:
   - Капитал (текущий, начальный, изменение %)
   - Открытые позиции (таблица)
   - PnL за сегодня/неделю/месяц

2. Метрики:
   - Win Rate
   - Total Trades
   - Sharpe Ratio
   - Max Drawdown

3. Графики:
   - График капитала (real-time)
   - График просадки
   - Распределение PnL
   - Win rate по дням

4. Топ-10 монет:
   - Текущий список
   - EV для каждой
   - P_up для каждой

5. Последние сделки:
   - Таблица с последними 20 сделками
   - Symbol, Entry, Exit, PnL, Duration

Запусти дашборд:
streamlit run paper_trading_dashboard.py

Или создай статический HTML:
python3 generate_paper_trading_report.py --output report.html

Проверяй дашборд каждый день и отслеживай:
- Стабильность винрейта (не должен сильно колебаться)
- Нет ли больших потерь
- Правильно ли работает EV-фильтр
- Достаточно ли монет в топ-10
```

---

## 📝 ПРОМТ 8.3: Анализ paper trading результатов

```
После 2-4 недель paper trading, проанализируй:

1. Загрузи данные:
   - paper_trades.csv
   - paper_capital.json

2. Рассчитай метрики:
   - Win Rate (должен быть 48-55%)
   - Sharpe Ratio (должен быть > 1.5)
   - Max Drawdown (должна быть < 15%)
   - Total Return за период

3. Сравни с бэктестом:
   - Метрики схожи?
   - Если сильно отличаются → проблема (overfitting? market shift?)

4. Анализ проблем:
   - Есть ли системные ошибки?
   - Проскальзывание выше ожидаемого?
   - EV-фильтр работает правильно?
   - Модели стабильны?

5. Решение о переходе на live:
   - ✅ Если Win Rate > 50% И Max DD < 15% И Sharpe > 1.5
   - ❌ Если любой из критериев не выполнен → продолжить paper trading

6. Создай финальный отчет:
   - Все метрики
   - Графики
   - Анализ сделок
   - Рекомендации

Сохрани отчет: paper_trading_final_report.md
```

---

# 🚀 ЭТАП 9: LIVE TRADING

## Цель
Запустить бота на реальные деньги

## Задачи
1. Настроить боевое окружение
2. Запустить с малым капиталом
3. Постепенно увеличивать капитал
4. Мониторить и оптимизировать

---

## 📝 ПРОМТ 9.1: Подготовка к live trading

```
Перед запуском на реальные деньги:

1. Создай боевой конфиг: /home/user/jjhbn/GGG3/binance_config_live.py
   - USE_TESTNET = False
   - Реальные API ключи
   - Малый начальный капитал ($500-1000)
   - Более консервативные параметры:
     * MAX_RISK_PER_POSITION = 0.015  # 1.5% вместо 2.5%
     * MAX_POSITIONS = 5  # начать с 5 вместо 10
     * MIN_EV_THRESHOLD = 0.03  # повысить порог

2. Настрой алерты:
   - Telegram бот для уведомлений
   - Алерты при:
     * Открытии/закрытии позиций
     * Просадке > 5%
     * Ошибках API
     * Неожиданном поведении моделей

3. Настрой мониторинг:
   - Логи в файл + stdout
   - Healthcheck endpoint (HTTP)
   - Автоматический restart при критических ошибках

4. Безопасность:
   - API ключи в .env файле
   - Ограничения на API (только spot trade, no withdrawal)
   - 2FA на Binance аккаунте
   - IP whitelist для API

5. Backup:
   - Автоматический backup моделей (каждый день)
   - Backup позиций (каждый час)
   - Backup истории сделок

Запусти в screen/tmux:
screen -S binance_bot
python3 /home/user/jjhbn/GGG3/binance_trading_bot.py \
    --config binance_config_live.py \
    --initial-capital 1000

Детач: Ctrl+A, D
Проверить: screen -r binance_bot
```

---

## 📝 ПРОМТ 9.2: План постепенного увеличения капитала

```
Стратегия масштабирования:

Неделя 1-2: $500-1000, 5 позиций
- Цель: проверить стабильность
- Ожидаемый PnL: +$50-100 (5-10%)
- Критерий успеха: Win Rate > 48%, Max DD < 10%

Неделя 3-4: $1500-2000, 7 позиций
- Увеличить капитал на 50%
- Ожидаемый PnL: +$150-200 (10%)
- Критерий успеха: стабильный винрейт, нет больших проблем

Неделя 5-8: $3000-5000, 10 позиций
- Полная стратегия
- Ожидаемый PnL: +$300-500 (10%)
- Критерий успеха: Sharpe > 1.5, винрейт стабилен

Месяц 3+: Постепенное увеличение до целевого капитала
- Добавлять +20-30% каждый месяц
- Реинвестировать прибыль

СТОП-УСЛОВИЯ (немедленная остановка):
1. Просадка > 15% от ATH
2. Win Rate < 45% за последние 50 сделок
3. 3 дня подряд с убытками > 2%
4. Технические проблемы (API errors, model failures)

При стоп-условии:
- Закрыть все позиции
- Проанализировать причины
- Вернуться к paper trading
- Исправить проблемы
```

---

## 📝 ПРОМТ 9.3: Оптимизация на живых данных

```
Непрерывная оптимизация:

Каждую неделю:
1. Проанализировать все сделки
2. Найти паттерны неудачных сделок:
   - Какие монеты?
   - Какие фазы рынка?
   - Какое время суток?
3. Обновить blacklist монет (если есть системные проигрыши)
4. Проверить качество моделей (не деградировали?)

Каждый месяц:
1. Переобучить модели на свежих данных
2. Обновить параметры:
   - TP/SL мультипликаторы (A/B тест разных значений)
   - Пороги EV
   - Количество позиций
3. Бэктест на последнем месяце
4. Если результаты хуже → откатиться к предыдущей версии

Каждый квартал:
1. Полный аудит стратегии
2. Сравнение с бенчмарками (Buy&Hold BTC, etc)
3. Анализ market regime changes
4. Возможные улучшения архитектуры

Метрики для отслеживания:
- Винрейт (rolling 50 trades)
- Sharpe ratio (rolling 30 days)
- Max DD
- EV accuracy (сколько монет из топ-10 действительно выросли)
- Model calibration (ECE)

Создай автоматический отчет:
python3 /home/user/jjhbn/GGG3/generate_monthly_report.py
```

---

# 📊 ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ

## Ключевые моменты

1. **Тестирование критически важно:**
   - Минимум 2 недели paper trading
   - Результаты должны быть близки к бэктесту
   - Не спешить на live без уверенности

2. **Начинать с малого:**
   - Первый месяц: $500-1000
   - Постепенно увеличивать при стабильных результатах
   - Не рисковать большими суммами сразу

3. **Мониторинг 24/7:**
   - Автоматические алерты
   - Ежедневная проверка метрик
   - Быстрая реакция на проблемы

4. **Консервативность:**
   - Лучше заработать 80% потенциала стабильно
   - Чем 120% с высоким риском краха
   - Stop-loss по просадке обязателен

5. **Непрерывное обучение:**
   - Онлайн обучение на новых данных
   - Адаптация к изменениям рынка
   - Регулярный рефакторинг и улучшения

## Ожидаемые результаты

**При винрейте 50%:**
- Месячный ROI: 6-8%
- Годовой ROI: 80-100%
- Max Drawdown: 12-15%
- Sharpe Ratio: 2.0-2.5

**При винрейте 55%:**
- Месячный ROI: 10-12%
- Годовой ROI: 120-150%
- Max Drawdown: 10-12%
- Sharpe Ratio: 2.5-3.0

## Риски

1. **Market regime shift** - рынок может измениться, модели устареют
2. **Overfitting** - хорошо на бэктесте, плохо на live
3. **Technical failures** - API errors, connectivity issues
4. **High correlation** - все монеты падают вместе
5. **Black swan events** - непредсказуемые события

**Защита:** Стоп-лоссы, диверсификация, консервативный риск, регулярный мониторинг

---

# 🎯 ЧЕКЛИСТ ГОТОВНОСТИ К LIVE

Перед запуском на реальные деньги убедись что:

- [ ] Бэктест показывает Win Rate > 50%
- [ ] Бэктест показывает Sharpe > 1.5
- [ ] Бэктест показывает Max DD < 20%
- [ ] Paper trading 2+ недели без критических проблем
- [ ] Paper trading винрейт близок к бэктесту (±5%)
- [ ] Все модели загружаются без ошибок
- [ ] EV-фильтр стабильно выбирает 10 монет
- [ ] Risk manager работает корректно
- [ ] Telegram алерты настроены
- [ ] Backup система работает
- [ ] API ключи настроены правильно
- [ ] Healthcheck endpoint работает
- [ ] Логирование полное и понятное
- [ ] Документация актуальна

Если все пункты ✅ → можно запускать!

---

**Удачи! 🚀**
