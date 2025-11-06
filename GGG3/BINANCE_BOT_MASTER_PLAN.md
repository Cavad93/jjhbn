# 🚀 ФИНАЛЬНЫЙ МАСТЕР-ПЛАН: МИГРАЦИЯ НА BINANCE FUTURES

**Версия:** 2.0 FINAL
**Дата:** 2025-11-06
**Статус:** Готов к реализации

---

## 📋 СОДЕРЖАНИЕ

1. [Обзор изменений](#обзор-изменений)
2. [Критические требования](#критические-требования)
3. [Архитектура и компоненты](#архитектура-и-компоненты)
4. [Этапы реализации](#этапы-реализации)
5. [Paper Trading (ОБЯЗАТЕЛЬНО)](#paper-trading-обязательно)
6. [Специальные режимы](#специальные-режимы)
7. [Чеклист готовности](#чеклист-готовности)

---

# 🎯 ОБЗОР ИЗМЕНЕНИЙ

## Было (PancakeSwap Prediction)

- **Рынок:** BSC Prediction Market (бинарные ставки UP/DOWN)
- **Таймфрейм:** 5 минут (1 раунд)
- **Позиций:** 1
- **Направление:** Только UP
- **Риск:** 1-3% капитала, без SL
- **Target:** Цена выше через 5 минут
- **Обучение:** Онлайн на результатах раундов

## Станет (Binance Futures 1x)

- **Рынок:** Binance Futures (маржинальная торговля)
- **Leverage:** 1x (без плеча, но можно SHORT)
- **Таймфрейм:** 4 часа (для ATR расчета)
- **Позиций:** 10 одновременно
- **Направление:** LONG и SHORT (bidirectional)
- **Риск:** 0.5-2.5% на позицию (Kelly criterion)
- **TP/SL:** TP = ±1.2×ATR, SL = ∓0.8×ATR (R/R = 1.5)
- **ATR:** Период 10 на 4h свечах
- **Target:** Достиг TP раньше SL
- **Обучение:** Онлайн на закрытых позициях
- **Break-even WR:** 40% (vs 50.76% в PancakeSwap)

## Сохраняется

✅ Базовая логика (4 сигнала: M, S, B, R)
✅ 4 ML эксперта (XGBoost, RF, ARF, NN)
✅ META нейросеть (CMA-ES обучение)
✅ 68 фич для экспертов
✅ Фазовая система (6 фаз рынка)
✅ Kelly criterion для риск-менеджмента
✅ Адаптивный порог (p_thr + δ)

## Добавляется

⚡ **EV Filter:** Скрининг ~400 монет, отбор топ-10 по Expected Value
⚡ **Bidirectional Trading:** LONG (p_up > threshold) и SHORT (p_up < 1-threshold)
⚡ **Snapshot Pattern:** Сохранение фич на момент входа (temporal consistency)
⚡ **Multi-timeframe:** 5m, 15m, 30m, 4h (adaptive selection)
⚡ **Portfolio Management:** Управление 10 позициями, агрессивная замена
⚡ **META SHADOW Mode:** Сначала в тени, переход в ACTIVE после доказательства
⚡ **Extended Blacklist:** Фильтрация мемкоинов, новых листингов, низколиквидных
⚡ **Special Modes:** Trailing stop, black swan protection, night mode, sector diversification
⚡ **Auto-reinvestment:** Полное увеличение капитала + 50% прибыли в резерв ежедневно

---

# ⚠️ КРИТИЧЕСКИЕ ТРЕБОВАНИЯ

## 1. TEMPORAL CONSISTENCY (КРИТИЧНО!)

### Проблема: Look-ahead Bias

❌ **НЕПРАВИЛЬНО:**
```python
def on_position_closed(position):
    # Рассчитываем фичи СЕЙЧАС (на момент закрытия)
    features = build_features()  # ❌ Содержат информацию о TP/SL!
    expert.train(features, outcome)
```

✅ **ПРАВИЛЬНО:**
```python
def open_position(symbol):
    # t0: МОМЕНТ ВХОДА
    features = build_features()  # Рассчитываем один раз
    predictions = get_all_predictions(features)

    # СОХРАНЯЕМ СНИМОК
    position = Position(
        entry_snapshot={
            'features': features.copy(),  # 68D
            'predictions': predictions,   # 5 экспертов
            'meta_context': context,      # 7 фич для META
            'timestamp': time.time()
        }
    )
    return position

def on_position_closed(position):
    # t1: МОМЕНТ ВЫХОДА
    outcome = 1 if position.exit_reason == 'TP' else 0

    # Используем СОХРАНЕННЫЕ фичи из t0
    snapshot = position.entry_snapshot
    expert.train(snapshot['features'], outcome)  # ✅
```

**Ключевой принцип:**
- **X (features)** = момент входа (t0)
- **y (outcome)** = момент выхода (t1)
- **НЕТ** расчета новых фич при закрытии!

---

## 2. BIDIRECTIONAL TRADING (LONG + SHORT)

### Логика направления

```python
def determine_direction(p_up: float, p_threshold: float) -> Optional[str]:
    """
    Определяет направление на основе вероятности

    Args:
        p_up: Вероятность роста от META (0-1)
        p_threshold: Адаптивный порог (p_thr + δ)

    Returns:
        'LONG'  - если p_up > p_threshold
        'SHORT' - если p_up < (1 - p_threshold)
        None    - неопределенность (skip)
    """

    if p_up > p_threshold:
        return 'LONG'   # Уверенность в росте

    elif p_up < (1 - p_threshold):
        return 'SHORT'  # Уверенность в падении

    else:
        return None     # Зона неопределенности
```

### TP/SL уровни

**LONG:**
```python
entry = 100.0
atr = 5.0

tp_price = entry + 1.2 * atr  # 106.0 (+6%)
sl_price = entry - 0.8 * atr  # 96.0 (-4%)

# Reward/Risk = 1.5:1
```

**SHORT:**
```python
entry = 100.0
atr = 5.0

tp_price = entry - 1.2 * atr  # 94.0 (-6% → прибыль)
sl_price = entry + 0.8 * atr  # 104.0 (+4% → убыток)

# Reward/Risk = 1.5:1 (тот же!)
```

### EV расчет для обоих направлений

```python
def calculate_ev_bidirectional(p_up: float) -> tuple[float, float]:
    """
    Рассчитывает Expected Value для LONG и SHORT

    EV = P(TP) × Reward - P(SL) × Risk

    При R/R = 1.5:
    - Reward = 1.5R
    - Risk = 1.0R
    """

    # EV для LONG
    ev_long = p_up * 1.5 - (1 - p_up) * 1.0
    ev_long = p_up * 2.5 - 1.0

    # EV для SHORT
    ev_short = (1 - p_up) * 1.5 - p_up * 1.0
    ev_short = 1.5 - p_up * 2.5

    return ev_long, ev_short

# Пример:
p_up = 0.62
ev_long, ev_short = calculate_ev_bidirectional(p_up)
# ev_long = +0.0500 (прибыльно)
# ev_short = -0.0500 (убыточно)
# Выбираем LONG
```

**Break-even вероятности:**
- EV > 0 требует p_up > 0.40 (для LONG)
- EV > 0 требует p_up < 0.60 (для SHORT)

---

## 3. ADAPTIVE THRESHOLD (p_thr + δ)

### Перенос текущей логики БЕЗ ИЗМЕНЕНИЙ

```python
def calculate_base_threshold(total_closed: int, recent_hour: int) -> float:
    """
    Базовый порог как в текущем боте

    Логика из bnbusdrt6.py (строки 26-28)
    """

    if total_closed < 500 or recent_hour == 0:
        return 0.51  # Консервативный порог в начале
    else:
        return 0.41  # После 500 сделок (чуть выше break-even 40%)

def adaptive_delta_protect(
    recent_wr: float,
    calib_error: float,
    n_trades: int
) -> float:
    """
    Динамическая δ на основе производительности

    ИЗ bnbusdrt6.py:348-383 (БЕЗ ИЗМЕНЕНИЙ!)
    """
    base_delta = 0.03

    # Масштаб по винрейту
    if recent_wr >= 0.58:
        wr_scale = 0.7      # Агрессивнее при отличном WR
    elif recent_wr >= 0.56:
        wr_scale = 0.85
    elif recent_wr >= 0.54:
        wr_scale = 1.0
    elif recent_wr >= 0.52:
        wr_scale = 1.2
    else:
        wr_scale = 1.4      # Осторожнее при плохом WR

    # Масштаб по калибровке
    if calib_error < 0.03:
        calib_scale = 0.9
    elif calib_error < 0.05:
        calib_scale = 1.0
    else:
        calib_scale = 1.15

    # Масштаб по активности
    activity_scale = 1.0 if n_trades >= 20 else 1.1

    delta_eff = base_delta * wr_scale * calib_scale * activity_scale
    return float(np.clip(delta_eff, 0.02, 0.08))

def get_adaptive_threshold(
    total_closed: int,
    recent_hour: int,
    recent_wr: float,
    calib_error: float
) -> float:
    """
    Финальный адаптивный порог

    ТОЧНО КАК В ТЕКУЩЕМ БОТЕ!
    """

    # Базовый порог
    p_thr = calculate_base_threshold(total_closed, recent_hour)

    # Адаптивная дельта
    delta = adaptive_delta_protect(recent_wr, calib_error, recent_hour)

    # Финальный порог
    p_threshold_final = p_thr + delta

    # Ограничиваем разумными пределами
    return np.clip(p_threshold_final, 0.45, 0.70)
```

---

## 4. META ARCHITECTURE

### Входные данные для META

META принимает **12 входов**, которые превращаются в **36D enriched features** внутри:

**5 predictions (от экспертов):**
1. p_xgb (XGBoost)
2. p_rf (RandomForest)
3. p_arf (AdaptiveRandomForest)
4. p_nn (Neural Network)
5. p_base (базовая логика - prob_up_down_at_time)

**7 context features (для attention module):**
1. vol_ratio (volatility ratio)
2. trend_macd (MACD trend)
3. jump_flag (price jump detection)
4. funding_sign (funding rate sign для futures)
5. book_imb (orderbook imbalance)
6. ofi_15s (order flow imbalance 15s)
7. basis_pct (basis percentage для futures)

### Snapshot для META

```python
entry_snapshot = {
    # Базовые фичи для экспертов (68D)
    'features_68d': features.copy(),

    # Предсказания экспертов (5)
    'predictions': {
        'p_xgb': float,
        'p_rf': float,
        'p_arf': float,
        'p_nn': float,
        'p_base': float  # from prob_up_down_at_time()
    },

    # Контекстные фичи для META (7)
    'meta_context': {
        'vol_ratio': float,
        'trend_macd': float,
        'jump_flag': float,
        'funding_sign': float,
        'book_imb': float,
        'ofi_15s': float,
        'basis_pct': float
    },

    # Финальное предсказание META
    'p_meta': float,

    # Временная метка
    'timestamp': float
}
```

### BASE LOGIC Modification

Текущая функция `prob_up_down_at_time()` использует 4 сигнала на одном таймфрейме.

**Для Binance нужно модифицировать:**
```python
def prob_up_down_at_time_multiTF(
    df_5m: pd.DataFrame,   # Для Momentum
    df_15m: pd.DataFrame,  # Для VWAP Slope, Bollinger
    df_30m: pd.DataFrame,  # Для Reversion
    timestamp: float,
    w_dyn: Optional[np.ndarray] = None,
    volAmp: float = 1.0,
    atr_norm: float = 1.0
) -> tuple[float, float, dict]:
    """
    Базовая логика с мультитаймфреймом

    Сигналы:
    - M (Momentum): на 5m
    - S (VWAP Slope): на 15m
    - B (Bollinger Breakout): на 15m
    - R (Reversion): на 30m

    Логика softmax + temperature: БЕЗ ИЗМЕНЕНИЙ
    """

    # Извлекаем сигналы с разных таймфреймов
    M_up, M_dn = calculate_momentum_signal(df_5m)
    S_up, S_dn = calculate_vwap_slope(df_15m)
    B_up, B_dn = calculate_bollinger_breakout(df_15m)
    R_up, R_dn = calculate_reversion_signal(df_30m)

    # Взвешенная комбинация (КАК В ТЕКУЩЕМ БОТЕ)
    Z_up = w_mom*M_up*volAmp + w_vwp*S_up + w_brk*B_up + w_rev*R_up
    Z_dn = w_mom*M_dn*volAmp + w_vwp*S_dn + w_brk*B_dn + w_rev*R_dn

    # Softmax with temperature
    P_up, P_dn = softmax2_with_temperature(Z_up, Z_dn, temperature=T)

    return P_up, P_dn, context
```

---

# 🏗️ АРХИТЕКТУРА И КОМПОНЕНТЫ

## Компоненты системы

```
┌─────────────────────────────────────────────────────────────────┐
│                    BINANCE TRADING BOT                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │  Binance API │    │  Data Feed   │    │  Portfolio   │     │
│  │   Client     │───▶│  (OHLCV,     │───▶│  Manager     │     │
│  │              │    │   4 TFs)     │    │  (10 pos)    │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌──────────────────────────────────────────────────────┐     │
│  │         FEATURE ENGINEERING (68D)                     │     │
│  │  - Multi-timeframe (5m, 15m, 30m, 4h)                │     │
│  │  - Momentum, VWAP, Bollinger, ATR, RSI, Volume       │     │
│  └──────────────────────────────────────────────────────┘     │
│         │                    │                    │            │
│         └────────────────────┼────────────────────┘            │
│                              ▼                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │              4 ML EXPERTS                             │     │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │     │
│  │  │ XGBoost │ │   RF    │ │  ARF    │ │   NN    │   │     │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │     │
│  │      │            │            │            │        │     │
│  │      └────────────┴────────────┴────────────┘        │     │
│  │                    │                                  │     │
│  │              [5 predictions]                          │     │
│  └──────────────────────────────────────────────────────┘     │
│                              │                                 │
│                              │                                 │
│         ┌────────────────────┼────────────────────┐           │
│         │                    │                    │           │
│         ▼                    ▼                    ▼           │
│  ┌──────────┐      ┌─────────────────┐    ┌──────────┐      │
│  │  BASE    │      │   META NEURAL   │    │ CONTEXT  │      │
│  │  LOGIC   │─────▶│    (CMA-ES)     │◀───│ (7 fts)  │      │
│  │ (4 sig)  │      │  SHADOW/ACTIVE  │    │          │      │
│  └──────────┘      └─────────────────┘    └──────────┘      │
│                              │                                 │
│                              ▼                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │            DECISION MAKING                            │     │
│  │  - Adaptive Threshold (p_thr + δ)                    │     │
│  │  - LONG vs SHORT vs SKIP                             │     │
│  │  - EV Filter (top-10 from ~400)                      │     │
│  └──────────────────────────────────────────────────────┘     │
│                              │                                 │
│                              ▼                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │         RISK MANAGEMENT                               │     │
│  │  - Kelly Criterion (0.5-2.5%)                        │     │
│  │  - Trailing Stop (+0.6×ATR)                          │     │
│  │  - Black Swan Protection                             │     │
│  │  - Night Mode (00:00-06:00 UTC)                      │     │
│  │  - Sector Diversification (max 3 per sector)         │     │
│  └──────────────────────────────────────────────────────┘     │
│                              │                                 │
│                              ▼                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │      EXECUTION & MONITORING                           │     │
│  │  - Market Orders (Entry)                             │     │
│  │  - TP/SL Orders                                       │     │
│  │  - Position Tracking                                  │     │
│  │  - Online Learning (TP=1, SL=0)                      │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Файловая структура

```
/home/user/jjhbn/GGG3/
│
├── binance_bot_main.py              # Главный файл бота
├── binance_config.py                # Конфигурация
│
├── binance_api/
│   ├── client.py                    # Binance API client
│   ├── data_feed.py                 # Real-time data feed
│   └── order_executor.py            # Исполнение ордеров
│
├── features/
│   ├── builder.py                   # Feature engineering (68D)
│   ├── multi_timeframe.py           # Multi-TF helpers
│   └── base_logic.py                # prob_up_down_at_time_multiTF
│
├── models/
│   ├── experts/                     # 4 эксперта
│   │   ├── xgb_expert.py
│   │   ├── rf_expert.py
│   │   ├── arf_expert.py
│   │   └── nn_expert.py
│   │
│   ├── meta_neural_cem.py           # META (existing, modified)
│   └── model_loader.py              # Загрузка моделей
│
├── portfolio/
│   ├── position.py                  # Position dataclass
│   ├── position_manager.py          # Управление позициями
│   └── snapshot_handler.py          # Temporal consistency
│
├── strategy/
│   ├── coin_selector.py             # EV filter, top-10 selection
│   ├── direction_logic.py           # LONG/SHORT decision
│   ├── threshold_adapter.py         # Adaptive threshold
│   └── kelly_sizer.py               # Kelly criterion
│
├── risk/
│   ├── risk_manager.py              # Risk limits
│   ├── trailing_stop.py             # Trailing stop logic
│   ├── black_swan.py                # Black swan protection
│   └── night_mode.py                # Night mode handler
│
├── training/
│   ├── online_trainer.py            # Online learning
│   ├── retrain_scheduler.py         # Retrain triggers
│   └── training_visualizer.py       # Live dashboard
│
├── backtesting/
│   ├── backtest_engine.py           # Backtesting
│   └── performance_metrics.py       # Metrics calculation
│
├── paper_trading/
│   ├── paper_exchange.py            # Paper trading simulator
│   ├── paper_bot.py                 # Paper trading bot
│   └── paper_dashboard.py           # Monitoring dashboard
│
├── data/
│   ├── historical/                  # Historical OHLCV
│   ├── models/                      # Saved models
│   ├── positions/                   # Position snapshots
│   └── trades/                      # Trade history
│
└── utils/
    ├── logger.py                    # Logging
    ├── telegram_notifier.py         # Telegram alerts
    └── helpers.py                   # Utility functions
```

---

# 🎬 ЭТАПЫ РЕАЛИЗАЦИИ

## ЭТАП 0: PAPER TRADING (ОБЯЗАТЕЛЬНО!) 🎯

**⚠️ КРИТИЧЕСКИ ВАЖНО: Бот ДОЛЖЕН начать с paper trading на $1000!**

### Цель
- Запустить бота в режиме **бумажной торговли** с виртуальным капиталом $1000
- Собрать статистику за **минимум 2-4 недели**
- Проверить стабильность алгоритмов БЕЗ риска реальных денег
- Сравнить результаты с бэктестом
- **ТОЛЬКО после успешного paper trading** переходить на реальные деньги

### Требования

```python
# binance_config.py

# PAPER TRADING MODE (обязательно включено!)
PAPER_TRADING_MODE = True  # ⚠️ НЕ МЕНЯТЬ до прохождения всех тестов!

# Стартовый капитал
INITIAL_CAPITAL_PAPER = 1000.0  # $1000

# Все остальные параметры как для LIVE
MAX_POSITIONS = 10
ATR_PERIOD = 10
TP_ATR_MULTIPLIER = 1.2
SL_ATR_MULTIPLIER = 0.8
MIN_RISK_PER_POSITION = 0.005  # 0.5%
MAX_RISK_PER_POSITION = 0.025  # 2.5%

# Binance API для получения реальных цен
BINANCE_API_KEY = "..."  # Для чтения данных (не торговли!)
BINANCE_SECRET_KEY = "..."  # Не используется в paper mode
```

### Промт 0.1: Paper Trading Engine

```
Создай модуль для paper trading: /home/user/jjhbn/GGG3/paper_trading/paper_exchange.py

Класс PaperExchange:
    """Симулятор биржи для бумажной торговли"""

    def __init__(self, initial_capital: float = 1000.0):
        self.balance = initial_capital  # Виртуальный баланс USDT
        self.positions = {}  # Открытые позиции
        self.orders = {}  # Ордера (TP/SL)
        self.trades_history = []  # История сделок

    def create_market_order(self, symbol: str, side: str, amount: float) -> dict:
        """
        Симулирует market order

        - Использует РЕАЛЬНУЮ цену с Binance API
        - Добавляет проскальзывание 0.05%
        - Списывает комиссию 0.1%
        - Обновляет баланс
        """
        # Получаем реальную цену
        real_price = get_binance_price(symbol)

        # Проскальзывание
        slippage = 0.0005
        if side == 'BUY':
            exec_price = real_price * (1 + slippage)
        else:
            exec_price = real_price * (1 - slippage)

        # Комиссия
        commission = 0.001
        cost = amount * exec_price
        total_cost = cost * (1 + commission)

        # Проверяем баланс
        if side == 'BUY' and total_cost > self.balance:
            raise InsufficientBalance(f"Need {total_cost}, have {self.balance}")

        # Исполняем
        if side == 'BUY':
            self.balance -= total_cost
        else:
            self.balance += cost * (1 - commission)

        return {
            'id': generate_order_id(),
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': exec_price,
            'cost': total_cost,
            'status': 'filled'
        }

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> dict:
        """Создает лимитный ордер (TP)"""
        order = {
            'id': generate_order_id(),
            'symbol': symbol,
            'side': side,
            'type': 'LIMIT',
            'amount': amount,
            'price': price,
            'status': 'open'
        }
        self.orders[order['id']] = order
        return order

    def create_stop_loss(self, symbol: str, side: str, amount: float, stop_price: float) -> dict:
        """Создает стоп-лосс ордер"""
        order = {
            'id': generate_order_id(),
            'symbol': symbol,
            'side': side,
            'type': 'STOP_LOSS',
            'amount': amount,
            'stop_price': stop_price,
            'status': 'open'
        }
        self.orders[order['id']] = order
        return order

    def check_orders(self):
        """
        Проверяет исполнение TP/SL ордеров

        - Получает текущие цены
        - Проверяет достижение TP/SL
        - Исполняет ордера
        - Обновляет позиции
        """
        for order_id, order in list(self.orders.items()):
            symbol = order['symbol']
            current_price = get_binance_price(symbol)

            executed = False

            if order['type'] == 'LIMIT':
                # TP order
                if order['side'] == 'SELL' and current_price >= order['price']:
                    executed = True
                elif order['side'] == 'BUY' and current_price <= order['price']:
                    executed = True

            elif order['type'] == 'STOP_LOSS':
                # SL order
                if order['side'] == 'SELL' and current_price <= order['stop_price']:
                    executed = True
                elif order['side'] == 'BUY' and current_price >= order['stop_price']:
                    executed = True

            if executed:
                # Исполняем ордер
                self.create_market_order(
                    symbol=order['symbol'],
                    side=order['side'],
                    amount=order['amount']
                )

                # Удаляем ордер
                del self.orders[order_id]

                print(f"[Paper] Order executed: {order['type']} {order['symbol']} @ {current_price}")

    def get_balance(self) -> float:
        """Возвращает текущий баланс"""
        return self.balance

    def get_equity(self) -> float:
        """
        Рассчитывает полный капитал (баланс + открытые позиции)
        """
        equity = self.balance

        for position in self.positions.values():
            current_price = get_binance_price(position['symbol'])

            if position['side'] == 'LONG':
                pnl = (current_price - position['entry_price']) * position['amount']
            else:  # SHORT
                pnl = (position['entry_price'] - current_price) * position['amount']

            equity += position['cost'] + pnl

        return equity

ВАЖНО:
1. Все цены берутся РЕАЛЬНО с Binance API
2. Проскальзывание и комиссии симулируются
3. TP/SL проверяются на реальных ценах (high/low)
4. Весь функционал идентичен реальной бирже
5. Сохранение состояния в JSON для восстановления
```

### Промт 0.2: Paper Trading Bot

```
Создай главный файл paper trading бота: /home/user/jjhbn/GGG3/paper_trading/paper_bot_main.py

Это ПОЛНАЯ копия реального бота, но использует PaperExchange вместо Binance.

Функционал:
1. Загрузка моделей (эксперты + META)
2. Основной торговый цикл:
   - Каждые 4 часа: обновление топ-10, управление позициями
   - Каждые 5 минут: проверка TP/SL
3. Все алгоритмы ИДЕНТИЧНЫ реальному боту:
   - EV filter
   - Adaptive threshold
   - LONG/SHORT logic
   - Snapshot pattern
   - Online learning
4. Логирование всех операций
5. Сохранение истории сделок в CSV

Запуск:
python3 /home/user/jjhbn/GGG3/paper_trading/paper_bot_main.py \
    --initial-capital 1000 \
    --max-positions 10

ВАЖНО:
- Все реально как в live режиме
- Только без отправки ордеров на биржу
- Сбор статистики для анализа
```

### Промт 0.3: Paper Trading Dashboard

```
Создай дашборд для мониторинга: /home/user/jjhbn/GGG3/paper_trading/paper_dashboard.py

Используй Plotly + Dash для интерактивного дашборда.

Разделы:

1. OVERVIEW
   - Текущий капитал ($1000 → $1050 → ...)
   - P&L сегодня/неделя/всего
   - Win Rate (rolling 50 trades)
   - Количество открытых позиций
   - Total trades

2. EQUITY CURVE
   - График капитала в реальном времени
   - Линия просадки (drawdown)
   - ATH marker

3. OPEN POSITIONS
   Таблица с колонками:
   - Symbol
   - Direction (LONG/SHORT)
   - Entry Price
   - Current Price
   - TP / SL
   - Unrealized P&L
   - Duration
   - EV при входе

4. TRADE HISTORY
   Таблица последних 50 сделок:
   - Symbol
   - Direction
   - Entry / Exit
   - Exit Reason (TP/SL/Timeout)
   - P&L ($)
   - P&L (%)
   - Duration
   - Outcome (✅/❌)

5. PERFORMANCE METRICS
   - Win Rate
   - Profit Factor
   - Sharpe Ratio
   - Sortino Ratio
   - Max Drawdown
   - Average Win / Average Loss
   - Average Trade Duration

6. TOP-10 COINS
   Текущий список с EV, p_up, направление

Запуск:
python3 /home/user/jjhbn/GGG3/paper_trading/paper_dashboard.py
# Доступно на http://localhost:8050

Auto-refresh каждые 10 секунд.
```

### Критерии успешного прохождения Paper Trading

**Минимум 2 недели paper trading!**

Для перехода на LIVE должны выполняться:

✅ **Win Rate ≥ 48%** (выше break-even 40%)
✅ **Sharpe Ratio ≥ 1.5**
✅ **Max Drawdown ≤ 15%**
✅ **Profit Factor ≥ 1.3**
✅ **Total Return > 0** за период
✅ **Стабильность:** нет аномальных провалов
✅ **Техническая стабильность:** нет критических ошибок
✅ **Comparison с бэктестом:** метрики схожи (±5-10%)

**Если хотя бы один критерий НЕ выполнен:**
→ Продолжить paper trading
→ Анализировать причины
→ Корректировать параметры
→ НЕ переходить на live!

---

## ЭТАП 1: Подготовка инфраструктуры

### Цель
Настроить окружение для работы с Binance API и создать базовые модули

### Промт 1.1: Установка зависимостей

```
Установи библиотеки для работы с Binance:

pip install python-binance ccxt pandas numpy ta-lib plotly dash

Создай файл requirements.txt:

python-binance>=1.0.19
ccxt>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
river>=0.20.0
torch>=2.0.0
cma>=3.3.0
plotly>=5.17.0
dash>=2.14.0
python-telegram-bot>=20.0
ta-lib>=0.4.28

Проверь установку:
python3 -c "import binance, ccxt, sklearn, xgboost, river, torch, cma, plotly, dash; print('✅ All libraries installed')"
```

### Промт 1.2: Binance API Client

```
Создай модуль для работы с Binance: /home/user/jjhbn/GGG3/binance_api/client.py

Класс BinanceClient:
    """
    Wrapper для Binance Futures API

    ВАЖНО: Futures (не Spot) для возможности SHORT
    Leverage: 1x (без плеча, но маржинальная торговля)
    """

    def __init__(self, api_key: str, secret_key: str, testnet: bool = True):
        """
        Args:
            testnet: True для testnet, False для live
        """
        from binance.client import Client

        if testnet:
            self.client = Client(api_key, secret_key, testnet=True)
            self.client.API_URL = 'https://testnet.binancefuture.com'
        else:
            self.client = Client(api_key, secret_key)

        # Устанавливаем leverage 1x для всех пар
        self.set_leverage_all(1)

    def set_leverage_all(self, leverage: int = 1):
        """Устанавливает кредитное плечо для всех пар"""
        # Для conservative торговли используем 1x
        pass

    def get_all_usdt_pairs(self, min_volume_24h: float = 1_000_000) -> List[str]:
        """
        Получает все торговые пары USDT

        Фильтры:
        - Только USDT пары (BTCUSDT, ETHUSDT, ...)
        - Только Futures
        - Минимальный объем 24h > $1M
        - Исключить стейблкоины (USDCUSDT, BUSDUSDT, ...)

        Returns:
            ~300-400 пар
        """
        exchange_info = self.client.futures_exchange_info()

        pairs = []
        for symbol_info in exchange_info['symbols']:
            symbol = symbol_info['symbol']

            # Фильтр 1: Только USDT пары
            if not symbol.endswith('USDT'):
                continue

            # Фильтр 2: Исключить стейблкоины
            if symbol in ['USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'DAIUSDT']:
                continue

            # Фильтр 3: Проверить объем
            ticker = self.client.futures_ticker(symbol=symbol)
            volume_24h = float(ticker['quoteVolume'])

            if volume_24h >= min_volume_24h:
                pairs.append(symbol)

        return pairs

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """
        Получает OHLCV данные

        Args:
            symbol: 'BTCUSDT'
            timeframe: '5m', '15m', '30m', '4h'
            limit: количество свечей

        Returns:
            DataFrame с колонками: timestamp, open, high, low, close, volume
        """
        klines = self.client.futures_klines(
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    def get_current_price(self, symbol: str) -> float:
        """Текущая цена"""
        ticker = self.client.futures_symbol_ticker(symbol=symbol)
        return float(ticker['price'])

    def get_orderbook(self, symbol: str, limit: int = 20) -> dict:
        """Стакан ордеров"""
        book = self.client.futures_order_book(symbol=symbol, limit=limit)
        return {
            'bids': [[float(p), float(q)] for p, q in book['bids']],
            'asks': [[float(p), float(q)] for p, q in book['asks']]
        }

    def create_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        """
        Market order

        Args:
            side: 'BUY' или 'SELL'
        """
        order = self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        return order

    def create_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        """TP limit order"""
        order = self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type='LIMIT',
            quantity=quantity,
            price=price,
            timeInForce='GTC'
        )
        return order

    def create_stop_market_order(self, symbol: str, side: str, quantity: float, stop_price: float) -> dict:
        """SL stop-market order"""
        order = self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type='STOP_MARKET',
            quantity=quantity,
            stopPrice=stop_price
        )
        return order

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Отмена ордера"""
        try:
            self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
            return True
        except Exception as e:
            print(f"Error canceling order: {e}")
            return False

    def get_balance(self, asset: str = 'USDT') -> float:
        """Баланс в Futures кошельке"""
        account = self.client.futures_account()
        for balance in account['assets']:
            if balance['asset'] == asset:
                return float(balance['availableBalance'])
        return 0.0

    def get_account_info(self) -> dict:
        """Информация об аккаунте"""
        return self.client.futures_account()

ВАЖНО:
1. Используй Futures API (не Spot)
2. Leverage = 1x
3. Обработка ошибок и retry логика
4. Rate limiting (не более 1200 req/min)
5. Логирование всех операций
```

### Промт 1.3: Position Manager

```
Создай менеджер позиций: /home/user/jjhbn/GGG3/portfolio/position_manager.py

PositionManager - управление 10 одновременными позициями с полным snapshot.

Класс Position (dataclass):
    symbol: str
    direction: str  # 'LONG' или 'SHORT'
    entry_time: datetime
    entry_price: float
    amount: float
    position_value: float

    tp_price: float
    sl_price: float
    atr_value: float

    entry_order_id: Optional[str]
    tp_order_id: Optional[str]
    sl_order_id: Optional[str]

    # ✅ КРИТИЧНО: Snapshot на момент входа
    entry_snapshot: Dict = field(default_factory=dict)
    # Структура:
    # {
    #   'features_68d': np.ndarray,
    #   'predictions': {'p_xgb', 'p_rf', 'p_arf', 'p_nn', 'p_base'},
    #   'meta_context': {7 features},
    #   'p_meta': float,
    #   'timestamp': float
    # }

    exit_time: Optional[datetime]
    exit_price: Optional[float]
    exit_reason: Optional[str]  # 'TP', 'SL', 'TIMEOUT', 'MANUAL'
    pnl: float
    pnl_pct: float

    status: str  # 'OPEN', 'CLOSED'

Класс PositionManager:
    """Управление портфелем из 10 позиций"""

    def __init__(self, max_positions: int = 10):
        self.positions: Dict[str, Position] = {}
        self.max_positions = max_positions
        self.closed_positions: List[Position] = []

    def can_open_position(self, symbol: str) -> bool:
        """
        Проверяет можно ли открыть позицию

        Правила:
        - Максимум 10 позиций
        - Не более 1 позиции на монету
        """
        if len(self.positions) >= self.max_positions:
            return False

        if symbol in self.positions:
            return False

        return True

    def add_position(self, position: Position):
        """Добавляет открытую позицию"""
        self.positions[position.symbol] = position
        self.save_to_file()

    def close_position(self, symbol: str, exit_price: float, exit_reason: str):
        """Закрывает позицию"""
        if symbol not in self.positions:
            raise ValueError(f"Position {symbol} not found")

        position = self.positions[symbol]
        position.exit_time = datetime.now()
        position.exit_price = exit_price
        position.exit_reason = exit_reason
        position.status = 'CLOSED'

        # Рассчитываем PnL
        if position.direction == 'LONG':
            position.pnl = (exit_price - position.entry_price) * position.amount
            position.pnl_pct = (exit_price / position.entry_price - 1) * 100
        else:  # SHORT
            position.pnl = (position.entry_price - exit_price) * position.amount
            position.pnl_pct = (1 - exit_price / position.entry_price) * 100

        # Перемещаем в закрытые
        self.closed_positions.append(position)
        del self.positions[symbol]

        self.save_to_file()

    def get_all_open(self) -> List[Position]:
        """Все открытые позиции"""
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        """Получить позицию по символу"""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Проверяет есть ли открытая позиция"""
        return symbol in self.positions

    def get_total_exposure(self) -> float:
        """Общая стоимость всех позиций"""
        return sum(p.position_value for p in self.positions.values())

    def get_pnl_summary(self) -> dict:
        """Статистика по PnL"""
        if not self.closed_positions:
            return {'total_pnl': 0, 'win_rate': 0, 'total_trades': 0}

        total_pnl = sum(p.pnl for p in self.closed_positions)
        wins = sum(1 for p in self.closed_positions if p.pnl > 0)
        total = len(self.closed_positions)
        win_rate = wins / total if total > 0 else 0

        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'total_trades': total,
            'wins': wins,
            'losses': total - wins
        }

    def save_to_file(self, path: str = '/home/user/jjhbn/GGG3/data/positions/positions.json'):
        """Сохраняет позиции в JSON"""
        data = {
            'open_positions': [p.to_dict() for p in self.positions.values()],
            'closed_positions': [p.to_dict() for p in self.closed_positions[-100:]]  # Последние 100
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, path: str = '/home/user/jjhbn/GGG3/data/positions/positions.json'):
        """Загружает позиции из JSON"""
        if not os.path.exists(path):
            return

        with open(path, 'r') as f:
            data = json.load(f)

        # Восстанавливаем открытые позиции
        for p_dict in data['open_positions']:
            position = Position.from_dict(p_dict)
            self.positions[position.symbol] = position

        # Восстанавливаем закрытые
        for p_dict in data['closed_positions']:
            position = Position.from_dict(p_dict)
            self.closed_positions.append(position)

ВАЖНО:
- Персистентность через JSON
- Snapshot сохраняется в каждой позиции
- При восстановлении после перезапуска все данные доступны
```

### Промт 1.4: Конфигурация

```
Создай конфигурацию: /home/user/jjhbn/GGG3/binance_config.py

import os
import numpy as np

# ═══════════════════════════════════════════════════════════
# PAPER TRADING (ОБЯЗАТЕЛЬНО для начала!)
# ═══════════════════════════════════════════════════════════

PAPER_TRADING_MODE = True  # ⚠️ ВСЕГДА True до прохождения тестов!
INITIAL_CAPITAL_PAPER = 1000.0  # $1000 стартовый капитал

# ═══════════════════════════════════════════════════════════
# API CREDENTIALS
# ═══════════════════════════════════════════════════════════

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
USE_TESTNET = os.getenv("USE_TESTNET", "1") == "1"

# ═══════════════════════════════════════════════════════════
# TRADING PARAMETERS
# ═══════════════════════════════════════════════════════════

# Портфель
MAX_POSITIONS = 10
MIN_POSITION_SIZE_USDT = 15.0
MAX_POSITION_SIZE_USDT = 1000.0

# ATR
ATR_PERIOD = 10
ATR_TIMEFRAME = "4h"
TP_ATR_MULTIPLIER = 1.2
SL_ATR_MULTIPLIER = 0.8

# Risk Management
MIN_RISK_PER_POSITION = 0.005  # 0.5%
MAX_RISK_PER_POSITION = 0.025  # 2.5%
DEFAULT_RISK = 0.009  # 0.9%

# Timeouts
MAX_POSITION_DURATION_HOURS = 168  # 7 дней
POSITION_CHECK_INTERVAL_SEC = 300  # 5 минут

# ═══════════════════════════════════════════════════════════
# EV FILTER & COIN SELECTION
# ═══════════════════════════════════════════════════════════

MIN_EV_THRESHOLD = 0.02  # Минимальный EV для входа
TOP_N_COINS = 10

# Ликвидность
MIN_VOLUME_24H_USDT = 5_000_000  # $5M
MAX_SPREAD_PCT = 0.002  # 0.2%
MIN_ATR_PCT = 0.02  # 2%
MAX_ATR_PCT = 0.15  # 15%

# Blacklist
BLACKLIST_MARKET_CAP_MIN = 50_000_000  # $50M минимум
BLACKLIST_LISTING_AGE_DAYS = 30  # Не торговать листинги < 30 дней
BLACKLIST_VOLUME_VS_BTC = 0.001  # Объем должен быть > 0.1% от BTC

# ═══════════════════════════════════════════════════════════
# MULTI-TIMEFRAME
# ═══════════════════════════════════════════════════════════

CANDLE_TIMEFRAMES = {
    "fast": "5m",     # Momentum, RSI
    "medium": "15m",  # VWAP, Bollinger
    "slow": "30m",    # Reversion
    "atr": "4h"       # ATR, Phases
}
CANDLE_LIMIT = 500

# ═══════════════════════════════════════════════════════════
# ADAPTIVE THRESHOLD
# ═══════════════════════════════════════════════════════════

# Базовый порог
BASE_THRESHOLD_INITIAL = 0.51  # До 500 сделок
BASE_THRESHOLD_TRAINED = 0.41  # После 500 сделок

# Delta параметры
DELTA_BASE = 0.03
DELTA_MIN = 0.02
DELTA_MAX = 0.08

# Финальные пределы
THRESHOLD_MIN = 0.45
THRESHOLD_MAX = 0.70

# ═══════════════════════════════════════════════════════════
# META NEURAL NETWORK
# ═══════════════════════════════════════════════════════════

META_MODE = "SHADOW"  # "SHADOW" или "ACTIVE"

# SHADOW → ACTIVE transition criteria
META_ACTIVATION_MIN_TRADES = 500
META_ACTIVATION_MIN_WR_IMPROVEMENT = 0.02  # +2% над средним экспертов
META_ACTIVATION_CONFIDENCE_LEVEL = 0.95  # 95% статистическая значимость

# ACTIVE → SHADOW revert criteria
META_REVERT_WR_THRESHOLD = 0.40  # Если WR < 40% → вернуться в SHADOW
META_REVERT_LOOKBACK = 100  # За последние 100 сделок

# ═══════════════════════════════════════════════════════════
# SPECIAL MODES
# ═══════════════════════════════════════════════════════════

# Trailing Stop
TRAILING_STOP_ENABLED = True
TRAILING_STOP_ACTIVATION = 0.6  # Активируется при +0.6×ATR (50% к TP)
TRAILING_STOP_DISTANCE = 0.3  # Отступ 0.3×ATR от максимума

# Black Swan Protection
BLACK_SWAN_ENABLED = True
BLACK_SWAN_BTC_DROP_THRESHOLD = -0.15  # BTC -15% за 4 часа
BLACK_SWAN_TIMEFRAME = "4h"
BLACK_SWAN_RESUME_STABLE_HOURS = 8  # Возобновить через 8ч стабильности

# Night Mode
NIGHT_MODE_ENABLED = True
NIGHT_MODE_START_UTC = 0  # 00:00 UTC
NIGHT_MODE_END_UTC = 6    # 06:00 UTC

# Sector Diversification
SECTOR_DIVERSIFICATION_ENABLED = True
MAX_COINS_PER_SECTOR = 3

# ═══════════════════════════════════════════════════════════
# AUTO-REINVESTMENT
# ═══════════════════════════════════════════════════════════

AUTO_REINVEST_ENABLED = True
REINVEST_FULL_CAPITAL_INCREASE = True  # Реинвестируем рост капитала
REINVEST_PROFIT_TO_RESERVE_PCT = 0.50  # 50% прибыли в резерв ежедневно

# ═══════════════════════════════════════════════════════════
# ONLINE LEARNING
# ═══════════════════════════════════════════════════════════

RETRAIN_EXPERTS_EVERY_N_TRADES = 100
RETRAIN_META_EVERY_N_TRADES = 500

# ═══════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════

BASE_DIR = "/home/user/jjhbn/GGG3"
DATA_DIR = f"{BASE_DIR}/data"
MODELS_DIR = f"{BASE_DIR}/data/models"
POSITIONS_FILE = f"{DATA_DIR}/positions/positions.json"
TRADES_CSV = f"{DATA_DIR}/trades/trades.csv"
LOGS_DIR = f"{BASE_DIR}/logs"

# ═══════════════════════════════════════════════════════════
# TELEGRAM ALERTS (optional)
# ═══════════════════════════════════════════════════════════

TELEGRAM_ENABLED = False
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ═══════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════

def validate_config():
    """Проверка корректности параметров"""

    assert MAX_POSITIONS > 0, "MAX_POSITIONS must be > 0"
    assert 0 < MIN_RISK_PER_POSITION < MAX_RISK_PER_POSITION < 1, "Risk percentages invalid"
    assert ATR_PERIOD > 0, "ATR_PERIOD must be > 0"
    assert TP_ATR_MULTIPLIER > SL_ATR_MULTIPLIER > 0, "TP/SL multipliers invalid"
    assert MIN_EV_THRESHOLD >= 0, "MIN_EV_THRESHOLD must be >= 0"
    assert TOP_N_COINS <= MAX_POSITIONS, "TOP_N_COINS > MAX_POSITIONS"
    assert 0 < THRESHOLD_MIN < THRESHOLD_MAX < 1, "Threshold limits invalid"

    # Paper trading проверка
    if PAPER_TRADING_MODE:
        print("⚠️  PAPER TRADING MODE ENABLED")
        print(f"   Initial capital: ${INITIAL_CAPITAL_PAPER}")
        print("   NO REAL MONEY AT RISK")

    print("✅ Configuration validated")

if __name__ == "__main__":
    validate_config()
```

---

## ЭТАП 2: Сбор исторических данных

### Цель
Собрать OHLCV данные для ~400 монет за последние 3 месяца

### Промт 2.1: Модуль сбора данных

```
Создай сборщик данных: /home/user/jjhbn/GGG3/data_collection/collect_binance_data.py

import time
from binance_api.client import BinanceClient
import pandas as pd
import os
from tqdm import tqdm

def get_all_tradeable_pairs(client: BinanceClient) -> List[str]:
    """
    Получает список всех торгуемых пар с фильтрами

    Фильтры:
    - Только USDT пары
    - Объем 24h > $1M
    - Исключить стейблкоины
    - Исключить новые листинги (<30 дней)

    Returns:
        ~300-400 пар
    """
    pairs = client.get_all_usdt_pairs(min_volume_24h=1_000_000)

    # Дополнительные фильтры...
    # (проверка листинга, blacklist, etc)

    return pairs

def download_ohlcv_data(
    client: BinanceClient,
    symbol: str,
    timeframes: List[str],
    output_dir: str
) -> bool:
    """
    Скачивает OHLCV для всех таймфреймов

    Args:
        symbol: 'BTCUSDT'
        timeframes: ['5m', '15m', '30m', '4h']
        output_dir: '/home/user/jjhbn/GGG3/data/historical'

    Saves:
        {output_dir}/{symbol}_5m.parquet
        {output_dir}/{symbol}_15m.parquet
        {output_dir}/{symbol}_30m.parquet
        {output_dir}/{symbol}_4h.parquet
    """

    os.makedirs(output_dir, exist_ok=True)

    for tf in timeframes:
        try:
            # Скачиваем максимум свечей (500 для 4h = ~83 дня)
            df = client.get_ohlcv(symbol, tf, limit=500)

            # Сохраняем в parquet (быстрый формат)
            path = f"{output_dir}/{symbol}_{tf}.parquet"
            df.to_parquet(path, index=False)

            print(f"✓ {symbol} {tf}: {len(df)} candles")

            time.sleep(0.1)  # Rate limiting

        except Exception as e:
            print(f"✗ {symbol} {tf}: {e}")
            return False

    return True

def collect_all_data(output_dir: str = '/home/user/jjhbn/GGG3/data/historical'):
    """
    Главная функция сбора данных
    """

    client = BinanceClient(api_key="", secret_key="", testnet=False)

    # Получаем список пар
    print("Fetching tradeable pairs...")
    pairs = get_all_tradeable_pairs(client)
    print(f"Found {len(pairs)} pairs")

    # Таймфреймы
    timeframes = ['5m', '15m', '30m', '4h']

    # Скачиваем данные
    print(f"\nDownloading data for {len(pairs)} pairs...")

    success_count = 0
    for symbol in tqdm(pairs):
        if download_ohlcv_data(client, symbol, timeframes, output_dir):
            success_count += 1

        time.sleep(0.2)  # Rate limiting

    print(f"\n✅ Downloaded: {success_count}/{len(pairs)} pairs")

    # Создаем индекс
    create_data_index(output_dir)

def create_data_index(data_dir: str):
    """
    Создает индекс data_index.json

    {
      "BTCUSDT": {
        "5m": "historical/BTCUSDT_5m.parquet",
        "15m": "historical/BTCUSDT_15m.parquet",
        "30m": "historical/BTCUSDT_30m.parquet",
        "4h": "historical/BTCUSDT_4h.parquet",
        "last_update": "2025-11-06T12:00:00",
        "rows_4h": 500
      },
      ...
    }
    """
    import glob
    import json

    index = {}

    for file_path in glob.glob(f"{data_dir}/*.parquet"):
        filename = os.path.basename(file_path)
        # BTCUSDT_5m.parquet -> BTCUSDT, 5m
        symbol, tf = filename.replace('.parquet', '').rsplit('_', 1)

        if symbol not in index:
            index[symbol] = {}

        df = pd.read_parquet(file_path)
        index[symbol][tf] = file_path
        index[symbol]['rows_' + tf] = len(df)
        index[symbol]['last_update'] = df['timestamp'].iloc[-1].isoformat()

    # Сохраняем индекс
    index_path = f"{data_dir}/data_index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"✅ Index created: {index_path}")
    print(f"   Total symbols: {len(index)}")

if __name__ == "__main__":
    collect_all_data()
```

Запусти:
```bash
python3 /home/user/jjhbn/GGG3/data_collection/collect_binance_data.py
```

Ожидаемый результат:
- ~400 монет × 4 таймфрейма = 1600 файлов
- Размер: ~500-1000 MB
- Время: 2-4 часа
```

---

## ЭТАП 3: Адаптация фич и target

### Цель
Модифицировать фичи для multi-timeframe и создать TP/SL target

### Промт 3.1: Multi-timeframe Features

```
Создай feature builder: /home/user/jjhbn/GGG3/features/builder.py

На основе текущих фич (prepare_training_features.py), но с multi-TF:

class BinanceFeatureBuilder:
    """
    Построение 68 фич для Binance Futures

    Входы: 4 таймфрейма (5m, 15m, 30m, 4h)
    Выход: 68D вектор (те же фичи что в текущем боте)
    """

    def build_extended_features(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_30m: pd.DataFrame,
        df_4h: pd.DataFrame,
        current_time: pd.Timestamp
    ) -> np.ndarray:
        """
        Рассчитывает 68 фич

        Распределение по таймфреймам:

        5m (fast):
        - Momentum indicators (RSI, MACD, Stoch)
        - Price changes (1/3/5 candles back)

        15m (medium):
        - VWAP slope
        - Bollinger bands
        - Volume indicators

        30m (slow):
        - Reversion signals
        - EMA crossovers
        - Trend indicators

        4h (phases):
        - ATR
        - Market phases (0-5)
        - Regime context

        Returns:
            np.ndarray (68,)
        """

        features = []

        # 5m features (14 фич)
        features.extend(self._calc_momentum_features(df_5m))

        # 15m features (22 фичи)
        features.extend(self._calc_vwap_bollinger_features(df_15m))

        # 30m features (21 фича)
        features.extend(self._calc_reversion_trend_features(df_30m))

        # 4h features (11 фич)
        features.extend(self._calc_atr_phase_features(df_4h))

        return np.array(features)

    def _calc_momentum_features(self, df: pd.DataFrame) -> List[float]:
        """Momentum на 5m (14 фич)"""
        # RSI
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        # MACD
        macd = ta.trend.MACD(df['close'])

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])

        # Price changes
        returns_1 = df['close'].pct_change(1)
        returns_3 = df['close'].pct_change(3)
        returns_5 = df['close'].pct_change(5)

        return [
            rsi.iloc[-1],
            (rsi.iloc[-1] - rsi.iloc[-5]) / 5,  # RSI slope
            macd.macd().iloc[-1],
            macd.macd_signal().iloc[-1],
            macd.macd_diff().iloc[-1],
            stoch.stoch().iloc[-1],
            stoch.stoch_signal().iloc[-1],
            returns_1.iloc[-1],
            returns_3.iloc[-1],
            returns_5.iloc[-1],
            df['volume'].iloc[-1] / df['volume'].iloc[-20:].mean(),
            # ... еще 3 фичи
        ]

    # Аналогично для других методов...

ВАЖНО:
- 68 фич как в текущем боте (совместимость с моделями)
- Распределение по таймфреймам (адаптивное)
- Использование ta-lib для индикаторов
```

### Промт 3.2: TP/SL Target для LONG и SHORT

```
Создай модуль target: /home/user/jjhbn/GGG3/features/target_calculator.py

def calculate_tp_sl_outcome_bidirectional(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    atr_value: float,
    direction: str,  # 'LONG' или 'SHORT'
    tp_mult: float = 1.2,
    sl_mult: float = 0.8,
    max_bars: int = 100
) -> Optional[dict]:
    """
    Определяет исход для LONG или SHORT позиции

    Для LONG:
    - TP = entry + 1.2×ATR
    - SL = entry - 0.8×ATR
    - Outcome = 1 если TP достигнут раньше SL
    - Outcome = 0 если SL достигнут раньше TP

    Для SHORT:
    - TP = entry - 1.2×ATR
    - SL = entry + 0.8×ATR
    - Outcome = 1 если TP достигнут раньше SL
    - Outcome = 0 если SL достигнут раньше TP

    Returns:
        {
          'outcome': 1 или 0,
          'direction': 'LONG' или 'SHORT',
          'bars_held': int,
          'exit_price': float,
          'pnl_pct': float
        }
        или None если timeout
    """

    if direction == 'LONG':
        tp_price = entry_price + tp_mult * atr_value
        sl_price = entry_price - sl_mult * atr_value

        for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            # Проверяем TP (достигнут раньше?)
            if high >= tp_price:
                return {
                    'outcome': 1,
                    'direction': 'LONG',
                    'bars_held': i - entry_idx,
                    'exit_price': tp_price,
                    'pnl_pct': (tp_price - entry_price) / entry_price * 100
                }

            # Проверяем SL
            if low <= sl_price:
                return {
                    'outcome': 0,
                    'direction': 'LONG',
                    'bars_held': i - entry_idx,
                    'exit_price': sl_price,
                    'pnl_pct': (sl_price - entry_price) / entry_price * 100
                }

    elif direction == 'SHORT':
        tp_price = entry_price - tp_mult * atr_value
        sl_price = entry_price + sl_mult * atr_value

        for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            # Проверяем TP (цена падает)
            if low <= tp_price:
                return {
                    'outcome': 1,
                    'direction': 'SHORT',
                    'bars_held': i - entry_idx,
                    'exit_price': tp_price,
                    'pnl_pct': (entry_price - tp_price) / entry_price * 100
                }

            # Проверяем SL (цена растет)
            if high >= sl_price:
                return {
                    'outcome': 0,
                    'direction': 'SHORT',
                    'bars_held': i - entry_idx,
                    'exit_price': sl_price,
                    'pnl_pct': (entry_price - sl_price) / entry_price * 100
                }

    return None  # Timeout

def prepare_training_dataset(data_dir: str, output_file: str):
    """
    Создает датасет для обучения

    Для каждой монеты и каждой свечи (4h):
    1. Рассчитать 68 фич (multi-TF)
    2. Рассчитать ATR
    3. Определить фазу рынка
    4. Для ОБОИХ направлений (LONG и SHORT):
       - Определить outcome (TP/SL)
       - Если outcome не None: добавить в датасет

    Результат:
    - DataFrame с колонками:
      * features (68D)
      * outcome (0 или 1)
      * direction ('LONG' или 'SHORT')
      * phase (0-5)
      * symbol
      * timestamp
      * bars_held
      * pnl_pct

    Сохраняет в parquet:
    - binance_dataset_train.parquet (первые 80% времени)
    - binance_dataset_test.parquet (последние 20%)
    """
    pass

ВАЖНО:
- Поддержка LONG и SHORT
- Симметричный R/R (1.5:1)
- Для каждой свечи проверяем ОБА направления
- Датасет будет содержать примеры и LONG и SHORT
```

---

## ЭТАП 4: EV Filter для LONG/SHORT

### Цель
Создать отбор топ-10 монет с учетом обоих направлений

### Промт 4.1: Coin Selector с Bidirectional EV

```
Создай селектор монет: /home/user/jjhbn/GGG3/strategy/coin_selector.py

class CoinSelector:
    """
    Отбирает топ-10 возможностей (LONG или SHORT)
    """

    def calculate_ev_bidirectional(self, p_up: float) -> tuple[float, float]:
        """
        Рассчитывает EV для LONG и SHORT

        EV = P(TP) × Reward - P(SL) × Risk

        При R/R = 1.5:
        - Reward = 1.5R
        - Risk = 1.0R

        EV_long = p_up × 1.5 - (1 - p_up) × 1.0
                = p_up × 2.5 - 1.0

        EV_short = (1 - p_up) × 1.5 - p_up × 1.0
                 = 1.5 - p_up × 2.5

        Break-even:
        - EV_long > 0 требует p_up > 0.40
        - EV_short > 0 требует p_up < 0.60
        """

        ev_long = p_up * 2.5 - 1.0
        ev_short = 1.5 - p_up * 2.5

        return ev_long, ev_short

    def apply_filters(self, symbol: str) -> bool:
        """
        Фильтры ликвидности и blacklist

        1. Объем 24h > $5M
        2. Спред < 0.2%
        3. ATR в пределах 2-15%
        4. Market cap > $50M
        5. Листинг > 30 дней назад
        6. Объем > 0.1% от BTC
        7. Не в blacklist (мемкоины, скамы)
        8. Sector limit (max 3 per sector)
        """
        # Реализация фильтров...
        return True

    def select_top_coins(
        self,
        all_pairs: List[str],
        experts: dict,
        meta: MetaNeuralCEM,
        base_model,
        top_n: int = 10
    ) -> List[dict]:
        """
        Отбирает топ-N возможностей

        Процесс:
        1. Для каждой монеты:
           - Применить фильтры
           - Получить 68 фич (multi-TF)
           - Получить 5 predictions (experts + base)
           - Получить p_meta
           - Рассчитать EV_long и EV_short
           - Выбрать направление с max(EV) > threshold
        2. Сортировать по abs(EV)
        3. Вернуть топ-N

        Returns:
            [
              {
                'symbol': 'BTCUSDT',
                'direction': 'LONG',
                'p_up': 0.67,
                'ev': 0.0675,
                'atr': 0.032,
                'phase': 3
              },
              {
                'symbol': 'ETHUSDT',
                'direction': 'SHORT',
                'p_up': 0.35,
                'ev': 0.0625,
                'atr': 0.041,
                'phase': 2
              },
              ...
            ]
        """

        opportunities = []

        for symbol in all_pairs:

            # Фильтры
            if not self.apply_filters(symbol):
                continue

            # Получаем данные (4 таймфрейма)
            df_5m = get_ohlcv(symbol, '5m', 200)
            df_15m = get_ohlcv(symbol, '15m', 200)
            df_30m = get_ohlcv(symbol, '30m', 200)
            df_4h = get_ohlcv(symbol, '4h', 100)

            # Фичи
            features = feature_builder.build_extended_features(
                df_5m, df_15m, df_30m, df_4h
            )

            # Контекст
            ctx = calculate_regime_context(df_4h)
            phase = ctx['phase']

            # Predictions экспертов
            p_xgb = experts['xgb'].proba_up(features, reg_ctx={'phase': phase})[0]
            p_rf = experts['rf'].proba_up(features, reg_ctx={'phase': phase})[0]
            p_arf = experts['arf'].proba_up(features, reg_ctx={'phase': phase})[0]
            p_nn = experts['nn'].proba_up(features, reg_ctx={'phase': phase})[0]

            # BASE logic
            p_base = base_model.prob_up_down_at_time_multiTF(
                df_5m, df_15m, df_30m, current_time=df_4h['timestamp'].iloc[-1]
            )[0]

            # META prediction
            if META_MODE == 'ACTIVE':
                p_meta = meta.predict(p_xgb, p_rf, p_arf, p_nn, p_base, phase)
            else:
                # SHADOW mode: используем среднее экспертов
                p_meta = np.mean([p_xgb, p_rf, p_arf, p_nn, p_base])

            # EV для обоих направлений
            ev_long, ev_short = self.calculate_ev_bidirectional(p_meta)

            # Выбираем лучшее направление
            if ev_long > MIN_EV_THRESHOLD and ev_long > ev_short:
                opportunities.append({
                    'symbol': symbol,
                    'direction': 'LONG',
                    'p_up': p_meta,
                    'ev': ev_long,
                    'atr': calculate_atr(df_4h, 10).iloc[-1] / df_4h['close'].iloc[-1],
                    'phase': phase,
                    'predictions': {
                        'p_xgb': p_xgb,
                        'p_rf': p_rf,
                        'p_arf': p_arf,
                        'p_nn': p_nn,
                        'p_base': p_base
                    },
                    'context': ctx
                })

            elif ev_short > MIN_EV_THRESHOLD and ev_short > ev_long:
                opportunities.append({
                    'symbol': symbol,
                    'direction': 'SHORT',
                    'p_up': p_meta,
                    'ev': ev_short,
                    'atr': calculate_atr(df_4h, 10).iloc[-1] / df_4h['close'].iloc[-1],
                    'phase': phase,
                    'predictions': {
                        'p_xgb': p_xgb,
                        'p_rf': p_rf,
                        'p_arf': p_arf,
                        'p_nn': p_nn,
                        'p_base': p_base
                    },
                    'context': ctx
                })

        # Сортируем по EV (descending)
        opportunities.sort(key=lambda x: x['ev'], reverse=True)

        # Применяем sector diversification
        opportunities = self._apply_sector_limits(opportunities)

        return opportunities[:top_n]

ВАЖНО:
- Скрининг ~400 монет
- Оба направления (LONG и SHORT)
- Выбор направления с максимальным EV
- Полный snapshot для каждой возможности
```

---

## ЭТАП 5-9: Обучение, Portfolio Management, Backtesting, Paper Trading, Live

[Аналогично оригинальному плану с учетом всех изменений...]

---

# 🎮 СПЕЦИАЛЬНЫЕ РЕЖИМЫ

## 1. Trailing Stop

```python
# /home/user/jjhbn/GGG3/risk/trailing_stop.py

class TrailingStopManager:
    """
    Trailing stop logic

    Активация: при достижении +0.6×ATR (50% к TP)
    Отступ: 0.3×ATR от максимума
    """

    def check_trailing_stop(self, position: Position, current_price: float) -> Optional[float]:
        """
        Проверяет и обновляет trailing stop

        Returns:
            Новый SL уровень или None
        """

        if not TRAILING_STOP_ENABLED:
            return None

        # Рассчитываем прогресс к TP
        if position.direction == 'LONG':
            progress = (current_price - position.entry_price) / (position.tp_price - position.entry_price)

            if progress >= TRAILING_STOP_ACTIVATION:
                # Активирован! Двигаем SL
                new_sl = current_price - TRAILING_STOP_DISTANCE * position.atr_value
                if new_sl > position.sl_price:
                    return new_sl

        elif position.direction == 'SHORT':
            progress = (position.entry_price - current_price) / (position.entry_price - position.tp_price)

            if progress >= TRAILING_STOP_ACTIVATION:
                new_sl = current_price + TRAILING_STOP_DISTANCE * position.atr_value
                if new_sl < position.sl_price:
                    return new_sl

        return None
```

## 2. Black Swan Protection

```python
# /home/user/jjhbn/GGG3/risk/black_swan.py

class BlackSwanProtection:
    """
    Защита от black swan events

    Триггер: BTC -15% за 4 часа
    Действие: Остановить новые входы
    Возобновление: Через 8 часов стабильности
    """

    def check_black_swan_event(self) -> bool:
        """
        Проверяет black swan event

        Returns:
            True если событие активно
        """

        if not BLACK_SWAN_ENABLED:
            return False

        # Получаем BTC данные за последние 4 часа
        df_btc = get_ohlcv('BTCUSDT', '4h', 2)

        if len(df_btc) < 2:
            return False

        # Рассчитываем изменение
        price_change = (df_btc['close'].iloc[-1] - df_btc['close'].iloc[-2]) / df_btc['close'].iloc[-2]

        if price_change <= BLACK_SWAN_BTC_DROP_THRESHOLD:
            print(f"⚠️ BLACK SWAN EVENT: BTC {price_change*100:.1f}% in 4h")
            print("   Stopping new entries...")
            return True

        return False
```

## 3. Night Mode

```python
# /home/user/jjhbn/GGG3/risk/night_mode.py

class NightModeManager:
    """
    Night mode: no new positions 00:00-06:00 UTC

    Причина: Низкая ликвидность, высокие спреды
    """

    def is_night_mode_active(self) -> bool:
        """
        Проверяет активен ли night mode

        Returns:
            True если 00:00-06:00 UTC
        """

        if not NIGHT_MODE_ENABLED:
            return False

        current_hour_utc = datetime.now(timezone.utc).hour

        if NIGHT_MODE_START_UTC <= current_hour_utc < NIGHT_MODE_END_UTC:
            return True

        return False
```

## 4. Sector Diversification

```python
# /home/user/jjhbn/GGG3/risk/sector_limits.py

# Mapping монет по секторам
SECTOR_MAP = {
    'BTC': ['BTCUSDT'],
    'ETH': ['ETHUSDT'],
    'L1': ['SOLUSDT', 'AVAXUSDT', 'ADAUSDT', 'DOTUSDT'],
    'L2': ['MATICUSDT', 'ARBUSDT', 'OPUSDT'],
    'DEFI': ['UNIUSDT', 'AAVEUSDT', 'CRVUSDT'],
    'MEME': ['DOGEUSDT', 'SHIBUSDT'],
    # ...
}

def get_coin_sector(symbol: str) -> str:
    """Определяет сектор монеты"""
    for sector, coins in SECTOR_MAP.items():
        if symbol in coins:
            return sector
    return 'OTHER'

def apply_sector_limits(opportunities: List[dict]) -> List[dict]:
    """
    Применяет ограничение: максимум 3 монеты из одного сектора

    Args:
        opportunities: Список возможностей, отсортированный по EV

    Returns:
        Отфильтрованный список
    """

    if not SECTOR_DIVERSIFICATION_ENABLED:
        return opportunities

    sector_counts = {}
    filtered = []

    for opp in opportunities:
        sector = get_coin_sector(opp['symbol'])
        count = sector_counts.get(sector, 0)

        if count < MAX_COINS_PER_SECTOR:
            filtered.append(opp)
            sector_counts[sector] = count + 1

    return filtered
```

## 5. Auto-Reinvestment

```python
# /home/user/jjhbn/GGG3/portfolio/reinvestment.py

class ReinvestmentManager:
    """
    Автоматическое реинвестирование

    Логика:
    1. Весь рост капитала реинвестируется
    2. 50% дневной прибыли → резервный фонд
    3. Резерв для защиты от drawdown
    """

    def __init__(self):
        self.initial_capital = INITIAL_CAPITAL_PAPER
        self.reserve_fund = 0.0
        self.last_reinvest_date = datetime.now().date()

    def calculate_daily_reinvestment(self, current_equity: float) -> dict:
        """
        Рассчитывает реинвестирование

        Returns:
            {
              'trading_capital': float,
              'reserve_fund': float,
              'total_equity': float
            }
        """

        if not AUTO_REINVEST_ENABLED:
            return {
                'trading_capital': current_equity,
                'reserve_fund': 0.0,
                'total_equity': current_equity
            }

        today = datetime.now().date()

        if today > self.last_reinvest_date:
            # Ежедневное реинвестирование

            # Дневная прибыль
            daily_profit = max(0, current_equity - self.initial_capital)

            if daily_profit > 0:
                # 50% прибыли → резерв
                to_reserve = daily_profit * REINVEST_PROFIT_TO_RESERVE_PCT
                self.reserve_fund += to_reserve

                # Остальное реинвестируем
                trading_capital = current_equity - to_reserve

                print(f"[Reinvest] Daily profit: ${daily_profit:.2f}")
                print(f"  To reserve: ${to_reserve:.2f}")
                print(f"  Trading capital: ${trading_capital:.2f}")
                print(f"  Reserve fund: ${self.reserve_fund:.2f}")

                self.last_reinvest_date = today

                return {
                    'trading_capital': trading_capital,
                    'reserve_fund': self.reserve_fund,
                    'total_equity': trading_capital + self.reserve_fund
                }

        return {
            'trading_capital': current_equity,
            'reserve_fund': self.reserve_fund,
            'total_equity': current_equity + self.reserve_fund
        }
```

---

# 📊 ЧЕКЛИСТ ГОТОВНОСТИ К LIVE

## Paper Trading (ОБЯЗАТЕЛЬНО)

- [ ] Paper trading запущен с $1000
- [ ] Минимум 2 недели работы без критических ошибок
- [ ] Win Rate ≥ 48%
- [ ] Sharpe Ratio ≥ 1.5
- [ ] Max Drawdown ≤ 15%
- [ ] Profit Factor ≥ 1.3
- [ ] Total Return > 0%
- [ ] Метрики близки к бэктесту (±10%)
- [ ] Дашборд работает стабильно
- [ ] Все алерты настроены

## Технические требования

- [ ] Temporal consistency реализован (snapshot pattern)
- [ ] LONG/SHORT logic работает
- [ ] Adaptive threshold адаптируется
- [ ] EV filter выбирает top-10
- [ ] Online learning работает (TP=1, SL=0)
- [ ] META в SHADOW mode
- [ ] Trailing stop работает
- [ ] Black swan protection активен
- [ ] Night mode блокирует ночную торговлю
- [ ] Sector diversification работает
- [ ] Auto-reinvestment реинвестирует

## Модели и данные

- [ ] Все 4 эксперта обучены
- [ ] META обучена
- [ ] Базовая логика модифицирована для multi-TF
- [ ] Бэктест показывает WR > 50%
- [ ] Калибровка моделей ECE < 0.05
- [ ] Загрузка/сохранение моделей работает

## Безопасность

- [ ] API ключи в .env
- [ ] Только Futures API (не Spot)
- [ ] Leverage = 1x
- [ ] IP whitelist настроен
- [ ] 2FA включен
- [ ] API права: только торговля (no withdrawal)
- [ ] Backup система работает

**Если ВСЕ пункты ✅ → Можно запускать LIVE с малым капиталом ($500-1000)**
**Иначе → Продолжить paper trading и исправить проблемы!**

---

**Версия:** 2.0 FINAL
**Последнее обновление:** 2025-11-06
**Статус:** Готов к реализации с ОБЯЗАТЕЛЬНЫМ paper trading на $1000

🚀 **УДАЧИ В МИГРАЦИИ!** 🚀
