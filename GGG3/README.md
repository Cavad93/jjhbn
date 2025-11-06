# 🤖 Binance Futures Trading Bot

Автоматизированный торговый бот для Binance Futures с ML-моделями, адаптивным управлением рисками и online learning.

---

## 🎯 Основные возможности

### 💹 Торговля
- ✅ **10 одновременных позиций** - диверсифицированный портфель
- ✅ **LONG и SHORT** - двунаправленная торговля
- ✅ **Expected Value (EV)** - выбор позиций на основе математического ожидания
- ✅ **Kelly Criterion** - оптимальный размер позиций
- ✅ **Adaptive Threshold** - динамическая подстройка порога входа

### 🧠 Машинное обучение
- ✅ **4 ML эксперта**: XGBoost, RandomForest, AdaptiveRF, NeuralNet
- ✅ **META модель** с CMA-ES обучением (~5200 параметров)
- ✅ **BASE Logic** - 4 сигнала (M, S, B, R) для pattern recognition
- ✅ **Online Learning** - непрерывное обучение на новых данных
- ✅ **6 фаз рынка** - phase-specific модели

### 🛡️ Управление рисками
- ✅ **Trailing Stop** - динамический stop loss при прибыли
- ✅ **Black Swan Protection** - защита от резких падений рынка
- ✅ **Night Mode** - снижение активности в ночное время
- ✅ **ATR-based TP/SL** - адаптивные уровни выхода
- ✅ **Sector Diversification** - ограничение по секторам

### 📊 Мониторинг
- ✅ **Telegram уведомления** - real-time информация о сделках
- ✅ **Paper Trading** - безопасное тестирование без реальных денег
- ✅ **Детальные логи** - полная история всех операций
- ✅ **Метрики производительности** - Win Rate, PnL, Sharpe Ratio

---

## 📋 Требования

- **Python**: 3.9 или выше (рекомендуется 3.11+)
- **RAM**: 4GB минимум, 8GB рекомендуется
- **Диск**: 2GB свободного места
- **ОС**:
  - ✅ **Windows** Server 2012 R2 / Windows 10 / Windows 11
  - ✅ **Linux** (Ubuntu, Debian, CentOS)
  - ✅ **macOS**

---

## 🚀 Быстрый старт (5 минут)

### Для Windows (Server 2012 R2 / 10 / 11)

```batch
REM Откройте Command Prompt (cmd.exe)
cd C:\path\to\jjhbn\GGG3

REM Запустите автоматическую установку
install.bat
```

**📚 Документация для Windows:**
- [QUICK_START_WINDOWS.md](QUICK_START_WINDOWS.md) - Быстрый старт
- [INSTALLATION_GUIDE_WINDOWS.md](INSTALLATION_GUIDE_WINDOWS.md) - Полное руководство

---

### Для Linux / macOS

```bash
# Клонируйте репозиторий (если ещё не сделано)
cd /home/user/jjhbn/GGG3

# Запустите автоматическую установку
./install.sh
```

Скрипт выполнит:
- ✅ Проверку системных требований
- ✅ Очистку старого окружения
- ✅ Создание виртуального окружения
- ✅ Установку всех зависимостей
- ✅ Настройку конфигурации
- ✅ Запуск тестов

### 2. Настройте API ключи

```bash
# Отредактируйте .env файл
nano .env
```

Добавьте ваши ключи:
```bash
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here
USE_TESTNET=true

# Опционально: Telegram уведомления
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 3. Запустите бота

```bash
# Paper trading (рекомендуется для начала)
./start_bot.sh

# Или запустите в фоновом режиме
./start_bot.sh --background
```

### 4. Проверьте статус

```bash
# Проверка статуса бота
./status_bot.sh

# Просмотр логов в реальном времени
tail -f logs/binance_bot.log
```

---

## 📚 Документация

### Для Windows
- **[QUICK_START_WINDOWS.md](QUICK_START_WINDOWS.md)** - Быстрый старт за 5 команд (Windows)
- **[INSTALLATION_GUIDE_WINDOWS.md](INSTALLATION_GUIDE_WINDOWS.md)** - Полное руководство (Windows)

### Для Linux / macOS
- **[QUICK_START.md](QUICK_START.md)** - Быстрый старт за 5 команд (Linux/macOS)
- **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Полное руководство (Linux/macOS)

### Общее
- **[README.md](README.md)** - Этот файл
- **[BOT_ARCHITECTURE.md](BOT_ARCHITECTURE.md)** - Детальное описание архитектуры

---

## 🔧 Управление ботом

### Для Windows

```batch
REM Запуск бота (интерактивный режим)
start_bot.bat

REM Запуск в фоновом режиме
start_bot.bat --background

REM Проверка статуса
status_bot.bat

REM Просмотр логов
tail_logs.bat

REM Остановка бота
stop_bot.bat
```

---

### Для Linux / macOS

```bash
# Интерактивный режим (paper trading)
./start_bot.sh

# Фоновый режим (paper trading)
./start_bot.sh --background

# Live trading (ОСТОРОЖНО!)
./start_bot.sh --live --background
```

### Остановка

```bash
# Graceful shutdown
./stop_bot.sh

# Или нажмите Ctrl+C в интерактивном режиме
```

### Проверка статуса

```bash
# Полный статус
./status_bot.sh

# Краткая проверка
ps aux | grep binance_bot_main.py
```

### Просмотр логов

```bash
# В реальном времени
tail -f logs/binance_bot.log

# Последние 100 строк
tail -n 100 logs/binance_bot.log

# Поиск ошибок
grep ERROR logs/binance_bot.log

# Просмотр открытия позиций
grep "ENTRY" logs/binance_bot.log

# Просмотр закрытия позиций
grep "EXIT" logs/binance_bot.log
```

---

## 🧪 Тестирование

### Быстрый тест (30 секунд)

```bash
source venv/bin/activate
python3 tests/quick_bot_test.py
```

Проверяет:
- ✅ BASE Logic - Pattern Recognition
- ✅ ML Models - XGBoost, RandomForest, NeuralNet
- ✅ Position Management
- ✅ Risk Management
- ✅ Adaptive Threshold
- ✅ Kelly Sizing

### Полный тест (2-3 минуты)

```bash
source venv/bin/activate
python3 tests/comprehensive_bot_test.py
```

Проверяет все 8 компонентов системы с детальными assertions.

---

## 📊 Архитектура бота

### Процесс принятия решений

```
┌─────────────────────────────────────────────────────────────────┐
│                     MARKET DATA (OHLCV)                         │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                  ┌──────────────────────┐
                  │  Feature Builder     │
                  │  (68 features)       │
                  └──────────┬───────────┘
                             ↓
          ┌──────────────────┴──────────────────┐
          ↓                                     ↓
┌──────────────────┐                 ┌──────────────────┐
│   BASE Logic     │                 │   ML Experts     │
│  (M, S, B, R)    │                 │  (XGB, RF, NN)   │
│   p_base         │                 │  p_xgb, p_rf...  │
└────────┬─────────┘                 └────────┬─────────┘
         │                                    │
         └──────────────┬─────────────────────┘
                        ↓
                ┌───────────────┐
                │  META Network │
                │  (Neural CMA) │
                │    p_meta     │
                └───────┬───────┘
                        ↓
                ┌───────────────┐
                │  EV Calculator│
                │  LONG / SHORT │
                └───────┬───────┘
                        ↓
           ┌────────────┴────────────┐
           ↓                         ↓
  ┌─────────────────┐      ┌─────────────────┐
  │ Risk Management │      │ Position Sizing │
  │  (TS, BS, NM)   │      │ (Kelly Criterion)│
  └────────┬────────┘      └────────┬─────────┘
           │                        │
           └────────┬───────────────┘
                    ↓
            ┌───────────────┐
            │ OPEN POSITION │
            │ (TP/SL set)   │
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │   MONITORING  │
            │  (5m checks)  │
            └───────┬───────┘
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
  ┌───────────────┐   ┌───────────────┐
  │ TP/SL HIT     │   │ TRAILING STOP │
  │ TIMEOUT       │   │ BLACK SWAN    │
  └───────┬───────┘   └───────┬───────┘
          │                   │
          └─────────┬─────────┘
                    ↓
            ┌───────────────┐
            │ CLOSE POSITION│
            │  (record PnL) │
            └───────┬───────┘
                    ↓
            ┌───────────────┐
            │ ONLINE LEARNING│
            │ (update models)│
            └───────────────┘
```

### Основные циклы

1. **4-часовой цикл** (Portfolio Rebalancing):
   - Сканирование ~400 монет
   - Отбор топ-10 по EV
   - Закрытие позиций не в топ-10
   - Открытие новых позиций

2. **5-минутный цикл** (Position Monitoring):
   - Проверка TP/SL
   - Обновление Trailing Stop
   - Проверка таймаута (48h)
   - Online learning на закрытых позициях

---

## ⚙️ Конфигурация

Основные параметры в `binance_config.py`:

```python
# Портфель
MAX_POSITIONS = 10              # Максимум одновременных позиций
TOP_N_COINS = 10                # Топ монет для портфеля

# Риск-менеджмент
LEVERAGE = 1                    # Плечо (1x - без плеча)
TP_ATR_MULTIPLIER = 1.2         # Take Profit (1.2 × ATR)
SL_ATR_MULTIPLIER = 0.8         # Stop Loss (0.8 × ATR)
MAX_POSITION_HOLD_TIME_HOURS = 48  # Максимальное время удержания

# Kelly Sizing
KELLY_FRACTION = 0.25           # 25% от полного Kelly
MIN_POSITION_SIZE_USD = 15      # Минимальный размер позиции
MAX_POSITION_SIZE_USD = 200     # Максимальный размер позиции

# Adaptive Threshold
THRESHOLD_MIN = 0.55            # Минимальный порог (консервативно)
THRESHOLD_MAX = 0.70            # Максимальный порог (агрессивно)

# Trailing Stop
TRAILING_STOP_ACTIVATION = 0.02 # Активация при +2%
TRAILING_STOP_CALLBACK = 0.01   # Откат на 1%

# Black Swan
BLACK_SWAN_BTC_DROP_PCT = 5.0   # Падение BTC >5% за час

# Paper Trading
PAPER_INITIAL_BALANCE = 10000   # Начальный капитал ($)
```

---

## 📈 Производительность

### Целевые метрики

- **Win Rate**: > 55%
- **Profit Factor**: > 1.5
- **Sharpe Ratio**: > 1.0
- **Max Drawdown**: < 15%
- **Average R/R**: 1.5:1

### Backtest результаты (пример)

```
Total Trades:     250
Win Rate:         58.4%
Profit Factor:    1.73
Total PnL:        +$2,450 (+24.5%)
Max Drawdown:     -$680 (-6.8%)
Sharpe Ratio:     1.42
```

---

## 🔒 Безопасность

### Рекомендации

1. ✅ **Используйте Paper Trading** минимум 1-2 недели перед live
2. ✅ **Ограничьте API права** - только Futures, только торговля
3. ✅ **Настройте IP whitelist** на Binance
4. ✅ **Начните с малого** - $100-500 для тестирования live
5. ✅ **Мониторьте ежедневно** - проверяйте логи и позиции
6. ✅ **Backup регулярно** - сохраняйте `data/` и `models/saved/`
7. ✅ **Не делитесь ключами** - никогда, ни с кем

### Защита API ключей

```bash
# .env файл НЕ должен коммититься в Git
cat .gitignore | grep .env

# Права доступа только для владельца
chmod 600 .env

# Проверка, что ключи не в репозитории
git log --all --full-history -- .env
```

---

## 🛠️ Решение проблем

### Бот не запускается

```bash
# Проверьте конфигурацию
python3 test_binance_config.py

# Проверьте зависимости
python3 verify_dependencies.py

# Посмотрите логи
cat logs/binance_bot.log
```

### Ошибки API Binance

```bash
# Проверьте ключи
cat .env | grep BINANCE

# Проверьте права API на Binance
# Должны быть включены:
# - Enable Futures
# - Enable Reading
# - Enable Spot & Margin Trading (для синхронизации)
```

### Telegram не работает

```bash
# Тест отправки уведомления
python3 -c "
from notifications.telegram_notifier import TelegramNotifier
import binance_config as config

telegram = TelegramNotifier(
    bot_token=config.TELEGRAM_BOT_TOKEN,
    chat_id=config.TELEGRAM_CHAT_ID,
    enabled=True
)
telegram.notify_bot_started(paper_mode=True, initial_capital=10000, max_positions=10)
"
```

### Полное руководство

См. раздел "Решение проблем" в [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)

---

## 📊 Структура проекта

```
GGG3/
├── binance_bot_main.py          # Главный файл бота
├── binance_config.py            # Конфигурация
├── meta_neural_cem.py           # META модель
│
├── binance_api/                 # Binance API клиент
│   └── client.py
│
├── features/                    # Feature Engineering
│   ├── builder.py               # 68D feature builder
│   └── base_logic.py            # BASE Logic (M,S,B,R)
│
├── models/                      # ML модели
│   ├── experts.py               # XGBoost, RF, NN
│   └── saved/                   # Сохранённые модели
│
├── portfolio/                   # Управление портфелем
│   └── position_manager.py      # Открытие/закрытие позиций
│
├── risk/                        # Риск-менеджмент
│   ├── trailing_stop.py
│   ├── black_swan.py
│   └── night_mode.py
│
├── strategy/                    # Торговая стратегия
│   ├── coin_selector.py         # Отбор топ-10
│   ├── threshold_adapter.py     # Adaptive threshold
│   └── kelly_sizer.py           # Kelly Criterion
│
├── notifications/               # Уведомления
│   └── telegram_notifier.py
│
├── paper_trading/               # Paper trading
│   └── paper_exchange.py
│
├── tests/                       # Тесты
│   ├── quick_bot_test.py
│   ├── comprehensive_bot_test.py
│   └── market_pattern_generator.py
│
├── data/                        # Данные
│   ├── positions/               # Позиции
│   └── market_data/             # Рыночные данные
│
├── logs/                        # Логи
│   └── binance_bot.log
│
├── install.sh                   # Скрипт установки
├── start_bot.sh                 # Запуск бота
├── stop_bot.sh                  # Остановка бота
├── status_bot.sh                # Проверка статуса
│
├── README.md                    # Этот файл
├── QUICK_START.md              # Быстрый старт
├── INSTALLATION_GUIDE.md       # Полное руководство
└── requirements.txt            # Зависимости
```

---

## 🤝 Поддержка

### Документация

- [Binance API Docs](https://binance-docs.github.io/apidocs/futures/en/)
- [Binance Testnet](https://testnet.binancefuture.com/)
- [Telegram Bot API](https://core.telegram.org/bots/api)

### Логи и диагностика

```bash
# Основной лог
tail -f logs/binance_bot.log

# Статус бота
./status_bot.sh

# Системная информация
./status_bot.sh | grep -A 10 "Использование ресурсов"
```

---

## ⚠️ Дисклеймер

**Этот бот создан для образовательных целей. Криптовалютная торговля несёт высокие риски. Используйте на свой страх и риск.**

- ❌ Авторы НЕ несут ответственности за финансовые потери
- ❌ Прошлые результаты НЕ гарантируют будущую прибыль
- ❌ Всегда начинайте с Paper Trading
- ❌ Не инвестируйте больше, чем можете потерять
- ✅ Используйте stop loss
- ✅ Мониторьте бота регулярно
- ✅ Читайте документацию перед использованием

---

## 📜 Лицензия

MIT License - см. LICENSE файл

---

## 🎯 Roadmap

### Версия 2.0 (планируется)
- [ ] WebSocket real-time данные
- [ ] Dashboard с визуализацией
- [ ] Дополнительные ML модели (LightGBM, CatBoost)
- [ ] Sentiment analysis (Twitter, Reddit)
- [ ] Multi-exchange support (Bybit, OKX)
- [ ] Advanced risk management (VaR, CVaR)
- [ ] Backtesting framework
- [ ] Strategy optimizer

---

**Удачной торговли! 🚀📈**

*Сделано с ❤️ Claude Code*

---

**Версия**: 1.0
**Последнее обновление**: 2025-11-06
**Python**: 3.9+
**Статус**: Production Ready (Paper Trading)
