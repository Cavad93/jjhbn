# AI Trading Module 🤖

Автономный торговый модуль на базе **Claude Sonnet 4.5**, который самостоятельно принимает решения о торговле криптовалютами на Binance Futures.

## 📋 Возможности

### Автономное принятие решений
- 🎯 **Отбор монет**: AI самостоятельно выбирает топ-20 монет из ~400 доступных пар
- 📊 **Технический анализ**: Использует RSI, MACD, Bollinger Bands, ATR и другие индикаторы
- 📰 **Фундаментальный анализ**: Поиск новостей и событий в интернете (опционально)
- 🧠 **Self-learning**: Анализирует исторические решения и учится на ошибках

### Управление позициями
- 📈 **Long и Short**: Открывает позиции в обоих направлениях
- 🎚️ **Динамический размер**: AI определяет размер позиции (1-10% капитала)
- 🎯 **TP/SL**: Устанавливает уровни Take Profit и Stop Loss
- ⚖️ **Риск-менеджмент**: Контроль экспозиции, диверсификация, корреляция

### Работа и мониторинг
- ⏰ **Ежедневные решения**: 1 раз в сутки (настраивается)
- 🔍 **Проверка позиций**: Каждый час мониторинг TP/SL
- 📱 **Telegram отчеты**: Подробные отчеты о каждом решении
- 💾 **Хранение данных**: Все решения сохраняются для анализа

### Защита и безопасность
- 📄 **Paper Trading**: Безопасный режим симуляции
- 🛡️ **Hard Limits**: Жесткие ограничения на размер позиций и риски
- 🚨 **Emergency Stop**: Автоматическая остановка при большой просадке
- 💰 **API Budget**: Контроль расходов на AI вызовы

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
cd GGG3

# Установите зависимости для AI модуля
pip install -r ai_trading_module/requirements.txt

# Или вручную:
pip install anthropic python-dotenv
```

### 2. Настройка переменных окружения

```bash
# Обязательно: API ключ Anthropic
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# Опционально: Telegram уведомления
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
export TELEGRAM_CHAT_ID="your_telegram_chat_id"

# Для live trading (НЕ рекомендуется на старте)
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_SECRET_KEY="your_binance_secret_key"
```

### 3. Запуск в Paper Trading режиме

```bash
# Базовый запуск с $1000 начальным капиталом
python3 ai_trader_main.py --paper --capital 1000

# С кастомными параметрами
python3 ai_trader_main.py \
  --paper \
  --capital 1000 \
  --decision-hour 10 \
  --max-api-cost 50
```

### 4. Мониторинг

Бот будет работать непрерывно:
- **Решения**: Принимаются в заданный час (по умолчанию 10:00 UTC)
- **Проверка позиций**: Каждый час
- **Логи**: Сохраняются в `ai_trading_module/data/ai_trader.log`
- **Telegram**: Отправляет ежедневные отчеты и алерты

---

## 📊 Структура модуля

```
ai_trading_module/
├── __init__.py                 # Инициализация модуля
├── config.py                   # Конфигурация (все параметры)
├── ai_trader.py                # Основной класс AITrader
├── ai_tools.py                 # Инструменты для Sonnet 4.5
├── decision_manager.py         # Управление решениями и позициями
├── README.md                   # Эта документация
└── data/                       # Данные модуля
    ├── decisions.json          # История всех решений
    ├── ai_positions.json       # Текущие открытые позиции
    ├── ai_state.json           # Состояние (баланс, PnL)
    ├── ai_trader.log           # Логи
    └── paper_exchange_state.json  # Состояние paper exchange
```

---

## ⚙️ Конфигурация

Основные параметры находятся в `config.py`:

### Торговые параметры
```python
INITIAL_CAPITAL = 1000.0           # Начальный капитал
MAX_POSITIONS = 10                 # Максимум позиций
MIN_POSITIONS = 3                  # Минимум позиций

MIN_POSITION_SIZE_PCT = 1.0        # Минимум 1% на позицию
MAX_POSITION_SIZE_PCT = 10.0       # Максимум 10% на позицию
MAX_TOTAL_EXPOSURE_PCT = 60.0      # Максимум 60% в позициях

MIN_RISK_REWARD_RATIO = 1.5        # Минимум 1.5:1 R:R
```

### Временные параметры
```python
DECISION_INTERVAL_HOURS = 24       # Решения 1 раз в сутки
DECISION_TIME_UTC = 10             # В 10:00 UTC
CHECK_POSITIONS_INTERVAL_MINUTES = 60  # Проверка позиций каждый час
```

### AI модель
```python
AI_MODEL = "claude-sonnet-4-5-20250929"
AI_TEMPERATURE = 0.7               # Баланс креативность/консистентность
MAX_DAILY_API_COST_USD = 50.0     # Максимум $50/день на API
```

### Безопасность
```python
EMERGENCY_STOP_DRAWDOWN_PCT = 20.0  # Остановка при 20% просадке
MAX_CONSECUTIVE_LOSSES = 5          # Остановка после 5 убытков подряд
MIN_CAPITAL_TO_CONTINUE = 500.0     # Минимум $500 для продолжения
```

---

## 🛠️ Инструменты доступные AI

AI имеет доступ к следующим инструментам (tools):

### Рыночные данные
- `get_market_data`: OHLCV данные для списка символов
- `get_technical_indicators`: RSI, MACD, ATR, Bollinger Bands и др.
- `get_all_available_symbols`: Список всех торгуемых пар с фильтрами
- `calculate_symbol_metrics`: Объем, волатильность, ликвидность

### Фундаментальный анализ
- `search_fundamental_info`: Поиск новостей и информации в интернете

### Исторический анализ
- `get_historical_decisions`: Получить прошлые решения
- `analyze_historical_performance`: Статистика производительности

### Управление позициями
- `open_position`: Открыть позицию с TP/SL
- `get_current_positions`: Текущие открытые позиции
- `get_portfolio_status`: Баланс, PnL, экспозиция

### Расчеты
- `calculate_position_size`: Оптимальный размер позиции
- `validate_trade_parameters`: Проверка TP/SL и риск:reward

---

## 📈 Workflow AI модуля

### Ежедневный цикл принятия решений (1 раз в сутки)

1. **Анализ портфеля**
   - Проверка текущего баланса и открытых позиций
   - Анализ последней производительности

2. **Скрининг рынка**
   - AI получает список ~400 торговых пар
   - Применяет фильтры: объем > $10M, волатильность 2-15%
   - Исключает stablecoins и blacklist

3. **Отбор топ-20**
   - AI формирует **свои критерии** отбора (адаптивно)
   - Анализирует метрики: объем, волатильность, тренд
   - Выбирает 20 наиболее перспективных монет

4. **Детальный анализ**
   - Для каждой из топ-20:
     - Технические индикаторы (4h, 1d таймфреймы)
     - Исторические решения по этому символу
     - Фундаментальные факторы (опционально)

5. **Принятие решений**
   - AI решает: LONG, SHORT или SKIP
   - Определяет размер позиции (1-10%)
   - Устанавливает TP/SL уровни
   - Валидирует риск:reward >= 1.5:1

6. **Открытие позиций**
   - Проверка всех ограничений
   - Исполнение на бирже (или paper trading)
   - Сохранение полного snapshot решения

7. **Отчет в Telegram**
   - Какие монеты выбраны и почему
   - Какие позиции открыты
   - Детальное обоснование каждой сделки

### Проверка позиций (каждый час)

1. **Мониторинг открытых позиций**
   - Получение текущих цен
   - Проверка достижения TP или SL

2. **Закрытие позиций**
   - При достижении TP/SL
   - Расчет PnL
   - Сохранение результата для обучения

3. **Уведомления**
   - Telegram алерт о закрытии
   - Успех/неудача, PnL, длительность

---

## 📱 Telegram отчеты

### Ежедневный отчет
```
🤖 AI Trading Daily Report

Portfolio Status
• Total Equity: $1,045.30
• Return: +4.53%
• Open Positions: 7/10
• Exposure: 45.2%

Decisions Made: 1
API Cost Today: $2.50

Open Positions:
📈 BTCUSDT LONG
   PnL: $12.30 (+2.45%)
📈 ETHUSDT LONG
   PnL: $8.50 (+1.70%)
...

AI Summary:
Selected 7 positions based on bullish market conditions...
```

### Алерт о закрытии позиции
```
✅ Position Closed (TP)

Symbol: BTCUSDT
Direction: LONG
Entry: $42,000.00
Exit: $43,470.00
PnL: $24.50 (+3.50%)
Duration: 18.5h
```

---

## 🎓 Self-Learning (Самообучение)

AI модуль учится на собственном опыте:

### Что хранится
- Все рыночные данные на момент решения (OHLCV, индикаторы)
- Обоснование AI (reasoning)
- Параметры позиции (размер, TP, SL)
- Результат (PnL, успех/неудача)

### Как AI использует историю
- `get_historical_decisions`: AI запрашивает прошлые решения по символу
- `analyze_historical_performance`: Анализирует паттерны успешных сделок
- AI адаптирует критерии отбора на основе результатов
- Избегает повторения неудачных стратегий

### Пример анализа
```python
# AI может запросить:
get_historical_decisions(symbol="BTCUSDT", outcome="success", limit=10)

# И увидеть что успешные сделки имели:
# - RSI 45-55 (нейтральная зона)
# - MACD бычий кроссовер
# - Объем выше среднего

# В следующий раз AI будет искать похожие условия
```

---

## 📊 Анализ производительности

### Получение статистики

```python
from ai_trading_module.decision_manager import DecisionManager

# В коде можно запросить:
stats = decision_manager.analyze_performance(groupby="all", days_back=30)

# Результат:
{
  "total_trades": 25,
  "win_rate_pct": 64.0,
  "total_pnl_usdt": 123.45,
  "average_win_usdt": 8.50,
  "average_loss_usdt": -3.20,
  "profit_factor": 2.66,
  "best_trade": {"symbol": "BTCUSDT", "pnl": 24.50},
  "worst_trade": {"symbol": "SOLUSDT", "pnl": -8.30}
}
```

### Анализ по направлениям

```python
stats = decision_manager.analyze_performance(groupby="direction", days_back=30)

# Покажет отдельную статистику для LONG и SHORT
```

---

## 🔒 Безопасность и ограничения

### Hard Limits (жесткие ограничения)

```python
# В config.py установлены строгие лимиты:

# Размер позиции
MAX_POSITION_SIZE_PCT = 10.0       # Максимум 10% на одну позицию
MAX_TOTAL_EXPOSURE_PCT = 60.0      # Максимум 60% в позициях

# TP/SL
MIN_TP_DISTANCE_PCT = 1.5          # TP минимум на 1.5% от входа
MAX_SL_DISTANCE_PCT = 5.0          # SL максимум на 5% от входа
MIN_RISK_REWARD_RATIO = 1.5        # Минимум 1.5:1 R:R

# Риск
MAX_SINGLE_RISK_PCT = 2.0          # Максимум 2% риска на позицию
MAX_CORRELATION = 0.7              # Избегать коррелированных позиций
```

### Emergency Stop (экстренная остановка)

AI модуль автоматически останавливает торговлю если:
- ❌ Просадка достигла 20% (`EMERGENCY_STOP_DRAWDOWN_PCT`)
- ❌ 5 убытков подряд (`MAX_CONSECUTIVE_LOSSES`)
- ❌ Капитал ниже $500 (`MIN_CAPITAL_TO_CONTINUE`)

### Validation (проверка параметров)

Перед открытием позиции AI **обязан** использовать `validate_trade_parameters`:
```python
validation = validate_trade_parameters(
    symbol="BTCUSDT",
    direction="LONG",
    entry_price=42000,
    tp_price=43470,
    sl_price=40530,
    size_pct=5.0
)

if not validation['valid']:
    # Открытие позиции будет отклонено
    print(validation['errors'])
```

---

## 🧪 Тестирование

### Paper Trading (рекомендуется)

```bash
# Запуск с симуляцией
python3 ai_trader_main.py --paper --capital 1000

# Преимущества:
# ✅ Безопасно (нет реальных денег)
# ✅ Реальные цены с Binance
# ✅ Полная симуляция комиссий и слипажа
# ✅ Можно тестировать стратегии
```

### Ручной тест инструментов

```python
from ai_trading_module.ai_tools import AIToolsExecutor
from paper_trading.paper_exchange import PaperExchange
from ai_trading_module.decision_manager import DecisionManager
from ai_trading_module.config import AITradingConfig

# Инициализация
config = AITradingConfig()
exchange = PaperExchange(initial_balance=1000.0)
decision_manager = DecisionManager(config, exchange)
tools = AIToolsExecutor(exchange, decision_manager, config)

# Тестирование инструментов
result = tools.execute_tool("get_all_available_symbols", {
    "min_volume_usd": 10_000_000,
    "exclude_stablecoins": True
})
print(f"Found {result['total_symbols']} symbols")

# Анализ символа
metrics = tools.execute_tool("calculate_symbol_metrics", {
    "symbol": "BTCUSDT"
})
print(f"BTC volatility: {metrics['volatility']['atr_pct']}%")
```

---

## 🎯 Рекомендации по использованию

### Первые шаги (Paper Trading)
1. ✅ Запустите в paper trading режиме с $1000
2. ✅ Дайте работать 30 дней
3. ✅ Анализируйте решения AI через Telegram отчеты
4. ✅ Проверьте итоговую производительность

### Если результаты хорошие (Win Rate > 55%, Profit Factor > 1.5)
5. ⚠️ Запустите paper trading с $5000 еще на 30 дней
6. ⚠️ Убедитесь что результаты стабильны
7. ⚠️ Изучите паттерны успешных сделок

### Переход на Live Trading (ОПАСНО)
8. 🚨 Начните с **очень малого** капитала ($100-200)
9. 🚨 Понизьте MAX_POSITION_SIZE_PCT до 3-5%
10. 🚨 Установите EMERGENCY_STOP_DRAWDOWN_PCT на 10%
11. 🚨 Мониторьте КАЖДЫЙ день первые 2 недели

### Никогда не делайте
- ❌ НЕ начинайте live trading без тестирования
- ❌ НЕ вкладывайте больше чем можете потерять
- ❌ НЕ отключайте safety limits
- ❌ НЕ запускайте без мониторинга
- ❌ НЕ изменяйте конфиг без понимания последствий

---

## 🔧 Troubleshooting

### "ANTHROPIC_API_KEY not set"
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### "Daily API budget exhausted"
AI модуль ограничивает стоимость API вызовов. Увеличьте лимит:
```python
# В config.py или через CLI
python3 ai_trader_main.py --max-api-cost 100
```

### "Emergency conditions detected"
Торговля приостановлена из-за просадки или убытков. Проверьте:
```bash
# Посмотрите логи
tail -f ai_trading_module/data/ai_trader.log

# Проверьте результаты
cat ai_trading_module/data/ai_state.json
```

### AI не открывает позиции
Возможные причины:
- Недостаточно ликвидных монет (понизьте MIN_24H_VOLUME_USD)
- Все монеты не прошли валидацию (проверьте логи)
- Emergency stop активирован
- AI просто не видит хороших возможностей (это нормально!)

### Telegram уведомления не приходят
```bash
# Проверьте переменные
echo $TELEGRAM_BOT_TOKEN
echo $TELEGRAM_CHAT_ID

# Проверьте настройки в config.py
TELEGRAM_ENABLED = True
TELEGRAM_SEND_DAILY_REPORT = True
```

---

## 📚 Дополнительные ресурсы

### Файлы для изучения
- `ai_trading_module/config.py` - Все параметры конфигурации
- `ai_trading_module/ai_trader.py` - Основная логика AI трейдера
- `ai_trading_module/ai_tools.py` - Инструменты доступные AI
- `ai_trading_module/decision_manager.py` - Управление решениями

### Логи и данные
- `ai_trading_module/data/ai_trader.log` - Подробные логи
- `ai_trading_module/data/decisions.json` - Все решения
- `ai_trading_module/data/ai_state.json` - Текущее состояние

### Мониторинг через Python
```python
import json

# Посмотреть текущее состояние
with open('ai_trading_module/data/ai_state.json') as f:
    state = json.load(f)
    print(f"Balance: ${state['current_balance']}")
    print(f"Total PnL: ${state['total_pnl']}")

# Посмотреть последние решения
with open('ai_trading_module/data/decisions.json') as f:
    decisions = json.load(f)['decisions']
    recent = decisions[-5:]  # Последние 5
    for d in recent:
        print(f"{d['symbol']} {d['direction']}: {d['reasoning'][:100]}...")
```

---

## ⚠️ Disclaimer

**ВАЖНО**: Этот AI модуль является **экспериментальным** и не гарантирует прибыль.

- Криптовалютная торговля крайне рискованна
- AI может принимать неверные решения
- Рынок может быть непредсказуемым
- Всегда тестируйте в paper trading режиме
- Не вкладывайте больше чем можете потерять
- Используйте на свой страх и риск

**Автор не несет ответственности за финансовые потери.**

---

## 📞 Поддержка

По вопросам и предложениям:
- Изучите логи: `ai_trading_module/data/ai_trader.log`
- Проверьте конфигурацию: `ai_trading_module/config.py`
- Посмотрите историю решений: `ai_trading_module/data/decisions.json`

---

**Удачи в торговле! 🚀**
