# Paper Trading Module

Модуль для симуляции биржевой торговли с реальными ценами Binance.

## ✅ ВАЖНО: Используется РЕАЛЬНЫЙ Binance Mainnet

**Этот модуль использует:**
- ✅ **РЕАЛЬНЫЙ Binance mainnet** (api.binance.com)
- ✅ **Публичные endpoints БЕЗ API ключей**
- ✅ **Реальные рыночные данные**
- ✅ **Fallback механизм** (4 альтернативных endpoints)
- ❌ **НЕ testnet** (testnet.binance.vision не используется)

Подробности: см. [BINANCE_API_INFO.md](BINANCE_API_INFO.md)

## Описание

`PaperExchange` - это симулятор биржи для бумажной торговли, который:
- Использует **реальные цены** с Binance mainnet API
- Симулирует проскальзывание (0.05%) и комиссии (0.1%)
- Поддерживает LONG и SHORT позиции
- Автоматически проверяет и исполняет TP/SL ордера
- Сохраняет состояние в JSON для восстановления

## Установка

Модуль использует стандартные зависимости из `requirements.txt`:
```bash
pip install requests
```

## Быстрый старт

```python
from paper_trading import PaperExchange

# Создаем биржу с начальным капиталом 1000 USDT
exchange = PaperExchange(initial_capital=1000.0)

# Открываем LONG позицию
position = exchange.open_position(
    symbol='BTCUSDT',
    side='LONG',
    amount=0.01,
    tp_price=50000.0,  # Take Profit
    sl_price=40000.0   # Stop Loss
)

# Проверяем баланс и equity
print(f"Balance: {exchange.get_balance():.2f} USDT")
print(f"Equity: {exchange.get_equity():.2f} USDT")

# Проверяем и исполняем ордера
exchange.check_orders()

# Закрываем позицию вручную
closed = exchange.close_position(position['id'], reason='manual')
print(f"PnL: {closed['pnl_after_commission']:.2f} USDT")

# Получаем статистику
stats = exchange.get_statistics()
print(f"Win Rate: {stats['winrate']:.2f}%")
print(f"ROI: {stats['roi']:.2f}%")
```

## Основные функции

### Создание биржи

```python
exchange = PaperExchange(initial_capital=1000.0)
```

### Открытие позиции

```python
position = exchange.open_position(
    symbol='BTCUSDT',      # Торговая пара
    side='LONG',           # 'LONG' или 'SHORT'
    amount=0.01,           # Размер позиции
    tp_price=50000.0,      # Take Profit (опционально)
    sl_price=40000.0,      # Stop Loss (опционально)
    metadata={'key': 'value'}  # Дополнительные данные (опционально)
)
```

### Закрытие позиции

```python
closed_position = exchange.close_position(
    position_id='PAPER_XXX',
    reason='manual'  # 'manual', 'tp', 'sl'
)
```

### Рыночные ордера

```python
# Покупка
order = exchange.create_market_order('BTCUSDT', 'BUY', 0.01)

# Продажа
order = exchange.create_market_order('BTCUSDT', 'SELL', 0.01)
```

### Проверка ордеров TP/SL

```python
# Автоматически проверяет и исполняет ордера
executed_orders = exchange.check_orders()
```

### Получение информации

```python
# Баланс
balance = exchange.get_balance()

# Полный капитал (баланс + нереализованный PnL)
equity = exchange.get_equity()

# Открытые позиции
positions = exchange.get_open_positions()

# Активные ордера
orders = exchange.get_open_orders()

# Конкретная позиция
position = exchange.get_position('PAPER_XXX')

# Статистика торговли
stats = exchange.get_statistics()
```

### Сохранение и загрузка состояния

```python
# Сохранение
exchange.save_state('my_state.json')

# Загрузка
exchange = PaperExchange.load_state('my_state.json')
```

## Статистика

Метод `get_statistics()` возвращает:
- `total_trades` - общее количество сделок
- `wins` / `losses` - количество прибыльных/убыточных сделок
- `winrate` - процент прибыльных сделок
- `total_pnl` - общий PnL
- `avg_pnl` - средний PnL на сделку
- `avg_win` / `avg_loss` - средняя прибыль/убыток
- `profit_factor` - отношение прибыли к убыткам
- `roi` - ROI в процентах
- `current_equity` - текущий капитал

## Особенности

### Проскальзывание
- BUY: цена исполнения = рыночная цена × 1.0005
- SELL: цена исполнения = рыночная цена × 0.9995

### Комиссии
- 0.1% (0.001) на каждую операцию
- Вычитаются из баланса при BUY
- Вычитаются из полученной суммы при SELL

### Проверка TP/SL
- Использует данные свечей (high/low) для точного определения
- TP (LIMIT): исполняется при достижении target цены
- SL (STOP_LOSS): исполняется при достижении stop цены
- Автоматически закрывает связанную позицию

### Temporal Consistency
Каждая позиция может содержать `metadata` с полным snapshot'ом:
- Фичи на момент входа
- Предсказания моделей
- Контекст META
- Timestamp

Это обеспечивает корректное обучение без look-ahead bias.

## Исключения

- `InsufficientBalance` - недостаточно средств
- `OrderNotFound` - ордер/позиция не найдены
- `InvalidOrder` - некорректные параметры ордера

## Тестирование

```bash
# Тесты с mock'ами (не требуют доступа к Binance API)
cd /home/user/jjhbn/GGG3/paper_trading
python test_paper_exchange_mock.py

# Тесты с реальным API (требуют доступ к Binance)
python test_paper_exchange.py
```

## Интеграция с ботом

```python
# В основном боте
from paper_trading import PaperExchange

# Вместо реальной биржи
if PAPER_MODE:
    exchange = PaperExchange(initial_capital=1000.0)
else:
    exchange = RealBinanceExchange()

# Дальше используем одинаковый интерфейс
position = exchange.open_position(...)
exchange.check_orders()
stats = exchange.get_statistics()
```

## Roadmap

- [x] Базовые market orders
- [x] LONG и SHORT позиции
- [x] TP/SL ордера
- [x] Сохранение состояния
- [x] Статистика торговли
- [ ] Фьючерсы с плечом (leverage > 1x)
- [ ] Funding rate симуляция
- [ ] Trailing stop
- [ ] Частичное закрытие позиций

## Версия

**1.0.0** - Первый релиз модуля paper trading

## Лицензия

Часть проекта Binance Trading Bot (GGG3)
