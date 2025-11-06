# Portfolio Management Module

Модуль для управления портфелем торговых позиций с поддержкой до 10 одновременных позиций.

## Особенности

✅ **Temporal Consistency** - полный snapshot на момент входа для обучения моделей
✅ **Управление портфелем** - до 10 одновременных позиций
✅ **LONG и SHORT** - поддержка обоих направлений
✅ **Расчет PnL** - автоматический расчет прибыли/убытка
✅ **TP/SL проверки** - методы для проверки достижения целей
✅ **Персистентность** - сохранение/загрузка из JSON
✅ **Статистика** - детальная статистика по закрытым позициям

## Установка

Модуль является частью проекта GGG3. Зависимости:

```bash
pip install numpy
```

## Быстрый старт

### Создание и управление позицией

```python
from portfolio import Position, PositionManager
from datetime import datetime
import numpy as np

# Создаем менеджер позиций
manager = PositionManager(max_positions=10)

# Создаем snapshot на момент входа (КРИТИЧНО!)
entry_snapshot = {
    'features_68d': np.array([0.1, 0.2, 0.3] + [0.0] * 65),
    'predictions': {
        'p_xgb': 0.65,
        'p_rf': 0.62,
        'p_arf': 0.64,
        'p_nn': 0.63,
        'p_base': 0.60
    },
    'meta_context': {
        'volatility': 0.05,
        'volume': 1000000,
        # ... другие метрики
    },
    'p_meta': 0.68,
    'ev_long': 0.0700,
    'ev_short': 0.0200,
    'selected_ev': 0.0700,
    'timestamp': datetime.now().timestamp()
}

# Открываем LONG позицию
position = Position(
    symbol='BTCUSDT',
    direction='LONG',
    entry_time=datetime.now(),
    entry_price=45000.0,
    amount=0.001,
    position_value=45.0,
    tp_price=46350.0,  # TP = +1.2*ATR
    sl_price=44100.0,  # SL = -0.8*ATR
    atr_value=1000.0,
    entry_order_id='ORDER_123',
    tp_order_id='TP_123',
    sl_order_id='SL_123',
    entry_snapshot=entry_snapshot  # ✅ КРИТИЧНО!
)

# Добавляем в менеджер
manager.add_position(position)
```

### Проверка TP/SL

```python
# Получаем текущую цену
current_price = 46500.0

# Проверяем достижение целей
if position.is_tp_hit(current_price):
    print("TP достигнут!")
    manager.close_position('BTCUSDT', exit_price=current_price, exit_reason='TP')

elif position.is_sl_hit(current_price):
    print("SL достигнут!")
    manager.close_position('BTCUSDT', exit_price=current_price, exit_reason='SL')
```

### Расчет нереализованного PnL

```python
# Для открытой позиции
current_price = 45500.0
pnl_usdt, pnl_pct = position.calculate_pnl(current_price)
print(f"Текущий PnL: ${pnl_usdt:.2f} ({pnl_pct:.2f}%)")
```

## Класс Position

### Поля

#### Основная информация
- `symbol: str` - торговая пара (например, 'BTCUSDT')
- `direction: str` - 'LONG' или 'SHORT'
- `entry_time: datetime` - время открытия
- `entry_price: float` - цена входа
- `amount: float` - количество актива
- `position_value: float` - стоимость позиции в USDT

#### Take Profit / Stop Loss
- `tp_price: float` - цена Take Profit
- `sl_price: float` - цена Stop Loss
- `atr_value: float` - значение ATR на момент входа

#### ID ордеров
- `entry_order_id: Optional[str]` - ID entry ордера на бирже
- `tp_order_id: Optional[str]` - ID TP ордера
- `sl_order_id: Optional[str]` - ID SL ордера

#### Entry Snapshot (КРИТИЧНО!)
- `entry_snapshot: Dict[str, Any]` - полный snapshot на момент входа

Структура `entry_snapshot`:
```python
{
    'features_68d': np.array([...]),  # 68 признаков
    'predictions': {                   # Предсказания моделей
        'p_xgb': float,
        'p_rf': float,
        'p_arf': float,
        'p_nn': float,
        'p_base': float
    },
    'meta_context': {                  # 7 признаков для META
        'feature1': float,
        ...
    },
    'p_meta': float,                   # Финальное предсказание META
    'ev_long': float,                  # EV для LONG
    'ev_short': float,                 # EV для SHORT
    'selected_ev': float,              # Выбранный EV
    'timestamp': float                 # Unix timestamp
}
```

#### Информация о закрытии
- `exit_time: Optional[datetime]` - время закрытия
- `exit_price: Optional[float]` - цена выхода
- `exit_reason: Optional[str]` - причина ('TP', 'SL', 'TIMEOUT', 'MANUAL', 'REPLACED')

#### PnL
- `pnl: float` - прибыль/убыток в USDT
- `pnl_pct: float` - прибыль/убыток в %

#### Статус
- `status: str` - 'OPEN' или 'CLOSED'

### Методы

#### calculate_pnl(current_price)
Рассчитывает текущий нереализованный PnL

```python
pnl_usdt, pnl_pct = position.calculate_pnl(46000.0)
# Для LONG: pnl = (current - entry) * amount
# Для SHORT: pnl = (entry - current) * amount
```

#### is_tp_hit(current_price)
Проверяет достиг ли цена TP

```python
if position.is_tp_hit(46500.0):
    print("TP достигнут!")
```

#### is_sl_hit(current_price)
Проверяет достиг ли цена SL

```python
if position.is_sl_hit(43500.0):
    print("SL достигнут!")
```

#### get_duration()
Возвращает длительность позиции в секундах

```python
duration = position.get_duration()
hours = duration / 3600
print(f"Позиция открыта {hours:.1f} часов")
```

#### to_dict() / from_dict()
Сериализация и десериализация для JSON

```python
# Сериализация
data = position.to_dict()

# Десериализация
restored = Position.from_dict(data)
```

## Класс PositionManager

### Инициализация

```python
manager = PositionManager(
    max_positions=10,  # Максимум одновременных позиций
    data_dir='/home/user/jjhbn/GGG3/data/positions'  # Директория для данных
)
```

### Основные методы

#### can_open_position(symbol)
Проверяет можно ли открыть позицию

```python
can_open, reason = manager.can_open_position('BTCUSDT')

if can_open:
    print("Можно открыть позицию")
else:
    print(f"Нельзя открыть: {reason}")
```

**Правила:**
1. Максимум `max_positions` одновременных позиций
2. Не более 1 позиции на символ

#### add_position(position)
Добавляет открытую позицию

```python
position = Position(...)
manager.add_position(position)
# Автоматически сохраняет в JSON
```

#### close_position(symbol, exit_price, exit_reason)
Закрывает позицию и рассчитывает PnL

```python
closed_position = manager.close_position(
    symbol='BTCUSDT',
    exit_price=46000.0,
    exit_reason='TP'  # 'TP', 'SL', 'TIMEOUT', 'MANUAL', 'REPLACED'
)

print(f"PnL: ${closed_position.pnl:.2f}")
```

#### get_all_open()
Возвращает все открытые позиции

```python
open_positions = manager.get_all_open()
for pos in open_positions:
    print(f"{pos.symbol}: {pos.direction} @ {pos.entry_price}")
```

#### get_position(symbol)
Получает позицию по символу

```python
position = manager.get_position('BTCUSDT')
if position:
    print(f"Позиция найдена: {position}")
```

#### has_position(symbol)
Проверяет наличие позиции

```python
if manager.has_position('BTCUSDT'):
    print("Позиция по BTCUSDT уже открыта")
```

#### get_total_exposure()
Рассчитывает общую стоимость всех позиций

```python
total = manager.get_total_exposure()
print(f"Общая экспозиция: ${total:.2f}")
```

#### get_open_count() / get_available_slots()
Информация о доступных слотах

```python
open_count = manager.get_open_count()
available = manager.get_available_slots()

print(f"Открыто: {open_count}/{manager.max_positions}")
print(f"Доступно слотов: {available}")
```

#### get_pnl_summary()
Статистика по закрытым позициям

```python
summary = manager.get_pnl_summary()

print(f"Всего сделок: {summary['total_trades']}")
print(f"Win rate: {summary['win_rate']:.1f}%")
print(f"Прибыль: {summary['wins']}, Убытки: {summary['losses']}")
print(f"Общий PnL: ${summary['total_pnl']:.2f}")
print(f"Средняя прибыль: ${summary['avg_win']:.2f}")
print(f"Средний убыток: ${summary['avg_loss']:.2f}")
print(f"Profit Factor: {summary['profit_factor']:.2f}")
```

Возвращаемые метрики:
- `total_pnl` - общий PnL в USDT
- `total_pnl_pct` - средний PnL в %
- `win_rate` - процент прибыльных сделок
- `total_trades` - количество сделок
- `wins` / `losses` - количество прибыльных/убыточных
- `avg_win` / `avg_loss` - средние значения
- `profit_factor` - отношение прибылей к убыткам
- `max_win` / `max_loss` - максимальные значения

#### get_positions_by_direction()
Подсчет по направлениям

```python
by_direction = manager.get_positions_by_direction()
print(f"LONG: {by_direction['LONG']}")
print(f"SHORT: {by_direction['SHORT']}")
```

#### get_recent_closed(limit)
Последние закрытые позиции

```python
recent = manager.get_recent_closed(limit=5)
for pos in recent:
    print(f"{pos.symbol}: PnL = {pos.pnl:+.2f} ({pos.exit_reason})")
```

### Персистентность

#### save_to_file(filename)
Сохраняет позиции в JSON

```python
manager.save_to_file('positions.json')
# Автоматически вызывается при add_position() и close_position()
```

Сохраняются:
- Все открытые позиции
- Последние 100 закрытых позиций
- Метаданные

#### load_from_file(filename)
Загружает позиции из JSON

```python
if manager.load_from_file('positions.json'):
    print("Позиции успешно загружены")
    print(f"Открыто: {manager.get_open_count()}")
    print(f"Закрыто: {len(manager.closed_positions)}")
```

## Примеры использования

### Пример 1: Открытие и закрытие LONG позиции

```python
from portfolio import Position, PositionManager
from datetime import datetime
import numpy as np

# Инициализация
manager = PositionManager(max_positions=10)

# Проверяем возможность открытия
can_open, reason = manager.can_open_position('BTCUSDT')
if not can_open:
    print(f"Не могу открыть: {reason}")
    exit()

# Создаем snapshot
snapshot = {
    'features_68d': np.random.rand(68),
    'predictions': {'p_xgb': 0.65, 'p_rf': 0.62},
    'p_meta': 0.68,
    'selected_ev': 0.0700,
    'timestamp': datetime.now().timestamp()
}

# Открываем LONG
position = Position(
    symbol='BTCUSDT',
    direction='LONG',
    entry_time=datetime.now(),
    entry_price=45000.0,
    amount=0.001,
    position_value=45.0,
    tp_price=46350.0,
    sl_price=44100.0,
    atr_value=1000.0,
    entry_snapshot=snapshot
)

manager.add_position(position)
print(f"Позиция открыта: {position.symbol}")

# Проверяем статус
print(f"Открыто позиций: {manager.get_open_count()}")
print(f"Общая экспозиция: ${manager.get_total_exposure():.2f}")

# Симулируем движение цены и закрываем
current_price = 46500.0

if position.is_tp_hit(current_price):
    closed = manager.close_position('BTCUSDT', current_price, 'TP')
    print(f"TP достигнут! PnL: ${closed.pnl:.2f} ({closed.pnl_pct:.2f}%)")
```

### Пример 2: Управление портфелем из 10 позиций

```python
from portfolio import Position, PositionManager
from datetime import datetime

manager = PositionManager(max_positions=10)

# Открываем 10 позиций на разные символы
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
           'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'AVAXUSDT', 'UNIUSDT']

for i, symbol in enumerate(symbols):
    position = Position(
        symbol=symbol,
        direction='LONG' if i % 2 == 0 else 'SHORT',
        entry_time=datetime.now(),
        entry_price=100.0 * (i + 1),
        amount=0.1,
        position_value=10.0 * (i + 1),
        tp_price=110.0 * (i + 1),
        sl_price=90.0 * (i + 1),
        atr_value=5.0,
        entry_snapshot={'p_meta': 0.6 + i * 0.01}
    )

    manager.add_position(position)
    print(f"#{i+1}: {symbol} {position.direction} добавлена")

# Проверяем статус
print(f"\nОткрыто: {manager.get_open_count()}/{manager.max_positions}")
print(f"Доступно слотов: {manager.get_available_slots()}")

# Пытаемся открыть 11-ю позицию
can_open, reason = manager.can_open_position('XRPUSDT')
print(f"\nМожно открыть XRPUSDT? {can_open} ({reason})")

# Закрываем несколько позиций
manager.close_position('BTCUSDT', 110.0, 'TP')
manager.close_position('ETHUSDT', 180.0, 'SL')

# Получаем статистику
summary = manager.get_pnl_summary()
print(f"\nСтатистика:")
print(f"  Сделок: {summary['total_trades']}")
print(f"  Win rate: {summary['win_rate']:.1f}%")
print(f"  Общий PnL: ${summary['total_pnl']:.2f}")
```

### Пример 3: Использование snapshot для обучения

```python
from portfolio import Position, PositionManager

manager = PositionManager()

# При открытии позиции сохраняем полный snapshot
position = Position(
    symbol='BTCUSDT',
    direction='LONG',
    entry_time=datetime.now(),
    entry_price=45000.0,
    amount=0.001,
    position_value=45.0,
    tp_price=46000.0,
    sl_price=44000.0,
    atr_value=1000.0,
    entry_snapshot={
        'features_68d': calculated_features,  # Признаки на момент t0
        'predictions': all_model_predictions,  # Предсказания на момент t0
        'p_meta': meta_prediction,             # META на момент t0
        'timestamp': time.time()
    }
)

manager.add_position(position)

# ... позиция живет какое-то время ...

# При закрытии используем СОХРАНЕННЫЕ признаки для обучения
closed = manager.close_position('BTCUSDT', 46000.0, 'TP')

# ✅ ПРАВИЛЬНО: Используем snapshot с момента входа (t0)
outcome = 1 if closed.exit_reason == 'TP' else 0
features_at_entry = closed.entry_snapshot['features_68d']
predictions_at_entry = closed.entry_snapshot['predictions']

# Обучаем модели на СТАРЫХ данных (temporal consistency)
for model_name, prediction in predictions_at_entry.items():
    model = get_model(model_name)
    model.train(features_at_entry, outcome)

# ❌ НЕПРАВИЛЬНО: Пересчитывать признаки на момент закрытия (t1)
# features_now = calculate_features()  # ← НЕ ДЕЛАТЬ ТАК!
# model.train(features_now, outcome)   # ← Look-ahead bias!
```

## Тестирование

Модуль включает полный набор из 13 тестов:

```bash
cd /home/user/jjhbn/GGG3
python3 portfolio/test_position_manager.py
```

**Результат:**
```
================================================================================
ТЕСТИРОВАНИЕ POSITION И POSITIONMANAGER
================================================================================

Тест 1: Импорт компонентов... ✓ ПРОЙДЕН
Тест 2: Создание Position... ✓ ПРОЙДЕН
Тест 3: Position с entry_snapshot... ✓ ПРОЙДЕН
Тест 4: Расчет PnL для LONG и SHORT... ✓ ПРОЙДЕН
Тест 5: Проверка достижения TP/SL... ✓ ПРОЙДЕН
Тест 6: Сериализация/десериализация Position... ✓ ПРОЙДЕН
Тест 7: Инициализация PositionManager... ✓ ПРОЙДЕН
Тест 8: Добавление позиций... ✓ ПРОЙДЕН
Тест 9: Закрытие позиций... ✓ ПРОЙДЕН
Тест 10: Статистика PnL... ✓ ПРОЙДЕН
Тест 11: Сохранение и загрузка из JSON... ✓ ПРОЙДЕН
Тест 12: Проверка дублирования символа... ✓ ПРОЙДЕН
Тест 13: Расчет общей экспозиции... ✓ ПРОЙДЕН

Пройдено: 13/13
✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ
```

## Структура модуля

```
portfolio/
├── __init__.py                 # Экспорты модуля
├── position_manager.py         # Классы Position и PositionManager
├── test_position_manager.py    # Тесты (13 тестов)
└── README.md                   # Документация
```

## Важные замечания

### ⚠️ Temporal Consistency

**КРИТИЧНО:** Всегда сохраняйте полный snapshot на момент входа в `entry_snapshot`. Это необходимо для корректного обучения моделей без look-ahead bias.

```python
# ✅ ПРАВИЛЬНО
entry_snapshot = {
    'features_68d': features_at_t0,      # Признаки НА МОМЕНТ входа
    'predictions': predictions_at_t0,    # Предсказания НА МОМЕНТ входа
    'timestamp': t0
}

# При закрытии используем СОХРАНЕННЫЕ данные
features = closed_position.entry_snapshot['features_68d']
model.train(features, outcome)  # Обучение на данных с момента t0

# ❌ НЕПРАВИЛЬНО
features = calculate_features_now()  # Данные НА МОМЕНТ закрытия
model.train(features, outcome)       # Look-ahead bias!
```

### ⚠️ Ограничения

1. **Максимум позиций:** По умолчанию 10, настраивается при инициализации
2. **Одна позиция на символ:** Нельзя открыть 2 позиции на один символ
3. **Персистентность:** Автоматическое сохранение при `add_position()` и `close_position()`
4. **История:** Сохраняются последние 100 закрытых позиций

### ⚠️ PnL расчет

**LONG позиция:**
```
PnL = (exit_price - entry_price) * amount
PnL% = (exit_price / entry_price - 1) * 100
```

**SHORT позиция:**
```
PnL = (entry_price - exit_price) * amount
PnL% = (1 - exit_price / entry_price) * 100
```

## API Reference

### Position

**Создание:**
```python
Position(
    symbol: str,
    direction: str,  # 'LONG' или 'SHORT'
    entry_time: datetime,
    entry_price: float,
    amount: float,
    position_value: float,
    tp_price: float,
    sl_price: float,
    atr_value: float,
    entry_order_id: Optional[str] = None,
    tp_order_id: Optional[str] = None,
    sl_order_id: Optional[str] = None,
    entry_snapshot: Dict = {},
    status: str = 'OPEN'
)
```

**Методы:**
- `calculate_pnl(current_price) -> (float, float)`
- `is_tp_hit(current_price) -> bool`
- `is_sl_hit(current_price) -> bool`
- `get_duration() -> float`
- `to_dict() -> Dict`
- `from_dict(data: Dict) -> Position` (classmethod)

### PositionManager

**Создание:**
```python
PositionManager(
    max_positions: int = 10,
    data_dir: str = '/home/user/jjhbn/GGG3/data/positions'
)
```

**Методы управления:**
- `can_open_position(symbol) -> (bool, str)`
- `add_position(position: Position)`
- `close_position(symbol, exit_price, exit_reason) -> Position`
- `get_all_open() -> List[Position]`
- `get_position(symbol) -> Optional[Position]`
- `has_position(symbol) -> bool`

**Методы статистики:**
- `get_total_exposure() -> float`
- `get_open_count() -> int`
- `get_available_slots() -> int`
- `get_pnl_summary() -> Dict`
- `get_positions_by_direction() -> Dict`
- `get_recent_closed(limit=10) -> List[Position]`

**Методы персистентности:**
- `save_to_file(filename='positions.json')`
- `load_from_file(filename='positions.json') -> bool`

## Соответствие BINANCE_BOT_MASTER_PLAN.md

Модуль полностью соответствует требованиям из мастер-плана:

- ✅ Dataclass Position с полным набором полей
- ✅ Entry snapshot для temporal consistency
- ✅ Поддержка LONG и SHORT
- ✅ Управление до 10 позиций
- ✅ Правила: не более 1 позиции на символ
- ✅ Расчет PnL для обоих направлений
- ✅ Проверки TP/SL
- ✅ Персистентность через JSON
- ✅ Статистика и метрики
- ✅ Полное тестирование (13/13 тестов)

## Лицензия

См. основной проект.
