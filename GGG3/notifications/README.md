# Telegram Уведомления

Система отправки уведомлений о работе бота в Telegram.

## Настройка

### 1. Создайте Telegram бота

1. Откройте [@BotFather](https://t.me/BotFather) в Telegram
2. Отправьте команду `/newbot`
3. Следуйте инструкциям и получите **токен бота**
4. Сохраните токен (формат: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### 2. Получите Chat ID

1. Откройте [@userinfobot](https://t.me/userinfobot) в Telegram
2. Запустите бота - он отправит вам **ваш Chat ID**
3. Сохраните Chat ID (формат: `123456789`)

### 3. Настройте переменные окружения

**СПОСОБ 1: Через .env файл (рекомендуется)**

```bash
# Скопируйте пример
cp .env.example .env

# Отредактируйте .env и добавьте ваши токены
nano .env
```

В файле `.env`:
```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

**СПОСОБ 2: Через export (временно, на текущую сессию)**

```bash
export TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
export TELEGRAM_CHAT_ID="123456789"
```

**СПОСОБ 3: Через .bashrc (постоянно)**

```bash
echo 'export TELEGRAM_BOT_TOKEN="your_token_here"' >> ~/.bashrc
echo 'export TELEGRAM_CHAT_ID="your_chat_id_here"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Протестируйте

```bash
cd /home/user/jjhbn/GGG3/notifications
python3 test_telegram.py
```

Вы должны получить 9 тестовых сообщений в Telegram.

## Типы уведомлений

### 🤖 Бот
- **bot_started**: Старт бота
- **bot_stopped**: Остановка бота

### 📊 Позиции
- **position_opened**: Открыта новая позиция
  - Символ, направление, цена входа
  - TP/SL уровни, R/R ratio
  - Вероятность роста, EV
  - Предсказания экспертов

- **position_closed**: Закрыта позиция
  - Цены входа/выхода
  - PnL в USDT и процентах
  - Причина закрытия (TP/SL/timeout)
  - Время удержания

### 🛡 Риск-менеджмент
- **trailing_stop**: Активирован trailing stop
  - Текущая цена, новый SL
  - Текущая прибыль

- **black_swan**: Black Swan Event
  - Процент падения BTC
  - Количество закрытых позиций
  - Блокировка новых входов

- **night_mode**: Night Mode
  - Активация/деактивация
  - Лимит позиций

### ⚖️ Портфель
- **rebalance**: Ребалансировка портфеля
  - Количество закрытых/открытых позиций
  - Топ-5 возможностей по EV

### 🧠 Обучение
- **online_learning**: Online learning (отключено по умолчанию)
  - Результат сделки
  - Обновленные эксперты

- **model_retrained**: Переобучение модели (отключено по умолчанию)
  - Метрики качества

### 📈 Статистика
- **daily_summary**: Ежедневная статистика
  - Win rate, Total PnL
  - Средние выигрыш/убыток
  - Лучшая/худшая сделка
  - Текущий баланс

## Настройка типов уведомлений

В файле `binance_config.py`:

```python
TELEGRAM_ALERT_TYPES = {
    'bot_started': True,          # Всегда включено
    'position_opened': True,       # Всегда включено
    'position_closed': True,       # Всегда включено
    'trailing_stop': True,         # Можно отключить если шумно
    'black_swan': True,            # Критично важно
    'night_mode': True,            # Можно отключить
    'rebalance': True,             # Раз в 4 часа
    'online_learning': False,      # Отключено (шумно)
    'model_retrained': False,      # Отключено (редко)
    'daily_summary': True,         # Раз в день
    'error': True,                 # Критично важно
}
```

## Примеры уведомлений

### Открытие позиции
```
🟢 ОТКРЫТА ПОЗИЦИЯ

BTCUSDT LONG x1
Вход: $35000.0000
Размер: $100.00

📊 TP: $36000.0000 (+2.86%)
🛡 SL: $34500.0000 (-1.43%)
📈 R/R: 2.00

🎯 P(up): 65.0%
💰 EV: +8.00%

Эксперты:
  • XGB: 67.0%
  • RF: 63.0%
  • NN: 66.0%
  • BASE: 64.0%

⏰ 2025-11-06 22:00:00 UTC
```

### Закрытие позиции
```
✅ ЗАКРЫТА ПОЗИЦИЯ

BTCUSDT LONG
Вход: $35000.0000
Выход: $35800.0000

💵 PnL: +$8.00 (+2.29%)
📝 Причина: 🎯 Take Profit
⏱ Время: 12.5h

⏰ 2025-11-06 22:00:00 UTC
```

### Black Swan Event
```
⚠️ BLACK SWAN EVENT!

BTC упал на 5.00% за 5 минут!
Все позиции экстренно закрыты.

Закрыто позиций: 8
Новые входы заблокированы на 2 часа.

⏰ 2025-11-06 22:00:00 UTC
```

## API Документация

### TelegramNotifier

```python
from notifications.telegram_notifier import TelegramNotifier

telegram = TelegramNotifier(
    bot_token="your_token",
    chat_id="your_chat_id",
    enabled=True
)

# Отправка уведомлений
telegram.notify_position_opened(position_data)
telegram.notify_position_closed(position_data)
telegram.notify_trailing_stop_activated(...)
telegram.notify_black_swan_triggered(...)
telegram.notify_rebalance_stats(stats)
```

## Troubleshooting

### Не приходят уведомления

1. Проверьте переменные окружения:
```bash
echo $TELEGRAM_BOT_TOKEN
echo $TELEGRAM_CHAT_ID
```

2. Проверьте настройку в боте:
```python
import binance_config as config
print(config.TELEGRAM_ALERTS_ENABLED)
```

3. Проверьте логи бота:
```bash
tail -f logs/binance_bot.log | grep telegram
```

### Ошибка "Chat not found"

- Убедитесь, что вы начали диалог с ботом (отправили `/start`)
- Проверьте правильность Chat ID

### Ошибка "Unauthorized"

- Проверьте правильность токена бота
- Убедитесь, что бот не был удален в BotFather

## Безопасность

⚠️ **ВАЖНО**:
- Никогда не коммитьте токены в Git
- Используйте переменные окружения
- Не делитесь токенами бота

## Дополнительно

- Telegram Bot API: https://core.telegram.org/bots/api
- Форматирование: поддерживается HTML
- Лимиты: 30 сообщений в секунду
