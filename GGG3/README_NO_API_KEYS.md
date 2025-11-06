# ✅ Запуск БЕЗ API ключей Binance

## 🎯 Главное

**ВАМ НЕ НУЖНЫ API КЛЮЧИ ДЛЯ PAPER TRADING!**

Бот работает в двух режимах:

1. **📄 Paper Trading** (рекомендуется) - **БЕЗ API ключей**
   - ✅ Реальные данные рынка через публичные API Binance
   - ✅ Виртуальные деньги ($10,000 начальный капитал)
   - ✅ Безопасно для тестирования и обучения
   - ✅ НЕ требует регистрации на Binance
   - ✅ НЕ требует API ключей

2. **💰 Live Trading** - требует API ключи
   - ⚠️ Реальные деньги
   - ⚠️ Требует API ключи Binance
   - ⚠️ Только после успешного тестирования в paper trading

---

## 🚀 Быстрый старт БЕЗ API ключей (3 минуты)

### Шаг 1: Установка

```batch
REM Windows
cd C:\path\to\jjhbn\GGG3
install.bat
```

```bash
# Linux / macOS
cd /path/to/jjhbn/GGG3
./install.sh
```

### Шаг 2: Настройка .env (опционально)

Файл `.env` уже создан автоматически! Он настроен для paper trading.

**Проверьте содержимое:**

```batch
REM Windows
notepad .env
```

```bash
# Linux / macOS
nano .env
```

**Должно быть:**

```ini
PAPER_TRADING_MODE=true
PAPER_INITIAL_BALANCE=10000
USE_TESTNET=false

# API ключи НЕ нужны, оставьте пустыми:
BINANCE_API_KEY=
BINANCE_SECRET_KEY=

# Telegram опционально:
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

### Шаг 3: Запуск

```batch
REM Windows
start_bot.bat
```

```bash
# Linux / macOS
./start_bot.sh
```

**Готово!** Бот запущен и торгует виртуальными деньгами на реальных данных рынка! 🎉

---

## ❓ FAQ - Частые вопросы

### 1. Мне действительно не нужны API ключи?

**Да!** Для paper trading API ключи **НЕ требуются**.

Бот использует **публичные API** Binance для получения данных:
- Цены токенов
- OHLCV свечи
- Объёмы торгов
- Orderbook данные

Эти данные доступны без авторизации.

---

### 2. Откуда бот берёт данные рынка?

Бот подключается к **публичным эндпоинтам Binance**:

- `https://api.binance.com/api/v3/ticker/price` - текущие цены
- `https://api.binance.com/api/v3/klines` - исторические данные (свечи)
- `https://api.binance.com/api/v3/depth` - стакан ордеров
- `https://api.binance.com/api/v3/ticker/24hr` - статистика 24ч

**НЕ требуют API ключей!**

---

### 3. Это реальный рынок или симуляция?

**Реальный рынок!** (mainnet, НЕ testnet)

- ✅ Реальные цены BTC, ETH, и других токенов
- ✅ Реальные объёмы торгов
- ✅ Реальная волатильность
- ✅ Реальные рыночные условия

**НО:**
- 💰 Виртуальные деньги (симулированный баланс)
- 📄 Ордера не отправляются на биржу
- 🛡️ Безопасно для тестирования

---

### 4. Зачем нужен USE_TESTNET=false?

**`USE_TESTNET=false`** означает использование **mainnet** (реального рынка).

**Сравнение:**

| Параметр | mainnet (`false`) | testnet (`true`) |
|----------|-------------------|------------------|
| Данные рынка | ✅ **Реальные** | ❌ Тестовые (не актуальные) |
| Цены | ✅ Актуальные | ❌ Симулированные |
| Объёмы | ✅ Реальные | ❌ Симулированные |
| API ключи | ❌ НЕ нужны | ❌ НЕ нужны |
| Paper trading | ✅ Работает | ✅ Работает |

**Для paper trading ВСЕГДА используйте mainnet** (реальные данные)!

---

### 5. Как работает paper trading?

**Paper trading** = виртуальная торговля на реальных данных.

**Процесс:**

1. **Получение данных** - Бот запрашивает реальные цены через публичные API
2. **Анализ рынка** - ML модели анализируют данные
3. **Принятие решений** - Бот решает открыть/закрыть позицию
4. **Симуляция ордера** - Ордер НЕ отправляется на биржу, а симулируется локально
5. **Обновление баланса** - Виртуальный баланс обновляется по реальным ценам

**Результат:**
- Вы видите как бот торгует
- Изучаете его стратегию
- Тестируете настройки
- **БЕЗ РИСКА** потери реальных денег

---

### 6. Можно ли использовать Telegram без API ключей?

**Да!** Telegram уведомления работают независимо от Binance API.

**Настройка (опционально):**

1. Создайте Telegram бота: https://t.me/BotFather → `/newbot`
2. Получите токен: `123456789:ABCdef...`
3. Получите Chat ID: https://t.me/userinfobot → `/start`
4. Добавьте в `.env`:
   ```ini
   TELEGRAM_BOT_TOKEN=123456789:ABCdef...
   TELEGRAM_CHAT_ID=123456789
   ```

**Бот будет отправлять уведомления о:**
- Открытии/закрытии позиций
- PnL (прибыль/убыток)
- Статистике торговли
- Ошибках

---

### 7. Когда понадобятся API ключи?

API ключи нужны **ТОЛЬКО** для **live trading** (реальная торговля):

**Live trading** = реальные деньги на вашем Binance аккаунте

**Для этого нужны:**
- ✅ API KEY - для идентификации
- ✅ SECRET KEY - для подписи запросов
- ✅ Права API: "Enable Futures", "Enable Spot & Margin Trading"
- ✅ IP Whitelist (рекомендуется)
- ❌ **НЕ включайте** "Enable Withdrawals"!

**Переход на live trading:**

1. Протестируйте в paper trading минимум 1-2 недели
2. Убедитесь что Win Rate > 55%
3. Проверьте PnL положительный
4. Получите API ключи на Binance
5. Измените в `.env`:
   ```ini
   PAPER_TRADING_MODE=false  # ОСТОРОЖНО!
   BINANCE_API_KEY=your_real_key
   BINANCE_SECRET_KEY=your_real_secret
   ```

⚠️ **НЕ ПЕРЕХОДИТЕ на live без тестирования!**

---

## 📋 Checklist для запуска БЕЗ API ключей

✅ **Шаг 1:** Установил Python 3.9+
✅ **Шаг 2:** Запустил `install.bat` (Windows) или `./install.sh` (Linux)
✅ **Шаг 3:** Проверил `.env` файл:
   - `PAPER_TRADING_MODE=true` ✅
   - `USE_TESTNET=false` ✅ (для реальных данных!)
   - `BINANCE_API_KEY=` (пустое) ✅
   - `BINANCE_SECRET_KEY=` (пустое) ✅
✅ **Шаг 4:** Запустил `start_bot.bat` (Windows) или `./start_bot.sh` (Linux)
✅ **Шаг 5:** Проверил статус: `status_bot.bat` или `./status_bot.sh`
✅ **Шаг 6:** Смотрю логи: `tail_logs.bat` или `tail -f logs/binance_bot.log`

---

## 🎯 Что бот делает БЕЗ API ключей

### ✅ Работает:

- ✅ Получение реальных цен токенов (BTC, ETH, и т.д.)
- ✅ Анализ рынка через ML модели
- ✅ Открытие/закрытие виртуальных позиций
- ✅ Расчёт PnL (прибыль/убыток)
- ✅ Статистика торговли (Win Rate, Sharpe Ratio, и т.д.)
- ✅ Trailing Stop, Black Swan Protection, Night Mode
- ✅ Telegram уведомления (если настроены)
- ✅ Логирование всех операций

### ❌ НЕ работает (требует API ключи):

- ❌ Реальные ордера на Binance
- ❌ Вывод средств
- ❌ Изменение leverage на вашем аккаунте
- ❌ Доступ к балансу вашего аккаунта

**Но это и не нужно для paper trading!** 🎉

---

## 🆘 Проблемы и решения

### Проблема: "ConnectionError: Failed to get price"

**Причина:** Не удалось подключиться к Binance API

**Решение:**

1. Проверьте интернет соединение
2. Убедитесь что Binance.com доступен в вашей стране
3. Проверьте firewall/антивирус
4. Попробуйте другой DNS (Google: 8.8.8.8)

---

### Проблема: "ImportError: No module named 'requests'"

**Причина:** Зависимости не установлены

**Решение:**

```batch
REM Windows
venv\Scripts\activate.bat
pip install -r requirements.txt
```

```bash
# Linux / macOS
source venv/bin/activate
pip install -r requirements.txt
```

---

### Проблема: Бот показывает нулевой баланс

**Причина:** Проверьте `.env` файл

**Решение:**

Убедитесь что в `.env`:

```ini
PAPER_TRADING_MODE=true
PAPER_INITIAL_BALANCE=10000
```

Перезапустите бота.

---

## 📚 Дополнительные ресурсы

- **QUICK_START_WINDOWS.md** - Быстрый старт для Windows
- **QUICK_START.md** - Быстрый старт для Linux/macOS
- **README.md** - Главная документация
- **INSTALLATION_GUIDE_WINDOWS.md** - Полное руководство (Windows)
- **INSTALLATION_GUIDE.md** - Полное руководство (Linux/macOS)

---

## ✨ Итого

### Для paper trading вам НЕ нужны:

❌ API ключи Binance
❌ Аккаунт на Binance
❌ Верификация KYC
❌ Реальные деньги
❌ Testnet access

### Вам нужно только:

✅ Python 3.9+
✅ Интернет соединение
✅ Запустить `install.bat` или `./install.sh`
✅ Запустить `start_bot.bat` или `./start_bot.sh`

**Готово!** 🎉🚀

---

**Версия:** 1.0
**Дата:** 2025-11-06
**Совместимость:** Windows Server 2012 R2, Windows 10/11, Linux, macOS
