# ⚡ Быстрый старт для Windows - 5 минут до запуска

Минимальная инструкция для быстрого запуска бота в режиме paper trading на Windows Server 2012 R2 / Windows 10/11.

---

## 🎯 Требования

- **Windows**: Server 2012 R2, Windows 10, или Windows 11
- **Python**: 3.9 или выше
- **PowerShell**: 3.0 или выше (встроен в Windows)
- **Интернет**: для установки зависимостей

---

## 🚀 Запуск за 5 команд

### 1. Откройте Command Prompt (cmd.exe)

```batch
REM Нажмите Win+R, введите "cmd" и нажмите Enter
REM Или найдите "Command Prompt" в меню Пуск
```

### 2. Перейдите в директорию проекта

```batch
cd C:\path\to\jjhbn\GGG3
```

### 3. Запустите автоматическую установку

```batch
install.bat
```

Это выполнит:
- ✅ Проверку Python
- ✅ Очистку старого окружения
- ✅ Создание виртуального окружения
- ✅ Установку всех зависимостей
- ✅ Настройку директорий

**Время**: 5-10 минут

### 4. Настройте .env файл

```batch
REM Откройте .env в Блокноте
notepad .env
```

**Минимальная конфигурация** (для начала):

```ini
# Обязательные настройки:
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here
USE_TESTNET=true

# Опциональные (можно оставить пустыми):
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
PAPER_TRADING_MODE=true
PAPER_INITIAL_BALANCE=10000
```

**Сохраните файл**: Файл → Сохранить, затем закройте Блокнот

### 5. Запустите бота

```batch
start_bot.bat
```

Или запустите в фоновом режиме:

```batch
start_bot.bat --background
```

---

## ✅ Проверка работы

После запуска вы должны увидеть:

```
================================================================================
🤖 INITIALIZING BINANCE TRADING BOT
================================================================================
  Mode: 📄 PAPER TRADING ($10000)
  Loaded 0 open positions

  Loading ML models...
    ✓ XGBoost
    ✓ RandomForest
    ✓ NeuralNet

✅ Bot initialized successfully
```

Если видите эти сообщения - **бот работает! 🎉**

---

## 📊 Мониторинг

### Проверка статуса

```batch
status_bot.bat
```

Покажет:
- ✅ Статус процесса (работает/остановлен)
- ✅ Использование CPU и памяти
- ✅ Открытые позиции
- ✅ Последние логи

### Просмотр логов в реальном времени

```batch
tail_logs.bat
```

Или в отдельном окне PowerShell:

```powershell
Get-Content logs\binance_bot.log -Wait -Tail 20
```

---

## 🛑 Остановка бота

### Интерактивный режим

Просто нажмите `Ctrl+C` в окне где запущен бот.

### Фоновый режим

```batch
stop_bot.bat
```

---

## 🔧 Полезные команды

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

REM Справка
start_bot.bat --help
```

---

## 🆘 Быстрое решение проблем

### Проблема: Python не найден

**Ошибка:**
```
'python' is not recognized as an internal or external command
```

**Решение:**

1. Скачайте Python: https://www.python.org/downloads/
2. При установке **обязательно** отметьте "Add Python to PATH"
3. Перезапустите Command Prompt
4. Проверьте: `python --version`

---

### Проблема: pip не работает

**Ошибка:**
```
'pip' is not recognized...
```

**Решение:**

```batch
python -m pip --version
python -m pip install --upgrade pip
```

---

### Проблема: Binance API ошибка

**Ошибка:**
```
BinanceAPIException: Invalid API-key
```

**Решение:**

1. Проверьте `.env` файл:
   ```batch
   type .env | findstr BINANCE
   ```

2. Убедитесь что ключи правильно скопированы (без пробелов)

3. Проверьте права API на Binance:
   - Enable Futures ✅
   - Enable Reading ✅
   - Enable Spot & Margin Trading ✅

---

### Проблема: Microsoft Visual C++ ошибка

**Ошибка:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Решение:**

Установите Microsoft C++ Build Tools:

1. Скачайте: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Установите "Desktop development with C++"
3. Перезапустите установку бота: `install.bat`

---

### Проблема: Доступ запрещен (Permission denied)

**Ошибка:**
```
PermissionError: [Errno 13] Permission denied
```

**Решение:**

Запустите Command Prompt от имени администратора:

1. Найдите "Command Prompt" в меню Пуск
2. Правый клик → "Run as administrator"
3. Повторите установку

---

## 📚 Документация

- **[INSTALLATION_GUIDE_WINDOWS.md](INSTALLATION_GUIDE_WINDOWS.md)** - Полное руководство для Windows
- **[README.md](README.md)** - Описание проекта

---

## 🎯 Важные напоминания

### ⚠️ Безопасность

1. ✅ Начните с **Paper Trading** режима
2. ✅ Тестируйте минимум **1-2 недели**
3. ✅ Не делитесь API ключами ни с кем
4. ✅ Используйте `USE_TESTNET=true` для тестирования
5. ❌ НЕ используйте live trading без опыта!

### 📊 Мониторинг

1. Проверяйте статус ежедневно: `status_bot.bat`
2. Следите за логами: `tail_logs.bat`
3. Настройте Telegram уведомления для мобильных алертов

### 💰 Капитал

**Paper Trading** (виртуальные деньги):
- Начальный капитал: $10,000
- Без реальных рисков
- Идеально для обучения

**Live Trading** (реальные деньги):
- Начните с малого: $100-500
- Убедитесь что Win Rate > 55%
- Мониторьте ежедневно

---

## 🚀 Следующие шаги

1. **Протестируйте 1-2 недели** в paper trading
2. **Настройте Telegram** для уведомлений
3. **Анализируйте результаты** через `status_bot.bat`
4. **Читайте полное руководство**: `type INSTALLATION_GUIDE_WINDOWS.md`
5. **Переходите на live** только после успешного тестирования

---

## 📞 Поддержка

### Проверка логов

```batch
REM Последние 50 строк
powershell Get-Content logs\binance_bot.log -Tail 50

REM Поиск ошибок
findstr /i "error" logs\binance_bot.log

REM Поиск открытия позиций
findstr /i "ENTRY" logs\binance_bot.log
```

### Проверка конфигурации

```batch
python test_binance_config.py
```

### Запуск тестов

```batch
REM Активируйте окружение
venv\Scripts\activate.bat

REM Запустите быстрый тест
python tests\quick_bot_test.py
```

---

**Готово! Теперь бот установлен и запущен на Windows.** 🎉

**Удачной торговли! 🚀📈**

---

**Версия**: 1.0 для Windows
**Совместимость**: Windows Server 2012 R2, Windows 10, Windows 11
**Python**: 3.9+
**Последнее обновление**: 2025-11-06
