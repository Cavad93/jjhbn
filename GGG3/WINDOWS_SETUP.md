# 🪟 Установка и запуск бота на Windows Server 2019

Подробная инструкция по установке и настройке Binance Trading Bot с интеграцией Haiku 4.5 на Windows Server 2019.

---

## 📋 Содержание

- [Требования](#требования)
- [Установка Python](#установка-python)
- [Установка Git](#установка-git)
- [Клонирование проекта](#клонирование-проекта)
- [Установка зависимостей](#установка-зависимостей)
- [Настройка конфигурации](#настройка-конфигурации)
- [Первый запуск](#первый-запуск)
- [Автозапуск при старте системы](#автозапуск-при-старте-системы)
- [Мониторинг и логи](#мониторинг-и-логи)
- [Troubleshooting](#troubleshooting)

---

## 💻 Требования

### Минимальные системные требования

- **ОС:** Windows Server 2019 (64-bit)
- **RAM:** 4 GB (рекомендуется 8 GB)
- **Диск:** 10 GB свободного места
- **Интернет:** Стабильное подключение (для API запросов к Binance и Anthropic)
- **Права:** Администратор (для установки софта)

### Необходимый софт

1. **Python 3.9+** (рекомендуется 3.11)
2. **Git** (для клонирования репозитория)
3. **PowerShell 5.1+** (встроен в Windows Server 2019)

---

## 🐍 Установка Python

### Шаг 1: Скачать Python

1. Откройте браузер (Edge/Chrome)
2. Перейдите на https://www.python.org/downloads/
3. Скачайте **Python 3.11.x** (Windows installer 64-bit)

### Шаг 2: Установить Python

1. Запустите скачанный `.exe` файл **от имени Администратора**
2. ✅ **ВАЖНО:** Отметьте галочку **"Add Python to PATH"**
3. Нажмите **"Install Now"**
4. Дождитесь завершения установки
5. Нажмите **"Close"**

### Шаг 3: Проверить установку

Откройте **PowerShell** (Win + X → Windows PowerShell (Admin)) и выполните:

```powershell
python --version
```

Должно вывести:
```
Python 3.11.x
```

Если команда не найдена, перезагрузите PowerShell или добавьте Python в PATH вручную.

---

## 📦 Установка Git

### Шаг 1: Скачать Git

1. Перейдите на https://git-scm.com/download/win
2. Скачайте **64-bit Git for Windows Setup**

### Шаг 2: Установить Git

1. Запустите установщик **от имени Администратора**
2. Используйте настройки по умолчанию:
   - ✅ Git from the command line and also from 3rd-party software
   - ✅ Use bundled OpenSSH
   - ✅ Use the OpenSSL library
   - ✅ Checkout Windows-style, commit Unix-style line endings
3. Нажмите **"Install"**
4. Дождитесь завершения
5. Нажмите **"Finish"**

### Шаг 3: Проверить установку

В PowerShell выполните:

```powershell
git --version
```

Должно вывести:
```
git version 2.x.x
```

---

## 📥 Клонирование проекта

### Шаг 1: Создать директорию для проекта

Откройте PowerShell и выполните:

```powershell
# Создать папку для проектов (например, в C:\Projects)
New-Item -Path "C:\Projects" -ItemType Directory -Force
cd C:\Projects
```

### Шаг 2: Клонировать репозиторий

```powershell
git clone https://github.com/Cavad93/jjhbn.git
cd jjhbn\GGG3
```

### Шаг 3: Переключиться на нужную ветку

```powershell
git checkout claude/code-analysis-russian-011CUvYwiHhugYL3qHzmwfe5
```

Или клонировать конкретную ветку сразу:

```powershell
git clone -b claude/code-analysis-russian-011CUvYwiHhugYL3qHzmwfe5 https://github.com/Cavad93/jjhbn.git
cd jjhbn\GGG3
```

---

## 📚 Установка зависимостей

### Шаг 1: Обновить pip

```powershell
python -m pip install --upgrade pip
```

### Шаг 2: Установить зависимости

```powershell
# Основные зависимости
pip install -r requirements.txt
```

**Ожидаемое время:** 2-5 минут (зависит от скорости интернета)

### Шаг 3: Проверить установку

```powershell
# Проверить основные пакеты
python -c "import numpy, pandas, anthropic; print('✅ All packages installed')"
```

Если видите `✅ All packages installed` — всё установлено корректно.

### Возможные проблемы

#### ❌ Ошибка: "Microsoft Visual C++ 14.0 is required"

**Решение:** Установить **Microsoft C++ Build Tools**

1. Скачайте: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Запустите установщик
3. Выберите **"Desktop development with C++"**
4. Нажмите **"Install"**
5. После установки повторите `pip install -r requirements.txt`

#### ❌ Ошибка: "Failed building wheel for TA-Lib"

**Решение:** TA-Lib не используется (удалён из requirements.txt), поэтому эта ошибка не должна возникать.

---

## ⚙️ Настройка конфигурации

### Шаг 1: Создать .env файл

```powershell
# Скопировать шаблон
Copy-Item .env.example .env

# Открыть в Notepad
notepad .env
```

### Шаг 2: Настроить Paper Trading (рекомендуется для начала)

Оставьте настройки по умолчанию:

```ini
# Режим paper trading (виртуальные деньги)
PAPER_TRADING_MODE=true

# Начальный баланс
PAPER_INITIAL_BALANCE=1000

# API ключи НЕ НУЖНЫ для paper trading
BINANCE_API_KEY=
BINANCE_SECRET_KEY=

# Testnet отключён (используем реальные данные рынка)
USE_TESTNET=false
```

### Шаг 3: Настроить Haiku 4.5 (опционально)

Если хотите использовать фундаментальный анализ с Haiku 4.5:

1. Зарегистрируйтесь на https://console.anthropic.com/
2. Создайте API ключ в **Settings → API Keys**
3. Скопируйте ключ (формат: `sk-ant-api...`)
4. Добавьте в `.env`:

```ini
# Haiku 4.5 для фундаментального анализа
ANTHROPIC_API_KEY=sk-ant-api-ваш-ключ-здесь
```

**Стоимость:** ~$6-12/месяц (20 монет × 6 раз/день)

### Шаг 4: Настроить Telegram уведомления (опционально)

1. Создайте бота: https://t.me/BotFather → `/newbot`
2. Получите токен (формат: `123456789:ABCdefGHI...`)
3. Получите Chat ID: https://t.me/userinfobot → `/start`
4. Добавьте в `.env`:

```ini
# Telegram уведомления
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

### Шаг 5: Сохранить .env

Нажмите **Ctrl + S** в Notepad, затем закройте файл.

---

## 🚀 Первый запуск

### Шаг 1: Запустить бота в тестовом режиме

```powershell
# Убедитесь что вы в директории GGG3
cd C:\Projects\jjhbn\GGG3

# Запустить бота
python binance_bot_main.py --paper
```

### Шаг 2: Проверить вывод

Вы должны увидеть:

```
================================================================================
🤖 INITIALIZING BINANCE TRADING BOT
================================================================================
  Mode: 📄 PAPER TRADING (New: $1000.0)
  Loaded 0 open positions

  Loading ML models...
  ✅ XGBoost loaded (trained: 2024-11-05)
  ✅ Random Forest loaded (trained: 2024-11-05)
  ✅ Neural Network loaded (trained: 2024-11-05)
  ✅ META Neural CEM loaded

  Initializing risk managers...
  Reinvestment: Enabled (50% profit → reserve)
  Capital tracker: Enabled
  Haiku 4.5 Analyzer: Enabled (batch_size=5)  ← Если настроен API ключ
  # ИЛИ
  Haiku 4.5 Analyzer: Disabled (no API key)   ← Если НЕ настроен

================================================================================
🚀 STARTING BINANCE TRADING BOT
================================================================================
```

### Шаг 3: Дождаться первой ребалансировки

Бот работает по циклу:
- **Каждые 4 часа:** Ребалансировка портфеля (поиск TOP-10 монет)
- **Каждые 5 минут:** Проверка открытых позиций (TP/SL)

Первая ребалансировка произойдёт в ближайший 4-часовой интервал:
- **00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC**

Или можно дождаться первой проверки позиций (через 5 минут).

### Шаг 4: Остановить бота (для тестов)

Нажмите **Ctrl + C** в PowerShell для остановки.

---

## 🔄 Автозапуск при старте системы

### Вариант 1: Task Scheduler (рекомендуется)

#### Шаг 1: Создать bat-файл запуска

Создайте файл `C:\Projects\jjhbn\GGG3\start_bot.bat`:

```powershell
notepad C:\Projects\jjhbn\GGG3\start_bot.bat
```

Содержимое:

```batch
@echo off
cd /d C:\Projects\jjhbn\GGG3
python binance_bot_main.py --paper
pause
```

Сохраните и закройте.

#### Шаг 2: Открыть Task Scheduler

1. Нажмите **Win + R**
2. Введите: `taskschd.msc`
3. Нажмите **Enter**

#### Шаг 3: Создать задачу

1. Нажмите **Create Task** (в правой панели)
2. **General** вкладка:
   - Name: `Binance Trading Bot`
   - Description: `Auto-start trading bot on system boot`
   - ✅ Run whether user is logged on or not
   - ✅ Run with highest privileges
   - Configure for: `Windows Server 2019`

3. **Triggers** вкладка:
   - Нажмите **New**
   - Begin the task: **At startup**
   - Delay task for: `1 minute` (дать системе загрузиться)
   - ✅ Enabled
   - Нажмите **OK**

4. **Actions** вкладка:
   - Нажмите **New**
   - Action: **Start a program**
   - Program/script: `C:\Projects\jjhbn\GGG3\start_bot.bat`
   - Start in: `C:\Projects\jjhbn\GGG3`
   - Нажмите **OK**

5. **Conditions** вкладка:
   - ❌ Снять галочку "Start the task only if the computer is on AC power"
   - ✅ Wake the computer to run this task (если нужно)

6. **Settings** вкладка:
   - ✅ Allow task to be run on demand
   - ✅ Run task as soon as possible after a scheduled start is missed
   - If the task fails, restart every: `1 minute`
   - Attempt to restart up to: `3 times`
   - If the running task does not end when requested, force it to stop: ❌

7. Нажмите **OK**

8. Введите пароль администратора (если запрашивается)

#### Шаг 4: Протестировать задачу

1. В Task Scheduler найдите задачу **Binance Trading Bot**
2. Правый клик → **Run**
3. Проверьте что бот запустился (откройте логи или Task Manager)

### Вариант 2: Windows Service (продвинутый)

Для запуска бота как Windows Service используйте **NSSM** (Non-Sucking Service Manager):

1. Скачайте NSSM: https://nssm.cc/download
2. Распакуйте в `C:\nssm`
3. Откройте PowerShell от Администратора:

```powershell
cd C:\nssm\win64
.\nssm.exe install BinanceTradingBot
```

4. В GUI укажите:
   - **Path:** `C:\Python311\python.exe` (путь к python.exe)
   - **Startup directory:** `C:\Projects\jjhbn\GGG3`
   - **Arguments:** `binance_bot_main.py --paper`
   - **Service name:** `BinanceTradingBot`

5. Нажмите **Install service**

6. Запустить сервис:

```powershell
Start-Service BinanceTradingBot
```

7. Проверить статус:

```powershell
Get-Service BinanceTradingBot
```

---

## 📊 Мониторинг и логи

### Просмотр логов

Логи сохраняются в:

```
C:\Projects\jjhbn\GGG3\logs\bot.log
```

#### Просмотр в реальном времени (PowerShell):

```powershell
Get-Content C:\Projects\jjhbn\GGG3\logs\bot.log -Wait -Tail 50
```

#### Просмотр последних 100 строк:

```powershell
Get-Content C:\Projects\jjhbn\GGG3\logs\bot.log -Tail 100
```

#### Поиск ошибок:

```powershell
Select-String -Path C:\Projects\jjhbn\GGG3\logs\bot.log -Pattern "ERROR|EXCEPTION"
```

### Проверка процесса

#### Найти процесс бота:

```powershell
Get-Process | Where-Object {$_.ProcessName -eq "python"}
```

#### Проверить использование CPU/RAM:

```powershell
Get-Process python | Select-Object Name, CPU, WorkingSet
```

### Статус бота

Откройте:

```
C:\Projects\jjhbn\GGG3\data\paper_exchange_state.json
```

Содержит текущий баланс и состояние paper trading.

### Dashboard (опционально)

Запустить веб-dashboard:

```powershell
cd C:\Projects\jjhbn\GGG3\paper_trading
python paper_dashboard.py
```

Откройте в браузере: http://localhost:8050

---

## 🔧 Troubleshooting

### ❌ Ошибка: "No module named 'anthropic'"

**Причина:** Не установлены зависимости для Haiku 4.5

**Решение:**

```powershell
pip install anthropic feedparser
```

### ❌ Ошибка: "ANTHROPIC_API_KEY not found"

**Причина:** Не настроен API ключ или .env не загружен

**Решение:**

1. Проверьте что файл `.env` существует:

```powershell
Test-Path C:\Projects\jjhbn\GGG3\.env
```

2. Проверьте содержимое:

```powershell
Get-Content C:\Projects\jjhbn\GGG3\.env | Select-String "ANTHROPIC"
```

3. Если ключ не нужен, отключите Haiku в `binance_config.py`:

```python
HAIKU_ANALYSIS_ENABLED = False
```

### ❌ Бот падает сразу после запуска

**Причина:** Возможно отсутствуют модели ML

**Решение:**

Проверьте наличие моделей:

```powershell
dir C:\Projects\jjhbn\GGG3\models
```

Должны быть файлы:
- `xgb_model.json`
- `rf_model.pkl`
- `meta_model.pth`

Если файлов нет, запустите обучение:

```powershell
python training\train_binance_experts.py
python training\train_binance_meta.py
```

### ❌ Ошибка: "Failed to connect to Binance API"

**Причина:** Проблемы с интернетом или Binance недоступен

**Решение:**

1. Проверьте интернет:

```powershell
Test-Connection binance.com -Count 4
```

2. Проверьте firewall:

```powershell
Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*Python*"}
```

3. Добавьте правило для Python:

```powershell
New-NetFirewallRule -DisplayName "Python 3.11" -Direction Outbound -Program "C:\Python311\python.exe" -Action Allow
```

### ❌ Высокое использование CPU

**Причина:** Бот использует ML модели, которые требуют вычислений

**Нормальное использование:**
- При ребалансировке (каждые 4 часа): 20-50% CPU (1-2 минуты)
- В остальное время: 1-5% CPU

**Если постоянно высокое:**
- Проверьте логи на ошибки (возможно бот в цикле)
- Перезапустите бота

### ❌ Бот не открывает позиции

**Причины:**

1. **Адаптивный порог слишком высокий:**
   - Проверьте в логах: `Adaptive threshold: X.XXXX`
   - Если > 0.75, это нормально в начале (холодный старт)

2. **Все монеты не проходят фильтры:**
   - Проверьте в логах сколько монет прошло фильтры
   - Нормально: 20-50 монет из ~400

3. **Night Mode активен:**
   - Проверьте время (ночью позиций меньше)

4. **Haiku отклонил все монеты:**
   - Проверьте логи Haiku: `Rejected: ...`

### ❌ Ошибка: "Permission denied" при запуске

**Причина:** Недостаточно прав

**Решение:**

Запустите PowerShell **от имени Администратора**:

1. Win + X
2. **Windows PowerShell (Admin)**
3. Повторите команду

---

## 🔒 Безопасность

### Рекомендации для Windows Server 2019

1. **Firewall:**
   - Разрешить исходящие соединения для Python
   - Заблокировать входящие (бот не требует)

2. **Антивирус:**
   - Добавить `C:\Projects\jjhbn` в исключения
   - Python может ложно срабатывать как вирус

3. **Обновления:**
   - Включить автоматические обновления Windows
   - Регулярно обновлять Python: `python -m pip install --upgrade pip`

4. **Backup:**
   - Регулярно бэкапить директорию:

```powershell
# Создать backup
Compress-Archive -Path C:\Projects\jjhbn\GGG3\data -DestinationPath C:\Backups\bot_data_$(Get-Date -Format 'yyyy-MM-dd').zip

# Автоматический backup (Task Scheduler, ежедневно)
```

5. **API ключи:**
   - Хранить `.env` в безопасном месте
   - НЕ коммитить `.env` в Git
   - Использовать только Read права для Binance API

---

## 📚 Полезные команды PowerShell

```powershell
# Обновить код из Git
cd C:\Projects\jjhbn\GGG3
git pull

# Обновить зависимости
pip install -r requirements.txt --upgrade

# Очистить кэш Python
py -m pip cache purge

# Проверить размер логов
Get-ChildItem C:\Projects\jjhbn\GGG3\logs -Recurse | Measure-Object -Property Length -Sum

# Очистить старые логи (старше 30 дней)
Get-ChildItem C:\Projects\jjhbn\GGG3\logs -Recurse | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-30)} | Remove-Item

# Перезапустить сервис (если используете NSSM)
Restart-Service BinanceTradingBot

# Просмотреть переменные окружения
Get-Content C:\Projects\jjhbn\GGG3\.env

# Экспортировать статистику
Copy-Item C:\Projects\jjhbn\GGG3\data\performance_metrics.json C:\Backups\
```

---

## 🎓 Дополнительные ресурсы

- **Основная документация:** `C:\Projects\jjhbn\GGG3\README.md`
- **Haiku интеграция:** `C:\Projects\jjhbn\GGG3\HAIKU_INTEGRATION.md`
- **Результаты тестов:** `C:\Projects\jjhbn\GGG3\HAIKU_TEST_REPORT.md`
- **Telegram канал:** (если есть)
- **GitHub Issues:** https://github.com/Cavad93/jjhbn/issues

---

## ✅ Чеклист запуска

- [ ] Python 3.11+ установлен
- [ ] Git установлен
- [ ] Проект склонирован
- [ ] Зависимости установлены (`pip install -r requirements.txt`)
- [ ] Файл `.env` создан и настроен
- [ ] Бот запускается без ошибок
- [ ] Логи пишутся в `logs/bot.log`
- [ ] Task Scheduler настроен (для автозапуска)
- [ ] Firewall разрешает Python
- [ ] Backup настроен

---

**Готово! Бот настроен и работает на Windows Server 2019** 🎉

**Автор:** Claude Code
**Дата:** 2025-11-08
**Версия:** 1.0
