# 📋 Шпаргалка команд для Windows Command Prompt

## 🔴 Удаление старого бота

### Вариант 1: Автоматический (Рекомендуется)

```batch
REM Скачайте и запустите скрипт очистки
cleanup_old_bot.bat
```

### Вариант 2: Вручную

```batch
REM 1. Остановить Python процессы
taskkill /F /IM python.exe
taskkill /F /IM pythonw.exe

REM 2. Перейти в папку старого бота
cd C:\path\to\old_bot

REM 3. Удалить виртуальное окружение
if exist venv rmdir /S /Q venv
if exist .venv rmdir /S /Q .venv
if exist env rmdir /S /Q env

REM 4. Очистить кэш pip
python -m pip cache purge

REM 5. (Опционально) Удалить PostgreSQL библиотеки глобально
python -m pip uninstall -y psycopg2-binary psycopg2 asyncpg
```

---

## 🟢 Установка нового бота

### Быстрая установка

```batch
REM 1. Перейти в папку нового бота
cd C:\path\to\jjhbn\GGG3

REM 2. Запустить автоматическую установку
install.bat

REM 3. Дождаться завершения (5-10 минут)
```

### Ручная установка

```batch
REM 1. Создать виртуальное окружение
python -m venv venv

REM 2. Активировать окружение
venv\Scripts\activate.bat

REM 3. Обновить pip
python -m pip install --upgrade pip

REM 4. Установить зависимости
pip install -r requirements.txt

REM 5. Создать .env файл
copy .env.example .env

REM 6. Настроить .env
notepad .env
```

---

## 🚀 Запуск бота

### Автоматический запуск

```batch
REM Интерактивный режим (с логами на экране)
start_bot.bat
REM Выберите: 1

REM Фоновый режим (работает в фоне)
start_bot.bat
REM Выберите: 2
```

### Ручной запуск

```batch
REM 1. Активировать окружение
venv\Scripts\activate.bat

REM 2. Запустить бота
python main.py

REM Или в фоновом режиме
start /B python main.py > logs\bot.log 2>&1
```

---

## 🛑 Остановка бота

### Автоматическая остановка

```batch
REM Мягкая остановка (рекомендуется)
stop_bot.bat
REM Выберите: 1

REM Принудительная остановка
stop_bot.bat
REM Выберите: 2
```

### Ручная остановка

```batch
REM Найти PID процесса
tasklist | findstr python

REM Остановить по PID
taskkill /PID <PID> /F

REM Или остановить все Python процессы
taskkill /F /IM python.exe
```

---

## 📊 Мониторинг

### Проверка статуса

```batch
REM Полная информация о боте
status_bot.bat

REM Просмотр логов в реальном времени
tail_logs.bat

REM Открыть файл логов
notepad logs\bot.log
```

### Проверка процессов

```batch
REM Список всех Python процессов
tasklist | findstr python

REM Детальная информация о процессе
wmic process where "name='python.exe'" get ProcessId,CommandLine
```

---

## ⚙️ Конфигурация

### Редактирование .env

```batch
REM Открыть в блокноте
notepad .env

REM Или в другом редакторе
start .env
```

### Основные настройки для Paper Trading

```ini
# Виртуальная торговля (рекомендуется для начала)
PAPER_TRADING_MODE=true
PAPER_INITIAL_BALANCE=10000

# Реальные данные рынка (НЕ testnet!)
USE_TESTNET=false

# API ключи (НЕ НУЖНЫ для paper trading!)
BINANCE_API_KEY=
BINANCE_SECRET_KEY=

# Торговая пара
SYMBOL=BTCUSDT
```

---

## 🧹 Очистка и обслуживание

### Очистка логов

```batch
REM Удалить все логи
del /Q logs\*.log

REM Удалить старые логи (старше 7 дней)
forfiles /P logs /S /M *.log /D -7 /C "cmd /c del @path"
```

### Очистка кэша

```batch
REM Очистить кэш pip
python -m pip cache purge

REM Очистить __pycache__
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
```

### Переустановка зависимостей

```batch
REM 1. Активировать окружение
venv\Scripts\activate.bat

REM 2. Переустановить все зависимости
pip install --force-reinstall -r requirements.txt
```

---

## 🐛 Диагностика проблем

### Проверка системы

```batch
REM Версия Python
python --version

REM Версия pip
python -m pip --version

REM Список установленных пакетов
pip list

REM Проверка PATH
echo %PATH%

REM Информация о системе
systeminfo | findstr /B /C:"OS Name" /C:"OS Version"
```

### Проверка зависимостей

```batch
REM Проверить конкретный пакет
pip show ccxt

REM Проверить все зависимости из requirements.txt
pip check

REM Найти устаревшие пакеты
pip list --outdated
```

### Проверка подключения к Binance API

```batch
REM Активировать окружение
venv\Scripts\activate.bat

REM Запустить тест
python -c "import ccxt; exchange = ccxt.binance(); print(exchange.fetch_ticker('BTC/USDT'))"
```

---

## 📦 Обновление бота

### Получить последние изменения

```batch
REM 1. Остановить бота
stop_bot.bat

REM 2. Получить обновления
git pull

REM 3. Обновить зависимости
venv\Scripts\activate.bat
pip install --upgrade -r requirements.txt

REM 4. Перезапустить бота
start_bot.bat
```

---

## 🔧 Работа с виртуальным окружением

### Активация/Деактивация

```batch
REM Активировать
venv\Scripts\activate.bat

REM Деактивировать
deactivate
```

### Пересоздание окружения

```batch
REM 1. Удалить старое окружение
if exist venv rmdir /S /Q venv

REM 2. Создать новое
python -m venv venv

REM 3. Активировать
venv\Scripts\activate.bat

REM 4. Установить зависимости
pip install -r requirements.txt
```

---

## 📁 Навигация по файлам

### Структура проекта

```batch
REM Показать структуру
tree /F /A

REM Показать только директории
tree /A

REM Список файлов в текущей папке
dir

REM Список файлов с деталями
dir /Q
```

### Важные файлы и папки

```
jjhbn\GGG3\
├── main.py                          # Главный файл бота
├── .env                             # Конфигурация (создаётся из .env.example)
├── .env.example                     # Шаблон конфигурации
├── requirements.txt                 # Зависимости Python
├── install.bat                      # Скрипт установки
├── start_bot.bat                    # Скрипт запуска
├── stop_bot.bat                     # Скрипт остановки
├── status_bot.bat                   # Проверка статуса
├── cleanup_old_bot.bat              # Удаление старого бота
├── CLEANUP_OLD_BOT_WINDOWS.md       # Инструкция по очистке
├── QUICK_START_WINDOWS.md           # Быстрый старт
├── README_NO_API_KEYS.md            # Работа без API ключей
├── logs\                            # Папка с логами
│   └── bot.log                      # Основной лог
├── models\                          # ML модели
├── paper_trading\                   # Модуль paper trading
│   ├── paper_exchange.py            # Публичные API Binance
│   └── ...
└── venv\                            # Виртуальное окружение (создаётся install.bat)
```

---

## 💡 Полезные команды Windows

### Управление процессами

```batch
REM Список всех процессов
tasklist

REM Найти процесс по имени
tasklist | findstr <имя>

REM Убить процесс по PID
taskkill /PID <PID> /F

REM Убить процесс по имени
taskkill /IM <имя.exe> /F
```

### Работа с файлами

```batch
REM Создать папку
mkdir <имя_папки>

REM Удалить папку (с подтверждением)
rmdir <имя_папки>

REM Удалить папку рекурсивно (без подтверждения)
rmdir /S /Q <имя_папки>

REM Копировать файл
copy <источник> <назначение>

REM Переместить файл
move <источник> <назначение>

REM Удалить файл
del <имя_файла>
```

### Просмотр файлов

```batch
REM Показать содержимое файла
type <имя_файла>

REM Показать первые 10 строк
type <имя_файла> | more

REM Открыть в блокноте
notepad <имя_файла>
```

---

## 🆘 Частые проблемы и решения

### "python не распознаётся"

```batch
REM Найти путь к Python
where python

REM Если не найден, проверить установку
C:\Python38\python.exe --version

REM Добавить в PATH (замените путь на свой)
set PATH=%PATH%;C:\Python38;C:\Python38\Scripts
```

### "Отказано в доступе"

```batch
REM 1. Закрыть все программы, использующие Python
taskkill /F /IM python.exe

REM 2. Запустить Command Prompt от администратора
REM Win + X -> Command Prompt (Admin)

REM 3. Повторить команду
```

### "Не удается найти указанный файл"

```batch
REM Проверить текущую папку
cd

REM Перейти в правильную папку
cd C:\path\to\jjhbn\GGG3

REM Проверить наличие файла
dir install.bat
```

### Бот не запускается

```batch
REM 1. Проверить логи
type logs\bot.log

REM 2. Проверить конфигурацию
type .env

REM 3. Проверить зависимости
venv\Scripts\activate.bat
pip check

REM 4. Переустановить зависимости
pip install --force-reinstall -r requirements.txt
```

---

## 📚 Дополнительная информация

### Документация

- **Быстрый старт:** `QUICK_START_WINDOWS.md`
- **Работа без API ключей:** `README_NO_API_KEYS.md`
- **Очистка старого бота:** `CLEANUP_OLD_BOT_WINDOWS.md`
- **Установка:** `INSTALL_GUIDE_WINDOWS.md`
- **Основной README:** `README.md`

### Открыть документацию

```batch
REM В блокноте
notepad QUICK_START_WINDOWS.md

REM В браузере (если Markdown viewer установлен)
start QUICK_START_WINDOWS.md
```

---

## 🎯 Краткая памятка для начинающих

### Первый запуск

```batch
1. cd C:\path\to\jjhbn\GGG3
2. install.bat
3. start_bot.bat
```

### Ежедневное использование

```batch
REM Запустить
start_bot.bat

REM Проверить статус
status_bot.bat

REM Посмотреть логи
tail_logs.bat

REM Остановить
stop_bot.bat
```

### При проблемах

```batch
REM Переустановить
cleanup_old_bot.bat
install.bat

REM Проверить логи
notepad logs\bot.log

REM Проверить конфигурацию
notepad .env
```

---

**Удачи в трейдинге! 🚀📈**
