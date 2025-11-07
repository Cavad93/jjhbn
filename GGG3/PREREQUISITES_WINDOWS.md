# 🔧 Предварительные требования для Windows Server 2012 R2

## 📋 Что нужно установить ПЕРЕД запуском install.bat

---

## ✅ ОБЯЗАТЕЛЬНО (для работы бота)

### 1. **Python 3.8 или выше**

**Проверка:**
```batch
python --version
```

Должно быть: `Python 3.8.x` или выше (рекомендуется 3.9-3.11)

**Если не установлен:**

#### Вариант A: Установка с python.org
1. Скачайте Python с https://www.python.org/downloads/
2. Выберите версию **3.9.13** или **3.10.11** (стабильные для Windows Server 2012 R2)
3. При установке **ОБЯЗАТЕЛЬНО**:
   - ✅ Поставьте галочку "Add Python to PATH"
   - ✅ Выберите "Install for all users"
   - ✅ Выберите "Customize installation"
   - ✅ Включите "pip" и "py launcher"

4. Перезагрузите Command Prompt после установки

**Проверка после установки:**
```batch
python --version
pip --version
```

#### Вариант B: Установка через Chocolatey (если установлен)
```batch
choco install python39 -y
```

---

### 2. **pip (обычно идёт с Python)**

**Проверка:**
```batch
pip --version
```

**Если не установлен:**
```batch
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

---

### 3. **Git (опционально, но рекомендуется)**

**Проверка:**
```batch
git --version
```

**Если не установлен:**
1. Скачайте с https://git-scm.com/download/win
2. Установите с настройками по умолчанию

Или:
```batch
choco install git -y
```

---

## ❌ НЕ НУЖНО (для Paper Trading)

### ❌ **PostgreSQL 12** - НЕ ТРЕБУЕТСЯ!

**ВАЖНО:** Новый бот **НЕ использует PostgreSQL** для paper trading!

- ✅ Старый бот (PancakeSwap) → PostgreSQL 12
- ✅ **Новый бот (Binance)** → **БЕЗ PostgreSQL!**

**Что использует новый бот:**
- JSON файлы для хранения позиций (`portfolio/positions.json`)
- JSON файлы для состояния (`portfolio/reinvestment_state.json`)
- Pickle файлы для ML моделей (`models/saved/*.pkl`)

**PostgreSQL нужен ТОЛЬКО если:**
- Вы хотите использовать Smart Collector (сборщик исторических данных)
- Вы хотите хранить большой объём исторических данных в БД

Для **обычной торговли в paper mode** PostgreSQL **НЕ НУЖЕН**.

---

### ❌ **Binance API ключи** - НЕ ТРЕБУЮТСЯ!

**ВАЖНО:** Для paper trading API ключи **НЕ НУЖНЫ**!

Бот использует **публичные API Binance**:
- `https://api.binance.com/api/v3/ticker/price` (цены)
- `https://api.binance.com/api/v3/klines` (свечи)
- `https://api.binance.com/api/v3/depth` (стакан)

**Никакой регистрации, никаких ключей!**

---

## 🎯 Краткий чеклист ПЕРЕД запуском install.bat

Откройте **Command Prompt** и выполните:

```batch
REM 1. Проверить Python
python --version

REM 2. Проверить pip
pip --version

REM 3. Проверить путь (опционально)
echo %PATH% | findstr Python
```

**Ожидаемый результат:**
```
Python 3.9.13 (или выше)
pip 23.x.x from ...
C:\Python39\;C:\Python39\Scripts\; ...
```

Если все 3 пункта OK → **можно запускать install.bat**!

---

## 🚀 Порядок установки

### Шаг 1: Подготовка системы

```batch
REM Открыть Command Prompt от администратора
REM Win + X → Command Prompt (Admin)

REM Обновить pip (рекомендуется)
python -m pip install --upgrade pip
```

### Шаг 2: Запуск install.bat

```batch
REM Перейти в папку бота
cd C:\path\to\jjhbn\GGG3

REM Запустить установку
install.bat
```

**Что произойдёт:**
1. ✅ Проверка Python 3.8+
2. ✅ Создание виртуального окружения `venv`
3. ✅ Установка Python зависимостей из `requirements.txt`
4. ✅ Создание файла конфигурации `.env`
5. ✅ Проверка установки

**Время установки:** ~5-10 минут

### Шаг 3: Настройка (опционально)

```batch
REM Проверить .env файл (для paper trading оставить как есть)
notepad .env
```

Для paper trading **НЕ НУЖНО ничего менять!**

### Шаг 4: Запуск бота

```batch
REM Запустить бота в paper trading mode
start_bot.bat

REM Выбрать: 1 (интерактивный режим)
```

---

## 📦 Что устанавливает install.bat?

**Python библиотеки:**
- `numpy`, `pandas`, `scipy` - вычисления
- `scikit-learn`, `xgboost`, `lightgbm` - ML модели
- `python-binance`, `ccxt` - API для бирж
- `requests` - HTTP запросы
- `python-dotenv` - работа с .env файлами

**PostgreSQL библиотеки (опциональные):**
- `asyncpg`, `psycopg2-binary` - **НЕ используются** в paper trading mode
- Устанавливаются автоматически, но **НЕ обязательны**

**Если возникнет ошибка** при установке `psycopg2-binary` или `asyncpg`:
- ❌ **Не страшно!** Они НЕ нужны для paper trading
- ✅ Бот будет работать без них
- ✅ Можно просто игнорировать эти ошибки

---

## 🐛 Решение проблем

### Проблема 1: "python не распознаётся"

**Решение:**
```batch
REM Добавить Python в PATH вручную
set PATH=%PATH%;C:\Python39;C:\Python39\Scripts

REM Или переустановить Python с галочкой "Add to PATH"
```

### Проблема 2: Ошибка при установке psycopg2-binary

**Сообщение:**
```
ERROR: Failed building wheel for psycopg2-binary
```

**Решение:**
```batch
REM Игнорировать эту ошибку!
REM PostgreSQL библиотеки НЕ нужны для paper trading

REM Если хочется исправить (опционально):
python -m pip install psycopg2-binary --no-cache-dir
```

Или просто **пропустить** - бот будет работать!

### Проблема 3: Ошибка при установке ccxt (coincurve)

**Сообщение:**
```
error: metadata-generation-failed
× Encountered error while generating package metadata.
╰─> coincurve
```

**Причина:** ccxt>=4.0.0 требует coincurve, который нужно компилировать.

**Решение:**
✅ **УЖЕ ИСПРАВЛЕНО!** requirements.txt использует `ccxt>=3.1.1,<4.0.0`

Версия 3.x работает БЕЗ компиляции на Windows и имеет все функции!

Если всё равно возникает ошибка, обновите pip:
```batch
python -m pip install --upgrade pip setuptools wheel
```

---

### Проблема 4: Ошибка при установке ta-lib

**Сообщение:**
```
ERROR: Failed building wheel for ta-lib
```

**Решение для Windows:**

1. Скачайте pre-built wheel с https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. Выберите подходящий файл:
   - `TA_Lib‑0.4.28‑cp39‑cp39‑win_amd64.whl` (для Python 3.9, 64-bit)
   - `TA_Lib‑0.4.28‑cp310‑cp310‑win_amd64.whl` (для Python 3.10, 64-bit)

3. Установите вручную:
   ```batch
   pip install C:\path\to\TA_Lib‑0.4.28‑cp39‑cp39‑win_amd64.whl
   ```

**Или пропустить** - бот работает БЕЗ ta-lib (используются альтернативные индикаторы).

### Проблема 5: Нет интернета при установке

**Решение:**
```batch
REM Скачать requirements.txt заранее на другом компьютере
pip download -r requirements.txt -d packages

REM Затем установить оффлайн:
pip install --no-index --find-links packages -r requirements.txt
```

---

## 📊 Требования к системе

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| **ОС** | Windows Server 2012 R2 | Windows 10/11, Server 2019+ |
| **Python** | 3.8.x | 3.9.x - 3.11.x |
| **RAM** | 2 GB | 4 GB+ |
| **HDD** | 1 GB | 5 GB+ |
| **Интернет** | Есть | Стабильный |
| **PostgreSQL** | ❌ НЕ нужен | ❌ НЕ нужен |
| **API ключи** | ❌ НЕ нужны | ❌ НЕ нужны |

---

## ✅ Готово к установке!

**Если у вас установлен Python 3.8+**, то можно сразу запускать:

```batch
cd C:\path\to\jjhbn\GGG3
install.bat
```

**Всё остальное установится автоматически!** 🚀

---

## 📚 Дополнительная информация

- **Быстрый старт:** `QUICK_START_WINDOWS.md`
- **Работа без API ключей:** `README_NO_API_KEYS.md`
- **Удаление старого бота:** `CLEANUP_OLD_BOT_WINDOWS.md`
- **Шпаргалка команд:** `COMMANDS_CHEATSHEET.md`

---

## 🆘 Нужна помощь?

Если что-то не работает:

1. Проверьте версию Python: `python --version` (должно быть 3.8+)
2. Проверьте pip: `pip --version`
3. Запустите install.bat от администратора
4. Проверьте логи: `logs\install.log` (если есть)

**Удачи с установкой!** 🎉
