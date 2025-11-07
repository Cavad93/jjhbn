# 🗄️ Установка PostgreSQL 12 на Windows Server 2012 R2 (ОПЦИОНАЛЬНО)

## ⚠️ ВАЖНО: PostgreSQL НЕ НУЖЕН для paper trading!

**Эта инструкция только для тех, кто хочет:**
- Использовать Smart Collector (сборщик исторических данных)
- Хранить большие объёмы данных в БД
- Использовать собственную БД для анализа

**Для обычного paper trading с публичными API Binance PostgreSQL НЕ ТРЕБУЕТСЯ!**

---

## 📦 Установка PostgreSQL 12

### Вариант 1: Установка через инсталлятор (Рекомендуется)

#### Шаг 1: Скачать PostgreSQL 12

1. Перейдите на https://www.enterprisedb.com/downloads/postgres-postgresql-downloads
2. Выберите **PostgreSQL 12.x** для Windows x86-64
3. Скачайте файл (например, `postgresql-12.17-1-windows-x64.exe`)

**Или прямая ссылка:**
```
https://sbp.enterprisedb.com/getfile.jsp?fileid=1258253
```

#### Шаг 2: Запустить инсталлятор

1. **Запустите** скачанный файл от администратора
2. **Выберите компоненты:**
   - ✅ PostgreSQL Server
   - ✅ pgAdmin 4 (графический интерфейс)
   - ✅ Stack Builder (опционально)
   - ✅ Command Line Tools

3. **Выберите папку установки:**
   - По умолчанию: `C:\Program Files\PostgreSQL\12`

4. **Выберите папку для данных:**
   - По умолчанию: `C:\Program Files\PostgreSQL\12\data`

5. **Установите пароль для superuser (postgres):**
   - Запомните этот пароль! (например, `postgres123`)

6. **Выберите порт:**
   - По умолчанию: `5432`

7. **Выберите локаль:**
   - По умолчанию: `[Default locale]` или `English, United States`

8. Нажмите **Next** → **Install**

**Время установки:** ~5-10 минут

#### Шаг 3: Проверка установки

Откройте **Command Prompt** и выполните:

```batch
REM Проверить версию PostgreSQL
psql --version

REM Должно быть: psql (PostgreSQL) 12.x
```

Если команда `psql` не распознаётся:

```batch
REM Добавить PostgreSQL в PATH
set PATH=%PATH%;C:\Program Files\PostgreSQL\12\bin

REM Или добавить постоянно через System Properties → Environment Variables
```

#### Шаг 4: Создать базу данных для бота

```batch
REM Войти в PostgreSQL
psql -U postgres

REM Введите пароль (который установили при установке)
```

**В консоли PostgreSQL:**
```sql
-- Создать базу данных для бота
CREATE DATABASE binance_bot;

-- Создать пользователя для бота
CREATE USER bot_user WITH PASSWORD 'bot_password_123';

-- Дать права пользователю
GRANT ALL PRIVILEGES ON DATABASE binance_bot TO bot_user;

-- Выйти
\q
```

#### Шаг 5: Настроить бота для использования PostgreSQL

Откройте `.env` файл:

```batch
notepad .env
```

Добавьте настройки PostgreSQL:

```ini
# PostgreSQL настройки (для Smart Collector)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=binance_bot
DB_USER=bot_user
DB_PASSWORD=bot_password_123
```

---

### Вариант 2: Установка через Chocolatey

Если у вас установлен Chocolatey:

```batch
REM Установить PostgreSQL 12
choco install postgresql12 --params '/Password:postgres123'

REM Проверить установку
psql --version
```

---

### Вариант 3: Portable PostgreSQL (без инсталлятора)

1. Скачайте PostgreSQL Portable с https://github.com/garethflowers/postgresql-portable/releases
2. Распакуйте в папку (например, `C:\PostgreSQL12`)
3. Запустите `initialize.bat`
4. Запустите `start.bat`

---

## 🔧 Настройка PostgreSQL для бота

### 1. Разрешить удалённые подключения (опционально)

Если PostgreSQL на другом сервере:

**Файл:** `C:\Program Files\PostgreSQL\12\data\postgresql.conf`

```ini
# Найти и изменить:
listen_addresses = '*'          # было 'localhost'
```

**Файл:** `C:\Program Files\PostgreSQL\12\data\pg_hba.conf`

Добавить в конец:
```
# IPv4 local connections:
host    all             all             0.0.0.0/0               md5
```

**Перезапустить PostgreSQL:**

```batch
REM Через Services
sc stop postgresql-x64-12
sc start postgresql-x64-12

REM Или через pgAdmin
```

### 2. Создать таблицы для Smart Collector

```sql
-- Войти в базу
psql -U bot_user -d binance_bot

-- Создать таблицу для исторических данных
CREATE TABLE ohlcv (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp BIGINT NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, timestamp)
);

-- Создать индексы для быстрого поиска
CREATE INDEX idx_ohlcv_symbol ON ohlcv(symbol);
CREATE INDEX idx_ohlcv_timestamp ON ohlcv(timestamp);
CREATE INDEX idx_ohlcv_symbol_timeframe ON ohlcv(symbol, timeframe);

-- Проверить таблицу
\dt

-- Выйти
\q
```

---

## 🧪 Проверка подключения из Python

Создайте тестовый скрипт `test_postgres.py`:

```python
import psycopg2

try:
    # Подключение к БД
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="binance_bot",
        user="bot_user",
        password="bot_password_123"
    )

    # Создать курсор
    cur = conn.cursor()

    # Проверить версию PostgreSQL
    cur.execute("SELECT version();")
    version = cur.fetchone()
    print(f"✅ PostgreSQL version: {version[0]}")

    # Закрыть подключение
    cur.close()
    conn.close()

    print("✅ Подключение к PostgreSQL успешно!")

except Exception as e:
    print(f"❌ Ошибка подключения: {e}")
```

Запустить:

```batch
python test_postgres.py
```

Ожидаемый результат:
```
✅ PostgreSQL version: PostgreSQL 12.17 ...
✅ Подключение к PostgreSQL успешно!
```

---

## 🐛 Решение проблем

### Проблема 1: "psql не распознаётся"

**Решение:**
```batch
REM Добавить PostgreSQL в PATH
set PATH=%PATH%;C:\Program Files\PostgreSQL\12\bin

REM Или указать полный путь
"C:\Program Files\PostgreSQL\12\bin\psql" --version
```

### Проблема 2: "connection refused"

**Проверить, запущен ли PostgreSQL:**

```batch
REM Через Services
sc query postgresql-x64-12

REM Если не запущен:
sc start postgresql-x64-12
```

**Или через Task Manager:**
- Нажмите `Ctrl+Shift+Esc`
- Перейдите на вкладку "Services"
- Найдите `postgresql-x64-12`
- Проверьте статус (должно быть "Running")

### Проблема 3: "password authentication failed"

**Решение:**

```batch
REM Сбросить пароль для пользователя postgres
psql -U postgres

-- В консоли PostgreSQL:
ALTER USER postgres WITH PASSWORD 'new_password';
\q
```

### Проблема 4: Ошибка при установке psycopg2-binary

**Если уже установлен PostgreSQL, но psycopg2-binary не устанавливается:**

```batch
REM Переустановить с указанием пути к PostgreSQL
set POSTGRES_HOME=C:\Program Files\PostgreSQL\12
pip install psycopg2-binary --no-cache-dir
```

**Или скачать pre-built wheel:**

1. Перейдите на https://www.lfd.uci.edu/~gohlke/pythonlibs/#psycopg
2. Скачайте подходящий файл (например, `psycopg2‑2.9.9‑cp39‑cp39‑win_amd64.whl`)
3. Установите:
   ```batch
   pip install C:\path\to\psycopg2‑2.9.9‑cp39‑cp39‑win_amd64.whl
   ```

---

## 🗑️ Удаление PostgreSQL 12

Если PostgreSQL больше не нужен:

### Способ 1: Через Control Panel

1. **Control Panel** → **Programs** → **Uninstall a program**
2. Найдите **PostgreSQL 12**
3. Нажмите **Uninstall**

### Способ 2: Через Command Prompt

```batch
REM Остановить сервис
sc stop postgresql-x64-12

REM Удалить сервис
sc delete postgresql-x64-12

REM Удалить папки вручную
rmdir /S /Q "C:\Program Files\PostgreSQL\12"
rmdir /S /Q "%APPDATA%\postgresql"
```

---

## 📊 Сравнение: С PostgreSQL vs БЕЗ PostgreSQL

| Функция | БЕЗ PostgreSQL | С PostgreSQL |
|---------|----------------|--------------|
| **Paper trading** | ✅ Работает | ✅ Работает |
| **Публичные API** | ✅ Работает | ✅ Работает |
| **Хранение позиций** | JSON файлы | JSON или PostgreSQL |
| **Smart Collector** | ❌ Не работает | ✅ Работает |
| **Исторические данные** | Ограниченные | Полные |
| **Быстродействие** | Отличное | Хорошее |
| **Сложность установки** | Простая | Средняя |
| **Требования к системе** | Минимальные | Средние |

---

## ✅ Итого

### Для **Paper Trading** (рекомендуется):
- ❌ **PostgreSQL НЕ нужен**
- ✅ Работает с публичными API
- ✅ JSON файлы для хранения
- ✅ Простая установка

### Для **Smart Collector** (опционально):
- ✅ PostgreSQL 12 нужен
- ✅ Хранение больших объёмов данных
- ✅ Сложная аналитика
- ⚠️ Дополнительная настройка

---

## 📚 Дополнительная информация

- **Предварительные требования:** `PREREQUISITES_WINDOWS.md`
- **Быстрый старт:** `QUICK_START_WINDOWS.md`
- **Работа без API ключей:** `README_NO_API_KEYS.md`
- **Удаление старого бота:** `CLEANUP_OLD_BOT_WINDOWS.md`

---

**Если сомневаетесь - начните БЕЗ PostgreSQL!** Его всегда можно добавить позже. 🚀
