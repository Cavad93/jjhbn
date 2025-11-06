# 🚀 Руководство по установке и запуску Binance Trading Bot

Полная инструкция по настройке окружения, установке зависимостей и запуску бота для торговли на Binance Futures.

---

## 📋 Содержание

1. [Требования к системе](#требования-к-системе)
2. [Шаг 1: Очистка старого окружения](#шаг-1-очистка-старого-окружения)
3. [Шаг 2: Создание нового виртуального окружения](#шаг-2-создание-нового-виртуального-окружения)
4. [Шаг 3: Установка зависимостей](#шаг-3-установка-зависимостей)
5. [Шаг 4: Настройка конфигурации](#шаг-4-настройка-конфигурации)
6. [Шаг 5: Запуск бота](#шаг-5-запуск-бота)
7. [Решение проблем](#решение-проблем)
8. [Тестирование](#тестирование)

---

## 🖥️ Требования к системе

### Минимальные требования:
- **Python**: 3.9 или выше (рекомендуется 3.11+)
- **RAM**: 4GB минимум, 8GB рекомендуется
- **Диск**: 2GB свободного места
- **ОС**: Linux, macOS, Windows (WSL2)

### Дополнительно:
- Git (для клонирования репозитория)
- Доступ к интернету (для API Binance)
- Telegram бот токен (для уведомлений)

---

## 🧹 Шаг 1: Очистка старого окружения

### 1.1 Удаление старого виртуального окружения

Если у вас уже есть старое окружение, удалите его:

```bash
# Перейдите в директорию проекта
cd /home/user/jjhbn/GGG3

# Деактивируйте текущее окружение (если активно)
deactivate 2>/dev/null || true

# Удалите старое виртуальное окружение
rm -rf venv/
rm -rf env/
rm -rf .venv/

echo "✅ Старое окружение удалено"
```

### 1.2 Очистка кэша Python

```bash
# Удалите скомпилированные файлы Python
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

echo "✅ Кэш Python очищен"
```

### 1.3 Удаление глобальных пакетов (опционально)

⚠️ **ВНИМАНИЕ**: Выполняйте только если вы уверены, что хотите удалить все установленные пакеты!

```bash
# Создайте список установленных пакетов
pip3 freeze > old_packages.txt

# Удалите все пакеты (ОСТОРОЖНО!)
# pip3 uninstall -y -r old_packages.txt

# Или удалите только конфликтующие пакеты
pip3 uninstall -y python-binance ccxt xgboost torch ta-lib cma river
```

---

## 🔧 Шаг 2: Создание нового виртуального окружения

### 2.1 Создание venv

```bash
# Перейдите в директорию проекта
cd /home/user/jjhbn/GGG3

# Создайте новое виртуальное окружение
python3 -m venv venv

echo "✅ Виртуальное окружение создано"
```

### 2.2 Активация окружения

```bash
# Активируйте окружение
source venv/bin/activate

# Проверьте, что используется правильный Python
which python3
# Должно показать: /home/user/jjhbn/GGG3/venv/bin/python3

python3 --version
# Должно показать версию >= 3.9
```

### 2.3 Обновление pip

```bash
# Обновите pip до последней версии
python3 -m pip install --upgrade pip setuptools wheel

echo "✅ pip обновлен"
```

---

## 📦 Шаг 3: Установка зависимостей

### 3.1 Установка основных зависимостей

```bash
# Убедитесь, что venv активировано
source venv/bin/activate

# Установите все зависимости из requirements.txt
pip3 install -r requirements.txt

# Это может занять 5-10 минут
```

### 3.2 Проверка установки

```bash
# Запустите скрипт проверки зависимостей
python3 verify_dependencies.py
```

**Ожидаемый вывод:**
```
✅ numpy: 1.24.0
✅ pandas: 2.0.0
✅ scikit-learn: 1.3.0
✅ xgboost: 2.0.0
✅ python-binance: 1.0.19
✅ ccxt: 4.0.0
✅ cma: 3.3.0
✅ river: 0.20.0
...
✅ Все зависимости установлены корректно!
```

### 3.3 Установка TA-Lib (опционально)

TA-Lib может требовать дополнительной установки системных библиотек:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Теперь установите Python wrapper
pip3 install ta-lib
```

**macOS:**
```bash
brew install ta-lib
pip3 install ta-lib
```

**Windows (WSL2):**
Следуйте инструкции для Ubuntu выше.

---

## ⚙️ Шаг 4: Настройка конфигурации

### 4.1 Создание .env файла

```bash
# Скопируйте шаблон
cp .env.example .env

# Отредактируйте файл
nano .env
```

### 4.2 Заполните .env файл

```bash
# ==========================================
# BINANCE API
# ==========================================
# Получите API ключи на https://www.binance.com/en/my/settings/api-management
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
USE_TESTNET=true  # true для тестовой сети, false для реальной торговли

# ==========================================
# TELEGRAM NOTIFICATIONS
# ==========================================
# 1. Создайте бота через @BotFather в Telegram
# 2. Получите токен (формат: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz

# 3. Получите ваш chat_id через @userinfobot
TELEGRAM_CHAT_ID=123456789

# ==========================================
# РЕЖИМ РАБОТЫ
# ==========================================
PAPER_TRADING_MODE=true  # true для виртуальной торговли, false для реальной

# ==========================================
# НАЧАЛЬНЫЙ КАПИТАЛ (только для paper trading)
# ==========================================
PAPER_INITIAL_BALANCE=10000  # $10,000 виртуальных
```

**Сохраните файл**: `Ctrl+O`, `Enter`, `Ctrl+X`

### 4.3 Проверка конфигурации

```bash
# Запустите тест конфигурации
python3 test_binance_config.py
```

**Ожидаемый вывод:**
```
✅ .env файл найден
✅ Binance API ключи настроены
✅ Telegram настроен
✅ Режим: PAPER TRADING
✅ Все настройки корректны!
```

---

## 🚀 Шаг 5: Запуск бота

### 5.1 Запуск в Paper Trading режиме (рекомендуется для начала)

```bash
# Убедитесь, что venv активировано
source venv/bin/activate

# Запустите бота в paper trading режиме
python3 binance_bot_main.py --paper
```

**Ожидаемый вывод:**
```
================================================================================
🤖 INITIALIZING BINANCE TRADING BOT
================================================================================
  Mode: 📄 PAPER TRADING ($10000)
  Loaded 0 open positions

  Loading ML models...
    ✓ XGBoost
    ✓ RandomForest
    ⚠ ARF (using fallback)
    ✓ NeuralNet

  Initializing risk managers...

✅ Bot initialized successfully

================================================================================
🚀 STARTING BINANCE TRADING BOT
================================================================================
```

### 5.2 Запуск в Live Trading режиме (ОСТОРОЖНО!)

⚠️ **ВНИМАНИЕ**: Используйте только после тщательного тестирования в paper trading режиме!

```bash
# Запустите бота в live режиме
python3 binance_bot_main.py --live

# Вам будет предложено подтверждение:
# ⚠️  WARNING: LIVE TRADING MODE
# ⚠️  This will use REAL money!
# Are you sure? Type 'YES' to continue: YES
```

### 5.3 Запуск в фоновом режиме

```bash
# Используйте nohup для запуска в фоне
nohup python3 binance_bot_main.py --paper > bot.log 2>&1 &

# Проверьте, что бот работает
tail -f bot.log

# Или используйте screen/tmux
screen -S binance_bot
python3 binance_bot_main.py --paper
# Нажмите Ctrl+A затем D для отсоединения
# Вернитесь: screen -r binance_bot
```

### 5.4 Остановка бота

```bash
# Если запущен в терминале, нажмите Ctrl+C

# Если запущен через nohup, найдите процесс и остановите:
ps aux | grep binance_bot_main.py
kill <PID>

# Или остановите все Python процессы бота (ОСТОРОЖНО!)
# pkill -f binance_bot_main.py
```

---

## 🔧 Решение проблем

### Проблема 1: ModuleNotFoundError

**Ошибка:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Решение:**
```bash
# Убедитесь, что venv активировано
source venv/bin/activate

# Переустановите зависимости
pip3 install -r requirements.txt
```

---

### Проблема 2: Ошибка импорта ta-lib

**Ошибка:**
```
ImportError: libtalib.so.0: cannot open shared object file
```

**Решение:**
```bash
# Ubuntu/Debian
sudo apt-get install -y libta-lib0 libta-lib0-dev

# Или установите TA-Lib из исходников (см. Шаг 3.3)

# Затем переустановите Python wrapper
pip3 install --force-reinstall ta-lib
```

---

### Проблема 3: Binance API ошибки

**Ошибка:**
```
BinanceAPIException: Invalid API-key, IP, or permissions
```

**Решение:**
1. Проверьте, что API ключи правильно скопированы в `.env`
2. Убедитесь, что API ключ имеет права на Futures торговлю
3. Проверьте IP whitelist на Binance (если настроен)
4. Используйте `USE_TESTNET=true` для тестирования

```bash
# Проверьте .env файл
cat .env | grep BINANCE
```

---

### Проблема 4: Telegram уведомления не приходят

**Ошибка:**
```
Telegram notification failed
```

**Решение:**
1. Проверьте токен бота через @BotFather
2. Убедитесь, что вы начали диалог с ботом (отправьте /start)
3. Проверьте chat_id через @userinfobot

```bash
# Тест отправки сообщения
python3 -c "
from notifications.telegram_notifier import TelegramNotifier
import binance_config as config

telegram = TelegramNotifier(
    bot_token=config.TELEGRAM_BOT_TOKEN,
    chat_id=config.TELEGRAM_CHAT_ID,
    enabled=True
)

telegram.notify_bot_started(
    paper_mode=True,
    initial_capital=10000,
    max_positions=10
)
print('✅ Telegram notification sent!')
"
```

---

### Проблема 5: Недостаточно памяти

**Ошибка:**
```
MemoryError: Unable to allocate array
```

**Решение:**
1. Уменьшите количество одновременных позиций в `binance_config.py`:
```python
MAX_POSITIONS = 5  # Вместо 10
```

2. Отключите неиспользуемые ML модели
3. Увеличьте swap память (Linux):
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

### Проблема 6: Permission denied для файлов

**Ошибка:**
```
PermissionError: [Errno 13] Permission denied: 'data/positions/positions.json'
```

**Решение:**
```bash
# Дайте права на запись для директорий данных
chmod -R 755 data/
chmod -R 755 logs/
chmod -R 755 models/saved/

# Или сделайте себя владельцем
sudo chown -R $USER:$USER /home/user/jjhbn/GGG3/
```

---

## 🧪 Тестирование

### Запуск быстрого теста

```bash
# Активируйте окружение
source venv/bin/activate

# Запустите быстрый тест (6 компонентов, ~30 секунд)
python3 tests/quick_bot_test.py
```

**Ожидаемый вывод:**
```
================================================================================
🧪 QUICK BOT TEST - 6 COMPONENTS
================================================================================

1️⃣ BASE LOGIC - Pattern Recognition
──────────────────────────────────────────────────────────────
Testing BASE Logic on 7 market patterns...
  ✓ Uptrend: P(up)=0.582 > 0.52 ✓ Recognized
  ✓ Downtrend: P(down)=0.537 > 0.50 ✓ Recognized
  ✓ Breakout: P(up)=0.828 > 0.53 ✓ Strong signal!
✅ BASE Logic works! Can find patterns even without ML.

...

================================================================================
✅ ALL TESTS PASSED (6/6)
================================================================================
```

### Запуск полного комплексного теста

```bash
# Запустите полный тест (8 компонентов, ~2-3 минуты)
python3 tests/comprehensive_bot_test.py
```

---

## 📊 Мониторинг работы бота

### Просмотр логов

```bash
# Просмотр основного лога
tail -f logs/binance_bot.log

# Просмотр только ошибок
tail -f logs/binance_bot.log | grep ERROR

# Просмотр открытия позиций
tail -f logs/binance_bot.log | grep "ENTRY"

# Просмотр закрытия позиций
tail -f logs/binance_bot.log | grep "EXIT"
```

### Проверка открытых позиций

```bash
# Просмотр JSON файла с позициями
cat data/positions/positions.json | python3 -m json.tool
```

### Проверка производительности

```bash
# Использование CPU и памяти
top -p $(pgrep -f binance_bot_main.py)

# Или используйте htop
htop -p $(pgrep -f binance_bot_main.py)
```

---

## 🎯 Рекомендации по запуску

### Для начинающих:

1. ✅ Начните с **Paper Trading** режима
2. ✅ Протестируйте бота минимум **1-2 недели**
3. ✅ Настройте Telegram уведомления
4. ✅ Следите за логами ежедневно
5. ✅ Убедитесь что Win Rate > 55% перед live торговлей

### Для продвинутых:

1. ⚙️ Настройте параметры в `binance_config.py`
2. 🔧 Обучите ML модели на свежих данных
3. 📈 Используйте dashboard для визуализации
4. 🛡️ Настройте дополнительные риск-менеджмент фильтры
5. 🔄 Регулярно обновляйте модели (раз в неделю)

---

## 📚 Дополнительные ресурсы

- **Документация Binance API**: https://binance-docs.github.io/apidocs/futures/en/
- **Binance Testnet**: https://testnet.binancefuture.com/
- **Telegram Bot API**: https://core.telegram.org/bots/api
- **Репозиторий проекта**: `/home/user/jjhbn/GGG3/`

---

## ⚠️ ВАЖНЫЕ ПРЕДУПРЕЖДЕНИЯ

1. **НЕ ИСПОЛЬЗУЙТЕ live trading** без тщательного тестирования!
2. **НЕ КОММИТЬТЕ** `.env` файл в Git (он в `.gitignore`)
3. **НЕ ДЕЛИТЕСЬ** API ключами с кем-либо
4. **НАЧНИТЕ С МАЛОГО** капитала при переходе на live
5. **МОНИТОРЬТЕ БОТА** ежедневно первые несколько недель
6. **ИСПОЛЬЗУЙТЕ STOP LOSS** - бот всегда устанавливает SL, не отключайте эту функцию
7. **BACKUP ДАННЫХ** регулярно сохраняйте `data/` и `models/saved/`

---

## 📞 Поддержка

Если возникли проблемы:

1. Проверьте [Решение проблем](#решение-проблем)
2. Посмотрите логи: `tail -f logs/binance_bot.log`
3. Запустите тесты: `python3 tests/quick_bot_test.py`
4. Проверьте конфигурацию: `python3 test_binance_config.py`

---

**Удачной торговли! 🚀📈**

*Этот бот создан для образовательных целей. Используйте на свой страх и риск. Авторы не несут ответственности за финансовые потери.*
