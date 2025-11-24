# Инструкция по установке AI Trading Module

## Шаг 1: Установка зависимостей

### Windows (PowerShell или CMD)
```powershell
cd GGG3
pip install -r ai_trading_module\requirements.txt
```

### Linux/Mac
```bash
cd GGG3
pip install -r ai_trading_module/requirements.txt
```

Или вручную:
```bash
pip install anthropic python-dotenv
```

---

## Шаг 2: Настройка переменных окружения

У вас уже есть файл `.env` в директории `GGG3`. Убедитесь что в нем указаны ключи:

### Откройте файл `GGG3/.env` и добавьте/проверьте:

```env
# ОБЯЗАТЕЛЬНО: API ключ Anthropic
ANTHROPIC_API_KEY=sk-ant-ваш_ключ_здесь

# ОПЦИОНАЛЬНО: Telegram уведомления
TELEGRAM_BOT_TOKEN=ваш_токен_здесь
TELEGRAM_CHAT_ID=939056216
```

**ВАЖНО:** Не используйте кавычки вокруг значений!

❌ **Неправильно:**
```env
ANTHROPIC_API_KEY="sk-ant-..."
ANTHROPIC_API_KEY='sk-ant-...'
```

✅ **Правильно:**
```env
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Шаг 3: Проверка установки

### Проверьте что зависимости установлены:
```bash
python -c "import anthropic; print('anthropic OK')"
python -c "import dotenv; print('dotenv OK')"
```

Вы должны увидеть:
```
anthropic OK
dotenv OK
```

### Проверьте что .env файл загружается:
```bash
python -c "from dotenv import load_dotenv; from pathlib import Path; load_dotenv(Path('GGG3/.env')); import os; print('ANTHROPIC_API_KEY:', 'SET' if os.getenv('ANTHROPIC_API_KEY') else 'NOT SET')"
```

Вы должны увидеть:
```
ANTHROPIC_API_KEY: SET
```

---

## Шаг 4: Запуск

```bash
cd GGG3
python ai_trader_main.py --paper --capital 1000
```

Вы должны увидеть:
```
[Setup] Loaded environment variables from GGG3\.env

================================================================================
                         AI TRADING MODULE
                    Powered by Claude Sonnet 4.5
================================================================================

[Setup] Using Paper Trading Exchange
...
```

---

## Troubleshooting

### Проблема: "ANTHROPIC_API_KEY not set"

**Решение 1:** Проверьте файл `.env`
```bash
# Windows
type GGG3\.env

# Linux/Mac
cat GGG3/.env
```

Убедитесь что `ANTHROPIC_API_KEY` указан БЕЗ кавычек:
```env
ANTHROPIC_API_KEY=sk-ant-api03-ваш_ключ_здесь
```

**Решение 2:** Проверьте что python-dotenv установлен
```bash
pip list | findstr dotenv    # Windows
pip list | grep dotenv       # Linux/Mac
```

Если не установлен:
```bash
pip install python-dotenv
```

**Решение 3:** Установите переменную вручную (временно)

Windows PowerShell:
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-ваш_ключ"
python ai_trader_main.py --paper --capital 1000
```

Windows CMD:
```cmd
set ANTHROPIC_API_KEY=sk-ant-ваш_ключ
python ai_trader_main.py --paper --capital 1000
```

---

### Проблема: "python-dotenv not installed"

```bash
pip install python-dotenv
```

---

### Проблема: Скрипт не видит .env файл

Проверьте что вы запускаете скрипт из директории `GGG3`:

```bash
# Проверьте текущую директорию
pwd           # Linux/Mac
cd            # Windows

# Вы должны быть в директории GGG3
# Например: C:\Users\Administrator\Desktop\GGG3
```

---

### Проблема: Ошибка импорта модулей

```bash
# Убедитесь что все зависимости установлены
pip install numpy pandas requests anthropic python-dotenv
```

---

## Проверка что все работает

После успешного запуска вы увидите:

```
[Setup] Loaded environment variables from ...
[AI Trader] Initialized with $1000.0 capital
[AI Trader] Model: claude-sonnet-4-5-20250929
[AI Trader] Paper Trading: True

================================================================================
AI TRADER STARTED
================================================================================

[2025-11-24 10:00:00] Starting daily decision cycle...
[AI] Analyzing market and making decisions...
```

Если видите это - **установка прошла успешно!** 🎉

---

## Дополнительная помощь

Если проблемы остались, проверьте:

1. **Python версия** (должна быть >= 3.8):
   ```bash
   python --version
   ```

2. **Pip версия**:
   ```bash
   pip --version
   ```

3. **Файл .env существует**:
   ```bash
   # Windows
   dir .env

   # Linux/Mac
   ls -la .env
   ```

4. **Логи**:
   ```bash
   # После запуска смотрите логи
   type ai_trading_module\data\ai_trader.log    # Windows
   cat ai_trading_module/data/ai_trader.log     # Linux/Mac
   ```
