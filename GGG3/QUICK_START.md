# ⚡ Быстрый старт - 5 минут до запуска

Минимальная инструкция для быстрого запуска бота в режиме paper trading.

---

## 🚀 Запуск за 5 команд

```bash
# 1. Перейдите в директорию проекта
cd /home/user/jjhbn/GGG3

# 2. Запустите автоматическую установку
chmod +x install.sh
./install.sh

# 3. Настройте .env файл (отредактируйте API ключи и Telegram)
nano .env

# 4. Запустите бота в paper trading режиме
source venv/bin/activate
python3 binance_bot_main.py --paper

# 5. Откройте новый терминал и следите за логами
tail -f logs/binance_bot.log
```

---

## 📋 Минимальная конфигурация .env

```bash
# Обязательные настройки (МИНИМУМ):
BINANCE_API_KEY=your_key_here
BINANCE_SECRET_KEY=your_secret_here
USE_TESTNET=true

# Опциональные (можно оставить пустыми для paper trading):
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
PAPER_TRADING_MODE=true
PAPER_INITIAL_BALANCE=10000
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

## 🛑 Остановка бота

Просто нажмите `Ctrl+C` в терминале где запущен бот.

---

## 📚 Полная документация

Для детального руководства смотрите [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)
