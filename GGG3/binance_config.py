#!/usr/bin/env python3
"""
Конфигурация для Binance торгового бота
Миграция с PancakeSwap на Binance Futures

ВАЖНО:
- По умолчанию используется paper trading (виртуальные деньги)
- Для реальной торговли установите PAPER_TRADING_MODE = False
- API ключи загружаются из переменных окружения
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 1. PAPER TRADING MODE
# ============================================================================

PAPER_TRADING_MODE = True  # True - виртуальные деньги, False - реальная торговля

# Начальный баланс для paper trading (в USDT)
PAPER_INITIAL_BALANCE = 1000.0


# ============================================================================
# 2. API CREDENTIALS
# ============================================================================

# API ключи загружаются из переменных окружения
# Установите их через: export BINANCE_API_KEY="ваш_ключ"
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')

# Использовать testnet (рекомендуется для тестирования)
USE_TESTNET = os.getenv('USE_TESTNET', 'True').lower() in ('true', '1', 'yes')

# Telegram бот (опционально)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')


# ============================================================================
# 3. TRADING PARAMETERS
# ============================================================================

# Максимальное количество одновременно открытых позиций
MAX_POSITIONS = 10

# Минимальный размер позиции (в USDT)
MIN_POSITION_SIZE_USDT = 15.0

# Максимальный размер позиции (в USDT)
MAX_POSITION_SIZE_USDT = 200.0

# Leverage (плечо)
LEVERAGE = 1  # 1x - консервативная торговля без плеча

# ATR параметры для расчёта TP/SL
ATR_PERIOD = 14  # Период для расчёта ATR
ATR_TIMEFRAME = '4h'  # Таймфрейм для расчёта ATR

# Множители для TP/SL на основе ATR
TP_ATR_MULTIPLIER = 2.5  # Take Profit = entry_price ± (ATR * 2.5)
SL_ATR_MULTIPLIER = 1.5  # Stop Loss = entry_price ± (ATR * 1.5)

# Минимальное расстояние до TP/SL (в процентах)
MIN_TP_DISTANCE_PCT = 1.0  # Минимум 1% до TP
MIN_SL_DISTANCE_PCT = 0.5  # Минимум 0.5% до SL

# Максимальное время удержания позиции (в часах)
MAX_POSITION_HOLD_TIME_HOURS = 72  # 3 дня


# ============================================================================
# 4. RISK MANAGEMENT
# ============================================================================

# Процент от баланса на одну позицию
MIN_RISK_PER_POSITION = 0.005  # 0.5%
MAX_RISK_PER_POSITION = 0.025  # 2.5%
DEFAULT_RISK_PER_POSITION = 0.009  # 0.9% по умолчанию

# Максимальная просадка (в процентах от начального баланса)
MAX_DRAWDOWN_PCT = 20.0  # Остановить торговлю при просадке 20%

# Максимальная просадка за день
MAX_DAILY_LOSS_PCT = 5.0  # Остановить торговлю при дневной просадке 5%

# Минимальный баланс для продолжения торговли (в USDT)
MIN_BALANCE_TO_TRADE = 50.0


# ============================================================================
# 5. EV FILTER & COIN SELECTION
# ============================================================================

# Минимальный порог EV (Expected Value) для открытия позиции
MIN_EV_THRESHOLD = 0.02  # 2% минимальный ожидаемый профит

# Количество топ монет по EV для рассмотрения
TOP_N_COINS = 10

# Минимальный объём торгов за 24ч (в USDT)
MIN_VOLUME_24H_USDT = 10_000_000  # 10 млн USDT

# Исключённые монеты (из-за высокой волатильности или других причин)
EXCLUDED_SYMBOLS = [
    'USDCUSDT',  # Стейблкоины
    'BUSDUSDT',
    'TUSDUSDT',
    'USDPUSDT',
]


# ============================================================================
# 6. MULTI-TIMEFRAME CONFIGURATION
# ============================================================================

# Конфигурация таймфреймов для анализа
CANDLE_TIMEFRAMES = {
    '5m': {
        'interval': '5m',
        'limit': 500,  # ~41 час данных
        'weight': 0.2
    },
    '15m': {
        'interval': '15m',
        'limit': 500,  # ~5 дней данных
        'weight': 0.25
    },
    '30m': {
        'interval': '30m',
        'limit': 500,  # ~10 дней данных
        'weight': 0.25
    },
    '4h': {
        'interval': '4h',
        'limit': 500,  # ~83 дня данных
        'weight': 0.3
    }
}

# Основной таймфрейм для принятия решений
PRIMARY_TIMEFRAME = '4h'

# Минимальное количество свечей для расчёта индикаторов
MIN_CANDLES_FOR_FEATURES = 100


# ============================================================================
# 7. ADAPTIVE THRESHOLD
# ============================================================================

# Начальный порог для модели (до обучения)
BASE_THRESHOLD_INITIAL = 0.65

# Порог для обученной модели
BASE_THRESHOLD_TRAINED = 0.58

# Параметры адаптации порога
DELTA_MIN = -0.05  # Минимальная коррекция порога
DELTA_MAX = 0.10   # Максимальная коррекция порога

# Коэффициент сглаживания для EMA адаптации
ADAPTIVE_SMOOTHING_FACTOR = 0.1

# Минимальное количество сделок для активации адаптивного порога
MIN_TRADES_FOR_ADAPTIVE = 50


# ============================================================================
# 8. META NEURAL NETWORK
# ============================================================================

# Режим работы META модели
META_MODE = "SHADOW"  # "SHADOW" или "ACTIVE"

# SHADOW режим: META работает параллельно, но не влияет на решения
# ACTIVE режим: META принимает окончательные решения

# Критерии активации META (переход из SHADOW в ACTIVE)
META_ACTIVATION_CRITERIA = {
    'min_shadow_trades': 100,      # Минимум 100 сделок в shadow режиме
    'min_shadow_profit': 0.05,     # Минимум 5% профита в shadow режиме
    'min_win_rate': 0.55,          # Минимум 55% win rate
    'min_sharpe_ratio': 1.2,       # Минимум Sharpe Ratio 1.2
}

# Критерии возврата в SHADOW режим (если META ухудшает результаты)
META_REVERT_CRITERIA = {
    'max_consecutive_losses': 5,   # 5 убыточных сделок подряд
    'max_drawdown': 0.10,          # Просадка 10% от пика
    'min_win_rate': 0.45,          # Win rate упал ниже 45%
}

# Архитектура META сети
META_NETWORK_CONFIG = {
    'input_size': 8,               # 8 предсказаний базовых моделей
    'hidden_layers': [16, 8],      # Два скрытых слоя
    'output_size': 1,              # 1 финальное предсказание
    'dropout': 0.2,                # Dropout для регуляризации
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
}


# ============================================================================
# 9. SPECIAL MODES
# ============================================================================

# Trailing Stop Loss
TRAILING_STOP_ENABLED = True
TRAILING_STOP_ACTIVATION_PCT = 2.0  # Активация при 2% прибыли
TRAILING_STOP_DISTANCE_PCT = 1.0    # Trailing на расстоянии 1%

# Black Swan Protection (защита от резких обвалов)
BLACK_SWAN_PROTECTION = True
BLACK_SWAN_PRICE_DROP_PCT = 5.0     # Закрыть все позиции при падении цены на 5% за 5 минут
BLACK_SWAN_CHECK_INTERVAL_SEC = 60  # Проверка каждую минуту

# Night Mode (снижение активности ночью)
NIGHT_MODE_ENABLED = True
NIGHT_MODE_START_HOUR = 23  # UTC
NIGHT_MODE_END_HOUR = 7     # UTC
NIGHT_MODE_MAX_POSITIONS = 5  # Максимум 5 позиций ночью

# Sector Diversification (диверсификация по секторам)
SECTOR_DIVERSIFICATION = True
MAX_POSITIONS_PER_SECTOR = 3  # Максимум 3 позиции на сектор

# Определение секторов
SECTORS = {
    'LAYER1': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT', 'SOLUSDT', 'AVAXUSDT'],
    'DEFI': ['UNIUSDT', 'AAVEUSDT', 'SUSHIUSDT', 'CAKEUSDT', 'COMPUSDT', 'CRVUSDT'],
    'LAYER2': ['MATICUSDT', 'OPUSDT', 'ARBUSDT', 'IMXUSDT'],
    'GAMING': ['AXSUSDT', 'SANDUSDT', 'MANAUSDT', 'GALAMUSDT', 'ENJUSDT'],
    'ORACLE': ['LINKUSDT', 'BANDUSDT', 'TLMUSDT'],
    'STORAGE': ['FILUSDT', 'ARUSDT', 'STORJUSDT'],
    'PRIVACY': ['XMRUSDT', 'ZECUSDT', 'SCRTUSDT'],
    'MEME': ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT'],
    'AI': ['FETUSDT', 'AGIXUSDT', 'OCEANUS DT', 'RENDERUSDT'],
    'OTHER': []  # Все остальные
}


# ============================================================================
# 10. AUTO-REINVESTMENT
# ============================================================================

# Автоматическое реинвестирование прибыли
AUTO_REINVEST_ENABLED = True

# Процент прибыли, который идёт в резерв (не реинвестируется)
REINVEST_PROFIT_TO_RESERVE_PCT = 0.3  # 30% прибыли в резерв

# Минимальная прибыль для реинвестирования (в USDT)
MIN_PROFIT_TO_REINVEST = 10.0

# Целевой процент увеличения размера позиции при реинвестировании
REINVEST_POSITION_SIZE_INCREASE_PCT = 0.1  # Увеличить размер позиции на 10%


# ============================================================================
# 11. ONLINE LEARNING
# ============================================================================

# Включить онлайн обучение моделей
ONLINE_LEARNING_ENABLED = True

# Минимальное количество закрытых позиций для переобучения
MIN_CLOSED_POSITIONS_FOR_RETRAIN = 20

# Максимальное количество позиций в обучающей выборке (для ограничения памяти)
MAX_TRAINING_SAMPLES = 1000

# Частота переобучения (в количестве новых закрытых позиций)
RETRAIN_EVERY_N_POSITIONS = 10

# Использовать только прибыльные сделки для обучения
USE_ONLY_PROFITABLE_FOR_TRAINING = False  # False - использовать все сделки


# ============================================================================
# 12. PATHS
# ============================================================================

# Базовая директория проекта
BASE_DIR = Path(__file__).parent

# Директория для данных
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

# Директория для моделей
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# Директория для логов
LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

# Директория для сохранения позиций
POSITIONS_DIR = DATA_DIR / 'positions'
POSITIONS_DIR.mkdir(exist_ok=True)

# Директория для бэкапов
BACKUPS_DIR = DATA_DIR / 'backups'
BACKUPS_DIR.mkdir(exist_ok=True)

# Файлы
PAPER_EXCHANGE_STATE_FILE = DATA_DIR / 'paper_exchange_state.json'
POSITIONS_FILE = POSITIONS_DIR / 'positions.json'
CLOSED_POSITIONS_FILE = POSITIONS_DIR / 'closed_positions.json'
BALANCE_HISTORY_FILE = DATA_DIR / 'balance_history.json'
TRADES_LOG_FILE = DATA_DIR / 'trades_log.csv'
PERFORMANCE_METRICS_FILE = DATA_DIR / 'performance_metrics.json'

# Файлы моделей
MODEL_XGB_FILE = MODELS_DIR / 'xgb_model.json'
MODEL_RF_FILE = MODELS_DIR / 'rf_model.pkl'
MODEL_LGBM_FILE = MODELS_DIR / 'lgbm_model.txt'
MODEL_META_FILE = MODELS_DIR / 'meta_model.pth'

# Конфигурация логирования
LOG_FILE = LOGS_DIR / 'bot.log'
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# ============================================================================
# 13. TELEGRAM ALERTS
# ============================================================================

# Включить Telegram уведомления
TELEGRAM_ALERTS_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# Типы уведомлений
TELEGRAM_ALERT_TYPES = {
    'position_opened': True,      # Открыта новая позиция
    'position_closed': True,      # Закрыта позиция
    'tp_hit': True,               # Достигнут TP
    'sl_hit': True,               # Достигнут SL
    'error': True,                # Критическая ошибка
    'daily_summary': True,        # Ежедневная сводка
    'weekly_summary': True,       # Еженедельная сводка
    'balance_milestone': True,    # Достигнут баланс milestone
    'drawdown_warning': True,     # Предупреждение о просадке
    'model_retrained': False,     # Модель переобучена (может быть шумно)
}

# Время отправки ежедневной сводки (UTC)
DAILY_SUMMARY_HOUR = 20  # 20:00 UTC


# ============================================================================
# 14. RATE LIMITING
# ============================================================================

# Максимальное количество запросов к Binance API в минуту
MAX_REQUESTS_PER_MINUTE = 1000  # Binance лимит 1200, оставляем запас

# Пауза между итерациями главного цикла (в секундах)
MAIN_LOOP_INTERVAL_SEC = 60  # Проверять рынок каждую минуту

# Пауза между запросами к API (в секундах)
API_REQUEST_DELAY_SEC = 0.1

# Максимальное количество повторных попыток при ошибках API
MAX_API_RETRIES = 3

# Задержка между повторными попытками (в секундах)
API_RETRY_DELAY_SEC = 1.0


# ============================================================================
# 15. VALIDATION FUNCTION
# ============================================================================

def validate_config() -> tuple[bool, List[str]]:
    """
    Валидация конфигурации

    Returns:
        tuple[bool, List[str]]: (valid, errors)
            - valid: True если конфигурация корректна
            - errors: Список ошибок валидации
    """
    errors = []

    # Проверка API ключей (только для реальной торговли)
    if not PAPER_TRADING_MODE:
        if not BINANCE_API_KEY:
            errors.append("BINANCE_API_KEY не установлен (требуется для реальной торговли)")
        if not BINANCE_SECRET_KEY:
            errors.append("BINANCE_SECRET_KEY не установлен (требуется для реальной торговли)")

    # Проверка числовых параметров
    if MAX_POSITIONS < 1:
        errors.append(f"MAX_POSITIONS должно быть >= 1, получено: {MAX_POSITIONS}")

    if MAX_POSITIONS > 20:
        errors.append(f"MAX_POSITIONS не должно превышать 20, получено: {MAX_POSITIONS}")

    if MIN_POSITION_SIZE_USDT < 5.0:
        errors.append(f"MIN_POSITION_SIZE_USDT должен быть >= 5.0, получено: {MIN_POSITION_SIZE_USDT}")

    if MAX_POSITION_SIZE_USDT < MIN_POSITION_SIZE_USDT:
        errors.append(f"MAX_POSITION_SIZE_USDT ({MAX_POSITION_SIZE_USDT}) < MIN_POSITION_SIZE_USDT ({MIN_POSITION_SIZE_USDT})")

    if LEVERAGE < 1 or LEVERAGE > 10:
        errors.append(f"LEVERAGE должно быть в диапазоне [1, 10], получено: {LEVERAGE}")

    # Проверка ATR параметров
    if ATR_PERIOD < 5 or ATR_PERIOD > 50:
        errors.append(f"ATR_PERIOD должен быть в диапазоне [5, 50], получено: {ATR_PERIOD}")

    if TP_ATR_MULTIPLIER < 1.0:
        errors.append(f"TP_ATR_MULTIPLIER должен быть >= 1.0, получено: {TP_ATR_MULTIPLIER}")

    if SL_ATR_MULTIPLIER < 0.5:
        errors.append(f"SL_ATR_MULTIPLIER должен быть >= 0.5, получено: {SL_ATR_MULTIPLIER}")

    if TP_ATR_MULTIPLIER <= SL_ATR_MULTIPLIER:
        errors.append(f"TP_ATR_MULTIPLIER ({TP_ATR_MULTIPLIER}) должен быть > SL_ATR_MULTIPLIER ({SL_ATR_MULTIPLIER})")

    # Проверка риск-менеджмента
    if not (0.001 <= MIN_RISK_PER_POSITION <= 0.05):
        errors.append(f"MIN_RISK_PER_POSITION должен быть в диапазоне [0.001, 0.05], получено: {MIN_RISK_PER_POSITION}")

    if not (0.001 <= MAX_RISK_PER_POSITION <= 0.1):
        errors.append(f"MAX_RISK_PER_POSITION должен быть в диапазоне [0.001, 0.1], получено: {MAX_RISK_PER_POSITION}")

    if MAX_RISK_PER_POSITION < MIN_RISK_PER_POSITION:
        errors.append(f"MAX_RISK_PER_POSITION ({MAX_RISK_PER_POSITION}) < MIN_RISK_PER_POSITION ({MIN_RISK_PER_POSITION})")

    if not (MIN_RISK_PER_POSITION <= DEFAULT_RISK_PER_POSITION <= MAX_RISK_PER_POSITION):
        errors.append(f"DEFAULT_RISK_PER_POSITION ({DEFAULT_RISK_PER_POSITION}) должен быть между MIN и MAX")

    # Проверка EV параметров
    if MIN_EV_THRESHOLD < 0:
        errors.append(f"MIN_EV_THRESHOLD должен быть >= 0, получено: {MIN_EV_THRESHOLD}")

    if TOP_N_COINS < 1 or TOP_N_COINS > 50:
        errors.append(f"TOP_N_COINS должен быть в диапазоне [1, 50], получено: {TOP_N_COINS}")

    # Проверка таймфреймов
    valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    for tf_name, tf_config in CANDLE_TIMEFRAMES.items():
        if tf_config['interval'] not in valid_intervals:
            errors.append(f"Неверный интервал {tf_config['interval']} в {tf_name}")

        if tf_config['limit'] < 100:
            errors.append(f"Limit для {tf_name} должен быть >= 100, получено: {tf_config['limit']}")

        if not (0 <= tf_config['weight'] <= 1):
            errors.append(f"Weight для {tf_name} должен быть в [0, 1], получено: {tf_config['weight']}")

    # Проверка суммы весов (должна быть ~1.0)
    total_weight = sum(tf['weight'] for tf in CANDLE_TIMEFRAMES.values())
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"Сумма весов таймфреймов должна быть ≈ 1.0, получено: {total_weight:.3f}")

    if PRIMARY_TIMEFRAME not in CANDLE_TIMEFRAMES:
        errors.append(f"PRIMARY_TIMEFRAME ({PRIMARY_TIMEFRAME}) не найден в CANDLE_TIMEFRAMES")

    # Проверка adaptive threshold
    if not (0.5 <= BASE_THRESHOLD_INITIAL <= 0.9):
        errors.append(f"BASE_THRESHOLD_INITIAL должен быть в [0.5, 0.9], получено: {BASE_THRESHOLD_INITIAL}")

    if not (0.5 <= BASE_THRESHOLD_TRAINED <= 0.9):
        errors.append(f"BASE_THRESHOLD_TRAINED должен быть в [0.5, 0.9], получено: {BASE_THRESHOLD_TRAINED}")

    if DELTA_MIN >= DELTA_MAX:
        errors.append(f"DELTA_MIN ({DELTA_MIN}) должен быть < DELTA_MAX ({DELTA_MAX})")

    # Проверка META режима
    if META_MODE not in ['SHADOW', 'ACTIVE']:
        errors.append(f"META_MODE должен быть 'SHADOW' или 'ACTIVE', получено: {META_MODE}")

    # Проверка путей
    required_dirs = [DATA_DIR, MODELS_DIR, LOGS_DIR, POSITIONS_DIR, BACKUPS_DIR]
    for dir_path in required_dirs:
        if not dir_path.exists():
            errors.append(f"Директория не существует: {dir_path}")

    # Проверка Telegram (если включен)
    if TELEGRAM_ALERTS_ENABLED:
        if not TELEGRAM_BOT_TOKEN:
            errors.append("TELEGRAM_ALERTS_ENABLED=True, но TELEGRAM_BOT_TOKEN пуст")
        if not TELEGRAM_CHAT_ID:
            errors.append("TELEGRAM_ALERTS_ENABLED=True, но TELEGRAM_CHAT_ID пуст")

    # Проверка rate limiting
    if MAX_REQUESTS_PER_MINUTE < 100 or MAX_REQUESTS_PER_MINUTE > 1200:
        errors.append(f"MAX_REQUESTS_PER_MINUTE должен быть в [100, 1200], получено: {MAX_REQUESTS_PER_MINUTE}")

    valid = len(errors) == 0

    return valid, errors


def print_config_summary():
    """Вывод краткой сводки конфигурации"""
    print("=" * 80)
    print("BINANCE BOT CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Mode: {'PAPER TRADING' if PAPER_TRADING_MODE else 'LIVE TRADING'}")
    print(f"Testnet: {USE_TESTNET}")
    print(f"Max Positions: {MAX_POSITIONS}")
    print(f"Position Size: {MIN_POSITION_SIZE_USDT} - {MAX_POSITION_SIZE_USDT} USDT")
    print(f"Leverage: {LEVERAGE}x")
    print(f"Risk per Position: {MIN_RISK_PER_POSITION*100:.1f}% - {MAX_RISK_PER_POSITION*100:.1f}%")
    print(f"TP/SL: {TP_ATR_MULTIPLIER}x ATR / {SL_ATR_MULTIPLIER}x ATR")
    print(f"Min EV Threshold: {MIN_EV_THRESHOLD*100:.1f}%")
    print(f"Top N Coins: {TOP_N_COINS}")
    print(f"Primary Timeframe: {PRIMARY_TIMEFRAME}")
    print(f"META Mode: {META_MODE}")
    print(f"Trailing Stop: {'Enabled' if TRAILING_STOP_ENABLED else 'Disabled'}")
    print(f"Black Swan Protection: {'Enabled' if BLACK_SWAN_PROTECTION else 'Disabled'}")
    print(f"Online Learning: {'Enabled' if ONLINE_LEARNING_ENABLED else 'Disabled'}")
    print(f"Telegram Alerts: {'Enabled' if TELEGRAM_ALERTS_ENABLED else 'Disabled'}")
    print("=" * 80)


# ============================================================================
# AUTO-VALIDATION ON IMPORT
# ============================================================================

if __name__ != '__main__':
    # Автоматическая валидация при импорте
    valid, errors = validate_config()
    if not valid:
        logger.warning("⚠️  Конфигурация содержит ошибки:")
        for error in errors:
            logger.warning(f"  - {error}")


# ============================================================================
# MAIN (для тестирования)
# ============================================================================

if __name__ == '__main__':
    print_config_summary()
    print()

    print("Валидация конфигурации...")
    valid, errors = validate_config()

    if valid:
        print("✅ Конфигурация корректна")
    else:
        print(f"❌ Найдено ошибок: {len(errors)}")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
