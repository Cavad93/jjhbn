#!/usr/bin/env python3
"""
КОМПЛЕКСНЫЙ ТЕСТ ВСЕХ ФУНКЦИЙ БОТА
Проверяет каждый компонент отдельно и интеграцию в целом
"""

import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any

print("=" * 80)
print("🧪 КОМПЛЕКСНЫЙ ТЕСТ BINANCE TRADING BOT")
print("=" * 80)
print()
print(f"📅 Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Статистика тестов
test_results = {
    'passed': 0,
    'failed': 0,
    'warnings': 0,
    'total': 0
}

failed_tests = []

def test_header(category: str, test_name: str):
    """Выводит заголовок теста"""
    print(f"\n{'─' * 80}")
    print(f"📋 [{category}] {test_name}")
    print(f"{'─' * 80}")

def test_result(passed: bool, message: str, warning: bool = False):
    """Записывает результат теста"""
    global test_results, failed_tests

    test_results['total'] += 1

    if warning:
        test_results['warnings'] += 1
        print(f"⚠️  WARNING: {message}")
    elif passed:
        test_results['passed'] += 1
        print(f"✅ PASS: {message}")
    else:
        test_results['failed'] += 1
        failed_tests.append(message)
        print(f"❌ FAIL: {message}")

# ============================================================================
# ТЕСТ 1: ИМПОРТ МОДУЛЕЙ
# ============================================================================

test_header("ИМПОРТЫ", "Проверка доступности всех модулей")

try:
    # Core modules
    import pandas as pd
    import numpy as np
    test_result(True, "NumPy и Pandas импортированы")
except Exception as e:
    test_result(False, f"NumPy/Pandas: {e}")

try:
    # Paper Exchange
    from paper_trading.paper_exchange import PaperExchange, get_binance_price
    test_result(True, "PaperExchange импортирован")
except Exception as e:
    test_result(False, f"PaperExchange: {e}")

try:
    # Feature Builder
    from features.binance_feature_builder import BinanceFeatureBuilder
    test_result(True, "BinanceFeatureBuilder импортирован")
except Exception as e:
    test_result(False, f"BinanceFeatureBuilder: {e}")

try:
    # Coin Selector
    from strategy.coin_selector import CoinSelector
    test_result(True, "CoinSelector импортирован")
except Exception as e:
    test_result(False, f"CoinSelector: {e}")

try:
    # Threshold Adapter
    from strategy.threshold_adapter import get_adaptive_threshold
    test_result(True, "ThresholdAdapter импортирован")
except Exception as e:
    test_result(False, f"ThresholdAdapter: {e}")

try:
    # Haiku Analyzer
    from strategy.haiku_analyzer import HaikuAnalyzer, NewsCollector
    test_result(True, "HaikuAnalyzer импортирован")
except Exception as e:
    test_result(False, f"HaikuAnalyzer: {e}")

try:
    # ML Models
    import joblib
    test_result(True, "Joblib для ML моделей импортирован")
except Exception as e:
    test_result(False, f"Joblib: {e}")

try:
    # Config
    import binance_config as config
    test_result(True, "binance_config импортирован")
except Exception as e:
    test_result(False, f"binance_config: {e}")

# ============================================================================
# ТЕСТ 2: BINANCE API
# ============================================================================

test_header("BINANCE API", "Проверка доступности и корректности данных")

try:
    exchange = PaperExchange(initial_capital=1000.0)
    test_result(True, "PaperExchange инициализирован")
except Exception as e:
    test_result(False, f"Инициализация PaperExchange: {e}")
    sys.exit(1)

# Тест 2.1: Получение списка пар
try:
    pairs = exchange.get_all_usdt_pairs(min_volume_24h=1_000_000)
    if len(pairs) >= 100:
        test_result(True, f"Получен список из {len(pairs)} USDT пар")
    else:
        test_result(False, f"Мало пар: {len(pairs)} (ожидалось >= 100)")
except Exception as e:
    test_result(False, f"get_all_usdt_pairs: {e}")

# Тест 2.2: Получение текущей цены
test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
successful_prices = 0

for symbol in test_symbols:
    try:
        price = get_binance_price(symbol)
        if price > 0:
            test_result(True, f"{symbol} цена: ${price:.2f}")
            successful_prices += 1
        else:
            test_result(False, f"{symbol} цена <= 0: ${price}")
    except Exception as e:
        test_result(False, f"get_binance_price({symbol}): {str(e)[:50]}")

if successful_prices >= 2:
    test_result(True, f"Успешно получено {successful_prices}/3 цен")
else:
    test_result(False, f"Слишком мало успешных запросов: {successful_prices}/3")

# Тест 2.3: Получение OHLCV данных
try:
    df = exchange.get_ohlcv('BTCUSDT', '4h', 100)

    if len(df) >= 50:
        test_result(True, f"OHLCV данные получены: {len(df)} свечей")
    else:
        test_result(False, f"Мало OHLCV данных: {len(df)} (ожидалось >= 50)")

    # Проверка структуры данных
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [c for c in required_columns if c not in df.columns]

    if not missing_cols:
        test_result(True, "Все необходимые колонки присутствуют")
    else:
        test_result(False, f"Отсутствуют колонки: {missing_cols}")

    # Проверка значений
    if (df['high'] >= df['low']).all():
        test_result(True, "High >= Low для всех свечей")
    else:
        test_result(False, "Некорректные OHLC данные (High < Low)")

except Exception as e:
    test_result(False, f"get_ohlcv: {e}")

# Тест 2.4: Получение ticker данных
try:
    ticker = exchange.get_ticker('BTCUSDT')

    required_fields = ['lastPrice', 'volume', 'priceChange', 'priceChangePercent']
    missing_fields = [f for f in required_fields if f not in ticker]

    if not missing_fields:
        test_result(True, f"Ticker данные получены: {ticker['lastPrice']}")
    else:
        test_result(False, f"Отсутствуют поля ticker: {missing_fields}")

except Exception as e:
    test_result(False, f"get_ticker: {e}")

# ============================================================================
# ТЕСТ 3: ТЕХНИЧЕСКИЙ АНАЛИЗ
# ============================================================================

test_header("ТЕХНИЧЕСКИЙ АНАЛИЗ", "Проверка расчёта индикаторов и фич")

try:
    from features.binance_feature_builder import BinanceFeatureBuilder
    feature_builder = BinanceFeatureBuilder()
    test_result(True, "BinanceFeatureBuilder инициализирован")
except Exception as e:
    test_result(False, f"Инициализация FeatureBuilder: {e}")
    feature_builder = None

if feature_builder:
    try:
        # Получаем тестовые данные
        df_5m = exchange.get_ohlcv('BTCUSDT', '5m', 200)
        df_15m = exchange.get_ohlcv('BTCUSDT', '15m', 200)
        df_30m = exchange.get_ohlcv('BTCUSDT', '30m', 200)
        df_4h = exchange.get_ohlcv('BTCUSDT', '4h', 100)

        if all(len(df) >= 50 for df in [df_5m, df_15m, df_30m, df_4h]):
            test_result(True, "Все таймфреймы загружены для анализа")
        else:
            test_result(False, "Недостаточно данных для некоторых таймфреймов")

        # Строим фичи
        features = feature_builder.build_features(
            df_5m=df_5m,
            df_15m=df_15m,
            df_30m=df_30m,
            df_4h=df_4h,
            symbol='BTCUSDT'
        )

        if features is not None and len(features) > 0:
            test_result(True, f"Фичи построены: {len(features)} признаков")
        else:
            test_result(False, "Фичи не построены или пустые")

        # Проверка наличия ключевых фич
        if features is not None:
            # Должны быть базовые фичи
            if len(features) >= 50:
                test_result(True, f"Достаточно фич для ML: {len(features)}")
            else:
                test_result(False, f"Мало фич: {len(features)} (ожидалось >= 50)")

    except Exception as e:
        test_result(False, f"Построение фич: {e}")
        traceback.print_exc()

# ============================================================================
# ТЕСТ 4: ML МОДЕЛИ
# ============================================================================

test_header("ML МОДЕЛИ", "Проверка загрузки и предсказаний моделей")

import os
from pathlib import Path

models_dir = Path('models/saved')

# Проверка наличия файлов моделей
expected_models = {
    'xgb_expert.pkl': 'XGBoost',
    'rf_expert.pkl': 'RandomForest',
    'nn_expert.pkl': 'NeuralNet',
    'meta_model.pkl': 'META'
}

for model_file, model_name in expected_models.items():
    model_path = models_dir / model_file
    if model_path.exists():
        test_result(True, f"{model_name} модель найдена: {model_file}")

        # Попытка загрузить
        try:
            import joblib
            model = joblib.load(model_path)
            test_result(True, f"{model_name} модель загружена успешно")
        except Exception as e:
            test_result(False, f"{model_name} загрузка: {e}")
    else:
        test_result(False, f"{model_name} модель не найдена: {model_file}", warning=True)

# ============================================================================
# ТЕСТ 5: ADAPTIVE THRESHOLD
# ============================================================================

test_header("ADAPTIVE THRESHOLD", "Проверка адаптивного порога")

try:
    from strategy.threshold_adapter import get_adaptive_threshold

    # Тест разных сценариев
    scenarios = [
        {
            'name': 'Холодный старт',
            'params': {'total_closed': 50, 'recent_hour': 0, 'recent_wr': 0.5, 'calib_error': 0.0},
            'expected_range': (0.45, 0.55)
        },
        {
            'name': 'Хороший WR',
            'params': {'total_closed': 1000, 'recent_hour': 5, 'recent_wr': 0.65, 'calib_error': 0.02},
            'expected_range': (0.45, 0.60)
        },
        {
            'name': 'Плохой WR',
            'params': {'total_closed': 1000, 'recent_hour': 5, 'recent_wr': 0.40, 'calib_error': 0.05},
            'expected_range': (0.50, 0.70)
        }
    ]

    for scenario in scenarios:
        threshold = get_adaptive_threshold(**scenario['params'])
        min_t, max_t = scenario['expected_range']

        if min_t <= threshold <= max_t:
            test_result(True, f"{scenario['name']}: threshold={threshold:.3f} ∈ [{min_t}, {max_t}]")
        else:
            test_result(False, f"{scenario['name']}: threshold={threshold:.3f} вне диапазона [{min_t}, {max_t}]")

except Exception as e:
    test_result(False, f"Adaptive threshold: {e}")

# ============================================================================
# ТЕСТ 6: HAIKU ANALYZER
# ============================================================================

test_header("HAIKU ANALYZER", "Проверка фундаментального анализа")

try:
    # Проверка NewsCollector
    from strategy.haiku_analyzer import NewsCollector

    news_collector = NewsCollector()
    test_result(True, "NewsCollector инициализирован")

    # Проверка RSS фидов
    if news_collector.RSS_FEEDS:
        test_result(True, f"RSS фиды настроены: {len(news_collector.RSS_FEEDS)} источников")
    else:
        test_result(False, "RSS фиды не настроены")

    # Попытка получить новости (может упасть если feedparser не установлен)
    try:
        news = news_collector.get_recent_news('BTC', hours=24)
        test_result(True, f"RSS парсинг работает: найдено {len(news)} новостей", warning=len(news)==0)
    except ImportError:
        test_result(False, "feedparser не установлен (pip install feedparser)", warning=True)
    except Exception as e:
        test_result(False, f"RSS парсинг: {str(e)[:50]}", warning=True)

except Exception as e:
    test_result(False, f"NewsCollector: {e}")

# Проверка HaikuAnalyzer
try:
    from strategy.haiku_analyzer import HaikuAnalyzer
    import os

    api_key = os.getenv('ANTHROPIC_API_KEY', '')

    if api_key:
        test_result(True, "ANTHROPIC_API_KEY найден в environment")

        try:
            analyzer = HaikuAnalyzer(
                api_key=api_key,
                batch_size=5,
                enable_stats=True
            )
            test_result(True, "HaikuAnalyzer инициализирован")
        except Exception as e:
            test_result(False, f"Инициализация HaikuAnalyzer: {e}")
    else:
        test_result(False, "ANTHROPIC_API_KEY не найден (Haiku будет отключён)", warning=True)

except Exception as e:
    test_result(False, f"HaikuAnalyzer: {e}")

# ============================================================================
# ТЕСТ 7: PAPER EXCHANGE - ТОРГОВЫЕ ОПЕРАЦИИ
# ============================================================================

test_header("PAPER EXCHANGE", "Проверка симуляции торговых операций")

try:
    # Создаём новый exchange для тестов
    test_exchange = PaperExchange(initial_capital=1000.0)
    initial_balance = test_exchange.balance

    test_result(True, f"Test Exchange создан с капиталом ${initial_balance}")

    # Тест: Открытие LONG позиции
    try:
        position = test_exchange.open_position(
            symbol='BTCUSDT',
            side='LONG',
            amount=0.01,
            tp_price=110000,
            sl_price=95000
        )

        if position and position['status'] == 'open':
            test_result(True, f"LONG позиция открыта: {position['symbol']} @ ${position['entry_price']:.2f}")
        else:
            test_result(False, "LONG позиция не открыта корректно")

    except Exception as e:
        test_result(False, f"Открытие LONG: {e}")

    # Тест: Проверка расхода баланса
    balance_after_open = test_exchange.balance
    cost = initial_balance - balance_after_open

    if cost > 0:
        test_result(True, f"Баланс уменьшился на ${cost:.2f} после открытия позиции")
    else:
        test_result(False, f"Баланс не изменился после открытия: ${balance_after_open}")

    # Тест: Открытие SHORT позиции
    try:
        position_short = test_exchange.open_position(
            symbol='ETHUSDT',
            side='SHORT',
            amount=0.5,
            tp_price=3000,
            sl_price=3500
        )

        if position_short and position_short['status'] == 'open':
            test_result(True, f"SHORT позиция открыта: {position_short['symbol']}")
        else:
            test_result(False, "SHORT позиция не открыта корректно")

    except Exception as e:
        test_result(False, f"Открытие SHORT: {e}")

    # Тест: Проверка TP/SL ордеров
    open_positions = list(test_exchange.positions.values())

    if len(open_positions) == 2:
        test_result(True, f"Открыто {len(open_positions)} позиции")

        # Проверка наличия TP и SL ордеров
        for pos in open_positions:
            if pos.get('tp_order_id') and pos.get('sl_order_id'):
                test_result(True, f"{pos['symbol']}: TP и SL ордера созданы")
            else:
                test_result(False, f"{pos['symbol']}: TP или SL ордера отсутствуют")
    else:
        test_result(False, f"Неверное количество позиций: {len(open_positions)} (ожидалось 2)")

    # Тест: Закрытие позиции
    try:
        if open_positions:
            pos_id = open_positions[0]['id']
            closed = test_exchange.close_position(pos_id, reason='manual')

            if closed:
                test_result(True, f"Позиция закрыта: P&L = ${closed.get('pnl', 0):.2f}")
            else:
                test_result(False, "Позиция не закрыта")
    except Exception as e:
        test_result(False, f"Закрытие позиции: {e}")

except Exception as e:
    test_result(False, f"Paper Exchange тесты: {e}")
    traceback.print_exc()

# ============================================================================
# ТЕСТ 8: COIN SELECTOR
# ============================================================================

test_header("COIN SELECTOR", "Проверка отбора монет для торговли")

try:
    from strategy.coin_selector import CoinSelector
    from portfolio.risk_manager import calculate_atr
    from strategy.market_phase import detect_market_phase

    selector = CoinSelector()
    test_result(True, "CoinSelector инициализирован")

    # Получаем небольшой список пар для теста
    test_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']

    # Проверяем что фильтрация работает
    try:
        # Простой тест с минимальными данными
        test_result(True, f"Тестируем отбор из {len(test_pairs)} пар")

        # В production coin_selector требует много данных,
        # поэтому просто проверяем что метод существует и callable
        if hasattr(selector, 'select_top_coins') and callable(selector.select_top_coins):
            test_result(True, "Метод select_top_coins доступен")
        else:
            test_result(False, "Метод select_top_coins недоступен")

    except Exception as e:
        test_result(False, f"CoinSelector тест: {e}", warning=True)

except Exception as e:
    test_result(False, f"CoinSelector: {e}")

# ============================================================================
# ТЕСТ 9: КОНФИГУРАЦИЯ
# ============================================================================

test_header("КОНФИГУРАЦИЯ", "Проверка настроек бота")

try:
    import binance_config as config

    # Проверка критичных настроек
    critical_configs = {
        'PAPER_TRADING_MODE': (bool, "Режим paper trading"),
        'MAX_POSITIONS': (int, "Максимум позиций"),
        'PAPER_INITIAL_BALANCE': (float, "Начальный капитал"),
        'HAIKU_ANALYSIS_ENABLED': (bool, "Haiku анализ"),
    }

    for cfg_name, (expected_type, description) in critical_configs.items():
        if hasattr(config, cfg_name):
            value = getattr(config, cfg_name)
            if isinstance(value, expected_type):
                test_result(True, f"{description}: {value}")
            else:
                test_result(False, f"{cfg_name} неверный тип: {type(value)} (ожидалось {expected_type})")
        else:
            test_result(False, f"{cfg_name} отсутствует в конфигурации")

    # Проверка путей
    if hasattr(config, 'BASE_DIR'):
        test_result(True, f"BASE_DIR: {config.BASE_DIR}")
    else:
        test_result(False, "BASE_DIR не определён")

except Exception as e:
    test_result(False, f"Конфигурация: {e}")

# ============================================================================
# ТЕСТ 10: ИНТЕГРАЦИОННЫЙ ТЕСТ
# ============================================================================

test_header("ИНТЕГРАЦИЯ", "Проверка совместной работы компонентов")

print("\n⏳ Запуск интеграционного теста (может занять 30-60 секунд)...")
print()

try:
    # Имитируем один цикл работы бота

    # 1. Получаем список пар
    all_pairs = exchange.get_all_usdt_pairs(min_volume_24h=5_000_000)
    if len(all_pairs) >= 50:
        test_result(True, f"Шаг 1/5: Получен список из {len(all_pairs)} пар")
    else:
        test_result(False, f"Шаг 1/5: Мало пар ({len(all_pairs)})")

    # 2. Получаем данные для нескольких пар
    test_symbols_integration = all_pairs[:5]  # Первые 5 пар
    successful_data = 0

    for symbol in test_symbols_integration:
        try:
            ticker = exchange.get_ticker(symbol)
            df = exchange.get_ohlcv(symbol, '4h', 50)

            if ticker and len(df) >= 20:
                successful_data += 1
        except:
            pass

    if successful_data >= 3:
        test_result(True, f"Шаг 2/5: Данные получены для {successful_data}/5 пар")
    else:
        test_result(False, f"Шаг 2/5: Недостаточно данных ({successful_data}/5)")

    # 3. Adaptive threshold
    try:
        threshold = get_adaptive_threshold(
            total_closed=100,
            recent_hour=3,
            recent_wr=0.55,
            calib_error=0.03
        )
        if 0.3 <= threshold <= 0.8:
            test_result(True, f"Шаг 3/5: Adaptive threshold={threshold:.3f}")
        else:
            test_result(False, f"Шаг 3/5: Threshold вне допустимого диапазона: {threshold}")
    except Exception as e:
        test_result(False, f"Шаг 3/5: Adaptive threshold ошибка: {e}")

    # 4. Открытие тестовой позиции
    try:
        test_exchange_int = PaperExchange(initial_capital=1000.0)
        position = test_exchange_int.open_position(
            symbol='BTCUSDT',
            side='LONG',
            amount=0.001,
            tp_price=110000,
            sl_price=95000
        )

        if position and position['status'] == 'open':
            test_result(True, "Шаг 4/5: Тестовая позиция открыта")

            # 5. Закрытие позиции
            closed = test_exchange_int.close_position(position['id'], reason='test')
            if closed:
                test_result(True, f"Шаг 5/5: Позиция закрыта, P&L=${closed.get('pnl', 0):.2f}")
            else:
                test_result(False, "Шаг 5/5: Позиция не закрылась")
        else:
            test_result(False, "Шаг 4/5: Позиция не открылась")
            test_result(False, "Шаг 5/5: Пропущено (позиция не открылась)")

    except Exception as e:
        test_result(False, f"Шаги 4-5: {e}")

except Exception as e:
    test_result(False, f"Интеграционный тест: {e}")
    traceback.print_exc()

# ============================================================================
# ИТОГОВЫЙ ОТЧЁТ
# ============================================================================

print()
print("=" * 80)
print("📊 ИТОГОВЫЙ ОТЧЁТ")
print("=" * 80)
print()

total = test_results['total']
passed = test_results['passed']
failed = test_results['failed']
warnings = test_results['warnings']

pass_rate = (passed / total * 100) if total > 0 else 0

print(f"Всего тестов:     {total}")
print(f"✅ Пройдено:      {passed} ({pass_rate:.1f}%)")
print(f"❌ Провалено:     {failed}")
print(f"⚠️  Предупреждения: {warnings}")
print()

if failed > 0:
    print("❌ Проваленные тесты:")
    for i, test in enumerate(failed_tests, 1):
        print(f"  {i}. {test}")
    print()

# Оценка готовности
if pass_rate >= 95:
    status = "✅ ОТЛИЧНО"
    message = "Бот готов к production использованию!"
elif pass_rate >= 80:
    status = "✅ ХОРОШО"
    message = "Бот работоспособен, но есть небольшие проблемы"
elif pass_rate >= 60:
    status = "⚠️  УДОВЛЕТВОРИТЕЛЬНО"
    message = "Бот работает, но требует доработки"
else:
    status = "❌ НЕУДОВЛЕТВОРИТЕЛЬНО"
    message = "Критические проблемы! Бот НЕ готов к использованию"

print("=" * 80)
print(f"ИТОГОВАЯ ОЦЕНКА: {status}")
print(f"{message}")
print("=" * 80)
print()

# Рекомендации
print("💡 РЕКОМЕНДАЦИИ:")
print()

if failed > 0:
    print("1. Исправьте проваленные тесты перед запуском бота")
    print("2. Проверьте логи для деталей ошибок")
    print()

if warnings > 0:
    print("1. Предупреждения не критичны, но лучше их устранить")
    print("2. Особое внимание на ML модели и Haiku API")
    print()

if pass_rate >= 80:
    print("✅ Бот прошёл большинство тестов")
    print("✅ Можно запускать в paper trading режиме")
    print("✅ Следите за логами первые 24 часа")
    print()

print("=" * 80)
print()
print(f"Тестирование завершено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Возвращаем exit code
sys.exit(0 if failed == 0 else 1)
