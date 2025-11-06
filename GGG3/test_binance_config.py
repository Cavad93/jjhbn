#!/usr/bin/env python3
"""
Тесты для binance_config.py

Проверка корректности всех параметров конфигурации
"""

import sys
import os
from pathlib import Path


def test_imports():
    """Тест 1: Импорт конфигурации"""
    print("Тест 1: Импорт конфигурации...", end=" ")

    try:
        import binance_config
        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_paper_trading_mode():
    """Тест 2: Paper trading mode"""
    print("Тест 2: Paper trading mode...", end=" ")

    try:
        import binance_config as config

        assert hasattr(config, 'PAPER_TRADING_MODE')
        assert isinstance(config.PAPER_TRADING_MODE, bool)
        assert config.PAPER_TRADING_MODE == True  # По умолчанию должен быть True

        assert hasattr(config, 'PAPER_INITIAL_BALANCE')
        assert config.PAPER_INITIAL_BALANCE > 0

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_api_credentials():
    """Тест 3: API credentials"""
    print("Тест 3: API credentials...", end=" ")

    try:
        import binance_config as config

        assert hasattr(config, 'BINANCE_API_KEY')
        assert hasattr(config, 'BINANCE_SECRET_KEY')
        assert hasattr(config, 'USE_TESTNET')

        assert isinstance(config.BINANCE_API_KEY, str)
        assert isinstance(config.BINANCE_SECRET_KEY, str)
        assert isinstance(config.USE_TESTNET, bool)

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_trading_parameters():
    """Тест 4: Trading parameters"""
    print("Тест 4: Trading parameters...", end=" ")

    try:
        import binance_config as config

        # MAX_POSITIONS
        assert hasattr(config, 'MAX_POSITIONS')
        assert 1 <= config.MAX_POSITIONS <= 20
        assert config.MAX_POSITIONS == 10

        # Position sizes
        assert hasattr(config, 'MIN_POSITION_SIZE_USDT')
        assert hasattr(config, 'MAX_POSITION_SIZE_USDT')
        assert config.MIN_POSITION_SIZE_USDT >= 5.0
        assert config.MAX_POSITION_SIZE_USDT >= config.MIN_POSITION_SIZE_USDT

        # Leverage
        assert hasattr(config, 'LEVERAGE')
        assert 1 <= config.LEVERAGE <= 10
        assert config.LEVERAGE == 1  # Должен быть 1x

        # ATR parameters
        assert hasattr(config, 'ATR_PERIOD')
        assert hasattr(config, 'ATR_TIMEFRAME')
        assert 5 <= config.ATR_PERIOD <= 50
        assert config.ATR_PERIOD == 14

        # TP/SL multipliers
        assert hasattr(config, 'TP_ATR_MULTIPLIER')
        assert hasattr(config, 'SL_ATR_MULTIPLIER')
        assert config.TP_ATR_MULTIPLIER > config.SL_ATR_MULTIPLIER
        assert config.TP_ATR_MULTIPLIER == 2.5
        assert config.SL_ATR_MULTIPLIER == 1.5

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_risk_management():
    """Тест 5: Risk management"""
    print("Тест 5: Risk management...", end=" ")

    try:
        import binance_config as config

        # Risk per position
        assert hasattr(config, 'MIN_RISK_PER_POSITION')
        assert hasattr(config, 'MAX_RISK_PER_POSITION')
        assert hasattr(config, 'DEFAULT_RISK_PER_POSITION')

        assert 0.001 <= config.MIN_RISK_PER_POSITION <= 0.05
        assert 0.001 <= config.MAX_RISK_PER_POSITION <= 0.1
        assert config.MAX_RISK_PER_POSITION >= config.MIN_RISK_PER_POSITION

        assert config.MIN_RISK_PER_POSITION <= config.DEFAULT_RISK_PER_POSITION <= config.MAX_RISK_PER_POSITION
        assert config.DEFAULT_RISK_PER_POSITION == 0.009

        # Drawdown limits
        assert hasattr(config, 'MAX_DRAWDOWN_PCT')
        assert hasattr(config, 'MAX_DAILY_LOSS_PCT')
        assert config.MAX_DRAWDOWN_PCT > 0
        assert config.MAX_DAILY_LOSS_PCT > 0

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_ev_filter():
    """Тест 6: EV filter & coin selection"""
    print("Тест 6: EV filter & coin selection...", end=" ")

    try:
        import binance_config as config

        # EV threshold
        assert hasattr(config, 'MIN_EV_THRESHOLD')
        assert config.MIN_EV_THRESHOLD >= 0
        assert config.MIN_EV_THRESHOLD == 0.02

        # Top N coins
        assert hasattr(config, 'TOP_N_COINS')
        assert 1 <= config.TOP_N_COINS <= 50
        assert config.TOP_N_COINS == 10

        # Volume filter
        assert hasattr(config, 'MIN_VOLUME_24H_USDT')
        assert config.MIN_VOLUME_24H_USDT > 0

        # Excluded symbols
        assert hasattr(config, 'EXCLUDED_SYMBOLS')
        assert isinstance(config.EXCLUDED_SYMBOLS, list)

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_multi_timeframe():
    """Тест 7: Multi-timeframe configuration"""
    print("Тест 7: Multi-timeframe configuration...", end=" ")

    try:
        import binance_config as config

        assert hasattr(config, 'CANDLE_TIMEFRAMES')
        assert isinstance(config.CANDLE_TIMEFRAMES, dict)

        # Проверка наличия всех требуемых таймфреймов
        required_timeframes = ['5m', '15m', '30m', '4h']
        for tf in required_timeframes:
            assert tf in config.CANDLE_TIMEFRAMES, f"Отсутствует таймфрейм {tf}"

            tf_config = config.CANDLE_TIMEFRAMES[tf]
            assert 'interval' in tf_config
            assert 'limit' in tf_config
            assert 'weight' in tf_config

            assert tf_config['limit'] >= 100
            assert 0 <= tf_config['weight'] <= 1

        # Проверка суммы весов
        total_weight = sum(tf['weight'] for tf in config.CANDLE_TIMEFRAMES.values())
        assert abs(total_weight - 1.0) < 0.01, f"Сумма весов {total_weight:.3f} != 1.0"

        # Primary timeframe
        assert hasattr(config, 'PRIMARY_TIMEFRAME')
        assert config.PRIMARY_TIMEFRAME in config.CANDLE_TIMEFRAMES
        assert config.PRIMARY_TIMEFRAME == '4h'

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_adaptive_threshold():
    """Тест 8: Adaptive threshold"""
    print("Тест 8: Adaptive threshold...", end=" ")

    try:
        import binance_config as config

        assert hasattr(config, 'BASE_THRESHOLD_INITIAL')
        assert hasattr(config, 'BASE_THRESHOLD_TRAINED')
        assert 0.5 <= config.BASE_THRESHOLD_INITIAL <= 0.9
        assert 0.5 <= config.BASE_THRESHOLD_TRAINED <= 0.9

        assert hasattr(config, 'DELTA_MIN')
        assert hasattr(config, 'DELTA_MAX')
        assert config.DELTA_MIN < config.DELTA_MAX

        assert hasattr(config, 'ADAPTIVE_SMOOTHING_FACTOR')
        assert hasattr(config, 'MIN_TRADES_FOR_ADAPTIVE')

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_meta_network():
    """Тест 9: META neural network"""
    print("Тест 9: META neural network...", end=" ")

    try:
        import binance_config as config

        # META mode
        assert hasattr(config, 'META_MODE')
        assert config.META_MODE in ['SHADOW', 'ACTIVE']
        assert config.META_MODE == 'SHADOW'

        # Activation criteria
        assert hasattr(config, 'META_ACTIVATION_CRITERIA')
        assert isinstance(config.META_ACTIVATION_CRITERIA, dict)
        assert 'min_shadow_trades' in config.META_ACTIVATION_CRITERIA
        assert 'min_shadow_profit' in config.META_ACTIVATION_CRITERIA
        assert 'min_win_rate' in config.META_ACTIVATION_CRITERIA

        # Revert criteria
        assert hasattr(config, 'META_REVERT_CRITERIA')
        assert isinstance(config.META_REVERT_CRITERIA, dict)

        # Network config
        assert hasattr(config, 'META_NETWORK_CONFIG')
        assert isinstance(config.META_NETWORK_CONFIG, dict)
        assert 'input_size' in config.META_NETWORK_CONFIG
        assert 'hidden_layers' in config.META_NETWORK_CONFIG
        assert 'output_size' in config.META_NETWORK_CONFIG

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_special_modes():
    """Тест 10: Special modes"""
    print("Тест 10: Special modes...", end=" ")

    try:
        import binance_config as config

        # Trailing stop
        assert hasattr(config, 'TRAILING_STOP_ENABLED')
        assert isinstance(config.TRAILING_STOP_ENABLED, bool)
        assert hasattr(config, 'TRAILING_STOP_ACTIVATION_PCT')
        assert hasattr(config, 'TRAILING_STOP_DISTANCE_PCT')

        # Black swan protection
        assert hasattr(config, 'BLACK_SWAN_PROTECTION')
        assert isinstance(config.BLACK_SWAN_PROTECTION, bool)
        assert hasattr(config, 'BLACK_SWAN_PRICE_DROP_PCT')
        assert hasattr(config, 'BLACK_SWAN_CHECK_INTERVAL_SEC')

        # Night mode
        assert hasattr(config, 'NIGHT_MODE_ENABLED')
        assert isinstance(config.NIGHT_MODE_ENABLED, bool)
        assert hasattr(config, 'NIGHT_MODE_START_HOUR')
        assert hasattr(config, 'NIGHT_MODE_END_HOUR')
        assert 0 <= config.NIGHT_MODE_START_HOUR < 24
        assert 0 <= config.NIGHT_MODE_END_HOUR < 24

        # Sector diversification
        assert hasattr(config, 'SECTOR_DIVERSIFICATION')
        assert isinstance(config.SECTOR_DIVERSIFICATION, bool)
        assert hasattr(config, 'MAX_POSITIONS_PER_SECTOR')
        assert hasattr(config, 'SECTORS')
        assert isinstance(config.SECTORS, dict)

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_auto_reinvestment():
    """Тест 11: Auto-reinvestment"""
    print("Тест 11: Auto-reinvestment...", end=" ")

    try:
        import binance_config as config

        assert hasattr(config, 'AUTO_REINVEST_ENABLED')
        assert isinstance(config.AUTO_REINVEST_ENABLED, bool)

        assert hasattr(config, 'REINVEST_PROFIT_TO_RESERVE_PCT')
        assert 0 <= config.REINVEST_PROFIT_TO_RESERVE_PCT <= 1

        assert hasattr(config, 'MIN_PROFIT_TO_REINVEST')
        assert config.MIN_PROFIT_TO_REINVEST > 0

        assert hasattr(config, 'REINVEST_POSITION_SIZE_INCREASE_PCT')
        assert config.REINVEST_POSITION_SIZE_INCREASE_PCT > 0

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_online_learning():
    """Тест 12: Online learning"""
    print("Тест 12: Online learning...", end=" ")

    try:
        import binance_config as config

        assert hasattr(config, 'ONLINE_LEARNING_ENABLED')
        assert isinstance(config.ONLINE_LEARNING_ENABLED, bool)

        assert hasattr(config, 'MIN_CLOSED_POSITIONS_FOR_RETRAIN')
        assert config.MIN_CLOSED_POSITIONS_FOR_RETRAIN > 0

        assert hasattr(config, 'MAX_TRAINING_SAMPLES')
        assert config.MAX_TRAINING_SAMPLES > 0

        assert hasattr(config, 'RETRAIN_EVERY_N_POSITIONS')
        assert config.RETRAIN_EVERY_N_POSITIONS > 0

        assert hasattr(config, 'USE_ONLY_PROFITABLE_FOR_TRAINING')
        assert isinstance(config.USE_ONLY_PROFITABLE_FOR_TRAINING, bool)

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_paths():
    """Тест 13: Paths configuration"""
    print("Тест 13: Paths configuration...", end=" ")

    try:
        import binance_config as config

        # Директории
        assert hasattr(config, 'BASE_DIR')
        assert hasattr(config, 'DATA_DIR')
        assert hasattr(config, 'MODELS_DIR')
        assert hasattr(config, 'LOGS_DIR')
        assert hasattr(config, 'POSITIONS_DIR')
        assert hasattr(config, 'BACKUPS_DIR')

        # Проверка, что директории созданы
        assert config.DATA_DIR.exists(), f"DATA_DIR не существует: {config.DATA_DIR}"
        assert config.MODELS_DIR.exists(), f"MODELS_DIR не существует: {config.MODELS_DIR}"
        assert config.LOGS_DIR.exists(), f"LOGS_DIR не существует: {config.LOGS_DIR}"
        assert config.POSITIONS_DIR.exists(), f"POSITIONS_DIR не существует: {config.POSITIONS_DIR}"
        assert config.BACKUPS_DIR.exists(), f"BACKUPS_DIR не существует: {config.BACKUPS_DIR}"

        # Файлы
        assert hasattr(config, 'PAPER_EXCHANGE_STATE_FILE')
        assert hasattr(config, 'POSITIONS_FILE')
        assert hasattr(config, 'CLOSED_POSITIONS_FILE')
        assert hasattr(config, 'MODEL_XGB_FILE')
        assert hasattr(config, 'MODEL_META_FILE')

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_telegram_alerts():
    """Тест 14: Telegram alerts"""
    print("Тест 14: Telegram alerts...", end=" ")

    try:
        import binance_config as config

        assert hasattr(config, 'TELEGRAM_BOT_TOKEN')
        assert hasattr(config, 'TELEGRAM_CHAT_ID')
        assert hasattr(config, 'TELEGRAM_ALERTS_ENABLED')

        assert isinstance(config.TELEGRAM_BOT_TOKEN, str)
        assert isinstance(config.TELEGRAM_CHAT_ID, str)
        assert isinstance(config.TELEGRAM_ALERTS_ENABLED, bool)

        assert hasattr(config, 'TELEGRAM_ALERT_TYPES')
        assert isinstance(config.TELEGRAM_ALERT_TYPES, dict)

        assert hasattr(config, 'DAILY_SUMMARY_HOUR')
        assert 0 <= config.DAILY_SUMMARY_HOUR < 24

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_rate_limiting():
    """Тест 15: Rate limiting"""
    print("Тест 15: Rate limiting...", end=" ")

    try:
        import binance_config as config

        assert hasattr(config, 'MAX_REQUESTS_PER_MINUTE')
        assert 100 <= config.MAX_REQUESTS_PER_MINUTE <= 1200

        assert hasattr(config, 'MAIN_LOOP_INTERVAL_SEC')
        assert config.MAIN_LOOP_INTERVAL_SEC > 0

        assert hasattr(config, 'API_REQUEST_DELAY_SEC')
        assert config.API_REQUEST_DELAY_SEC > 0

        assert hasattr(config, 'MAX_API_RETRIES')
        assert config.MAX_API_RETRIES > 0

        assert hasattr(config, 'API_RETRY_DELAY_SEC')
        assert config.API_RETRY_DELAY_SEC > 0

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_validation_function():
    """Тест 16: Validation function"""
    print("Тест 16: Validation function...", end=" ")

    try:
        import binance_config as config

        # Проверка наличия функции
        assert hasattr(config, 'validate_config')
        assert callable(config.validate_config)

        # Вызов функции валидации
        valid, errors = config.validate_config()

        # Результат должен быть tuple из bool и list
        assert isinstance(valid, bool), "validate_config должна возвращать (bool, list)"
        assert isinstance(errors, list), "validate_config должна возвращать (bool, list)"

        # При корректной конфигурации должно быть valid=True и errors=[]
        if not valid:
            print(f"\n  Предупреждение: найдены ошибки валидации:")
            for error in errors:
                print(f"    - {error}")

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_print_config_summary():
    """Тест 17: Print config summary function"""
    print("Тест 17: Print config summary function...", end=" ")

    try:
        import binance_config as config

        assert hasattr(config, 'print_config_summary')
        assert callable(config.print_config_summary)

        # Просто проверяем, что функция не падает
        # (не выводим результат, чтобы не замусоривать тесты)
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()  # Перехватываем вывод

        config.print_config_summary()

        sys.stdout = old_stdout  # Восстанавливаем вывод

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_full_validation():
    """Тест 18: Полная валидация конфигурации"""
    print("Тест 18: Полная валидация конфигурации...", end=" ")

    try:
        import binance_config as config

        valid, errors = config.validate_config()

        # Для paper trading режима конфигурация должна быть полностью валидна
        if config.PAPER_TRADING_MODE:
            if not valid:
                print("\n  Найдены ошибки валидации:")
                for error in errors:
                    print(f"    - {error}")
                return False

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def run_all_tests():
    """Запуск всех тестов"""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ BINANCE_CONFIG.PY")
    print("=" * 80)
    print()

    tests = [
        test_imports,
        test_paper_trading_mode,
        test_api_credentials,
        test_trading_parameters,
        test_risk_management,
        test_ev_filter,
        test_multi_timeframe,
        test_adaptive_threshold,
        test_meta_network,
        test_special_modes,
        test_auto_reinvestment,
        test_online_learning,
        test_paths,
        test_telegram_alerts,
        test_rate_limiting,
        test_validation_function,
        test_print_config_summary,
        test_full_validation,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print()
    print("=" * 80)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 80)

    passed = sum(results)
    total = len(results)

    print(f"Пройдено: {passed}/{total}")

    if passed == total:
        print()
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
        return True
    else:
        print()
        print(f"❌ ПРОВАЛЕНО ТЕСТОВ: {total - passed}")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
