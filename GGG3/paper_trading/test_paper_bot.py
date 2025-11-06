"""
Test Paper Trading Bot

Быстрый тест для проверки работоспособности paper_bot_main.py
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Добавляем путь
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock для моделей
def mock_get_binance_price(symbol):
    """Mock цены"""
    prices = {
        'BTCUSDT': 45000.0,
        'ETHUSDT': 2500.0,
        'BNBUSDT': 350.0
    }
    return prices.get(symbol, 1000.0)


def mock_get_binance_kline(symbol, interval, limit):
    """Mock свечи"""
    price = mock_get_binance_price(symbol)
    return {
        'open': price * 0.99,
        'high': price * 1.01,
        'low': price * 0.98,
        'close': price,
        'volume': 1000000.0,
        'timestamp': int(time.time() * 1000)
    }


def test_imports():
    """Тест 1: Импорт модулей"""
    print("\n" + "=" * 80)
    print("TEST 1: Imports")
    print("=" * 80)

    try:
        from paper_trading import paper_bot_main
        print("✓ paper_bot_main imported successfully")

        # Проверяем основные классы
        assert hasattr(paper_bot_main, 'PaperTradingBot')
        assert hasattr(paper_bot_main, 'Config')
        assert hasattr(paper_bot_main, 'ModelsManager')
        assert hasattr(paper_bot_main, 'EVFilter')
        assert hasattr(paper_bot_main, 'ATRCalculator')

        print("✓ All classes found")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_configuration():
    """Тест 2: Конфигурация"""
    print("\n" + "=" * 80)
    print("TEST 2: Configuration")
    print("=" * 80)

    try:
        from paper_trading.paper_bot_main import Config

        config = Config()

        print(f"✓ MAX_POSITIONS: {config.MAX_POSITIONS}")
        print(f"✓ INITIAL_CAPITAL: {config.INITIAL_CAPITAL}")
        print(f"✓ ATR_PERIOD: {config.ATR_PERIOD}")
        print(f"✓ TP_MULTIPLIER: {config.TP_MULTIPLIER}")
        print(f"✓ SL_MULTIPLIER: {config.SL_MULTIPLIER}")

        assert config.MAX_POSITIONS > 0
        assert config.INITIAL_CAPITAL > 0
        assert config.TP_MULTIPLIER > config.SL_MULTIPLIER  # R/R > 1

        print("✓ Configuration valid")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_models_manager():
    """Тест 3: Менеджер моделей"""
    print("\n" + "=" * 80)
    print("TEST 3: Models Manager")
    print("=" * 80)

    try:
        from paper_trading.paper_bot_main import ModelsManager, Config

        config = Config()
        models_dir = config.MODELS_DIR

        manager = ModelsManager(models_dir)

        print(f"✓ Models directory: {models_dir}")
        print("✓ ModelsManager created")

        # Попытка загрузки (может не найти файлы - это нормально для теста)
        models = manager.load_models()

        print(f"✓ Models loaded: {sum(1 for v in models.values() if v is not None)}/{len(models)}")

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


@patch('paper_trading.paper_exchange.get_binance_price', side_effect=mock_get_binance_price)
@patch('paper_trading.paper_exchange.get_binance_kline', side_effect=mock_get_binance_kline)
def test_ev_filter(mock_kline, mock_price):
    """Тест 4: EV Filter"""
    print("\n" + "=" * 80)
    print("TEST 4: EV Filter")
    print("=" * 80)

    try:
        from paper_trading.paper_bot_main import EVFilter, Config

        config = Config()
        models = {}  # Пустые модели для теста

        ev_filter = EVFilter(models, config)

        # Тест расчета EV
        p_up = 0.62
        ev_long, ev_short = ev_filter.calculate_ev(p_up)

        print(f"✓ p_up = {p_up}")
        print(f"✓ EV_long = {ev_long:+.4f}")
        print(f"✓ EV_short = {ev_short:+.4f}")

        assert ev_long > 0  # При p_up=0.62 LONG должен быть прибыльным
        assert ev_short < 0  # SHORT убыточен

        # Тест с обратной вероятностью
        p_up = 0.38
        ev_long, ev_short = ev_filter.calculate_ev(p_up)

        print(f"\n✓ p_up = {p_up}")
        print(f"✓ EV_long = {ev_long:+.4f}")
        print(f"✓ EV_short = {ev_short:+.4f}")

        assert ev_long < 0  # При p_up=0.38 LONG убыточен
        assert ev_short > 0  # SHORT прибылен

        print("\n✓ EV Filter logic correct")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


@patch('paper_trading.paper_exchange.get_binance_price', side_effect=mock_get_binance_price)
@patch('paper_trading.paper_exchange.get_binance_kline', side_effect=mock_get_binance_kline)
def test_atr_calculator(mock_kline, mock_price):
    """Тест 5: ATR Calculator"""
    print("\n" + "=" * 80)
    print("TEST 5: ATR Calculator")
    print("=" * 80)

    try:
        from paper_trading.paper_bot_main import ATRCalculator

        calc = ATRCalculator()

        atr = calc.calculate_atr('BTCUSDT', period=10, timeframe='4h')

        print(f"✓ ATR for BTCUSDT: {atr:.2f}")

        assert atr > 0
        assert atr < 10000  # Разумный диапазон

        print("✓ ATR Calculator works")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


@patch('paper_trading.paper_exchange.get_binance_price', side_effect=mock_get_binance_price)
@patch('paper_trading.paper_exchange.get_binance_kline', side_effect=mock_get_binance_kline)
def test_bot_initialization(mock_kline, mock_price):
    """Тест 6: Инициализация бота"""
    print("\n" + "=" * 80)
    print("TEST 6: Bot Initialization")
    print("=" * 80)

    try:
        from paper_trading.paper_bot_main import PaperTradingBot

        bot = PaperTradingBot(
            initial_capital=1000.0,
            max_positions=5
        )

        print(f"✓ Bot initialized")
        print(f"✓ Exchange balance: {bot.exchange.get_balance():.2f} USDT")
        print(f"✓ Max positions: {bot.config.MAX_POSITIONS}")

        assert bot.exchange.get_balance() == 1000.0
        assert bot.config.MAX_POSITIONS == 5

        # Проверка методов
        assert hasattr(bot, 'update_portfolio')
        assert hasattr(bot, 'check_orders')
        assert hasattr(bot, 'open_position')

        print("✓ Bot initialization successful")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


@patch('paper_trading.paper_exchange.get_binance_price', side_effect=mock_get_binance_price)
@patch('paper_trading.paper_exchange.get_binance_kline', side_effect=mock_get_binance_kline)
def test_get_universe(mock_kline, mock_price):
    """Тест 7: Получение universe"""
    print("\n" + "=" * 80)
    print("TEST 7: Get Universe")
    print("=" * 80)

    try:
        from paper_trading.paper_bot_main import PaperTradingBot

        bot = PaperTradingBot(initial_capital=1000.0, max_positions=5)

        universe = bot.get_binance_universe()

        print(f"✓ Universe size: {len(universe)} symbols")
        print(f"✓ Sample symbols: {universe[:5]}")

        assert len(universe) > 0
        assert 'BTCUSDT' in universe or 'ETHUSDT' in universe

        print("✓ Universe retrieval works")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Запуск всех тестов"""
    print("\n" + "=" * 80)
    print("PAPER BOT TEST SUITE")
    print("=" * 80)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Models Manager", test_models_manager),
        ("EV Filter", test_ev_filter),
        ("ATR Calculator", test_atr_calculator),
        ("Bot Initialization", test_bot_initialization),
        ("Get Universe", test_get_universe),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:.<50} {status}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print("\n" + "=" * 80)
    if passed == total:
        print(f"✓ ALL TESTS PASSED ({passed}/{total})")
        print("=" * 80)
        return True
    else:
        print(f"✗ SOME TESTS FAILED ({passed}/{total} passed)")
        print("=" * 80)
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
