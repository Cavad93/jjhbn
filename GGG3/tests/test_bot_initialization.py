#!/usr/bin/env python3
"""
Тест инициализации бота

Проверяет что бот может загрузить все модули и инициализировать компоненты.
БЕЗ реального запуска бота.
"""

import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_imports():
    """Тест: все импорты наших новых модулей работают"""
    print("\n=== Test 1: New Modules Imports ===")

    try:
        # Основные импорты
        from datetime import datetime
        from typing import Dict, List, Optional
        import logging

        print("✅ Standard library imports")

        # Config
        import binance_config as config
        print("✅ Config imported")

        # Risk management (наш новый модуль)
        from risk.trailing_stop import TrailingStopManager
        print("✅ Risk management modules (ATR-based trailing)")

        # Analytics (наши новые модули)
        from analytics.performance_tracker import PerformanceTracker, TradeStats, PerformanceMetrics
        from analytics.performance_notifier import PerformanceNotifier
        print("✅ Analytics modules (Performance system)")

        print("\n✅ ALL NEW MODULES IMPORTED SUCCESSFULLY")
        return True

    except ImportError as e:
        print(f"\n❌ IMPORT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_component_initialization():
    """Тест: инициализация компонентов"""
    print("\n=== Test 2: Component Initialization ===")

    try:
        import binance_config as config
        from risk.trailing_stop import TrailingStopManager
        from analytics.performance_tracker import PerformanceTracker
        from analytics.performance_notifier import PerformanceNotifier

        # Trailing Stop
        trailing_stop = TrailingStopManager()
        assert trailing_stop.enabled == config.TRAILING_STOP_ENABLED
        assert trailing_stop.activation_pct == config.TRAILING_STOP_ACTIVATION_PCT
        assert trailing_stop.distance_atr_mult == config.TRAILING_STOP_DISTANCE_ATR_MULT
        print("✅ TrailingStopManager initialized")

        # Performance Tracker
        tracker = PerformanceTracker(
            windows=config.PERFORMANCE_WINDOWS,
            history_dir=config.PERFORMANCE_HISTORY_DIR,
            enable_history=False  # Отключаем для теста
        )
        assert tracker.windows == config.PERFORMANCE_WINDOWS
        print("✅ PerformanceTracker initialized")

        # Performance Notifier
        notifier = PerformanceNotifier(telegram_notifier=None)
        assert notifier.telegram is None
        print("✅ PerformanceNotifier initialized")

        print("\n✅ ALL COMPONENTS INITIALIZED")
        return True

    except Exception as e:
        print(f"\n❌ INITIALIZATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_values():
    """Тест: правильность значений конфигурации"""
    print("\n=== Test 3: Configuration Values ===")

    try:
        import binance_config as config

        # Trailing Stop config
        assert config.TRAILING_STOP_ENABLED == True
        assert config.TRAILING_STOP_ACTIVATION_PCT == 3.0
        assert config.TRAILING_STOP_DISTANCE_ATR_MULT == 1.0
        print("✅ Trailing Stop configuration")

        # Performance Analytics config
        assert config.PERFORMANCE_ANALYTICS_ENABLED == True
        assert config.PERFORMANCE_MIN_TRADES == 50
        assert config.PERFORMANCE_CONFIDENCE_LEVEL == 0.90
        print("✅ Performance Analytics configuration")

        # Windows
        assert 'short' in config.PERFORMANCE_WINDOWS
        assert 'medium' in config.PERFORMANCE_WINDOWS
        assert 'long' in config.PERFORMANCE_WINDOWS
        assert 'all' in config.PERFORMANCE_WINDOWS
        print("✅ Analysis windows configuration")

        print("\n✅ ALL CONFIGURATION VALUES CORRECT")
        return True

    except AssertionError as e:
        print(f"\n❌ CONFIGURATION ERROR: {e}")
        return False


def test_integration_points():
    """Тест: точки интеграции с основным ботом"""
    print("\n=== Test 4: Integration Points ===")

    try:
        from risk.trailing_stop import TrailingStopManager
        from analytics.performance_tracker import PerformanceTracker, TradeStats
        from analytics.performance_notifier import PerformanceNotifier
        from datetime import datetime

        # Создаем компоненты
        trailing_stop = TrailingStopManager()
        tracker = PerformanceTracker(enable_history=False)
        notifier = PerformanceNotifier(telegram_notifier=None)

        # Симулируем позицию (как в боте)
        class MockPosition:
            symbol = 'BTCUSDT'
            direction = 'LONG'
            entry_price = 100.0
            sl_price = 97.0
            atr_value = 2.0

        position = MockPosition()

        # 1. Проверяем trailing stop (как в bot.close_position)
        current_price = 104.0
        new_sl = trailing_stop.check_trailing_stop(position, current_price)
        assert new_sl is not None, "Trailing should work with position"
        print("✅ Trailing stop integration point")

        # 2. Проверяем добавление сделки (как в bot.close_position)
        trade_stats = TradeStats(
            symbol='BTCUSDT',
            direction='LONG',
            entry_price=100.0,
            exit_price=103.0,
            pnl=3.0,
            pnl_pct=3.0,
            exit_reason='TRAILING_SL',
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            duration_hours=2.0
        )
        tracker.add_trade(trade_stats)
        assert len(tracker.trades) == 1, "Trade should be added"
        print("✅ Performance tracking integration point")

        # 3. Проверяем сброс позиции (как в bot.close_position)
        trailing_stop.reset_position('BTCUSDT')
        peak = trailing_stop.get_peak_price('BTCUSDT')
        assert peak is None, "Peak should be reset"
        print("✅ Position reset integration point")

        print("\n✅ ALL INTEGRATION POINTS WORKING")
        return True

    except Exception as e:
        print(f"\n❌ INTEGRATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Запуск всех тестов инициализации"""
    print("\n" + "="*60)
    print("ТЕСТЫ ИНИЦИАЛИЗАЦИИ БОТА")
    print("="*60)

    results = []
    results.append(test_imports())
    results.append(test_component_initialization())
    results.append(test_config_values())
    results.append(test_integration_points())

    if all(results):
        print("\n" + "="*60)
        print("✅ ВСЕ ТЕСТЫ ИНИЦИАЛИЗАЦИИ ПРОЙДЕНЫ!")
        print("="*60)
        print("\nБот готов к запуску с новыми компонентами:")
        print("  • ATR-based Trailing Stop")
        print("  • Performance Analytics System")
        print("  • Smart Notifications")
        return True
    else:
        print("\n" + "="*60)
        print("❌ НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ")
        print("="*60)
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
