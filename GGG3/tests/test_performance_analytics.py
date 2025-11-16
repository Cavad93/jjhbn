#!/usr/bin/env python3
"""
Тесты для системы Performance Analytics

Проверяет:
- Расчет всех метрик
- Доверительные интервалы
- Геометрическую прогрессию
- Умные уведомления
- Автоматические рекомендации
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analytics.performance_tracker import PerformanceTracker, TradeStats, PerformanceMetrics
from analytics.performance_notifier import PerformanceNotifier


def create_mock_trade(pnl_pct: float, symbol: str = 'BTCUSDT', direction: str = 'LONG') -> TradeStats:
    """Создает мок-сделку с заданной прибылью"""
    entry_time = datetime.now() - timedelta(hours=2)
    exit_time = datetime.now()

    return TradeStats(
        symbol=symbol,
        direction=direction,
        entry_price=100.0,
        exit_price=100.0 * (1 + pnl_pct / 100),
        pnl=pnl_pct,  # Упрощенно
        pnl_pct=pnl_pct,
        exit_reason='TP' if pnl_pct > 0 else 'SL',
        entry_time=entry_time,
        exit_time=exit_time,
        duration_hours=2.0
    )


def test_basic_metrics():
    """Тест: базовые метрики для 50 сделок"""
    print("\n=== Test 1: Basic Metrics ===")

    tracker = PerformanceTracker(
        windows={'short': 50},
        enable_history=False
    )

    # Создаем 50 сделок: 30 wins (+4%) и 20 losses (-3%)
    for i in range(30):
        tracker.add_trade(create_mock_trade(+4.0))

    for i in range(20):
        tracker.add_trade(create_mock_trade(-3.0))

    # Получаем метрики
    metrics = tracker.get_metrics('short')

    assert metrics is not None, "Metrics should not be None"
    assert metrics.total_trades == 50, f"Expected 50 trades, got {metrics.total_trades}"
    assert metrics.win_count == 30, f"Expected 30 wins, got {metrics.win_count}"
    assert metrics.loss_count == 20, f"Expected 20 losses, got {metrics.loss_count}"
    assert metrics.win_rate == 0.6, f"Expected 60% win rate, got {metrics.win_rate*100}%"

    print(f"✅ Total trades: {metrics.total_trades}")
    print(f"✅ Win rate: {metrics.win_rate*100:.1f}%")
    print(f"✅ Wins: {metrics.win_count}, Losses: {metrics.loss_count}")


def test_required_win_rate():
    """Тест: расчет необходимого винрейта"""
    print("\n=== Test 2: Required Win Rate ===")

    tracker = PerformanceTracker(enable_history=False)

    # R:R = 1.33:1 (avg_win=4%, avg_loss=3%)
    for i in range(30):
        tracker.add_trade(create_mock_trade(+4.0))
    for i in range(20):
        tracker.add_trade(create_mock_trade(-3.0))

    metrics = tracker.get_metrics('short')

    # Required WR = 1 / (1 + R:R) = 1 / (1 + 4/3) = 1 / 2.33 = 0.43 (43%)
    expected_required_wr = 1 / (1 + (4.0 / 3.0))

    assert abs(metrics.required_win_rate - expected_required_wr) < 0.01, \
        f"Expected required WR {expected_required_wr:.2f}, got {metrics.required_win_rate:.2f}"

    print(f"✅ Average Win: {metrics.avg_win:.2f}%")
    print(f"✅ Average Loss: {metrics.avg_loss:.2f}%")
    print(f"✅ R:R: {metrics.avg_win_loss_ratio:.2f}:1")
    print(f"✅ Required Win Rate: {metrics.required_win_rate*100:.1f}%")


def test_geometric_progression():
    """Тест: геометрическая прогрессия"""
    print("\n=== Test 3: Geometric Progression ===")

    tracker = PerformanceTracker(enable_history=False)

    # 50 сделок с +2% каждая
    for i in range(50):
        tracker.add_trade(create_mock_trade(+2.0))

    metrics = tracker.get_metrics('short')

    # Геометрическое среднее должно быть близко к +2%
    assert abs(metrics.geometric_mean_return - 2.0) < 0.1, \
        f"Expected ~2% geometric mean, got {metrics.geometric_mean_return:.2f}%"

    assert metrics.geometric_mean_return > 0, "Geometric progression should be positive"

    print(f"✅ Geometric Mean Return: {metrics.geometric_mean_return:.2f}% per trade")
    print(f"✅ Compound Growth Rate: {metrics.compound_growth_rate:.2f}% per trade")


def test_confidence_interval():
    """Тест: 90% доверительный интервал"""
    print("\n=== Test 4: Confidence Interval ===")

    tracker = PerformanceTracker(enable_history=False)

    # 60% win rate (30 wins, 20 losses)
    for i in range(30):
        tracker.add_trade(create_mock_trade(+4.0))
    for i in range(20):
        tracker.add_trade(create_mock_trade(-3.0))

    metrics = tracker.get_metrics('short')

    # CI должен быть вокруг 60%
    assert 0 < metrics.ci_lower < metrics.win_rate, \
        f"CI lower ({metrics.ci_lower:.2f}) should be below win rate ({metrics.win_rate:.2f})"
    assert metrics.win_rate < metrics.ci_upper < 1, \
        f"CI upper ({metrics.ci_upper:.2f}) should be above win rate ({metrics.win_rate:.2f})"

    # При 60% WR, нижний CI должен быть > 0 (хорошая стратегия)
    assert metrics.ci_lower > 0, f"Lower CI should be > 0, got {metrics.ci_lower:.2f}"

    print(f"✅ Win Rate: {metrics.win_rate*100:.1f}%")
    print(f"✅ 90% CI: [{metrics.ci_lower*100:.1f}%, {metrics.ci_upper*100:.1f}%]")
    print(f"✅ CI lower > 0: {metrics.ci_lower > 0}")


def test_profit_factor():
    """Тест: Profit Factor"""
    print("\n=== Test 5: Profit Factor ===")

    tracker = PerformanceTracker(enable_history=False)

    # 30 wins × +4% = +120%
    # 20 losses × -3% = -60%
    # PF = 120 / 60 = 2.0
    for i in range(30):
        tracker.add_trade(create_mock_trade(+4.0))
    for i in range(20):
        tracker.add_trade(create_mock_trade(-3.0))

    metrics = tracker.get_metrics('short')

    expected_pf = (30 * 4.0) / (20 * 3.0)  # 2.0
    assert abs(metrics.profit_factor - expected_pf) < 0.1, \
        f"Expected PF ~{expected_pf:.2f}, got {metrics.profit_factor:.2f}"

    print(f"✅ Profit Factor: {metrics.profit_factor:.2f}")
    print(f"✅ Expectancy: {metrics.expectancy:.2f}% per trade")


def test_conditions_check():
    """Тест: проверка всех условий"""
    print("\n=== Test 6: Conditions Check ===")

    tracker = PerformanceTracker(enable_history=False)

    # Хорошая стратегия: 60% WR, +4%/-3% R:R
    for i in range(30):
        tracker.add_trade(create_mock_trade(+4.0))
    for i in range(20):
        tracker.add_trade(create_mock_trade(-3.0))

    metrics = tracker.get_metrics('short')

    # Все условия должны быть выполнены
    assert metrics.conditions['win_rate_above_required'], "WR should be above required"
    assert metrics.conditions['positive_geometric_growth'], "Geometric growth should be positive"
    assert metrics.conditions['ci_above_zero'], "CI lower should be above 0"
    assert metrics.all_conditions_met, "All conditions should be met"

    print(f"✅ Win Rate > Required: {metrics.conditions['win_rate_above_required']}")
    print(f"✅ Positive Geo Growth: {metrics.conditions['positive_geometric_growth']}")
    print(f"✅ CI > 0: {metrics.conditions['ci_above_zero']}")
    print(f"✅ ALL CONDITIONS MET: {metrics.all_conditions_met}")


def test_notifications():
    """Тест: умные уведомления"""
    print("\n=== Test 7: Smart Notifications ===")

    class MockTelegram:
        def __init__(self):
            self.messages = []

        def send_message(self, message, parse_mode=None):
            self.messages.append(message)

    tracker = PerformanceTracker(enable_history=False)
    telegram = MockTelegram()
    notifier = PerformanceNotifier(telegram_notifier=telegram)

    # Добавляем 50 сделок (milestone)
    for i in range(50):
        tracker.add_trade(create_mock_trade(+3.0 if i < 30 else -2.0))

    metrics = tracker.get_metrics('short')

    # Проверяем milestone notification
    should_notify, reason = notifier.should_notify(metrics)
    assert should_notify, "Should notify on milestone 50"
    assert 'milestone' in reason, f"Reason should contain 'milestone', got '{reason}'"

    # Отправляем уведомление
    notifier.notify(metrics, 'short', reason)
    assert len(telegram.messages) > 0, "Should send at least one message"

    print(f"✅ Should notify on milestone: {should_notify}")
    print(f"✅ Reason: {reason}")
    print(f"✅ Messages sent: {len(telegram.messages)}")


def test_multiple_windows():
    """Тест: множественные окна анализа"""
    print("\n=== Test 8: Multiple Windows ===")

    tracker = PerformanceTracker(
        windows={'short': 50, 'medium': 100, 'all': None},
        enable_history=False
    )

    # Добавляем 100 сделок
    for i in range(60):
        tracker.add_trade(create_mock_trade(+3.0))
    for i in range(40):
        tracker.add_trade(create_mock_trade(-2.0))

    # Получаем метрики для всех окон
    all_metrics = tracker.get_all_metrics()

    assert 'short' in all_metrics, "Should have 'short' window"
    assert 'medium' in all_metrics, "Should have 'medium' window"
    assert 'all' in all_metrics, "Should have 'all' window"

    print(f"✅ Windows available: {list(all_metrics.keys())}")
    print(f"✅ Short window trades: {all_metrics['short'].total_trades}")
    print(f"✅ Medium window trades: {all_metrics['medium'].total_trades}")
    print(f"✅ All window trades: {all_metrics['all'].total_trades}")


def run_all_tests():
    """Запуск всех тестов"""
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ PERFORMANCE ANALYTICS SYSTEM")
    print("="*60)

    try:
        test_basic_metrics()
        test_required_win_rate()
        test_geometric_progression()
        test_confidence_interval()
        test_profit_factor()
        test_conditions_check()
        test_notifications()
        test_multiple_windows()

        print("\n" + "="*60)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("="*60)
        return True

    except AssertionError as e:
        print(f"\n❌ ТЕСТ ПРОВАЛЕН: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
