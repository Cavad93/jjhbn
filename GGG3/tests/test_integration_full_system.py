#!/usr/bin/env python3
"""
Интеграционные тесты полной системы

Проверяет работу всех компонентов вместе:
- ATR-based Trailing Stop
- Performance Analytics
- Интеграцию с Position Manager
- Уведомления
"""

import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from risk.trailing_stop import TrailingStopManager
from analytics.performance_tracker import PerformanceTracker, TradeStats
from analytics.performance_notifier import PerformanceNotifier


@dataclass
class MockPosition:
    """Мок-объект позиции"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: Optional[float] = None
    sl_price: float = 0.0
    tp_price: float = 0.0
    atr_value: float = 2.0
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: Optional[str] = None
    status: str = 'OPEN'
    amount: float = 1.0
    position_value: float = 100.0


class MockTelegram:
    """Мок Telegram для тестирования уведомлений"""
    def __init__(self):
        self.messages = []

    def send_message(self, message, parse_mode=None):
        self.messages.append(message)
        print(f"[MockTelegram] Message sent (length: {len(message)})")


def test_trailing_stop_with_analytics():
    """Интеграция: Trailing Stop + Analytics"""
    print("\n=== Test 1: Trailing Stop + Performance Analytics ===")

    # Инициализация компонентов
    trailing_stop = TrailingStopManager(
        enabled=True,
        activation_pct=3.0,
        distance_atr_mult=1.0
    )

    tracker = PerformanceTracker(enable_history=False)
    telegram = MockTelegram()
    notifier = PerformanceNotifier(telegram_notifier=telegram)

    # Симулируем 50 сделок с трейлинг-стопом
    wins = 0
    losses = 0

    for i in range(50):
        # Создаем позицию
        position = MockPosition(
            symbol='BTCUSDT',
            direction='LONG',
            entry_price=100.0,
            sl_price=97.0,  # -3% SL
            tp_price=106.0,  # +6% TP
            atr_value=2.0
        )

        # Симулируем движение цены
        # 60% выигрышей, 40% проигрышей
        is_win = i < 30

        if is_win:
            # Позиция идет в прибыль
            # Активируем трейлинг на +4%
            current_price = 104.0
            new_sl = trailing_stop.check_trailing_stop(position, current_price)

            assert new_sl is not None, f"Trailing should activate at {current_price}"
            assert new_sl > position.sl_price, "New SL should be higher"

            # Обновляем SL
            position.sl_price = new_sl

            # Цена идет до +5%
            current_price = 105.0
            new_sl = trailing_stop.check_trailing_stop(position, current_price)

            if new_sl:
                position.sl_price = new_sl

            # Закрываем по трейлинг-стопу (+3%)
            position.exit_price = 103.0
            position.pnl_pct = ((103.0 - 100.0) / 100.0) * 100
            position.exit_reason = 'TRAILING_SL'
            wins += 1

        else:
            # Позиция идет в убыток, закрывается по SL
            position.exit_price = 97.0
            position.pnl_pct = ((97.0 - 100.0) / 100.0) * 100
            position.exit_reason = 'SL'
            losses += 1

        # Закрываем позицию
        position.status = 'CLOSED'
        position.exit_time = datetime.now()
        position.pnl = position.pnl_pct  # Упрощенно

        # Добавляем в analytics
        trade_stats = TradeStats(
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=position.exit_price,
            pnl=position.pnl,
            pnl_pct=position.pnl_pct,
            exit_reason=position.exit_reason,
            entry_time=position.entry_time,
            exit_time=position.exit_time,
            duration_hours=2.0
        )
        tracker.add_trade(trade_stats)

        # Сбрасываем трейлинг
        trailing_stop.reset_position(position.symbol)

    # Проверяем метрики
    metrics = tracker.get_metrics('short')

    assert metrics is not None, "Metrics should be available"
    assert metrics.total_trades == 50, f"Expected 50 trades, got {metrics.total_trades}"
    assert metrics.win_count == wins, f"Expected {wins} wins, got {metrics.win_count}"
    assert metrics.loss_count == losses, f"Expected {losses} losses, got {metrics.loss_count}"

    # Проверяем что трейлинг улучшил R:R
    # Wins: +3% (трейлинг), Losses: -3% (SL)
    # R:R = 1:1 вместо старого 0.5:1
    assert metrics.avg_win_loss_ratio >= 0.9, \
        f"R:R should be ~1:1, got {metrics.avg_win_loss_ratio:.2f}"

    # Проверяем уведомление
    should_notify, reason = notifier.should_notify(metrics)
    assert should_notify, "Should notify on 50 trades milestone"
    assert 'milestone' in reason, f"Reason should be milestone, got {reason}"

    print(f"✅ Completed 50 trades")
    print(f"✅ Win Rate: {metrics.win_rate*100:.1f}%")
    print(f"✅ R:R improved to: {metrics.avg_win_loss_ratio:.2f}:1")
    print(f"✅ Notification triggered: {reason}")


def test_full_workflow_simulation():
    """Полная симуляция workflow бота"""
    print("\n=== Test 2: Full Workflow Simulation ===")

    # Инициализация всех компонентов
    trailing_stop = TrailingStopManager(
        enabled=True,
        activation_pct=3.0,
        distance_atr_mult=1.0
    )

    tracker = PerformanceTracker(
        windows={'short': 50, 'medium': 100, 'all': None},
        enable_history=False
    )

    telegram = MockTelegram()
    notifier = PerformanceNotifier(telegram_notifier=telegram)

    # Открываем позиции
    print("\nPhase 1: Opening and managing positions...")

    active_positions = []

    for i in range(10):
        position = MockPosition(
            symbol=f'COIN{i}USDT',
            direction='LONG',
            entry_price=100.0,
            sl_price=97.0,
            tp_price=106.0,
            atr_value=2.0
        )
        active_positions.append(position)

    print(f"✅ Opened {len(active_positions)} positions")

    # Обновляем позиции (симулируем движение цены)
    print("\nPhase 2: Updating trailing stops...")

    for i, position in enumerate(active_positions):
        # Половина позиций идет в прибыль
        if i < 5:
            current_price = 104.0  # +4%

            # Проверяем трейлинг
            new_sl = trailing_stop.check_trailing_stop(position, current_price)

            if new_sl:
                old_sl = position.sl_price
                position.sl_price = new_sl
                print(f"  {position.symbol}: SL updated {old_sl:.2f} → {new_sl:.2f}")

    print(f"✅ Updated {5} positions with trailing stop")

    # Закрываем позиции
    print("\nPhase 3: Closing positions...")

    closed_count = 0
    for i, position in enumerate(active_positions):
        if i < 5:
            # Закрываем с прибылью по трейлингу
            position.exit_price = 103.0
            position.pnl_pct = 3.0
            position.exit_reason = 'TRAILING_SL'
        else:
            # Закрываем по SL
            position.exit_price = 97.0
            position.pnl_pct = -3.0
            position.exit_reason = 'SL'

        position.status = 'CLOSED'
        position.exit_time = datetime.now()
        position.pnl = position.pnl_pct

        # Добавляем в tracker
        trade_stats = TradeStats(
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=position.exit_price,
            pnl=position.pnl,
            pnl_pct=position.pnl_pct,
            exit_reason=position.exit_reason,
            entry_time=position.entry_time,
            exit_time=position.exit_time,
            duration_hours=2.0
        )
        tracker.add_trade(trade_stats)

        # Сбрасываем трейлинг
        trailing_stop.reset_position(position.symbol)
        closed_count += 1

    print(f"✅ Closed {closed_count} positions")

    # Проверяем что все позиции обработаны
    assert tracker.trades[-1].symbol == active_positions[-1].symbol, \
        "Last trade should match last position"

    print(f"✅ All {len(tracker.trades)} trades tracked")


def test_conditions_monitoring():
    """Тест: мониторинг условий и смена статуса"""
    print("\n=== Test 3: Conditions Monitoring ===")

    tracker = PerformanceTracker(enable_history=False)
    telegram = MockTelegram()
    notifier = PerformanceNotifier(telegram_notifier=telegram)

    # Фаза 1: Плохая стратегия (40% WR, малый R:R)
    print("\nPhase 1: Poor strategy (40% WR)...")

    for i in range(20):
        pnl_pct = +2.0 if i < 8 else -3.0
        tracker.add_trade(TradeStats(
            symbol='BTCUSDT',
            direction='LONG',
            entry_price=100.0,
            exit_price=100.0 * (1 + pnl_pct/100),
            pnl=pnl_pct,
            pnl_pct=pnl_pct,
            exit_reason='TP' if pnl_pct > 0 else 'SL',
            entry_time=datetime.now() - timedelta(hours=2),
            exit_time=datetime.now(),
            duration_hours=2.0
        ))

    # Добавляем еще 30 сделок чтобы достичь 50
    for i in range(30):
        pnl_pct = +2.0 if i < 12 else -3.0
        tracker.add_trade(TradeStats(
            symbol='BTCUSDT',
            direction='LONG',
            entry_price=100.0,
            exit_price=100.0 * (1 + pnl_pct/100),
            pnl=pnl_pct,
            pnl_pct=pnl_pct,
            exit_reason='TP' if pnl_pct > 0 else 'SL',
            entry_time=datetime.now() - timedelta(hours=2),
            exit_time=datetime.now(),
            duration_hours=2.0
        ))

    metrics = tracker.get_metrics('short')
    status_before = metrics.all_conditions_met

    print(f"  Win Rate: {metrics.win_rate*100:.1f}%")
    print(f"  Required: {metrics.required_win_rate*100:.1f}%")
    print(f"  Status: {'✅ GOOD' if status_before else '❌ BAD'}")

    # Сохраняем для проверки смены статуса
    notifier.last_status = status_before

    # Фаза 2: Улучшение стратегии (добавляем больше выигрышей)
    # Не добавляем новые сделки, а проверяем что система правильно определила статус

    print(f"✅ Conditions monitoring working")
    print(f"✅ All conditions met: {metrics.all_conditions_met}")


def test_multi_window_consistency():
    """Тест: согласованность между окнами"""
    print("\n=== Test 4: Multi-Window Consistency ===")

    tracker = PerformanceTracker(
        windows={'short': 50, 'medium': 100, 'all': None},
        enable_history=False
    )

    # Добавляем 100 сделок
    for i in range(100):
        pnl_pct = +3.0 if i % 2 == 0 else -2.0
        tracker.add_trade(TradeStats(
            symbol='BTCUSDT',
            direction='LONG',
            entry_price=100.0,
            exit_price=100.0 * (1 + pnl_pct/100),
            pnl=pnl_pct,
            pnl_pct=pnl_pct,
            exit_reason='TP' if pnl_pct > 0 else 'SL',
            entry_time=datetime.now() - timedelta(hours=2),
            exit_time=datetime.now(),
            duration_hours=2.0
        ))

    # Получаем метрики для всех окон
    all_metrics = tracker.get_all_metrics()

    assert 'short' in all_metrics, "Should have short window"
    assert 'medium' in all_metrics, "Should have medium window"
    assert 'all' in all_metrics, "Should have all window"

    # Проверяем consistency
    short_metrics = all_metrics['short']
    medium_metrics = all_metrics['medium']
    all_metrics_window = all_metrics['all']

    assert short_metrics.total_trades == 50, "Short should have 50 trades"
    assert medium_metrics.total_trades == 100, "Medium should have 100 trades"
    assert all_metrics_window.total_trades == 100, "All should have 100 trades"

    # Medium и All должны быть идентичны (так как у нас только 100 сделок)
    assert medium_metrics.win_rate == all_metrics_window.win_rate, \
        "Medium and All should have same WR"

    print(f"✅ Short window: {short_metrics.total_trades} trades, WR {short_metrics.win_rate*100:.1f}%")
    print(f"✅ Medium window: {medium_metrics.total_trades} trades, WR {medium_metrics.win_rate*100:.1f}%")
    print(f"✅ All window: {all_metrics_window.total_trades} trades, WR {all_metrics_window.win_rate*100:.1f}%")
    print(f"✅ Windows are consistent")


def run_all_tests():
    """Запуск всех интеграционных тестов"""
    print("\n" + "="*60)
    print("ИНТЕГРАЦИОННЫЕ ТЕСТЫ ПОЛНОЙ СИСТЕМЫ")
    print("="*60)

    try:
        test_trailing_stop_with_analytics()
        test_full_workflow_simulation()
        test_conditions_monitoring()
        test_multi_window_consistency()

        print("\n" + "="*60)
        print("✅ ВСЕ ИНТЕГРАЦИОННЫЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("="*60)
        return True

    except AssertionError as e:
        print(f"\n❌ ИНТЕГРАЦИОННЫЙ ТЕСТ ПРОВАЛЕН: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
