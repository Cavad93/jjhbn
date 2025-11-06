#!/usr/bin/env python3
"""
Тест Telegram уведомлений

Использование:
    export TELEGRAM_BOT_TOKEN="your_bot_token"
    export TELEGRAM_CHAT_ID="your_chat_id"
    python3 test_telegram.py
"""

import sys
import os
from pathlib import Path

# Добавляем путь к GGG3
sys.path.insert(0, str(Path(__file__).parent.parent))

from notifications.telegram_notifier import TelegramNotifier
import binance_config as config


def test_telegram_notifications():
    """Тестирует все типы Telegram уведомлений"""

    print("="*80)
    print("ТЕСТ TELEGRAM УВЕДОМЛЕНИЙ")
    print("="*80)

    # Проверяем настройки
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print("\n⚠️  ВНИМАНИЕ: Telegram не настроен!")
        print("\nНастройте переменные окружения:")
        print("  export TELEGRAM_BOT_TOKEN='your_bot_token'")
        print("  export TELEGRAM_CHAT_ID='your_chat_id'")
        print("\nПолучить токен: https://t.me/BotFather")
        print("Получить chat_id: https://t.me/userinfobot")
        return

    print(f"\n✓ Bot token: {'*' * 10}{config.TELEGRAM_BOT_TOKEN[-5:]}")
    print(f"✓ Chat ID: {config.TELEGRAM_CHAT_ID}")

    # Создаём notifier
    telegram = TelegramNotifier(
        bot_token=config.TELEGRAM_BOT_TOKEN,
        chat_id=config.TELEGRAM_CHAT_ID,
        enabled=True
    )

    print("\n" + "="*80)
    print("1. ТЕСТОВОЕ СООБЩЕНИЕ")
    print("="*80)
    success = telegram.test_connection()
    print(f"{'✓' if success else '✗'} Test message sent")

    print("\n" + "="*80)
    print("2. СТАРТ БОТА")
    print("="*80)
    telegram.notify_bot_started(
        paper_mode=True,
        initial_capital=1000.0,
        max_positions=10
    )
    print("✓ Bot started notification sent")

    print("\n" + "="*80)
    print("3. ОТКРЫТИЕ ПОЗИЦИИ")
    print("="*80)
    position_data = {
        'symbol': 'BTCUSDT',
        'direction': 'LONG',
        'entry_price': 35000.0,
        'position_size': 100.0,
        'leverage': 1,
        'tp_price': 36000.0,
        'sl_price': 34500.0,
        'p_up': 0.65,
        'ev': 0.08,
        'predictions': {
            'xgb': 0.67,
            'rf': 0.63,
            'nn': 0.66,
            'base': 0.64
        }
    }
    telegram.notify_position_opened(position_data)
    print("✓ Position opened notification sent")

    print("\n" + "="*80)
    print("4. TRAILING STOP")
    print("="*80)
    telegram.notify_trailing_stop_activated(
        symbol='BTCUSDT',
        direction='LONG',
        current_price=35700.0,
        new_sl=35000.0,
        profit_pct=2.0
    )
    print("✓ Trailing stop notification sent")

    print("\n" + "="*80)
    print("5. ЗАКРЫТИЕ ПОЗИЦИИ")
    print("="*80)
    close_data = {
        'symbol': 'BTCUSDT',
        'direction': 'LONG',
        'entry_price': 35000.0,
        'exit_price': 35800.0,
        'position_size': 100.0,
        'pnl': 8.0,
        'pnl_percent': 2.29,
        'exit_reason': 'TP',
        'holding_time': '12.5h'
    }
    telegram.notify_position_closed(close_data)
    print("✓ Position closed notification sent")

    print("\n" + "="*80)
    print("6. РЕБАЛАНСИРОВКА")
    print("="*80)
    rebalance_stats = {
        'closed_positions': 2,
        'opened_positions': 3,
        'total_positions': 8,
        'top_opportunities': [
            {'symbol': 'BTCUSDT', 'ev': 0.08, 'p_up': 0.65},
            {'symbol': 'ETHUSDT', 'ev': 0.07, 'p_up': 0.63},
            {'symbol': 'BNBUSDT', 'ev': 0.06, 'p_up': 0.62}
        ]
    }
    telegram.notify_rebalance_stats(rebalance_stats)
    print("✓ Rebalance notification sent")

    print("\n" + "="*80)
    print("7. NIGHT MODE")
    print("="*80)
    telegram.notify_night_mode_activated(max_positions=5)
    print("✓ Night mode activated notification sent")

    print("\n" + "="*80)
    print("8. СТАТИСТИКА")
    print("="*80)
    daily_stats = {
        'total_trades': 25,
        'win_rate': 0.60,
        'total_pnl': 150.0,
        'avg_win': 10.0,
        'avg_loss': -5.0,
        'best_trade': 25.0,
        'worst_trade': -8.0,
        'current_balance': 1150.0,
        'open_positions': 8
    }
    telegram.notify_daily_stats(daily_stats)
    print("✓ Daily stats notification sent")

    print("\n" + "="*80)
    print("9. ОСТАНОВКА БОТА")
    print("="*80)
    telegram.notify_bot_stopped(reason="Test completed")
    print("✓ Bot stopped notification sent")

    print("\n" + "="*80)
    print("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("="*80)
    print("\nПроверьте Telegram чат - должно быть 9 сообщений")


if __name__ == '__main__':
    test_telegram_notifications()
