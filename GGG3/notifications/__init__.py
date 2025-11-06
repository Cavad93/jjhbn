"""
Notifications Package

Отправка уведомлений о работе бота через различные каналы:
- Telegram
"""

from .telegram_notifier import TelegramNotifier

__all__ = ['TelegramNotifier']
