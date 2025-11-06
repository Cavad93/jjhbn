"""
Risk Management модули для Binance торгового бота

Включает:
- TrailingStopManager: Trailing stop loss для защиты прибыли
- BlackSwanProtection: Защита от резких обвалов
- NightModeManager: Управление ночным режимом торговли
"""

from .trailing_stop import TrailingStopManager
from .black_swan import BlackSwanProtection
from .night_mode import NightModeManager

__all__ = [
    'TrailingStopManager',
    'BlackSwanProtection',
    'NightModeManager'
]
