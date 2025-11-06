"""
Strategy module for Binance Futures Bot

Модуль стратегии для выбора торговых возможностей:
- CoinSelector - отбор топ-10 монет по Expected Value
- Sector diversification - диверсификация по секторам
- Blacklist - фильтрация проблемных монет
- Filters - фильтры ликвидности и рисков

Автор: Claude Code
Дата: 2025-11-06
"""

from .coin_selector import CoinSelector, create_mock_predictions_func
from .sector_config import (
    get_coin_sector,
    get_sector_coins,
    get_all_sectors,
    get_sector_limit,
    count_coins_per_sector,
    SECTOR_MAP,
    MAX_COINS_PER_SECTOR
)
from .blacklist import (
    is_blacklisted,
    is_high_risk,
    get_risk_level,
    get_max_allocation,
    get_blacklist_reason,
    filter_blacklisted,
    get_blacklist_summary,
    BLACKLIST_SYMBOLS,
    HIGH_RISK_CATEGORIES
)

__all__ = [
    # CoinSelector
    'CoinSelector',
    'create_mock_predictions_func',

    # Sector config
    'get_coin_sector',
    'get_sector_coins',
    'get_all_sectors',
    'get_sector_limit',
    'count_coins_per_sector',
    'SECTOR_MAP',
    'MAX_COINS_PER_SECTOR',

    # Blacklist
    'is_blacklisted',
    'is_high_risk',
    'get_risk_level',
    'get_max_allocation',
    'get_blacklist_reason',
    'filter_blacklisted',
    'get_blacklist_summary',
    'BLACKLIST_SYMBOLS',
    'HIGH_RISK_CATEGORIES'
]

__version__ = '1.0.0'
__author__ = 'Claude Code'
__date__ = '2025-11-06'
