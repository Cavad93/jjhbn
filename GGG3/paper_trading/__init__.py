"""
Paper Trading Module

Модуль для симуляции торговли с реальными ценами Binance
"""

from .paper_exchange import (
    PaperExchange,
    InsufficientBalance,
    OrderNotFound,
    InvalidOrder,
    get_binance_price,
    get_binance_kline,
    generate_order_id
)

__all__ = [
    'PaperExchange',
    'InsufficientBalance',
    'OrderNotFound',
    'InvalidOrder',
    'get_binance_price',
    'get_binance_kline',
    'generate_order_id'
]

__version__ = '1.0.0'
