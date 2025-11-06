"""
Binance API Module

Модуль для работы с Binance Futures API.

Основные компоненты:
- BinanceClient: Wrapper для Binance Futures API
- Обработка ошибок и retry логика
- Rate limiting
- Логирование всех операций
"""

from .client import BinanceClient
from .exceptions import (
    BinanceAPIError,
    RateLimitError,
    InsufficientBalanceError,
    OrderError
)

__all__ = [
    'BinanceClient',
    'BinanceAPIError',
    'RateLimitError',
    'InsufficientBalanceError',
    'OrderError'
]

__version__ = '1.0.0'
