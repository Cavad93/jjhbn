"""
Исключения для Binance API модуля
"""


class BinanceAPIError(Exception):
    """Базовое исключение для ошибок Binance API"""
    pass


class RateLimitError(BinanceAPIError):
    """Превышен лимит запросов"""
    pass


class InsufficientBalanceError(BinanceAPIError):
    """Недостаточно средств на балансе"""
    pass


class OrderError(BinanceAPIError):
    """Ошибка при создании/отмене ордера"""
    pass


class NetworkError(BinanceAPIError):
    """Сетевая ошибка при обращении к API"""
    pass


class InvalidSymbolError(BinanceAPIError):
    """Неверный символ торговой пары"""
    pass
