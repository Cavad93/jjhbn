"""
Binance Futures API Client

Wrapper для работы с Binance Futures API с поддержкой:
- Leverage 1x
- LONG и SHORT позиций
- Обработки ошибок и retry логики
- Rate limiting
- Логирования всех операций
"""

import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from .exceptions import (
    BinanceAPIError,
    RateLimitError,
    InsufficientBalanceError,
    OrderError,
    NetworkError,
    InvalidSymbolError
)


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter для Binance API

    Binance имеет лимиты:
    - 1200 запросов в минуту (weight-based)
    - 10 ордеров в секунду
    """

    def __init__(self, max_requests_per_minute: int = 1000):
        """
        Args:
            max_requests_per_minute: Максимум запросов в минуту (оставляем запас)
        """
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.window = 60  # 60 секунд

    def wait_if_needed(self):
        """Ждет, если достигнут лимит запросов"""
        now = time.time()

        # Удаляем старые запросы (> 60 секунд)
        self.requests = [req_time for req_time in self.requests if now - req_time < self.window]

        # Если достигнут лимит - ждем
        if len(self.requests) >= self.max_requests:
            sleep_time = self.window - (now - self.requests[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self.requests = []

        # Добавляем текущий запрос
        self.requests.append(now)


class BinanceClient:
    """
    Wrapper для Binance Futures API

    ВАЖНО:
    - Futures (не Spot) для возможности SHORT
    - Leverage: 1x (без плеча, но маржинальная торговля)
    - Testnet для тестирования, Live для боевого режима
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        testnet: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Инициализация клиента

        Args:
            api_key: API ключ Binance
            secret_key: Secret ключ Binance
            testnet: True для testnet, False для live
            max_retries: Максимум попыток при ошибке
            retry_delay: Задержка между попытками (секунды)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Инициализация клиента
        try:
            if testnet:
                # Testnet для Futures
                self.client = Client(api_key, secret_key, testnet=True)
                self.client.API_URL = 'https://testnet.binancefuture.com'
                logger.info("Initialized Binance Futures TESTNET client")
            else:
                # Live Futures
                self.client = Client(api_key, secret_key)
                logger.info("Initialized Binance Futures LIVE client")
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise BinanceAPIError(f"Client initialization failed: {e}")

        # Rate limiter
        self.rate_limiter = RateLimiter(max_requests_per_minute=1000)

        # Устанавливаем leverage 1x для всех пар
        try:
            self.set_leverage_all(1)
        except Exception as e:
            logger.warning(f"Failed to set leverage: {e}")

    def _retry_request(self, func, *args, **kwargs):
        """
        Выполняет запрос с retry логикой

        Args:
            func: Функция для выполнения
            *args, **kwargs: Аргументы функции

        Returns:
            Результат функции

        Raises:
            BinanceAPIError: После всех попыток
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                self.rate_limiter.wait_if_needed()

                # Выполняем запрос
                result = func(*args, **kwargs)

                # Логируем успешный запрос
                func_name = getattr(func, '__name__', 'unknown')
                logger.debug(f"Request {func_name} succeeded on attempt {attempt + 1}")

                return result

            except BinanceAPIException as e:
                last_error = e
                error_code = e.code
                error_msg = e.message

                logger.warning(f"Binance API error (attempt {attempt + 1}/{self.max_retries}): {error_code} - {error_msg}")

                # Обработка специфичных ошибок
                if error_code == -1003:  # TOO_MANY_REQUESTS
                    raise RateLimitError(f"Rate limit exceeded: {error_msg}")
                elif error_code == -2010:  # INSUFFICIENT_BALANCE
                    raise InsufficientBalanceError(f"Insufficient balance: {error_msg}")
                elif error_code in [-1021, -1022]:  # Timestamp errors
                    logger.warning("Timestamp error, syncing time...")
                    time.sleep(1)
                    continue
                elif error_code in [-1000, -1001]:  # Internal error
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    raise BinanceAPIError(f"Binance internal error: {error_msg}")
                else:
                    # Для других ошибок пробуем еще раз
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    raise BinanceAPIError(f"Binance API error: {error_msg}")

            except BinanceRequestException as e:
                last_error = e
                logger.warning(f"Network error (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise NetworkError(f"Network error after {self.max_retries} attempts: {e}")

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise BinanceAPIError(f"Unexpected error: {e}")

        # Если все попытки провалились
        raise BinanceAPIError(f"Request failed after {self.max_retries} attempts: {last_error}")

    def set_leverage_all(self, leverage: int = 1):
        """
        Устанавливает кредитное плечо для всех торгуемых пар

        Args:
            leverage: Размер плеча (1 для консервативной торговли)
        """
        logger.info(f"Setting leverage to {leverage}x for all symbols")

        # Для conservative торговли используем 1x
        # Метод можно расширить для установки leverage для каждой пары отдельно
        # Но для начала просто логируем
        logger.info(f"Leverage will be set to {leverage}x when opening positions")

    def set_leverage_for_symbol(self, symbol: str, leverage: int = 1):
        """
        Устанавливает leverage для конкретного символа

        Args:
            symbol: Торговая пара (например, 'BTCUSDT')
            leverage: Размер плеча
        """
        try:
            result = self._retry_request(
                self.client.futures_change_leverage,
                symbol=symbol,
                leverage=leverage
            )
            logger.info(f"Set leverage {leverage}x for {symbol}: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")
            raise

    def get_all_usdt_pairs(self, min_volume_24h: float = 1_000_000) -> List[str]:
        """
        Получает все торговые пары USDT с Futures

        Фильтры:
        - Только USDT пары (BTCUSDT, ETHUSDT, ...)
        - Только Futures
        - Минимальный объем 24h > $1M
        - Исключить стейблкоины (USDCUSDT, BUSDUSDT, ...)

        Args:
            min_volume_24h: Минимальный объем за 24ч в USD

        Returns:
            Список символов (~300-400 пар)
        """
        logger.info(f"Fetching all USDT pairs with min volume {min_volume_24h:,.0f}")

        try:
            # Получаем информацию о всех парах
            exchange_info = self._retry_request(self.client.futures_exchange_info)

            # Список стейблкоинов для исключения
            stablecoins = ['USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'DAIUSDT', 'USDPUSDT', 'FDUSDUSDT']

            pairs = []

            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info['symbol']
                status = symbol_info['status']

                # Фильтр 1: Только активные пары
                if status != 'TRADING':
                    continue

                # Фильтр 2: Только USDT пары
                if not symbol.endswith('USDT'):
                    continue

                # Фильтр 3: Исключить стейблкоины
                if symbol in stablecoins:
                    continue

                # Фильтр 4: Проверить объем за 24ч
                try:
                    ticker = self._retry_request(
                        self.client.futures_ticker,
                        symbol=symbol
                    )
                    volume_24h = float(ticker['quoteVolume'])

                    if volume_24h >= min_volume_24h:
                        pairs.append(symbol)
                        logger.debug(f"Added {symbol} with 24h volume: ${volume_24h:,.0f}")

                except Exception as e:
                    logger.debug(f"Skipping {symbol}: {e}")
                    continue

            logger.info(f"Found {len(pairs)} USDT pairs meeting criteria")
            return pairs

        except Exception as e:
            logger.error(f"Failed to get USDT pairs: {e}")
            raise BinanceAPIError(f"Failed to get pairs: {e}")

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Получает OHLCV данные

        Args:
            symbol: Торговая пара (например, 'BTCUSDT')
            timeframe: Таймфрейм ('5m', '15m', '30m', '4h', '1d')
            limit: Количество свечей (max 1500)
            start_time: Начальное время (timestamp ms)
            end_time: Конечное время (timestamp ms)

        Returns:
            DataFrame с колонками: timestamp, open, high, low, close, volume
        """
        logger.debug(f"Fetching OHLCV for {symbol} {timeframe} (limit={limit})")

        try:
            # Получаем klines
            klines = self._retry_request(
                self.client.futures_klines,
                symbol=symbol,
                interval=timeframe,
                limit=limit,
                startTime=start_time,
                endTime=end_time
            )

            # Преобразуем в DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Конвертируем типы
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[
                ['open', 'high', 'low', 'close', 'volume']
            ].astype(float)

            # Возвращаем только нужные колонки
            result = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            logger.debug(f"Retrieved {len(result)} candles for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Failed to get OHLCV for {symbol}: {e}")
            raise BinanceAPIError(f"Failed to get OHLCV: {e}")

    def get_current_price(self, symbol: str) -> float:
        """
        Получает текущую цену символа

        Args:
            symbol: Торговая пара

        Returns:
            Текущая цена
        """
        try:
            ticker = self._retry_request(
                self.client.futures_symbol_ticker,
                symbol=symbol
            )
            price = float(ticker['price'])
            logger.debug(f"Current price for {symbol}: {price}")
            return price

        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            raise BinanceAPIError(f"Failed to get price: {e}")

    def get_orderbook(self, symbol: str, limit: int = 20) -> dict:
        """
        Получает стакан ордеров

        Args:
            symbol: Торговая пара
            limit: Глубина стакана (5, 10, 20, 50, 100, 500, 1000)

        Returns:
            Dict с bids и asks
        """
        try:
            book = self._retry_request(
                self.client.futures_order_book,
                symbol=symbol,
                limit=limit
            )

            result = {
                'bids': [[float(p), float(q)] for p, q in book['bids']],
                'asks': [[float(p), float(q)] for p, q in book['asks']]
            }

            logger.debug(f"Order book for {symbol}: {len(result['bids'])} bids, {len(result['asks'])} asks")
            return result

        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            raise BinanceAPIError(f"Failed to get order book: {e}")

    def create_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False
    ) -> dict:
        """
        Создает market ордер

        Args:
            symbol: Торговая пара
            side: 'BUY' или 'SELL'
            quantity: Количество
            reduce_only: Только закрытие позиции

        Returns:
            Информация об ордере
        """
        logger.info(f"Creating market order: {side} {quantity} {symbol}")

        try:
            order = self._retry_request(
                self.client.futures_create_order,
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity,
                reduceOnly='true' if reduce_only else 'false'
            )

            logger.info(f"Market order created: {order['orderId']} - {order['status']}")
            return order

        except Exception as e:
            logger.error(f"Failed to create market order for {symbol}: {e}")
            raise OrderError(f"Failed to create market order: {e}")

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        reduce_only: bool = False,
        time_in_force: str = 'GTC'
    ) -> dict:
        """
        Создает limit ордер (для TP)

        Args:
            symbol: Торговая пара
            side: 'BUY' или 'SELL'
            quantity: Количество
            price: Цена
            reduce_only: Только закрытие позиции
            time_in_force: 'GTC', 'IOC', 'FOK'

        Returns:
            Информация об ордере
        """
        logger.info(f"Creating limit order: {side} {quantity} {symbol} @ {price}")

        try:
            order = self._retry_request(
                self.client.futures_create_order,
                symbol=symbol,
                side=side,
                type='LIMIT',
                quantity=quantity,
                price=price,
                timeInForce=time_in_force,
                reduceOnly='true' if reduce_only else 'false'
            )

            logger.info(f"Limit order created: {order['orderId']} - {order['status']}")
            return order

        except Exception as e:
            logger.error(f"Failed to create limit order for {symbol}: {e}")
            raise OrderError(f"Failed to create limit order: {e}")

    def create_stop_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = False
    ) -> dict:
        """
        Создает stop-market ордер (для SL)

        Args:
            symbol: Торговая пара
            side: 'BUY' или 'SELL'
            quantity: Количество
            stop_price: Цена срабатывания стопа
            reduce_only: Только закрытие позиции

        Returns:
            Информация об ордере
        """
        logger.info(f"Creating stop-market order: {side} {quantity} {symbol} @ stop {stop_price}")

        try:
            order = self._retry_request(
                self.client.futures_create_order,
                symbol=symbol,
                side=side,
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=stop_price,
                reduceOnly='true' if reduce_only else 'false'
            )

            logger.info(f"Stop-market order created: {order['orderId']} - {order['status']}")
            return order

        except Exception as e:
            logger.error(f"Failed to create stop-market order for {symbol}: {e}")
            raise OrderError(f"Failed to create stop-market order: {e}")

    def cancel_order(self, symbol: str, order_id: int) -> bool:
        """
        Отменяет ордер

        Args:
            symbol: Торговая пара
            order_id: ID ордера

        Returns:
            True если успешно отменен
        """
        logger.info(f"Canceling order {order_id} for {symbol}")

        try:
            result = self._retry_request(
                self.client.futures_cancel_order,
                symbol=symbol,
                orderId=order_id
            )

            logger.info(f"Order {order_id} canceled: {result['status']}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id} for {symbol}: {e}")
            return False

    def cancel_all_orders(self, symbol: str) -> bool:
        """
        Отменяет все открытые ордера для символа

        Args:
            symbol: Торговая пара

        Returns:
            True если успешно
        """
        logger.info(f"Canceling all orders for {symbol}")

        try:
            result = self._retry_request(
                self.client.futures_cancel_all_open_orders,
                symbol=symbol
            )

            logger.info(f"All orders for {symbol} canceled")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel all orders for {symbol}: {e}")
            return False

    def get_balance(self, asset: str = 'USDT') -> float:
        """
        Получает доступный баланс

        Args:
            asset: Актив ('USDT' по умолчанию)

        Returns:
            Доступный баланс
        """
        try:
            account = self._retry_request(self.client.futures_account)

            for balance in account['assets']:
                if balance['asset'] == asset:
                    available = float(balance['availableBalance'])
                    logger.debug(f"Available {asset} balance: {available}")
                    return available

            logger.warning(f"Asset {asset} not found in account")
            return 0.0

        except Exception as e:
            logger.error(f"Failed to get balance for {asset}: {e}")
            raise BinanceAPIError(f"Failed to get balance: {e}")

    def get_account_info(self) -> dict:
        """
        Получает полную информацию об аккаунте

        Returns:
            Dict с информацией об аккаунте
        """
        try:
            account = self._retry_request(self.client.futures_account)
            logger.debug("Retrieved account info")
            return account

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise BinanceAPIError(f"Failed to get account info: {e}")

    def get_position_info(self, symbol: Optional[str] = None) -> List[dict]:
        """
        Получает информацию о позициях

        Args:
            symbol: Конкретная пара (если None - все позиции)

        Returns:
            Список позиций
        """
        try:
            positions = self._retry_request(
                self.client.futures_position_information,
                symbol=symbol
            )

            # Фильтруем только активные позиции
            active_positions = [
                pos for pos in positions
                if float(pos['positionAmt']) != 0
            ]

            logger.debug(f"Found {len(active_positions)} active positions")
            return active_positions

        except Exception as e:
            logger.error(f"Failed to get position info: {e}")
            raise BinanceAPIError(f"Failed to get position info: {e}")

    def get_open_orders(self, symbol: Optional[str] = None) -> List[dict]:
        """
        Получает открытые ордера

        Args:
            symbol: Конкретная пара (если None - все ордера)

        Returns:
            Список открытых ордеров
        """
        try:
            orders = self._retry_request(
                self.client.futures_get_open_orders,
                symbol=symbol
            )

            logger.debug(f"Found {len(orders)} open orders")
            return orders

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise BinanceAPIError(f"Failed to get open orders: {e}")

    def get_ticker(self, symbol: str) -> dict:
        """
        Получает 24h ticker данные для символа

        Args:
            symbol: Торговая пара

        Returns:
            dict: Ticker данные (price, volume, bid, ask, etc.)
        """
        try:
            ticker = self._retry_request(
                self.client.futures_ticker,
                symbol=symbol
            )

            return {
                'symbol': ticker['symbol'],
                'priceChange': float(ticker['priceChange']),
                'priceChangePercent': float(ticker['priceChangePercent']),
                'lastPrice': float(ticker['lastPrice']),
                'bidPrice': float(ticker.get('bidPrice', ticker['lastPrice'])),
                'askPrice': float(ticker.get('askPrice', ticker['lastPrice'])),
                'volume': float(ticker['volume']),
                'quoteVolume': float(ticker['quoteVolume']),
                'openTime': int(ticker['openTime']),
                'closeTime': int(ticker['closeTime']),
            }

        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise BinanceAPIError(f"Failed to get ticker: {e}")
