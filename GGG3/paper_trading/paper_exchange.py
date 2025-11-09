"""
Paper Trading Exchange Module

Симулятор биржи для бумажной торговли с реальными ценами Binance
"""

import json
import time
import uuid
from typing import Dict, List, Optional
from datetime import datetime
import requests


# ============================================================================
# EXCEPTIONS
# ============================================================================

class InsufficientBalance(Exception):
    """Недостаточно средств для совершения операции"""
    pass


class OrderNotFound(Exception):
    """Ордер не найден"""
    pass


class InvalidOrder(Exception):
    """Некорректный ордер"""
    pass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_order_id() -> str:
    """Генерирует уникальный ID для ордера"""
    return f"PAPER_{uuid.uuid4().hex[:12].upper()}"


def get_binance_price(symbol: str) -> float:
    """
    Получает текущую цену с Binance API

    Uses MAINNET endpoints (not testnet):
    - Primary: https://api.binance.com
    - Fallback: https://api1.binance.com, https://api-gcp.binance.com

    NO API KEYS REQUIRED - uses public market data endpoints

    Args:
        symbol: Торговая пара (например, 'BTCUSDT')

    Returns:
        float: Текущая цена

    Raises:
        Exception: Ошибка при получении цены
    """
    # Убираем возможные суффиксы и форматируем символ
    symbol = symbol.replace('/', '').upper()

    # MAINNET endpoints (НЕ testnet!)
    base_urls = [
        "https://api.binance.com",           # Primary mainnet
        "https://api1.binance.com",          # Alternative mainnet
        "https://api-gcp.binance.com",       # GCP CDN mainnet
        "https://data-api.binance.vision"    # Historical data mainnet
    ]

    params = {'symbol': symbol}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }

    last_error = None

    # Try each endpoint with fallback
    for base_url in base_urls:
        try:
            url = f"{base_url}/api/v3/ticker/price"
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            price = float(data['price'])

            return price

        except requests.exceptions.RequestException as e:
            last_error = e
            continue  # Try next endpoint
        except (KeyError, ValueError) as e:
            last_error = e
            continue

    # All endpoints failed
    raise Exception(f"Failed to get price for {symbol} from all endpoints: {last_error}")


def get_binance_kline(symbol: str, interval: str = '4h', limit: int = 1) -> Dict:
    """
    Получает данные свечи (OHLC) с Binance API

    Uses MAINNET endpoints (not testnet):
    - Primary: https://api.binance.com
    - Fallback: https://api1.binance.com, https://api-gcp.binance.com

    NO API KEYS REQUIRED - uses public market data endpoints

    Args:
        symbol: Торговая пара (например, 'BTCUSDT')
        interval: Интервал свечи ('1m', '5m', '1h', '4h', '1d' и т.д.)
        limit: Количество свечей (по умолчанию 1 - последняя)

    Returns:
        Dict: Словарь с данными свечи (open, high, low, close, volume)
    """
    symbol = symbol.replace('/', '').upper()

    # MAINNET endpoints (НЕ testnet!)
    base_urls = [
        "https://api.binance.com",           # Primary mainnet
        "https://api1.binance.com",          # Alternative mainnet
        "https://api-gcp.binance.com",       # GCP CDN mainnet
        "https://data-api.binance.vision"    # Historical data mainnet
    ]

    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }

    last_error = None

    # Try each endpoint with fallback
    for base_url in base_urls:
        try:
            url = f"{base_url}/api/v3/klines"
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            if not data:
                raise Exception(f"No kline data for {symbol}")

            # Последняя свеча
            kline = data[-1]

            return {
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
                'timestamp': int(kline[0])
            }

        except requests.exceptions.RequestException as e:
            last_error = e
            continue  # Try next endpoint
        except (KeyError, ValueError, IndexError) as e:
            last_error = e
            continue

    # All endpoints failed
    raise Exception(f"Failed to get kline for {symbol} from all endpoints: {last_error}")


def get_binance_ticker(symbol: str) -> Dict:
    """
    Получает 24h ticker данные с Binance API

    Uses MAINNET endpoints (not testnet):
    - Primary: https://api.binance.com
    - Fallback: https://api1.binance.com, https://api-gcp.binance.com

    NO API KEYS REQUIRED - uses public market data endpoints

    Args:
        symbol: Торговая пара (например, 'BTCUSDT')

    Returns:
        Dict: Ticker данные (price, volume, bid, ask, etc.)
    """
    symbol = symbol.replace('/', '').upper()

    # MAINNET endpoints (НЕ testnet!)
    base_urls = [
        "https://api.binance.com",
        "https://api1.binance.com",
        "https://api-gcp.binance.com",
    ]

    params = {'symbol': symbol}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }

    last_error = None

    # Try each endpoint with fallback
    for base_url in base_urls:
        try:
            url = f"{base_url}/api/v3/ticker/24hr"
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            return {
                'symbol': data['symbol'],
                'priceChange': float(data['priceChange']),
                'priceChangePercent': float(data['priceChangePercent']),
                'lastPrice': float(data['lastPrice']),
                'bidPrice': float(data['bidPrice']),
                'askPrice': float(data['askPrice']),
                'volume': float(data['volume']),
                'quoteVolume': float(data['quoteVolume']),
                'openTime': int(data['openTime']),
                'closeTime': int(data['closeTime']),
            }

        except requests.exceptions.RequestException as e:
            last_error = e
            continue  # Try next endpoint
        except (KeyError, ValueError) as e:
            last_error = e
            continue

    # All endpoints failed
    raise Exception(f"Failed to get ticker for {symbol} from all endpoints: {last_error}")


# ============================================================================
# PAPER EXCHANGE CLASS
# ============================================================================

class PaperExchange:
    """
    Симулятор биржи для бумажной торговли

    Особенности:
    - Использует РЕАЛЬНЫЕ цены с Binance API
    - Симулирует проскальзывание 0.05%
    - Применяет комиссию 0.1%
    - Поддерживает LONG и SHORT позиции
    - Автоматическая проверка и исполнение TP/SL ордеров
    - Сохранение состояния в JSON
    """

    def __init__(self, initial_capital: float = 1000.0):
        """
        Инициализация бумажной биржи

        Args:
            initial_capital: Начальный капитал в USDT
        """
        self.balance = initial_capital  # Виртуальный баланс USDT
        self.initial_capital = initial_capital
        self.positions = {}  # Открытые позиции {position_id: position_data}
        self.orders = {}  # Активные ордера {order_id: order_data}
        self.trades_history = []  # История сделок
        self.closed_positions = []  # История закрытых позиций

        # Параметры симуляции
        self.slippage = 0.0005  # 0.05% проскальзывание
        self.commission = 0.001  # 0.1% комиссия

        print(f"[PaperExchange] Initialized with capital: {initial_capital} USDT")

    def create_market_order(self, symbol: str, side: str, amount: float,
                           position_id: Optional[str] = None, is_closing: bool = False) -> dict:
        """
        Симулирует рыночный ордер (market order)

        ФЬЮЧЕРСЫ: Balance НЕ меняется при открытии позиции!
        Balance меняется только при ЗАКРЫТИИ позиции (добавляется PnL).

        Args:
            symbol: Торговая пара (например, 'BTCUSDT')
            side: Направление ('BUY' или 'SELL')
            amount: Количество базовой валюты
            position_id: ID позиции (для закрытия позиции)
            is_closing: True если это закрытие позиции

        Returns:
            dict: Данные исполненного ордера

        Raises:
            InsufficientBalance: Недостаточно средств
            InvalidOrder: Некорректные параметры
        """
        side = side.upper()
        if side not in ['BUY', 'SELL']:
            raise InvalidOrder(f"Invalid side: {side}. Must be 'BUY' or 'SELL'")

        if amount <= 0:
            raise InvalidOrder(f"Invalid amount: {amount}. Must be positive")

        # Получаем реальную цену с Binance
        try:
            real_price = get_binance_price(symbol)
        except Exception as e:
            raise Exception(f"Failed to get market price: {e}")

        # Применяем проскальзывание
        if side == 'BUY':
            exec_price = real_price * (1 + self.slippage)
        else:  # SELL
            exec_price = real_price * (1 - self.slippage)

        # Рассчитываем стоимость
        cost = amount * exec_price
        commission_amount = cost * self.commission

        # ФЬЮЧЕРСЫ: Balance НЕ меняется при открытии!
        # Balance изменяется только при ЗАКРЫТИИ позиции в методе close_position()

        # Создаем запись о сделке
        order = {
            'id': generate_order_id(),
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'amount': amount,
            'price': exec_price,
            'real_price': real_price,
            'cost': cost,
            'commission': commission_amount,
            'total': cost + commission_amount if side == 'BUY' else cost - commission_amount,
            'status': 'filled',
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'position_id': position_id,
            'is_closing': is_closing
        }

        self.trades_history.append(order.copy())

        print(f"[PaperExchange] Market {side}: {amount:.6f} {symbol} @ {exec_price:.2f} "
              f"(slippage: {abs(exec_price - real_price):.2f}, commission: {commission_amount:.2f})")

        return order

    def open_position(self, symbol: str, side: str, amount: float,
                     entry_price: Optional[float] = None,
                     tp_price: Optional[float] = None,
                     sl_price: Optional[float] = None,
                     metadata: Optional[Dict] = None,
                     leverage: float = 1.0) -> dict:
        """
        Открывает новую позицию (LONG или SHORT)

        Args:
            symbol: Торговая пара
            side: Направление ('LONG' или 'SHORT')
            amount: Размер позиции
            entry_price: Цена входа (если None, используется текущая рыночная)
            tp_price: Цена Take Profit (опционально)
            sl_price: Цена Stop Loss (опционально)
            metadata: Дополнительные данные (entry_snapshot и т.д.)
            leverage: Плечо (по умолчанию 1x)

        Returns:
            dict: Данные открытой позиции

        Raises:
            InsufficientBalance: Недостаточно маржи для открытия позиции
        """
        side = side.upper()
        if side not in ['LONG', 'SHORT']:
            raise InvalidOrder(f"Invalid side: {side}. Must be 'LONG' or 'SHORT'")

        # ФЬЮЧЕРСЫ: Проверяем достаточность маржи ПЕРЕД открытием
        if entry_price is None:
            entry_price = get_binance_price(symbol)

        position_value = entry_price * amount
        required_margin = position_value / leverage

        free_margin = self.get_free_margin(leverage)

        if required_margin > free_margin:
            raise InsufficientBalance(
                f"Insufficient margin to open position: "
                f"need ${required_margin:.2f}, have ${free_margin:.2f} free margin "
                f"(balance: ${self.balance:.2f}, used: ${self.get_used_margin(leverage):.2f})"
            )

        # Открываем позицию через market order
        market_side = 'BUY' if side == 'LONG' else 'SELL'

        try:
            order = self.create_market_order(symbol, market_side, amount)
        except Exception as e:
            raise Exception(f"Failed to open position: {e}")

        # Создаем позицию
        position_id = generate_order_id()
        position = {
            'id': position_id,
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'entry_price': order['price'],
            'entry_order_id': order['id'],
            'status': 'open',
            'opened_at': time.time(),
            'opened_datetime': datetime.now().isoformat(),
            'tp_price': tp_price,
            'sl_price': sl_price,
            'tp_order_id': None,
            'sl_order_id': None,
            'metadata': metadata or {}
        }

        # Создаем TP/SL ордера если указаны
        if tp_price:
            tp_side = 'SELL' if side == 'LONG' else 'BUY'
            tp_order = self.create_limit_order(
                symbol, tp_side, amount, tp_price, position_id=position_id
            )
            position['tp_order_id'] = tp_order['id']

        if sl_price:
            sl_side = 'SELL' if side == 'LONG' else 'BUY'
            sl_order = self.create_stop_loss(
                symbol, sl_side, amount, sl_price, position_id=position_id
            )
            position['sl_order_id'] = sl_order['id']

        self.positions[position_id] = position

        print(f"[PaperExchange] Position opened: {side} {amount:.6f} {symbol} "
              f"@ {order['price']:.2f} (TP: {tp_price}, SL: {sl_price})")

        return position

    def close_position(self, position_id: str, reason: str = "manual") -> dict:
        """
        Закрывает открытую позицию

        Args:
            position_id: ID позиции
            reason: Причина закрытия ('manual', 'tp', 'sl')

        Returns:
            dict: Данные закрытой позиции с PnL
        """
        if position_id not in self.positions:
            raise OrderNotFound(f"Position {position_id} not found")

        position = self.positions[position_id]

        # Закрываем позицию противоположным ордером
        close_side = 'SELL' if position['side'] == 'LONG' else 'BUY'

        try:
            close_order = self.create_market_order(
                position['symbol'],
                close_side,
                position['amount'],
                position_id=position_id,
                is_closing=True
            )
        except Exception as e:
            raise Exception(f"Failed to close position: {e}")

        # Рассчитываем PnL
        if position['side'] == 'LONG':
            pnl = (close_order['price'] - position['entry_price']) * position['amount']
        else:  # SHORT
            pnl = (position['entry_price'] - close_order['price']) * position['amount']

        # Вычитаем комиссии на вход и выход
        total_commission = close_order['commission']
        # Находим комиссию при входе
        for trade in reversed(self.trades_history):
            if trade['id'] == position['entry_order_id']:
                total_commission += trade['commission']
                break

        pnl_after_commission = pnl - total_commission
        pnl_percent = (pnl_after_commission / (position['entry_price'] * position['amount'])) * 100

        # ФЬЮЧЕРСЫ: Добавляем PnL в balance при закрытии
        self.balance += pnl_after_commission

        # Обновляем позицию
        position['status'] = 'closed'
        position['close_price'] = close_order['price']
        position['close_order_id'] = close_order['id']
        position['closed_at'] = time.time()
        position['closed_datetime'] = datetime.now().isoformat()
        position['pnl'] = pnl
        position['pnl_after_commission'] = pnl_after_commission
        position['pnl_percent'] = pnl_percent
        position['close_reason'] = reason
        position['duration'] = position['closed_at'] - position['opened_at']

        # Удаляем связанные ордера TP/SL
        if position.get('tp_order_id') and position['tp_order_id'] in self.orders:
            del self.orders[position['tp_order_id']]
        if position.get('sl_order_id') and position['sl_order_id'] in self.orders:
            del self.orders[position['sl_order_id']]

        # Перемещаем в историю
        self.closed_positions.append(position.copy())
        del self.positions[position_id]

        print(f"[PaperExchange] Position closed: {position['side']} {position['symbol']} "
              f"PnL: {pnl_after_commission:.2f} USDT ({pnl_percent:.2f}%) - Reason: {reason}")

        return position

    def create_limit_order(self, symbol: str, side: str, amount: float,
                          price: float, position_id: Optional[str] = None) -> dict:
        """
        Создает лимитный ордер (для Take Profit)

        Args:
            symbol: Торговая пара
            side: Направление ('BUY' или 'SELL')
            amount: Количество
            price: Цена исполнения
            position_id: ID связанной позиции

        Returns:
            dict: Данные созданного ордера
        """
        side = side.upper()
        if side not in ['BUY', 'SELL']:
            raise InvalidOrder(f"Invalid side: {side}")

        order = {
            'id': generate_order_id(),
            'symbol': symbol,
            'side': side,
            'type': 'LIMIT',
            'amount': amount,
            'price': price,
            'status': 'open',
            'position_id': position_id,
            'created_at': time.time(),
            'created_datetime': datetime.now().isoformat()
        }

        self.orders[order['id']] = order

        print(f"[PaperExchange] Limit order created: {side} {amount:.6f} {symbol} @ {price:.2f}")

        return order

    def create_stop_loss(self, symbol: str, side: str, amount: float,
                        stop_price: float, position_id: Optional[str] = None) -> dict:
        """
        Создает стоп-лосс ордер

        Args:
            symbol: Торговая пара
            side: Направление ('BUY' или 'SELL')
            amount: Количество
            stop_price: Цена срабатывания стопа
            position_id: ID связанной позиции

        Returns:
            dict: Данные созданного ордера
        """
        side = side.upper()
        if side not in ['BUY', 'SELL']:
            raise InvalidOrder(f"Invalid side: {side}")

        order = {
            'id': generate_order_id(),
            'symbol': symbol,
            'side': side,
            'type': 'STOP_LOSS',
            'amount': amount,
            'stop_price': stop_price,
            'status': 'open',
            'position_id': position_id,
            'created_at': time.time(),
            'created_datetime': datetime.now().isoformat()
        }

        self.orders[order['id']] = order

        print(f"[PaperExchange] Stop-loss order created: {side} {amount:.6f} {symbol} @ {stop_price:.2f}")

        return order

    def create_stop_market_order(self, symbol: str, side: str, amount: float,
                                 stop_price: float, position_id: Optional[str] = None) -> dict:
        """
        Создает stop-market ордер (алиас для create_stop_loss для совместимости с BinanceClient)

        Args:
            symbol: Торговая пара
            side: Направление ('BUY' или 'SELL')
            amount: Количество
            stop_price: Цена срабатывания стопа
            position_id: ID связанной позиции

        Returns:
            dict: Данные созданного ордера
        """
        return self.create_stop_loss(symbol, side, amount, stop_price, position_id)

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Отменяет ордер

        Args:
            symbol: Торговая пара
            order_id: ID ордера

        Returns:
            True если успешно отменен
        """
        if order_id in self.orders:
            del self.orders[order_id]
            print(f"[PaperExchange] Order {order_id} cancelled for {symbol}")
            return True
        else:
            print(f"[PaperExchange] Warning: Order {order_id} not found for {symbol}")
            return False

    def check_orders(self) -> List[dict]:
        """
        Проверяет и исполняет TP/SL ордера на основе реальных цен

        Использует данные свечей (high/low) для точного определения
        достижения цен TP/SL

        Returns:
            List[dict]: Список исполненных ордеров
        """
        executed_orders = []

        for order_id, order in list(self.orders.items()):
            symbol = order['symbol']

            try:
                # Получаем данные свечи для проверки high/low
                kline = get_binance_kline(symbol, interval='1m', limit=1)
                current_price = kline['close']
                high_price = kline['high']
                low_price = kline['low']

            except Exception as e:
                print(f"[PaperExchange] Warning: Failed to check order {order_id}: {e}")
                continue

            executed = False

            if order['type'] == 'LIMIT':
                # Take Profit order
                if order['side'] == 'SELL':
                    # LONG TP: цена выросла до target
                    if high_price >= order['price']:
                        executed = True
                elif order['side'] == 'BUY':
                    # SHORT TP: цена упала до target
                    if low_price <= order['price']:
                        executed = True

            elif order['type'] == 'STOP_LOSS':
                # Stop Loss order
                if order['side'] == 'SELL':
                    # LONG SL: цена упала до stop
                    if low_price <= order['stop_price']:
                        executed = True
                elif order['side'] == 'BUY':
                    # SHORT SL: цена выросла до stop
                    if high_price >= order['stop_price']:
                        executed = True

            if executed:
                # Исполняем ордер
                try:
                    position_id = order.get('position_id')

                    # Закрываем позицию
                    if position_id and position_id in self.positions:
                        reason = 'tp' if order['type'] == 'LIMIT' else 'sl'
                        closed_position = self.close_position(position_id, reason=reason)

                        executed_orders.append({
                            'order': order,
                            'position': closed_position,
                            'execution_price': current_price
                        })

                        print(f"[PaperExchange] Order executed: {order['type']} {order['symbol']} "
                              f"@ {current_price:.2f} (target: {order.get('price') or order.get('stop_price'):.2f})")
                    else:
                        # Просто исполняем ордер без позиции
                        self.create_market_order(
                            symbol=order['symbol'],
                            side=order['side'],
                            amount=order['amount']
                        )
                        del self.orders[order_id]

                except Exception as e:
                    print(f"[PaperExchange] Error executing order {order_id}: {e}")
                    continue

        return executed_orders

    def get_used_margin(self, leverage: float = 1.0) -> float:
        """
        Рассчитывает используемую маржу (заблокированную в позициях)

        ФЬЮЧЕРСЫ: Маржа = сумма всех position_value / leverage

        Args:
            leverage: Плечо (по умолчанию 1x)

        Returns:
            Используемая маржа в USDT
        """
        used_margin = 0.0

        for position in self.positions.values():
            # Position value при входе
            position_value = position['entry_price'] * position['amount']
            # С плечом 1x: маржа = position_value
            # С плечом 10x: маржа = position_value / 10
            used_margin += position_value / leverage

        return used_margin

    def get_free_margin(self, leverage: float = 1.0) -> float:
        """
        Рассчитывает свободную маржу (доступную для новых позиций)

        ФЬЮЧЕРСЫ: Free Margin = Balance - Used Margin

        Args:
            leverage: Плечо (по умолчанию 1x)

        Returns:
            Свободная маржа в USDT
        """
        used_margin = self.get_used_margin(leverage)
        free_margin = self.balance - used_margin
        return max(0.0, free_margin)  # Не может быть отрицательной

    def get_balance(self, asset: str = 'USDT') -> float:
        """
        Возвращает текущий баланс (без учета маржи)

        ВАЖНО ДЛЯ ФЬЮЧЕРСОВ:
        - Этот метод возвращает ПОЛНЫЙ баланс
        - Для расчета доступных средств используйте get_free_margin()

        Args:
            asset: Актив ('USDT' по умолчанию) - для совместимости с BinanceClient

        Returns:
            Текущий баланс
        """
        return self.balance

    def get_current_price(self, symbol: str) -> float:
        """
        Получает текущую цену символа с Binance API

        Args:
            symbol: Торговая пара (например, 'BTCUSDT')

        Returns:
            Текущая цена

        Raises:
            Exception: Ошибка при получении цены
        """
        return get_binance_price(symbol)

    def get_equity(self) -> float:
        """
        Рассчитывает полный капитал (баланс + unrealized PnL)

        ФЬЮЧЕРСЫ: Balance НЕ меняется при открытии позиции!
        Equity = balance + unrealized_pnl всех открытых позиций.

        При открытии:
        - balance остается прежним (маржа блокируется, но баланс НЕ меняется)
        - unrealized_pnl = 0 (цена не изменилась)
        - equity = balance

        При изменении цены:
        - balance НЕ меняется
        - unrealized_pnl растет/падает
        - equity = balance + unrealized_pnl

        При закрытии:
        - balance += pnl_after_commission
        - позиция удаляется
        - equity = balance (новый)

        Returns:
            float: Общий капитал в USDT
        """
        equity = self.balance  # Баланс фиксирован до закрытия позиций

        # Добавляем нереализованный PnL всех открытых позиций
        for position in self.positions.values():
            try:
                current_price = get_binance_price(position['symbol'])

                # ФЬЮЧЕРСЫ: Считаем unrealized PnL
                if position['side'] == 'LONG':
                    unrealized_pnl = (current_price - position['entry_price']) * position['amount']
                else:  # SHORT
                    unrealized_pnl = (position['entry_price'] - current_price) * position['amount']

                # Вычитаем комиссию на вход (комиссия на выход будет при закрытии)
                for trade in reversed(self.trades_history):
                    if trade['id'] == position['entry_order_id']:
                        unrealized_pnl -= trade['commission']
                        break

                equity += unrealized_pnl

            except Exception as e:
                print(f"[PaperExchange] Warning: Failed to get equity for {position['id']}: {e}")
                continue

        return equity

    def get_position(self, position_id: str) -> Optional[dict]:
        """Возвращает данные позиции по ID"""
        return self.positions.get(position_id)

    def get_open_positions(self) -> List[dict]:
        """Возвращает список всех открытых позиций"""
        return list(self.positions.values())

    def get_open_orders(self) -> List[dict]:
        """Возвращает список всех активных ордеров"""
        return list(self.orders.values())

    def get_all_usdt_pairs(self, min_volume_24h: float = 1_000_000) -> List[str]:
        """
        Получает все торговые пары USDT с Binance

        Фильтры:
        - Только USDT пары (BTCUSDT, ETHUSDT, ...)
        - Только активные пары (status = 'TRADING')
        - Минимальный объем 24h > $1M (по умолчанию)
        - Исключить стейблкоины (USDCUSDT, BUSDUSDT, ...)

        Args:
            min_volume_24h: Минимальный объем за 24ч в USD

        Returns:
            Список символов (~300-400 пар)
        """
        print(f"[PaperExchange] Fetching all USDT pairs with min volume ${min_volume_24h:,.0f}")

        # MAINNET endpoints (НЕ testnet!)
        base_urls = [
            "https://api.binance.com",
            "https://api1.binance.com",
            "https://api-gcp.binance.com",
        ]

        # Список стейблкоинов для исключения
        stablecoins = ['USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'DAIUSDT', 'USDPUSDT', 'FDUSDUSDT']

        try:
            # Получаем информацию о всех парах
            exchange_info = None
            last_error = None

            for base_url in base_urls:
                try:
                    url = f"{base_url}/api/v3/exchangeInfo"
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    exchange_info = response.json()
                    break
                except Exception as e:
                    last_error = e
                    continue

            if exchange_info is None:
                raise Exception(f"Failed to get exchange info from all endpoints: {last_error}")

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
                    ticker = None
                    for base_url in base_urls:
                        try:
                            url = f"{base_url}/api/v3/ticker/24hr"
                            response = requests.get(url, params={'symbol': symbol}, timeout=10)
                            response.raise_for_status()
                            ticker = response.json()
                            break
                        except:
                            continue

                    if ticker is None:
                        continue

                    volume_24h = float(ticker['quoteVolume'])

                    if volume_24h >= min_volume_24h:
                        pairs.append(symbol)

                except Exception:
                    continue

            print(f"[PaperExchange] Found {len(pairs)} USDT pairs meeting criteria")
            return pairs

        except Exception as e:
            print(f"[PaperExchange] Error getting USDT pairs: {e}")
            # Возвращаем хотя бы топовые пары если не получилось
            return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT',
                    'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'AVAXUSDT']

    def get_ticker(self, symbol: str) -> dict:
        """
        Получает ticker данные для символа

        Args:
            symbol: Торговая пара

        Returns:
            dict: Ticker данные
        """
        return get_binance_ticker(symbol)

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> 'pd.DataFrame':
        """
        Получает OHLCV данные для символа

        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            limit: Количество свечей

        Returns:
            pd.DataFrame: DataFrame с колонками [timestamp, open, high, low, close, volume]
        """
        import pandas as pd

        # Конвертация timeframe в Binance формат
        tf_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '4h', '1d': '1d'
        }
        interval = tf_map.get(timeframe, timeframe)

        # Получаем klines через публичное API
        base_urls = [
            "https://api.binance.com",
            "https://api1.binance.com",
            "https://api-gcp.binance.com",
        ]

        params = {
            'symbol': symbol.replace('/', '').upper(),
            'interval': interval,
            'limit': limit
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }

        for base_url in base_urls:
            try:
                url = f"{base_url}/api/v3/klines"
                response = requests.get(url, params=params, headers=headers, timeout=10)
                response.raise_for_status()

                data = response.json()

                # Преобразуем в DataFrame
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])

                # Оставляем только нужные колонки и конвертируем типы
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)

                # Устанавливаем timestamp как DatetimeIndex (требуется для BaseLogic)
                df = df.set_index('timestamp')
                # Добавляем timestamp как колонку для совместимости
                df['timestamp'] = df.index

                return df

            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Failed to fetch from {base_url}: {e}")
                continue

        # Все endpoints failed - возвращаем пустой DataFrame с DatetimeIndex
        import pandas as pd
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"All endpoints failed for {symbol} {timeframe}, returning empty DataFrame")

        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        empty_df.index = pd.DatetimeIndex([])
        empty_df.index.name = 'timestamp'
        empty_df['timestamp'] = empty_df.index
        return empty_df

    def get_statistics(self) -> dict:
        """
        Возвращает статистику по торговле

        Returns:
            dict: Статистика (винрейт, средний PnL, и т.д.)
        """
        if not self.closed_positions:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'winrate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'roi': 0.0
            }

        wins = [p for p in self.closed_positions if p['pnl_after_commission'] > 0]
        losses = [p for p in self.closed_positions if p['pnl_after_commission'] <= 0]

        total_pnl = sum(p['pnl_after_commission'] for p in self.closed_positions)
        total_wins = sum(p['pnl_after_commission'] for p in wins) if wins else 0
        total_losses = abs(sum(p['pnl_after_commission'] for p in losses)) if losses else 0

        return {
            'total_trades': len(self.closed_positions),
            'wins': len(wins),
            'losses': len(losses),
            'winrate': len(wins) / len(self.closed_positions) * 100 if self.closed_positions else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(self.closed_positions),
            'avg_win': total_wins / len(wins) if wins else 0,
            'avg_loss': total_losses / len(losses) if losses else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'roi': (total_pnl / self.initial_capital) * 100,
            'current_equity': self.get_equity()
        }

    def save_state(self, filepath: str = "paper_exchange_state.json"):
        """
        Сохраняет состояние биржи в JSON файл

        Args:
            filepath: Путь к файлу для сохранения
        """
        state = {
            'balance': self.balance,
            'initial_capital': self.initial_capital,
            'positions': self.positions,
            'orders': self.orders,
            'trades_history': self.trades_history,
            'closed_positions': self.closed_positions,
            'slippage': self.slippage,
            'commission': self.commission,
            'saved_at': time.time(),
            'saved_datetime': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"[PaperExchange] State saved to {filepath}")

    @classmethod
    def load_state(cls, filepath: str = "paper_exchange_state.json") -> 'PaperExchange':
        """
        Загружает состояние биржи из JSON файла

        Args:
            filepath: Путь к файлу состояния

        Returns:
            PaperExchange: Восстановленный экземпляр биржи
        """
        with open(filepath, 'r') as f:
            state = json.load(f)

        # Создаем новый экземпляр
        exchange = cls(initial_capital=state['initial_capital'])

        # Восстанавливаем состояние
        exchange.balance = state['balance']
        exchange.positions = state['positions']
        exchange.orders = state['orders']
        exchange.trades_history = state['trades_history']
        exchange.closed_positions = state['closed_positions']
        exchange.slippage = state.get('slippage', 0.0005)
        exchange.commission = state.get('commission', 0.001)

        print(f"[PaperExchange] State loaded from {filepath}")
        print(f"[PaperExchange] Balance: {exchange.balance:.2f} USDT, "
              f"Open positions: {len(exchange.positions)}, "
              f"Active orders: {len(exchange.orders)}")

        return exchange


# ============================================================================
# MAIN (для тестирования)
# ============================================================================

if __name__ == "__main__":
    # Пример использования
    print("=" * 80)
    print("PAPER EXCHANGE - TEST RUN")
    print("=" * 80)

    # Создаем биржу
    exchange = PaperExchange(initial_capital=1000.0)

    # Тест 1: Открытие LONG позиции
    print("\n[TEST 1] Opening LONG position...")
    try:
        position = exchange.open_position(
            symbol='BTCUSDT',
            side='LONG',
            amount=0.01,
            tp_price=None,  # Установим позже
            sl_price=None
        )
        print(f"✓ Position opened: {position['id']}")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Тест 2: Проверка баланса и equity
    print("\n[TEST 2] Checking balance and equity...")
    print(f"Balance: {exchange.get_balance():.2f} USDT")
    print(f"Equity: {exchange.get_equity():.2f} USDT")

    # Тест 3: Сохранение состояния
    print("\n[TEST 3] Saving state...")
    exchange.save_state("/tmp/test_paper_exchange.json")

    # Тест 4: Загрузка состояния
    print("\n[TEST 4] Loading state...")
    loaded_exchange = PaperExchange.load_state("/tmp/test_paper_exchange.json")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
