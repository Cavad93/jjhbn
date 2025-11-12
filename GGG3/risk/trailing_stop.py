#!/usr/bin/env python3
"""
Trailing Stop Loss Manager

Автоматически подтягивает stop-loss, когда позиция в прибыли,
для защиты прибыли при развороте цены.

Usage:
    from risk.trailing_stop import TrailingStopManager

    manager = TrailingStopManager()
    new_sl = manager.check_trailing_stop(position, current_price)

    if new_sl is not None:
        # Обновить SL ордер на бирже
        update_stop_loss(position.symbol, new_sl)
"""

from typing import Optional
import binance_config as config


class TrailingStopManager:
    """
    Управление trailing stop loss

    Активация:
    - Когда прибыль достигает TRAILING_STOP_ACTIVATION_PCT

    Trailing:
    - SL подтягивается на расстоянии TRAILING_STOP_DISTANCE_PCT от текущей цены
    """

    def __init__(
        self,
        enabled: bool = None,
        activation_pct: float = None,
        distance_pct: float = None
    ):
        """
        Args:
            enabled: Включить trailing stop (default from config)
            activation_pct: Процент прибыли для активации (default from config)
            distance_pct: Расстояние trailing stop от цены (default from config)
        """
        self.enabled = enabled if enabled is not None else config.TRAILING_STOP_ENABLED
        self.activation_pct = activation_pct if activation_pct is not None else config.TRAILING_STOP_ACTIVATION_PCT
        self.distance_pct = distance_pct if distance_pct is not None else config.TRAILING_STOP_DISTANCE_PCT

        # Отслеживание максимальной/минимальной цены для каждой позиции
        self._peak_prices = {}  # symbol -> peak_price

    def check_trailing_stop(self, position, current_price: float) -> Optional[float]:
        """
        Проверяет, нужно ли обновить SL для trailing stop

        Args:
            position: Position объект
            current_price: Текущая цена

        Returns:
            new_sl_price: Новый уровень SL, если нужно обновить
            None: Если обновление не требуется
        """
        if not self.enabled:
            return None

        symbol = position.symbol
        direction = position.direction
        entry_price = position.entry_price
        current_sl = position.sl_price

        # Рассчитываем текущую прибыль (в процентах)
        if direction == 'LONG':
            profit_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # SHORT
            profit_pct = ((entry_price - current_price) / entry_price) * 100

        # Проверяем, достигнут ли порог активации
        if profit_pct < self.activation_pct:
            return None

        # Отслеживаем максимальную/минимальную цену
        if symbol not in self._peak_prices:
            self._peak_prices[symbol] = current_price

        if direction == 'LONG':
            # Для LONG: отслеживаем максимальную цену
            if current_price > self._peak_prices[symbol]:
                self._peak_prices[symbol] = current_price

            # Рассчитываем новый SL (ниже максимальной цены на distance_pct)
            new_sl = self._peak_prices[symbol] * (1 - self.distance_pct / 100)

            # КРИТИЧНО: SL не должен подниматься выше entry_price для LONG!
            # Trailing stop защищает прибыль, но не должен создавать убыток
            if new_sl > entry_price:
                new_sl = entry_price

            # Обновляем SL только если он выше текущего
            if new_sl > current_sl:
                return new_sl

        else:  # SHORT
            # Для SHORT: отслеживаем минимальную цену
            if current_price < self._peak_prices[symbol]:
                self._peak_prices[symbol] = current_price

            # Рассчитываем новый SL (выше минимальной цены на distance_pct)
            new_sl = self._peak_prices[symbol] * (1 + self.distance_pct / 100)

            # КРИТИЧНО: SL не должен опускаться ниже entry_price для SHORT!
            # Trailing stop защищает прибыль, но не должен создавать убыток
            if new_sl < entry_price:
                new_sl = entry_price

            # Обновляем SL только если он ниже текущего
            if new_sl < current_sl:
                return new_sl

        return None

    def reset_position(self, symbol: str):
        """Сбрасывает отслеживание для закрытой позиции"""
        if symbol in self._peak_prices:
            del self._peak_prices[symbol]

    def get_peak_price(self, symbol: str) -> Optional[float]:
        """Возвращает отслеживаемую peak цену для позиции"""
        return self._peak_prices.get(symbol)
