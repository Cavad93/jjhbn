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

    Trailing (два режима):
    1. Процентный (legacy): SL подтягивается на расстоянии distance_pct от пиковой цены
    2. ATR-based (рекомендуется): SL подтягивается на расстоянии (ATR * distance_atr_mult) от пиковой цены
    """

    def __init__(
        self,
        enabled: bool = None,
        activation_pct: float = None,
        distance_pct: float = None,
        distance_atr_mult: float = None
    ):
        """
        Args:
            enabled: Включить trailing stop (default from config)
            activation_pct: Процент прибыли для активации (default from config)
            distance_pct: Расстояние trailing stop в % (default from config, legacy mode)
            distance_atr_mult: Множитель ATR для расстояния trailing stop (default from config)
                              Если задан, используется вместо distance_pct (ATR-based mode)
        """
        self.enabled = enabled if enabled is not None else config.TRAILING_STOP_ENABLED
        self.activation_pct = activation_pct if activation_pct is not None else config.TRAILING_STOP_ACTIVATION_PCT
        self.distance_pct = distance_pct if distance_pct is not None else config.TRAILING_STOP_DISTANCE_PCT

        # ATR-based trailing (приоритет над процентным)
        self.distance_atr_mult = distance_atr_mult if distance_atr_mult is not None else getattr(config, 'TRAILING_STOP_DISTANCE_ATR_MULT', None)

        # Отслеживание максимальной/минимальной цены для каждой позиции
        self._peak_prices = {}  # symbol -> peak_price

    def check_trailing_stop(self, position, current_price: float, atr: float = None) -> Optional[float]:
        """
        Проверяет, нужно ли обновить SL для trailing stop

        Args:
            position: Position объект или dict (должен иметь atr_value если atr не передан)
            current_price: Текущая цена
            atr: Значение ATR (опционально, если не передано - берется из position.atr_value или metadata)

        Returns:
            new_sl_price: Новый уровень SL, если нужно обновить
            None: Если обновление не требуется
        """
        if not self.enabled:
            return None

        # Универсальный доступ к полям (работает с Position объектом и dict)
        def get_field(obj, field):
            if isinstance(obj, dict):
                return obj.get(field)
            return getattr(obj, field, None)

        symbol = get_field(position, 'symbol')
        direction = get_field(position, 'direction')
        entry_price = get_field(position, 'entry_price')
        current_sl = get_field(position, 'sl_price')

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

        # Определяем режим: ATR-based или процентный
        use_atr_mode = self.distance_atr_mult is not None

        if use_atr_mode:
            # ATR-based режим: получаем ATR
            if atr is None:
                # Пытаемся взять ATR из позиции (объект Position)
                atr = getattr(position, 'atr_value', None)

                # Если не нашли, пробуем взять из metadata (для PaperExchange dict)
                if atr is None and isinstance(position, dict):
                    metadata = position.get('metadata', {})
                    atr = metadata.get('atr_value')

                # Если всё ещё None - fallback к процентному режиму
                if atr is None:
                    # Fallback: используем процентный режим
                    use_atr_mode = False

        if direction == 'LONG':
            # Для LONG: отслеживаем максимальную цену
            if current_price > self._peak_prices[symbol]:
                self._peak_prices[symbol] = current_price

            # Рассчитываем новый SL
            if use_atr_mode:
                # ATR-based: SL = peak_price - (ATR * multiplier)
                new_sl = self._peak_prices[symbol] - (atr * self.distance_atr_mult)
            else:
                # Процентный: SL = peak_price * (1 - distance_pct/100)
                new_sl = self._peak_prices[symbol] * (1 - self.distance_pct / 100)

            # КРИТИЧЕСКИ ВАЖНО: новый SL не должен быть ниже точки входа!
            # Трейлинг-стоп предназначен для защиты прибыли, а не создания убытков
            new_sl = max(new_sl, entry_price)

            # Обновляем SL только если он выше текущего
            # Trailing stop МОЖЕТ и ДОЛЖЕН поднимать SL выше entry_price для защиты прибыли
            if new_sl > current_sl:
                return new_sl

        else:  # SHORT
            # Для SHORT: отслеживаем минимальную цену
            if current_price < self._peak_prices[symbol]:
                self._peak_prices[symbol] = current_price

            # Рассчитываем новый SL
            if use_atr_mode:
                # ATR-based: SL = peak_price + (ATR * multiplier)
                new_sl = self._peak_prices[symbol] + (atr * self.distance_atr_mult)
            else:
                # Процентный: SL = peak_price * (1 + distance_pct/100)
                new_sl = self._peak_prices[symbol] * (1 + self.distance_pct / 100)

            # КРИТИЧЕСКИ ВАЖНО: новый SL не должен быть выше точки входа!
            # Трейлинг-стоп предназначен для защиты прибыли, а не создания убытков
            new_sl = min(new_sl, entry_price)

            # Обновляем SL только если он ниже текущего
            # Trailing stop МОЖЕТ и ДОЛЖЕН опускать SL ниже entry_price для защиты прибыли в SHORT
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
