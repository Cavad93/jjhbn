#!/usr/bin/env python3
"""
Black Swan Protection

Защита от резких обвалов рынка (Black Swan events).
Закрывает все позиции при резком падении цены Bitcoin.

Usage:
    from risk.black_swan import BlackSwanProtection

    protection = BlackSwanProtection()

    if protection.check_black_swan_event():
        # Закрыть все позиции немедленно
        close_all_positions()
"""

import time
from typing import Optional, Dict
import binance_config as config


class BlackSwanProtection:
    """
    Защита от Black Swan events (резкие обвалы рынка)

    Отслеживает:
    - Падение Bitcoin на X% за Y минут
    - Резкое изменение волатильности

    При обнаружении:
    - Закрывает все открытые позиции
    - Блокирует новые входы на N часов
    """

    def __init__(
        self,
        enabled: bool = None,
        price_drop_pct: float = None,
        check_interval_sec: int = None
    ):
        """
        Args:
            enabled: Включить защиту (default from config)
            price_drop_pct: Процент падения для триггера (default from config)
            check_interval_sec: Интервал проверки в секундах (default from config)
        """
        self.enabled = enabled if enabled is not None else config.BLACK_SWAN_PROTECTION
        self.price_drop_pct = price_drop_pct if price_drop_pct is not None else config.BLACK_SWAN_PRICE_DROP_PCT
        self.check_interval = check_interval_sec if check_interval_sec is not None else config.BLACK_SWAN_CHECK_INTERVAL_SEC

        # История цен BTC для отслеживания
        self._price_history: Dict[float, float] = {}  # timestamp -> price
        self._last_check_time = 0
        self._event_triggered = False
        self._event_trigger_time = 0
        self._block_duration_hours = 2  # Блокировка на 2 часа после события
        self._last_price_drop_pct = 0.0  # Последний рассчитанный процент падения BTC

    def update_btc_price(self, price: float):
        """
        Обновляет историю цен BTC

        Args:
            price: Текущая цена Bitcoin
        """
        if not self.enabled:
            return

        current_time = time.time()

        # Добавляем текущую цену
        self._price_history[current_time] = price

        # Удаляем старые данные (старше 10 минут)
        cutoff_time = current_time - 600  # 10 минут
        self._price_history = {
            t: p for t, p in self._price_history.items()
            if t >= cutoff_time
        }

    def check_black_swan_event(self, btc_price: Optional[float] = None) -> bool:
        """
        Проверяет наличие Black Swan события

        Args:
            btc_price: Текущая цена BTC (если None, использует последнюю известную)

        Returns:
            True если обнаружено событие, False иначе
        """
        if not self.enabled:
            return False

        current_time = time.time()

        # Если событие уже произошло, проверяем, не истек ли блок
        if self._event_triggered:
            time_since_event = current_time - self._event_trigger_time
            if time_since_event < self._block_duration_hours * 3600:
                # Блокировка все еще активна
                return True
            else:
                # Блокировка истекла, сбрасываем флаг
                self._event_triggered = False
                return False

        # Обновляем цену BTC если передана
        if btc_price is not None:
            self.update_btc_price(btc_price)

        # Проверяем только если прошел интервал
        if current_time - self._last_check_time < self.check_interval:
            return False

        self._last_check_time = current_time

        # Нужно минимум 2 точки данных
        if len(self._price_history) < 2:
            return False

        # Получаем самую старую и самую новую цену
        timestamps = sorted(self._price_history.keys())
        oldest_time = timestamps[0]
        newest_time = timestamps[-1]

        # Проверяем, прошло ли достаточно времени (минимум 5 минут)
        time_diff = newest_time - oldest_time
        if time_diff < 300:  # 5 минут
            return False

        oldest_price = self._price_history[oldest_time]
        newest_price = self._price_history[newest_time]

        # Рассчитываем падение в процентах
        price_drop = ((oldest_price - newest_price) / oldest_price) * 100

        # Сохраняем для последующего использования
        self._last_price_drop_pct = price_drop

        # Проверяем, превышен ли порог
        if price_drop >= self.price_drop_pct:
            print(f"\n⚠️  BLACK SWAN EVENT DETECTED!")
            print(f"   BTC dropped {price_drop:.2f}% in {time_diff/60:.1f} minutes")
            print(f"   Price: ${oldest_price:.2f} → ${newest_price:.2f}")
            print(f"   Blocking new entries for {self._block_duration_hours}h")

            self._event_triggered = True
            self._event_trigger_time = current_time
            return True

        return False

    def is_event_active(self) -> bool:
        """Проверяет, активно ли сейчас Black Swan событие"""
        if not self._event_triggered:
            return False

        current_time = time.time()
        time_since_event = current_time - self._event_trigger_time

        if time_since_event >= self._block_duration_hours * 3600:
            self._event_triggered = False
            return False

        return True

    def reset_event(self):
        """Сбрасывает Black Swan событие (вручную)"""
        self._event_triggered = False
        self._event_trigger_time = 0
        print("Black Swan event manually reset")

    def get_time_until_unblock(self) -> Optional[float]:
        """
        Возвращает время до разблокировки в секундах

        Returns:
            seconds: Секунды до разблокировки, или None если не заблокировано
        """
        if not self._event_triggered:
            return None

        current_time = time.time()
        time_since_event = current_time - self._event_trigger_time
        remaining_time = (self._block_duration_hours * 3600) - time_since_event

        return max(0, remaining_time)

    def get_last_price_drop_pct(self) -> float:
        """
        Возвращает последний рассчитанный процент падения BTC

        Returns:
            float: Процент падения BTC (положительное число означает падение)
                   0.0 если событие не было обнаружено или недостаточно данных
        """
        return self._last_price_drop_pct
