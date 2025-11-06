#!/usr/bin/env python3
"""
Night Mode Manager

Управляет ночным режимом торговли для снижения рисков
в периоды низкой ликвидности.

Usage:
    from risk.night_mode import NightModeManager

    manager = NightModeManager()

    if manager.is_night_mode_active():
        # Ограничить количество новых позиций
        max_positions = manager.get_max_positions()
"""

from datetime import datetime
from typing import Optional
import binance_config as config


class NightModeManager:
    """
    Управление ночным режимом торговли

    Ночной режим (UTC 23:00 - 07:00):
    - Ограничивает количество открытых позиций
    - Не открывает новые позиции в высокорискованных активах
    - Увеличивает минимальный порог вероятности для входа

    Причина: низкая ликвидность ночью увеличивает риски
    """

    def __init__(
        self,
        enabled: bool = None,
        start_hour: int = None,
        end_hour: int = None,
        max_positions: int = None
    ):
        """
        Args:
            enabled: Включить ночной режим (default from config)
            start_hour: Час начала ночного режима UTC (default from config)
            end_hour: Час окончания ночного режима UTC (default from config)
            max_positions: Максимум позиций ночью (default from config)
        """
        self.enabled = enabled if enabled is not None else config.NIGHT_MODE_ENABLED
        self.start_hour = start_hour if start_hour is not None else config.NIGHT_MODE_START_HOUR
        self.end_hour = end_hour if end_hour is not None else config.NIGHT_MODE_END_HOUR
        self.max_positions_night = max_positions if max_positions is not None else config.NIGHT_MODE_MAX_POSITIONS
        self.max_positions_day = config.MAX_POSITIONS

    def is_night_mode_active(self, current_time: Optional[datetime] = None) -> bool:
        """
        Проверяет, активен ли сейчас ночной режим

        Args:
            current_time: Время для проверки (default: сейчас UTC)

        Returns:
            True если ночной режим, False иначе
        """
        if not self.enabled:
            return False

        if current_time is None:
            current_time = datetime.utcnow()

        current_hour = current_time.hour

        # Проверяем, попадает ли текущий час в ночной диапазон
        if self.start_hour < self.end_hour:
            # Обычный случай (например, 23:00 - 7:00 через полночь не переходит)
            return self.start_hour <= current_hour < self.end_hour
        else:
            # Диапазон через полночь (например, 23:00 - 07:00)
            return current_hour >= self.start_hour or current_hour < self.end_hour

    def get_max_positions(self, current_time: Optional[datetime] = None) -> int:
        """
        Возвращает максимальное количество позиций для текущего времени

        Args:
            current_time: Время для проверки (default: сейчас UTC)

        Returns:
            max_positions: Максимальное количество позиций
        """
        if self.is_night_mode_active(current_time):
            return self.max_positions_night
        return self.max_positions_day

    def get_threshold_adjustment(self, current_time: Optional[datetime] = None) -> float:
        """
        Возвращает корректировку порога вероятности для ночного режима

        Args:
            current_time: Время для проверки (default: сейчас UTC)

        Returns:
            adjustment: Корректировка порога (добавляется к базовому порогу)
                       +0.02 ночью (повышаем требования)
                       +0.00 днем
        """
        if self.is_night_mode_active(current_time):
            return 0.02  # Повышаем порог на 2% ночью
        return 0.0

    def should_skip_high_risk_coins(self, current_time: Optional[datetime] = None) -> bool:
        """
        Проверяет, нужно ли пропускать высокорискованные монеты

        Args:
            current_time: Время для проверки (default: сейчас UTC)

        Returns:
            True если нужно пропустить, False иначе
        """
        return self.is_night_mode_active(current_time)

    def get_mode_info(self, current_time: Optional[datetime] = None) -> dict:
        """
        Возвращает информацию о текущем режиме

        Args:
            current_time: Время для проверки (default: сейчас UTC)

        Returns:
            dict с информацией о режиме
        """
        is_night = self.is_night_mode_active(current_time)

        if current_time is None:
            current_time = datetime.utcnow()

        return {
            'is_night_mode': is_night,
            'current_hour_utc': current_time.hour,
            'max_positions': self.get_max_positions(current_time),
            'threshold_adjustment': self.get_threshold_adjustment(current_time),
            'mode': '🌙 NIGHT' if is_night else '☀️ DAY'
        }

    def get_time_until_mode_change(self, current_time: Optional[datetime] = None) -> int:
        """
        Возвращает количество часов до смены режима

        Args:
            current_time: Время для проверки (default: сейчас UTC)

        Returns:
            hours: Часы до смены режима
        """
        if current_time is None:
            current_time = datetime.utcnow()

        current_hour = current_time.hour
        is_night = self.is_night_mode_active(current_time)

        if is_night:
            # Сейчас ночь, считаем до утра (end_hour)
            if current_hour < self.end_hour:
                return self.end_hour - current_hour
            else:
                return 24 - current_hour + self.end_hour
        else:
            # Сейчас день, считаем до вечера (start_hour)
            if current_hour < self.start_hour:
                return self.start_hour - current_hour
            else:
                return 24 - current_hour + self.start_hour
