"""
Portfolio Management Module

Модуль для управления портфелем торговых позиций.

Основные компоненты:
- Position: Dataclass для хранения информации о позиции
- PositionManager: Менеджер для управления до 10 одновременными позициями
"""

from .position_manager import Position, PositionManager

__all__ = [
    'Position',
    'PositionManager'
]

__version__ = '1.0.0'
