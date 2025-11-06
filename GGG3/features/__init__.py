"""
Features module for Binance trading bot

Модуль для расчёта технических признаков (features) на основе multi-timeframe данных
и генерации датасетов для обучения ML моделей.
"""

from .builder import BinanceFeatureBuilder
from .target_calculator import (
    calculate_atr,
    calculate_tp_sl_outcome_bidirectional,
    detect_market_phase,
    prepare_training_dataset
)

__all__ = [
    'BinanceFeatureBuilder',
    'calculate_atr',
    'calculate_tp_sl_outcome_bidirectional',
    'detect_market_phase',
    'prepare_training_dataset'
]
