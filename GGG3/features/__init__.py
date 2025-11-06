"""
Features module for Binance trading bot

Модуль для расчёта технических признаков (features) на основе multi-timeframe данных
"""

from .builder import BinanceFeatureBuilder

__all__ = ['BinanceFeatureBuilder']
