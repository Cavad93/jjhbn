"""
AI Trading Module - Autonomous trading using Claude Sonnet 4.5

This module implements an AI-powered trading system that makes decisions
independently based on market analysis, fundamental research, and historical
performance.

Features:
- Autonomous coin selection (top-20 from ~400 pairs)
- Technical and fundamental analysis
- Self-learning from historical decisions
- Position sizing and risk management
- Daily decision-making cycle
"""

from .ai_trader import AITrader
from .config import AITradingConfig

__all__ = ['AITrader', 'AITradingConfig']
__version__ = '1.0.0'
