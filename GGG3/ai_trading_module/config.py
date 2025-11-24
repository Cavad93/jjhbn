"""
Configuration for AI Trading Module
"""
import os
from typing import Dict, List


class AITradingConfig:
    """Configuration for AI-powered trading module"""

    # ===== MODE =====
    PAPER_TRADING = True
    INITIAL_CAPITAL = 1000.0  # $1000 для AI модуля

    # ===== AI MODEL =====
    AI_MODEL = "claude-sonnet-4-5-20250929"
    AI_TEMPERATURE = 0.7  # Баланс между креативностью и консистентностью
    AI_MAX_TOKENS = 4096  # Reduced to minimize token usage

    # ===== TRADING PARAMETERS =====
    MAX_POSITIONS = 10
    MIN_POSITIONS = 2  # Минимальное количество позиций для поддержания (reduced to avoid rate limits)

    # Размер позиции (% от капитала)
    MIN_POSITION_SIZE_PCT = 1.0   # 1%
    MAX_POSITION_SIZE_PCT = 10.0  # 10%
    DEFAULT_POSITION_SIZE_PCT = 5.0  # 5%

    # Риск-менеджмент
    MAX_TOTAL_EXPOSURE_PCT = 60.0  # Максимум 60% капитала в позициях
    MAX_SINGLE_RISK_PCT = 2.0      # Максимум 2% риска на позицию
    MAX_CORRELATION = 0.7          # Максимальная корреляция между позициями

    # ===== DECISION CYCLE =====
    DECISION_INTERVAL_HOURS = 24  # Принятие решений 1 раз в сутки
    DECISION_TIME_UTC = 10        # В 10:00 UTC

    # Проверка закрытия позиций
    CHECK_POSITIONS_INTERVAL_MINUTES = 60  # Каждый час

    # ===== COIN SELECTION =====
    TOP_COINS_TO_ANALYZE = 20  # AI выбирает топ-20 для детального анализа

    # Фильтры для первичного скрининга
    MIN_24H_VOLUME_USD = 10_000_000  # $10M минимум
    MIN_VOLATILITY_PCT = 2.0          # 2% минимальная волатильность
    MAX_VOLATILITY_PCT = 15.0         # 15% максимальная волатильность

    # Blacklist (высокий риск)
    BLACKLIST_SYMBOLS = [
        'LUNA', 'LUNC', 'UST', 'USTC',  # Crashed stablecoins
        'FTT',                           # FTX token
    ]

    # Stablecoins (исключить)
    STABLECOIN_SYMBOLS = [
        'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP',
        'FDUSD', 'GUSD', 'USDD'
    ]

    # ===== TP/SL PARAMETERS =====
    # AI определяет TP/SL, но с ограничениями
    MIN_TP_DISTANCE_PCT = 1.5   # Минимум 1.5% до TP
    MAX_TP_DISTANCE_PCT = 20.0  # Максимум 20% до TP

    MIN_SL_DISTANCE_PCT = 0.5   # Минимум 0.5% до SL
    MAX_SL_DISTANCE_PCT = 5.0   # Максимум 5% до SL

    MIN_RISK_REWARD_RATIO = 1.5  # Минимум 1.5:1

    # ===== FUNDAMENTAL ANALYSIS =====
    ENABLE_WEB_SEARCH = True
    MAX_SEARCH_QUERIES_PER_DAY = 50  # Ограничение на поиск

    # Whitelist источников для фундаментального анализа
    TRUSTED_SOURCES = [
        'coindesk.com',
        'cointelegraph.com',
        'decrypt.co',
        'theblock.co',
        'cryptoslate.com',
        'messari.io'
    ]

    # ===== SELF-LEARNING =====
    ENABLE_LEARNING = True
    MIN_HISTORICAL_DECISIONS = 10  # Минимум решений для анализа
    LEARNING_WINDOW_DAYS = 90      # Анализировать последние 90 дней

    # ===== API COSTS MANAGEMENT =====
    MAX_DAILY_API_COST_USD = 50.0  # Максимум $50/день на API вызовы

    # Примерная стоимость (для отслеживания бюджета)
    COST_PER_DECISION = 2.0        # ~$2 за полный цикл принятия решения
    COST_PER_SEARCH = 0.5          # ~$0.5 за веб-поиск
    COST_PER_ANALYSIS = 1.0        # ~$1 за анализ исторических данных

    # ===== DATA STORAGE =====
    # Use relative path to work on both Windows and Linux
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    DECISIONS_FILE = os.path.join(DATA_DIR, "decisions.json")
    POSITIONS_FILE = os.path.join(DATA_DIR, "ai_positions.json")
    STATE_FILE = os.path.join(DATA_DIR, "ai_state.json")

    # ===== TELEGRAM NOTIFICATIONS =====
    TELEGRAM_ENABLED = True
    TELEGRAM_SEND_DAILY_REPORT = True
    TELEGRAM_SEND_POSITION_ALERTS = True

    # ===== TECHNICAL INDICATORS =====
    # Таймфреймы для анализа
    TIMEFRAMES = ['5m', '15m', '30m', '4h', '1d']
    PRIMARY_TIMEFRAME = '4h'

    # Индикаторы для расчета
    INDICATORS = [
        'rsi',
        'macd',
        'stochastic',
        'atr',
        'bollinger_bands',
        'adx',
        'williams_r',
        'cci',
        'roc',
        'ema',
        'volume_indicators'
    ]

    # ===== SAFETY LIMITS =====
    # Hard limits для защиты
    EMERGENCY_STOP_DRAWDOWN_PCT = 20.0  # Остановить торговлю при 20% просадке
    MAX_CONSECUTIVE_LOSSES = 5          # Остановить после 5 убытков подряд
    MIN_CAPITAL_TO_CONTINUE = 500.0     # Минимум $500 для продолжения

    # ===== LOGGING =====
    LOG_LEVEL = "INFO"
    LOG_FILE = os.path.join(DATA_DIR, "ai_trader.log")

    @classmethod
    def validate(cls) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        if cls.MAX_POSITION_SIZE_PCT > 100 / cls.MIN_POSITIONS:
            errors.append(
                f"MAX_POSITION_SIZE_PCT ({cls.MAX_POSITION_SIZE_PCT}%) too high "
                f"for MIN_POSITIONS ({cls.MIN_POSITIONS})"
            )

        if cls.MIN_RISK_REWARD_RATIO < 1.0:
            errors.append("MIN_RISK_REWARD_RATIO should be >= 1.0")

        if cls.MAX_TOTAL_EXPOSURE_PCT > 100:
            errors.append("MAX_TOTAL_EXPOSURE_PCT cannot exceed 100%")

        return errors

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get system prompt for AI Trading Agent"""
        return f"""You are an AI Trading Agent managing a cryptocurrency portfolio on Binance Futures.

CAPITAL: ${cls.INITIAL_CAPITAL}
OBJECTIVE: Maximize profit while managing risk through data-driven decisions.

CONSTRAINTS:
- Maximum {cls.MAX_POSITIONS} simultaneous positions
- Position size: {cls.MIN_POSITION_SIZE_PCT}-{cls.MAX_POSITION_SIZE_PCT}% of capital
- Total exposure: max {cls.MAX_TOTAL_EXPOSURE_PCT}% of capital
- Mandatory TP/SL for every position (Risk:Reward >= {cls.MIN_RISK_REWARD_RATIO}:1)
- Decisions made once per {cls.DECISION_INTERVAL_HOURS}h
- **API EFFICIENCY**: Minimize tool calls - batch requests when possible, avoid redundant analysis
- **Rate Limits**: Stay within API limits (30k tokens/min) - be concise and focused

DECISION PROCESS:
1. MARKET SCREENING: Analyze ~400 trading pairs
   - Filter by volume (>${cls.MIN_24H_VOLUME_USD/1e6}M USD), volatility ({cls.MIN_VOLATILITY_PCT}-{cls.MAX_VOLATILITY_PCT}%)
   - Exclude blacklist and stablecoins
   - Select EXACTLY top-{cls.TOP_COINS_TO_ANALYZE} candidates (DO NOT EXCEED THIS LIMIT)

2. DEEP ANALYSIS (for each top-{cls.TOP_COINS_TO_ANALYZE}):
   - Technical analysis: Use indicators across {', '.join(cls.TIMEFRAMES)} timeframes
   - Fundamental analysis: Search news, events, on-chain data (when relevant)
   - Historical performance: Learn from past decisions

3. POSITION DECISIONS:
   - Determine: LONG/SHORT/SKIP
   - Set position size ({cls.MIN_POSITION_SIZE_PCT}-{cls.MAX_POSITION_SIZE_PCT}%)
   - Define TP/SL levels (validate risk:reward >= {cls.MIN_RISK_REWARD_RATIO}:1)
   - Provide detailed reasoning

4. PORTFOLIO MANAGEMENT:
   - Maintain {cls.MIN_POSITIONS}-{cls.MAX_POSITIONS} positions
   - Ensure diversification (max correlation {cls.MAX_CORRELATION})
   - Monitor risk exposure (total <= {cls.MAX_TOTAL_EXPOSURE_PCT}%)

5. CONTINUOUS LEARNING:
   - Analyze outcomes of closed positions
   - Identify patterns in successful/unsuccessful decisions
   - Adapt selection criteria and risk parameters

REQUIREMENTS:
- Every decision MUST be justified with concrete data
- Consider both technical AND fundamental factors
- Learn from historical successes and failures
- Manage risk: diversification, position sizing, correlation
- Be adaptive: market conditions change

AVAILABLE TOOLS:
- get_market_data: Retrieve OHLCV data
- get_technical_indicators: Calculate technical indicators
- get_all_available_symbols: List all tradeable pairs
- calculate_symbol_metrics: Get volume, volatility, liquidity
- search_fundamental_info: Web search for news/events
- get_historical_decisions: Analyze past decisions
- open_position: Execute trade with TP/SL
- get_current_positions: View open positions
- get_portfolio_status: Check balance, PnL, exposure
- calculate_position_size: Size position based on risk

Remember: Your performance is measured by total return, Sharpe ratio, and risk-adjusted metrics.
Make decisions that balance opportunity with prudent risk management."""
