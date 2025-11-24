"""
AI Tools - Инструменты для Sonnet 4.5 Trading Agent

Эти инструменты предоставляют AI доступ к рыночным данным,
техническим индикаторам, фундаментальному анализу и управлению позициями.
"""
import sys
import os
import json
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from indicators.technical import (
    rsi, macd, stochastic, atr, bollinger_bands,
    adx, williams_r, cci, roc, ema, sma
)


# ===== TOOL DEFINITIONS (Anthropic Format) =====

TOOLS = [
    {
        "name": "get_market_data",
        "description": "Retrieve OHLCV (Open, High, Low, Close, Volume) market data for specified symbols and timeframes. Returns candlestick data that can be used for technical analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT'])"
                },
                "timeframes": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["5m", "15m", "30m", "4h", "1d"]},
                    "description": "List of timeframes to retrieve"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of candles to retrieve (default: 100, max: 500)",
                    "default": 100
                }
            },
            "required": ["symbols", "timeframes"]
        }
    },
    {
        "name": "get_technical_indicators",
        "description": "Calculate technical indicators for a given symbol and timeframe. Returns RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX, Williams %R, CCI, and ROC.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading symbol (e.g., 'BTCUSDT')"
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["5m", "15m", "30m", "4h", "1d"],
                    "description": "Timeframe for indicators"
                },
                "indicators": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["rsi", "macd", "bollinger_bands", "atr", "stochastic", "adx", "williams_r", "cci", "roc", "all"]
                    },
                    "description": "List of indicators to calculate, or 'all' for all indicators",
                    "default": ["all"]
                }
            },
            "required": ["symbol", "timeframe"]
        }
    },
    {
        "name": "get_all_available_symbols",
        "description": "Get list of all available trading pairs on Binance Futures with basic filters applied (volume, volatility). Returns filtered list of symbols ready for analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_volume_usd": {
                    "type": "number",
                    "description": "Minimum 24h volume in USD (default: 10M)",
                    "default": 10000000
                },
                "exclude_stablecoins": {
                    "type": "boolean",
                    "description": "Exclude stablecoin pairs (default: true)",
                    "default": True
                },
                "exclude_blacklist": {
                    "type": "boolean",
                    "description": "Exclude blacklisted symbols (default: true)",
                    "default": True
                }
            },
            "required": []
        }
    },
    {
        "name": "calculate_symbol_metrics",
        "description": "Calculate detailed metrics for a symbol: 24h volume, volatility (ATR%), liquidity score, price change %, and market phase. Useful for coin selection and ranking.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading symbol (e.g., 'BTCUSDT')"
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["4h", "1d"],
                    "description": "Timeframe for volatility calculation (default: 4h)",
                    "default": "4h"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "search_fundamental_info",
        "description": "Search for fundamental information about a cryptocurrency from CryptoCompare news API and CoinGecko trending data. Returns: recent news articles (titles, sources, dates), sentiment analysis (positive/negative/neutral with score), trending status on CoinGecko, and a summary. Use this to filter out coins with negative news or identify coins with positive catalysts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'Bitcoin recent news', 'Ethereum upgrade', 'Solana developments')"
                },
                "coin_name": {
                    "type": "string",
                    "description": "Cryptocurrency name or symbol (e.g., 'Bitcoin', 'BTC', 'BTCUSDT'). Used to filter news and check trending status."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_historical_decisions",
        "description": "Retrieve and analyze historical trading decisions made by the AI. Use this to learn from past successes and failures. Can filter by symbol, outcome, or time period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Filter by specific symbol (optional)"
                },
                "outcome": {
                    "type": "string",
                    "enum": ["success", "failure", "all"],
                    "description": "Filter by outcome (default: all)",
                    "default": "all"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of decisions to retrieve (default: 20)",
                    "default": 20
                },
                "days_back": {
                    "type": "integer",
                    "description": "Number of days to look back (default: 90)",
                    "default": 90
                }
            },
            "required": []
        }
    },
    {
        "name": "analyze_historical_performance",
        "description": "Get statistical analysis of historical decisions: win rate, average PnL, best/worst trades, common patterns in successful trades. Use this for self-learning and strategy improvement.",
        "input_schema": {
            "type": "object",
            "properties": {
                "groupby": {
                    "type": "string",
                    "enum": ["symbol", "direction", "timeframe", "all"],
                    "description": "Group analysis by symbol, direction (long/short), or timeframe",
                    "default": "all"
                },
                "days_back": {
                    "type": "integer",
                    "description": "Number of days to analyze (default: 90)",
                    "default": 90
                }
            },
            "required": []
        }
    },
    {
        "name": "open_position",
        "description": "Open a new trading position (LONG or SHORT) with specified take-profit and stop-loss levels. This executes the trade on the exchange (or paper trading environment).",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading symbol (e.g., 'BTCUSDT')"
                },
                "direction": {
                    "type": "string",
                    "enum": ["LONG", "SHORT"],
                    "description": "Position direction"
                },
                "size_pct": {
                    "type": "number",
                    "description": "Position size as percentage of capital (1-10%)",
                    "minimum": 1.0,
                    "maximum": 10.0
                },
                "tp_price": {
                    "type": "number",
                    "description": "Take profit price level"
                },
                "sl_price": {
                    "type": "number",
                    "description": "Stop loss price level"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Detailed reasoning for this trade decision (required for learning)"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence level in this trade (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.7
                }
            },
            "required": ["symbol", "direction", "size_pct", "tp_price", "sl_price", "reasoning"]
        }
    },
    {
        "name": "get_current_positions",
        "description": "Get all currently open positions with their current PnL, entry details, and TP/SL levels. Use this to monitor the portfolio.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_portfolio_status",
        "description": "Get comprehensive portfolio status: current balance, unrealized PnL, total exposure, number of positions, available capital, and risk metrics.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "calculate_position_size",
        "description": "Calculate optimal position size based on risk parameters, ATR, and current capital. Returns recommended position size in USDT and percentage of capital.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading symbol"
                },
                "direction": {
                    "type": "string",
                    "enum": ["LONG", "SHORT"],
                    "description": "Position direction"
                },
                "risk_pct": {
                    "type": "number",
                    "description": "Risk percentage per trade (default: 1.5%)",
                    "default": 1.5,
                    "minimum": 0.5,
                    "maximum": 3.0
                },
                "sl_distance_pct": {
                    "type": "number",
                    "description": "Stop loss distance as percentage from entry"
                }
            },
            "required": ["symbol", "sl_distance_pct"]
        }
    },
    {
        "name": "validate_trade_parameters",
        "description": "Validate trade parameters (TP/SL levels, position size, risk:reward ratio) before opening position. Returns validation result and any warnings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading symbol"
                },
                "direction": {
                    "type": "string",
                    "enum": ["LONG", "SHORT"],
                    "description": "Position direction"
                },
                "entry_price": {
                    "type": "number",
                    "description": "Expected entry price"
                },
                "tp_price": {
                    "type": "number",
                    "description": "Take profit price"
                },
                "sl_price": {
                    "type": "number",
                    "description": "Stop loss price"
                },
                "size_pct": {
                    "type": "number",
                    "description": "Position size as % of capital"
                }
            },
            "required": ["symbol", "direction", "entry_price", "tp_price", "sl_price", "size_pct"]
        }
    }
]


class AIToolsExecutor:
    """
    Executor for AI Trading Tools

    Handles execution of tool calls from Claude Sonnet 4.5
    """

    def __init__(self, exchange, decision_manager, config):
        """
        Initialize tools executor

        Args:
            exchange: Exchange client (BinanceClient or PaperExchange)
            decision_manager: DecisionManager instance
            config: AITradingConfig instance
        """
        self.exchange = exchange
        self.decision_manager = decision_manager
        self.config = config
        self.search_count = 0  # Track web searches for budget

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool and return result

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Tool execution result
        """
        try:
            if tool_name == "get_market_data":
                return self._get_market_data(**tool_input)
            elif tool_name == "get_technical_indicators":
                return self._get_technical_indicators(**tool_input)
            elif tool_name == "get_all_available_symbols":
                return self._get_all_available_symbols(**tool_input)
            elif tool_name == "calculate_symbol_metrics":
                return self._calculate_symbol_metrics(**tool_input)
            elif tool_name == "search_fundamental_info":
                return self._search_fundamental_info(**tool_input)
            elif tool_name == "get_historical_decisions":
                return self._get_historical_decisions(**tool_input)
            elif tool_name == "analyze_historical_performance":
                return self._analyze_historical_performance(**tool_input)
            elif tool_name == "open_position":
                return self._open_position(**tool_input)
            elif tool_name == "get_current_positions":
                return self._get_current_positions()
            elif tool_name == "get_portfolio_status":
                return self._get_portfolio_status()
            elif tool_name == "calculate_position_size":
                return self._calculate_position_size(**tool_input)
            elif tool_name == "validate_trade_parameters":
                return self._validate_trade_parameters(**tool_input)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": f"Tool execution error: {str(e)}"}

    def _get_market_data(self, symbols: List[str], timeframes: List[str],
                         limit: int = 100) -> Dict[str, Any]:
        """Get OHLCV data for symbols"""
        result = {}

        for symbol in symbols:
            result[symbol] = {}
            for timeframe in timeframes:
                try:
                    klines = self.exchange.get_klines(symbol, timeframe, limit=limit)

                    # Convert to structured format
                    result[symbol][timeframe] = {
                        "timestamps": [k[0] for k in klines],
                        "open": [float(k[1]) for k in klines],
                        "high": [float(k[2]) for k in klines],
                        "low": [float(k[3]) for k in klines],
                        "close": [float(k[4]) for k in klines],
                        "volume": [float(k[5]) for k in klines],
                        "count": len(klines)
                    }

                    # Add summary statistics
                    closes = result[symbol][timeframe]["close"]
                    result[symbol][timeframe]["current_price"] = closes[-1]
                    result[symbol][timeframe]["price_change_pct"] = (
                        (closes[-1] - closes[0]) / closes[0] * 100
                    )

                except Exception as e:
                    result[symbol][timeframe] = {"error": str(e)}

        return result

    def _get_technical_indicators(self, symbol: str, timeframe: str,
                                   indicators: List[str] = ["all"]) -> Dict[str, Any]:
        """Calculate technical indicators"""
        try:
            # Get market data
            klines = self.exchange.get_klines(symbol, timeframe, limit=200)

            # Convert to numpy arrays
            timestamps = np.array([k[0] for k in klines])
            opens = np.array([float(k[1]) for k in klines])
            highs = np.array([float(k[2]) for k in klines])
            lows = np.array([float(k[3]) for k in klines])
            closes = np.array([float(k[4]) for k in klines])
            volumes = np.array([float(k[5]) for k in klines])

            result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": float(closes[-1]),
                "indicators": {}
            }

            calc_all = "all" in indicators

            # RSI
            if calc_all or "rsi" in indicators:
                rsi_values = rsi(closes, period=14)
                result["indicators"]["rsi"] = {
                    "current": float(rsi_values[-1]),
                    "previous": float(rsi_values[-2]),
                    "overbought": rsi_values[-1] > 70,
                    "oversold": rsi_values[-1] < 30
                }

            # MACD
            if calc_all or "macd" in indicators:
                macd_line, signal_line, histogram = macd(closes)
                result["indicators"]["macd"] = {
                    "macd": float(macd_line[-1]),
                    "signal": float(signal_line[-1]),
                    "histogram": float(histogram[-1]),
                    "bullish_crossover": histogram[-1] > 0 and histogram[-2] <= 0,
                    "bearish_crossover": histogram[-1] < 0 and histogram[-2] >= 0
                }

            # Bollinger Bands
            if calc_all or "bollinger_bands" in indicators:
                upper, middle, lower = bollinger_bands(closes)
                bb_width = (upper[-1] - lower[-1]) / middle[-1] * 100
                result["indicators"]["bollinger_bands"] = {
                    "upper": float(upper[-1]),
                    "middle": float(middle[-1]),
                    "lower": float(lower[-1]),
                    "width_pct": float(bb_width),
                    "price_position": (closes[-1] - lower[-1]) / (upper[-1] - lower[-1])
                }

            # ATR
            if calc_all or "atr" in indicators:
                atr_values = atr(highs, lows, closes, period=14)
                atr_pct = atr_values[-1] / closes[-1] * 100
                result["indicators"]["atr"] = {
                    "value": float(atr_values[-1]),
                    "pct": float(atr_pct),
                    "volatility": "high" if atr_pct > 5 else "medium" if atr_pct > 2 else "low"
                }

            # Stochastic
            if calc_all or "stochastic" in indicators:
                k, d = stochastic(highs, lows, closes)
                result["indicators"]["stochastic"] = {
                    "k": float(k[-1]),
                    "d": float(d[-1]),
                    "overbought": k[-1] > 80,
                    "oversold": k[-1] < 20
                }

            # ADX
            if calc_all or "adx" in indicators:
                adx_values = adx(highs, lows, closes, period=14)
                result["indicators"]["adx"] = {
                    "value": float(adx_values[-1]),
                    "trend_strength": (
                        "strong" if adx_values[-1] > 25 else
                        "weak" if adx_values[-1] < 20 else "moderate"
                    )
                }

            # Williams %R
            if calc_all or "williams_r" in indicators:
                wr = williams_r(highs, lows, closes, period=14)
                result["indicators"]["williams_r"] = {
                    "value": float(wr[-1]),
                    "overbought": wr[-1] > -20,
                    "oversold": wr[-1] < -80
                }

            # CCI
            if calc_all or "cci" in indicators:
                cci_values = cci(highs, lows, closes, period=20)
                result["indicators"]["cci"] = {
                    "value": float(cci_values[-1]),
                    "overbought": cci_values[-1] > 100,
                    "oversold": cci_values[-1] < -100
                }

            # ROC
            if calc_all or "roc" in indicators:
                roc_values = roc(closes, period=12)
                result["indicators"]["roc"] = {
                    "value": float(roc_values[-1]),
                    "momentum": "positive" if roc_values[-1] > 0 else "negative"
                }

            return result

        except Exception as e:
            return {"error": f"Failed to calculate indicators: {str(e)}"}

    def _get_all_available_symbols(self, min_volume_usd: float = 10_000_000,
                                    exclude_stablecoins: bool = True,
                                    exclude_blacklist: bool = True) -> Dict[str, Any]:
        """Get filtered list of available symbols"""
        try:
            # Get all futures symbols from exchange
            exchange_info = self.exchange.futures_exchange_info()
            all_symbols = [
                s['symbol'] for s in exchange_info['symbols']
                if s['status'] == 'TRADING' and s['symbol'].endswith('USDT')
            ]

            filtered = []

            for symbol in all_symbols:
                # Exclude stablecoins
                if exclude_stablecoins:
                    base = symbol.replace('USDT', '')
                    if base in self.config.STABLECOIN_SYMBOLS:
                        continue

                # Exclude blacklist
                if exclude_blacklist:
                    base = symbol.replace('USDT', '')
                    if base in self.config.BLACKLIST_SYMBOLS:
                        continue

                # Check volume
                try:
                    ticker = self.exchange.get_ticker(symbol)
                    volume_24h = float(ticker.get('quoteVolume', 0))

                    if volume_24h >= min_volume_usd:
                        filtered.append({
                            "symbol": symbol,
                            "volume_24h_usd": volume_24h,
                            "price": float(ticker.get('lastPrice', 0)),
                            "price_change_pct": float(ticker.get('priceChangePercent', 0))
                        })
                except:
                    continue

            # Sort by volume
            filtered.sort(key=lambda x: x['volume_24h_usd'], reverse=True)

            return {
                "total_symbols": len(filtered),
                "symbols": filtered,
                "filters_applied": {
                    "min_volume_usd": min_volume_usd,
                    "exclude_stablecoins": exclude_stablecoins,
                    "exclude_blacklist": exclude_blacklist
                }
            }

        except Exception as e:
            return {"error": f"Failed to get symbols: {str(e)}"}

    def _calculate_symbol_metrics(self, symbol: str, timeframe: str = "4h") -> Dict[str, Any]:
        """Calculate detailed metrics for a symbol"""
        try:
            # Get ticker
            ticker = self.exchange.get_ticker(symbol)

            # Get klines for volatility
            klines = self.exchange.get_klines(symbol, timeframe, limit=100)
            highs = np.array([float(k[2]) for k in klines])
            lows = np.array([float(k[3]) for k in klines])
            closes = np.array([float(k[4]) for k in klines])
            volumes = np.array([float(k[5]) for k in klines])

            # Calculate ATR for volatility
            atr_values = atr(highs, lows, closes, period=14)
            atr_pct = atr_values[-1] / closes[-1] * 100

            # Calculate market phase (simple MACD + RSI based)
            macd_line, signal_line, _ = macd(closes)
            rsi_values = rsi(closes, period=14)

            if macd_line[-1] > signal_line[-1] and rsi_values[-1] > 50:
                phase = "bullish"
            elif macd_line[-1] < signal_line[-1] and rsi_values[-1] < 50:
                phase = "bearish"
            else:
                phase = "neutral"

            return {
                "symbol": symbol,
                "current_price": float(ticker['lastPrice']),
                "volume_24h_usd": float(ticker['quoteVolume']),
                "price_change_24h_pct": float(ticker['priceChangePercent']),
                "volatility": {
                    "atr": float(atr_values[-1]),
                    "atr_pct": float(atr_pct),
                    "classification": (
                        "high" if atr_pct > 5 else
                        "medium" if atr_pct > 2 else "low"
                    )
                },
                "liquidity_score": min(float(ticker['quoteVolume']) / 10_000_000, 10.0),
                "market_phase": phase,
                "technical_summary": {
                    "rsi": float(rsi_values[-1]),
                    "macd_signal": "bullish" if macd_line[-1] > signal_line[-1] else "bearish"
                }
            }

        except Exception as e:
            return {"error": f"Failed to calculate metrics: {str(e)}"}

    def _search_fundamental_info(self, query: str, coin_name: str = "") -> Dict[str, Any]:
        """
        Search for fundamental information using crypto news APIs

        Sources:
        - CryptoCompare News API (free, no key required)
        - CoinGecko trending coins
        - Sentiment analysis from keywords
        """
        # Budget check
        if self.search_count >= self.config.MAX_SEARCH_QUERIES_PER_DAY:
            return {
                "error": "Daily search quota exceeded",
                "searches_used": self.search_count,
                "max_searches": self.config.MAX_SEARCH_QUERIES_PER_DAY
            }

        self.search_count += 1

        results = {
            "query": query,
            "coin_name": coin_name,
            "searches_used": self.search_count,
            "news": [],
            "sentiment": "neutral",
            "trending": False,
            "sources_checked": []
        }

        try:
            # 1. Get latest crypto news from CryptoCompare (free API)
            news_data = self._fetch_cryptocompare_news(coin_name or query)
            if news_data:
                results["news"] = news_data["articles"][:5]  # Top 5 news
                results["sources_checked"].append("CryptoCompare")

            # 2. Check if coin is trending on CoinGecko
            trending_data = self._fetch_coingecko_trending(coin_name)
            if trending_data:
                results["trending"] = trending_data["is_trending"]
                results["trending_rank"] = trending_data.get("rank")
                results["sources_checked"].append("CoinGecko")

            # 3. Analyze sentiment from news headlines
            if results["news"]:
                results["sentiment"] = self._analyze_sentiment(results["news"])
                results["sentiment_score"] = self._calculate_sentiment_score(results["news"])

            # 4. Summary
            results["summary"] = self._generate_fundamental_summary(results)

        except Exception as e:
            results["error"] = f"Search failed: {str(e)}"
            results["note"] = "Partial results may be available"

        return results

    def _fetch_cryptocompare_news(self, coin_name: str, limit: int = 10) -> Optional[Dict]:
        """
        Fetch latest news from CryptoCompare API (free, no key required)
        """
        try:
            # CryptoCompare news endpoint (public, no auth)
            url = "https://min-api.cryptocompare.com/data/v2/news/"
            params = {
                "lang": "EN",
                "sortOrder": "latest"
            }

            # Add coin filter if provided
            if coin_name:
                # Try to get coin symbol (BTC, ETH, etc)
                coin_symbol = self._extract_coin_symbol(coin_name)
                if coin_symbol:
                    params["categories"] = coin_symbol

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get("Response") == "Success" and data.get("Data"):
                articles = []
                for item in data["Data"][:limit]:
                    articles.append({
                        "title": item.get("title", ""),
                        "source": item.get("source", ""),
                        "published": datetime.fromtimestamp(item.get("published_on", 0)).isoformat(),
                        "url": item.get("url", ""),
                        "categories": item.get("categories", ""),
                        "body_preview": item.get("body", "")[:200] + "..." if item.get("body") else ""
                    })

                return {"articles": articles, "total": len(articles)}

        except Exception as e:
            print(f"[Warning] CryptoCompare news fetch failed: {e}")
            return None

    def _fetch_coingecko_trending(self, coin_name: str) -> Optional[Dict]:
        """
        Check if coin is trending on CoinGecko (free API)
        """
        try:
            url = "https://api.coingecko.com/api/v3/search/trending"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            if "coins" in data:
                coin_symbol = self._extract_coin_symbol(coin_name).lower()

                for idx, coin in enumerate(data["coins"]):
                    coin_item = coin.get("item", {})
                    symbol = coin_item.get("symbol", "").lower()
                    name = coin_item.get("name", "").lower()

                    if coin_symbol in symbol or coin_symbol in name:
                        return {
                            "is_trending": True,
                            "rank": idx + 1,
                            "name": coin_item.get("name"),
                            "symbol": coin_item.get("symbol"),
                            "market_cap_rank": coin_item.get("market_cap_rank")
                        }

            return {"is_trending": False}

        except Exception as e:
            print(f"[Warning] CoinGecko trending fetch failed: {e}")
            return None

    def _extract_coin_symbol(self, coin_name: str) -> str:
        """
        Extract coin symbol from name or trading pair
        Examples: BTCUSDT -> BTC, Bitcoin -> BTC, ETH -> ETH
        """
        coin_name = coin_name.upper()

        # Common mappings
        mappings = {
            "BITCOIN": "BTC",
            "ETHEREUM": "ETH",
            "BINANCE": "BNB",
            "CARDANO": "ADA",
            "SOLANA": "SOL",
            "RIPPLE": "XRP",
            "POLKADOT": "DOT",
            "DOGECOIN": "DOGE",
            "AVALANCHE": "AVAX",
            "POLYGON": "MATIC",
            "CHAINLINK": "LINK"
        }

        if coin_name in mappings:
            return mappings[coin_name]

        # Remove USDT, BUSD, etc
        for suffix in ["USDT", "BUSD", "USDC", "USD", "BTC", "ETH"]:
            if coin_name.endswith(suffix):
                return coin_name[:-len(suffix)]

        # If short (2-5 chars), probably already a symbol
        if 2 <= len(coin_name) <= 5:
            return coin_name

        return coin_name[:4]  # Take first 4 chars as fallback

    def _analyze_sentiment(self, articles: List[Dict]) -> str:
        """
        Analyze sentiment from news articles using keyword analysis
        """
        positive_keywords = [
            "bullish", "surge", "rally", "gain", "profit", "breakthrough",
            "partnership", "adoption", "upgrade", "positive", "growth",
            "milestone", "success", "win", "launch", "innovation"
        ]

        negative_keywords = [
            "bearish", "crash", "drop", "loss", "scam", "hack", "ban",
            "regulation", "lawsuit", "decline", "fail", "warning", "risk",
            "concern", "investigation", "fraud", "dump"
        ]

        positive_count = 0
        negative_count = 0

        for article in articles:
            title = article.get("title", "").lower()
            body = article.get("body_preview", "").lower()
            text = f"{title} {body}"

            for keyword in positive_keywords:
                if keyword in text:
                    positive_count += 1

            for keyword in negative_keywords:
                if keyword in text:
                    negative_count += 1

        if positive_count > negative_count * 1.5:
            return "positive"
        elif negative_count > positive_count * 1.5:
            return "negative"
        else:
            return "neutral"

    def _calculate_sentiment_score(self, articles: List[Dict]) -> float:
        """
        Calculate sentiment score from -1.0 (very negative) to 1.0 (very positive)
        """
        positive_keywords = [
            "bullish", "surge", "rally", "gain", "profit", "breakthrough",
            "partnership", "adoption", "upgrade", "positive", "growth"
        ]

        negative_keywords = [
            "bearish", "crash", "drop", "loss", "scam", "hack", "ban",
            "regulation", "lawsuit", "decline", "fail", "warning"
        ]

        total_score = 0
        count = 0

        for article in articles:
            title = article.get("title", "").lower()
            body = article.get("body_preview", "").lower()
            text = f"{title} {body}"

            score = 0
            for keyword in positive_keywords:
                if keyword in text:
                    score += 1

            for keyword in negative_keywords:
                if keyword in text:
                    score -= 1

            total_score += score
            count += 1

        if count == 0:
            return 0.0

        # Normalize to -1.0 to 1.0
        avg_score = total_score / count
        return max(-1.0, min(1.0, avg_score / 3))

    def _generate_fundamental_summary(self, results: Dict) -> str:
        """
        Generate human-readable summary of fundamental analysis
        """
        parts = []

        # News summary
        if results.get("news"):
            news_count = len(results["news"])
            parts.append(f"Found {news_count} recent news articles")

        # Sentiment
        sentiment = results.get("sentiment", "neutral")
        sentiment_score = results.get("sentiment_score", 0)

        if sentiment == "positive":
            parts.append(f"Overall sentiment: POSITIVE (score: {sentiment_score:.2f})")
        elif sentiment == "negative":
            parts.append(f"Overall sentiment: NEGATIVE (score: {sentiment_score:.2f})")
        else:
            parts.append(f"Overall sentiment: NEUTRAL (score: {sentiment_score:.2f})")

        # Trending
        if results.get("trending"):
            rank = results.get("trending_rank", "")
            parts.append(f"Currently TRENDING on CoinGecko (rank #{rank})")

        # Sources
        sources = results.get("sources_checked", [])
        if sources:
            parts.append(f"Sources: {', '.join(sources)}")

        return ". ".join(parts) if parts else "No significant fundamental data found"

    def _get_historical_decisions(self, symbol: Optional[str] = None,
                                   outcome: str = "all", limit: int = 20,
                                   days_back: int = 90) -> Dict[str, Any]:
        """Get historical decisions"""
        decisions = self.decision_manager.get_historical_decisions(
            symbol=symbol,
            outcome=outcome,
            limit=limit,
            days_back=days_back
        )

        return {
            "count": len(decisions),
            "decisions": decisions,
            "filters": {
                "symbol": symbol,
                "outcome": outcome,
                "days_back": days_back
            }
        }

    def _analyze_historical_performance(self, groupby: str = "all",
                                        days_back: int = 90) -> Dict[str, Any]:
        """Analyze historical performance"""
        return self.decision_manager.analyze_performance(
            groupby=groupby,
            days_back=days_back
        )

    def _open_position(self, symbol: str, direction: str, size_pct: float,
                       tp_price: float, sl_price: float, reasoning: str,
                       confidence: float = 0.7) -> Dict[str, Any]:
        """Open a new position (handled by AITrader)"""
        # This will be called from AITrader, not directly
        return {
            "status": "queued",
            "symbol": symbol,
            "direction": direction,
            "size_pct": size_pct,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "reasoning": reasoning,
            "confidence": confidence
        }

    def _get_current_positions(self) -> Dict[str, Any]:
        """Get current open positions"""
        return self.decision_manager.get_open_positions()

    def _get_portfolio_status(self) -> Dict[str, Any]:
        """Get portfolio status"""
        return self.decision_manager.get_portfolio_status()

    def _calculate_position_size(self, symbol: str, sl_distance_pct: float,
                                  direction: str = "LONG",
                                  risk_pct: float = 1.5) -> Dict[str, Any]:
        """Calculate optimal position size"""
        try:
            # Get current capital
            portfolio = self.decision_manager.get_portfolio_status()
            available_capital = portfolio['available_capital']

            # Get current price
            ticker = self.exchange.get_ticker(symbol)
            current_price = float(ticker['lastPrice'])

            # Calculate position size based on risk
            risk_amount = available_capital * (risk_pct / 100)
            position_size_usdt = risk_amount / (sl_distance_pct / 100)

            # Apply limits
            max_size = available_capital * (self.config.MAX_POSITION_SIZE_PCT / 100)
            min_size = available_capital * (self.config.MIN_POSITION_SIZE_PCT / 100)

            position_size_usdt = min(max(position_size_usdt, min_size), max_size)
            size_pct = (position_size_usdt / available_capital) * 100

            return {
                "recommended_size_usdt": round(position_size_usdt, 2),
                "recommended_size_pct": round(size_pct, 2),
                "risk_amount_usdt": round(risk_amount, 2),
                "risk_pct": risk_pct,
                "sl_distance_pct": sl_distance_pct,
                "available_capital": round(available_capital, 2),
                "current_price": current_price
            }

        except Exception as e:
            return {"error": f"Failed to calculate position size: {str(e)}"}

    def _validate_trade_parameters(self, symbol: str, direction: str,
                                    entry_price: float, tp_price: float,
                                    sl_price: float, size_pct: float) -> Dict[str, Any]:
        """Validate trade parameters"""
        errors = []
        warnings = []

        # Calculate distances
        if direction == "LONG":
            tp_distance_pct = (tp_price - entry_price) / entry_price * 100
            sl_distance_pct = (entry_price - sl_price) / entry_price * 100

            if tp_price <= entry_price:
                errors.append("TP must be above entry price for LONG")
            if sl_price >= entry_price:
                errors.append("SL must be below entry price for LONG")
        else:  # SHORT
            tp_distance_pct = (entry_price - tp_price) / entry_price * 100
            sl_distance_pct = (sl_price - entry_price) / entry_price * 100

            if tp_price >= entry_price:
                errors.append("TP must be below entry price for SHORT")
            if sl_price <= entry_price:
                errors.append("SL must be above entry price for SHORT")

        # Validate TP distance
        if tp_distance_pct < self.config.MIN_TP_DISTANCE_PCT:
            errors.append(f"TP too close: {tp_distance_pct:.2f}% < {self.config.MIN_TP_DISTANCE_PCT}%")
        if tp_distance_pct > self.config.MAX_TP_DISTANCE_PCT:
            warnings.append(f"TP very far: {tp_distance_pct:.2f}% > {self.config.MAX_TP_DISTANCE_PCT}%")

        # Validate SL distance
        if sl_distance_pct < self.config.MIN_SL_DISTANCE_PCT:
            errors.append(f"SL too close: {sl_distance_pct:.2f}% < {self.config.MIN_SL_DISTANCE_PCT}%")
        if sl_distance_pct > self.config.MAX_SL_DISTANCE_PCT:
            warnings.append(f"SL very far: {sl_distance_pct:.2f}% > {self.config.MAX_SL_DISTANCE_PCT}%")

        # Calculate risk:reward ratio
        rr_ratio = tp_distance_pct / sl_distance_pct if sl_distance_pct > 0 else 0

        if rr_ratio < self.config.MIN_RISK_REWARD_RATIO:
            errors.append(f"Risk:Reward too low: {rr_ratio:.2f} < {self.config.MIN_RISK_REWARD_RATIO}")

        # Validate position size
        if size_pct < self.config.MIN_POSITION_SIZE_PCT:
            errors.append(f"Position size too small: {size_pct:.2f}% < {self.config.MIN_POSITION_SIZE_PCT}%")
        if size_pct > self.config.MAX_POSITION_SIZE_PCT:
            errors.append(f"Position size too large: {size_pct:.2f}% > {self.config.MAX_POSITION_SIZE_PCT}%")

        # Check portfolio exposure
        portfolio = self.decision_manager.get_portfolio_status()
        new_exposure = portfolio['exposure_pct'] + size_pct

        if new_exposure > self.config.MAX_TOTAL_EXPOSURE_PCT:
            errors.append(
                f"Total exposure would exceed limit: {new_exposure:.1f}% > "
                f"{self.config.MAX_TOTAL_EXPOSURE_PCT}%"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "metrics": {
                "tp_distance_pct": round(tp_distance_pct, 2),
                "sl_distance_pct": round(sl_distance_pct, 2),
                "risk_reward_ratio": round(rr_ratio, 2),
                "position_size_pct": round(size_pct, 2),
                "new_total_exposure_pct": round(new_exposure, 2)
            }
        }
