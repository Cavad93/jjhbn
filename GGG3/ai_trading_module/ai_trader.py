"""
AI Trader - Autonomous Trading Agent using Claude Sonnet 4.5

Основной класс для AI-управляемой торговли
"""
import sys
import os
import io
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import anthropic

# Setup logger
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_trading_module.config import AITradingConfig
from ai_trading_module.decision_manager import DecisionManager
from ai_trading_module.ai_tools import AIToolsExecutor, TOOLS


class AITrader:
    """
    AI Trading Agent using Claude Sonnet 4.5

    Autonomous trading system that:
    - Selects coins based on AI analysis
    - Makes LONG/SHORT decisions
    - Manages positions with TP/SL
    - Learns from historical performance
    - Reports to Telegram
    """

    def __init__(self, exchange, config: AITradingConfig = None,
                 telegram_notifier=None):
        """
        Initialize AI Trader

        Args:
            exchange: Exchange client (PaperExchange or BinanceClient)
            config: Configuration (uses default if None)
            telegram_notifier: TelegramNotifier instance (optional)
        """
        self.config = config or AITradingConfig()
        self.exchange = exchange
        self.telegram = telegram_notifier

        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Configuration errors: {errors}")

        # Initialize Anthropic client
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(api_key=api_key)

        # Initialize managers
        self.decision_manager = DecisionManager(self.config, self.exchange)
        self.tools_executor = AIToolsExecutor(
            self.exchange,
            self.decision_manager,
            self.config
        )

        # State
        self.last_decision_time = None
        self.last_position_check_time = None
        self.conversation_history = []
        self.daily_api_cost = 0.0

        # Statistics
        self.stats = {
            "decisions_made": 0,
            "positions_opened": 0,
            "positions_closed": 0,
            "total_api_calls": 0,
            "total_api_cost_usd": 0.0
        }

        # Watchdog для предотвращения зависания
        self._last_activity = time.time()
        self._watchdog_enabled = True
        self._activity_lock = threading.Lock()
        self._watchdog_thread = threading.Thread(target=self._watchdog_monitor, daemon=True)
        self._watchdog_thread.start()

        print(f"[AI Trader] Initialized with ${self.config.INITIAL_CAPITAL} capital")
        print(f"[AI Trader] Model: {self.config.AI_MODEL}")
        print(f"[AI Trader] Paper Trading: {self.config.PAPER_TRADING}")
        print(f"[AI Trader] Watchdog: Enabled")

    def run(self):
        """
        Main loop for AI Trader

        Runs continuously:
        - Daily decision cycle (coin selection + position opening)
        - Hourly position checks (TP/SL monitoring)
        """
        print(f"\n{'='*80}")
        print(f"AI TRADER STARTED")
        print(f"{'='*80}\n")

        if self.telegram:
            self.telegram.send_message(
                f"🤖 AI Trading Module Started\n\n"
                f"Capital: ${self.config.INITIAL_CAPITAL}\n"
                f"Mode: {'Paper Trading' if self.config.PAPER_TRADING else 'Live Trading'}\n"
                f"Model: {self.config.AI_MODEL}"
            )

        try:
            while True:
                current_time = datetime.now()
                self._update_activity()  # Watchdog: update activity

                # Check if it's time for daily decision cycle
                if self._should_make_decisions(current_time):
                    print(f"\n[{current_time}] Starting daily decision cycle...")
                    self._daily_decision_cycle()
                    self.last_decision_time = current_time
                    self._update_activity()  # Watchdog: after decisions
                else:
                    # Show that bot is alive but waiting
                    if self.last_decision_time:
                        time_until_next = self._time_until_next_decision(current_time)
                        print(f"[{current_time.strftime('%H:%M:%S')}] Waiting for next decision cycle (in {time_until_next})")

                # Check positions periodically
                if self._should_check_positions(current_time):
                    print(f"\n[{current_time}] Checking positions...")
                    self._check_and_close_positions()
                    self.last_position_check_time = current_time
                    self._update_activity()  # Watchdog: after position check

                # Sleep for a minute
                time.sleep(60)

        except KeyboardInterrupt:
            print("\n\nAI Trader stopped by user")
            self._watchdog_enabled = False  # Stop watchdog
            self._send_shutdown_report()
        except Exception as e:
            print(f"\n\nCritical error: {e}")
            self._watchdog_enabled = False  # Stop watchdog
            if self.telegram:
                self.telegram.send_message(f"🚨 AI Trader Error: {e}")
            raise

    def _should_make_decisions(self, current_time: datetime) -> bool:
        """Check if it's time to make daily decisions"""
        if self.last_decision_time is None:
            return True

        # Check if decision hour has passed
        if current_time.hour == self.config.DECISION_TIME_UTC:
            # Check if we haven't made decisions today
            time_since_last = current_time - self.last_decision_time
            return time_since_last.total_seconds() >= (self.config.DECISION_INTERVAL_HOURS * 3600)

        return False

    def _time_until_next_decision(self, current_time: datetime) -> str:
        """Calculate time until next decision cycle"""
        if self.last_decision_time is None:
            return "now"

        next_decision_time = self.last_decision_time + timedelta(hours=self.config.DECISION_INTERVAL_HOURS)
        time_diff = next_decision_time - current_time

        if time_diff.total_seconds() <= 0:
            return "now"

        hours = int(time_diff.total_seconds() // 3600)
        minutes = int((time_diff.total_seconds() % 3600) // 60)

        return f"{hours}h {minutes}m"

    def _should_check_positions(self, current_time: datetime) -> bool:
        """Check if it's time to check positions"""
        if not self.decision_manager.open_positions:
            return False

        if self.last_position_check_time is None:
            return True

        time_since_last = current_time - self.last_position_check_time
        return time_since_last.total_seconds() >= (self.config.CHECK_POSITIONS_INTERVAL_MINUTES * 60)

    def _daily_decision_cycle(self):
        """
        Daily decision cycle:
        1. Analyze current portfolio
        2. Select top-20 coins for analysis
        3. Make decisions (open/skip positions)
        4. Generate report
        """
        print("\n" + "="*80)
        print("DAILY DECISION CYCLE")
        print("="*80 + "\n")

        # Check budget
        if self.daily_api_cost >= self.config.MAX_DAILY_API_COST_USD:
            print(f"[WARNING] Daily API budget exhausted: ${self.daily_api_cost:.2f}")
            return

        # Check emergency conditions
        portfolio = self.decision_manager.get_portfolio_status()
        if self._check_emergency_conditions(portfolio):
            print("[WARNING] Emergency conditions detected - skipping new positions")
            return

        try:
            # Build context prompt
            context = self._build_decision_context()

            # Ask AI to make decisions
            print("[AI] Analyzing market and making decisions...")
            start_time = time.time()

            response = self._call_ai_with_tools(
                user_message=self._get_decision_prompt(context),
                system_prompt=self.config.get_system_prompt()
            )

            duration = time.time() - start_time
            print(f"[AI] Decision complete in {duration:.1f}s")

            # Track API cost (rough estimate)
            estimated_cost = self.config.COST_PER_DECISION
            self.daily_api_cost += estimated_cost
            self.stats['total_api_cost_usd'] += estimated_cost

            # Process AI response
            self._process_ai_decision_response(response)

            # Send report to Telegram
            self._send_daily_report(response)

            self.stats['decisions_made'] += 1

        except Exception as e:
            print(f"[ERROR] Decision cycle failed: {e}")
            if self.telegram:
                self.telegram.send_message(f"🚨 AI Decision Error: {e}")

    def _build_decision_context(self) -> Dict[str, Any]:
        """Build context for AI decision making"""
        portfolio = self.decision_manager.get_portfolio_status()
        open_positions = self.decision_manager.get_open_positions()

        # Get recent performance
        recent_performance = self.decision_manager.analyze_performance(
            groupby="all",
            days_back=7
        )

        return {
            "portfolio": portfolio,
            "open_positions": open_positions,
            "recent_performance": recent_performance,
            "timestamp": datetime.now().isoformat()
        }

    def _get_decision_prompt(self, context: Dict) -> str:
        """Generate decision prompt for AI"""
        portfolio = context['portfolio']
        open_positions = context['open_positions']
        recent_perf = context['recent_performance']

        prompt = f"""# DAILY TRADING DECISION

## Current Portfolio Status
- **Total Equity**: ${portfolio['total_equity']:.2f}
- **Available Capital**: ${portfolio['available_capital']:.2f}
- **Open Positions**: {portfolio['num_open_positions']}/{self.config.MAX_POSITIONS}
- **Current Exposure**: {portfolio['exposure_pct']:.1f}%
- **Total Return**: {portfolio['total_return_pct']:.2f}%

## Recent Performance (Last 7 Days)
"""

        if recent_perf.get('total_trades', 0) > 0:
            prompt += f"""- **Trades**: {recent_perf['total_trades']} (Win Rate: {recent_perf['win_rate_pct']:.1f}%)
- **Total PnL**: ${recent_perf['total_pnl_usdt']:.2f}
- **Avg Win**: ${recent_perf['average_win_usdt']:.2f} / **Avg Loss**: ${recent_perf['average_loss_usdt']:.2f}
- **Profit Factor**: {recent_perf['profit_factor']:.2f}
"""
        else:
            prompt += "- No trades yet\n"

        prompt += f"""
## Open Positions ({open_positions['count']})
"""
        for pos in open_positions['positions']:
            prompt += f"""
- **{pos['symbol']}** {pos['direction']}
  - Entry: ${pos['entry_price']:.2f} | Current: ${pos['current_price']:.2f}
  - Unrealized PnL: ${pos['unrealized_pnl_usdt']:.2f} ({pos['unrealized_pnl_pct']:.2f}%)
  - To TP: {pos['distance_to_tp_pct']:.2f}% | To SL: {pos['distance_to_sl_pct']:.2f}%
"""

        # Calculate target positions to open
        positions_needed = max(
            self.config.MIN_POSITIONS - open_positions['count'],
            self.config.MAX_POSITIONS - open_positions['count']
        )

        prompt += f"""
## Your Task

You need to analyze the market and decide on new positions to open.

**Positions to consider**: {positions_needed} new positions
**Available capital**: ${portfolio['available_capital']:.2f}

### Process:
1. **Screen the market** (use `get_all_available_symbols`):
   - Get list of available trading pairs
   - Filter by volume, volatility

2. **Select top-{self.config.TOP_COINS_TO_ANALYZE} candidates**:
   - Use `calculate_symbol_metrics` for each candidate
   - Rank based on YOUR criteria (opportunity, risk, momentum, etc.)
   - Explain your selection criteria

3. **Deep analysis** for top candidates:
   - Get technical indicators (`get_technical_indicators`)
   - Analyze market data (`get_market_data`)
   - Review historical decisions for these symbols (`get_historical_decisions`)
   - Consider fundamental factors if relevant

4. **Make trading decisions**:
   - For each candidate: LONG, SHORT, or SKIP
   - Define TP/SL levels using `validate_trade_parameters`
   - Calculate position size using `calculate_position_size`
   - Open positions using `open_position` (provide detailed reasoning)

5. **Portfolio management**:
   - Ensure diversification (avoid correlated positions)
   - Manage total exposure (≤{self.config.MAX_TOTAL_EXPOSURE_PCT}%)
   - Target {self.config.MIN_POSITIONS}-{self.config.MAX_POSITIONS} total positions

### Important Guidelines:
- **Be selective**: Only open positions with strong conviction
- **Learn from history**: Review past decisions on similar symbols
- **Manage risk**: Proper position sizing and TP/SL levels
- **Diversify**: Avoid concentration in correlated assets
- **Provide reasoning**: Explain EVERY decision clearly

Now proceed with your analysis and decisions.
"""

        return prompt

    def _call_ai_with_tools(self, user_message: str, system_prompt: str,
                           max_iterations: int = 10) -> str:
        """
        Call AI with tool support (agentic loop)

        Args:
            user_message: User prompt
            system_prompt: System prompt
            max_iterations: Max tool use iterations

        Returns:
            Final AI response text
        """
        messages = [{"role": "user", "content": user_message}]

        for iteration in range(max_iterations):
            print(f"[AI] Iteration {iteration + 1}/{max_iterations}")

            # Call Claude API
            response = self.client.messages.create(
                model=self.config.AI_MODEL,
                max_tokens=self.config.AI_MAX_TOKENS,
                temperature=self.config.AI_TEMPERATURE,
                system=system_prompt,
                tools=TOOLS,
                messages=messages
            )

            self.stats['total_api_calls'] += 1

            # Check stop reason
            if response.stop_reason == "end_turn":
                # Extract text response
                text_responses = [
                    block.text for block in response.content
                    if hasattr(block, 'text')
                ]
                return "\n".join(text_responses)

            elif response.stop_reason == "tool_use":
                # Process tool calls
                assistant_message = {"role": "assistant", "content": response.content}
                messages.append(assistant_message)

                # Execute tools
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  [Tool] {block.name}({json.dumps(block.input, indent=2)[:100]}...)")

                        # Execute tool
                        result = self.tools_executor.execute_tool(block.name, block.input)

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result)
                        })

                        # Special handling for open_position
                        if block.name == "open_position":
                            self._execute_position_opening(block.input, result)

                # Add tool results to conversation
                messages.append({"role": "user", "content": tool_results})

            else:
                # Unexpected stop reason
                print(f"[WARNING] Unexpected stop_reason: {response.stop_reason}")
                break

        # Max iterations reached
        print(f"[WARNING] Max iterations ({max_iterations}) reached")
        return "Decision process incomplete - max iterations reached"

    def _execute_position_opening(self, position_params: Dict, validation_result: Dict):
        """
        Actually execute position opening

        Args:
            position_params: Parameters from AI
            validation_result: Result from open_position tool
        """
        try:
            symbol = position_params['symbol']
            direction = position_params['direction']
            size_pct = position_params['size_pct']
            ai_tp_price = position_params['tp_price']
            ai_sl_price = position_params['sl_price']
            reasoning = position_params['reasoning']
            confidence = position_params.get('confidence', 0.7)

            # Get current price
            ticker = self.exchange.get_ticker(symbol)
            entry_price = float(ticker['lastPrice'])

            # Get market data for record
            market_data = self._get_market_snapshot(symbol)

            # CRITICAL FIX: Recalculate TP/SL based on REAL entry price using ATR
            # AI may have wrong price expectations, so we use ATR-based calculation
            tp_price, sl_price = self._recalculate_tp_sl_from_atr(
                symbol=symbol,
                entry_price=entry_price,
                direction=direction,
                market_data=market_data,
                ai_tp_price=ai_tp_price,
                ai_sl_price=ai_sl_price
            )

            print(f"[TP/SL Adjustment] AI suggested TP: ${ai_tp_price:.2f}, SL: ${ai_sl_price:.2f}")
            print(f"[TP/SL Adjustment] Recalculated TP: ${tp_price:.2f}, SL: ${sl_price:.2f} (based on entry: ${entry_price:.2f})")

            # Record decision
            decision_id = self.decision_manager.record_decision(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                tp_price=tp_price,  # Use recalculated TP
                sl_price=sl_price,  # Use recalculated SL
                size_pct=size_pct,
                reasoning=reasoning,
                confidence=confidence,
                market_data=market_data,
                fundamental_factors=[],  # TODO: Extract from reasoning
                technical_factors=market_data.get('indicators', {})
            )

            # Execute on exchange (paper or live)
            self._execute_on_exchange(
                symbol=symbol,
                direction=direction,
                size_pct=size_pct,
                entry_price=entry_price,
                tp_price=tp_price,  # Use recalculated TP
                sl_price=sl_price   # Use recalculated SL
            )

            self.stats['positions_opened'] += 1

            print(f"[SUCCESS] Opened {direction} position on {symbol}")
            print(f"  Entry: ${entry_price:.2f} | TP: ${tp_price:.2f} | SL: ${sl_price:.2f}")
            print(f"  Size: {size_pct:.1f}% | Confidence: {confidence:.2f}")

        except Exception as e:
            print(f"[ERROR] Failed to open position: {e}")

    def _recalculate_tp_sl_from_atr(self, symbol: str, entry_price: float,
                                    direction: str, market_data: Dict,
                                    ai_tp_price: float, ai_sl_price: float) -> tuple:
        """
        Recalculate TP/SL based on actual entry price and ATR

        AI may suggest TP/SL based on wrong price expectations.
        This method recalculates using ATR-based approach relative to actual entry.

        Returns:
            (tp_price, sl_price) tuple
        """
        try:
            # Try to get ATR from market data
            indicators = market_data.get('indicators', {})
            atr = indicators.get('atr', {}).get('current', None)

            if atr is None or atr <= 0:
                # Fallback: calculate ATR from AI's intended distances
                print(f"[Warning] No ATR available for {symbol}, using AI's intended distances")
                if direction == "LONG":
                    tp_distance_pct = (ai_tp_price - entry_price) / entry_price * 100
                    sl_distance_pct = (entry_price - ai_sl_price) / entry_price * 100
                else:
                    tp_distance_pct = (entry_price - ai_tp_price) / entry_price * 100
                    sl_distance_pct = (ai_sl_price - entry_price) / entry_price * 100

                # Apply to actual entry price
                if direction == "LONG":
                    tp_price = entry_price * (1 + tp_distance_pct / 100)
                    sl_price = entry_price * (1 - sl_distance_pct / 100)
                else:
                    tp_price = entry_price * (1 - tp_distance_pct / 100)
                    sl_price = entry_price * (1 + sl_distance_pct / 100)

            else:
                # Use ATR-based calculation (same as main bot)
                # TP = 2.5x ATR, SL = 1.5x ATR
                tp_multiplier = 2.5
                sl_multiplier = 1.5

                if direction == "LONG":
                    tp_price = entry_price + (atr * tp_multiplier)
                    sl_price = entry_price - (atr * sl_multiplier)
                else:  # SHORT
                    tp_price = entry_price - (atr * tp_multiplier)
                    sl_price = entry_price + (atr * sl_multiplier)

                print(f"[ATR-based TP/SL] ATR: ${atr:.2f}, TP: {tp_multiplier}x ATR, SL: {sl_multiplier}x ATR")

            # Validate bounds
            if direction == "LONG":
                tp_distance_pct = (tp_price - entry_price) / entry_price * 100
                sl_distance_pct = (entry_price - sl_price) / entry_price * 100

                # Ensure TP is above entry and SL is below
                if tp_price <= entry_price:
                    tp_price = entry_price * 1.02  # Min 2% TP
                if sl_price >= entry_price:
                    sl_price = entry_price * 0.98  # Min 2% SL
            else:  # SHORT
                tp_distance_pct = (entry_price - tp_price) / entry_price * 100
                sl_distance_pct = (sl_price - entry_price) / entry_price * 100

                # Ensure TP is below entry and SL is above
                if tp_price >= entry_price:
                    tp_price = entry_price * 0.98  # Min 2% TP
                if sl_price <= entry_price:
                    sl_price = entry_price * 1.02  # Min 2% SL

            # Ensure minimum risk:reward ratio
            rr_ratio = tp_distance_pct / sl_distance_pct if sl_distance_pct > 0 else 0
            if rr_ratio < self.config.MIN_RISK_REWARD_RATIO:
                # Adjust TP to meet minimum RR
                tp_distance_pct = sl_distance_pct * self.config.MIN_RISK_REWARD_RATIO
                if direction == "LONG":
                    tp_price = entry_price * (1 + tp_distance_pct / 100)
                else:
                    tp_price = entry_price * (1 - tp_distance_pct / 100)

                print(f"[TP/SL Adjustment] Adjusted TP to meet min R:R {self.config.MIN_RISK_REWARD_RATIO}")

            return (tp_price, sl_price)

        except Exception as e:
            print(f"[ERROR] Failed to recalculate TP/SL: {e}")
            # Fallback to AI's values
            return (ai_tp_price, ai_sl_price)

    def _execute_on_exchange(self, symbol: str, direction: str, size_pct: float,
                            entry_price: float, tp_price: float, sl_price: float):
        """
        Execute trade on exchange

        For paper trading: Just record
        For live trading: Actually place orders
        """
        portfolio = self.decision_manager.get_portfolio_status()
        size_usdt = portfolio['available_capital'] * (size_pct / 100)

        # Calculate quantity
        quantity = size_usdt / entry_price

        if self.config.PAPER_TRADING:
            # Paper trading - just log
            print(f"[PAPER] Would execute {direction} {quantity:.6f} {symbol} @ ${entry_price:.2f}")
        else:
            # Live trading - place orders
            side = "BUY" if direction == "LONG" else "SELL"

            # Market order to enter
            self.exchange.create_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="MARKET"
            )

            # Set TP/SL orders
            tp_side = "SELL" if direction == "LONG" else "BUY"
            sl_side = "SELL" if direction == "LONG" else "BUY"

            # Take profit limit order
            self.exchange.create_limit_order(
                symbol=symbol,
                side=tp_side,
                quantity=quantity,
                price=tp_price
            )

            # Stop loss order
            # Note: Binance Futures uses stopMarket for SL
            # Implementation depends on exchange API

    def _get_market_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Get market snapshot for a symbol"""
        try:
            # Get OHLCV data
            klines_4h = self.exchange.get_klines(symbol, '4h', limit=100)
            klines_1d = self.exchange.get_klines(symbol, '1d', limit=100)

            # Get indicators
            indicators = self.tools_executor._get_technical_indicators(
                symbol=symbol,
                timeframe='4h',
                indicators=['all']
            )

            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "klines_4h": klines_4h[-10:],  # Last 10 candles
                "klines_1d": klines_1d[-10:],
                "indicators": indicators.get('indicators', {})
            }
        except Exception as e:
            return {"error": str(e)}

    def _process_ai_decision_response(self, response: str):
        """Process and log AI decision response"""
        print("\n" + "="*80)
        print("AI DECISION SUMMARY")
        print("="*80)
        print(response)
        print("="*80 + "\n")

    def _check_and_close_positions(self):
        """Check open positions and close if TP/SL hit"""
        open_positions = self.decision_manager.get_open_positions()

        if open_positions['count'] == 0:
            print("[Info] No open positions to check")
            return

        print(f"\n[Check] Monitoring {open_positions['count']} open positions...")

        for pos in open_positions['positions']:
            symbol = pos['symbol']
            direction = pos['direction']
            current_price = pos['current_price']
            tp_price = pos['tp_price']
            sl_price = pos['sl_price']

            # Check TP
            if direction == "LONG" and current_price >= tp_price:
                self._close_position(symbol, current_price, "TP")
            elif direction == "SHORT" and current_price <= tp_price:
                self._close_position(symbol, current_price, "TP")

            # Check SL
            elif direction == "LONG" and current_price <= sl_price:
                self._close_position(symbol, current_price, "SL")
            elif direction == "SHORT" and current_price >= sl_price:
                self._close_position(symbol, current_price, "SL")

    def _close_position(self, symbol: str, exit_price: float, exit_reason: str):
        """Close a position"""
        decision = self.decision_manager.close_position(symbol, exit_price, exit_reason)

        if decision:
            self.stats['positions_closed'] += 1

            result_emoji = "✅" if decision.success else "❌"
            print(f"[{exit_reason}] Closed {symbol}: {result_emoji} "
                  f"PnL: ${decision.pnl_usdt:.2f} ({decision.pnl_pct:.2f}%)")

            # Send Telegram notification
            if self.telegram and self.config.TELEGRAM_SEND_POSITION_ALERTS:
                self.telegram.send_message(
                    f"{result_emoji} Position Closed ({exit_reason})\n\n"
                    f"Symbol: {symbol}\n"
                    f"Direction: {decision.direction}\n"
                    f"Entry: ${decision.entry_price:.2f}\n"
                    f"Exit: ${exit_price:.2f}\n"
                    f"PnL: ${decision.pnl_usdt:.2f} ({decision.pnl_pct:.2f}%)\n"
                    f"Duration: {(decision.exit_time - decision.timestamp).total_seconds() / 3600:.1f}h"
                )

            # Save exchange state after closing position (for paper trading)
            if self.config.PAPER_TRADING and hasattr(self.exchange, 'save_state'):
                try:
                    state_file = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "ai_trading_module",
                        "data",
                        "ai_paper_exchange_state.json"
                    )
                    self.exchange.save_state(state_file)
                except Exception as e:
                    print(f"[Warning] Failed to save exchange state: {e}")

    def _check_emergency_conditions(self, portfolio: Dict) -> bool:
        """Check for emergency stop conditions"""
        emergency_triggered = False
        emergency_reasons = []

        # Check drawdown
        total_return_pct = portfolio['total_return_pct']
        if total_return_pct < -self.config.EMERGENCY_STOP_DRAWDOWN_PCT:
            emergency_triggered = True
            emergency_reasons.append(f"Drawdown: {total_return_pct:.2f}%")
            print(f"[EMERGENCY] Drawdown limit reached: {total_return_pct:.2f}%")

        # Check minimum capital
        if portfolio['total_equity'] < self.config.MIN_CAPITAL_TO_CONTINUE:
            emergency_triggered = True
            emergency_reasons.append(f"Capital below minimum: ${portfolio['total_equity']:.2f}")
            print(f"[EMERGENCY] Capital below minimum: ${portfolio['total_equity']:.2f}")

        # Check consecutive losses
        recent_decisions = self.decision_manager.get_historical_decisions(
            outcome="failure",
            limit=self.config.MAX_CONSECUTIVE_LOSSES,
            days_back=7
        )

        if len(recent_decisions) >= self.config.MAX_CONSECUTIVE_LOSSES:
            emergency_triggered = True
            emergency_reasons.append(f"{len(recent_decisions)} consecutive losses in last 7 days")
            print(f"[EMERGENCY] {len(recent_decisions)} consecutive losses detected")
            print(f"[EMERGENCY] To reset: Delete ai_trading_module/data/decisions.json or wait 7 days")

        # Send comprehensive emergency alert
        if emergency_triggered:
            msg = (
                f"🚨 EMERGENCY STOP ACTIVATED\n\n"
                f"New positions paused due to:\n"
                + "\n".join([f"• {reason}" for reason in emergency_reasons]) +
                f"\n\n📊 Current Status:\n"
                f"• Equity: ${portfolio['total_equity']:.2f}\n"
                f"• Return: {total_return_pct:.2f}%\n"
                f"• Open Positions: {portfolio['num_open_positions']}\n\n"
                f"Bot will continue monitoring existing positions.\n"
                f"To reset: Delete decisions.json or adjust config."
            )

            if self.telegram:
                self.telegram.send_message(msg)

            print("\n" + "="*80)
            print(msg)
            print("="*80 + "\n")

        return emergency_triggered

    def _send_daily_report(self, ai_response: str):
        """Send daily decision report to Telegram"""
        if not self.telegram or not self.config.TELEGRAM_SEND_DAILY_REPORT:
            return

        portfolio = self.decision_manager.get_portfolio_status()
        open_positions = self.decision_manager.get_open_positions()

        report = f"""🤖 **AI Trading Daily Report**

**Portfolio Status**
• Total Equity: ${portfolio['total_equity']:.2f}
• Return: {portfolio['total_return_pct']:.2f}%
• Open Positions: {portfolio['num_open_positions']}/{self.config.MAX_POSITIONS}
• Exposure: {portfolio['exposure_pct']:.1f}%

**Decisions Made:** {self.stats['decisions_made']}
**API Cost Today:** ${self.daily_api_cost:.2f}

**Open Positions:**
"""

        for pos in open_positions['positions'][:5]:  # Top 5
            pnl_emoji = "📈" if pos['unrealized_pnl_usdt'] > 0 else "📉"
            report += (
                f"\n{pnl_emoji} {pos['symbol']} {pos['direction']}\n"
                f"   PnL: ${pos['unrealized_pnl_usdt']:.2f} ({pos['unrealized_pnl_pct']:.2f}%)"
            )

        # Add AI summary (first 500 chars)
        report += f"\n\n**AI Summary:**\n{ai_response[:500]}..."

        self.telegram.send_message(report)

    def _send_shutdown_report(self):
        """Send shutdown report"""
        if not self.telegram:
            return

        portfolio = self.decision_manager.get_portfolio_status()

        self.telegram.send_message(
            f"🛑 AI Trader Stopped\n\n"
            f"Final Equity: ${portfolio['total_equity']:.2f}\n"
            f"Total Return: {portfolio['total_return_pct']:.2f}%\n"
            f"Positions Opened: {self.stats['positions_opened']}\n"
            f"Positions Closed: {self.stats['positions_closed']}\n"
            f"Total API Cost: ${self.stats['total_api_cost_usd']:.2f}"
        )

    def _update_activity(self):
        """Update watchdog activity timestamp"""
        with self._activity_lock:
            self._last_activity = time.time()

    def _watchdog_monitor(self):
        """
        Watchdog thread - monitors for bot hanging

        Если бот не проявляет активности 600 секунд (10 минут),
        выводит предупреждение для отладки
        """
        TIMEOUT_SECONDS = 600  # 10 минут
        CHECK_INTERVAL = 30    # Проверка каждые 30 сек

        logger.info("[Watchdog] Monitoring started")

        while self._watchdog_enabled:
            try:
                time.sleep(CHECK_INTERVAL)

                with self._activity_lock:
                    idle_time = time.time() - self._last_activity

                if idle_time > TIMEOUT_SECONDS:
                    logger.warning(f"[WATCHDOG] No activity for {idle_time:.0f}s - possible hang!")
                    print(f"[WATCHDOG WARNING] No activity for {idle_time:.0f} seconds!")

                    # Reset activity to avoid spam
                    self._update_activity()

                    # Send Telegram alert if available
                    if self.telegram:
                        self.telegram.send_message(
                            f"⚠️ Watchdog Alert\n\n"
                            f"No bot activity for {idle_time:.0f}s\n"
                            f"Possible hang detected"
                        )

            except Exception as e:
                logger.error(f"[Watchdog] Error: {e}")

        logger.info("[Watchdog] Monitoring stopped")
