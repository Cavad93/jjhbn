"""
AI Trader - Autonomous Trading Agent using Claude Sonnet 4.5

Основной класс для AI-управляемой торговли
"""
import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import anthropic

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

        print(f"[AI Trader] Initialized with ${self.config.INITIAL_CAPITAL} capital")
        print(f"[AI Trader] Model: {self.config.AI_MODEL}")
        print(f"[AI Trader] Paper Trading: {self.config.PAPER_TRADING}")

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

                # Check if it's time for daily decision cycle
                if self._should_make_decisions(current_time):
                    print(f"\n[{current_time}] Starting daily decision cycle...")
                    self._daily_decision_cycle()
                    self.last_decision_time = current_time

                # Check positions periodically
                if self._should_check_positions(current_time):
                    print(f"\n[{current_time}] Checking positions...")
                    self._check_and_close_positions()
                    self.last_position_check_time = current_time

                # Sleep for a minute
                time.sleep(60)

        except KeyboardInterrupt:
            print("\n\nAI Trader stopped by user")
            self._send_shutdown_report()
        except Exception as e:
            print(f"\n\nCritical error: {e}")
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
            tp_price = position_params['tp_price']
            sl_price = position_params['sl_price']
            reasoning = position_params['reasoning']
            confidence = position_params.get('confidence', 0.7)

            # Get current price
            ticker = self.exchange.get_ticker(symbol)
            entry_price = float(ticker['lastPrice'])

            # Get market data for record
            market_data = self._get_market_snapshot(symbol)

            # Record decision
            decision_id = self.decision_manager.record_decision(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                tp_price=tp_price,
                sl_price=sl_price,
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
                tp_price=tp_price,
                sl_price=sl_price
            )

            self.stats['positions_opened'] += 1

            print(f"[SUCCESS] Opened {direction} position on {symbol}")
            print(f"  Entry: ${entry_price:.2f} | TP: ${tp_price:.2f} | SL: ${sl_price:.2f}")
            print(f"  Size: {size_pct:.1f}% | Confidence: {confidence:.2f}")

        except Exception as e:
            print(f"[ERROR] Failed to open position: {e}")

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

    def _check_emergency_conditions(self, portfolio: Dict) -> bool:
        """Check for emergency stop conditions"""
        # Check drawdown
        total_return_pct = portfolio['total_return_pct']
        if total_return_pct < -self.config.EMERGENCY_STOP_DRAWDOWN_PCT:
            print(f"[EMERGENCY] Drawdown limit reached: {total_return_pct:.2f}%")
            if self.telegram:
                self.telegram.send_message(
                    f"🚨 EMERGENCY STOP\n\n"
                    f"Drawdown: {total_return_pct:.2f}%\n"
                    f"Trading paused"
                )
            return True

        # Check minimum capital
        if portfolio['total_equity'] < self.config.MIN_CAPITAL_TO_CONTINUE:
            print(f"[EMERGENCY] Capital below minimum: ${portfolio['total_equity']:.2f}")
            return True

        # Check consecutive losses
        recent_decisions = self.decision_manager.get_historical_decisions(
            outcome="failure",
            limit=self.config.MAX_CONSECUTIVE_LOSSES,
            days_back=7
        )

        if len(recent_decisions) >= self.config.MAX_CONSECUTIVE_LOSSES:
            print(f"[EMERGENCY] {self.config.MAX_CONSECUTIVE_LOSSES} consecutive losses")
            return True

        return False

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
