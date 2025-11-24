"""
Decision Manager - Управление решениями и позициями AI Trading Module

Хранит:
- Все решения AI (открытие позиций)
- Результаты закрытых позиций
- Статистику для self-learning
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid


@dataclass
class AIDecision:
    """Решение AI о открытии позиции"""
    decision_id: str
    timestamp: datetime
    symbol: str
    direction: str  # LONG/SHORT

    # Данные на момент принятия решения
    entry_price: float
    tp_price: float
    sl_price: float
    size_pct: float
    size_usdt: float

    # AI reasoning
    reasoning: str
    confidence: float

    # Данные для анализа
    market_data: Dict[str, Any]  # OHLCV, indicators
    fundamental_factors: List[str]  # Ключевые факторы из фундаментального анализа
    technical_factors: Dict[str, Any]  # Технические сигналы

    # Результат (заполняется после закрытия)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # TP, SL, MANUAL
    pnl_usdt: Optional[float] = None
    pnl_pct: Optional[float] = None
    success: Optional[bool] = None  # True if profit

    # Метаданные
    status: str = "OPEN"  # OPEN, CLOSED

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        if self.exit_time:
            d['exit_time'] = self.exit_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'AIDecision':
        """Create from dictionary"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('exit_time'):
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
        return cls(**data)


class DecisionManager:
    """
    Управление решениями и позициями AI модуля

    Функции:
    - Сохранение всех решений AI
    - Хранение результатов
    - Анализ исторической производительности
    - Управление капиталом
    """

    def __init__(self, config, exchange):
        """
        Initialize DecisionManager

        Args:
            config: AITradingConfig
            exchange: Exchange client (PaperExchange or BinanceClient)
        """
        self.config = config
        self.exchange = exchange

        # Storage
        self.decisions: Dict[str, AIDecision] = {}  # decision_id -> AIDecision
        self.open_positions: Dict[str, AIDecision] = {}  # symbol -> AIDecision

        # Capital tracking
        self.initial_capital = config.INITIAL_CAPITAL
        self.current_balance = config.INITIAL_CAPITAL
        self.total_pnl = 0.0

        # Load existing data
        self._load_state()

    def record_decision(self, symbol: str, direction: str, entry_price: float,
                       tp_price: float, sl_price: float, size_pct: float,
                       reasoning: str, confidence: float,
                       market_data: Dict, fundamental_factors: List[str],
                       technical_factors: Dict) -> str:
        """
        Record a trading decision

        Returns:
            decision_id
        """
        decision_id = str(uuid.uuid4())

        size_usdt = self.current_balance * (size_pct / 100)

        decision = AIDecision(
            decision_id=decision_id,
            timestamp=datetime.now(),
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            size_pct=size_pct,
            size_usdt=size_usdt,
            reasoning=reasoning,
            confidence=confidence,
            market_data=market_data,
            fundamental_factors=fundamental_factors,
            technical_factors=technical_factors,
            status="OPEN"
        )

        self.decisions[decision_id] = decision
        self.open_positions[symbol] = decision

        # Update balance (reduce available capital)
        # Note: In paper trading, capital is allocated but not spent

        self._save_state()

        return decision_id

    def close_position(self, symbol: str, exit_price: float, exit_reason: str) -> Optional[AIDecision]:
        """
        Close a position and record result

        Args:
            symbol: Trading symbol
            exit_price: Exit price
            exit_reason: Reason for closing (TP, SL, MANUAL)

        Returns:
            Closed decision or None if not found
        """
        if symbol not in self.open_positions:
            return None

        decision = self.open_positions[symbol]
        decision.exit_time = datetime.now()
        decision.exit_price = exit_price
        decision.exit_reason = exit_reason
        decision.status = "CLOSED"

        # Calculate PnL
        if decision.direction == "LONG":
            pnl_pct = (exit_price - decision.entry_price) / decision.entry_price * 100
        else:  # SHORT
            pnl_pct = (decision.entry_price - exit_price) / decision.entry_price * 100

        pnl_usdt = decision.size_usdt * (pnl_pct / 100)

        # Account for fees (0.1% entry + 0.1% exit)
        fees = decision.size_usdt * 0.002
        pnl_usdt -= fees

        decision.pnl_usdt = pnl_usdt
        decision.pnl_pct = pnl_pct
        decision.success = pnl_usdt > 0

        # Update balance
        self.current_balance += pnl_usdt
        self.total_pnl += pnl_usdt

        # Remove from open positions
        del self.open_positions[symbol]

        self._save_state()

        return decision

    def get_open_positions(self) -> Dict[str, Any]:
        """Get all open positions with current PnL"""
        positions = []

        for symbol, decision in self.open_positions.items():
            try:
                # Get current price
                ticker = self.exchange.get_ticker(symbol)
                current_price = float(ticker['lastPrice'])

                # Calculate unrealized PnL
                if decision.direction == "LONG":
                    unrealized_pnl_pct = (current_price - decision.entry_price) / decision.entry_price * 100
                else:
                    unrealized_pnl_pct = (decision.entry_price - current_price) / decision.entry_price * 100

                unrealized_pnl_usdt = decision.size_usdt * (unrealized_pnl_pct / 100)

                # Distance to TP/SL
                if decision.direction == "LONG":
                    distance_to_tp_pct = (decision.tp_price - current_price) / current_price * 100
                    distance_to_sl_pct = (current_price - decision.sl_price) / current_price * 100
                else:
                    distance_to_tp_pct = (current_price - decision.tp_price) / current_price * 100
                    distance_to_sl_pct = (decision.sl_price - current_price) / current_price * 100

                positions.append({
                    "symbol": symbol,
                    "direction": decision.direction,
                    "entry_price": decision.entry_price,
                    "current_price": current_price,
                    "tp_price": decision.tp_price,
                    "sl_price": decision.sl_price,
                    "size_usdt": decision.size_usdt,
                    "unrealized_pnl_usdt": round(unrealized_pnl_usdt, 2),
                    "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
                    "distance_to_tp_pct": round(distance_to_tp_pct, 2),
                    "distance_to_sl_pct": round(distance_to_sl_pct, 2),
                    "entry_time": decision.timestamp.isoformat(),
                    "holding_hours": (datetime.now() - decision.timestamp).total_seconds() / 3600,
                    "confidence": decision.confidence
                })
            except Exception as e:
                print(f"Error getting position info for {symbol}: {e}")

        return {
            "count": len(positions),
            "positions": positions
        }

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status"""
        # Get unrealized PnL
        unrealized_pnl = 0.0
        total_allocated = 0.0

        for symbol, decision in self.open_positions.items():
            try:
                ticker = self.exchange.get_ticker(symbol)
                current_price = float(ticker['lastPrice'])

                if decision.direction == "LONG":
                    pnl_pct = (current_price - decision.entry_price) / decision.entry_price * 100
                else:
                    pnl_pct = (decision.entry_price - current_price) / decision.entry_price * 100

                unrealized_pnl += decision.size_usdt * (pnl_pct / 100)
                total_allocated += decision.size_usdt
            except:
                total_allocated += decision.size_usdt

        total_equity = self.current_balance + unrealized_pnl
        exposure_pct = (total_allocated / self.initial_capital) * 100
        available_capital = self.current_balance - total_allocated

        # Calculate returns
        total_return_pct = ((total_equity - self.initial_capital) / self.initial_capital) * 100

        return {
            "initial_capital": self.initial_capital,
            "current_balance": round(self.current_balance, 2),
            "total_allocated": round(total_allocated, 2),
            "available_capital": round(available_capital, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_equity": round(total_equity, 2),
            "total_return_pct": round(total_return_pct, 2),
            "exposure_pct": round(exposure_pct, 2),
            "num_open_positions": len(self.open_positions),
            "max_positions": self.config.MAX_POSITIONS,
            "total_realized_pnl": round(self.total_pnl, 2)
        }

    def get_historical_decisions(self, symbol: Optional[str] = None,
                                outcome: str = "all", limit: int = 20,
                                days_back: int = 90) -> List[Dict]:
        """
        Get historical decisions

        Args:
            symbol: Filter by symbol (optional)
            outcome: 'success', 'failure', or 'all'
            limit: Maximum number of decisions
            days_back: Look back period in days

        Returns:
            List of decisions
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)

        filtered = []
        for decision in self.decisions.values():
            # Filter by date
            if decision.timestamp < cutoff_date:
                continue

            # Filter by symbol
            if symbol and decision.symbol != symbol:
                continue

            # Filter by outcome (only closed positions)
            if outcome != "all" and decision.status == "CLOSED":
                if outcome == "success" and not decision.success:
                    continue
                if outcome == "failure" and decision.success:
                    continue

            filtered.append({
                "decision_id": decision.decision_id,
                "timestamp": decision.timestamp.isoformat(),
                "symbol": decision.symbol,
                "direction": decision.direction,
                "entry_price": decision.entry_price,
                "tp_price": decision.tp_price,
                "sl_price": decision.sl_price,
                "size_pct": decision.size_pct,
                "reasoning": decision.reasoning[:200] + "..." if len(decision.reasoning) > 200 else decision.reasoning,
                "confidence": decision.confidence,
                "status": decision.status,
                "exit_reason": decision.exit_reason,
                "pnl_usdt": decision.pnl_usdt,
                "pnl_pct": decision.pnl_pct,
                "success": decision.success
            })

        # Sort by timestamp (newest first)
        filtered.sort(key=lambda x: x['timestamp'], reverse=True)

        return filtered[:limit]

    def analyze_performance(self, groupby: str = "all", days_back: int = 90) -> Dict[str, Any]:
        """
        Analyze historical performance

        Args:
            groupby: 'all', 'symbol', 'direction', or 'timeframe'
            days_back: Look back period

        Returns:
            Performance statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)

        # Get closed positions
        closed = [
            d for d in self.decisions.values()
            if d.status == "CLOSED" and d.timestamp >= cutoff_date
        ]

        if not closed:
            return {
                "message": "No closed positions in the specified period",
                "days_back": days_back
            }

        # Overall statistics
        total_trades = len(closed)
        winning_trades = sum(1 for d in closed if d.success)
        losing_trades = total_trades - winning_trades

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_pnl = sum(d.pnl_usdt for d in closed)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        winners = [d for d in closed if d.success]
        losers = [d for d in closed if not d.success]

        avg_win = sum(d.pnl_usdt for d in winners) / len(winners) if winners else 0
        avg_loss = sum(d.pnl_usdt for d in losers) / len(losers) if losers else 0

        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')

        # Best and worst trades
        best_trade = max(closed, key=lambda d: d.pnl_usdt)
        worst_trade = min(closed, key=lambda d: d.pnl_usdt)

        result = {
            "period_days": days_back,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_pct": round(win_rate, 2),
            "total_pnl_usdt": round(total_pnl, 2),
            "average_pnl_usdt": round(avg_pnl, 2),
            "average_win_usdt": round(avg_win, 2),
            "average_loss_usdt": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "best_trade": {
                "symbol": best_trade.symbol,
                "direction": best_trade.direction,
                "pnl_usdt": round(best_trade.pnl_usdt, 2),
                "pnl_pct": round(best_trade.pnl_pct, 2)
            },
            "worst_trade": {
                "symbol": worst_trade.symbol,
                "direction": worst_trade.direction,
                "pnl_usdt": round(worst_trade.pnl_usdt, 2),
                "pnl_pct": round(worst_trade.pnl_pct, 2)
            }
        }

        # Group by analysis
        if groupby == "direction":
            long_trades = [d for d in closed if d.direction == "LONG"]
            short_trades = [d for d in closed if d.direction == "SHORT"]

            result["by_direction"] = {
                "LONG": self._calculate_group_stats(long_trades),
                "SHORT": self._calculate_group_stats(short_trades)
            }

        elif groupby == "symbol":
            # Group by symbol
            symbols = set(d.symbol for d in closed)
            result["by_symbol"] = {}
            for symbol in symbols:
                symbol_trades = [d for d in closed if d.symbol == symbol]
                result["by_symbol"][symbol] = self._calculate_group_stats(symbol_trades)

        return result

    def _calculate_group_stats(self, trades: List[AIDecision]) -> Dict[str, Any]:
        """Calculate statistics for a group of trades"""
        if not trades:
            return {"count": 0}

        total = len(trades)
        winners = sum(1 for t in trades if t.success)

        return {
            "count": total,
            "win_rate_pct": round(winners / total * 100, 2),
            "total_pnl_usdt": round(sum(t.pnl_usdt for t in trades), 2),
            "avg_pnl_usdt": round(sum(t.pnl_usdt for t in trades) / total, 2)
        }

    def _save_state(self):
        """Save state to files"""
        try:
            # Ensure data directory exists
            os.makedirs(self.config.DATA_DIR, exist_ok=True)

            # Save decisions
            decisions_data = {
                "decisions": [d.to_dict() for d in self.decisions.values()]
            }
            with open(self.config.DECISIONS_FILE, 'w') as f:
                json.dump(decisions_data, f, indent=2)

            # Save positions
            positions_data = {
                "open_positions": [d.to_dict() for d in self.open_positions.values()]
            }
            with open(self.config.POSITIONS_FILE, 'w') as f:
                json.dump(positions_data, f, indent=2)

            # Save state
            state_data = {
                "initial_capital": self.initial_capital,
                "current_balance": self.current_balance,
                "total_pnl": self.total_pnl,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.config.STATE_FILE, 'w') as f:
                json.dump(state_data, f, indent=2)

        except Exception as e:
            print(f"Error saving state: {e}")

    def _load_state(self):
        """Load state from files"""
        try:
            # Load decisions
            if os.path.exists(self.config.DECISIONS_FILE):
                with open(self.config.DECISIONS_FILE, 'r') as f:
                    data = json.load(f)
                    for d in data.get('decisions', []):
                        decision = AIDecision.from_dict(d)
                        self.decisions[decision.decision_id] = decision

            # Load open positions
            if os.path.exists(self.config.POSITIONS_FILE):
                with open(self.config.POSITIONS_FILE, 'r') as f:
                    data = json.load(f)
                    for d in data.get('open_positions', []):
                        decision = AIDecision.from_dict(d)
                        self.open_positions[decision.symbol] = decision

            # Load state
            if os.path.exists(self.config.STATE_FILE):
                with open(self.config.STATE_FILE, 'r') as f:
                    data = json.load(f)
                    self.current_balance = data.get('current_balance', self.initial_capital)
                    self.total_pnl = data.get('total_pnl', 0.0)

        except Exception as e:
            print(f"Error loading state: {e}")
