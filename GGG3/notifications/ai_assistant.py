#!/usr/bin/env python3
"""
AI Trading Assistant - Claude Sonnet 4.5 Interactive Chat

Возможности:
- Интерактивный чат с ботом через Telegram
- Полный контекст о состоянии бота (позиции, сделки, капитал, модели)
- Анализ решений и объяснение действий бота
- Диагностика проблем и рекомендации
- Prompt caching для экономии ~90% токенов
- Session management (вкл/выкл для экономии)

Примеры вопросов:
- "Почему бот открыл позицию по BTCUSDT?"
- "Сколько я заработал на ETHUSDT?"
- "Есть ли проблемы с ботом?"
- "Что с капиталом за неделю?"
- "Какая модель работает лучше всего?"

Автор: Claude Code
Дата: 2025-11-12
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

# Anthropic API
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None
    logging.warning("anthropic not installed. Run: pip install anthropic")

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class AIAssistantConfig:
    """Конфигурация AI ассистента"""

    # Модель
    MODEL_ID = "claude-sonnet-4-5-20250929"  # Актуальная версия от 29 сентября 2025
    MODEL_TEMPERATURE = 0.7  # Баланс между креативностью и точностью
    MAX_TOKENS = 4096  # Максимум токенов в ответе

    # Стоимость (Sonnet 4.5: $3/$15 per million tokens)
    COST_PER_MILLION_INPUT_TOKENS = 3.0
    COST_PER_MILLION_OUTPUT_TOKENS = 15.0
    COST_PER_MILLION_CACHE_WRITE = 3.75  # Cache write (первый раз)
    COST_PER_MILLION_CACHE_READ = 0.30   # Cache read (экономия ~90%)

    # Бюджет
    MONTHLY_BUDGET_USD = 30.0  # Максимальный месячный бюджет

    # Сессия
    SESSION_TIMEOUT_MINUTES = 30  # Таймаут неактивности
    MAX_CONTEXT_MESSAGES = 20  # Максимум сообщений в контексте

    # Кэш
    CACHE_DIR = "./data/ai_assistant_cache"
    CONTEXT_CACHE_TTL_SECONDS = 300  # 5 минут (контекст бота обновляется)

    # Ограничения
    MAX_MESSAGE_LENGTH = 2000  # Максимум символов в одном сообщении Telegram
    MAX_TRADES_IN_CONTEXT = 50  # Максимум сделок в контексте
    MAX_POSITIONS_IN_CONTEXT = 20  # Максимум позиций в контексте


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

@dataclass
class UserSession:
    """Сессия пользователя"""

    user_id: str
    chat_id: str
    active: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    # История диалога
    message_history: List[Dict[str, str]] = field(default_factory=list)

    # Статистика
    total_messages: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0

    def is_expired(self, timeout_minutes: int = AIAssistantConfig.SESSION_TIMEOUT_MINUTES) -> bool:
        """Проверка истечения сессии"""
        return (datetime.utcnow() - self.last_activity).total_seconds() > (timeout_minutes * 60)

    def update_activity(self):
        """Обновление времени последней активности"""
        self.last_activity = datetime.utcnow()

    def add_message(self, role: str, content: str):
        """Добавление сообщения в историю"""
        self.message_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Ограничиваем размер истории
        if len(self.message_history) > AIAssistantConfig.MAX_CONTEXT_MESSAGES * 2:
            # Удаляем старые, но оставляем первое системное сообщение
            self.message_history = self.message_history[:1] + self.message_history[-(AIAssistantConfig.MAX_CONTEXT_MESSAGES * 2 - 1):]

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Получить историю для API (без timestamp)"""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.message_history]

    def update_stats(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0):
        """Обновление статистики использования"""
        self.total_messages += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cached_tokens += cached_tokens

        # Расчет стоимости
        cost_input = (input_tokens / 1_000_000) * AIAssistantConfig.COST_PER_MILLION_INPUT_TOKENS
        cost_output = (output_tokens / 1_000_000) * AIAssistantConfig.COST_PER_MILLION_OUTPUT_TOKENS
        cost_cache = (cached_tokens / 1_000_000) * AIAssistantConfig.COST_PER_MILLION_CACHE_READ

        self.total_cost_usd += (cost_input + cost_output - cost_cache)

    def to_dict(self) -> dict:
        """Сериализация в словарь"""
        return {
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_history": self.message_history,
            "total_messages": self.total_messages,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_cost_usd": self.total_cost_usd
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserSession":
        """Десериализация из словаря"""
        session = cls(
            user_id=data["user_id"],
            chat_id=data["chat_id"],
            active=data.get("active", False),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"])
        )
        session.message_history = data.get("message_history", [])
        session.total_messages = data.get("total_messages", 0)
        session.total_input_tokens = data.get("total_input_tokens", 0)
        session.total_output_tokens = data.get("total_output_tokens", 0)
        session.total_cached_tokens = data.get("total_cached_tokens", 0)
        session.total_cost_usd = data.get("total_cost_usd", 0.0)
        return session


# ============================================================================
# BOT CONTEXT COLLECTOR
# ============================================================================

class BotContextCollector:
    """
    Сборщик контекста о состоянии бота

    Собирает информацию о:
    - Открытых позициях
    - Истории сделок
    - Капитале и изменениях
    - Моделях и их точности
    - Активных режимах (night mode, black swan, etc.)
    - Ошибках и предупреждениях
    """

    def __init__(self, position_manager, capital_tracker, config, exchange=None):
        """
        Args:
            position_manager: PositionManager instance
            capital_tracker: CapitalTracker instance
            config: Bot configuration
            exchange: Exchange instance для получения текущих цен (опционально)
        """
        self.position_manager = position_manager
        self.capital_tracker = capital_tracker
        self.config = config
        self.exchange = exchange

        logger.debug("[BotContextCollector] Initialized")

    def collect_full_context(self) -> Dict[str, Any]:
        """
        Сбор полного контекста о боте

        Returns:
            Словарь с данными о боте
        """
        context = {
            "timestamp": datetime.utcnow().isoformat(),
            "positions": self._collect_positions(),
            "trades": self._collect_trades(),
            "capital": self._collect_capital(),
            "statistics": self._collect_statistics(),
            "modes": self._collect_modes(),
            "errors": self._collect_errors()
        }

        logger.debug(f"[BotContextCollector] Collected context: {len(context['positions']['open'])} open positions, "
                    f"{len(context['trades']['recent'])} recent trades")

        return context

    def _collect_positions(self) -> Dict[str, Any]:
        """Сбор информации о позициях с текущими ценами и PnL"""
        open_positions = self.position_manager.get_all_open()

        positions_data = []
        total_unrealized_pnl = 0.0

        for pos in open_positions:
            pos_data = {
                "symbol": pos.symbol,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
                "position_value": pos.position_value,
                "tp_price": pos.tp_price,
                "sl_price": pos.sl_price,
                "atr_value": pos.atr_value,
                "predictions": pos.entry_snapshot.get("predictions", {}),
                "p_meta": pos.entry_snapshot.get("p_meta", 0.0),
                "ev": pos.entry_snapshot.get("selected_ev", 0.0),
                "metadata": pos.metadata
            }

            # Получаем текущую цену и вычисляем нереализованный PnL
            if self.exchange:
                try:
                    ticker = self.exchange.get_ticker(pos.symbol)
                    current_price = ticker.get('lastPrice', pos.entry_price)

                    # Вычисляем нереализованный PnL
                    pnl_usdt, pnl_pct = pos.calculate_pnl(current_price)

                    pos_data["current_price"] = current_price
                    pos_data["unrealized_pnl"] = pnl_usdt
                    pos_data["unrealized_pnl_pct"] = pnl_pct

                    total_unrealized_pnl += pnl_usdt
                except Exception as e:
                    logger.warning(f"[BotContextCollector] Failed to get current price for {pos.symbol}: {e}")
                    pos_data["current_price"] = None
                    pos_data["unrealized_pnl"] = None
                    pos_data["unrealized_pnl_pct"] = None
            else:
                pos_data["current_price"] = None
                pos_data["unrealized_pnl"] = None
                pos_data["unrealized_pnl_pct"] = None

            positions_data.append(pos_data)

        return {
            "open": positions_data,
            "count": len(positions_data),
            "total_value": sum(p["position_value"] for p in positions_data),
            "total_unrealized_pnl": total_unrealized_pnl
        }

    def _collect_trades(self) -> Dict[str, Any]:
        """Сбор истории сделок"""
        closed_positions = self.position_manager.get_recent_closed(limit=AIAssistantConfig.MAX_TRADES_IN_CONTEXT)

        # Последние N сделок
        recent_trades = []
        for pos in closed_positions:
            recent_trades.append({
                "symbol": pos.symbol,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "exit_price": pos.exit_price,
                "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
                "exit_time": pos.exit_time.isoformat() if pos.exit_time else None,
                "pnl": pos.pnl,
                "pnl_pct": pos.pnl_pct,
                "exit_reason": pos.exit_reason,
                "p_meta": pos.entry_snapshot.get("p_meta", 0.0),
                "ev": pos.entry_snapshot.get("selected_ev", 0.0)
            })

        # Группировка по символам
        by_symbol = {}
        for pos in closed_positions:
            symbol = pos.symbol
            if symbol not in by_symbol:
                by_symbol[symbol] = {"trades": 0, "pnl": 0.0, "wins": 0}
            by_symbol[symbol]["trades"] += 1
            by_symbol[symbol]["pnl"] += pos.pnl
            if pos.pnl > 0:
                by_symbol[symbol]["wins"] += 1

        return {
            "recent": recent_trades,
            "by_symbol": by_symbol,
            "total_count": len(closed_positions)
        }

    def _collect_capital(self) -> Dict[str, Any]:
        """Сбор информации о капитале"""
        # Получаем последний snapshot
        latest_snapshot = self.capital_tracker.get_latest_snapshot()

        if not latest_snapshot:
            return {
                "total_equity": 0.0,
                "free_capital": 0.0,
                "locked_capital": 0.0,
                "reserve_fund": 0.0,
                "changes": {}
            }

        equity = latest_snapshot.get("total_equity", 0.0)
        free = latest_snapshot.get("free_balance", 0.0)
        locked = latest_snapshot.get("locked_in_positions", 0.0)
        reserve = latest_snapshot.get("reserve_fund", 0.0)

        # Изменения за периоды
        changes = {}
        for period_name, hours in [("1h", 1), ("24h", 24), ("7d", 168), ("30d", 720)]:
            change_data = self.capital_tracker.get_change_over_period(equity, hours)
            if change_data and change_data.get("available"):
                changes[period_name] = {
                    "absolute": change_data["absolute"],
                    "percent": change_data["percent"],
                    "available": change_data["available"]
                }

        return {
            "total_equity": equity,
            "free_capital": free,
            "locked_capital": locked,
            "reserve_fund": reserve,
            "changes": changes
        }

    def _collect_statistics(self) -> Dict[str, Any]:
        """Сбор торговой статистики"""
        stats = self.position_manager.get_pnl_summary()

        return {
            "total_trades": stats.get("total_trades", 0),
            "win_rate": stats.get("win_rate", 0.0),
            "total_pnl": stats.get("total_pnl", 0.0),
            "avg_win": stats.get("avg_win", 0.0),
            "avg_loss": stats.get("avg_loss", 0.0),
            "best_trade": stats.get("max_win", 0.0),
            "worst_trade": stats.get("max_loss", 0.0),
            "sharpe_ratio": 0.0,  # Not calculated by PositionManager
            "max_drawdown": 0.0   # Not calculated by PositionManager
        }

    def _collect_modes(self) -> Dict[str, Any]:
        """Сбор информации об активных режимах"""
        # TODO: Добавить проверку активных режимов из bot main
        return {
            "night_mode": False,
            "black_swan_protection": False,
            "paper_trading": getattr(self.config, 'PAPER_TRADING_MODE', True)
        }

    def _collect_errors(self) -> List[str]:
        """Сбор последних ошибок"""
        # TODO: Добавить чтение из лог-файла или error tracker
        return []

    def format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """
        Форматирование контекста для промпта

        Args:
            context: Контекст от collect_full_context()

        Returns:
            Форматированный текст для системного промпта
        """
        lines = []

        # Заголовок
        lines.append("=== BOT STATE CONTEXT ===")
        lines.append(f"Timestamp: {context['timestamp']}")
        lines.append("")

        # Позиции
        positions = context['positions']
        lines.append(f"OPEN POSITIONS: {positions['count']}")
        for i, pos in enumerate(positions['open'], 1):
            lines.append(
                f"{i}. {pos['symbol']} {pos['direction']} @ ${pos['entry_price']:.4f} "
                f"(value=${pos['position_value']:.2f}, TP=${pos['tp_price']:.4f}, SL=${pos['sl_price']:.4f})"
            )
            lines.append(f"   META p_up={pos['p_meta']:.3f}, EV={pos['ev']:.4f}")
        lines.append("")

        # Капитал
        capital = context['capital']
        lines.append(f"CAPITAL: Total=${capital['total_equity']:,.2f}, Free=${capital['free_capital']:,.2f}, "
                    f"Locked=${capital['locked_capital']:,.2f}")
        if capital['changes']:
            lines.append("Changes:")
            for period, data in capital['changes'].items():
                if data['available']:
                    sign = "+" if data['absolute'] >= 0 else ""
                    lines.append(f"  {period}: {sign}${data['absolute']:.2f} ({sign}{data['percent']:.2f}%)")
        lines.append("")

        # Статистика
        stats = context['statistics']
        lines.append(f"STATISTICS: {stats['total_trades']} trades, WR={stats['win_rate']:.1f}%, "
                    f"PnL=${stats['total_pnl']:+.2f}")
        lines.append(f"Avg Win=${stats['avg_win']:+.2f}, Avg Loss=${stats['avg_loss']:+.2f}, "
                    f"Best=${stats['best_trade']:+.2f}, Worst=${stats['worst_trade']:+.2f}")
        lines.append("")

        # Последние сделки
        trades = context['trades']
        lines.append(f"RECENT TRADES: (last {min(5, len(trades['recent']))})")
        for trade in trades['recent'][-5:]:
            result = "WIN" if trade['pnl'] > 0 else "LOSS"
            lines.append(
                f"  {trade['symbol']} {trade['direction']}: ${trade['pnl']:+.2f} ({trade['pnl_pct']:+.2f}%) - {result} - {trade['exit_reason']}"
            )
        lines.append("")

        # Режимы
        modes = context['modes']
        active_modes = [k for k, v in modes.items() if v]
        if active_modes:
            lines.append(f"ACTIVE MODES: {', '.join(active_modes)}")
            lines.append("")

        # Ошибки
        errors = context['errors']
        if errors:
            lines.append(f"RECENT ERRORS: {len(errors)}")
            for err in errors[-5:]:
                lines.append(f"  - {err}")
            lines.append("")

        return "\n".join(lines)


# ============================================================================
# AI ASSISTANT
# ============================================================================

class AITradingAssistant:
    """
    AI Trading Assistant на базе Claude Sonnet 4.5

    Возможности:
    - Интерактивный чат через Telegram
    - Полный контекст о боте
    - Анализ решений и диагностика
    - Prompt caching для экономии токенов
    - Session management
    """

    # Системный промпт (будет закэширован)
    SYSTEM_PROMPT = """You are an AI Trading Assistant for a cryptocurrency trading bot. You help the user understand the bot's actions, analyze performance, and diagnose issues.

Your capabilities:
- Explain why the bot opened/closed specific positions
- Analyze trading performance and statistics
- Identify problems and provide recommendations
- Answer questions about capital, PnL, models, and strategies
- Provide insights based on historical data

Guidelines:
- Be concise and clear (Telegram messages should be short)
- Use emojis sparingly and only when appropriate
- Always base answers on the provided bot context
- If you don't have enough data, say so
- Provide specific numbers and references when available
- Be helpful and proactive in identifying issues

Communication style:
- Professional but friendly
- Data-driven and factual
- Clear explanations without jargon overload
- Russian language for user communication"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        context_collector: Optional[BotContextCollector] = None,
        cache_dir: str = AIAssistantConfig.CACHE_DIR
    ):
        """
        Args:
            api_key: Anthropic API key
            context_collector: BotContextCollector instance
            cache_dir: Directory for caching
        """
        if Anthropic is None:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in env or constructor")

        self.client = Anthropic(api_key=self.api_key)
        self.context_collector = context_collector

        # Кэш
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Сессии пользователей
        self.sessions: Dict[str, UserSession] = {}
        self.sessions_file = self.cache_dir / "sessions.json"
        self._load_sessions()

        # Статистика
        self.stats_file = self.cache_dir / "assistant_stats.json"
        self.stats = self._load_stats()

        logger.info(f"[AITradingAssistant] Initialized with model {AIAssistantConfig.MODEL_ID}")

    def _load_sessions(self):
        """Загрузка сохраненных сессий"""
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, 'r') as f:
                    data = json.load(f)
                    for user_id, session_data in data.items():
                        self.sessions[user_id] = UserSession.from_dict(session_data)
                logger.info(f"[AITradingAssistant] Loaded {len(self.sessions)} sessions")
            except Exception as e:
                logger.warning(f"[AITradingAssistant] Failed to load sessions: {e}")

    def _save_sessions(self):
        """Сохранение сессий"""
        try:
            data = {uid: session.to_dict() for uid, session in self.sessions.items()}
            with open(self.sessions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"[AITradingAssistant] Failed to save sessions: {e}")

    def _load_stats(self) -> dict:
        """Загрузка статистики"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[AITradingAssistant] Failed to load stats: {e}")

        return {
            "total_sessions": 0,
            "total_messages": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cached_tokens": 0,
            "total_cost_usd": 0.0,
            "sessions_by_user": {}
        }

    def _save_stats(self):
        """Сохранение статистики"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.warning(f"[AITradingAssistant] Failed to save stats: {e}")

    def activate_session(self, user_id: str, chat_id: str) -> UserSession:
        """
        Активация сессии пользователя

        Args:
            user_id: Telegram user ID
            chat_id: Telegram chat ID

        Returns:
            UserSession instance
        """
        if user_id not in self.sessions:
            self.sessions[user_id] = UserSession(user_id=user_id, chat_id=chat_id)
            self.stats["total_sessions"] += 1
            logger.info(f"[AITradingAssistant] Created new session for user {user_id}")
        else:
            # Обновляем chat_id если пользователь активирует сессию в другом чате
            self.sessions[user_id].chat_id = chat_id
            logger.info(f"[AITradingAssistant] Updated chat_id for user {user_id} to {chat_id}")

        session = self.sessions[user_id]
        session.active = True
        session.update_activity()

        # Добавляем начальный контекст если это новая сессия
        if not session.message_history:
            if self.context_collector:
                context = self.context_collector.collect_full_context()
                context_text = self.context_collector.format_context_for_prompt(context)
                session.add_message("system", f"{self.SYSTEM_PROMPT}\n\n{context_text}")
            else:
                session.add_message("system", self.SYSTEM_PROMPT)

        self._save_sessions()
        logger.info(f"[AITradingAssistant] Session activated for user {user_id}")

        return session

    def deactivate_session(self, user_id: str):
        """
        Деактивация сессии пользователя

        Args:
            user_id: Telegram user ID
        """
        if user_id in self.sessions:
            self.sessions[user_id].active = False
            self._save_sessions()
            logger.info(f"[AITradingAssistant] Session deactivated for user {user_id}")

    def is_session_active(self, user_id: str) -> bool:
        """Проверка активности сессии"""
        if user_id not in self.sessions:
            return False

        session = self.sessions[user_id]

        # Проверяем таймаут
        if session.is_expired():
            session.active = False
            self._save_sessions()
            logger.info(f"[AITradingAssistant] Session expired for user {user_id}")
            return False

        return session.active

    def chat(self, user_id: str, message: str) -> str:
        """
        Отправка сообщения ассистенту и получение ответа

        Args:
            user_id: Telegram user ID
            message: Сообщение пользователя

        Returns:
            Ответ ассистента
        """
        if user_id not in self.sessions or not self.sessions[user_id].active:
            return "⚠️ Сессия не активна. Используйте кнопку 'Активировать AI' для начала."

        session = self.sessions[user_id]
        session.update_activity()

        # Проверяем бюджет
        if session.total_cost_usd > AIAssistantConfig.MONTHLY_BUDGET_USD:
            logger.error(f"[AITradingAssistant] Budget exceeded for user {user_id}: ${session.total_cost_usd:.2f}")
            return f"⚠️ Месячный бюджет превышен (${session.total_cost_usd:.2f} / ${AIAssistantConfig.MONTHLY_BUDGET_USD:.2f})"

        # Обновляем контекст бота (если доступен)
        if self.context_collector:
            try:
                context = self.context_collector.collect_full_context()
                context_text = self.context_collector.format_context_for_prompt(context)

                # Обновляем системное сообщение с новым контекстом
                if session.message_history and session.message_history[0]["role"] == "system":
                    session.message_history[0] = {
                        "role": "system",
                        "content": f"{self.SYSTEM_PROMPT}\n\n{context_text}",
                        "timestamp": datetime.utcnow().isoformat()
                    }
            except Exception as e:
                logger.warning(f"[AITradingAssistant] Failed to update context: {e}")

        # Добавляем сообщение пользователя
        session.add_message("user", message)

        # Отправляем запрос к Claude
        try:
            logger.info(f"[AITradingAssistant] Sending message to Claude for user {user_id}")
            start_time = time.time()

            # Формируем сообщения для API
            messages = session.get_conversation_history()

            # Системное сообщение отдельно для prompt caching
            system_message = messages[0]["content"] if messages and messages[0].get("role") == "system" else self.SYSTEM_PROMPT
            user_messages = [msg for msg in messages if msg.get("role") != "system"]

            response = self.client.messages.create(
                model=AIAssistantConfig.MODEL_ID,
                max_tokens=AIAssistantConfig.MAX_TOKENS,
                temperature=AIAssistantConfig.MODEL_TEMPERATURE,
                system=[
                    {
                        "type": "text",
                        "text": system_message,
                        "cache_control": {"type": "ephemeral"}  # Кэшируем системный промпт
                    }
                ],
                messages=user_messages
            )

            elapsed = time.time() - start_time
            logger.info(f"[AITradingAssistant] Response received in {elapsed:.2f}s")

            # Извлекаем ответ
            assistant_message = response.content[0].text

            # Обновляем статистику
            usage = response.usage
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            cached_tokens = getattr(usage, 'cache_read_input_tokens', 0)

            session.update_stats(input_tokens, output_tokens, cached_tokens)

            # Глобальная статистика
            self.stats["total_messages"] += 1
            self.stats["total_input_tokens"] += input_tokens
            self.stats["total_output_tokens"] += output_tokens
            self.stats["total_cached_tokens"] += cached_tokens

            # Стоимость
            cost_input = (input_tokens / 1_000_000) * AIAssistantConfig.COST_PER_MILLION_INPUT_TOKENS
            cost_output = (output_tokens / 1_000_000) * AIAssistantConfig.COST_PER_MILLION_OUTPUT_TOKENS
            cost_cache = (cached_tokens / 1_000_000) * AIAssistantConfig.COST_PER_MILLION_CACHE_READ
            total_cost = cost_input + cost_output - cost_cache

            self.stats["total_cost_usd"] += total_cost

            logger.info(
                f"[AITradingAssistant] Usage: in={input_tokens}, out={output_tokens}, "
                f"cached={cached_tokens}, cost=${total_cost:.4f}"
            )

            # Добавляем ответ в историю
            session.add_message("assistant", assistant_message)

            # Сохраняем
            self._save_sessions()
            self._save_stats()

            return assistant_message

        except Exception as e:
            logger.error(f"[AITradingAssistant] Error during chat: {e}", exc_info=True)
            return f"❌ Ошибка при обращении к AI: {str(e)[:100]}"

    def get_session_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Получить статистику сессии пользователя

        Args:
            user_id: Telegram user ID

        Returns:
            Словарь со статистикой или None
        """
        if user_id not in self.sessions:
            return None

        session = self.sessions[user_id]

        return {
            "active": session.active,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "total_messages": session.total_messages,
            "total_input_tokens": session.total_input_tokens,
            "total_output_tokens": session.total_output_tokens,
            "total_cached_tokens": session.total_cached_tokens,
            "total_cost_usd": session.total_cost_usd,
            "cache_efficiency_pct": (session.total_cached_tokens / session.total_input_tokens * 100)
                                   if session.total_input_tokens > 0 else 0
        }

    def get_ai_status_report(self, user_id: str) -> str:
        """
        Генерация AI-анализа статуса бота (замена /status)

        Args:
            user_id: Telegram user ID

        Returns:
            Подробный AI-анализ состояния бота
        """
        if not self.context_collector:
            return "⚠️ Context collector not available"

        # Собираем контекст
        try:
            context = self.context_collector.collect_full_context()
        except Exception as e:
            logger.error(f"[AITradingAssistant] Failed to collect context: {e}")
            return f"❌ Ошибка сбора контекста: {str(e)[:100]}"

        # Формируем специальный промпт для статуса
        status_prompt = f"""Проанализируй текущее состояние бота и предоставь краткий отчет (максимум 10 предложений):

1. Открытые позиции: количество, качество, риски
2. Капитал: изменения, тренд, проблемы
3. Производительность: win rate, PnL, сравнение с целевыми показателями
4. Проблемы: если есть, укажи что не так
5. Рекомендации: что можно улучшить

Будь конкретным и используй цифры из контекста."""

        # Если сессия не активна, создаем временную
        temp_session = user_id not in self.sessions or not self.sessions[user_id].active

        if temp_session:
            self.activate_session(user_id, "temp")

        try:
            response = self.chat(user_id, status_prompt)
            return response
        finally:
            if temp_session:
                self.deactivate_session(user_id)

    def get_global_stats(self) -> str:
        """Получить глобальную статистику использования AI"""
        return f"""📊 <b>AI Assistant Statistics</b>

Sessions: {self.stats['total_sessions']}
Messages: {self.stats['total_messages']}
Input tokens: {self.stats['total_input_tokens']:,}
Output tokens: {self.stats['total_output_tokens']:,}
Cached tokens: {self.stats['total_cached_tokens']:,}
Cache efficiency: {(self.stats['total_cached_tokens'] / self.stats['total_input_tokens'] * 100) if self.stats['total_input_tokens'] > 0 else 0:.1f}%
Total cost: ${self.stats['total_cost_usd']:.2f}
Budget: ${AIAssistantConfig.MONTHLY_BUDGET_USD:.2f}"""


# ============================================================================
# MAIN (для тестирования)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("AI Trading Assistant module loaded successfully!")
    print(f"Model: {AIAssistantConfig.MODEL_ID}")
    print(f"Monthly budget: ${AIAssistantConfig.MONTHLY_BUDGET_USD}")
