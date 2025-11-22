#!/usr/bin/env python3
"""
AI Assistant Tools - Инструменты для поиска данных бота

Позволяет AI ассистенту запрашивать данные по требованию вместо
постоянной отправки всего контекста (экономия токенов).

Доступные инструменты:

Positions & Trading:
1. get_open_positions - список открытых позиций
2. get_position_details - детали конкретной позиции
3. get_closed_positions - история закрытых позиций
4. search_positions - поиск позиций по критериям
5. get_bot_status - общий статус бота

ML Models & Learning:
6. get_ml_models_stats - статистика всех ML моделей
7. get_expert_details - детали конкретного эксперта
8. get_expert_performance - производительность экспертов

Market Data:
9. get_market_overview - обзор рынка (волатильность, топ позиции)
10. get_symbol_metrics - метрики для конкретного символа
11. get_active_symbols - список активных символов
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ============================================================================
# TOOL DEFINITIONS (для Claude API)
# ============================================================================

TOOLS = [
    {
        "name": "get_open_positions",
        "description": "Получить список всех открытых позиций с данными на момент входа (entry price, p_up, EV, ATR, current PnL)",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_details": {
                    "type": "boolean",
                    "description": "Включить полные детали (entry_snapshot, signals)",
                    "default": False
                }
            }
        }
    },
    {
        "name": "get_position_details",
        "description": "Получить детальную информацию о конкретной позиции (entry_snapshot, индикаторы, причина открытия, рыночные условия)",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Символ позиции (например, BTCUSDT)"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_closed_positions",
        "description": "Получить историю закрытых позиций с результатами",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Количество последних позиций",
                    "default": 10
                },
                "result_filter": {
                    "type": "string",
                    "description": "Фильтр по результату: 'WIN', 'LOSS', или 'ALL'",
                    "enum": ["WIN", "LOSS", "ALL"],
                    "default": "ALL"
                }
            }
        }
    },
    {
        "name": "search_positions",
        "description": "Поиск позиций по различным критериям (символ, дата, статус, результат)",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol_pattern": {
                    "type": "string",
                    "description": "Паттерн символа (например, 'BTC*', 'ETH*')"
                },
                "status": {
                    "type": "string",
                    "description": "Статус позиции",
                    "enum": ["open", "closed", "all"]
                },
                "date_from": {
                    "type": "string",
                    "description": "Дата начала поиска (ISO format: 2025-11-17)"
                },
                "date_to": {
                    "type": "string",
                    "description": "Дата окончания поиска (ISO format)"
                }
            }
        }
    },
    {
        "name": "get_bot_status",
        "description": "Получить общий статус бота (баланс, equity, количество позиций, winrate, статистика моделей)",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_ml_models_stats",
        "description": "Получить статистику всех ML моделей (accuracy, samples, training stats, online learning)",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_diversity": {
                    "type": "boolean",
                    "description": "Включить статистику diversity monitor",
                    "default": True
                }
            }
        }
    },
    {
        "name": "get_expert_details",
        "description": "Получить детальную информацию о конкретном ML эксперте (XGBoost, RandomForest, NeuralNet, AdaptiveRF, META)",
        "input_schema": {
            "type": "object",
            "properties": {
                "expert_name": {
                    "type": "string",
                    "description": "Название эксперта: 'xgb', 'rf', 'arf', 'nn', 'meta'",
                    "enum": ["xgb", "rf", "arf", "nn", "meta"]
                }
            },
            "required": ["expert_name"]
        }
    },
    {
        "name": "get_expert_performance",
        "description": "Получить статистику производительности экспертов (winrate, wins/losses по каждому эксперту)",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_market_overview",
        "description": "Получить обзор рынка: активные символы, общая волатильность, фаза рынка, топ прибыльные/убыточные позиции",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_prices": {
                    "type": "boolean",
                    "description": "Включить текущие цены активных символов",
                    "default": True
                }
            }
        }
    },
    {
        "name": "get_symbol_metrics",
        "description": "Получить детальные метрики для конкретного символа (цена, ATR, волатильность, momentum)",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Символ (например, BTCUSDT)"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_active_symbols",
        "description": "Получить список всех активных символов бота с их статусами (есть ли позиции, в watchlist, etc.)",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]


# ============================================================================
# TOOL EXECUTOR
# ============================================================================

class AIAssistantTools:
    """Исполнитель инструментов для AI ассистента"""

    def __init__(self, bot_instance):
        """
        Args:
            bot_instance: Экземпляр TradingBot с доступом к позициям и состоянию
        """
        self.bot = bot_instance
        logger.info("[AIAssistantTools] Initialized with bot instance")

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Выполняет инструмент и возвращает результат в текстовом формате

        Args:
            tool_name: Название инструмента
            tool_input: Параметры инструмента

        Returns:
            str: Результат выполнения (JSON-строка или текст)
        """
        try:
            logger.info(f"[AIAssistantTools] Executing tool: {tool_name} with input: {tool_input}")

            if tool_name == "get_open_positions":
                return self._get_open_positions(**tool_input)
            elif tool_name == "get_position_details":
                return self._get_position_details(**tool_input)
            elif tool_name == "get_closed_positions":
                return self._get_closed_positions(**tool_input)
            elif tool_name == "search_positions":
                return self._search_positions(**tool_input)
            elif tool_name == "get_bot_status":
                return self._get_bot_status()
            elif tool_name == "get_ml_models_stats":
                return self._get_ml_models_stats(**tool_input)
            elif tool_name == "get_expert_details":
                return self._get_expert_details(**tool_input)
            elif tool_name == "get_expert_performance":
                return self._get_expert_performance()
            elif tool_name == "get_market_overview":
                return self._get_market_overview(**tool_input)
            elif tool_name == "get_symbol_metrics":
                return self._get_symbol_metrics(**tool_input)
            elif tool_name == "get_active_symbols":
                return self._get_active_symbols()
            else:
                return f"❌ Unknown tool: {tool_name}"

        except Exception as e:
            logger.error(f"[AIAssistantTools] Error executing {tool_name}: {e}", exc_info=True)
            return f"❌ Error executing {tool_name}: {str(e)}"

    # ========================================================================
    # TOOL IMPLEMENTATIONS
    # ========================================================================

    def _get_open_positions(self, include_details: bool = False) -> str:
        """Получить открытые позиции"""
        import json

        try:
            open_positions = self.bot.position_manager.get_all_open()

            if not open_positions:
                return "📊 Нет открытых позиций"

            result = []
            for pos in open_positions:
                # Базовая информация
                ticker = self.bot.exchange.get_ticker(pos.symbol)
                current_price = ticker['lastPrice']
                entry_price = pos.entry_price

                if pos.direction == 'LONG':
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                else:  # SHORT
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100

                pos_data = {
                    "symbol": pos.symbol,
                    "direction": pos.direction,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "entry_time": pos.entry_time.isoformat() if hasattr(pos.entry_time, 'isoformat') else str(pos.entry_time),
                    "tp_price": pos.tp_price,
                    "sl_price": pos.sl_price,
                    "current_pnl_pct": round(pnl_pct, 2),
                    "amount": pos.amount,
                    "position_value": pos.position_value,
                    "atr_value": getattr(pos, 'atr_value', None)
                }

                # Данные из entry_snapshot
                if include_details and hasattr(pos, 'entry_snapshot') and pos.entry_snapshot:
                    snapshot = pos.entry_snapshot
                    pos_data["p_up"] = snapshot.get("predictions", {}).get("p_up")
                    pos_data["ev"] = snapshot.get("ev")
                    pos_data["expert_predictions"] = snapshot.get("predictions", {})

                result.append(pos_data)

            return json.dumps(result, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _get_position_details(self, symbol: str) -> str:
        """Получить детали позиции"""
        import json

        try:
            # Ищем в открытых
            open_positions = self.bot.position_manager.get_all_open()
            position = next((p for p in open_positions if p.symbol == symbol), None)

            if not position:
                # Ищем в закрытых
                closed_positions = self.bot.position_manager.get_all_closed()
                position = next((p for p in closed_positions if p.symbol == symbol), None)

            if not position:
                return f"❌ Позиция {symbol} не найдена"

            # Полная информация
            details = {
                "symbol": position.symbol,
                "direction": position.direction,
                "status": position.status,
                "entry_price": position.entry_price,
                "entry_time": position.entry_time.isoformat() if hasattr(position.entry_time, 'isoformat') else str(position.entry_time),
                "tp_price": position.tp_price,
                "sl_price": position.sl_price,
                "amount": position.amount,
                "position_value": position.position_value,
                "atr_value": getattr(position, 'atr_value', None),
            }

            # Entry snapshot (причина открытия, сигналы, индикаторы)
            if hasattr(position, 'entry_snapshot') and position.entry_snapshot:
                snapshot = position.entry_snapshot
                details["entry_snapshot"] = {
                    "timestamp": snapshot.get("timestamp"),
                    "predictions": snapshot.get("predictions"),
                    "ev": snapshot.get("ev"),
                    "direction": snapshot.get("direction"),
                    "features_count": len(snapshot.get("features_68d", [])),
                    # Фичи не включаем чтобы не засорять (68 значений)
                }

            # Результат (для закрытых)
            if position.status == 'CLOSED':
                details["exit_price"] = position.exit_price
                details["exit_time"] = position.exit_time.isoformat() if hasattr(position.exit_time, 'isoformat') else str(position.exit_time)
                details["exit_reason"] = position.exit_reason
                details["pnl"] = position.pnl
                details["pnl_pct"] = position.pnl_pct

            return json.dumps(details, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _get_closed_positions(self, limit: int = 10, result_filter: str = "ALL") -> str:
        """Получить закрытые позиции"""
        import json

        try:
            closed_positions = self.bot.position_manager.get_all_closed()

            if not closed_positions:
                return "📊 Нет закрытых позиций"

            # Фильтруем по результату
            if result_filter == "WIN":
                closed_positions = [p for p in closed_positions if p.pnl > 0]
            elif result_filter == "LOSS":
                closed_positions = [p for p in closed_positions if p.pnl <= 0]

            # Берем последние N
            closed_positions = closed_positions[-limit:]

            result = []
            for pos in closed_positions:
                pos_data = {
                    "symbol": pos.symbol,
                    "direction": pos.direction,
                    "entry_price": pos.entry_price,
                    "exit_price": pos.exit_price,
                    "entry_time": pos.entry_time.isoformat() if hasattr(pos.entry_time, 'isoformat') else str(pos.entry_time),
                    "exit_time": pos.exit_time.isoformat() if hasattr(pos.exit_time, 'isoformat') else str(pos.exit_time),
                    "exit_reason": pos.exit_reason,
                    "pnl": round(pos.pnl, 2),
                    "pnl_pct": round(pos.pnl_pct, 2),
                    "result": "WIN" if pos.pnl > 0 else "LOSS"
                }

                # Добавляем данные о предсказаниях экспертов из entry_snapshot
                if hasattr(pos, 'entry_snapshot') and pos.entry_snapshot:
                    snapshot = pos.entry_snapshot
                    predictions = snapshot.get("predictions", {})
                    if predictions:
                        # Добавляем предсказания всех экспертов
                        pos_data["expert_predictions"] = {
                            "xgb": predictions.get("p_xgb"),
                            "rf": predictions.get("p_rf"),
                            "arf": predictions.get("p_arf"),
                            "nn": predictions.get("p_nn"),
                            "base": predictions.get("p_base"),
                            "meta": predictions.get("p_meta")
                        }
                        # Также добавляем какой эксперт "победил" (имел самое высокое предсказание)
                        valid_predictions = {k: v for k, v in predictions.items() if v is not None and k.startswith('p_')}
                        if valid_predictions:
                            best_expert = max(valid_predictions.items(), key=lambda x: x[1])
                            pos_data["best_expert"] = best_expert[0].replace('p_', '')  # xgb, rf, etc.

                result.append(pos_data)

            return json.dumps(result, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _search_positions(self, symbol_pattern: Optional[str] = None,
                         status: str = "all",
                         date_from: Optional[str] = None,
                         date_to: Optional[str] = None) -> str:
        """Поиск позиций по критериям"""
        import json
        import fnmatch

        try:
            # Собираем все позиции
            positions = []
            if status in ["open", "all"]:
                positions.extend(self.bot.position_manager.get_all_open())
            if status in ["closed", "all"]:
                positions.extend(self.bot.position_manager.get_all_closed())

            # Фильтр по символу
            if symbol_pattern:
                positions = [p for p in positions if fnmatch.fnmatch(p.symbol, symbol_pattern)]

            # Фильтр по дате
            if date_from:
                date_from_dt = datetime.fromisoformat(date_from)
                positions = [p for p in positions if p.entry_time >= date_from_dt]

            if date_to:
                date_to_dt = datetime.fromisoformat(date_to)
                positions = [p for p in positions if p.entry_time <= date_to_dt]

            if not positions:
                return "📊 Позиции не найдены по заданным критериям"

            # Форматируем результат
            result = []
            for pos in positions:
                pos_data = {
                    "symbol": pos.symbol,
                    "direction": pos.direction,
                    "status": pos.status,
                    "entry_time": pos.entry_time.isoformat() if hasattr(pos.entry_time, 'isoformat') else str(pos.entry_time),
                    "entry_price": pos.entry_price,
                }

                if pos.status == 'CLOSED':
                    pos_data["pnl_pct"] = round(pos.pnl_pct, 2)
                    pos_data["result"] = "WIN" if pos.pnl > 0 else "LOSS"

                result.append(pos_data)

            return json.dumps(result, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _get_bot_status(self) -> str:
        """Получить общий статус бота"""
        import json

        try:
            # Баланс и equity
            balance = self.bot.exchange.get_balance() if hasattr(self.bot.exchange, 'get_balance') else "N/A"
            equity = self.bot.exchange.get_equity() if hasattr(self.bot.exchange, 'get_equity') else "N/A"

            # Позиции
            open_positions = self.bot.position_manager.get_all_open()
            closed_positions = self.bot.position_manager.get_all_closed()

            # Статистика
            wins = sum(1 for p in closed_positions if p.pnl > 0)
            losses = sum(1 for p in closed_positions if p.pnl <= 0)
            total_closed = len(closed_positions)
            winrate = (wins / total_closed * 100) if total_closed > 0 else 0

            status = {
                "timestamp": datetime.now().isoformat(),
                "balance": balance,
                "equity": equity,
                "open_positions_count": len(open_positions),
                "closed_positions_count": total_closed,
                "winrate": round(winrate, 2),
                "wins": wins,
                "losses": losses,
                "paper_mode": getattr(self.bot, 'paper_mode', False)
            }

            return json.dumps(status, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _get_ml_models_stats(self, include_diversity: bool = True) -> str:
        """Получить статистику всех ML моделей"""
        import json

        try:
            stats = {
                "experts": {},
                "meta": {},
                "diversity": {} if include_diversity else None
            }

            # Статистика экспертов
            if hasattr(self.bot, 'experts') and self.bot.experts:
                for name, expert in self.bot.experts.items():
                    if expert is not None:
                        expert_stats = {
                            "trained": expert.is_trained if hasattr(expert, 'is_trained') else True,
                            "samples": expert.train_samples if hasattr(expert, 'train_samples') else "N/A",
                            "type": type(expert).__name__
                        }

                        # Дополнительные метрики если доступны
                        if hasattr(expert, 'train_accuracy'):
                            expert_stats["accuracy"] = round(expert.train_accuracy, 4)
                        if hasattr(expert, 'n_estimators'):
                            expert_stats["n_estimators"] = expert.n_estimators

                        stats["experts"][name] = expert_stats

            # Статистика META модели
            if hasattr(self.bot, 'meta') and self.bot.meta is not None:
                meta = self.bot.meta
                stats["meta"] = {
                    "trained": meta.is_trained if hasattr(meta, 'is_trained') else True,
                    "samples": meta.train_samples if hasattr(meta, 'train_samples') else 0,
                    "type": type(meta).__name__
                }

                # Дополнительные метрики для SimplifiedEnsembleMETA
                if hasattr(meta, 'mode'):
                    stats["meta"]["mode"] = meta.mode  # SHADOW/ACTIVE
                if hasattr(meta, 'status'):
                    status = meta.status()
                    stats["meta"]["wr_shadow"] = status.get("wr_shadow", "N/A")
                    stats["meta"]["wr_active"] = status.get("wr_active", "N/A")

                if hasattr(meta, 'train_accuracy'):
                    stats["meta"]["accuracy"] = round(meta.train_accuracy, 4)

            # Diversity Monitor
            if include_diversity and hasattr(self.bot, 'diversity_monitor'):
                monitor = self.bot.diversity_monitor
                stats["diversity"] = {
                    "enabled": True,
                    "window_size": getattr(monitor, 'window_size', "N/A"),
                    "current_metrics": getattr(monitor, 'current_metrics', {})
                }

            return json.dumps(stats, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _get_expert_details(self, expert_name: str) -> str:
        """Получить детальную информацию об эксперте"""
        import json

        try:
            expert = None

            # Поиск эксперта
            if expert_name == "meta":
                expert = getattr(self.bot, 'meta', None)
            elif hasattr(self.bot, 'experts') and self.bot.experts:
                expert = self.bot.experts.get(expert_name)

            if expert is None:
                return f"❌ Эксперт '{expert_name}' не найден"

            # Собираем детали
            details = {
                "name": expert_name,
                "type": type(expert).__name__,
                "trained": expert.is_trained if hasattr(expert, 'is_trained') else True,
                "samples": expert.train_samples if hasattr(expert, 'train_samples') else 0
            }

            # Специфичные метрики для META
            if expert_name == "meta":
                if hasattr(expert, 'mode'):
                    details["mode"] = expert.mode
                if hasattr(expert, 'status'):
                    status = expert.status()
                    details["wr_shadow"] = status.get("wr_shadow", "N/A")
                    details["wr_active"] = status.get("wr_active", "N/A")
                    details["n_shadow"] = status.get("n_shadow", "0")
                    details["n_active"] = status.get("n_active", "0")

            # Специфичные для типа метрики
            if hasattr(expert, 'train_accuracy'):
                details["train_accuracy"] = round(expert.train_accuracy, 4)

            if hasattr(expert, 'n_estimators'):
                details["n_estimators"] = expert.n_estimators

            if hasattr(expert, 'max_depth'):
                details["max_depth"] = expert.max_depth

            if hasattr(expert, 'learning_rate'):
                details["learning_rate"] = expert.learning_rate

            if hasattr(expert, 'random_state'):
                details["random_state"] = expert.random_state

            # Параметры NeuralNet
            if hasattr(expert, 'n_epochs'):
                details["n_epochs"] = expert.n_epochs

            if hasattr(expert, 'hidden_dim'):
                details["hidden_dim"] = expert.hidden_dim

            # Adaptive RF специфичные
            if hasattr(expert, 'n_forests'):
                details["n_forests"] = expert.n_forests

            return json.dumps(details, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _get_expert_performance(self) -> str:
        """Получить статистику производительности экспертов"""
        import json

        try:
            if not hasattr(self.bot, 'expert_stats'):
                return "❌ Статистика экспертов недоступна"

            performance = {}

            for expert_name, stats in self.bot.expert_stats.items():
                total = stats.get('total', 0)
                wins = stats.get('wins', 0)
                losses = stats.get('losses', 0)
                winrate = (wins / total * 100) if total > 0 else 0

                performance[expert_name] = {
                    "wins": wins,
                    "losses": losses,
                    "total": total,
                    "winrate": round(winrate, 2)
                }

            return json.dumps(performance, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _get_market_overview(self, include_prices: bool = True) -> str:
        """Получить обзор рынка"""
        import json

        try:
            overview = {
                "timestamp": datetime.now().isoformat(),
                "active_symbols": [],
                "market_stats": {},
                "top_positions": {}
            }

            # Получаем все открытые позиции
            open_positions = self.bot.position_manager.get_all_open()

            if open_positions:
                # Активные символы
                active_symbols = list(set(p.symbol for p in open_positions))
                overview["active_symbols"] = active_symbols

                # Статистика по позициям
                total_positions = len(open_positions)
                long_count = sum(1 for p in open_positions if p.direction == 'LONG')
                short_count = sum(1 for p in open_positions if p.direction == 'SHORT')

                # Средний PnL
                pnl_values = []
                for pos in open_positions:
                    ticker = self.bot.exchange.get_ticker(pos.symbol)
                    current_price = ticker['lastPrice']
                    entry_price = pos.entry_price
                    if pos.direction == 'LONG':
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    else:
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    pnl_values.append(pnl_pct)

                avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0

                # Средний ATR (волатильность)
                atr_values = [getattr(p, 'atr_value', None) for p in open_positions]
                atr_values = [a for a in atr_values if a is not None]
                avg_atr = sum(atr_values) / len(atr_values) if atr_values else None

                overview["market_stats"] = {
                    "total_positions": total_positions,
                    "long_positions": long_count,
                    "short_positions": short_count,
                    "avg_pnl_percent": round(avg_pnl, 2),
                    "avg_atr": round(avg_atr, 4) if avg_atr else None
                }

                # Топ прибыльные и убыточные
                positions_with_pnl = list(zip(open_positions, pnl_values))
                positions_with_pnl.sort(key=lambda x: x[1], reverse=True)

                if len(positions_with_pnl) >= 3:
                    top_profitable = [
                        {"symbol": p.symbol, "pnl_pct": round(pnl, 2)}
                        for p, pnl in positions_with_pnl[:3]
                    ]
                    top_losing = [
                        {"symbol": p.symbol, "pnl_pct": round(pnl, 2)}
                        for p, pnl in positions_with_pnl[-3:]
                    ]
                else:
                    top_profitable = [
                        {"symbol": p.symbol, "pnl_pct": round(pnl, 2)}
                        for p, pnl in positions_with_pnl
                    ]
                    top_losing = []

                overview["top_positions"] = {
                    "most_profitable": top_profitable,
                    "most_losing": top_losing
                }

                # Текущие цены (опционально)
                if include_prices:
                    prices = {}
                    for symbol in active_symbols:
                        try:
                            ticker = self.bot.exchange.get_ticker(symbol)
                            prices[symbol] = ticker['lastPrice']
                        except:
                            prices[symbol] = None
                    overview["current_prices"] = prices

            else:
                overview["market_stats"] = {
                    "total_positions": 0,
                    "message": "Нет открытых позиций"
                }

            return json.dumps(overview, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _get_symbol_metrics(self, symbol: str) -> str:
        """Получить метрики для конкретного символа"""
        import json

        try:
            metrics = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }

            # Текущая цена
            try:
                ticker = self.bot.exchange.get_ticker(symbol)
                current_price = ticker['lastPrice']
                metrics["current_price"] = current_price
            except Exception as e:
                metrics["current_price"] = None
                metrics["price_error"] = str(e)

            # Проверяем есть ли позиция по этому символу
            open_positions = self.bot.position_manager.get_all_open()
            position = next((p for p in open_positions if p.symbol == symbol), None)

            if position:
                metrics["has_position"] = True
                metrics["position_direction"] = position.direction
                metrics["entry_price"] = position.entry_price

                # ATR
                atr = getattr(position, 'atr_value', None)
                if atr:
                    metrics["atr"] = atr
                    # Волатильность как процент от цены
                    metrics["volatility_pct"] = round((atr / position.entry_price) * 100, 2)

                # Текущий PnL
                if current_price:
                    if position.direction == 'LONG':
                        pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                    else:
                        pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100
                    metrics["current_pnl_pct"] = round(pnl_pct, 2)

                # Entry snapshot (если есть)
                if hasattr(position, 'entry_snapshot') and position.entry_snapshot:
                    snapshot = position.entry_snapshot
                    metrics["entry_conditions"] = {
                        "p_up": snapshot.get("predictions", {}).get("p_up"),
                        "ev": snapshot.get("ev"),
                        "timestamp": snapshot.get("timestamp")
                    }
            else:
                metrics["has_position"] = False

                # Пытаемся получить ATR из недавних данных (если доступно)
                # Примечание: в текущей архитектуре ATR вычисляется только при открытии позиции
                # Здесь мы можем только сказать что позиции нет
                metrics["message"] = f"Нет открытой позиции по {symbol}"

            return json.dumps(metrics, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _get_active_symbols(self) -> str:
        """Получить список активных символов"""
        import json

        try:
            # Символы из открытых позиций
            open_positions = self.bot.position_manager.get_all_open()
            open_symbols = set(p.symbol for p in open_positions)

            # Символы из watchlist (если есть)
            watchlist_symbols = set()
            if hasattr(self.bot, 'symbol_watchlist') and self.bot.symbol_watchlist:
                watchlist_symbols = set(self.bot.symbol_watchlist)

            # Все уникальные символы
            all_symbols = open_symbols | watchlist_symbols

            symbols_data = []
            for symbol in sorted(all_symbols):
                symbol_info = {
                    "symbol": symbol,
                    "has_open_position": symbol in open_symbols,
                    "in_watchlist": symbol in watchlist_symbols
                }

                # Если есть позиция - добавляем детали
                if symbol in open_symbols:
                    position = next(p for p in open_positions if p.symbol == symbol)
                    symbol_info["direction"] = position.direction

                    # Текущий PnL
                    try:
                        ticker = self.bot.exchange.get_ticker(symbol)
                        current_price = ticker['lastPrice']
                        entry_price = position.entry_price
                        if position.direction == 'LONG':
                            pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        else:
                            pnl_pct = ((entry_price - current_price) / entry_price) * 100
                        symbol_info["current_pnl_pct"] = round(pnl_pct, 2)
                    except:
                        symbol_info["current_pnl_pct"] = None

                symbols_data.append(symbol_info)

            result = {
                "timestamp": datetime.now().isoformat(),
                "total_symbols": len(symbols_data),
                "symbols_with_positions": len(open_symbols),
                "watchlist_symbols": len(watchlist_symbols),
                "symbols": symbols_data
            }

            return json.dumps(result, indent=2, ensure_ascii=False)

        except Exception as e:
            return f"❌ Error: {str(e)}"
