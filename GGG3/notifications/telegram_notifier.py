#!/usr/bin/env python3
"""
Telegram Notifier - отправка уведомлений о работе бота в Telegram

Типы уведомлений:
- Старт/стоп бота
- Открытие/закрытие позиций
- Статистика торговли
- Обучение моделей
- Защитные режимы (black swan, night mode, trailing stop)
- Ребалансировка портфеля
"""

import logging
import requests
from typing import Dict, Optional, List, Callable
from datetime import datetime
import traceback
import threading
import time

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Отправка уведомлений в Telegram через Bot API
    """

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True, ai_assistant=None,
                 group_chat_id: str = "", allowed_user_id: str = "939056216"):
        """
        Инициализация Telegram notifier

        Args:
            bot_token: Токен Telegram бота (получить у @BotFather)
            chat_id: ID личного чата для отправки сообщений и команд
            enabled: Включить/выключить уведомления
            ai_assistant: AI Assistant instance (опционально)
            group_chat_id: ID группового чата для отправки уведомлений (опционально)
            allowed_user_id: ID пользователя, который может отправлять команды (по умолчанию: 939056216)
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.group_chat_id = group_chat_id
        self.allowed_user_id = allowed_user_id
        self.enabled = enabled
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.get_updates_url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
        self.answer_callback_url = f"https://api.telegram.org/bot{bot_token}/answerCallbackQuery"
        self.edit_message_url = f"https://api.telegram.org/bot{bot_token}/editMessageText"

        # Для обработки команд
        self.last_update_id = 0
        self.command_handlers: Dict[str, Callable] = {}
        self.command_thread: Optional[threading.Thread] = None
        self.command_thread_running = False

        # AI Assistant
        self.ai_assistant = ai_assistant

        # Close all positions handler (будет установлен из main)
        self.closeall_handler: Optional[Callable] = None

        if not enabled:
            logger.info("Telegram notifications disabled")
        elif not bot_token or not chat_id:
            logger.warning("Telegram bot_token or chat_id not configured")
            self.enabled = False
        else:
            chats_info = f"chat_id: {chat_id}"
            if group_chat_id:
                chats_info += f", group_chat_id: {group_chat_id}"
            logger.info(f"Telegram notifier initialized ({chats_info}, allowed_user_id: {allowed_user_id})")

    def _send_message(self, text: str, parse_mode: str = "HTML", reply_markup: Optional[Dict] = None,
                      target_chat_id: Optional[str] = None) -> bool:
        """
        Отправка сообщения в Telegram

        Args:
            text: Текст сообщения
            parse_mode: Формат текста (HTML, Markdown, MarkdownV2, None)
            reply_markup: Inline keyboard markup (опционально)
            target_chat_id: ID чата для отправки (если None, отправляется во все настроенные чаты)

        Returns:
            True если успешно отправлено хотя бы в один чат, False иначе
        """
        if not self.enabled:
            return False

        # Определяем список чатов для отправки
        if target_chat_id:
            # Отправляем только в указанный чат
            chat_ids = [target_chat_id]
        else:
            # Отправляем во все настроенные чаты
            chat_ids = [self.chat_id]
            if self.group_chat_id:
                chat_ids.append(self.group_chat_id)

        success = False
        for chat_id in chat_ids:
            try:
                payload = {
                    "chat_id": chat_id,
                    "text": text,
                    "disable_web_page_preview": True
                }

                # Добавляем parse_mode только если он указан
                if parse_mode:
                    payload["parse_mode"] = parse_mode

                # Добавляем кнопки если указаны
                if reply_markup:
                    payload["reply_markup"] = reply_markup

                response = requests.post(self.api_url, json=payload, timeout=10)
                response.raise_for_status()

                logger.debug(f"Telegram message sent to {chat_id}: {text[:50]}...")
                success = True

            except requests.exceptions.HTTPError as e:
                # При ошибке 400 логируем содержимое сообщения для отладки
                if e.response.status_code == 400:
                    logger.error(f"Telegram API 400 Bad Request for chat {chat_id}")
                    logger.error(f"Message length: {len(text)} chars")
                    logger.error(f"First 500 chars: {text[:500]}")
                    logger.error(f"Last 500 chars: {text[-500:]}")
                    try:
                        error_detail = e.response.json()
                        logger.error(f"Telegram error detail: {error_detail}")
                    except:
                        pass
                logger.error(f"Failed to send Telegram message to {chat_id}: {e}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to send Telegram message to {chat_id}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error sending Telegram message to {chat_id}: {e}")

        return success

    # ========================================================================
    # BOT STATUS NOTIFICATIONS
    # ========================================================================

    def notify_bot_started(self, paper_mode: bool, initial_capital: float, max_positions: int):
        """Уведомление о старте бота"""
        mode = "📄 PAPER TRADING" if paper_mode else "⚠️ LIVE TRADING"
        text = f"""
🤖 <b>БОТ ЗАПУЩЕН</b>

Режим: {mode}
Капитал: ${initial_capital:.2f}
Макс. позиций: {max_positions}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """.strip()
        self._send_message(text)

    def notify_bot_stopped(self, reason: str = ""):
        """Уведомление об остановке бота"""
        text = f"""
🛑 <b>БОТ ОСТАНОВЛЕН</b>

{f"Причина: {reason}" if reason else ""}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """.strip()
        self._send_message(text)

    # ========================================================================
    # POSITION NOTIFICATIONS
    # ========================================================================

    def notify_position_opened(self, position_data: Dict):
        """
        Уведомление об открытии позиции

        Args:
            position_data: Словарь с данными позиции:
                - symbol: Символ
                - direction: LONG/SHORT
                - entry_price: Цена входа
                - position_size: Размер позиции (USDT)
                - leverage: Плечо
                - tp_price: Take Profit
                - sl_price: Stop Loss
                - p_up: Вероятность роста (META)
                - ev: Expected Value
                - predictions: Предсказания экспертов (опционально)
                - reserve_fund: Резервный фонд (опционально)
                - trading_capital: Торговый капитал (опционально)
                - total_equity: Общий капитал (опционально)
        """
        symbol = position_data.get('symbol', 'UNKNOWN')
        direction = position_data.get('direction', 'UNKNOWN')
        entry_price = position_data.get('entry_price', 0.0)
        position_size = position_data.get('position_size', 0.0)
        leverage = position_data.get('leverage', 1)
        tp_price = position_data.get('tp_price', 0.0)
        sl_price = position_data.get('sl_price', 0.0)
        p_up = position_data.get('p_up', 0.0)
        ev = position_data.get('ev', 0.0)
        atr = position_data.get('atr', 0.0)  # ATR в долях (0.02 = 2%)

        # Направление эмодзи
        direction_emoji = "🟢" if direction == "LONG" else "🔴"

        # R/R ratio
        if direction == "LONG":
            rr_ratio = (tp_price - entry_price) / (entry_price - sl_price) if sl_price > 0 else 0
        else:
            rr_ratio = (entry_price - tp_price) / (sl_price - entry_price) if sl_price > 0 else 0

        # ИСПРАВЛЕНИЕ: EV приходит в множителях ATR, конвертируем в проценты цены
        # EV в множителях ATR (например, 1.57) × ATR в % (например, 2%) = EV в % цены (3.14%)
        ev_percent = ev * atr if atr > 0 else ev

        text = f"""
{direction_emoji} <b>ОТКРЫТА ПОЗИЦИЯ</b>

<b>{symbol}</b> {direction} x{leverage}
Вход: ${entry_price:.4f}
Размер: ${position_size:.2f}

📊 TP: ${tp_price:.4f} ({self._calc_percent(entry_price, tp_price, direction):+.2f}%)
🛡 SL: ${sl_price:.4f} ({self._calc_percent(entry_price, sl_price, direction):+.2f}%)
📈 R/R: {rr_ratio:.2f}

🎯 P(up): {p_up:.1%}
💰 EV: {ev_percent:+.2%}
        """.strip()

        # Добавляем информацию о резервном фонде и капитале (если есть)
        reserve_fund = position_data.get('reserve_fund')
        trading_capital = position_data.get('trading_capital')
        locked_in_positions = position_data.get('locked_in_positions')
        total_equity = position_data.get('total_equity')
        period_changes = position_data.get('period_changes')

        if total_equity is not None:
            text += f"\n\n<b>💼 Капитал:</b>"
            text += f"\n  • Свободный: ${trading_capital:,.2f}" if trading_capital is not None else ""
            text += f"\n  • В позициях: ${locked_in_positions:,.2f}" if locked_in_positions is not None else ""
            if reserve_fund is not None and reserve_fund > 0:
                text += f"\n  • Резервный: ${reserve_fund:,.2f}"
            text += f"\n  • <b>Общий: ${total_equity:,.2f}</b>"
            if reserve_fund is not None and total_equity > 0:
                reserve_pct = (reserve_fund / total_equity * 100)
                text += f"\n  • Резерв: {reserve_pct:.1f}% от капитала"

            # Добавляем изменения за периоды
            if period_changes:
                text += f"\n\n<b>📊 Изменения:</b>"
                for period_name, data in period_changes.items():
                    if data.get('available'):
                        sign = "+" if data['absolute'] >= 0 else ""
                        text += f"\n  • {period_name}: {sign}${data['absolute']:.2f} ({sign}{data['percent']:.2f}%)"

        # Добавляем предсказания экспертов если есть
        predictions = position_data.get('predictions')
        if predictions:
            pred_text = "\n\n<b>Эксперты:</b>"
            for expert, prob in predictions.items():
                pred_text += f"\n  • {expert.upper()}: {prob:.1%}"
            text += pred_text

        text += f"\n\n⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"

        self._send_message(text)

    def notify_position_closed(self, position_data: Dict):
        """
        Уведомление о закрытии позиции

        Args:
            position_data: Словарь с данными закрытой позиции:
                - symbol: Символ
                - direction: LONG/SHORT
                - entry_price: Цена входа
                - exit_price: Цена выхода
                - position_size: Размер позиции
                - pnl: PnL в USDT
                - pnl_percent: PnL в процентах
                - exit_reason: Причина закрытия (TP/SL/timeout/manual)
                - holding_time: Время удержания позиции
                - win_rates: Win rates экспертов (опционально)
        """
        symbol = position_data.get('symbol', 'UNKNOWN')
        direction = position_data.get('direction', 'UNKNOWN')
        entry_price = position_data.get('entry_price', 0.0)
        exit_price = position_data.get('exit_price', 0.0)
        pnl = position_data.get('pnl', 0.0)
        pnl_percent = position_data.get('pnl_percent', 0.0)
        exit_reason = position_data.get('exit_reason', 'UNKNOWN')
        holding_time = position_data.get('holding_time', 'N/A')
        win_rates = position_data.get('win_rates', {})

        # Эмодзи в зависимости от результата
        if pnl > 0:
            result_emoji = "✅"
        elif pnl < 0:
            result_emoji = "❌"
        else:
            result_emoji = "⚪"

        # Причина закрытия
        reason_map = {
            'TP': '🎯 Take Profit',
            'SL': '🛡 Stop Loss',
            'trailing_sl': '📈 Trailing Stop',
            'timeout': '⏱ Timeout',
            'manual': '👤 Manual',
            'rebalance': '⚖️ Rebalance',
            'black_swan': '⚠️ Black Swan',
            'night_mode': '🌙 Night Mode'
        }
        reason_text = reason_map.get(exit_reason, exit_reason)

        text = f"""
{result_emoji} <b>ЗАКРЫТА ПОЗИЦИЯ</b>

<b>{symbol}</b> {direction}
Вход: ${entry_price:.4f}
Выход: ${exit_price:.4f}

💵 PnL: ${pnl:+.2f} ({pnl_percent:+.2f}%)
📝 Причина: {reason_text}
⏱ Время: {holding_time}
        """.strip()

        # ✅ Добавляем Win Rates экспертов если доступны
        if win_rates:
            text += "\n\n<b>📊 Win Rates:</b>"
            # Порядок отображения: BASE, XGB, RF, ARF, NN, META
            expert_order = [
                ('BASE', 'base'),
                ('XGB', 'xgb'),
                ('RF', 'rf'),
                ('ARF', 'arf'),
                ('NN', 'nn'),
                ('META', 'meta')
            ]
            for display_name, expert_key in expert_order:
                if expert_key in win_rates:
                    wr_data = win_rates[expert_key]
                    wr = wr_data['win_rate']
                    total = wr_data['total']
                    if total > 0:
                        text += f"\n  • {display_name}: {wr:.1%} ({total} сделок)"

        text += f"\n\n⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"

        self._send_message(text)

    # ========================================================================
    # STATISTICS NOTIFICATIONS
    # ========================================================================

    def notify_daily_stats(self, stats: Dict):
        """
        Уведомление с ежедневной статистикой

        Args:
            stats: Статистика торговли:
                - total_trades: Всего сделок
                - win_rate: Win rate
                - total_pnl: Общий PnL
                - avg_win: Средний выигрыш
                - avg_loss: Средний убыток
                - best_trade: Лучшая сделка
                - worst_trade: Худшая сделка
                - current_balance: Текущий баланс
                - open_positions: Количество открытых позиций
        """
        total_trades = stats.get('total_trades', 0)
        win_rate = stats.get('win_rate', 0.0)
        total_pnl = stats.get('total_pnl', 0.0)
        avg_win = stats.get('avg_win', 0.0)
        avg_loss = stats.get('avg_loss', 0.0)
        best_trade = stats.get('best_trade', 0.0)
        worst_trade = stats.get('worst_trade', 0.0)
        current_balance = stats.get('current_balance', 0.0)
        open_positions = stats.get('open_positions', 0)

        text = f"""
📊 <b>ЕЖЕДНЕВНАЯ СТАТИСТИКА</b>

💼 Баланс: ${current_balance:.2f}
📈 Открытых позиций: {open_positions}

🎲 Всего сделок: {total_trades}
✅ Win Rate: {win_rate:.1%}
💰 Total PnL: ${total_pnl:+.2f}

📊 Средний выигрыш: ${avg_win:+.2f}
📉 Средний убыток: ${avg_loss:+.2f}

🏆 Лучшая сделка: ${best_trade:+.2f}
💔 Худшая сделка: ${worst_trade:+.2f}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """.strip()

        self._send_message(text)

    def notify_rebalance_stats(self, stats: Dict):
        """
        Уведомление о ребалансировке портфеля

        Args:
            stats: Статистика ребалансировки:
                - closed_positions: Количество закрытых позиций
                - opened_positions: Количество новых позиций
                - total_positions: Всего открытых позиций
                - top_opportunities: Топ возможностей по EV
        """
        closed = stats.get('closed_positions', 0)
        opened = stats.get('opened_positions', 0)
        total = stats.get('total_positions', 0)
        opportunities = stats.get('top_opportunities', [])

        text = f"""
⚖️ <b>РЕБАЛАНСИРОВКА ПОРТФЕЛЯ</b>

Закрыто позиций: {closed}
Открыто позиций: {opened}
Всего открытых: {total}
        """.strip()

        # Добавляем топ возможностей
        if opportunities:
            text += "\n\n<b>Топ возможностей по EV:</b>"
            for i, opp in enumerate(opportunities[:5], 1):
                symbol = opp.get('symbol', 'UNKNOWN')
                ev = opp.get('ev', 0.0)
                atr = opp.get('atr', 0.0)
                p_up = opp.get('p_up', 0.0)
                # ИСПРАВЛЕНИЕ: EV в множителях ATR, конвертируем в проценты
                ev_percent = ev * atr if atr > 0 else ev
                text += f"\n{i}. {symbol}: EV={ev_percent:+.2%}, P(up)={p_up:.1%}"

        text += f"\n\n⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"

        self._send_message(text)

    # ========================================================================
    # RISK MANAGEMENT NOTIFICATIONS
    # ========================================================================

    def notify_trailing_stop_activated(self, symbol: str, direction: str,
                                       current_price: float, new_sl: float, profit_pct: float):
        """Уведомление об активации trailing stop"""
        text = f"""
📈 <b>TRAILING STOP АКТИВИРОВАН</b>

<b>{symbol}</b> {direction}
Текущая цена: ${current_price:.4f}
Новый SL: ${new_sl:.4f}
Прибыль: +{profit_pct:.2f}%

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """.strip()
        self._send_message(text)

    def notify_black_swan_triggered(self, btc_drop: float, positions_closed: int):
        """Уведомление о срабатывании Black Swan protection"""
        text = f"""
⚠️ <b>BLACK SWAN EVENT!</b>

BTC упал на {btc_drop:.2f}% за 5 минут!
Все позиции экстренно закрыты.

Закрыто позиций: {positions_closed}
Новые входы заблокированы на 2 часа.

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """.strip()
        self._send_message(text)

    def notify_night_mode_activated(self, max_positions: int):
        """Уведомление об активации night mode"""
        text = f"""
🌙 <b>NIGHT MODE АКТИВИРОВАН</b>

UTC 23:00 - 07:00
Макс. позиций: {max_positions}
Порог входа повышен на +2%

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """.strip()
        self._send_message(text)

    def notify_night_mode_deactivated(self):
        """Уведомление о деактивации night mode"""
        text = f"""
☀️ <b>NIGHT MODE ДЕАКТИВИРОВАН</b>

Возврат к нормальному режиму работы.
Макс. позиций: 10

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """.strip()
        self._send_message(text)

    # ========================================================================
    # LEARNING NOTIFICATIONS
    # ========================================================================

    def notify_online_learning(self, learning_data: Dict):
        """
        Уведомление об online learning

        Args:
            learning_data: Данные об обучении:
                - symbol: Символ
                - outcome: Результат (win/loss)
                - experts_updated: Список обновленных экспертов
                - meta_updated: Обновлена ли META
        """
        symbol = learning_data.get('symbol', 'UNKNOWN')
        outcome = learning_data.get('outcome', 'UNKNOWN')
        experts = learning_data.get('experts_updated', [])
        meta_updated = learning_data.get('meta_updated', False)

        outcome_emoji = "✅" if outcome == 'win' else "❌"

        text = f"""
🧠 <b>ONLINE LEARNING</b>

{outcome_emoji} {symbol}: {outcome}

Обновлены эксперты:
        """.strip()

        for expert in experts:
            text += f"\n  • {expert.upper()}"

        if meta_updated:
            text += "\n  • META"

        text += f"\n\n⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"

        self._send_message(text)

    def notify_model_retrained(self, model_name: str, performance: Dict):
        """
        Уведомление о переобучении модели

        Args:
            model_name: Название модели
            performance: Метрики качества:
                - accuracy: Точность
                - win_rate: Win rate
                - calibration_error: Ошибка калибровки
        """
        accuracy = performance.get('accuracy', 0.0)
        win_rate = performance.get('win_rate', 0.0)
        calib_error = performance.get('calibration_error', 0.0)

        text = f"""
🔄 <b>МОДЕЛЬ ПЕРЕОБУЧЕНА</b>

Модель: <b>{model_name}</b>

Метрики:
  • Accuracy: {accuracy:.1%}
  • Win Rate: {win_rate:.1%}
  • Calibration Error: {calib_error:.3f}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """.strip()

        self._send_message(text)

    # ========================================================================
    # ERROR NOTIFICATIONS
    # ========================================================================

    def notify_error(self, error_msg: str, traceback_str: Optional[str] = None):
        """Уведомление об ошибке"""
        text = f"""
🚨 <b>ОШИБКА</b>

{error_msg}
        """.strip()

        if traceback_str:
            # Ограничиваем длину traceback
            if len(traceback_str) > 500:
                traceback_str = traceback_str[:500] + "..."
            text += f"\n\n<code>{traceback_str}</code>"

        text += f"\n\n⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"

        self._send_message(text)

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _calc_percent(self, entry: float, target: float, direction: str) -> float:
        """Вычисляет процентное изменение с учетом направления"""
        if direction == "LONG":
            return ((target - entry) / entry) * 100
        else:
            return ((entry - target) / entry) * 100

    def test_connection(self) -> bool:
        """Тестовая отправка сообщения"""
        text = f"""
🧪 <b>ТЕСТОВОЕ СООБЩЕНИЕ</b>

Telegram уведомления работают корректно!

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """.strip()

        return self._send_message(text)

    def send_message(self, text: str) -> bool:
        """
        Публичный метод для отправки сообщения

        Args:
            text: Текст сообщения

        Returns:
            True если успешно, False иначе
        """
        return self._send_message(text)

    # ========================================================================
    # COMMAND HANDLER (для обработки команд типа /status)
    # ========================================================================

    def register_command_handler(self, command: str, handler: Callable):
        """
        Регистрация обработчика команды

        Args:
            command: Команда (например, 'status' для /status)
            handler: Функция-обработчик, принимает () и возвращает str
        """
        self.command_handlers[command] = handler
        logger.info(f"Registered command handler: /{command}")

    def start_command_listener(self):
        """Запуск потока для прослушивания команд"""
        if not self.enabled:
            logger.warning("Cannot start command listener: notifications disabled")
            return

        if self.command_thread_running:
            logger.warning("Command listener already running")
            return

        self.command_thread_running = True
        self.command_thread = threading.Thread(target=self._command_listener_loop, daemon=True)
        self.command_thread.start()
        logger.info("Command listener started")

    def stop_command_listener(self):
        """Остановка потока прослушивания команд"""
        if self.command_thread_running:
            self.command_thread_running = False
            logger.info("Command listener stopped")

    def _command_listener_loop(self):
        """Основной цикл прослушивания команд"""
        logger.info("Command listener loop started")

        while self.command_thread_running:
            try:
                # Получаем новые обновления
                params = {
                    "offset": self.last_update_id + 1,
                    "timeout": 30,
                    "allowed_updates": ["message", "callback_query"]  # Добавляем callback_query для кнопок
                }

                response = requests.get(self.get_updates_url, params=params, timeout=35)
                response.raise_for_status()

                data = response.json()

                if not data.get("ok"):
                    logger.error(f"Failed to get updates: {data}")
                    time.sleep(5)
                    continue

                updates = data.get("result", [])

                for update in updates:
                    self.last_update_id = max(self.last_update_id, update["update_id"])

                    # ═══════════════════════════════════════════════════════════════
                    # ОБРАБОТКА CALLBACK QUERIES (кнопки)
                    # ═══════════════════════════════════════════════════════════════
                    if "callback_query" in update:
                        self._handle_callback_query(update["callback_query"])
                        continue

                    # ═══════════════════════════════════════════════════════════════
                    # ОБРАБОТКА ТЕКСТОВЫХ СООБЩЕНИЙ
                    # ═══════════════════════════════════════════════════════════════
                    message = update.get("message", {})
                    chat_id = str(message.get("chat", {}).get("id", ""))
                    user_id = str(message.get("from", {}).get("id", ""))

                    # Проверяем, что сообщение от разрешенного пользователя
                    if user_id != self.allowed_user_id:
                        logger.debug(f"Ignoring message from unauthorized user {user_id}")
                        continue

                    text = message.get("text", "")
                    if not text:
                        continue

                    # Проверяем команды
                    if text.startswith("/"):
                        # Парсим команду
                        command = text.split()[0][1:]  # Убираем '/'

                        # Вызываем обработчик
                        if command in self.command_handlers:
                            try:
                                response_text = self.command_handlers[command]()
                                # Отправляем сообщение только если есть текст
                                # (обработчик может сам отправить сообщение и вернуть пустую строку)
                                # Ответ отправляем в тот же чат, откуда пришла команда
                                if response_text:
                                    self._send_message(response_text, target_chat_id=chat_id)
                            except Exception as e:
                                logger.error(f"Error handling command /{command}: {e}", exc_info=True)
                                self._send_message(f"❌ Ошибка при выполнении команды /{command}: {str(e)}",
                                                 target_chat_id=chat_id)
                        else:
                            logger.debug(f"Unknown command: /{command}")
                    else:
                        # Не команда - проверяем активна ли AI сессия
                        self._handle_ai_message(text, user_id, chat_id)

            except requests.exceptions.Timeout:
                # Это нормально для long polling
                continue
            except Exception as e:
                logger.error(f"Error in command listener loop: {e}", exc_info=True)
                time.sleep(5)

        logger.info("Command listener loop stopped")

    # ========================================================================
    # AI ASSISTANT INTEGRATION
    # ========================================================================

    def set_ai_assistant(self, ai_assistant):
        """
        Установить AI Assistant instance

        Args:
            ai_assistant: AITradingAssistant instance
        """
        self.ai_assistant = ai_assistant
        logger.info("[TelegramNotifier] AI Assistant connected")

    def send_ai_control_panel(self):
        """Отправка панели управления AI ассистентом"""
        if not self.ai_assistant:
            logger.warning("[TelegramNotifier] AI Assistant not configured")
            return False

        # Проверяем активность сессии
        user_id = self.chat_id  # Используем chat_id как user_id
        is_active = self.ai_assistant.is_session_active(user_id)

        # Формируем текст
        if is_active:
            text = "🤖 <b>AI АССИСТЕНТ АКТИВЕН</b>\n\nМожете задавать вопросы о боте в чате."

            # Показываем статистику сессии
            stats = self.ai_assistant.get_session_stats(user_id)
            if stats:
                text += f"\n\n📊 Статистика:"
                text += f"\n  • Сообщений: {stats['total_messages']}"
                text += f"\n  • Стоимость: ${stats['total_cost_usd']:.4f}"
                text += f"\n  • Cache эфф.: {stats['cache_efficiency_pct']:.1f}%"
        else:
            text = "🤖 <b>AI АССИСТЕНТ</b>\n\nАктивируйте AI для интерактивного чата с ботом."

        # Формируем кнопки
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "💤 Деактивировать" if is_active else "⚡ Активировать",
                     "callback_data": "ai_deactivate" if is_active else "ai_activate"}
                ],
                [
                    {"text": "📊 AI Status", "callback_data": "ai_status"}
                ]
            ]
        }

        return self._send_message(text, reply_markup=keyboard)

    def send_closeall_confirmation(self, open_positions_count: int, total_value: float):
        """
        Отправка панели подтверждения закрытия всех позиций

        Args:
            open_positions_count: Количество открытых позиций
            total_value: Общая стоимость позиций
        """
        if open_positions_count == 0:
            text = "ℹ️ <b>НЕТ ОТКРЫТЫХ ПОЗИЦИЙ</b>\n\nВсе позиции уже закрыты."
            return self._send_message(text)

        text = f"""
⚠️ <b>ЗАКРЫТИЕ ВСЕХ ПОЗИЦИЙ</b>

Вы уверены что хотите закрыть ВСЕ открытые позиции?

📊 Будет закрыто: <b>{open_positions_count}</b> позиций
💰 Общий объём: <b>${total_value:.2f}</b>

⚠️ Это действие нельзя отменить!
        """.strip()

        # Формируем кнопки подтверждения
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "✅ Да, закрыть ВСЕ", "callback_data": "closeall_confirm"},
                    {"text": "❌ Отмена", "callback_data": "closeall_cancel"}
                ]
            ]
        }

        return self._send_message(text, reply_markup=keyboard)

    def _handle_callback_query(self, callback_query: Dict):
        """
        Обработка нажатия на inline кнопку

        Args:
            callback_query: Callback query от Telegram
        """
        callback_id = callback_query.get("id")
        data = callback_query.get("data", "")
        user_id = str(callback_query.get("from", {}).get("id", ""))
        # Получаем chat_id из сообщения, к которому относится callback
        chat_id = str(callback_query.get("message", {}).get("chat", {}).get("id", ""))

        logger.info(f"[TelegramNotifier] Callback query: {data} from user {user_id} in chat {chat_id}")

        # Проверяем, что callback от разрешенного пользователя
        if user_id != self.allowed_user_id:
            logger.warning(f"[TelegramNotifier] Ignoring callback from unauthorized user {user_id}")
            self._answer_callback_query(callback_id, "❌ Доступ запрещен")
            return

        try:
            # Обработка AI команд
            if data == "ai_activate":
                self._handle_ai_activate(user_id, chat_id)
            elif data == "ai_deactivate":
                self._handle_ai_deactivate(user_id)
            elif data == "ai_status":
                self._handle_ai_status(user_id)
            # Обработка closeall команд
            elif data == "closeall_confirm":
                self._handle_closeall_confirm()
            elif data == "closeall_cancel":
                self._handle_closeall_cancel()
            else:
                logger.warning(f"[TelegramNotifier] Unknown callback data: {data}")

            # Подтверждаем обработку callback
            self._answer_callback_query(callback_id)

        except Exception as e:
            logger.error(f"[TelegramNotifier] Error handling callback: {e}", exc_info=True)
            self._answer_callback_query(callback_id, f"Ошибка: {str(e)[:50]}")

    def _answer_callback_query(self, callback_id: str, text: Optional[str] = None):
        """Ответ на callback query"""
        try:
            payload = {"callback_query_id": callback_id}
            if text:
                payload["text"] = text

            requests.post(self.answer_callback_url, json=payload, timeout=5)
        except Exception as e:
            logger.warning(f"[TelegramNotifier] Failed to answer callback: {e}")

    def _handle_ai_activate(self, user_id: str, chat_id: str):
        """Активация AI сессии"""
        if not self.ai_assistant:
            self._send_message("❌ AI Assistant не настроен. Проверьте ANTHROPIC_API_KEY в .env", target_chat_id=chat_id)
            return

        try:
            session = self.ai_assistant.activate_session(user_id, chat_id)
            logger.info(f"[TelegramNotifier] AI session activated for user {user_id} in chat {chat_id}")

            # Обновляем панель управления (отправляем в тот же чат)
            # self.send_ai_control_panel()  # Эта функция всегда отправляет в личный чат

            # Приветственное сообщение
            self._send_message(
                "✅ <b>AI Ассистент активирован!</b>\n\n"
                "Теперь вы можете задавать вопросы о боте прямо в чате:\n\n"
                "• Почему бот открыл позицию?\n"
                "• Сколько я заработал?\n"
                "• Есть ли проблемы?\n"
                "• Что с капиталом?\n\n"
                "Просто пишите вопросы текстом.",
                target_chat_id=chat_id
            )

        except Exception as e:
            logger.error(f"[TelegramNotifier] Failed to activate AI: {e}", exc_info=True)
            self._send_message(f"❌ Ошибка активации AI: {str(e)[:100]}", target_chat_id=chat_id)

    def _handle_ai_deactivate(self, user_id: str):
        """Деактивация AI сессии"""
        if not self.ai_assistant:
            return

        try:
            self.ai_assistant.deactivate_session(user_id)
            logger.info(f"[TelegramNotifier] AI session deactivated for user {user_id}")

            # Обновляем панель управления
            self.send_ai_control_panel()

            self._send_message("💤 <b>AI Ассистент деактивирован.</b>\n\nДля экономии токенов сессия остановлена.")

        except Exception as e:
            logger.error(f"[TelegramNotifier] Failed to deactivate AI: {e}", exc_info=True)
            self._send_message(f"❌ Ошибка деактивации AI: {str(e)[:100]}")

    def _handle_ai_status(self, user_id: str):
        """Генерация AI Status отчета"""
        if not self.ai_assistant:
            self._send_message("❌ AI Assistant не настроен")
            return

        try:
            # Отправляем уведомление о генерации
            self._send_message("🤖 Генерирую AI-анализ состояния бота...\n\nЭто займет 3-5 секунд.")

            # Получаем AI анализ
            report = self.ai_assistant.get_ai_status_report(user_id)

            # Отправляем отчет
            self._send_message(f"📊 <b>AI STATUS REPORT</b>\n\n{report}")

        except Exception as e:
            logger.error(f"[TelegramNotifier] Failed to generate AI status: {e}", exc_info=True)
            self._send_message(f"❌ Ошибка генерации AI Status: {str(e)[:100]}")

    def _handle_closeall_confirm(self):
        """Обработка подтверждения закрытия всех позиций"""
        if not self.closeall_handler:
            self._send_message("❌ Обработчик closeall не настроен")
            return

        try:
            logger.info("[TelegramNotifier] User confirmed closeall - executing...")
            self._send_message("⏳ <b>ЗАКРЫТИЕ ВСЕХ ПОЗИЦИЙ...</b>\n\nПожалуйста, подождите...")

            # Вызываем обработчик закрытия позиций
            result = self.closeall_handler()

            # Отправляем результат
            if result.get("success"):
                closed_count = result.get("closed_count", 0)
                failed_count = result.get("failed_count", 0)
                total_pnl = result.get("total_pnl", 0.0)
                details = result.get("details", [])

                text = f"""
✅ <b>ВСЕ ПОЗИЦИИ ЗАКРЫТЫ</b>

📊 Закрыто: <b>{closed_count}</b> позиций
💰 Общий PnL: <b>${total_pnl:+.2f}</b>
                """.strip()

                if failed_count > 0:
                    text += f"\n\n⚠️ Не удалось закрыть: {failed_count} позиций"

                if details:
                    text += "\n\n<b>Детали:</b>"
                    for detail in details[:5]:  # Показываем первые 5
                        symbol = detail.get("symbol", "?")
                        pnl = detail.get("pnl", 0.0)
                        text += f"\n  • {symbol}: ${pnl:+.2f}"

                    if len(details) > 5:
                        text += f"\n  ... и ещё {len(details) - 5}"

                self._send_message(text)
            else:
                error_msg = result.get("error", "Unknown error")
                self._send_message(f"❌ <b>ОШИБКА</b>\n\n{error_msg}")

        except Exception as e:
            logger.error(f"[TelegramNotifier] Error executing closeall: {e}", exc_info=True)
            self._send_message(f"❌ <b>ОШИБКА ЗАКРЫТИЯ</b>\n\n{str(e)[:200]}")

    def _handle_closeall_cancel(self):
        """Обработка отмены закрытия всех позиций"""
        logger.info("[TelegramNotifier] User cancelled closeall")
        self._send_message("❌ <b>ОТМЕНЕНО</b>\n\nЗакрытие всех позиций отменено.")

    def _handle_ai_message(self, message: str, user_id: str, chat_id: str):
        """
        Обработка текстового сообщения для AI чата

        Args:
            message: Текст сообщения
            user_id: Telegram user ID
            chat_id: Telegram chat ID (для отправки ответа в правильный чат)
        """
        if not self.ai_assistant:
            return

        # Проверяем активна ли сессия
        if not self.ai_assistant.is_session_active(user_id):
            logger.debug(f"[TelegramNotifier] AI session not active for user {user_id}, ignoring message")
            return

        try:
            logger.info(f"[TelegramNotifier] Processing AI message from user {user_id}: {message[:50]}...")

            # Отправляем в AI
            response = self.ai_assistant.chat(user_id, message)

            # Разбиваем длинный ответ на части (Telegram limit 4096 chars)
            MAX_LENGTH = 4000
            if len(response) <= MAX_LENGTH:
                self._send_message(response, target_chat_id=chat_id)
            else:
                # Разбиваем на части
                parts = []
                while response:
                    if len(response) <= MAX_LENGTH:
                        parts.append(response)
                        break

                    # Ищем последний перенос строки перед лимитом
                    split_pos = response.rfind('\n', 0, MAX_LENGTH)
                    if split_pos == -1:
                        split_pos = MAX_LENGTH

                    parts.append(response[:split_pos])
                    response = response[split_pos:].lstrip()

                # Отправляем по частям
                for i, part in enumerate(parts, 1):
                    if len(parts) > 1:
                        part = f"[{i}/{len(parts)}]\n\n{part}"
                    self._send_message(part, target_chat_id=chat_id)
                    time.sleep(0.5)  # Небольшая пауза между сообщениями

        except Exception as e:
            logger.error(f"[TelegramNotifier] Error processing AI message: {e}", exc_info=True)
            self._send_message(f"❌ Ошибка обработки сообщения: {str(e)[:100]}", target_chat_id=chat_id)

    # ========================================================================
    # DETAILED STATUS REPORT (для команды /status)
    # ========================================================================

    def _safe_format_number(self, value: float, format_str: str = ".2f", default: str = "N/A") -> str:
        """
        Безопасное форматирование числа, обрабатывает inf/nan

        Args:
            value: число для форматирования
            format_str: строка формата (например, ".2f", "+.2f")
            default: значение по умолчанию для inf/nan

        Returns:
            Отформатированная строка
        """
        import math

        if value is None or math.isnan(value) or math.isinf(value):
            return default

        try:
            return f"{value:{format_str}}"
        except (ValueError, TypeError):
            return default

    def _escape_html(self, text: str) -> str:
        """
        Экранирование специальных HTML символов для Telegram

        Args:
            text: текст для экранирования

        Returns:
            Экранированный текст
        """
        if not text:
            return ""

        # Экранируем только символы, которые не являются частью HTML тегов
        # Telegram поддерживает только определенные теги: <b>, <i>, <code>, <pre>, <a>
        text = str(text)
        # Не экранируем, если это уже HTML теги
        return text

    def notify_status_report(self, status_data: Dict) -> bool:
        """
        Отправка подробного отчета о статусе бота

        Args:
            status_data: Словарь с данными статуса:
                - open_positions: List[Dict] - список открытых позиций с текущими ценами и PnL
                - total_unrealized_pnl: float - общий нереализованный PnL
                - total_unrealized_pnl_pct: float - общий нереализованный PnL в %
                - balance: float - текущий баланс
                - locked_in_positions: float - заблокировано в позициях (маржа)
                - free_balance: float - свободный баланс
                - total_equity: float - общий капитал
                - reserve_fund: float - резервный фонд
                - closed_trades_today: int - закрытых сделок сегодня
                - daily_pnl: float - дневной PnL
                - total_trades: int - всего сделок
                - win_rate: float - win rate
                - total_realized_pnl: float - общий реализованный PnL
                - best_trade: float - лучшая сделка
                - worst_trade: float - худшая сделка
                - avg_win: float - средний выигрыш
                - avg_loss: float - средний убыток
                - period_changes: Dict - изменения за периоды (1h, 24h, 7d, 30d)

        Returns:
            True если успешно отправлено
        """
        # Заголовок
        text = "📊 <b>СТАТУС БОТА</b>\n\n"

        # ═══════════════════════════════════════════════════════════════
        # РАЗДЕЛ 1: ОТКРЫТЫЕ ПОЗИЦИИ
        # ═══════════════════════════════════════════════════════════════
        open_positions = status_data.get('open_positions', [])
        total_unrealized_pnl = status_data.get('total_unrealized_pnl', 0.0)
        total_unrealized_pnl_pct = status_data.get('total_unrealized_pnl_pct', 0.0)

        if open_positions:
            text += f"<b>🔓 ОТКРЫТЫЕ ПОЗИЦИИ ({len(open_positions)})</b>\n"

            for i, pos in enumerate(open_positions, 1):
                symbol = pos.get('symbol', 'UNKNOWN')
                direction = pos.get('direction', 'UNKNOWN')
                entry_price = pos.get('entry_price', 0.0)
                current_price = pos.get('current_price', 0.0)
                pnl_usdt = pos.get('pnl_usdt', 0.0)
                pnl_pct = pos.get('pnl_pct', 0.0)
                position_value = pos.get('position_value', 0.0)
                tp_price = pos.get('tp_price', 0.0)
                sl_price = pos.get('sl_price', 0.0)
                duration = pos.get('duration', 'N/A')

                # Эмодзи для направления и результата
                dir_emoji = "🟢" if direction == "LONG" else "🔴"
                pnl_emoji = "✅" if pnl_usdt >= 0 else "❌"

                # Безопасное форматирование чисел
                entry_str = self._safe_format_number(entry_price, ".4f")
                current_str = self._safe_format_number(current_price, ".4f")
                pnl_usdt_str = self._safe_format_number(pnl_usdt, "+.2f")
                pnl_pct_str = self._safe_format_number(pnl_pct, "+.2f")
                pos_value_str = self._safe_format_number(position_value, ".2f")
                tp_str = self._safe_format_number(tp_price, ".4f")
                sl_str = self._safe_format_number(sl_price, ".4f")

                text += f"\n{i}. {dir_emoji} <b>{symbol}</b> {direction}\n"
                text += f"   Вход: ${entry_str} | Сейчас: ${current_str}\n"
                text += f"   {pnl_emoji} PnL: ${pnl_usdt_str} ({pnl_pct_str}%)\n"
                text += f"   Размер: ${pos_value_str} | TP: ${tp_str} | SL: ${sl_str}\n"
                text += f"   Время: {duration}\n"

            # Итого по открытым позициям
            total_emoji = "✅" if total_unrealized_pnl >= 0 else "❌"
            total_pnl_str = self._safe_format_number(total_unrealized_pnl, "+.2f")
            total_pnl_pct_str = self._safe_format_number(total_unrealized_pnl_pct, "+.2f")
            text += f"\n{total_emoji} <b>Итого нереализованный PnL: ${total_pnl_str} ({total_pnl_pct_str}%)</b>\n"
        else:
            text += "<b>🔓 ОТКРЫТЫЕ ПОЗИЦИИ</b>\nНет открытых позиций\n"

        # ═══════════════════════════════════════════════════════════════
        # РАЗДЕЛ 2: КАПИТАЛ
        # ═══════════════════════════════════════════════════════════════
        total_equity = status_data.get('total_equity', 0.0)
        free_balance = status_data.get('free_balance', 0.0)
        locked_in_positions = status_data.get('locked_in_positions', 0.0)
        reserve_fund = status_data.get('reserve_fund', 0.0)

        text += f"\n<b>💼 КАПИТАЛ</b>\n"
        text += f"  • Свободный: ${free_balance:,.2f}\n"
        text += f"  • В позициях (маржа): ${locked_in_positions:,.2f}\n"
        if reserve_fund > 0:
            reserve_pct = (reserve_fund / total_equity * 100) if total_equity > 0 else 0
            text += f"  • Резервный: ${reserve_fund:,.2f} ({reserve_pct:.1f}%)\n"
        text += f"  • <b>Общий капитал: ${total_equity:,.2f}</b>\n"

        # Изменения за периоды
        period_changes = status_data.get('period_changes', {})
        if period_changes:
            text += f"\n<b>📊 ИЗМЕНЕНИЯ КАПИТАЛА</b>\n"
            for period_name, data in period_changes.items():
                if data.get('available'):
                    absolute_val = data['absolute']
                    percent_val = data['percent']
                    sign = "+" if absolute_val >= 0 else ""
                    emoji = "📈" if absolute_val >= 0 else "📉"

                    # Безопасное форматирование
                    abs_str = self._safe_format_number(absolute_val, ".2f")
                    pct_str = self._safe_format_number(percent_val, ".2f")

                    text += f"  {emoji} {period_name}: {sign}${abs_str} ({sign}{pct_str}%)\n"

        # ═══════════════════════════════════════════════════════════════
        # РАЗДЕЛ 3: ТОРГОВАЯ СТАТИСТИКА
        # ═══════════════════════════════════════════════════════════════
        total_trades = status_data.get('total_trades', 0)
        win_rate = status_data.get('win_rate', 0.0)
        total_realized_pnl = status_data.get('total_realized_pnl', 0.0)
        daily_pnl = status_data.get('daily_pnl', 0.0)
        closed_today = status_data.get('closed_trades_today', 0)

        # Безопасное форматирование win rate (уже в процентах от position_manager)
        win_rate_str = self._safe_format_number(win_rate, ".1f", "0.0")
        realized_pnl_str = self._safe_format_number(total_realized_pnl, "+.2f")
        daily_pnl_str = self._safe_format_number(daily_pnl, "+.2f")

        text += f"\n<b>📈 ТОРГОВАЯ СТАТИСТИКА</b>\n"
        text += f"  • Всего сделок: {total_trades}\n"
        text += f"  • Win Rate: {win_rate_str}%\n"
        text += f"  • Реализованный PnL: ${realized_pnl_str}\n"
        text += f"  • Сегодня закрыто: {closed_today} сделок\n"
        text += f"  • Дневной PnL: ${daily_pnl_str}\n"

        # Дополнительная статистика
        avg_win = status_data.get('avg_win', 0.0)
        avg_loss = status_data.get('avg_loss', 0.0)
        best_trade = status_data.get('best_trade', 0.0)
        worst_trade = status_data.get('worst_trade', 0.0)

        if total_trades > 0:
            # Безопасное форматирование статистики
            avg_win_str = self._safe_format_number(avg_win, "+.2f")
            avg_loss_str = self._safe_format_number(avg_loss, "+.2f")
            best_trade_str = self._safe_format_number(best_trade, "+.2f")
            worst_trade_str = self._safe_format_number(worst_trade, "+.2f")

            text += f"\n<b>📊 ДЕТАЛИ</b>\n"
            text += f"  • Средний выигрыш: ${avg_win_str}\n"
            text += f"  • Средний убыток: ${avg_loss_str}\n"
            text += f"  • Лучшая сделка: ${best_trade_str}\n"
            text += f"  • Худшая сделка: ${worst_trade_str}\n"

        # Время
        text += f"\n⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"

        # Проверка длины сообщения (Telegram лимит 4096 символов)
        if len(text) > 4096:
            logger.warning(f"Status message too long ({len(text)} chars), truncating...")
            text = text[:4090] + "\n...(обрезано)"

        # Попытка отправить с HTML форматированием
        success = self._send_message(text)

        # Если не удалось, пробуем без HTML (plain text)
        if not success:
            logger.warning("Retrying status message without HTML formatting...")
            # Удаляем все HTML теги
            import re
            plain_text = re.sub(r'<[^>]+>', '', text)
            return self._send_message(plain_text, parse_mode=None)

        return success
