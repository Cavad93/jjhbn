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
from typing import Dict, Optional, List
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Отправка уведомлений в Telegram через Bot API
    """

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        """
        Инициализация Telegram notifier

        Args:
            bot_token: Токен Telegram бота (получить у @BotFather)
            chat_id: ID чата для отправки сообщений
            enabled: Включить/выключить уведомления
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        if not enabled:
            logger.info("Telegram notifications disabled")
        elif not bot_token or not chat_id:
            logger.warning("Telegram bot_token or chat_id not configured")
            self.enabled = False
        else:
            logger.info(f"Telegram notifier initialized (chat_id: {chat_id})")

    def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Отправка сообщения в Telegram

        Args:
            text: Текст сообщения
            parse_mode: Формат текста (HTML, Markdown, MarkdownV2)

        Returns:
            True если успешно отправлено, False иначе
        """
        if not self.enabled:
            return False

        try:
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }

            response = requests.post(self.api_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.debug(f"Telegram message sent: {text[:50]}...")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            return False

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

        # Направление эмодзи
        direction_emoji = "🟢" if direction == "LONG" else "🔴"

        # R/R ratio
        if direction == "LONG":
            rr_ratio = (tp_price - entry_price) / (entry_price - sl_price) if sl_price > 0 else 0
        else:
            rr_ratio = (entry_price - tp_price) / (sl_price - entry_price) if sl_price > 0 else 0

        text = f"""
{direction_emoji} <b>ОТКРЫТА ПОЗИЦИЯ</b>

<b>{symbol}</b> {direction} x{leverage}
Вход: ${entry_price:.4f}
Размер: ${position_size:.2f}

📊 TP: ${tp_price:.4f} ({self._calc_percent(entry_price, tp_price, direction):+.2f}%)
🛡 SL: ${sl_price:.4f} ({self._calc_percent(entry_price, sl_price, direction):+.2f}%)
📈 R/R: {rr_ratio:.2f}

🎯 P(up): {p_up:.1%}
💰 EV: {ev:+.2%}
        """.strip()

        # Добавляем информацию о резервном фонде (если есть)
        reserve_fund = position_data.get('reserve_fund')
        trading_capital = position_data.get('trading_capital')
        total_equity = position_data.get('total_equity')

        if reserve_fund is not None and trading_capital is not None:
            text += f"\n\n<b>💼 Капитал:</b>"
            text += f"\n  • Торговый: ${trading_capital:,.2f}"
            text += f"\n  • Резервный: ${reserve_fund:,.2f}"
            if total_equity is not None:
                text += f"\n  • Общий: ${total_equity:,.2f}"
                reserve_pct = (reserve_fund / total_equity * 100) if total_equity > 0 else 0
                text += f"\n  • Резерв: {reserve_pct:.1f}% от капитала"

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
        """
        symbol = position_data.get('symbol', 'UNKNOWN')
        direction = position_data.get('direction', 'UNKNOWN')
        entry_price = position_data.get('entry_price', 0.0)
        exit_price = position_data.get('exit_price', 0.0)
        pnl = position_data.get('pnl', 0.0)
        pnl_percent = position_data.get('pnl_percent', 0.0)
        exit_reason = position_data.get('exit_reason', 'UNKNOWN')
        holding_time = position_data.get('holding_time', 'N/A')

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

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """.strip()

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
                p_up = opp.get('p_up', 0.0)
                text += f"\n{i}. {symbol}: EV={ev:+.2%}, P(up)={p_up:.1%}"

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
        text = """
🧪 <b>ТЕСТОВОЕ СООБЩЕНИЕ</b>

Telegram уведомления работают корректно!

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """.strip()

        return self._send_message(text)
