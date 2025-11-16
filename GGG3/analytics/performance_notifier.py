#!/usr/bin/env python3
"""
Performance Notifier

Умные уведомления о производительности стратегии с:
- Красивым форматированием
- Автоматическими рекомендациями
- Milestone notifications
- Status change notifications
"""

from typing import Optional, Dict
from datetime import datetime, timedelta
from .performance_tracker import PerformanceMetrics


class PerformanceNotifier:
    """
    Отправляет умные уведомления о производительности

    Уведомления отправляются при:
    1. Достижении milestones (50, 100, 200, 500 сделок)
    2. Смене статуса (all_conditions_met)
    3. Периодически (раз в час если были новые сделки)
    """

    MILESTONES = [50, 100, 200, 500, 1000]

    def __init__(self, telegram_notifier=None):
        """
        Args:
            telegram_notifier: Объект для отправки Telegram уведомлений
        """
        self.telegram = telegram_notifier

        # Отслеживание состояния
        self.last_trade_count = 0
        self.last_status = None
        self.last_notification_time = None
        self.notified_milestones = set()

    def should_notify(
        self,
        metrics: PerformanceMetrics,
        force: bool = False
    ) -> tuple:
        """
        Определяет нужно ли отправлять уведомление

        Args:
            metrics: Метрики производительности
            force: Принудительная отправка

        Returns:
            (should_notify, reason)
        """
        if force:
            return (True, 'manual')

        # 1. Проверяем milestones
        if metrics.total_trades in self.MILESTONES and metrics.total_trades not in self.notified_milestones:
            self.notified_milestones.add(metrics.total_trades)
            return (True, f'milestone_{metrics.total_trades}')

        # 2. Проверяем смену статуса
        if self.last_status is not None and self.last_status != metrics.all_conditions_met:
            return (True, 'status_change')

        # 3. Проверяем периодические уведомления (раз в час если были новые сделки)
        now = datetime.now()
        hour_passed = (
            self.last_notification_time is None or
            (now - self.last_notification_time) >= timedelta(hours=1)
        )
        new_trades = metrics.total_trades > self.last_trade_count

        if hour_passed and new_trades:
            return (True, 'periodic')

        return (False, '')

    def notify(
        self,
        metrics: PerformanceMetrics,
        window_name: str = 'short',
        reason: str = ''
    ) -> None:
        """
        Отправляет уведомление о производительности

        Args:
            metrics: Метрики производительности
            window_name: Название окна анализа
            reason: Причина уведомления
        """
        # Формируем сообщение
        message = self._format_message(metrics, window_name, reason)

        # Отправляем в Telegram
        if self.telegram:
            try:
                self.telegram.send_message(message, parse_mode='HTML')
            except Exception as e:
                print(f"Error sending Telegram notification: {e}")

        # Обновляем состояние
        self.last_trade_count = metrics.total_trades
        self.last_status = metrics.all_conditions_met
        self.last_notification_time = datetime.now()

        # Выводим в консоль
        print(f"\n{message}\n")

    def _format_message(
        self,
        metrics: PerformanceMetrics,
        window_name: str,
        reason: str
    ) -> str:
        """
        Форматирует красивое сообщение с метриками

        Args:
            metrics: Метрики
            window_name: Название окна
            reason: Причина уведомления

        Returns:
            Отформатированное сообщение (HTML для Telegram)
        """
        # Заголовок с причиной
        header = self._format_header(metrics.total_trades, window_name, reason)

        # Базовые метрики
        core_metrics = self._format_core_metrics(metrics)

        # Дополнительные метрики качества
        quality_metrics = self._format_quality_metrics(metrics)

        # Прогрессия капитала
        capital_info = self._format_capital_info(metrics)

        # Win/Loss breakdown
        breakdown = self._format_breakdown(metrics)

        # Статус и условия
        status = self._format_status(metrics)

        # Автоматические рекомендации
        recommendations = self._format_recommendations(metrics)

        # Footer
        footer = self._format_footer(metrics)

        # Собираем все вместе
        message = f"{header}\n\n{core_metrics}\n\n{quality_metrics}\n\n{capital_info}\n\n{breakdown}\n\n{status}"

        if recommendations:
            message += f"\n\n{recommendations}"

        message += f"\n\n{footer}"

        return message

    def _format_header(self, total_trades: int, window_name: str, reason: str) -> str:
        """Форматирует заголовок"""
        window_labels = {
            'short': 'Last 50 trades',
            'medium': 'Last 100 trades',
            'long': 'Last 200 trades',
            'all': 'All trades since start'
        }

        reason_emoji = {
            'milestone_50': '🎯',
            'milestone_100': '💯',
            'milestone_200': '🚀',
            'milestone_500': '⭐',
            'milestone_1000': '🏆',
            'status_change': '🔄',
            'periodic': '📊',
            'manual': '👤'
        }

        emoji = reason_emoji.get(reason, '📊')
        window_label = window_labels.get(window_name, f'{total_trades} trades')

        header = f"{emoji} <b>PERFORMANCE REPORT</b> ({window_label})\n"
        header += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        return header

    def _format_core_metrics(self, m: PerformanceMetrics) -> str:
        """Форматирует базовые метрики"""
        # Проверки условий
        wr_check = "✅" if m.conditions.get('win_rate_above_required') else "❌"
        geo_check = "✅" if m.conditions.get('positive_geometric_growth') else "❌"
        ci_check = "✅" if m.conditions.get('ci_above_zero') else "❌"

        text = "<b>🎯 CORE METRICS:</b>\n"
        text += f"{wr_check} Win Rate: <b>{m.win_rate*100:.1f}%</b> (Required: {m.required_win_rate*100:.1f}%)\n"
        text += f"{geo_check} Geometric Growth: <b>{m.geometric_mean_return:+.2f}%</b> per trade\n"
        text += f"{ci_check} 90% CI: <b>[{m.ci_lower*100:.1f}%, {m.ci_upper*100:.1f}%]</b>"

        return text

    def _format_quality_metrics(self, m: PerformanceMetrics) -> str:
        """Форматирует метрики качества"""
        # Оценка Profit Factor
        if m.profit_factor >= 2.5:
            pf_rating = "🌟 Excellent"
        elif m.profit_factor >= 2.0:
            pf_rating = "✅ Great"
        elif m.profit_factor >= 1.5:
            pf_rating = "👍 Good"
        elif m.profit_factor >= 1.0:
            pf_rating = "⚠️ Marginal"
        else:
            pf_rating = "❌ Poor"

        pf_display = f"{m.profit_factor:.2f}" if m.profit_factor != float('inf') else "∞"

        text = "<b>📈 QUALITY METRICS:</b>\n"
        text += f"• Profit Factor: <b>{pf_display}</b> {pf_rating}\n"
        text += f"• Expectancy: <b>{m.expectancy:+.2f}%</b> per trade\n"
        text += f"• Avg Win: <b>+{m.avg_win:.2f}%</b> | Avg Loss: <b>-{m.avg_loss:.2f}%</b>\n"
        text += f"• Realized R:R: <b>{m.avg_win_loss_ratio:.2f}:1</b>\n"
        text += f"• Max Drawdown: <b>{m.max_drawdown_pct:.1f}%</b>"

        return text

    def _format_capital_info(self, m: PerformanceMetrics) -> str:
        """Форматирует информацию о капитале"""
        text = "<b>💰 CAPITAL PROGRESSION:</b>\n"
        text += f"${m.starting_capital:.0f} → ${m.ending_capital:.0f} (<b>{m.total_return_pct:+.1f}%</b>)\n"
        text += f"Compound Growth Rate: <b>{m.compound_growth_rate:+.2f}%</b> per trade"

        return text

    def _format_breakdown(self, m: PerformanceMetrics) -> str:
        """Форматирует Win/Loss breakdown"""
        text = "<b>📊 WIN/LOSS BREAKDOWN:</b>\n"
        text += f"🟢 Wins: <b>{m.win_count}</b> ({m.win_rate*100:.0f}%) | "
        text += f"🔴 Losses: <b>{m.loss_count}</b> ({(1-m.win_rate)*100:.0f}%)\n"
        text += f"Longest Win Streak: <b>{m.max_consecutive_wins}</b> | "
        text += f"Longest Loss Streak: <b>{m.max_consecutive_losses}</b>\n"

        # Текущая серия
        if m.current_streak > 0:
            text += f"Current Streak: <b>{m.current_streak} wins</b> 🔥"
        elif m.current_streak < 0:
            text += f"Current Streak: <b>{abs(m.current_streak)} losses</b> ⚠️"
        else:
            text += f"Current Streak: <b>None</b>"

        return text

    def _format_status(self, m: PerformanceMetrics) -> str:
        """Форматирует статус стратегии"""
        if m.all_conditions_met:
            icon = "🟢"
            status_text = "ALL CONDITIONS MET"
            description = "Strategy is performing above expectations!"
        else:
            icon = "🟡"
            status_text = "CONDITIONS NOT MET"
            failed = [k for k, v in m.conditions.items() if not v]
            description = f"Failed: {', '.join(failed)}"

        text = f"<b>STATUS: {icon} {status_text}</b>\n"
        text += description

        return text

    def _format_recommendations(self, m: PerformanceMetrics) -> str:
        """Генерирует автоматические рекомендации"""
        recommendations = []

        # Низкий винрейт
        if m.win_rate < m.required_win_rate and m.profit_factor < 1.5:
            recommendations.append(
                "⚠️ <b>Низкий винрейт:</b> Рассмотрите ужесточение критериев входа или "
                "улучшение фильтрации сигналов"
            )

        # Малый R:R
        if m.avg_win_loss_ratio < 1.5 and m.win_rate < 0.6:
            recommendations.append(
                "⚠️ <b>Малый R:R:</b> Улучшите exits - позволяйте прибыли расти дольше "
                "или сокращайте убытки быстрее"
            )

        # Высокая просадка
        if m.max_drawdown_pct > 15:
            recommendations.append(
                "⚠️ <b>Высокая просадка:</b> Снизьте risk per trade или улучшите "
                "управление позициями в drawdown"
            )

        # Низкий Profit Factor
        if m.profit_factor < 1.5 and m.total_trades >= 50:
            recommendations.append(
                "⚠️ <b>Низкий Profit Factor:</b> Стратегия малоприбыльна - "
                "необходима оптимизация параметров"
            )

        # Отрицательный Expectancy
        if m.expectancy < 0:
            recommendations.append(
                "🚨 <b>КРИТИЧНО:</b> Отрицательный Expectancy - стратегия убыточна! "
                "Остановите торговлю и пересмотрите подход"
            )

        # Длинная серия поражений
        if m.current_streak < -5:
            recommendations.append(
                "⚠️ <b>Длинная серия поражений:</b> Возможно изменились рыночные условия. "
                "Рассмотрите временную паузу для переоценки"
            )

        # Хорошая производительность
        if m.all_conditions_met and m.profit_factor >= 2.0 and m.max_drawdown_pct < 10:
            recommendations.append(
                "✅ <b>Отличная производительность:</b> Стратегия работает в рамках ожиданий. "
                "Продолжайте следить за метриками"
            )

        if not recommendations:
            return ""

        text = "<b>💡 RECOMMENDATIONS:</b>\n"
        text += "\n".join(recommendations)

        return text

    def _format_footer(self, m: PerformanceMetrics) -> str:
        """Форматирует footer с временной меткой"""
        text = f"<i>Last updated: {m.last_updated.strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        return text

    def notify_all_windows(self, all_metrics: Dict[str, PerformanceMetrics]) -> None:
        """
        Отправляет сводный отчет по всем окнам

        Args:
            all_metrics: Словарь {window_name: metrics}
        """
        if not all_metrics:
            return

        # Используем короткое окно для определения причины
        short_metrics = all_metrics.get('short')
        if not short_metrics:
            return

        should_notify, reason = self.should_notify(short_metrics)

        if not should_notify:
            return

        # Отправляем уведомление для основного окна
        self.notify(short_metrics, 'short', reason)

        # Если это milestone, также отправляем summary по всем окнам
        if 'milestone' in reason and len(all_metrics) > 1:
            summary = self._format_multi_window_summary(all_metrics)
            if self.telegram:
                try:
                    self.telegram.send_message(summary, parse_mode='HTML')
                except Exception as e:
                    print(f"Error sending summary: {e}")

    def _format_multi_window_summary(self, all_metrics: Dict[str, PerformanceMetrics]) -> str:
        """Форматирует сводку по всем окнам"""
        text = "📊 <b>MULTI-WINDOW SUMMARY</b>\n"
        text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        window_labels = {
            'short': '50 trades',
            'medium': '100 trades',
            'long': '200 trades',
            'all': 'All trades'
        }

        for window_name in ['short', 'medium', 'long', 'all']:
            metrics = all_metrics.get(window_name)
            if not metrics:
                continue

            label = window_labels.get(window_name, window_name)
            status_icon = "🟢" if metrics.all_conditions_met else "🟡"

            text += f"<b>{label}:</b> {status_icon}\n"
            text += f"  Win Rate: {metrics.win_rate*100:.1f}% | "
            text += f"Geo Growth: {metrics.geometric_mean_return:+.2f}% | "
            text += f"PF: {metrics.profit_factor:.2f}\n\n"

        return text
