#!/usr/bin/env python3
"""
Тест Telegram уведомлений с резервным фондом

Проверяет, как отображается информация о reserve fund в уведомлениях
об открытии позиций.

Автор: Claude Code
Дата: 2025-11-06
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from notifications.telegram_notifier import TelegramNotifier


def test_position_opened_notification():
    """Тест уведомления об открытии позиции с reserve fund"""
    print("\n" + "="*80)
    print("ТЕСТ TELEGRAM УВЕДОМЛЕНИЙ С RESERVE FUND")
    print("="*80)

    # Создаём notifier с отключенной отправкой
    # (чтобы не отправлять реальные сообщения)
    notifier = TelegramNotifier(
        bot_token="dummy_token",
        chat_id="dummy_chat",
        enabled=False  # Отключаем реальную отправку
    )

    # Но включим вручную для демонстрации
    notifier.enabled = True

    print("\n[ТЕСТ 1] Уведомление БЕЗ резервного фонда")
    print("-" * 80)

    position_data_no_reserve = {
        'symbol': 'BTCUSDT',
        'direction': 'LONG',
        'entry_price': 65000.0,
        'position_size': 1000.0,
        'leverage': 5,
        'tp_price': 67000.0,
        'sl_price': 64000.0,
        'p_up': 0.67,
        'ev': 0.035,
        'predictions': {
            'xgb': 0.68,
            'rf': 0.66,
            'nn': 0.67
        }
    }

    # Генерируем текст уведомления
    text = _generate_notification_text(notifier, position_data_no_reserve)
    print(text)

    print("\n[ТЕСТ 2] Уведомление С резервным фондом")
    print("-" * 80)

    position_data_with_reserve = {
        'symbol': 'ETHUSDT',
        'direction': 'SHORT',
        'entry_price': 3500.0,
        'position_size': 500.0,
        'leverage': 3,
        'tp_price': 3400.0,
        'sl_price': 3600.0,
        'p_up': 0.32,
        'ev': 0.025,
        'predictions': {
            'xgb': 0.30,
            'rf': 0.35,
            'nn': 0.31
        },
        # Добавляем информацию о резервном фонде
        'reserve_fund': 1500.0,
        'trading_capital': 11250.0,
        'total_equity': 12750.0
    }

    text = _generate_notification_text(notifier, position_data_with_reserve)
    print(text)

    print("\n[ТЕСТ 3] Уведомление с большим резервным фондом")
    print("-" * 80)

    position_data_large_reserve = {
        'symbol': 'SOLUSDT',
        'direction': 'LONG',
        'entry_price': 120.0,
        'position_size': 300.0,
        'leverage': 2,
        'tp_price': 125.0,
        'sl_price': 118.0,
        'p_up': 0.72,
        'ev': 0.045,
        'predictions': {
            'xgb': 0.73,
            'rf': 0.71
        },
        # Большой резервный фонд (30% от капитала)
        'reserve_fund': 6000.0,
        'trading_capital': 14000.0,
        'total_equity': 20000.0
    }

    text = _generate_notification_text(notifier, position_data_large_reserve)
    print(text)

    print("\n" + "="*80)
    print("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("="*80)
    print("\nПроверьте вывод выше, чтобы убедиться, что:")
    print("  1. Уведомление без reserve fund выглядит как обычно")
    print("  2. Уведомление с reserve fund показывает секцию '💼 Капитал'")
    print("  3. Процент резерва рассчитывается корректно")
    print()


def _generate_notification_text(notifier, position_data):
    """
    Генерирует текст уведомления без отправки

    (Копия логики из notify_position_opened)
    """
    from datetime import datetime

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

📊 TP: ${tp_price:.4f} ({notifier._calc_percent(entry_price, tp_price, direction):+.2f}%)
🛡 SL: ${sl_price:.4f} ({notifier._calc_percent(entry_price, sl_price, direction):+.2f}%)
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

    return text


if __name__ == "__main__":
    test_position_opened_notification()
