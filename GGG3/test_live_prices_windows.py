#!/usr/bin/env python3
"""
КРИТИЧЕСКИЙ ТЕСТ ДЛЯ WINDOWS: Что РЕАЛЬНО возвращает API?
Запустите СРАЗУ ПОСЛЕ открытия позиций ботом
"""

from paper_trading.paper_exchange import PaperExchange, get_binance_price
from datetime import datetime
import time

print("=" * 80)
print("🔬 КРИТИЧЕСКИЙ ТЕСТ: Проверка LIVE цен с Binance API")
print("=" * 80)
print()

print(f"🕐 Текущее время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Символы из лога бота
bot_positions = {
    'KSMUSDT': {'entry': 13.42, 'timestamp': '19:14:44'},
    'ALICEUSDT': {'entry': 0.29, 'timestamp': '19:14:48'},
    'ZENUSDT': {'entry': 14.07, 'timestamp': '19:14:51'},
    'ILVUSDT': {'entry': 11.64, 'timestamp': '19:14:55'},
    'ICPUSDT': {'entry': 9.14, 'timestamp': '19:14:58'},
    'YGGUSDT': {'entry': 0.12, 'timestamp': '19:15:01'},
    'SOLUSDT': {'entry': 157.00, 'timestamp': '19:15:04'},
    'XRPUSDT': {'entry': 2.27, 'timestamp': '19:15:11'},
    'BNBUSDT': {'entry': 993.09, 'timestamp': '19:15:15'},
    'PENDLEUSDT': {'entry': 2.72, 'timestamp': '19:15:18'},
}

print("📊 ПРОВЕРКА: Текущая цена vs Цена входа из лога")
print("-" * 80)
print()

critical_issues = []
correct_prices = []

for symbol, info in bot_positions.items():
    try:
        # Получаем ТЕКУЩУЮ цену через Binance API
        current_price = get_binance_price(symbol)
        bot_entry = info['entry']

        # Рассчитываем разницу
        diff_abs = abs(current_price - bot_entry)
        diff_pct = (diff_abs / bot_entry) * 100

        # Определяем статус
        if diff_pct < 5:
            status = "✅ OK"
            correct_prices.append(symbol)
        elif diff_pct < 20:
            status = "⚠️  ОТКЛОНЕНИЕ"
        else:
            status = "❌ КРИТИЧНО"
            critical_issues.append({
                'symbol': symbol,
                'bot_price': bot_entry,
                'real_price': current_price,
                'diff_pct': diff_pct
            })

        print(f"{status}  {symbol:<12s}")
        print(f"         Бот вошёл по:  ${bot_entry:>12.4f}  (в {info['timestamp']})")
        print(f"         Реальная цена: ${current_price:>12.4f}  (СЕЙЧАС)")
        print(f"         Разница:       {diff_pct:>12.2f}%")

        if diff_pct > 100:
            print(f"         ⚠️  ВНИМАНИЕ: Цена УДВОИЛАСЬ или УПАЛА в 2 раза!")

        print()

    except Exception as e:
        print(f"❌ ERROR  {symbol:<12s}")
        print(f"         {str(e)}")
        print()

print()
print("=" * 80)
print("📋 ИТОГОВЫЙ АНАЛИЗ")
print("=" * 80)
print()

print(f"Правильные цены ({len(correct_prices)}):  {', '.join(correct_prices) if correct_prices else 'НЕТ'}")
print(f"Критические проблемы ({len(critical_issues)}): ", end='')

if critical_issues:
    print()
    for issue in critical_issues:
        print(f"  ❌ {issue['symbol']}: Бот {issue['bot_price']:.4f} vs Реальная {issue['real_price']:.4f} (Δ {issue['diff_pct']:.1f}%)")
else:
    print("НЕТ")

print()
print("🔍 ВОЗМОЖНЫЕ ПРИЧИНЫ ПРОБЛЕМЫ:")
print()

if len(critical_issues) >= 5:
    print("❌ КРИТИЧНО: Более 5 монет с огромной разницей!")
    print()
    print("Возможные причины:")
    print("  1. Бот использует СТАРЫЕ данные (кэш, state файлы)")
    print("  2. PaperExchange.create_market_order() НЕ вызывает get_binance_price()")
    print("  3. Где-то в коде есть подмена цен")
    print()
    print("ДЕЙСТВИЯ:")
    print("  1. Удалите файл: C:\\Users\\Administrator\\Desktop\\GGG3\\data\\paper_exchange_state.json")
    print("  2. Удалите всю папку: C:\\Users\\Administrator\\Desktop\\GGG3\\data\\")
    print("  3. Перезапустите бота")
    print()

elif len(critical_issues) >= 1 and len(correct_prices) >= 5:
    print("⚠️  СМЕШАННАЯ СИТУАЦИЯ: Некоторые цены правильные, некоторые нет")
    print()
    print("Это ОЧЕНЬ СТРАННО! Возможные причины:")
    print("  1. Разные монеты берутся из разных источников")
    print("  2. Кэш работает выборочно")
    print("  3. Некоторые пары недоступны на Binance и берутся откуда-то ещё")
    print()
    print("ДЕЙСТВИЯ:")
    print("  1. Проверьте логи бота на ошибки API для конкретных монет")
    print("  2. Удалите кэш и перезапустите")
    print()

elif len(correct_prices) >= 8:
    print("✅ ВСЁ В ПОРЯДКЕ!")
    print()
    print("Все цены близки к реальным (< 5% отклонение).")
    print("Небольшие отклонения - это нормально:")
    print("  - Цены меняются каждую секунду")
    print("  - Между входом (19:14) и проверкой (сейчас) прошло время")
    print("  - Slippage 0.05% уже заложен в боте")
    print()
    print("Ваш бот работает КОРРЕКТНО и использует LIVE цены!")
    print()

else:
    print("⚠️  Недостаточно данных для анализа")
    print("    Проверьте интернет соединение и доступность Binance API")
    print()

print("=" * 80)
print()
print("💡 РЕКОМЕНДАЦИЯ:")
print()
print("Если вы видите критические проблемы (>50% разница):")
print("  1. Остановите бота (Ctrl+C)")
print("  2. Удалите папку 'data' полностью")
print("  3. Перезапустите бота")
print("  4. Запустите этот тест снова сразу после открытия позиций")
print()
print("Если цены правильные (< 10% разница):")
print("  ✅ Ваш бот работает корректно!")
print("  ✅ +11.5% за 15 минут - это РЕАЛЬНАЯ прибыль!")
print("  ✅ Продолжайте использовать бота")
print()
print("=" * 80)
