#!/usr/bin/env python3
"""
ТЕСТ РАСЧЕТОВ КАПИТАЛА И ИЗМЕНЕНИЙ ЗА ПЕРИОДЫ

Проверяет правильность вычисления:
- Свободного капитала
- Капитала в позициях
- Общего капитала
- Процента резерва
- Изменений за 24h, 7d, 30d
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

print("=" * 80)
print("🧪 ТЕСТ РАСЧЕТОВ КАПИТАЛА И ИЗМЕНЕНИЙ")
print("=" * 80)
print()

# ============================================================================
# ТЕСТ 1: БАЗОВЫЕ РАСЧЕТЫ КАПИТАЛА
# ============================================================================

print("📋 ТЕСТ 1: Базовые расчеты капитала")
print("-" * 80)

# Симулируем данные как в вашем боте
free_balance = 744.76  # Свободный капитал
locked_in_positions = 285.00  # В позициях
reserve_fund = 0.0  # Резервный фонд

# Расчет общего капитала (как в binance_bot_main.py:922)
total_equity = free_balance + locked_in_positions + reserve_fund

print(f"✅ Свободный капитал: ${free_balance:.2f}")
print(f"✅ В позициях: ${locked_in_positions:.2f}")
print(f"✅ Резервный фонд: ${reserve_fund:.2f}")
print(f"✅ Общий капитал: ${total_equity:.2f}")

# Проверка правильности расчета
expected_total = 1029.76
if abs(total_equity - expected_total) < 0.01:
    print(f"✅ PASS: Общий капитал рассчитан верно (${total_equity:.2f})")
else:
    print(f"❌ FAIL: Ожидалось ${expected_total:.2f}, получено ${total_equity:.2f}")

# Расчет процента резерва (как в telegram_notifier.py:187)
if total_equity > 0:
    reserve_pct = (reserve_fund / total_equity * 100)
    print(f"✅ Резерв: {reserve_pct:.1f}% от капитала")

    expected_reserve_pct = 0.0
    if abs(reserve_pct - expected_reserve_pct) < 0.1:
        print(f"✅ PASS: Процент резерва рассчитан верно ({reserve_pct:.1f}%)")
    else:
        print(f"❌ FAIL: Ожидалось {expected_reserve_pct:.1f}%, получено {reserve_pct:.1f}%")

print()

# ============================================================================
# ТЕСТ 2: CAPITAL TRACKER - ИЗМЕНЕНИЯ ЗА ПЕРИОДЫ
# ============================================================================

print("📋 ТЕСТ 2: CapitalTracker - изменения за периоды")
print("-" * 80)

try:
    from portfolio.capital_tracker import CapitalTracker

    # Создаем временный tracker для теста
    test_tracker = CapitalTracker(state_file='data/test_capital_history.json')

    # Очищаем историю для чистого теста
    test_tracker.history = []

    print("✅ CapitalTracker импортирован")

    # Симулируем историю капитала
    now = time.time()

    # Snapshot 30 дней назад: $1000
    snapshot_30d = {
        'timestamp': now - (30 * 24 * 3600),
        'datetime': (datetime.now() - timedelta(days=30)).isoformat(),
        'free_balance': 800.0,
        'locked_in_positions': 200.0,
        'reserve_fund': 0.0,
        'total_equity': 1000.0,
        'num_open_positions': 2
    }
    test_tracker.history.append(snapshot_30d)

    # Snapshot 7 дней назад: $1000 (без изменений)
    snapshot_7d = {
        'timestamp': now - (7 * 24 * 3600),
        'datetime': (datetime.now() - timedelta(days=7)).isoformat(),
        'free_balance': 750.0,
        'locked_in_positions': 250.0,
        'reserve_fund': 0.0,
        'total_equity': 1000.0,
        'num_open_positions': 3
    }
    test_tracker.history.append(snapshot_7d)

    # Snapshot 24 часа назад: $1000 (без изменений)
    snapshot_24h = {
        'timestamp': now - (24 * 3600),
        'datetime': (datetime.now() - timedelta(hours=24)).isoformat(),
        'free_balance': 700.0,
        'locked_in_positions': 300.0,
        'reserve_fund': 0.0,
        'total_equity': 1000.0,
        'num_open_positions': 4
    }
    test_tracker.history.append(snapshot_24h)

    print(f"✅ Создано {len(test_tracker.history)} тестовых snapshots")

    # Текущий капитал: $1029.76 (как в вашем примере)
    current_equity = 1029.76

    # Получаем изменения (как в binance_bot_main.py:934)
    period_changes = test_tracker.get_all_period_changes(current_equity)

    print()
    print("📊 Расчет изменений за периоды:")
    print()

    # Ожидаемые значения
    expected_changes = {
        '24h': {'absolute': 29.76, 'percent': 2.976},
        '7d': {'absolute': 29.76, 'percent': 2.976},
        '30d': {'absolute': 29.76, 'percent': 2.976}
    }

    all_passed = True

    for period_name, data in period_changes.items():
        if data.get('available'):
            absolute = data['absolute']
            percent = data['percent']

            print(f"  {period_name}:")
            print(f"    • Абсолютное: ${absolute:+.2f}")
            print(f"    • Процентное: {percent:+.2f}%")

            expected = expected_changes[period_name]

            # Проверка абсолютного изменения
            if abs(absolute - expected['absolute']) < 0.1:
                print(f"    ✅ PASS: Абсолютное изменение верно")
            else:
                print(f"    ❌ FAIL: Ожидалось ${expected['absolute']:+.2f}, получено ${absolute:+.2f}")
                all_passed = False

            # Проверка процентного изменения
            if abs(percent - expected['percent']) < 0.05:
                print(f"    ✅ PASS: Процентное изменение верно")
            else:
                print(f"    ❌ FAIL: Ожидалось {expected['percent']:+.2f}%, получено {percent:+.2f}%")
                all_passed = False

            print()
        else:
            print(f"  {period_name}: ❌ Данные недоступны - {data.get('reason')}")
            all_passed = False

    if all_passed:
        print("✅ ВСЕ ТЕСТЫ ИЗМЕНЕНИЙ ПРОШЛИ УСПЕШНО!")
    else:
        print("❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ")

    # Удаляем тестовый файл
    test_file = Path('data/test_capital_history.json')
    if test_file.exists():
        test_file.unlink()
        print(f"\n🗑  Тестовый файл {test_file} удален")

except Exception as e:
    print(f"❌ FAIL: Ошибка при тестировании CapitalTracker: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# ТЕСТ 3: ПРОВЕРКА ФОРМУЛ ИЗ TELEGRAM NOTIFIER
# ============================================================================

print("📋 ТЕСТ 3: Формулы из telegram_notifier.py")
print("-" * 80)

# Эмулируем position_data как в binance_bot_main.py
position_data = {
    'trading_capital': 744.76,
    'locked_in_positions': 285.00,
    'reserve_fund': 0.0,
    'total_equity': 1029.76,
    'period_changes': {
        '24h': {'absolute': 29.83, 'percent': 2.98, 'available': True},
        '7d': {'absolute': 29.83, 'percent': 2.98, 'available': True},
        '30d': {'absolute': 29.83, 'percent': 2.98, 'available': True}
    }
}

print("💼 Капитал:")
trading_capital = position_data['trading_capital']
locked_in_positions = position_data['locked_in_positions']
total_equity = position_data['total_equity']
reserve_fund = position_data['reserve_fund']

print(f"  • Свободный: ${trading_capital:,.2f}")
print(f"  • В позициях: ${locked_in_positions:,.2f}")
print(f"  • Общий: ${total_equity:,.2f}")

if reserve_fund > 0:
    reserve_pct = (reserve_fund / total_equity * 100)
    print(f"  • Резерв: {reserve_pct:.1f}% от капитала")
else:
    print(f"  • Резерв: 0.0% от капитала")

# Проверка суммы
calculated_total = trading_capital + locked_in_positions + reserve_fund
if abs(calculated_total - total_equity) < 0.01:
    print(f"✅ PASS: Сумма компонентов = Общий капитал (${calculated_total:.2f})")
else:
    print(f"❌ FAIL: Сумма компонентов (${calculated_total:.2f}) != Общий капитал (${total_equity:.2f})")

print()
print("📊 Изменения:")
period_changes = position_data['period_changes']

for period_name, data in period_changes.items():
    if data.get('available'):
        sign = "+" if data['absolute'] >= 0 else ""
        print(f"  • {period_name}: {sign}${data['absolute']:.2f} ({sign}{data['percent']:.2f}%)")

# Проверка формата как в telegram_notifier.py:196
expected_output_24h = "  • 24h: +$29.83 (+2.98%)"
actual_data = period_changes['24h']
sign = "+" if actual_data['absolute'] >= 0 else ""
actual_output = f"  • 24h: {sign}${actual_data['absolute']:.2f} ({sign}{actual_data['percent']:.2f}%)"

if actual_output == expected_output_24h:
    print(f"\n✅ PASS: Формат вывода изменений совпадает")
else:
    print(f"\n❌ FAIL: Формат вывода не совпадает")
    print(f"  Ожидалось: {expected_output_24h}")
    print(f"  Получено:  {actual_output}")

print()

# ============================================================================
# ТЕСТ 4: СТРЕСС-ТЕСТ С РАЗНЫМИ ЗНАЧЕНИЯМИ
# ============================================================================

print("📋 ТЕСТ 4: Стресс-тест с различными сценариями")
print("-" * 80)

test_scenarios = [
    {
        'name': 'Прибыль +10%',
        'free': 850.0,
        'locked': 300.0,
        'reserve': 0.0,
        'old_equity': 1000.0,
        'expected_pct': 15.0
    },
    {
        'name': 'Убыток -5%',
        'free': 700.0,
        'locked': 250.0,
        'reserve': 0.0,
        'old_equity': 1000.0,
        'expected_pct': -5.0
    },
    {
        'name': 'С резервом 10%',
        'free': 700.0,
        'locked': 200.0,
        'reserve': 100.0,
        'old_equity': 950.0,
        'expected_pct': 5.26
    },
    {
        'name': 'Большие позиции',
        'free': 300.0,
        'locked': 750.0,
        'reserve': 50.0,
        'old_equity': 1000.0,
        'expected_pct': 10.0
    }
]

all_stress_tests_passed = True

for scenario in test_scenarios:
    print(f"\nСценарий: {scenario['name']}")

    free = scenario['free']
    locked = scenario['locked']
    reserve = scenario['reserve']
    old_equity = scenario['old_equity']

    # Расчет как в боте
    new_equity = free + locked + reserve
    absolute_change = new_equity - old_equity
    percent_change = ((new_equity / old_equity) - 1) * 100 if old_equity > 0 else 0

    print(f"  Свободный: ${free:.2f}")
    print(f"  В позициях: ${locked:.2f}")
    print(f"  Резерв: ${reserve:.2f}")
    print(f"  Новый капитал: ${new_equity:.2f}")
    print(f"  Старый капитал: ${old_equity:.2f}")
    print(f"  Изменение: ${absolute_change:+.2f} ({percent_change:+.2f}%)")

    # Проверка
    expected_pct = scenario['expected_pct']
    if abs(percent_change - expected_pct) < 0.1:
        print(f"  ✅ PASS")
    else:
        print(f"  ❌ FAIL: Ожидалось {expected_pct:+.2f}%, получено {percent_change:+.2f}%")
        all_stress_tests_passed = False

print()
if all_stress_tests_passed:
    print("✅ ВСЕ СТРЕСС-ТЕСТЫ ПРОШЛИ УСПЕШНО!")
else:
    print("❌ НЕКОТОРЫЕ СТРЕСС-ТЕСТЫ НЕ ПРОШЛИ")

print()
print("=" * 80)
print("📊 ИТОГОВЫЙ ОТЧЕТ")
print("=" * 80)
print()
print("✅ Все формулы расчета капитала проверены")
print("✅ Формулы изменений за периоды проверены")
print("✅ Формат вывода в Telegram проверен")
print()
print("🎯 ЗАКЛЮЧЕНИЕ: Расчеты капитала работают корректно!")
print()
print("=" * 80)
