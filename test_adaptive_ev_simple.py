#!/usr/bin/env python3
"""
Тест: Демонстрация разницы между фиксированными и адаптивными множителями TP/SL в расчёте EV
(Упрощённая версия без импорта классов)
"""

def get_adaptive_tp_sl_multipliers(atr_pct: float) -> tuple:
    """Адаптивные множители из binance_config"""
    if atr_pct < 2.0:  # Низкая волатильность
        return (3.0, 1.5)  # R:R = 2.0:1
    elif atr_pct > 5.0:  # Высокая волатильность
        return (2.0, 1.5)  # R:R = 1.33:1
    else:  # Средняя волатильность (2% - 5%)
        return (2.5, 1.5)  # R:R = 1.67:1


def calculate_ev(p_up: float, tp_mult: float, sl_mult: float) -> tuple:
    """Расчёт EV с заданными множителями"""
    commission = 0.001  # 0.1% комиссия
    slippage = 0.0005   # 0.05% проскальзывание
    total_cost = (commission + slippage) * 2  # 0.003 (0.3%)

    real_tp = tp_mult - total_cost
    real_sl = sl_mult + total_cost

    ev_long = p_up * real_tp - (1.0 - p_up) * real_sl
    ev_short = (1.0 - p_up) * real_tp - p_up * real_sl

    return ev_long, ev_short


print("="*80)
print("ДЕМОНСТРАЦИЯ БАГА: EV с фиксированными vs адаптивными множителями")
print("="*80)
print()

# Тестовые сценарии с разной волатильностью
test_cases = [
    {
        'atr_pct': 1.5,
        'description': 'Низкая волатильность',
    },
    {
        'atr_pct': 3.5,
        'description': 'Средняя волатильность',
    },
    {
        'atr_pct': 6.0,
        'description': 'Высокая волатильность',
    }
]

p_up = 0.60  # Вероятность роста 60%

print(f"Вероятность роста (p_up): {p_up:.1%}")
print()

for case in test_cases:
    atr_pct = case['atr_pct']
    description = case['description']

    # Получаем адаптивные множители
    tp_mult, sl_mult = get_adaptive_tp_sl_multipliers(atr_pct)

    print(f"{'='*80}")
    print(f"{description}: ATR = {atr_pct:.1f}%")
    print(f"{'='*80}")
    print()
    print(f"Адаптивные множители: TP = {tp_mult:.1f}×, SL = {sl_mult:.1f}×")
    print(f"Risk:Reward = {tp_mult:.1f} / {sl_mult:.1f} = {tp_mult/sl_mult:.2f}:1")
    print()

    # 1. EV с ФИКСИРОВАННЫМИ множителями (старая логика - БАГ!)
    ev_long_fixed, ev_short_fixed = calculate_ev(p_up, tp_mult=2.5, sl_mult=1.5)

    print(f"❌ БАГ: EV с ФИКСИРОВАННЫМИ множителями (2.5/1.5):")
    print(f"  EV LONG  = {ev_long_fixed:+.4f} ({ev_long_fixed*100:+.2f}%)")
    print(f"  EV SHORT = {ev_short_fixed:+.4f} ({ev_short_fixed*100:+.2f}%)")
    print()

    # 2. EV с АДАПТИВНЫМИ множителями (исправленная логика)
    ev_long_adaptive, ev_short_adaptive = calculate_ev(p_up, tp_mult=tp_mult, sl_mult=sl_mult)

    print(f"✅ ИСПРАВЛЕНО: EV с АДАПТИВНЫМИ множителями ({tp_mult:.1f}/{sl_mult:.1f}):")
    print(f"  EV LONG  = {ev_long_adaptive:+.4f} ({ev_long_adaptive*100:+.2f}%)")
    print(f"  EV SHORT = {ev_short_adaptive:+.4f} ({ev_short_adaptive*100:+.2f}%)")
    print()

    # 3. Разница
    diff_long = ev_long_adaptive - ev_long_fixed
    diff_short = ev_short_adaptive - ev_short_fixed

    print(f"🔍 РАЗНИЦА:")
    print(f"  LONG:  {diff_long:+.4f} ({diff_long*100:+.2f}%)")
    print(f"  SHORT: {diff_short:+.4f} ({diff_short*100:+.2f}%)")

    if abs(diff_long) > 0.01:
        if diff_long > 0:
            print(f"  ⚠️  Монета была НЕДООЦЕНЕНА на {diff_long*100:.2f}%!")
        else:
            print(f"  ⚠️  Монета была ПЕРЕОЦЕНЕНА на {abs(diff_long)*100:.2f}%!")
    else:
        print(f"  ✓ Разница незначительна")

    print()

print("="*80)
print("ВЫВОДЫ:")
print("="*80)
print()
print("1. При НИЗКОЙ волатильности (ATR < 2%):")
print("   - Адаптивный R:R = 3.0/1.5 = 2.0:1")
print("   - Фиксированный R:R = 2.5/1.5 = 1.67:1")
print("   - EV был ЗАНИЖЕН → монеты недооценивались")
print("   → Могли упустить хорошие сделки!")
print()
print("2. При ВЫСОКОЙ волатильности (ATR > 5%):")
print("   - Адаптивный R:R = 2.0/1.5 = 1.33:1")
print("   - Фиксированный R:R = 2.5/1.5 = 1.67:1")
print("   - EV был ЗАВЫШЕН → монеты переоценивались")
print("   → Могли взять плохие сделки!")
print()
print("3. При СРЕДНЕЙ волатильности (ATR 2-5%):")
print("   - Адаптивный R:R = 2.5/1.5 = 1.67:1")
print("   - Фиксированный R:R = 2.5/1.5 = 1.67:1")
print("   - EV был ПРАВИЛЬНЫЙ ✓")
print()
print("✅ ИСПРАВЛЕНИЕ: Теперь EV рассчитывается с адаптивными множителями!")
print("   Топ-20 будет ранжироваться ПРАВИЛЬНО для всех уровней волатильности.")
print("="*80)
