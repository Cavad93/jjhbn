#!/usr/bin/env python3
"""
Тест: Демонстрация разницы между фиксированными и адаптивными множителями TP/SL в расчёте EV
"""

import sys
sys.path.append('GGG3')

from strategy.coin_selector import CoinSelector
import binance_config

print("="*80)
print("ДЕМОНСТРАЦИЯ БАГА: EV с фиксированными vs адаптивными множителями")
print("="*80)
print()

selector = CoinSelector(min_ev_threshold=0.02)

# Тестовые сценарии с разной волатильностью
test_cases = [
    {
        'atr_pct': 1.5,
        'description': 'Низкая волатильность',
        'expected_mult': (3.0, 1.5)
    },
    {
        'atr_pct': 3.5,
        'description': 'Средняя волатильность',
        'expected_mult': (2.5, 1.5)
    },
    {
        'atr_pct': 6.0,
        'description': 'Высокая волатильность',
        'expected_mult': (2.0, 1.5)
    }
]

p_up = 0.60  # Вероятность роста 60%

print(f"Вероятность роста (p_up): {p_up:.1%}")
print()

for case in test_cases:
    atr_pct = case['atr_pct']
    description = case['description']
    expected_mult = case['expected_mult']

    # Получаем адаптивные множители
    tp_mult, sl_mult = binance_config.get_adaptive_tp_sl_multipliers(atr_pct)

    print(f"{'='*80}")
    print(f"{description}: ATR = {atr_pct:.1f}%")
    print(f"{'='*80}")
    print()
    print(f"Адаптивные множители: TP = {tp_mult:.1f}×, SL = {sl_mult:.1f}×")
    print(f"Risk:Reward = {tp_mult:.1f} / {sl_mult:.1f} = {tp_mult/sl_mult:.2f}:1")
    print()

    # 1. EV с ФИКСИРОВАННЫМИ множителями (старая логика - БАГ!)
    ev_long_fixed, ev_short_fixed = selector.calculate_ev_bidirectional(p_up)

    print(f"❌ БАГ: EV с ФИКСИРОВАННЫМИ множителями (2.5/1.5):")
    print(f"  EV LONG  = {ev_long_fixed:+.4f} ({ev_long_fixed*100:+.2f}%)")
    print(f"  EV SHORT = {ev_short_fixed:+.4f} ({ev_short_fixed*100:+.2f}%)")
    print()

    # 2. EV с АДАПТИВНЫМИ множителями (исправленная логика)
    ev_long_adaptive, ev_short_adaptive = selector.calculate_ev_bidirectional(
        p_up,
        tp_mult=tp_mult,
        sl_mult=sl_mult
    )

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
print()
print("2. При ВЫСОКОЙ волатильности (ATR > 5%):")
print("   - Адаптивный R:R = 2.0/1.5 = 1.33:1")
print("   - Фиксированный R:R = 2.5/1.5 = 1.67:1")
print("   - EV был ЗАВЫШЕН → монеты переоценивались")
print()
print("3. При СРЕДНЕЙ волатильности (ATR 2-5%):")
print("   - Адаптивный R:R = 2.5/1.5 = 1.67:1")
print("   - Фиксированный R:R = 2.5/1.5 = 1.67:1")
print("   - EV был ПРАВИЛЬНЫЙ")
print()
print("ИСПРАВЛЕНИЕ: Теперь EV рассчитывается с адаптивными множителями!")
print("="*80)
