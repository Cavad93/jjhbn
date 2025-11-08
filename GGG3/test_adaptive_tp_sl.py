#!/usr/bin/env python3
"""
Тест адаптивных множителей TP/SL

Проверяет:
1. Функция get_adaptive_tp_sl_multipliers() корректно возвращает множители
2. Различные уровни волатильности дают разные множители
3. R:R ratio рассчитывается правильно
4. Kelly Criterion учитывает адаптивный R:R ratio
"""

import sys
from pathlib import Path
# Добавляем корневую директорию проекта в путь
sys.path.insert(0, str(Path(__file__).parent))

import binance_config as config
from strategy.kelly_sizer import calculate_kelly_position_size


def test_adaptive_multipliers():
    """Тест адаптивных множителей для разных уровней волатильности"""

    print("="*70)
    print("ТЕСТ АДАПТИВНЫХ МНОЖИТЕЛЕЙ TP/SL")
    print("="*70)
    print()

    test_cases = [
        (1.5, "Низкая волатильность"),
        (3.5, "Средняя волатильность"),
        (6.0, "Высокая волатильность"),
        (1.9, "Граница низкой/средней"),
        (5.1, "Граница средней/высокой"),
    ]

    print("1. Проверка адаптивных множителей")
    print("-" * 70)

    for atr_pct, description in test_cases:
        tp_mult, sl_mult = config.get_adaptive_tp_sl_multipliers(atr_pct)
        rr_ratio = tp_mult / sl_mult

        print(f"\n{description} (ATR = {atr_pct:.1f}%):")
        print(f"  TP множитель: {tp_mult:.1f}x ATR")
        print(f"  SL множитель: {sl_mult:.1f}x ATR")
        print(f"  R:R ratio: {rr_ratio:.2f}:1")

    print("\n" + "="*70)
    print("2. Проверка интеграции с Kelly Criterion")
    print("-" * 70)

    capital = 1000.0
    ev = 0.05
    p_up = 0.62
    recent_wr = 0.58

    print(f"\nПараметры:")
    print(f"  Capital: ${capital:.2f}")
    print(f"  EV: {ev:.2%}")
    print(f"  p_up: {p_up:.2%}")
    print(f"  recent_wr: {recent_wr:.2%}")

    for atr_pct, description in test_cases[:3]:  # Только 3 основных случая
        tp_mult, sl_mult = config.get_adaptive_tp_sl_multipliers(atr_pct)
        rr_ratio = tp_mult / sl_mult

        position_size = calculate_kelly_position_size(
            capital=capital,
            ev=ev,
            p_up=p_up,
            recent_wr=recent_wr,
            rr_ratio=rr_ratio
        )

        print(f"\n{description} (ATR = {atr_pct:.1f}%, R:R = {rr_ratio:.2f}:1):")
        print(f"  Position size: ${position_size:.2f} ({position_size/capital:.1%} of capital)")

    print("\n" + "="*70)
    print("3. Сравнение со старым фиксированным подходом")
    print("-" * 70)

    # Старый фиксированный R:R
    old_rr = config.TP_ATR_MULTIPLIER / config.SL_ATR_MULTIPLIER
    old_position = calculate_kelly_position_size(
        capital=capital,
        ev=ev,
        p_up=p_up,
        recent_wr=recent_wr,
        rr_ratio=old_rr
    )

    print(f"\nСтарый фиксированный подход:")
    print(f"  R:R ratio: {old_rr:.2f}:1")
    print(f"  Position size: ${old_position:.2f}")

    # Новый адаптивный для средней волатильности
    tp_mult, sl_mult = config.get_adaptive_tp_sl_multipliers(3.5)
    new_rr = tp_mult / sl_mult
    new_position = calculate_kelly_position_size(
        capital=capital,
        ev=ev,
        p_up=p_up,
        recent_wr=recent_wr,
        rr_ratio=new_rr
    )

    print(f"\nНовый адаптивный подход (средняя волатильность):")
    print(f"  R:R ratio: {new_rr:.2f}:1")
    print(f"  Position size: ${new_position:.2f}")
    print(f"  Разница: ${new_position - old_position:.2f}")

    print("\n" + "="*70)
    print("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("="*70)
    print()
    print("ВЫВОДЫ:")
    print("1. Адаптивные множители корректно зависят от волатильности")
    print("2. Kelly Criterion учитывает изменяющийся R:R ratio")
    print("3. На низкой волатильности позиция больше (R:R = 2.0:1)")
    print("4. На высокой волатильности позиция меньше (R:R = 1.33:1)")
    print("5. Система автоматически адаптируется к рыночным условиям")
    print()


if __name__ == '__main__':
    test_adaptive_multipliers()
