#!/usr/bin/env python3
"""
Adaptive Threshold для вероятности входа в позицию

Динамически адаптирует порог вероятности p_up на основе:
- Общего количества сделок
- Win rate за последние 100 сделок
- Calibration error модели
- Количества недавних сделок (за последний час)

Usage:
    from strategy.threshold_adapter import get_adaptive_threshold

    threshold = get_adaptive_threshold(
        total_closed=150,
        recent_hour=5,
        recent_wr=0.58,
        calib_error=0.03
    )
"""

import binance_config as config


def get_adaptive_threshold(
    total_closed: int,
    recent_hour: int,
    recent_wr: float,
    calib_error: float
) -> float:
    """
    Рассчитывает адаптивный порог вероятности для входа в позицию

    Логика:
    1. Базовый порог зависит от количества сделок:
       - Если < MIN_TRADES_FOR_ADAPTIVE: используем BASE_THRESHOLD_INITIAL (0.65)
       - Если >= MIN_TRADES_FOR_ADAPTIVE: используем BASE_THRESHOLD_TRAINED (0.58)

    2. Корректировки:
       - Win rate < 50%: повышаем порог (+0.05)
       - Win rate > 60%: снижаем порог (-0.03)
       - Calibration error > 0.05: повышаем порог (+0.02)
       - Много недавних сделок (>10/час): повышаем порог (+0.02)
       - Мало недавних сделок (<2/час): снижаем порог (-0.01)

    Args:
        total_closed: Общее количество закрытых сделок
        recent_hour: Количество сделок за последний час
        recent_wr: Win rate за последние 100 сделок (0-1)
        calib_error: Expected Calibration Error модели (0-1)

    Returns:
        threshold: Адаптивный порог вероятности (0-1)
    """

    # Базовый порог
    if total_closed < config.MIN_TRADES_FOR_ADAPTIVE:
        base_threshold = config.BASE_THRESHOLD_INITIAL
    else:
        base_threshold = config.BASE_THRESHOLD_TRAINED

    # Корректировки
    delta = 0.0

    # 1. Корректировка по win rate
    if recent_wr < 0.50:
        # Плохой win rate - повышаем порог (более строгий отбор)
        delta += 0.05
    elif recent_wr > 0.60:
        # Хороший win rate - снижаем порог (можем брать больше сделок)
        delta -= 0.03

    # 2. Корректировка по calibration error
    if calib_error > 0.05:
        # Плохая калибровка - повышаем порог
        delta += 0.02

    # 3. Корректировка по частоте сделок
    if recent_hour > 10:
        # Слишком много сделок - повышаем порог (снижаем активность)
        delta += 0.02
    elif recent_hour < 2:
        # Мало сделок - снижаем порог (увеличиваем активность)
        delta -= 0.01

    # Применяем ограничения
    delta = max(config.DELTA_MIN, min(delta, config.DELTA_MAX))

    # Финальный порог
    threshold = base_threshold + delta

    # Ограничиваем диапазон [0.5, 0.8]
    threshold = max(0.5, min(threshold, 0.8))

    return threshold


def get_threshold_explanation(
    total_closed: int,
    recent_hour: int,
    recent_wr: float,
    calib_error: float,
    threshold: float
) -> str:
    """
    Возвращает текстовое объяснение адаптивного порога

    Args:
        total_closed: Общее количество закрытых сделок
        recent_hour: Количество сделок за последний час
        recent_wr: Win rate за последние 100 сделок (0-1)
        calib_error: Expected Calibration Error модели (0-1)
        threshold: Рассчитанный порог

    Returns:
        explanation: Текстовое объяснение
    """
    lines = []
    lines.append(f"Adaptive Threshold: {threshold:.3f}")
    lines.append(f"  Total trades: {total_closed}")
    lines.append(f"  Recent win rate: {recent_wr:.1%}")
    lines.append(f"  Calibration error: {calib_error:.3f}")
    lines.append(f"  Recent trades/hour: {recent_hour}")

    # Базовый порог
    if total_closed < config.MIN_TRADES_FOR_ADAPTIVE:
        lines.append(f"  Base: {config.BASE_THRESHOLD_INITIAL:.3f} (initial, < {config.MIN_TRADES_FOR_ADAPTIVE} trades)")
    else:
        lines.append(f"  Base: {config.BASE_THRESHOLD_TRAINED:.3f} (trained)")

    # Корректировки
    adjustments = []
    if recent_wr < 0.50:
        adjustments.append("WR low (+0.05)")
    elif recent_wr > 0.60:
        adjustments.append("WR high (-0.03)")

    if calib_error > 0.05:
        adjustments.append("Cal error high (+0.02)")

    if recent_hour > 10:
        adjustments.append("Too many trades (+0.02)")
    elif recent_hour < 2:
        adjustments.append("Too few trades (-0.01)")

    if adjustments:
        lines.append(f"  Adjustments: {', '.join(adjustments)}")
    else:
        lines.append(f"  Adjustments: none")

    return '\n'.join(lines)
