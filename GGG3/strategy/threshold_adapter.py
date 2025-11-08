#!/usr/bin/env python3
"""
Adaptive Threshold для вероятности входа в позицию (Enhanced Version)

Динамически адаптирует порог вероятности p_up на основе:
- Общего количества сделок
- Win rate за последние 50 и 200 сделок (с трендом)
- Calibration error модели
- Количества недавних сделок (за последний час)
- Текущей просадки от пика
- Количества отклоненных сигналов (для проверки активности)

Улучшения v2.0:
✅ Плавные функции вместо ступенчатых (устранены резкие скачки)
✅ Учет тренда изменения win rate (обнаружение деградации)
✅ Учет текущей просадки (повышение осторожности при убытках)
✅ Улучшенная проверка активности (только при наличии отклоненных сигналов)
✅ Более точная калибровка корректировок

Usage:
    from strategy.threshold_adapter import get_adaptive_threshold

    threshold = get_adaptive_threshold(
        total_closed=150,
        recent_hour=5,
        recent_wr=0.58,
        calib_error=0.03,
        wr_50=0.56,
        wr_200=0.60,
        current_drawdown=0.08,
        signals_rejected_hour=3
    )
"""

import binance_config as config


def _smooth_step(x: float, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    """
    Плавная интерполяция между y_min и y_max по мере изменения x от x_min до x_max

    Args:
        x: Текущее значение
        x_min: Минимальное значение x (соответствует y_min)
        x_max: Максимальное значение x (соответствует y_max)
        y_min: Минимальное значение y
        y_max: Максимальное значение y

    Returns:
        Интерполированное значение y
    """
    if x <= x_min:
        return y_min
    if x >= x_max:
        return y_max

    # Линейная интерполяция
    ratio = (x - x_min) / (x_max - x_min)
    return y_min + ratio * (y_max - y_min)


def get_adaptive_threshold(
    total_closed: int,
    recent_hour: int,
    recent_wr: float,
    calib_error: float,
    last_trade_time: float = None,
    wr_50: float = None,
    wr_200: float = None,
    current_drawdown: float = 0.0,
    signals_rejected_hour: int = 0
) -> float:
    """
    Рассчитывает адаптивный порог вероятности для входа в позицию (Enhanced)

    Логика:
    1. ХОЛОДНЫЙ СТАРТ (первые 500 сделок):
       - Фиксированный порог 0.5 для быстрого накопления статистики

    2. БЕЗ АКТИВНОСТИ (нет сделок 1+ час И есть отклоненные сигналы):
       - Снижаем порог до 0.52 для возобновления торговли
       - Только если действительно есть сигналы, но мы их отклоняем

    3. Базовый порог зависит от количества сделок:
       - Если < MIN_TRADES_FOR_ADAPTIVE: используем BASE_THRESHOLD_INITIAL (0.65)
       - Если >= MIN_TRADES_FOR_ADAPTIVE: используем BASE_THRESHOLD_TRAINED (0.58)

    4. ПЛАВНЫЕ корректировки:
       - Win rate: плавная функция от 0.35 (крит.) до 0.70 (отл.)
       - Calibration error: плавная от 0.01 до 0.10
       - Частота сделок: плавная от 0 до 20/час
       - Тренд WR: сравнение WR_50 vs WR_200 (деградация)
       - Просадка: плавная от 0% до 20%

    Args:
        total_closed: Общее количество закрытых сделок
        recent_hour: Количество сделок за последний час
        recent_wr: Win rate за последние 100 сделок (0-1)
        calib_error: Expected Calibration Error модели (0-1)
        last_trade_time: Timestamp последней сделки (Unix time) для проверки активности
        wr_50: Win rate за последние 50 сделок (для тренда)
        wr_200: Win rate за последние 200 сделок (для тренда)
        current_drawdown: Текущая просадка от пика (0-1)
        signals_rejected_hour: Количество отклоненных сигналов за последний час

    Returns:
        threshold: Адаптивный порог вероятности (0-1)
    """
    import time

    # ===== ХОЛОДНЫЙ СТАРТ: Первые 500 сделок =====
    if total_closed < 500:
        # Фиксированный низкий порог для быстрого накопления данных
        return 0.5

    # ===== ПРОВЕРКА АКТИВНОСТИ: Нет сделок > 1 часа И есть отклоненные сигналы =====
    if last_trade_time is not None and signals_rejected_hour > 0:
        time_since_last_trade = time.time() - last_trade_time
        hours_since_last_trade = time_since_last_trade / 3600.0

        # УЛУЧШЕНО: снижаем порог только если есть отклоненные сигналы
        # Это значит, что рынок дает возможности, но порог слишком высокий
        if hours_since_last_trade >= 1.0 and signals_rejected_hour >= 3:
            # Нет сделок больше часа, но есть сигналы - немного снижаем порог
            return 0.52  # Не 0.5, а чуть выше для осторожности

    # ===== ОБЫЧНЫЙ РЕЖИМ =====

    # Базовый порог
    if total_closed < config.MIN_TRADES_FOR_ADAPTIVE:
        base_threshold = config.BASE_THRESHOLD_INITIAL
    else:
        base_threshold = config.BASE_THRESHOLD_TRAINED

    # Корректировки (все плавные функции)
    delta = 0.0

    # 1. ✅ ПЛАВНАЯ корректировка по win rate
    # WR от 0.35 (критично) до 0.70 (отлично)
    # Корректировка от +0.08 до -0.05
    wr_delta = _smooth_step(
        x=recent_wr,
        x_min=0.35,  # При WR=0.35 → delta = +0.08
        x_max=0.70,  # При WR=0.70 → delta = -0.05
        y_min=0.08,
        y_max=-0.05
    )
    delta += wr_delta

    # 2. ✅ ПЛАВНАЯ корректировка по calibration error
    # CalErr от 0.01 (отлично) до 0.10 (плохо)
    # Корректировка от 0.00 до +0.04
    cal_delta = _smooth_step(
        x=calib_error,
        x_min=0.01,
        x_max=0.10,
        y_min=0.0,
        y_max=0.04
    )
    delta += cal_delta

    # 3. ✅ ПЛАВНАЯ корректировка по частоте сделок
    # Freq от 0 (мало) до 20 (много) сделок/час
    # Корректировка от -0.02 до +0.03
    freq_delta = _smooth_step(
        x=recent_hour,
        x_min=0,
        x_max=20,
        y_min=-0.02,
        y_max=0.03
    )
    delta += freq_delta

    # 4. ✅ НОВОЕ: Корректировка по тренду WR (деградация модели)
    # Сравниваем WR_50 vs WR_200
    if wr_50 is not None and wr_200 is not None and total_closed >= 200:
        wr_trend_diff = wr_50 - wr_200

        # Если WR падает (wr_50 < wr_200), повышаем порог
        # Падение на 0.10 (10%) → +0.05 к порогу
        # Рост на 0.10 (10%) → -0.02 к порогу
        trend_delta = _smooth_step(
            x=wr_trend_diff,
            x_min=-0.10,  # WR падает на 10%
            x_max=0.10,   # WR растет на 10%
            y_min=0.05,   # Повышаем порог
            y_max=-0.02   # Снижаем порог
        )
        delta += trend_delta

    # 5. ✅ НОВОЕ: Корректировка по текущей просадке
    # Drawdown от 0% до 20%
    # Корректировка от 0.00 до +0.06
    dd_delta = _smooth_step(
        x=current_drawdown,
        x_min=0.0,
        x_max=0.20,  # 20% просадка
        y_min=0.0,
        y_max=0.06   # Значительно повышаем порог при большой просадке
    )
    delta += dd_delta

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
    threshold: float,
    wr_50: float = None,
    wr_200: float = None,
    current_drawdown: float = 0.0
) -> str:
    """
    Возвращает текстовое объяснение адаптивного порога (Enhanced)

    Args:
        total_closed: Общее количество закрытых сделок
        recent_hour: Количество сделок за последний час
        recent_wr: Win rate за последние 100 сделок (0-1)
        calib_error: Expected Calibration Error модели (0-1)
        threshold: Рассчитанный порог
        wr_50: Win rate за последние 50 сделок
        wr_200: Win rate за последние 200 сделок
        current_drawdown: Текущая просадка от пика (0-1)

    Returns:
        explanation: Текстовое объяснение
    """
    lines = []
    lines.append(f"Adaptive Threshold: {threshold:.3f}")
    lines.append(f"  Total trades: {total_closed}")
    lines.append(f"  Recent win rate (100): {recent_wr:.1%}")

    if wr_50 is not None and wr_200 is not None:
        lines.append(f"  Win rate (50): {wr_50:.1%}")
        lines.append(f"  Win rate (200): {wr_200:.1%}")
        trend = wr_50 - wr_200
        trend_str = f"+{trend:.1%}" if trend >= 0 else f"{trend:.1%}"
        lines.append(f"  WR Trend (50 vs 200): {trend_str}")

    lines.append(f"  Calibration error: {calib_error:.3f}")
    lines.append(f"  Recent trades/hour: {recent_hour}")
    lines.append(f"  Current drawdown: {current_drawdown:.1%}")

    # Базовый порог
    if total_closed < config.MIN_TRADES_FOR_ADAPTIVE:
        lines.append(f"  Base: {config.BASE_THRESHOLD_INITIAL:.3f} (initial, < {config.MIN_TRADES_FOR_ADAPTIVE} trades)")
    else:
        lines.append(f"  Base: {config.BASE_THRESHOLD_TRAINED:.3f} (trained)")

    # Корректировки
    adjustments = []

    # Win rate
    if recent_wr < 0.45:
        adjustments.append(f"WR very low (+)")
    elif recent_wr < 0.50:
        adjustments.append(f"WR low (+)")
    elif recent_wr > 0.65:
        adjustments.append(f"WR excellent (-)")
    elif recent_wr > 0.60:
        adjustments.append(f"WR good (-)")

    # Calibration error
    if calib_error > 0.07:
        adjustments.append(f"CalErr high (+)")
    elif calib_error > 0.05:
        adjustments.append(f"CalErr medium (+)")

    # Frequency
    if recent_hour > 15:
        adjustments.append(f"Too many trades (+)")
    elif recent_hour < 2:
        adjustments.append(f"Too few trades (-)")

    # Trend
    if wr_50 is not None and wr_200 is not None:
        trend = wr_50 - wr_200
        if trend < -0.05:
            adjustments.append(f"WR declining (+)")
        elif trend > 0.05:
            adjustments.append(f"WR improving (-)")

    # Drawdown
    if current_drawdown > 0.15:
        adjustments.append(f"DD critical (+)")
    elif current_drawdown > 0.10:
        adjustments.append(f"DD high (+)")
    elif current_drawdown > 0.05:
        adjustments.append(f"DD medium (+)")

    if adjustments:
        lines.append(f"  Adjustments: {', '.join(adjustments)}")
    else:
        lines.append(f"  Adjustments: none")

    return '\n'.join(lines)
