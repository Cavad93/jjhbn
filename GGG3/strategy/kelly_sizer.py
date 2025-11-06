#!/usr/bin/env python3
"""
Kelly Criterion Position Sizing

Рассчитывает оптимальный размер позиции на основе Kelly Criterion
с учетом EV и вероятности успеха.

Usage:
    from strategy.kelly_sizer import calculate_kelly_position_size

    position_size = calculate_kelly_position_size(
        capital=1000.0,
        ev=0.05,
        p_up=0.62,
        recent_wr=0.58
    )
"""

import binance_config as config


def calculate_kelly_position_size(
    capital: float,
    ev: float,
    p_up: float,
    recent_wr: float = 0.5,
    kelly_fraction: float = 0.25
) -> float:
    """
    Рассчитывает размер позиции на основе Kelly Criterion

    Полный Kelly:
        f = (p * b - q) / b
        где:
        - p = вероятность выигрыша
        - q = 1 - p = вероятность проигрыша
        - b = odds (выигрыш / проигрыш)

    Но мы используем fractional Kelly (kelly_fraction=0.25) для снижения риска.

    Args:
        capital: Доступный капитал (USDT)
        ev: Expected Value (ожидаемая прибыль в %)
        p_up: Вероятность достижения TP (0-1)
        recent_wr: Реальный win rate за последние сделки (0-1)
        kelly_fraction: Фракция Kelly (0.25 = четверть Kelly)

    Returns:
        position_size: Размер позиции в USDT
    """

    # Рассчитываем вероятность выигрыша
    # Берем среднее между предсказанием модели и реальным win rate
    p_win = (p_up + recent_wr) / 2.0

    # Вероятность проигрыша
    p_loss = 1.0 - p_win

    # Odds (R/R ratio из конфига)
    # TP = 2.5 ATR, SL = 1.5 ATR => R/R = 2.5/1.5 = 1.667
    rr_ratio = config.TP_ATR_MULTIPLIER / config.SL_ATR_MULTIPLIER

    # Kelly formula
    # f = (p * b - q) / b
    # где b = rr_ratio
    kelly_full = (p_win * rr_ratio - p_loss) / rr_ratio

    # Fractional Kelly для снижения риска
    kelly = kelly_full * kelly_fraction

    # Ограничиваем Kelly положительными значениями
    if kelly <= 0:
        kelly = config.MIN_RISK_PER_POSITION  # Минимальный риск

    # Ограничиваем диапазон [MIN_RISK, MAX_RISK]
    kelly = max(config.MIN_RISK_PER_POSITION, min(kelly, config.MAX_RISK_PER_POSITION))

    # Рассчитываем размер позиции
    position_size = capital * kelly

    # Ограничиваем размер позиции
    position_size = max(config.MIN_POSITION_SIZE_USDT, min(position_size, config.MAX_POSITION_SIZE_USDT))

    return position_size


def get_kelly_explanation(
    capital: float,
    ev: float,
    p_up: float,
    recent_wr: float,
    position_size: float,
    kelly_fraction: float = 0.25
) -> str:
    """
    Возвращает текстовое объяснение расчета Kelly

    Args:
        capital: Доступный капитал (USDT)
        ev: Expected Value
        p_up: Вероятность TP
        recent_wr: Реальный win rate
        position_size: Рассчитанный размер позиции
        kelly_fraction: Фракция Kelly

    Returns:
        explanation: Текстовое объяснение
    """
    p_win = (p_up + recent_wr) / 2.0
    p_loss = 1.0 - p_win
    rr_ratio = config.TP_ATR_MULTIPLIER / config.SL_ATR_MULTIPLIER
    kelly_full = (p_win * rr_ratio - p_loss) / rr_ratio
    kelly = kelly_full * kelly_fraction
    kelly = max(config.MIN_RISK_PER_POSITION, min(kelly, config.MAX_RISK_PER_POSITION))

    lines = []
    lines.append(f"Kelly Position Sizing:")
    lines.append(f"  Capital: ${capital:.2f}")
    lines.append(f"  EV: {ev:.2%}")
    lines.append(f"  p_up (model): {p_up:.2%}")
    lines.append(f"  recent_wr (actual): {recent_wr:.2%}")
    lines.append(f"  p_win (avg): {p_win:.2%}")
    lines.append(f"  R/R ratio: {rr_ratio:.2f}")
    lines.append(f"  Kelly full: {kelly_full:.2%}")
    lines.append(f"  Kelly fraction: {kelly_fraction:.2f}")
    lines.append(f"  Kelly adjusted: {kelly:.2%}")
    lines.append(f"  Position size: ${position_size:.2f} ({position_size/capital:.2%} of capital)")

    return '\n'.join(lines)
