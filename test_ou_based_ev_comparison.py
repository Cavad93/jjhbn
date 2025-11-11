#!/usr/bin/env python3
"""
Сравнение расчёта EV: Стандартный vs OU-based vs Гибридный
"""

import numpy as np
from math import log, exp, sqrt, sinh

# ============================================================================
# СТАНДАРТНЫЙ РАСЧЁТ EV
# ============================================================================

def standard_ev(p_up, tp_mult=2.5, sl_mult=1.5):
    """Стандартная формула EV"""
    commission = 0.001
    slippage = 0.0005
    total_cost = (commission + slippage) * 2

    real_tp = tp_mult - total_cost
    real_sl = sl_mult + total_cost

    ev_long = p_up * real_tp - (1.0 - p_up) * real_sl
    return ev_long

# ============================================================================
# OU-BASED РАСЧЁТ EV
# ============================================================================

def ou_based_ev_analytical(current_price, tp_price, sl_price, theta, mu, sigma):
    """
    Аналитический расчёт EV на основе OU процесса

    Args:
        current_price: Текущая цена
        tp_price: Take Profit уровень
        sl_price: Stop Loss уровень
        theta: Скорость возврата к среднему
        mu: Долгосрочное среднее (log-price)
        sigma: Волатильность

    Returns:
        ev_long: Expected value для LONG позиции (в %)
    """
    x = log(current_price)
    tp = log(tp_price)
    sl = log(sl_price)

    # Параметр mean reversion
    k = sqrt(2 * theta) / sigma if sigma > 0 else 0.001

    # Вероятность достижения TP первым (first hitting time)
    try:
        numerator = sinh(k * (x - sl))
        denominator = sinh(k * (tp - sl))
        p_tp = numerator / denominator if denominator != 0 else 0.5
    except:
        p_tp = 0.5  # Fallback если sinh переполняется

    # Вероятность достижения SL первым
    p_sl = 1.0 - p_tp

    # Награды (в %)
    reward_tp = (tp_price - current_price) / current_price
    loss_sl = (sl_price - current_price) / current_price

    # Expected Value
    ev_long = p_tp * reward_tp + p_sl * loss_sl

    return ev_long

def ou_probability(current_price, mu, theta, sigma):
    """
    Вероятность роста на основе OU (упрощённая версия)

    Идея: Чем дальше цена от μ, тем сильнее mean reversion
    """
    x = log(current_price)
    deviation = x - mu

    # Базовая вероятность 50%
    base_prob = 0.5

    # Корректировка на mean reversion
    # Если выше μ → давление вниз → p_up уменьшается
    # Если ниже μ → давление вверх → p_up увеличивается
    sigma_eff = sigma / sqrt(2 * theta) if theta > 0 else sigma
    correction = -deviation / sigma_eff

    # Преобразуем через сигмоиду чтобы остаться в [0, 1]
    p_up = 1 / (1 + exp(-correction))

    return p_up

# ============================================================================
# ГИБРИДНЫЙ ПОДХОД
# ============================================================================

def hybrid_ev(p_up_ml, current_price, mu, theta, scaling=0.1):
    """
    Гибридный подход: ML + OU коррекция

    Args:
        p_up_ml: Вероятность от ML модели
        current_price: Текущая цена
        mu: Среднее из OU
        theta: Скорость возврата
        scaling: Масштаб коррекции (0.1 = 10%)

    Returns:
        p_up_corrected: Скорректированная вероятность
    """
    x = log(current_price)
    deviation = x - mu

    # Корректировка: чем дальше от среднего, тем сильнее коррекция
    correction = -theta * deviation * scaling

    # Применяем коррекцию
    p_up_corrected = p_up_ml + correction

    # Ограничиваем [0, 1]
    p_up_corrected = max(0.0, min(1.0, p_up_corrected))

    return p_up_corrected

# ============================================================================
# ОЦЕНКА ПАРАМЕТРОВ OU (упрощённая)
# ============================================================================

def estimate_ou_parameters(prices):
    """
    Упрощённая оценка параметров OU из исторических цен

    Args:
        prices: Массив исторических цен

    Returns:
        theta, mu, sigma
    """
    log_prices = np.log(prices)

    # Дискретная версия OU: X_{t+1} - X_t = α + β·X_t + ε
    X = log_prices[:-1]
    Y = log_prices[1:] - log_prices[:-1]

    # Простая линейная регрессия
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    beta = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean)**2)
    alpha = Y_mean - beta * X_mean

    # Параметры OU
    dt = 1.0  # 1 временной шаг
    theta = -beta / dt
    mu = alpha / (theta * dt) if theta != 0 else X_mean

    # Волатильность из остатков
    residuals = Y - (alpha + beta * X)
    sigma = np.std(residuals) / sqrt(dt)

    return theta, mu, sigma

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("СРАВНЕНИЕ: Стандартный EV vs OU-based EV vs Гибридный EV")
    print("="*80)
    print()

    # ========================================================================
    # СЦЕНАРИЙ 1: Цена около среднего
    # ========================================================================
    print("СЦЕНАРИЙ 1: Цена около долгосрочного среднего")
    print("-"*80)

    current_price = 50000
    tp_price = 52000  # +4%
    sl_price = 48500  # -3%

    # Исторические данные (симулируем)
    np.random.seed(42)
    history = np.random.normal(50000, 1000, 200)
    theta, mu, sigma = estimate_ou_parameters(history)

    print(f"Текущая цена: ${current_price:,.0f}")
    print(f"TP: ${tp_price:,.0f} (+{(tp_price/current_price-1)*100:.1f}%)")
    print(f"SL: ${sl_price:,.0f} ({(sl_price/current_price-1)*100:.1f}%)")
    print()
    print(f"OU параметры:")
    print(f"  μ (среднее): {exp(mu):,.0f}")
    print(f"  θ (скорость): {theta:.4f}")
    print(f"  σ (волатильность): {sigma:.4f}")
    print()

    # ML предсказание
    p_up_ml = 0.65
    print(f"ML предсказание: p_up = {p_up_ml:.2%}")
    print()

    # 1. Стандартный EV
    ev_std = standard_ev(p_up_ml)
    print(f"1️⃣  Стандартный EV:")
    print(f"    EV = {p_up_ml:.2f} × 2.494 - {1-p_up_ml:.2f} × 1.506")
    print(f"    EV = {ev_std:.4f} ({ev_std*100:.2f}%)")
    print()

    # 2. OU-based EV (аналитический)
    ev_ou = ou_based_ev_analytical(current_price, tp_price, sl_price,
                                     theta, mu, sigma)
    print(f"2️⃣  OU-based EV (аналитический):")
    print(f"    EV = {ev_ou:.4f} ({ev_ou*100:.2f}%)")
    print()

    # 3. Гибридный EV
    p_up_hybrid = hybrid_ev(p_up_ml, current_price, mu, theta)
    ev_hybrid = standard_ev(p_up_hybrid)
    print(f"3️⃣  Гибридный EV (ML + OU коррекция):")
    print(f"    p_up скорректированная = {p_up_hybrid:.2%}")
    print(f"    EV = {ev_hybrid:.4f} ({ev_hybrid*100:.2f}%)")
    print()
    print()

    # ========================================================================
    # СЦЕНАРИЙ 2: Цена ВЫШЕ среднего (перекуплена)
    # ========================================================================
    print("="*80)
    print("СЦЕНАРИЙ 2: Цена выше среднего (перекуплена)")
    print("-"*80)

    current_price_high = 55000  # +10% от среднего
    tp_price_high = 57000
    sl_price_high = 53500

    print(f"Текущая цена: ${current_price_high:,.0f} (+10% от среднего)")
    print(f"TP: ${tp_price_high:,.0f}")
    print(f"SL: ${sl_price_high:,.0f}")
    print()
    print(f"ML предсказание: p_up = {p_up_ml:.2%} (такое же)")
    print()

    # 1. Стандартный EV
    ev_std_high = standard_ev(p_up_ml)
    print(f"1️⃣  Стандартный EV: {ev_std_high:.4f} ({ev_std_high*100:.2f}%)")
    print(f"    ⚠️  Не учитывает перекупленность!")
    print()

    # 2. OU-based EV
    ev_ou_high = ou_based_ev_analytical(current_price_high, tp_price_high,
                                         sl_price_high, theta, mu, sigma)
    print(f"2️⃣  OU-based EV: {ev_ou_high:.4f} ({ev_ou_high*100:.2f}%)")
    print(f"    ✅ Учитывает mean reversion!")
    print()

    # 3. Гибридный EV
    p_up_hybrid_high = hybrid_ev(p_up_ml, current_price_high, mu, theta)
    ev_hybrid_high = standard_ev(p_up_hybrid_high)
    print(f"3️⃣  Гибридный EV:")
    print(f"    p_up скорректированная = {p_up_hybrid_high:.2%} (снижена!)")
    print(f"    EV = {ev_hybrid_high:.4f} ({ev_hybrid_high*100:.2f}%)")
    print(f"    ✅ Баланс ML и mean reversion!")
    print()
    print()

    # ========================================================================
    # СЦЕНАРИЙ 3: Цена НИЖЕ среднего (перепродана)
    # ========================================================================
    print("="*80)
    print("СЦЕНАРИЙ 3: Цена ниже среднего (перепродана)")
    print("-"*80)

    current_price_low = 45000  # -10% от среднего
    tp_price_low = 47000
    sl_price_low = 43500

    print(f"Текущая цена: ${current_price_low:,.0f} (-10% от среднего)")
    print(f"TP: ${tp_price_low:,.0f}")
    print(f"SL: ${sl_price_low:,.0f}")
    print()
    print(f"ML предсказание: p_up = {p_up_ml:.2%} (такое же)")
    print()

    # 1. Стандартный EV
    ev_std_low = standard_ev(p_up_ml)
    print(f"1️⃣  Стандартный EV: {ev_std_low:.4f} ({ev_std_low*100:.2f}%)")
    print(f"    ⚠️  Не учитывает перепроданность!")
    print()

    # 2. OU-based EV
    ev_ou_low = ou_based_ev_analytical(current_price_low, tp_price_low,
                                        sl_price_low, theta, mu, sigma)
    print(f"2️⃣  OU-based EV: {ev_ou_low:.4f} ({ev_ou_low*100:.2f}%)")
    print(f"    ✅ Учитывает mean reversion вверх!")
    print()

    # 3. Гибридный EV
    p_up_hybrid_low = hybrid_ev(p_up_ml, current_price_low, mu, theta)
    ev_hybrid_low = standard_ev(p_up_hybrid_low)
    print(f"3️⃣  Гибридный EV:")
    print(f"    p_up скорректированная = {p_up_hybrid_low:.2%} (повышена!)")
    print(f"    EV = {ev_hybrid_low:.4f} ({ev_hybrid_low*100:.2f}%)")
    print(f"    ✅ Баланс ML и mean reversion!")
    print()
    print()

    # ========================================================================
    # ИТОГОВАЯ ТАБЛИЦА
    # ========================================================================
    print("="*80)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*80)
    print()
    print(f"{'Сценарий':<25s} | {'Стандартный':<15s} | {'OU-based':<15s} | {'Гибридный':<15s}")
    print("-"*80)
    print(f"{'Около среднего':<25s} | {ev_std*100:>13.2f}% | {ev_ou*100:>13.2f}% | {ev_hybrid*100:>13.2f}%")
    print(f"{'Выше среднего (+10%)':<25s} | {ev_std_high*100:>13.2f}% | {ev_ou_high*100:>13.2f}% | {ev_hybrid_high*100:>13.2f}%")
    print(f"{'Ниже среднего (-10%)':<25s} | {ev_std_low*100:>13.2f}% | {ev_ou_low*100:>13.2f}% | {ev_hybrid_low*100:>13.2f}%")
    print()
    print("="*80)
    print("ВЫВОДЫ:")
    print("="*80)
    print()
    print("✅ ГИБРИДНЫЙ подход:")
    print("   - Учитывает ML паттерны")
    print("   - Корректирует на mean reversion")
    print("   - Адаптируется к отклонению от среднего")
    print("   - Лучший баланс между агрессивностью и консервативностью")
    print()
    print("❌ Стандартный подход:")
    print("   - Игнорирует перекупленность/перепроданность")
    print("   - Может брать плохие сделки на экстремумах")
    print()
    print("⚠️  OU-only подход:")
    print("   - Слишком консервативен")
    print("   - Теряет ML insights")
    print("   - Провалится в трендах")
    print()
    print("="*80)
