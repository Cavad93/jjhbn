#!/usr/bin/env python3
"""
Простое сравнение: Стандартный EV vs Гибридный (ML + OU)
"""

from math import log, exp, sqrt

# ============================================================================
# СТАНДАРТНЫЙ EV
# ============================================================================

def standard_ev(p_up, tp_mult=2.5, sl_mult=1.5):
    """Стандартная формула EV"""
    real_tp = tp_mult - 0.003  # С учётом комиссий
    real_sl = sl_mult + 0.003
    ev_long = p_up * real_tp - (1.0 - p_up) * real_sl
    return ev_long

# ============================================================================
# ГИБРИДНЫЙ EV (ML + OU коррекция)
# ============================================================================

def hybrid_ev_correction(p_up_ml, current_price, mu_price, theta, scaling=0.15):
    """
    Гибридная коррекция вероятности на основе OU

    Args:
        p_up_ml: ML вероятность (0-1)
        current_price: Текущая цена
        mu_price: Долгосрочное среднее (в обычной цене, не log)
        theta: Скорость mean reversion (обычно 0.1-0.5)
        scaling: Масштаб коррекции

    Returns:
        p_up_corrected: Скорректированная вероятность
    """
    # Отклонение от среднего (в %)
    deviation_pct = (current_price - mu_price) / mu_price

    # Коррекция: если выше среднего → снижаем p_up, если ниже → повышаем
    correction = -theta * deviation_pct * scaling

    # Применяем
    p_up_corrected = p_up_ml + correction

    # Ограничиваем [0, 1]
    p_up_corrected = max(0.0, min(1.0, p_up_corrected))

    return p_up_corrected


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

print("="*80)
print("СРАВНЕНИЕ: Стандартный EV vs Гибридный EV (ML + OU)")
print("="*80)
print()

# Параметры OU (оценены из истории)
mu_price = 50000  # Долгосрочное среднее BTC
theta = 0.20  # Скорость mean reversion (0.20 = умеренная)

# ML предсказание (одинаково для всех)
p_up_ml = 0.65

print(f"ML модель предсказывает: p_up = {p_up_ml:.1%} (для всех случаев)")
print(f"OU параметры: среднее μ = ${mu_price:,.0f}, скорость θ = {theta}")
print()

# ========================================================================
# СЦЕНАРИЙ 1: Цена около среднего
# ========================================================================
print("="*80)
print("СЦЕНАРИЙ 1: Цена около среднего ($50,000)")
print("="*80)

current_1 = 50000
dev_1 = (current_1 - mu_price) / mu_price * 100

p_up_hybrid_1 = hybrid_ev_correction(p_up_ml, current_1, mu_price, theta)
ev_std_1 = standard_ev(p_up_ml)
ev_hybrid_1 = standard_ev(p_up_hybrid_1)

print(f"Цена: ${current_1:,.0f} (отклонение: {dev_1:+.1f}%)")
print(f"  Стандартный: p_up = {p_up_ml:.2%} → EV = {ev_std_1*100:+.2f}%")
print(f"  Гибридный:   p_up = {p_up_hybrid_1:.2%} → EV = {ev_hybrid_1*100:+.2f}%")
print(f"  Разница: {(ev_hybrid_1 - ev_std_1)*100:+.2f}%")
print()

# ========================================================================
# СЦЕНАРИЙ 2: Цена выше среднего (перекуплена)
# ========================================================================
print("="*80)
print("СЦЕНАРИЙ 2: Цена выше среднего ($55,000) - ПЕРЕКУПЛЕНА")
print("="*80)

current_2 = 55000
dev_2 = (current_2 - mu_price) / mu_price * 100

p_up_hybrid_2 = hybrid_ev_correction(p_up_ml, current_2, mu_price, theta)
ev_std_2 = standard_ev(p_up_ml)
ev_hybrid_2 = standard_ev(p_up_hybrid_2)

print(f"Цена: ${current_2:,.0f} (отклонение: {dev_2:+.1f}%)")
print(f"  Стандартный: p_up = {p_up_ml:.2%} → EV = {ev_std_2*100:+.2f}%")
print(f"  Гибридный:   p_up = {p_up_hybrid_2:.2%} → EV = {ev_hybrid_2*100:+.2f}%")
print(f"  Разница: {(ev_hybrid_2 - ev_std_2)*100:+.2f}%")
print(f"  💡 OU коррекция СНИЗИЛА EV (mean reversion вниз)")
print()

# ========================================================================
# СЦЕНАРИЙ 3: Цена сильно выше среднего
# ========================================================================
print("="*80)
print("СЦЕНАРИЙ 3: Цена сильно выше среднего ($60,000) - ОЧЕНЬ ПЕРЕКУПЛЕНА")
print("="*80)

current_3 = 60000
dev_3 = (current_3 - mu_price) / mu_price * 100

p_up_hybrid_3 = hybrid_ev_correction(p_up_ml, current_3, mu_price, theta)
ev_std_3 = standard_ev(p_up_ml)
ev_hybrid_3 = standard_ev(p_up_hybrid_3)

print(f"Цена: ${current_3:,.0f} (отклонение: {dev_3:+.1f}%)")
print(f"  Стандартный: p_up = {p_up_ml:.2%} → EV = {ev_std_3*100:+.2f}%")
print(f"  Гибридный:   p_up = {p_up_hybrid_3:.2%} → EV = {ev_hybrid_3*100:+.2f}%")
print(f"  Разница: {(ev_hybrid_3 - ev_std_3)*100:+.2f}%")
print(f"  💡 Сильная коррекция! Защита от плохих сделок")
print()

# ========================================================================
# СЦЕНАРИЙ 4: Цена ниже среднего (перепродана)
# ========================================================================
print("="*80)
print("СЦЕНАРИЙ 4: Цена ниже среднего ($45,000) - ПЕРЕПРОДАНА")
print("="*80)

current_4 = 45000
dev_4 = (current_4 - mu_price) / mu_price * 100

p_up_hybrid_4 = hybrid_ev_correction(p_up_ml, current_4, mu_price, theta)
ev_std_4 = standard_ev(p_up_ml)
ev_hybrid_4 = standard_ev(p_up_hybrid_4)

print(f"Цена: ${current_4:,.0f} (отклонение: {dev_4:+.1f}%)")
print(f"  Стандартный: p_up = {p_up_ml:.2%} → EV = {ev_std_4*100:+.2f}%")
print(f"  Гибридный:   p_up = {p_up_hybrid_4:.2%} → EV = {ev_hybrid_4*100:+.2f}%")
print(f"  Разница: {(ev_hybrid_4 - ev_std_4)*100:+.2f}%")
print(f"  💡 OU коррекция ПОВЫСИЛА EV (mean reversion вверх)")
print()

# ========================================================================
# СЦЕНАРИЙ 5: Цена сильно ниже среднего
# ========================================================================
print("="*80)
print("СЦЕНАРИЙ 5: Цена сильно ниже среднего ($40,000) - ОЧЕНЬ ПЕРЕПРОДАНА")
print("="*80)

current_5 = 40000
dev_5 = (current_5 - mu_price) / mu_price * 100

p_up_hybrid_5 = hybrid_ev_correction(p_up_ml, current_5, mu_price, theta)
ev_std_5 = standard_ev(p_up_ml)
ev_hybrid_5 = standard_ev(p_up_hybrid_5)

print(f"Цена: ${current_5:,.0f} (отклонение: {dev_5:+.1f}%)")
print(f"  Стандартный: p_up = {p_up_ml:.2%} → EV = {ev_std_5*100:+.2f}%")
print(f"  Гибридный:   p_up = {p_up_hybrid_5:.2%} → EV = {ev_hybrid_5*100:+.2f}%")
print(f"  Разница: {(ev_hybrid_5 - ev_std_5)*100:+.2f}%")
print(f"  💡 Сильное увеличение! Не упустим хорошие сделки")
print()

# ========================================================================
# ИТОГОВАЯ ТАБЛИЦА
# ========================================================================
print("="*80)
print("ИТОГОВАЯ ТАБЛИЦА")
print("="*80)
print()
print(f"{'Цена':<12s} | {'Откл.':<8s} | {'p_up ML':<10s} | {'p_up Гибрид':<12s} | {'EV Станд.':<12s} | {'EV Гибрид':<12s} | {'Δ':<8s}")
print("-"*80)

cases = [
    (40000, p_up_ml, p_up_hybrid_5, ev_std_5, ev_hybrid_5),
    (45000, p_up_ml, p_up_hybrid_4, ev_std_4, ev_hybrid_4),
    (50000, p_up_ml, p_up_hybrid_1, ev_std_1, ev_hybrid_1),
    (55000, p_up_ml, p_up_hybrid_2, ev_std_2, ev_hybrid_2),
    (60000, p_up_ml, p_up_hybrid_3, ev_std_3, ev_hybrid_3),
]

for price, p_ml, p_hyb, ev_std, ev_hyb in cases:
    dev = (price - mu_price) / mu_price * 100
    diff = (ev_hyb - ev_std) * 100
    print(f"${price:>6,} | {dev:>+6.1f}% | {p_ml:>8.1%} | {p_hyb:>10.1%} | {ev_std*100:>+10.2f}% | {ev_hyb*100:>+10.2f}% | {diff:>+6.2f}%")

print()
print("="*80)
print("ВЫВОДЫ:")
print("="*80)
print()
print("✅ ГИБРИДНЫЙ подход автоматически:")
print("   1. СНИЖАЕТ EV для перекупленных активов → защита от плохих сделок")
print("   2. ПОВЫШАЕТ EV для перепроданных активов → не упускаем возможности")
print("   3. Сохраняет EV около среднего → доверяем ML")
print()
print("❌ Стандартный подход:")
print("   1. Игнорирует mean reversion")
print("   2. Одинаковый EV для $40k и $60k (если ML даёт ту же p_up)")
print("   3. Может брать плохие сделки на экстремумах")
print()
print("💡 РЕКОМЕНДАЦИЯ:")
print("   Добавить OU коррекцию как ДОПОЛНЕНИЕ к ML предсказаниям!")
print("   Это улучшит качество топ-20 в mean-reverting рынках.")
print("="*80)
