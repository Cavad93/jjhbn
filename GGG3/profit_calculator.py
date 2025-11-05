#!/usr/bin/env python3
"""
Калькулятор прибыльности для PancakeSwap Prediction

Показывает:
1. Какой winrate нужен для прибыльности (с учетом комиссии 3%)
2. Какую месячную доходность можно ожидать при разных winrate
3. Влияние размера ставки и Kelly criterion
"""

def calculate_roi(winrate, n_bets=1000, commission=0.03):
    """
    Рассчитывает ROI с учетом комиссии PancakeSwap

    Args:
        winrate: Процент выигрышей (0.0 - 1.0)
        n_bets: Количество ставок
        commission: Комиссия платформы (3% = 0.03)

    Returns:
        total_roi: Общий ROI в USDT
        roi_per_bet: ROI на одну ставку
    """
    wins = int(n_bets * winrate)
    losses = n_bets - wins

    # При выигрыше получаем ~2x минус комиссия
    # Если ставка 1 USDT, выигрыш = 2 * (1 - commission) - 1 = 0.97
    profit_per_win = 1.0 * (1 - commission)  # Чистая прибыль

    # При проигрыше теряем всю ставку
    loss_per_loss = 1.0

    total_profit = wins * profit_per_win
    total_loss = losses * loss_per_loss
    total_roi = total_profit - total_loss

    roi_per_bet = total_roi / n_bets if n_bets > 0 else 0

    return total_roi, roi_per_bet


def break_even_winrate(commission=0.03):
    """Минимальный winrate для безубыточности"""
    # Для безубыточности: wins * (1 - commission) = losses
    # wins / (wins + losses) = ?
    # Обозначим winrate = w
    # w * (1 - commission) = (1 - w)
    # w * (1 - commission) + w = 1
    # w = 1 / (2 - commission)
    return 1.0 / (2.0 - commission)


def kelly_criterion(winrate, odds=2.0, commission=0.03):
    """
    Kelly Criterion для оптимального размера ставки

    Args:
        winrate: Вероятность выигрыша
        odds: Коэффициент (2.0 для 50/50 после комиссии)
        commission: Комиссия

    Returns:
        Оптимальная доля банкролла для ставки (0.0 - 1.0)
    """
    # Adjusted odds after commission
    net_odds = odds * (1 - commission)

    # Kelly: f = (p * net_odds - q) / net_odds
    # где p = winrate, q = 1 - winrate
    p = winrate
    q = 1 - winrate

    kelly = (p * net_odds - q) / net_odds

    # Ограничиваем (безопасный Kelly = 50% от полного)
    kelly = max(0, min(kelly, 1.0))

    return kelly


def simulate_month(winrate, initial_bankroll=100, bet_size_pct=0.05,
                   bets_per_day=288, days=30):
    """
    Симулирует торговлю в течение месяца

    Args:
        winrate: Процент выигрышей
        initial_bankroll: Начальный капитал в USDT
        bet_size_pct: Процент от банкролла на ставку
        bets_per_day: Количество ставок в день (288 = каждые 5 минут)
        days: Количество дней

    Returns:
        final_bankroll, monthly_roi, monthly_pct
    """
    import random
    random.seed(42)

    bankroll = initial_bankroll
    commission = 0.03

    total_bets = bets_per_day * days

    for _ in range(total_bets):
        if bankroll <= 0:
            break

        bet = bankroll * bet_size_pct

        # Outcome
        won = random.random() < winrate

        if won:
            bankroll += bet * (1 - commission)
        else:
            bankroll -= bet

    final_roi = bankroll - initial_bankroll
    monthly_pct = (final_roi / initial_bankroll) * 100

    return bankroll, final_roi, monthly_pct


def main():
    """Главная функция"""
    print("="*70)
    print("💰 КАЛЬКУЛЯТОР ПРИБЫЛЬНОСТИ - PancakeSwap Prediction")
    print("="*70)

    commission = 0.03  # 3% комиссия

    # Минимальный winrate для безубыточности
    breakeven = break_even_winrate(commission)
    print(f"\n📊 Минимальный winrate для безубыточности: {breakeven:.4f} ({breakeven*100:.2f}%)")
    print(f"   С комиссией {commission*100}%, нужно выигрывать >{breakeven*100:.2f}% для прибыли")

    # Таблица winrate vs ROI
    print(f"\n{'='*70}")
    print(f"📈 WINRATE vs ROI (на 1000 ставок)")
    print(f"{'='*70}")
    print(f"{'Winrate':>10} {'ROI Total':>12} {'ROI/bet':>10} {'Kelly %':>10} {'Статус':>15}")
    print(f"{'-'*70}")

    winrates = [0.48, 0.50, 0.51, 0.515, 0.52, 0.525, 0.53, 0.54, 0.55, 0.56, 0.58, 0.60]

    for wr in winrates:
        total_roi, roi_per_bet = calculate_roi(wr, n_bets=1000, commission=commission)
        kelly_pct = kelly_criterion(wr, commission=commission) * 100

        if total_roi > 0:
            status = "✅ ПРИБЫЛЬ"
        elif total_roi == 0:
            status = "⚖️  БЕЗУБЫТОК"
        else:
            status = "❌ УБЫТОК"

        print(f"{wr*100:>9.1f}% {total_roi:>+11.2f} {roi_per_bet:>+9.4f} {kelly_pct:>9.1f}% {status:>15}")

    # Месячная доходность при разных winrate
    print(f"\n{'='*70}")
    print(f"📅 ПРОГНОЗ МЕСЯЧНОЙ ДОХОДНОСТИ (начальный банк: 100 USDT)")
    print(f"{'='*70}")
    print(f"{'Winrate':>10} {'Ставка/банк':>12} {'Конечный $':>12} {'ROI':>10} {'Месяц %':>10}")
    print(f"{'-'*70}")

    test_winrates = [0.515, 0.52, 0.525, 0.53, 0.54, 0.55]
    bet_sizes = [0.02, 0.03, 0.05]  # 2%, 3%, 5% от банкролла

    for wr in test_winrates:
        kelly_optimal = kelly_criterion(wr, commission=commission)

        # Используем консервативный размер (половина от Kelly)
        bet_size = min(0.05, kelly_optimal * 0.5)

        final, roi, pct = simulate_month(
            winrate=wr,
            initial_bankroll=100,
            bet_size_pct=bet_size,
            bets_per_day=288,  # Каждые 5 минут
            days=30
        )

        print(f"{wr*100:>9.1f}% {bet_size*100:>11.1f}% {final:>11.2f} {roi:>+9.2f} {pct:>+9.1f}%")

    # Выводы
    print(f"\n{'='*70}")
    print(f"🎯 КЛЮЧЕВЫЕ ВЫВОДЫ")
    print(f"{'='*70}\n")

    print(f"1. МИНИМАЛЬНЫЙ WINRATE для прибыли: {breakeven*100:.2f}%")
    print(f"   Все что ниже → убыточно из-за комиссии 3%\n")

    print(f"2. РЕАЛИСТИЧНЫЕ ЦЕЛИ:")
    print(f"   • Winrate 52-53% → 2-5% в месяц (консервативно)")
    print(f"   • Winrate 54-55% → 5-10% в месяц (оптимистично)")
    print(f"   • Winrate 56%+   → 10-20% в месяц (очень сложно достичь)\n")

    print(f"3. УПРАВЛЕНИЕ КАПИТАЛОМ:")
    print(f"   • Используйте Kelly Criterion для оптимального размера ставки")
    print(f"   • Безопасно: 25-50% от полного Kelly")
    print(f"   • Максимум 5% от банкролла на одну ставку\n")

    print(f"4. РИСКИ:")
    print(f"   • Variance: даже при 55% winrate возможны серии из 10+ проигрышей")
    print(f"   • Drawdown: будьте готовы к просадкам 15-25%")
    print(f"   • Overfitting: тестовый winrate ≠ реальный winrate\n")

    print(f"5. NEXT STEPS:")
    print(f"   ✅ Обучить модель на реальных исторических данных")
    print(f"   ✅ Проверить winrate на hold-out тестовой выборке")
    print(f"   ✅ Paper trading 1 месяц (без реальных ставок)")
    print(f"   ✅ Если paper trading winrate > 52% → малый капитал (50-100 USDT)")
    print(f"   ✅ Постепенное масштабирование по мере подтверждения результатов")

    print(f"\n{'='*70}\n")

    # Интерактивный расчет
    print("💡 Хотите рассчитать свой сценарий?")
    print("   Отредактируйте этот скрипт и измените параметры:")
    print("   - initial_bankroll: Ваш начальный капитал")
    print("   - winrate: Ожидаемый winrate модели")
    print("   - bet_size_pct: Процент от банка на ставку")


if __name__ == "__main__":
    main()
