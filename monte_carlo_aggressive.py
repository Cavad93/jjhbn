#!/usr/bin/env python3
"""
Агрессивная Монте-Карло симуляция для проверки возможности 100-200% годовых

Пересмотр параметров:
1. Более агрессивные сценарии
2. Учёт compound interest
3. Более высокая частота сделок
4. Эффект от Haiku + DuckDuckGo
"""

import random
import math
from typing import List, Dict, Tuple

# Для воспроизводимости
random.seed(42)


class AggressiveMonteCarloSimulator:
    """Симулятор с более агрессивными параметрами"""

    def __init__(
        self,
        initial_capital: float = 1000.0,
        monthly_deposit: float = 300.0,
        simulation_days: int = 365,
        kelly_fraction: float = 0.25,
        max_position_size: float = 200.0,
        min_position_size: float = 15.0
    ):
        self.initial_capital = initial_capital
        self.monthly_deposit = monthly_deposit
        self.simulation_days = simulation_days
        self.kelly_fraction = kelly_fraction
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size

    def calculate_kelly_size(self, capital: float, win_rate: float, rr_ratio: float) -> float:
        """Рассчитать размер позиции по Kelly"""
        # Kelly formula: f = (p*b - q) / b
        p = win_rate
        q = 1 - win_rate
        b = rr_ratio

        kelly_full = (p * b - q) / b
        kelly = kelly_full * self.kelly_fraction

        # Ограничения
        kelly = max(0.005, min(kelly, 0.04))  # 0.5% - 4%

        position = capital * kelly
        return max(self.min_position_size, min(position, self.max_position_size))

    def simulate_trade(
        self,
        position_size: float,
        win_rate: float,
        rr_ratio: float,
        trailing_prob: float,
        trailing_bonus: float,
        commission: float,
        slippage: float
    ) -> Tuple[bool, float]:
        """Симулировать одну сделку"""

        is_win = random.random() < win_rate

        if is_win:
            # Проверка trailing stop
            if random.random() < trailing_prob:
                profit_mult = rr_ratio + trailing_bonus
            else:
                profit_mult = rr_ratio

            gross_profit = position_size * profit_mult
            costs = position_size * (commission + slippage) * 2
            return True, gross_profit - costs
        else:
            loss = position_size * 1.0
            costs = position_size * (commission + slippage) * 2
            return False, -(loss + costs)

    def simulate_trajectory(
        self,
        scenario_params: dict,
        seed: int
    ) -> dict:
        """Симулировать одну траекторию"""

        random.seed(seed)

        balance = self.initial_capital
        reserve = 0.0
        total_deposited = self.initial_capital

        max_balance = balance
        min_balance = balance

        total_trades = 0
        win_trades = 0

        daily_balances = []

        for day in range(self.simulation_days):
            # Ежемесячное пополнение
            if day > 0 and day % 30 == 0:
                balance += self.monthly_deposit
                total_deposited += self.monthly_deposit

            day_start_balance = balance

            # Количество сделок в день (Poisson-like)
            avg_trades = scenario_params['trades_per_day']
            num_trades = int(random.gauss(avg_trades, avg_trades * 0.3))
            num_trades = max(0, min(num_trades, 5))  # 0-5 сделок

            for _ in range(num_trades):
                if balance < self.min_position_size:
                    break

                # Размер позиции
                position_size = self.calculate_kelly_size(
                    balance,
                    scenario_params['win_rate'],
                    scenario_params['rr_ratio']
                )

                if position_size > balance * 0.9:
                    position_size = balance * 0.025

                # Симуляция сделки
                is_win, pnl = self.simulate_trade(
                    position_size,
                    scenario_params['win_rate'],
                    scenario_params['rr_ratio'],
                    scenario_params['trailing_prob'],
                    scenario_params['trailing_bonus'],
                    scenario_params['commission'],
                    scenario_params['slippage']
                )

                balance += pnl
                total_trades += 1

                if is_win:
                    win_trades += 1

                max_balance = max(max_balance, balance)
                min_balance = min(min_balance, balance)

            # Реинвестирование
            day_profit = balance - day_start_balance
            if day_profit > 0:
                to_reserve = day_profit * scenario_params['reserve_pct']
                reserve += to_reserve
                balance -= to_reserve
            elif day_profit < 0:
                loss = abs(day_profit)
                from_reserve = min(loss, reserve)
                reserve -= from_reserve
                balance += from_reserve

            daily_balances.append(balance + reserve)

        final_balance = balance + reserve
        total_return = ((final_balance - total_deposited) / total_deposited) * 100
        max_dd = ((max_balance - min_balance) / max_balance) * 100 if max_balance > 0 else 0
        actual_wr = (win_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            'final_balance': final_balance,
            'total_return': total_return,
            'net_profit': final_balance - total_deposited,
            'total_deposited': total_deposited,
            'max_drawdown': max_dd,
            'total_trades': total_trades,
            'win_rate': actual_wr,
            'daily_balances': daily_balances
        }

    def run_monte_carlo(
        self,
        scenario_params: dict,
        num_simulations: int = 10000
    ) -> List[dict]:
        """Запустить Монте-Карло симуляцию"""

        print(f"\n{'='*80}")
        print(f"Запуск {num_simulations:,} симуляций: {scenario_params['name']}")
        print(f"{'='*80}")

        results = []

        for i in range(num_simulations):
            result = self.simulate_trajectory(scenario_params, seed=i)
            results.append(result)

            if (i + 1) % 1000 == 0:
                print(f"  Прогресс: {i+1:,}/{num_simulations:,} ({(i+1)/num_simulations*100:.1f}%)")

        print(f"✅ Завершено!\n")
        return results

    def analyze_results(self, results: List[dict], scenario_name: str) -> dict:
        """Анализ результатов"""

        returns = sorted([r['total_return'] for r in results])
        final_balances = sorted([r['final_balance'] for r in results])
        net_profits = sorted([r['net_profit'] for r in results])

        total_deposited = results[0]['total_deposited']

        # Percentiles
        def percentile(data, p):
            idx = int(len(data) * p / 100)
            return data[idx]

        # Вероятности
        profitable = sum(1 for r in results if r['total_return'] > 0)
        doubled = sum(1 for r in results if r['total_return'] > 100)
        tripled = sum(1 for r in results if r['total_return'] > 200)

        analysis = {
            'scenario': scenario_name,
            'num_sims': len(results),
            'deposited': total_deposited,

            # Returns
            'return_mean': sum(returns) / len(returns),
            'return_median': percentile(returns, 50),
            'return_min': min(returns),
            'return_max': max(returns),
            'return_p5': percentile(returns, 5),
            'return_p10': percentile(returns, 10),
            'return_p25': percentile(returns, 25),
            'return_p75': percentile(returns, 75),
            'return_p90': percentile(returns, 90),
            'return_p95': percentile(returns, 95),
            'return_p99': percentile(returns, 99),

            # Net profit
            'profit_median': percentile(net_profits, 50),
            'profit_p95': percentile(net_profits, 95),
            'profit_max': max(net_profits),

            # Final balance
            'balance_median': percentile(final_balances, 50),
            'balance_p75': percentile(final_balances, 75),
            'balance_p90': percentile(final_balances, 90),
            'balance_p95': percentile(final_balances, 95),
            'balance_p99': percentile(final_balances, 99),
            'balance_max': max(final_balances),

            # Probabilities
            'prob_profit': (profitable / len(results)) * 100,
            'prob_double': (doubled / len(results)) * 100,
            'prob_triple': (tripled / len(results)) * 100,

            # Trades
            'avg_trades': sum(r['total_trades'] for r in results) / len(results),
            'avg_wr': sum(r['win_rate'] for r in results) / len(results)
        }

        return analysis

    def print_analysis(self, analysis: dict):
        """Вывод анализа"""

        print(f"\n{'='*80}")
        print(f"АНАЛИЗ: {analysis['scenario']}")
        print(f"{'='*80}\n")

        print(f"Симуляций: {analysis['num_sims']:,}")
        print(f"Всего внесено: ${analysis['deposited']:,.0f}")
        print()

        print("📊 ДОХОДНОСТЬ (ROI):")
        print(f"  Средняя:       {analysis['return_mean']:>7.1f}%")
        print(f"  Медиана:       {analysis['return_median']:>7.1f}%")
        print(f"  Минимум:       {analysis['return_min']:>7.1f}%")
        print(f"  Максимум:      {analysis['return_max']:>7.1f}%")
        print()
        print("  Процентили:")
        print(f"    P5  (5% хуже):        {analysis['return_p5']:>7.1f}%")
        print(f"    P10 (10% хуже):       {analysis['return_p10']:>7.1f}%")
        print(f"    P25 (25% хуже):       {analysis['return_p25']:>7.1f}%")
        print(f"    P75 (25% лучше):      {analysis['return_p75']:>7.1f}%")
        print(f"    P90 (10% лучше):      {analysis['return_p90']:>7.1f}%")
        print(f"    P95 (5% лучше):       {analysis['return_p95']:>7.1f}%")
        print(f"    P99 (1% лучше):       {analysis['return_p99']:>7.1f}%")
        print()

        print("💰 ФИНАЛЬНЫЙ БАЛАНС:")
        print(f"  Медиана:       ${analysis['balance_median']:>8,.0f}")
        print(f"  P75:           ${analysis['balance_p75']:>8,.0f}")
        print(f"  P90:           ${analysis['balance_p90']:>8,.0f}")
        print(f"  P95:           ${analysis['balance_p95']:>8,.0f}")
        print(f"  P99:           ${analysis['balance_p99']:>8,.0f}")
        print(f"  Максимум:      ${analysis['balance_max']:>8,.0f}")
        print()

        print("💵 ЧИСТАЯ ПРИБЫЛЬ:")
        print(f"  Медиана:       ${analysis['profit_median']:>8,.0f}")
        print(f"  P95:           ${analysis['profit_p95']:>8,.0f}")
        print(f"  Максимум:      ${analysis['profit_max']:>8,.0f}")
        print()

        print("🎯 ВЕРОЯТНОСТИ:")
        print(f"  Прибыль (>0%):         {analysis['prob_profit']:>6.2f}%")
        print(f"  Удвоение (>100%):      {analysis['prob_double']:>6.2f}%")
        print(f"  Утроение (>200%):      {analysis['prob_triple']:>6.2f}%")
        print()

        print("📈 СДЕЛКИ:")
        print(f"  Среднее количество:    {analysis['avg_trades']:>6.0f}")
        print(f"  Средний WR:            {analysis['avg_wr']:>6.2f}%")
        print()


def main():
    """Главная функция"""

    print("\n" + "="*80)
    print("АГРЕССИВНАЯ МОНТЕ-КАРЛО СИМУЛЯЦИЯ")
    print("Проверка возможности 100-200% годовых")
    print("="*80)
    print()
    print("Параметры:")
    print("  - Начальный капитал: $1,000")
    print("  - Ежемесячное пополнение: $300")
    print("  - Период: 365 дней")
    print("  - Всего будет внесено: $4,600")
    print("  - Симуляций на сценарий: 10,000")
    print()

    simulator = AggressiveMonteCarloSimulator(
        initial_capital=1000.0,
        monthly_deposit=300.0,
        simulation_days=365,
        kelly_fraction=0.25,
        max_position_size=200.0,
        min_position_size=15.0
    )

    # ========================================================================
    # СЦЕНАРИЙ 1: ОЧЕНЬ ОПТИМИСТИЧНЫЙ (АГРЕССИВНЫЙ)
    # ========================================================================
    # Haiku работает отлично + Bull market + Высокая частота
    scenario_aggressive = {
        'name': '1. ОЧЕНЬ ОПТИМИСТИЧНЫЙ (Агрессивный)',
        'win_rate': 0.60,           # 60% WR (отличная работа Haiku + ML)
        'rr_ratio': 1.667,          # TP/SL = 2.5/1.5
        'trades_per_day': 1.5,      # 1.5 сделки/день (много возможностей)
        'trailing_prob': 0.30,      # 30% сделок с trailing stop
        'trailing_bonus': 0.5,      # +0.5R бонус от trailing
        'commission': 0.001,        # 0.1%
        'slippage': 0.0005,         # 0.05%
        'reserve_pct': 0.25         # 25% в резерв (больше остаётся в торговле)
    }

    # ========================================================================
    # СЦЕНАРИЙ 2: ОПТИМИСТИЧНЫЙ
    # ========================================================================
    scenario_optimistic = {
        'name': '2. ОПТИМИСТИЧНЫЙ',
        'win_rate': 0.58,           # 58% WR
        'rr_ratio': 1.667,
        'trades_per_day': 1.2,      # 1.2 сделки/день
        'trailing_prob': 0.25,      # 25% с trailing
        'trailing_bonus': 0.4,      # +0.4R
        'commission': 0.001,
        'slippage': 0.0005,
        'reserve_pct': 0.30         # 30% в резерв
    }

    # ========================================================================
    # СЦЕНАРИЙ 3: РЕАЛИСТИЧНЫЙ
    # ========================================================================
    scenario_realistic = {
        'name': '3. РЕАЛИСТИЧНЫЙ',
        'win_rate': 0.53,           # 53% WR
        'rr_ratio': 1.667,
        'trades_per_day': 0.9,      # 0.9 сделки/день
        'trailing_prob': 0.15,      # 15% с trailing
        'trailing_bonus': 0.3,      # +0.3R
        'commission': 0.001,
        'slippage': 0.0005,
        'reserve_pct': 0.30
    }

    scenarios = [scenario_aggressive, scenario_optimistic, scenario_realistic]
    all_analyses = []

    for scenario in scenarios:
        print(f"\n{'─'*80}")
        print(f"СЦЕНАРИЙ: {scenario['name']}")
        print(f"{'─'*80}")
        print(f"  Win Rate: {scenario['win_rate']*100:.0f}%")
        print(f"  Сделок/день: {scenario['trades_per_day']:.1f}")
        print(f"  Trailing prob: {scenario['trailing_prob']*100:.0f}%")

        results = simulator.run_monte_carlo(scenario, num_simulations=10000)
        analysis = simulator.analyze_results(results, scenario['name'])
        all_analyses.append(analysis)

        simulator.print_analysis(analysis)

    # ========================================================================
    # СРАВНИТЕЛЬНАЯ ТАБЛИЦА
    # ========================================================================
    print(f"\n{'='*80}")
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print(f"{'='*80}\n")

    print(f"{'Метрика':<35} {'Агрессивный':>15} {'Оптимист.':>15} {'Реалист.':>15}")
    print(f"{'-'*35} {'-'*15} {'-'*15} {'-'*15}")

    metrics = [
        ('Медианная доходность', 'return_median', '%'),
        ('P90 доходность', 'return_p90', '%'),
        ('P95 доходность', 'return_p95', '%'),
        ('Вероятность прибыли', 'prob_profit', '%'),
        ('Вероятность удвоения (>100%)', 'prob_double', '%'),
        ('Вероятность утроения (>200%)', 'prob_triple', '%'),
        ('Медианный баланс', 'balance_median', '$'),
        ('P95 баланс', 'balance_p95', '$'),
    ]

    for metric_name, key, unit in metrics:
        a = all_analyses[0][key]
        o = all_analyses[1][key]
        r = all_analyses[2][key]

        if unit == '%':
            print(f"{metric_name:<35} {a:>14.1f}% {o:>14.1f}% {r:>14.1f}%")
        elif unit == '$':
            print(f"{metric_name:<35} ${a:>13,.0f} ${o:>13,.0f} ${r:>13,.0f}")

    # ========================================================================
    # ВЫВОДЫ
    # ========================================================================
    print(f"\n{'='*80}")
    print("ВЫВОДЫ: ВОЗМОЖНЫ ЛИ 100-200% ГОДОВЫХ?")
    print(f"{'='*80}\n")

    agg = all_analyses[0]
    opt = all_analyses[1]
    real = all_analyses[2]

    print("🎯 АГРЕССИВНЫЙ СЦЕНАРИЙ (WR 60%, 1.5 сделок/день):")
    print(f"   Медианная доходность:     {agg['return_median']:.1f}%")
    print(f"   P90 доходность:           {agg['return_p90']:.1f}%")
    print(f"   P95 доходность:           {agg['return_p95']:.1f}%")
    print(f"   Вероятность >100%:        {agg['prob_double']:.1f}%")
    print(f"   Вероятность >200%:        {agg['prob_triple']:.1f}%")

    if agg['return_median'] >= 100:
        print(f"   ✅ Медиана >100% - ОЧЕНЬ ВЕРОЯТНО удвоение!")
    elif agg['return_p75'] >= 100:
        print(f"   ✅ P75 >100% - 25% шанс удвоения!")
    elif agg['return_p90'] >= 100:
        print(f"   ✅ P90 >100% - 10% шанс удвоения!")

    print()

    print("📊 ОПТИМИСТИЧНЫЙ СЦЕНАРИЙ (WR 58%, 1.2 сделок/день):")
    print(f"   Медианная доходность:     {opt['return_median']:.1f}%")
    print(f"   P90 доходность:           {opt['return_p90']:.1f}%")
    print(f"   P95 доходность:           {opt['return_p95']:.1f}%")
    print(f"   Вероятность >100%:        {opt['prob_double']:.1f}%")
    print(f"   Вероятность >200%:        {opt['prob_triple']:.1f}%")

    if opt['return_p90'] >= 100:
        print(f"   ✅ P90 >100% - реально достичь удвоения!")

    print()

    print("🔵 РЕАЛИСТИЧНЫЙ СЦЕНАРИЙ (WR 53%, 0.9 сделок/день):")
    print(f"   Медианная доходность:     {real['return_median']:.1f}%")
    print(f"   P90 доходность:           {real['return_p90']:.1f}%")
    print(f"   P95 доходность:           {real['return_p95']:.1f}%")
    print(f"   Вероятность >100%:        {real['prob_double']:.1f}%")

    print()
    print("="*80)
    print("ИТОГОВЫЙ ОТВЕТ:")
    print("="*80)

    if agg['prob_double'] >= 50:
        print("✅ ДА! 100-200% ВПОЛНЕ ВОЗМОЖНО в агрессивном сценарии!")
        print(f"   Вероятность удвоения: {agg['prob_double']:.0f}%")
    elif opt['prob_double'] >= 25:
        print("✅ ДА! 100-200% ВОЗМОЖНО в оптимистичном сценарии!")
        print(f"   Вероятность удвоения: {opt['prob_double']:.0f}%")
    elif real['prob_double'] >= 10:
        print("🟡 ВОЗМОЖНО, но маловероятно даже в реалистичном сценарии")
        print(f"   Вероятность удвоения: {real['prob_double']:.0f}%")
    else:
        print("⚠️  100-200% маловероятно во всех сценариях")

    print()
    print("💡 УСЛОВИЯ ДЛЯ 100-200%:")
    print("   1. Win Rate должен быть 58-60% (отличная работа Haiku)")
    print("   2. Высокая частота сделок: 1.2-1.5/день")
    print("   3. Эффективный trailing stop: 25-30% сделок")
    print("   4. Bull market с множеством возможностей")
    print("   5. Минимальные комиссии и проскальзывание")
    print("   6. Geometric compounding работает в плюс")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
