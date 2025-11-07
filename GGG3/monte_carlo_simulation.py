#!/usr/bin/env python3
"""
Симуляция доходности бота методом Монте-Карло

Симулирует 10,000 траекторий для 3 сценариев:
- Оптимистичный (WR=60%, хорошие условия)
- Реалистичный (WR=52%, средние условия)
- Пессимистичный (WR=48%, плохие условия)

Учитывает:
- Kelly sizing (quarter Kelly)
- R/R ratio 2.5/1.5 = 1.667
- Trailing stop (2% активация, 1% дистанция)
- Ребалансировка каждые 4 часа
- Максимум 10 позиций одновременно
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time
from collections import defaultdict


@dataclass
class ScenarioParams:
    """Параметры сценария"""
    name: str
    win_rate: float                      # Вероятность выигрыша
    avg_trades_per_day: float            # Среднее количество сделок в день
    tp_multiplier: float                 # TP ATR множитель
    sl_multiplier: float                 # SL ATR множитель
    trailing_stop_prob: float            # Вероятность активации trailing stop
    trailing_stop_bonus: float           # Бонус от trailing stop (в R/R)
    commission: float                    # Комиссия за сделку
    slippage: float                      # Проскальзывание
    max_drawdown_before_stop: float      # Макс просадка до остановки торговли


@dataclass
class SimulationResult:
    """Результат одной симуляции"""
    scenario_name: str
    final_balance: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    win_trades: int
    loss_trades: int
    actual_win_rate: float
    max_balance: float
    min_balance: float
    days_to_bust: int  # 0 если не обанкротился


class MonteCarloSimulator:
    """Симулятор Монте-Карло для бота"""

    def __init__(
        self,
        initial_capital: float = 1000.0,
        max_positions: int = 10,
        min_position_size: float = 15.0,
        max_position_size: float = 200.0,
        kelly_fraction: float = 0.25,
        simulation_days: int = 90
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.kelly_fraction = kelly_fraction
        self.simulation_days = simulation_days

    def calculate_position_size(
        self,
        capital: float,
        win_rate: float,
        rr_ratio: float
    ) -> float:
        """Рассчитывает размер позиции по Kelly Criterion"""

        # Kelly formula: f = (p * b - q) / b
        p_win = win_rate
        p_loss = 1.0 - win_rate

        kelly_full = (p_win * rr_ratio - p_loss) / rr_ratio

        # Fractional Kelly
        kelly = kelly_full * self.kelly_fraction

        # Ограничения
        kelly = max(0.005, min(kelly, 0.025))  # 0.5% - 2.5%

        # Размер позиции
        position_size = capital * kelly

        # Ограничения по размеру
        position_size = max(self.min_position_size, min(position_size, self.max_position_size))

        return position_size

    def simulate_trade(
        self,
        params: ScenarioParams,
        position_size: float
    ) -> Tuple[bool, float]:
        """
        Симулирует одну сделку

        Returns:
            (is_win, pnl)
        """

        # Определяем выигрыш/проигрыш
        is_win = np.random.random() < params.win_rate

        # Рассчитываем R/R ratio
        rr_ratio = params.tp_multiplier / params.sl_multiplier

        if is_win:
            # Проверяем активацию trailing stop
            if np.random.random() < params.trailing_stop_prob:
                # Trailing stop сработал - бонус к профиту
                profit_multiplier = rr_ratio + params.trailing_stop_bonus
            else:
                # Обычный TP
                profit_multiplier = rr_ratio

            # Прибыль
            gross_profit = position_size * profit_multiplier

            # Вычитаем комиссии и проскальзывание
            commission_cost = position_size * params.commission * 2  # Вход + выход
            slippage_cost = position_size * params.slippage * 2

            net_profit = gross_profit - commission_cost - slippage_cost

            return True, net_profit
        else:
            # Убыток
            loss = position_size  # Теряем 1R (SL)

            # Вычитаем комиссии и проскальзывание
            commission_cost = position_size * params.commission * 2
            slippage_cost = position_size * params.slippage * 2

            net_loss = -(loss + commission_cost + slippage_cost)

            return False, net_loss

    def simulate_single_trajectory(
        self,
        params: ScenarioParams,
        seed: int = None
    ) -> SimulationResult:
        """Симулирует одну траекторию капитала"""

        if seed is not None:
            np.random.seed(seed)

        # Начальные значения
        balance = self.initial_capital
        max_balance = self.initial_capital
        min_balance = self.initial_capital

        balances = [balance]
        daily_returns = []

        total_trades = 0
        win_trades = 0
        loss_trades = 0

        days_to_bust = 0
        is_busted = False

        # R/R ratio
        rr_ratio = params.tp_multiplier / params.sl_multiplier

        # Симулируем каждый день
        for day in range(self.simulation_days):
            day_start_balance = balance

            # Количество сделок в день (Пуассоновское распределение)
            num_trades_today = np.random.poisson(params.avg_trades_per_day)
            num_trades_today = min(num_trades_today, self.max_positions * 3)  # Максимум 30 сделок в день

            for _ in range(num_trades_today):
                # Проверка на банкротство
                if balance < self.min_position_size:
                    is_busted = True
                    days_to_bust = day
                    break

                # Проверка на макс. просадку
                current_drawdown = (max_balance - balance) / max_balance
                if current_drawdown >= params.max_drawdown_before_stop:
                    is_busted = True
                    days_to_bust = day
                    break

                # Рассчитываем размер позиции
                position_size = self.calculate_position_size(
                    capital=balance,
                    win_rate=params.win_rate,
                    rr_ratio=rr_ratio
                )

                # Проверка достаточности капитала
                if position_size > balance * 0.9:  # Не используем больше 90% капитала
                    position_size = balance * 0.025  # Используем минимальный риск

                # Симулируем сделку
                is_win, pnl = self.simulate_trade(params, position_size)

                # Обновляем баланс
                balance += pnl
                total_trades += 1

                if is_win:
                    win_trades += 1
                else:
                    loss_trades += 1

                # Обновляем макс/мин баланс
                max_balance = max(max_balance, balance)
                min_balance = min(min_balance, balance)

            if is_busted:
                break

            # Записываем баланс на конец дня
            balances.append(balance)

            # Дневная доходность
            if day_start_balance > 0:
                daily_return = (balance - day_start_balance) / day_start_balance
                daily_returns.append(daily_return)

        # Финальные метрики
        final_balance = balance
        total_return_pct = ((final_balance - self.initial_capital) / self.initial_capital) * 100

        # Максимальная просадка
        if max_balance > 0:
            max_drawdown_pct = ((max_balance - min_balance) / max_balance) * 100
        else:
            max_drawdown_pct = 0

        # Sharpe ratio (предполагаем risk-free rate = 0)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365)
        else:
            sharpe_ratio = 0.0

        # Actual win rate
        actual_win_rate = win_trades / total_trades if total_trades > 0 else 0.0

        return SimulationResult(
            scenario_name=params.name,
            final_balance=final_balance,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            total_trades=total_trades,
            win_trades=win_trades,
            loss_trades=loss_trades,
            actual_win_rate=actual_win_rate,
            max_balance=max_balance,
            min_balance=min_balance,
            days_to_bust=days_to_bust if is_busted else 0
        )

    def run_monte_carlo(
        self,
        params: ScenarioParams,
        num_simulations: int = 10000,
        show_progress: bool = True
    ) -> List[SimulationResult]:
        """Запускает Монте-Карло симуляции"""

        results = []

        print(f"\n{'='*80}")
        print(f"Запуск {num_simulations:,} симуляций для сценария: {params.name}")
        print(f"{'='*80}")

        start_time = time.time()

        for i in range(num_simulations):
            result = self.simulate_single_trajectory(params, seed=i)
            results.append(result)

            # Progress bar
            if show_progress and (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (num_simulations - i - 1) / rate
                print(f"  Прогресс: {i+1:,}/{num_simulations:,} ({(i+1)/num_simulations*100:.1f}%) | "
                      f"Скорость: {rate:.0f} sim/s | Осталось: {remaining:.0f}s")

        elapsed = time.time() - start_time
        print(f"\n✅ Завершено за {elapsed:.1f}s ({num_simulations/elapsed:.0f} sim/s)")

        return results

    def analyze_results(
        self,
        results: List[SimulationResult]
    ) -> Dict:
        """Анализирует результаты симуляций"""

        # Извлекаем метрики
        final_balances = [r.final_balance for r in results]
        returns = [r.total_return_pct for r in results]
        drawdowns = [r.max_drawdown_pct for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results if r.sharpe_ratio != 0]

        # Банкротства
        busts = sum(1 for r in results if r.final_balance < 50)
        bust_rate = busts / len(results) * 100

        # Прибыльные траектории
        profitable = sum(1 for r in results if r.total_return_pct > 0)
        profitable_rate = profitable / len(results) * 100

        # Статистика
        analysis = {
            'scenario_name': results[0].scenario_name,
            'num_simulations': len(results),

            # Final Balance
            'final_balance_mean': np.mean(final_balances),
            'final_balance_median': np.median(final_balances),
            'final_balance_std': np.std(final_balances),
            'final_balance_min': np.min(final_balances),
            'final_balance_max': np.max(final_balances),
            'final_balance_p5': np.percentile(final_balances, 5),
            'final_balance_p25': np.percentile(final_balances, 25),
            'final_balance_p75': np.percentile(final_balances, 75),
            'final_balance_p95': np.percentile(final_balances, 95),

            # Returns
            'return_mean': np.mean(returns),
            'return_median': np.median(returns),
            'return_std': np.std(returns),
            'return_min': np.min(returns),
            'return_max': np.max(returns),
            'return_p5': np.percentile(returns, 5),
            'return_p25': np.percentile(returns, 25),
            'return_p75': np.percentile(returns, 75),
            'return_p95': np.percentile(returns, 95),

            # Drawdowns
            'drawdown_mean': np.mean(drawdowns),
            'drawdown_median': np.median(drawdowns),
            'drawdown_max': np.max(drawdowns),
            'drawdown_p95': np.percentile(drawdowns, 95),

            # Sharpe
            'sharpe_mean': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'sharpe_median': np.median(sharpe_ratios) if sharpe_ratios else 0,

            # Risk metrics
            'bust_rate': bust_rate,
            'profitable_rate': profitable_rate,

            # Win rates
            'avg_win_rate': np.mean([r.actual_win_rate for r in results]),
            'avg_trades': np.mean([r.total_trades for r in results]),
        }

        return analysis

    def print_analysis(self, analysis: Dict):
        """Выводит анализ результатов"""

        print(f"\n{'='*80}")
        print(f"АНАЛИЗ РЕЗУЛЬТАТОВ: {analysis['scenario_name']}")
        print(f"{'='*80}")
        print(f"Количество симуляций: {analysis['num_simulations']:,}")
        print()

        print("📊 ДОХОДНОСТЬ (90 дней):")
        print(f"  Средняя:     {analysis['return_mean']:+7.2f}%")
        print(f"  Медиана:     {analysis['return_median']:+7.2f}%")
        print(f"  Ст. откл.:   {analysis['return_std']:7.2f}%")
        print(f"  Минимум:     {analysis['return_min']:+7.2f}%")
        print(f"  Максимум:    {analysis['return_max']:+7.2f}%")
        print()
        print(f"  Percentiles:")
        print(f"    P5:        {analysis['return_p5']:+7.2f}%  (5% хуже)")
        print(f"    P25:       {analysis['return_p25']:+7.2f}%  (25% хуже)")
        print(f"    P75:       {analysis['return_p75']:+7.2f}%  (25% лучше)")
        print(f"    P95:       {analysis['return_p95']:+7.2f}%  (5% лучше)")
        print()

        print("💰 ФИНАЛЬНЫЙ БАЛАНС:")
        print(f"  Средний:     ${analysis['final_balance_mean']:8.2f}")
        print(f"  Медианный:   ${analysis['final_balance_median']:8.2f}")
        print(f"  Ст. откл.:   ${analysis['final_balance_std']:8.2f}")
        print(f"  Диапазон:    ${analysis['final_balance_min']:8.2f} - ${analysis['final_balance_max']:8.2f}")
        print()

        print("📉 ПРОСАДКИ:")
        print(f"  Средняя:     {analysis['drawdown_mean']:6.2f}%")
        print(f"  Медианная:   {analysis['drawdown_median']:6.2f}%")
        print(f"  Максимум:    {analysis['drawdown_max']:6.2f}%")
        print(f"  P95:         {analysis['drawdown_p95']:6.2f}%  (95% лучше)")
        print()

        print("📈 SHARPE RATIO:")
        print(f"  Средний:     {analysis['sharpe_mean']:6.3f}")
        print(f"  Медианный:   {analysis['sharpe_median']:6.3f}")
        print()

        print("⚠️  РИСКИ:")
        print(f"  Вероятность банкротства: {analysis['bust_rate']:5.2f}%")
        print(f"  Вероятность прибыли:     {analysis['profitable_rate']:5.2f}%")
        print()

        print("📊 СТАТИСТИКА СДЕЛОК:")
        print(f"  Среднее кол-во сделок:   {analysis['avg_trades']:.0f}")
        print(f"  Средний win rate:        {analysis['avg_win_rate']*100:.2f}%")
        print()


def create_scenarios() -> List[ScenarioParams]:
    """Создает 3 сценария: оптимистичный, реалистичный, пессимистичный"""

    scenarios = []

    # 1. ОПТИМИСТИЧНЫЙ СЦЕНАРИЙ
    scenarios.append(ScenarioParams(
        name="ОПТИМИСТИЧНЫЙ",
        win_rate=0.60,                    # 60% WR
        avg_trades_per_day=1.5,           # ~1-2 сделки в день
        tp_multiplier=2.5,
        sl_multiplier=1.5,
        trailing_stop_prob=0.30,          # 30% сделок с trailing stop
        trailing_stop_bonus=0.5,          # +0.5R бонус
        commission=0.0004,                # 0.04% комиссия
        slippage=0.0002,                  # 0.02% проскальзывание
        max_drawdown_before_stop=0.25     # Стоп при 25% просадке
    ))

    # 2. РЕАЛИСТИЧНЫЙ СЦЕНАРИЙ
    scenarios.append(ScenarioParams(
        name="РЕАЛИСТИЧНЫЙ",
        win_rate=0.52,                    # 52% WR
        avg_trades_per_day=1.0,           # ~1 сделка в день
        tp_multiplier=2.5,
        sl_multiplier=1.5,
        trailing_stop_prob=0.15,          # 15% сделок с trailing stop
        trailing_stop_bonus=0.3,          # +0.3R бонус
        commission=0.0006,                # 0.06% комиссия
        slippage=0.0004,                  # 0.04% проскальзывание
        max_drawdown_before_stop=0.20     # Стоп при 20% просадке
    ))

    # 3. ПЕССИМИСТИЧНЫЙ СЦЕНАРИЙ
    scenarios.append(ScenarioParams(
        name="ПЕССИМИСТИЧНЫЙ",
        win_rate=0.48,                    # 48% WR (ниже 50%)
        avg_trades_per_day=0.7,           # ~0.7 сделки в день
        tp_multiplier=2.5,
        sl_multiplier=1.5,
        trailing_stop_prob=0.05,          # 5% сделок с trailing stop
        trailing_stop_bonus=0.2,          # +0.2R бонус
        commission=0.0008,                # 0.08% комиссия
        slippage=0.0006,                  # 0.06% проскальзывание
        max_drawdown_before_stop=0.20     # Стоп при 20% просадке
    ))

    return scenarios


def main():
    """Главная функция"""

    print("\n" + "="*80)
    print("МОНТЕ-КАРЛО СИМУЛЯЦИЯ ДОХОДНОСТИ БОТА")
    print("="*80)
    print()
    print("Параметры симуляции:")
    print(f"  - Начальный капитал: $1,000")
    print(f"  - Период: 90 дней (~3 месяца)")
    print(f"  - Количество траекторий: 10,000 на сценарий")
    print(f"  - Kelly sizing: Quarter Kelly (0.25)")
    print(f"  - R/R ratio: 2.5 / 1.5 = 1.667")
    print(f"  - Максимум позиций: 10")
    print(f"  - Размер позиции: $15-200")
    print()

    # Создаем симулятор
    simulator = MonteCarloSimulator(
        initial_capital=1000.0,
        max_positions=10,
        min_position_size=15.0,
        max_position_size=200.0,
        kelly_fraction=0.25,
        simulation_days=90
    )

    # Создаем сценарии
    scenarios = create_scenarios()

    # Запускаем симуляции для каждого сценария
    all_results = {}
    all_analyses = {}

    for scenario in scenarios:
        print(f"\n{'─'*80}")
        print(f"СЦЕНАРИЙ: {scenario.name}")
        print(f"{'─'*80}")
        print(f"Win Rate: {scenario.win_rate*100:.0f}%")
        print(f"Trades/day: {scenario.avg_trades_per_day:.1f}")
        print(f"Trailing Stop prob: {scenario.trailing_stop_prob*100:.0f}%")
        print(f"Commission: {scenario.commission*100:.2f}%")

        results = simulator.run_monte_carlo(scenario, num_simulations=10000)
        all_results[scenario.name] = results

        analysis = simulator.analyze_results(results)
        all_analyses[scenario.name] = analysis

        simulator.print_analysis(analysis)

    # Сравнительная таблица
    print(f"\n{'='*80}")
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА СЦЕНАРИЕВ")
    print(f"{'='*80}\n")

    print(f"{'Метрика':<30} {'Оптимист.':>15} {'Реалист.':>15} {'Пессимист.':>15}")
    print(f"{'-'*30} {'-'*15} {'-'*15} {'-'*15}")

    metrics = [
        ('Средняя доходность', 'return_mean', '%'),
        ('Медианная доходность', 'return_median', '%'),
        ('Вероятность прибыли', 'profitable_rate', '%'),
        ('Средний баланс', 'final_balance_mean', '$'),
        ('Медианный баланс', 'final_balance_median', '$'),
        ('Средняя просадка', 'drawdown_mean', '%'),
        ('Sharpe Ratio', 'sharpe_mean', ''),
        ('Риск банкротства', 'bust_rate', '%'),
    ]

    for metric_name, metric_key, unit in metrics:
        opt = all_analyses["ОПТИМИСТИЧНЫЙ"][metric_key]
        real = all_analyses["РЕАЛИСТИЧНЫЙ"][metric_key]
        pess = all_analyses["ПЕССИМИСТИЧНЫЙ"][metric_key]

        if unit == '%':
            print(f"{metric_name:<30} {opt:>14.2f}% {real:>14.2f}% {pess:>14.2f}%")
        elif unit == '$':
            print(f"{metric_name:<30} ${opt:>13.2f} ${real:>13.2f} ${pess:>13.2f}")
        else:
            print(f"{metric_name:<30} {opt:>15.3f} {real:>15.3f} {pess:>15.3f}")

    print(f"\n{'='*80}")
    print("ВЫВОДЫ И РЕКОМЕНДАЦИИ")
    print(f"{'='*80}\n")

    real_analysis = all_analyses["РЕАЛИСТИЧНЫЙ"]

    print(f"📊 РЕАЛИСТИЧНЫЙ СЦЕНАРИЙ (наиболее вероятный):")
    print(f"   - Ожидаемая доходность за 3 месяца: {real_analysis['return_median']:+.1f}%")
    print(f"   - Вероятность получить прибыль: {real_analysis['profitable_rate']:.1f}%")
    print(f"   - Риск банкротства: {real_analysis['bust_rate']:.2f}%")
    print(f"   - Sharpe Ratio: {real_analysis['sharpe_median']:.2f}")
    print()

    # Оценка качества стратегии
    if real_analysis['return_median'] > 10:
        verdict = "🟢 ОТЛИЧНАЯ СТРАТЕГИЯ"
        comment = "Высокая ожидаемая доходность с приемлемым риском"
    elif real_analysis['return_median'] > 5:
        verdict = "🟡 ХОРОШАЯ СТРАТЕГИЯ"
        comment = "Положительная доходность, можно использовать"
    elif real_analysis['return_median'] > 0:
        verdict = "🟠 ПОСРЕДСТВЕННАЯ СТРАТЕГИЯ"
        comment = "Низкая доходность, нужна оптимизация"
    else:
        verdict = "🔴 УБЫТОЧНАЯ СТРАТЕГИЯ"
        comment = "Отрицательная ожидаемая доходность"

    print(f"   Вердикт: {verdict}")
    print(f"   {comment}")
    print()

    print(f"💡 РЕКОМЕНДАЦИИ:")

    if real_analysis['bust_rate'] > 5:
        print(f"   ⚠️  Высокий риск банкротства ({real_analysis['bust_rate']:.1f}%)")
        print(f"       Рекомендуется: снизить Kelly fraction или увеличить начальный капитал")

    if real_analysis['sharpe_median'] < 1.0:
        print(f"   ⚠️  Низкий Sharpe Ratio ({real_analysis['sharpe_median']:.2f})")
        print(f"       Рекомендуется: улучшить качество сигналов или R/R ratio")

    if real_analysis['return_median'] < 5:
        print(f"   ⚠️  Низкая доходность ({real_analysis['return_median']:.1f}%)")
        print(f"       Рекомендуется: увеличить частоту сделок или улучшить win rate")

    if real_analysis['profitable_rate'] < 70:
        print(f"   ⚠️  Низкая вероятность прибыли ({real_analysis['profitable_rate']:.1f}%)")
        print(f"       Рекомендуется: улучшить стабильность стратегии")

    print()
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
