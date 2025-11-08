#!/usr/bin/env python3
"""
Монте-Карло анализ риска банкротства при разных MAX_RISK_PER_POSITION

Симулирует 10000 торговых траекторий для оценки:
- Риска банкротства (drawdown > 50%)
- Максимального drawdown
- Итоговой доходности

Сценарии:
1. Оптимистичный: WR=55%, avg_win=8%, avg_loss=4%
2. Реалистичный: WR=45%, avg_win=6%, avg_loss=4%
3. Пессимистичный: WR=35%, avg_win=5%, avg_loss=5%
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ScenarioParams:
    """Параметры торгового сценария"""
    name: str
    win_rate: float  # Вероятность выигрыша (0-1)
    avg_win_pct: float  # Средний выигрыш в %
    avg_loss_pct: float  # Средний проигрыш в %
    win_std: float = 2.0  # Стандартное отклонение выигрышей
    loss_std: float = 1.5  # Стандартное отклонение проигрышей


@dataclass
class RiskSettings:
    """Настройки риск-менеджмента"""
    max_risk_per_position: float  # Максимальный риск на позицию (0.025 = 2.5%)
    max_positions: int = 10
    initial_capital: float = 1000.0
    kelly_fraction: float = 0.25


def simulate_single_trajectory(
    scenario: ScenarioParams,
    risk_settings: RiskSettings,
    n_trades: int = 1000
) -> Dict:
    """
    Симулирует одну торговую траекторию

    Args:
        scenario: Параметры сценария
        risk_settings: Настройки риск-менеджмента
        n_trades: Количество сделок

    Returns:
        dict с результатами траектории
    """
    capital = risk_settings.initial_capital
    capital_history = [capital]
    max_drawdown = 0.0
    peak_capital = capital

    for _ in range(n_trades):
        # Открываем позиции (до max_positions)
        num_positions = min(risk_settings.max_positions,
                           int(np.random.uniform(5, risk_settings.max_positions + 1)))

        # Размер каждой позиции (Kelly)
        position_size = capital * risk_settings.max_risk_per_position

        # Суммарный PnL за цикл ребалансировки
        cycle_pnl = 0.0

        for _ in range(num_positions):
            # Определяем исход сделки
            is_win = np.random.random() < scenario.win_rate

            if is_win:
                # Выигрыш с вариацией
                pnl_pct = np.random.normal(scenario.avg_win_pct, scenario.win_std)
                pnl_pct = max(0.1, pnl_pct)  # Минимум 0.1%
            else:
                # Проигрыш с вариацией
                pnl_pct = -np.random.normal(scenario.avg_loss_pct, scenario.loss_std)
                pnl_pct = min(-0.1, pnl_pct)  # Минимум -0.1%

            # PnL в USDT
            pnl = position_size * (pnl_pct / 100.0)
            cycle_pnl += pnl

        # Обновляем капитал
        capital += cycle_pnl

        # Защита от отрицательного капитала
        if capital <= 0:
            capital = 0.01  # Минимальный остаток

        capital_history.append(capital)

        # Обновляем пик и drawdown
        if capital > peak_capital:
            peak_capital = capital

        current_drawdown = (peak_capital - capital) / peak_capital
        max_drawdown = max(max_drawdown, current_drawdown)

    # Финальные метрики
    final_capital = capital
    total_return = (final_capital / risk_settings.initial_capital - 1) * 100
    is_bankrupt = max_drawdown > 0.5  # Банкротство = drawdown > 50%

    return {
        'final_capital': final_capital,
        'total_return_pct': total_return,
        'max_drawdown': max_drawdown,
        'is_bankrupt': is_bankrupt,
        'capital_history': capital_history
    }


def run_monte_carlo_simulation(
    scenario: ScenarioParams,
    risk_settings: RiskSettings,
    n_simulations: int = 10000,
    n_trades: int = 1000
) -> Dict:
    """
    Запускает Монте-Карло симуляцию

    Args:
        scenario: Параметры сценария
        risk_settings: Настройки риск-менеджмента
        n_simulations: Количество траекторий
        n_trades: Количество сделок в траектории

    Returns:
        dict с агрегированными результатами
    """
    results = []

    print(f"Запуск {n_simulations} симуляций для сценария '{scenario.name}'...")
    print(f"  MAX_RISK: {risk_settings.max_risk_per_position*100:.1f}%")
    print(f"  Win Rate: {scenario.win_rate*100:.0f}%")
    print(f"  Avg Win: +{scenario.avg_win_pct:.1f}%, Avg Loss: -{scenario.avg_loss_pct:.1f}%")

    for i in range(n_simulations):
        result = simulate_single_trajectory(scenario, risk_settings, n_trades)
        results.append(result)

        if (i + 1) % 1000 == 0:
            print(f"  Прогресс: {i+1}/{n_simulations}")

    # Агрегация результатов
    final_capitals = [r['final_capital'] for r in results]
    total_returns = [r['total_return_pct'] for r in results]
    max_drawdowns = [r['max_drawdown'] for r in results]
    bankruptcies = sum(r['is_bankrupt'] for r in results)

    # Перцентили
    percentiles = [5, 25, 50, 75, 95]
    return_percentiles = np.percentile(total_returns, percentiles)
    drawdown_percentiles = np.percentile(max_drawdowns, percentiles)

    return {
        'scenario': scenario.name,
        'max_risk': risk_settings.max_risk_per_position,
        'n_simulations': n_simulations,
        'bankruptcy_count': bankruptcies,
        'bankruptcy_rate': bankruptcies / n_simulations * 100,
        'avg_return': np.mean(total_returns),
        'median_return': np.median(total_returns),
        'std_return': np.std(total_returns),
        'return_percentiles': dict(zip(percentiles, return_percentiles)),
        'avg_max_drawdown': np.mean(max_drawdowns),
        'median_max_drawdown': np.median(max_drawdowns),
        'drawdown_percentiles': dict(zip(percentiles, drawdown_percentiles)),
        'positive_return_rate': sum(1 for r in total_returns if r > 0) / n_simulations * 100,
        'all_results': results
    }


def print_results(results: Dict):
    """Выводит результаты симуляции"""
    print("\n" + "="*80)
    print(f"РЕЗУЛЬТАТЫ: {results['scenario']}")
    print(f"MAX_RISK_PER_POSITION: {results['max_risk']*100:.1f}%")
    print("="*80)

    print(f"\n💀 РИСК БАНКРОТСТВА:")
    print(f"  Траектории с drawdown > 50%: {results['bankruptcy_count']:,} из {results['n_simulations']:,}")
    print(f"  Вероятность банкротства: {results['bankruptcy_rate']:.2f}%")

    print(f"\n📊 ДОХОДНОСТЬ:")
    print(f"  Средняя: {results['avg_return']:+.2f}%")
    print(f"  Медианная: {results['median_return']:+.2f}%")
    print(f"  Стд. отклонение: {results['std_return']:.2f}%")
    print(f"  Процент прибыльных: {results['positive_return_rate']:.1f}%")

    print(f"\n  Перцентили доходности:")
    for p, val in results['return_percentiles'].items():
        print(f"    {p}%: {val:+.2f}%")

    print(f"\n📉 МАКСИМАЛЬНЫЙ DRAWDOWN:")
    print(f"  Средний: {results['avg_max_drawdown']*100:.2f}%")
    print(f"  Медианный: {results['median_max_drawdown']*100:.2f}%")

    print(f"\n  Перцентили drawdown:")
    for p, val in results['drawdown_percentiles'].items():
        print(f"    {p}%: {val*100:.2f}%")

    print("="*80)


def compare_risk_levels(
    scenario: ScenarioParams,
    risk_levels: List[float],
    n_simulations: int = 10000
):
    """
    Сравнивает разные уровни риска для одного сценария

    Args:
        scenario: Параметры сценария
        risk_levels: Список уровней риска (например, [0.025, 0.05, 0.09])
        n_simulations: Количество симуляций
    """
    print("\n" + "="*80)
    print(f"СРАВНЕНИЕ УРОВНЕЙ РИСКА: {scenario.name}")
    print("="*80)

    all_results = []

    for risk_level in risk_levels:
        risk_settings = RiskSettings(max_risk_per_position=risk_level)
        results = run_monte_carlo_simulation(scenario, risk_settings, n_simulations)
        all_results.append(results)
        print_results(results)

    # Сравнительная таблица
    print("\n" + "="*80)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print("="*80)
    print(f"{'MAX_RISK':<12} | {'Банкротство':>12} | {'Медианная доходность':>20} | {'Медианный DD':>15}")
    print("-" * 80)

    for r in all_results:
        print(f"{r['max_risk']*100:>10.1f}% | {r['bankruptcy_rate']:>11.2f}% | "
              f"{r['median_return']:>19.2f}% | {r['median_max_drawdown']*100:>14.2f}%")

    print("="*80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("МОНТЕ-КАРЛО АНАЛИЗ РИСКА БАНКРОТСТВА")
    print("="*80)
    print("\nПараметры:")
    print("  - Симуляций: 10,000")
    print("  - Сделок в траектории: 1,000")
    print("  - Начальный капитал: $1,000")
    print("  - Максимум позиций: 10")
    print("  - Kelly fraction: 0.25")

    # Определяем сценарии
    scenarios = [
        ScenarioParams(
            name="Оптимистичный",
            win_rate=0.55,
            avg_win_pct=8.0,
            avg_loss_pct=4.0
        ),
        ScenarioParams(
            name="Реалистичный",
            win_rate=0.45,
            avg_win_pct=6.0,
            avg_loss_pct=4.0
        ),
        ScenarioParams(
            name="Пессимистичный",
            win_rate=0.35,
            avg_win_pct=5.0,
            avg_loss_pct=5.0
        )
    ]

    # Уровни риска для сравнения
    risk_levels = [0.025, 0.05, 0.09]  # 2.5%, 5%, 9%

    # Запускаем симуляции для каждого сценария
    for scenario in scenarios:
        compare_risk_levels(scenario, risk_levels, n_simulations=10000)

    print("\n" + "="*80)
    print("✅ АНАЛИЗ ЗАВЕРШЕН")
    print("="*80)
    print("\nРекомендации:")
    print("  - Если банкротство < 5% → риск приемлемый")
    print("  - Если банкротство 5-10% → риск умеренный")
    print("  - Если банкротство > 10% → риск высокий")
    print("  - Оптимальный MAX_RISK балансирует доходность и безопасность")
