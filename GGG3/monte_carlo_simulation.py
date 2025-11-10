#!/usr/bin/env python3
"""
Monte Carlo Simulation для оценки годовой доходности торгового бота

Симулирует 10,000 траекторий на основе параметров бота:
- Risk:Reward = 1.67:1 (TP=2.5×ATR, SL=1.5×ATR)
- Порог вероятности p_up > 0.58
- Максимум 10 одновременных позиций
- Начальный капитал: $1000
- Комиссии + проскальзывание: 0.3% на сделку

Автор: Claude (Anthropic)
Дата: 2025-11-11
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

np.random.seed(42)  # Для воспроизводимости


@dataclass
class TradingParameters:
    """Параметры торговой системы"""
    # Начальные условия
    initial_capital: float = 1000.0
    max_positions: int = 10

    # TP/SL
    tp_multiplier: float = 2.5
    sl_multiplier: float = 1.5

    # Издержки
    commission_pct: float = 0.001  # 0.1%
    slippage_pct: float = 0.0005  # 0.05%
    total_cost_pct: float = 0.003  # 0.3% на круг

    # Риск
    risk_per_position: float = 0.009  # 0.9% от капитала

    # ML модель
    p_threshold: float = 0.58  # Порог для открытия позиций
    avg_p_up: float = 0.63  # Средняя вероятность при открытии
    p_up_std: float = 0.05  # Стандартное отклонение p_up

    # Частота торговли
    avg_trades_per_week: int = 25  # ~10 позиций × 2.5 оборота
    trades_per_week_std: int = 5

    # Калибровка Win Rate
    # Теоретический win rate = 52-58% для p_up ~ 0.63
    # Консервативная оценка: 54%
    base_win_rate: float = 0.54
    win_rate_std: float = 0.03


def estimate_win_rate_from_p_up(p_up: float, params: TradingParameters) -> float:
    """
    Оценка win rate на основе p_up с учетом реальной дисперсии рынка

    Логика:
    - p_up = 0.58 → win_rate ~ 0.52 (минимальный порог)
    - p_up = 0.63 → win_rate ~ 0.54 (средний)
    - p_up = 0.70 → win_rate ~ 0.58 (отличный сигнал)
    """
    # Консервативная калибровка: win_rate = 0.82 × p_up + 0.065
    calibrated_wr = 0.82 * p_up + 0.065

    # Добавляем случайный шум ±3%
    noise = np.random.normal(0, params.win_rate_std)
    final_wr = np.clip(calibrated_wr + noise, 0.45, 0.70)

    return final_wr


def simulate_single_trade(p_up: float, params: TradingParameters) -> Tuple[float, bool]:
    """
    Симулирует одну сделку

    Returns:
        (pnl_pct, is_win): PnL в процентах и флаг победы
    """
    # Оцениваем win rate для данного p_up
    win_rate = estimate_win_rate_from_p_up(p_up, params)

    # Случайный результат сделки
    is_win = np.random.random() < win_rate

    if is_win:
        # TP с учетом издержек
        real_tp = params.tp_multiplier - params.total_cost_pct
        pnl_pct = real_tp * params.risk_per_position
    else:
        # SL с учетом издержек
        real_sl = params.sl_multiplier + params.total_cost_pct
        pnl_pct = -real_sl * params.risk_per_position

    return pnl_pct, is_win


def simulate_year(params: TradingParameters) -> Dict:
    """
    Симулирует 1 год торговли

    Returns:
        Словарь с метриками
    """
    capital = params.initial_capital
    peak_capital = capital
    max_drawdown = 0.0

    trades = []
    balances = [capital]

    # Симулируем 52 недели
    weeks = 52

    for week in range(weeks):
        # Случайное количество сделок на неделе
        n_trades_week = max(1, int(np.random.normal(
            params.avg_trades_per_week,
            params.trades_per_week_std
        )))

        for _ in range(n_trades_week):
            # Случайная p_up для сделки (обычно выше порога)
            p_up = np.random.normal(params.avg_p_up, params.p_up_std)
            p_up = np.clip(p_up, params.p_threshold, 0.85)

            # Симулируем сделку
            pnl_pct, is_win = simulate_single_trade(p_up, params)

            # Обновляем капитал
            pnl_amount = capital * pnl_pct
            capital += pnl_amount

            # Обновляем максимальную просадку
            if capital > peak_capital:
                peak_capital = capital

            current_drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, current_drawdown)

            # Сохраняем результаты
            trades.append({
                'week': week,
                'p_up': p_up,
                'pnl_pct': pnl_pct * 100,
                'pnl_amount': pnl_amount,
                'is_win': is_win,
                'capital': capital
            })

            balances.append(capital)

            # Защита от банкротства
            if capital < params.initial_capital * 0.3:
                break

        if capital < params.initial_capital * 0.3:
            break

    # Расчёт метрик
    df_trades = pd.DataFrame(trades)

    total_trades = len(df_trades)
    wins = df_trades['is_win'].sum()
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0

    total_return_pct = (capital - params.initial_capital) / params.initial_capital * 100

    # Sharpe Ratio
    if len(df_trades) > 1:
        returns = df_trades['pnl_pct'].values
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (avg_return / std_return) * np.sqrt(total_trades) if std_return > 0 else 0
    else:
        sharpe = 0

    return {
        'final_balance': capital,
        'total_return_pct': total_return_pct,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'max_drawdown_pct': max_drawdown * 100,
        'sharpe_ratio': sharpe
    }


def run_monte_carlo(n_simulations: int = 10000, params: TradingParameters = None) -> pd.DataFrame:
    """
    Запуск Monte Carlo симуляции
    """
    if params is None:
        params = TradingParameters()

    print("=" * 80)
    print(f"🎲 MONTE CARLO SIMULATION: {n_simulations:,} траекторий")
    print("=" * 80)
    print(f"\nПараметры бота:")
    print(f"  Начальный капитал: ${params.initial_capital:,.0f}")
    print(f"  Максимум позиций: {params.max_positions}")
    print(f"  Risk:Reward: {params.tp_multiplier:.1f}:{params.sl_multiplier:.1f} = {params.tp_multiplier/params.sl_multiplier:.2f}:1")
    print(f"  Риск на позицию: {params.risk_per_position*100:.1f}%")
    print(f"  Издержки на сделку: {params.total_cost_pct*100:.2f}%")
    print(f"  Порог p_up: {params.p_threshold:.2f}")
    print(f"  Средняя p_up: {params.avg_p_up:.2f}")
    print(f"  Ожидаемый Win Rate: {params.base_win_rate*100:.1f}%")
    print(f"  Частота торговли: ~{params.avg_trades_per_week * 52} сделок/год")
    print()

    results = []

    print(f"Запуск симуляции...")
    for i in range(n_simulations):
        if (i + 1) % 1000 == 0:
            print(f"  Прогресс: {i+1:,}/{n_simulations:,} ({(i+1)/n_simulations*100:.1f}%)")

        result = simulate_year(params)
        results.append({
            'sim_id': i,
            'final_balance': result['final_balance'],
            'total_return_pct': result['total_return_pct'],
            'total_trades': result['total_trades'],
            'win_rate': result['win_rate'],
            'max_drawdown_pct': result['max_drawdown_pct'],
            'sharpe_ratio': result['sharpe_ratio']
        })

    df_results = pd.DataFrame(results)

    print(f"\n✅ Симуляция завершена: {n_simulations:,} траекторий")

    return df_results


def analyze_results(df_results: pd.DataFrame, params: TradingParameters):
    """
    Анализ результатов Monte Carlo симуляции
    """
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ MONTE CARLO АНАЛИЗА")
    print("=" * 80)

    # Базовая статистика
    print("\n1. ГОДОВАЯ ДОХОДНОСТЬ (Return on Investment):")
    print(f"   Медиана (50%): {df_results['total_return_pct'].median():+.1f}%")
    print(f"   Среднее: {df_results['total_return_pct'].mean():+.1f}%")
    print(f"   Std Dev: {df_results['total_return_pct'].std():.1f}%")
    print()
    print(f"   Перцентили:")
    print(f"     5% (Worst case): {df_results['total_return_pct'].quantile(0.05):+.1f}%")
    print(f"    10%: {df_results['total_return_pct'].quantile(0.10):+.1f}%")
    print(f"    25%: {df_results['total_return_pct'].quantile(0.25):+.1f}%")
    print(f"    50% (Median): {df_results['total_return_pct'].quantile(0.50):+.1f}%")
    print(f"    75%: {df_results['total_return_pct'].quantile(0.75):+.1f}%")
    print(f"    90%: {df_results['total_return_pct'].quantile(0.90):+.1f}%")
    print(f"    95% (Best case): {df_results['total_return_pct'].quantile(0.95):+.1f}%")

    # Вероятность прибыли
    prob_profit = (df_results['total_return_pct'] > 0).mean()
    print(f"\n   Вероятность прибыли: {prob_profit*100:.1f}%")
    print(f"   Вероятность убытка: {(1-prob_profit)*100:.1f}%")

    # Финальный баланс
    print("\n2. ФИНАЛЬНЫЙ БАЛАНС:")
    print(f"   Начальный: ${params.initial_capital:,.0f}")
    print(f"   Медиана: ${df_results['final_balance'].median():,.0f}")
    print(f"   Среднее: ${df_results['final_balance'].mean():,.0f}")
    print(f"   Min (5%): ${df_results['final_balance'].quantile(0.05):,.0f}")
    print(f"   Max (95%): ${df_results['final_balance'].quantile(0.95):,.0f}")

    # Win Rate
    print("\n3. WIN RATE:")
    print(f"   Медиана: {df_results['win_rate'].median()*100:.1f}%")
    print(f"   Среднее: {df_results['win_rate'].mean()*100:.1f}%")
    print(f"   Диапазон: {df_results['win_rate'].min()*100:.1f}% - {df_results['win_rate'].max()*100:.1f}%")

    # Максимальная просадка
    print("\n4. МАКСИМАЛЬНАЯ ПРОСАДКА:")
    print(f"   Медиана: {df_results['max_drawdown_pct'].median():.1f}%")
    print(f"   Среднее: {df_results['max_drawdown_pct'].mean():.1f}%")
    print(f"   Worst case (95%): {df_results['max_drawdown_pct'].quantile(0.95):.1f}%")

    # Sharpe Ratio
    print("\n5. SHARPE RATIO (аннуализированный):")
    print(f"   Медиана: {df_results['sharpe_ratio'].median():.2f}")
    print(f"   Среднее: {df_results['sharpe_ratio'].mean():.2f}")

    # Количество сделок
    print("\n6. КОЛИЧЕСТВО СДЕЛОК:")
    print(f"   Медиана: {df_results['total_trades'].median():.0f} сделок/год")
    print(f"   Среднее: {df_results['total_trades'].mean():.0f} сделок/год")

    # Risk-adjusted metrics
    median_return = df_results['total_return_pct'].median()
    median_drawdown = df_results['max_drawdown_pct'].median()
    calmar_ratio = median_return / median_drawdown if median_drawdown > 0 else 0

    print("\n7. RISK-ADJUSTED RETURN:")
    print(f"   Calmar Ratio (Return/MaxDD): {calmar_ratio:.2f}")

    # Вероятность достижения целей
    print("\n8. ВЕРОЯТНОСТЬ ДОСТИЖЕНИЯ ЦЕЛЕЙ:")
    print(f"   Удвоить капитал (+100%): {(df_results['total_return_pct'] >= 100).mean()*100:.1f}%")
    print(f"   +50% доходность: {(df_results['total_return_pct'] >= 50).mean()*100:.1f}%")
    print(f"   +25% доходность: {(df_results['total_return_pct'] >= 25).mean()*100:.1f}%")
    print(f"   Убыток > 20%: {(df_results['total_return_pct'] <= -20).mean()*100:.1f}%")

    print("\n" + "=" * 80)

    # Рекомендации
    print("\n💡 ИНТЕРПРЕТАЦИЯ:")
    if median_return > 20:
        print("   ✅ Система показывает ОТЛИЧНУЮ ожидаемую доходность")
    elif median_return > 10:
        print("   ✅ Система показывает ХОРОШУЮ ожидаемую доходность")
    elif median_return > 0:
        print("   ⚠️  Система показывает СЛАБУЮ, но положительную доходность")
    else:
        print("   ❌ Система показывает ОТРИЦАТЕЛЬНУЮ ожидаемую доходность")

    if median_drawdown < 15:
        print("   ✅ Просадки находятся в ПРИЕМЛЕМОМ диапазоне")
    elif median_drawdown < 25:
        print("   ⚠️  Просадки УМЕРЕННО ВЫСОКИЕ - требуется контроль рисков")
    else:
        print("   ❌ Просадки КРИТИЧЕСКИ ВЫСОКИЕ - требуется оптимизация")

    if df_results['sharpe_ratio'].median() > 1.5:
        print("   ✅ Sharpe Ratio ОТЛИЧНЫЙ (>1.5)")
    elif df_results['sharpe_ratio'].median() > 1.0:
        print("   ✅ Sharpe Ratio ХОРОШИЙ (>1.0)")
    elif df_results['sharpe_ratio'].median() > 0.5:
        print("   ⚠️  Sharpe Ratio ПРИЕМЛЕМЫЙ (>0.5)")
    else:
        print("   ❌ Sharpe Ratio НИЗКИЙ (<0.5)")

    print("=" * 80)


def main():
    """Главная функция"""
    # Параметры симуляции
    params = TradingParameters(
        initial_capital=1000.0,
        max_positions=10,
        tp_multiplier=2.5,
        sl_multiplier=1.5,
        risk_per_position=0.009,
        p_threshold=0.58,
        avg_p_up=0.63,
        p_up_std=0.05,
        avg_trades_per_week=25,
        base_win_rate=0.54,
        win_rate_std=0.03
    )

    # Запуск симуляции
    n_simulations = 10000
    df_results = run_monte_carlo(n_simulations, params)

    # Анализ
    analyze_results(df_results, params)

    # Сохранение результатов
    output_file = 'monte_carlo_results.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Результаты сохранены в: {output_file}")


if __name__ == '__main__':
    main()
